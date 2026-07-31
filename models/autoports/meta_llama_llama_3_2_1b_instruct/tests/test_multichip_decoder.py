# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import ttnn
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.autoports.meta_llama_llama_3_2_1b_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    MODEL_ID,
    PCC_THRESHOLD,
    DecodeRotaryHelper,
    _assert_pcc,
    _make_page_table,
    _meta_rot_mats_prefill,
    _real_layer_state_dict,
    _synthetic_layer_state_dict,
    _to_tt_decode,
    _to_tt_prefill,
    _tt_to_layer_output,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    MultichipDecoder,
    _MultichipLlamaMLP,
    _all_gather_hidden,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import OptimizedDecoder


ARTIFACT_DIR = Path(
    os.getenv(
        "MD_MULTICHIP_ARTIFACT_DIR",
        "models/autoports/meta_llama_llama_3_2_1b_instruct/doc/multichip_decoder",
    )
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _require_target_mesh() -> None:
    if ttnn.get_num_devices() < 8:
        pytest.skip("multichip decoder requires the local 8-chip T3K target mesh")


@contextmanager
def _open_mesh(shape: tuple[int, int], *, trace_region_size: int = 100000000):
    fabric_config = None if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D_RING
    if fabric_config is not None:
        ttnn.set_fabric_config(fabric_config)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(shape),
        trace_region_size=trace_region_size,
        num_command_queues=1,
    )
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _mesh_shape_tuple(mesh_device: ttnn.MeshDevice) -> tuple[int, int]:
    return int(mesh_device.shape[0]), int(mesh_device.shape[1])


def _current_pos_to_tt(current_pos: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _run_tt_prefill_and_decode(
    *,
    decoder,
    mesh_device: ttnn.MeshDevice,
    rotary_emb: LlamaRotaryEmbedding,
    hidden_states: torch.Tensor,
    decode_hidden_states: torch.Tensor,
    page_table: torch.Tensor,
    page_table_tt: ttnn.Tensor,
    page_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    batch, prefill_seq_len, hidden_size = hidden_states.shape
    _, tt_rot_mats = _meta_rot_mats_prefill(rotary_emb, hidden_states, 0, prefill_seq_len, mesh_device)
    prefill_input = _to_tt_prefill(hidden_states, mesh_device)

    prefill_out = decoder.prefill_forward(prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt, user_id=0)
    ttnn.synchronize_device(mesh_device)
    prefill_torch = _tt_to_layer_output(prefill_out, batch=batch, seq_len=prefill_seq_len, hidden_size=hidden_size)

    current_pos_host = torch.full((batch,), prefill_seq_len, dtype=torch.int32)
    current_pos_tt = _current_pos_to_tt(current_pos_host, mesh_device)
    rot_mats = DecodeRotaryHelper(rotary_emb, prefill_seq_len + 2, hidden_size // 32, mesh_device).get_rot_mats(
        current_pos_host
    )
    decode_input_tt = _to_tt_decode(decode_hidden_states, decoder, mesh_device)

    eager_out = decoder.decode_forward(
        decode_input_tt,
        current_pos=current_pos_tt,
        rot_mats=rot_mats,
        page_table=page_table_tt,
    )
    ttnn.synchronize_device(mesh_device)
    eager_torch = _tt_to_layer_output(eager_out, batch=batch, seq_len=1, hidden_size=hidden_size)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = decoder.decode_forward(
        decode_input_tt,
        current_pos=current_pos_tt,
        rot_mats=rot_mats,
        page_table=page_table_tt,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    replay_torch_1 = _tt_to_layer_output(traced_out, batch=batch, seq_len=1, hidden_size=hidden_size)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    replay_torch_2 = _tt_to_layer_output(traced_out, batch=batch, seq_len=1, hidden_size=hidden_size)
    ttnn.release_trace(mesh_device, trace_id)

    keys, values = decoder.kv_cache
    local_key_shapes = [list(device_tensor.shape) for device_tensor in ttnn.get_device_tensors(keys)]
    local_value_shapes = [list(device_tensor.shape) for device_tensor in ttnn.get_device_tensors(values)]
    layout_info = {
        "mesh_shape": list(_mesh_shape_tuple(mesh_device)),
        "page_table": page_table.tolist(),
        "current_pos": current_pos_host.tolist(),
        "kv_cache_shape": list(keys.shape),
        "local_key_shapes": local_key_shapes,
        "local_value_shapes": local_value_shapes,
        "prefill_output_topology": repr(prefill_out.tensor_topology()),
        "decode_output_topology": repr(traced_out.tensor_topology()),
        "page_block_size": page_block_size,
    }
    return prefill_torch, eager_torch, replay_torch_1, replay_torch_2, layout_info


def _make_decoder(decoder_cls, state_dict, hf_config, mesh_device, *, page_block_size: int, max_seq_len: int):
    return decoder_cls.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )


def test_multichip_decoder_contract_and_runtime_fallback_audit():
    source = inspect.getsource(MultichipDecoder.prefill_forward)
    source += inspect.getsource(MultichipDecoder.decode_forward)
    source += inspect.getsource(_MultichipLlamaMLP.prefill_forward)
    source += inspect.getsource(_MultichipLlamaMLP.decode_forward)
    source += inspect.getsource(_all_gather_hidden)

    for forbidden in ("torch", "from_torch", "to_torch", "cpu("):
        assert forbidden not in source

    module_source = inspect.getsource(__import__(MultichipDecoder.__module__, fromlist=["_"]))
    assert "OptimizedDecoder.from_state_dict" in module_source
    assert "_reduce_scatter_hidden" in module_source
    assert "all_gather_async" in module_source


def test_multichip_static_mesh_plan_uses_optimized_baseline():
    _require_target_mesh()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)

    with _open_mesh(TARGET_MESH_SHAPE) as mesh_device:
        decoder = _make_decoder(
            MultichipDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=64,
            max_seq_len=256,
        )
        assert isinstance(decoder, OptimizedDecoder)
        assert isinstance(decoder.mlp, _MultichipLlamaMLP)
        assert decoder.attention.config.topology == ttnn.Topology.Ring
        assert decoder.attention.config.use_fused_all_gather_matmul is True
        assert decoder.mesh_plan.mesh_shape == TARGET_MESH_SHAPE
        assert decoder.mesh_plan.local_query_heads == 4
        assert decoder.mesh_plan.local_kv_heads == 1
        assert decoder.mesh_plan.local_qkv_width == 384
        assert decoder.mesh_plan.local_intermediate == 1024

        keys, values = decoder.kv_cache
        assert len(ttnn.get_device_tensors(keys)) == 8
        assert len(ttnn.get_device_tensors(values)) == 8
        assert keys.shape[-3] == 1
        assert values.shape[-3] == 1

        _write_json(
            ARTIFACT_DIR / "mesh_strategy.json",
            {
                "status": "selected",
                "hardware": {
                    "arch": ttnn.get_arch_name(),
                    "cluster_type": str(ttnn.cluster.get_cluster_type()),
                    "num_devices": ttnn.get_num_devices(),
                },
                "plan": decoder.mesh_strategy,
                "attention": {
                    "topology": str(decoder.attention.config.topology),
                    "use_fused_all_gather_matmul": decoder.attention.config.use_fused_all_gather_matmul,
                    "decode_residual_memcfg": str(decoder.decode_residual_memcfg),
                },
                "mlp": {
                    "implementation": type(decoder.mlp).__name__,
                    "decode_input_memcfg": str(decoder.mlp.config.decode_input_memcfg),
                    "decode_residual_memcfg": str(decoder.mlp.config.decode_residual_memcfg),
                },
            },
        )


def test_synthetic_multichip_paged_prefill_decode_trace_and_determinism():
    _require_target_mesh()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256
    torch.manual_seed(5)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15

    with _open_mesh((1, 1)) as mesh_device:
        page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=13
        )
        baseline = _make_decoder(
            OptimizedDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
        )
        baseline_prefill, baseline_eager, baseline_replay_1, baseline_replay_2, _ = _run_tt_prefill_and_decode(
            decoder=baseline,
            mesh_device=mesh_device,
            rotary_emb=rotary_emb,
            hidden_states=hidden_states,
            decode_hidden_states=decode_hidden,
            page_table=page_table,
            page_table_tt=page_table_tt,
            page_block_size=page_block_size,
        )

    with _open_mesh(TARGET_MESH_SHAPE) as mesh_device:
        multi_page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=13
        )
        assert torch.equal(page_table, multi_page_table)
        decoder = _make_decoder(
            MultichipDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
        )
        multi_prefill, multi_eager, multi_replay_1, multi_replay_2, layout_info = _run_tt_prefill_and_decode(
            decoder=decoder,
            mesh_device=mesh_device,
            rotary_emb=rotary_emb,
            hidden_states=hidden_states,
            decode_hidden_states=decode_hidden,
            page_table=page_table,
            page_table_tt=page_table_tt,
            page_block_size=page_block_size,
        )

    prefill_pcc = _assert_pcc("multichip synthetic prefill vs optimized single-chip", baseline_prefill, multi_prefill)
    eager_decode_pcc = _assert_pcc("multichip synthetic eager decode vs optimized single-chip", baseline_eager, multi_eager)
    replay_decode_pcc = _assert_pcc(
        "multichip synthetic traced replay decode vs optimized single-chip", baseline_replay_1, multi_replay_1
    )
    repeated_pcc = _assert_pcc("multichip synthetic repeated trace replay", multi_replay_1, multi_replay_2, threshold=0.9999)
    baseline_repeated_pcc = _assert_pcc("optimized baseline repeated trace replay", baseline_replay_1, baseline_replay_2, threshold=0.9999)

    _write_json(
        ARTIFACT_DIR / "synthetic_correctness.json",
        {
            "baseline": "OptimizedDecoder single-chip TTNN",
            "mesh_shape": list(TARGET_MESH_SHAPE),
            "prefill_seq_len": prefill_seq_len,
            "decode_current_pos": prefill_seq_len,
            "page_block_size": page_block_size,
            "prefill_pcc_vs_single_chip": prefill_pcc,
            "eager_decode_pcc_vs_single_chip": eager_decode_pcc,
            "traced_decode_replay_pcc_vs_single_chip": replay_decode_pcc,
            "repeated_trace_replay_pcc": repeated_pcc,
            "baseline_repeated_trace_replay_pcc": baseline_repeated_pcc,
            "threshold": PCC_THRESHOLD,
            "layout_info": layout_info,
            "status": "passed",
        },
    )


@pytest.mark.real_weights
def test_real_weights_multichip_paged_prefill_and_decode_trace():
    _require_target_mesh()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _real_layer_state_dict()
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = int(os.getenv("MD_PREFILL_SEQ_LEN", "128"))
    max_seq_len = max(256, ((prefill_seq_len + page_block_size) + page_block_size - 1) // page_block_size * page_block_size)
    torch.manual_seed(99)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15

    with _open_mesh((1, 1)) as mesh_device:
        page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=29
        )
        baseline = _make_decoder(
            OptimizedDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
        )
        baseline_prefill, _, baseline_replay_1, _, _ = _run_tt_prefill_and_decode(
            decoder=baseline,
            mesh_device=mesh_device,
            rotary_emb=rotary_emb,
            hidden_states=hidden_states,
            decode_hidden_states=decode_hidden,
            page_table=page_table,
            page_table_tt=page_table_tt,
            page_block_size=page_block_size,
        )

    with _open_mesh(TARGET_MESH_SHAPE) as mesh_device:
        multi_page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=29
        )
        assert torch.equal(page_table, multi_page_table)
        decoder = _make_decoder(
            MultichipDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
        )
        multi_prefill, _, multi_replay_1, multi_replay_2, layout_info = _run_tt_prefill_and_decode(
            decoder=decoder,
            mesh_device=mesh_device,
            rotary_emb=rotary_emb,
            hidden_states=hidden_states,
            decode_hidden_states=decode_hidden,
            page_table=page_table,
            page_table_tt=page_table_tt,
            page_block_size=page_block_size,
        )

    prefill_pcc = _assert_pcc("multichip real-weight prefill vs optimized single-chip", baseline_prefill, multi_prefill)
    decode_pcc = _assert_pcc("multichip real-weight traced decode vs optimized single-chip", baseline_replay_1, multi_replay_1)
    repeated_pcc = _assert_pcc("multichip real-weight repeated trace replay", multi_replay_1, multi_replay_2, threshold=0.9999)
    artifact_name = (
        "real_weight_correctness.json"
        if prefill_seq_len == 128
        else f"real_weight_correctness_prefill_{prefill_seq_len}.json"
    )
    _write_json(
        ARTIFACT_DIR / artifact_name,
        {
            "baseline": "OptimizedDecoder single-chip TTNN",
            "mesh_shape": list(TARGET_MESH_SHAPE),
            "prefill_seq_len": prefill_seq_len,
            "decode_current_pos": prefill_seq_len,
            "prefill_pcc_vs_single_chip": prefill_pcc,
            "traced_decode_replay_pcc_vs_single_chip": decode_pcc,
            "repeated_trace_replay_pcc": repeated_pcc,
            "threshold": PCC_THRESHOLD,
            "layout_info": layout_info,
            "status": "passed",
        },
    )


def test_runtime_fallback_audit_measured_multichip_prefill_and_traced_decode(monkeypatch):
    _require_target_mesh()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256

    with _open_mesh(TARGET_MESH_SHAPE) as mesh_device:
        page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=71
        )
        decoder = _make_decoder(
            MultichipDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
        )
        torch.manual_seed(73)
        hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
        _, tt_rot_mats = _meta_rot_mats_prefill(rotary_emb, hidden_states, 0, prefill_seq_len, mesh_device)
        tt_prefill_input = _to_tt_prefill(hidden_states, mesh_device)
        decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
        decode_input_tt = _to_tt_decode(decode_hidden, decoder, mesh_device)
        current_pos_host = torch.tensor([prefill_seq_len], dtype=torch.int32)
        current_pos_tt = _current_pos_to_tt(current_pos_host, mesh_device)
        rot_mats = DecodeRotaryHelper(rotary_emb, prefill_seq_len + 2, hf_config.head_dim, mesh_device).get_rot_mats(
            current_pos_host
        )

        decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
        decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
        ttnn.synchronize_device(mesh_device)

        def forbidden_host_bridge(*_args, **_kwargs):
            raise AssertionError("host fallback bridge called inside measured multichip TTNN pass")

        monkeypatch.setattr(ttnn, "from_torch", forbidden_host_bridge)
        monkeypatch.setattr(ttnn, "to_torch", forbidden_host_bridge, raising=False)

        prefill_out = decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        decode_out = decoder.decode_forward(
            decode_input_tt,
            current_pos=current_pos_tt,
            rot_mats=rot_mats,
            page_table=page_table_tt,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ttnn.release_trace(mesh_device, trace_id)

    assert prefill_out is not None
    assert decode_out is not None
    payload = {
        "mesh_shape": list(TARGET_MESH_SHAPE),
        "prefill_seq_len": prefill_seq_len,
        "decode_current_pos": prefill_seq_len,
        "guarded_python_bridges": ["ttnn.from_torch", "ttnn.to_torch"],
        "measured_passes": ["prefill_forward", "decode_forward_trace_capture_and_replay"],
        "status": "passed",
    }
    _write_json(ARTIFACT_DIR / "runtime_fallback_audit.json", payload)
    if os.getenv("TT_METAL_WATCHER"):
        _write_json(
            ARTIFACT_DIR / "watcher" / "watcher_summary.json",
            {
                **payload,
                "watcher_env": {
                    "TT_METAL_WATCHER": os.getenv("TT_METAL_WATCHER"),
                    "TT_METAL_WATCHER_NOINLINE": os.getenv("TT_METAL_WATCHER_NOINLINE"),
                    "TT_METAL_WATCHER_APPEND": os.getenv("TT_METAL_WATCHER_APPEND"),
                },
            },
        )


def test_multichip_repeated_run_stress():
    _require_target_mesh()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256
    torch.manual_seed(11)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15

    with _open_mesh((1, 1)) as mesh_device:
        page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=43
        )
        baseline = _make_decoder(
            OptimizedDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
        )
        baseline_prefill, _, _, _, _ = _run_tt_prefill_and_decode(
            decoder=baseline,
            mesh_device=mesh_device,
            rotary_emb=rotary_emb,
            hidden_states=hidden_states,
            decode_hidden_states=decode_hidden,
            page_table=page_table,
            page_table_tt=page_table_tt,
            page_block_size=page_block_size,
        )

    pccs = []
    with _open_mesh(TARGET_MESH_SHAPE) as mesh_device:
        multi_page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=43
        )
        assert torch.equal(page_table, multi_page_table)
        for _ in range(int(os.getenv("MD_STRESS_ITERS", "3"))):
            decoder = _make_decoder(
                MultichipDecoder,
                state_dict,
                hf_config,
                mesh_device,
                page_block_size=page_block_size,
                max_seq_len=max_seq_len,
            )
            multi_prefill, _, _, _, _ = _run_tt_prefill_and_decode(
                decoder=decoder,
                mesh_device=mesh_device,
                rotary_emb=rotary_emb,
                hidden_states=hidden_states,
                decode_hidden_states=decode_hidden,
                page_table=page_table,
                page_table_tt=page_table_tt,
                page_block_size=page_block_size,
            )
            pccs.append(_assert_pcc("multichip repeated prefill vs optimized single-chip", baseline_prefill, multi_prefill))

    _write_json(
        ARTIFACT_DIR / "stress_repeated_runs.json",
        {
            "iterations": len(pccs),
            "mesh_shape": list(TARGET_MESH_SHAPE),
            "prefill_seq_len": prefill_seq_len,
            "prefill_pccs_vs_single_chip": pccs,
            "threshold": PCC_THRESHOLD,
            "status": "passed",
        },
    )


@pytest.mark.perf_artifact
def test_perf_artifact_signposted_multichip_prefill_and_decode():
    from tracy import signpost

    _require_target_mesh()
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = int(os.getenv("MD_PERF_PREFILL_SEQ_LEN", "128"))
    max_seq_len = max(256, prefill_seq_len + page_block_size)
    torch.manual_seed(53)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15

    with _open_mesh(TARGET_MESH_SHAPE) as mesh_device:
        page_table, page_table_tt = _make_page_table(
            mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=53
        )
        decoder = _make_decoder(
            MultichipDecoder,
            state_dict,
            hf_config,
            mesh_device,
            page_block_size=page_block_size,
            max_seq_len=max_seq_len,
        )
        _, tt_rot_mats = _meta_rot_mats_prefill(rotary_emb, hidden_states, 0, prefill_seq_len, mesh_device)
        tt_prefill_input = _to_tt_prefill(hidden_states, mesh_device)

        decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
        ttnn.synchronize_device(mesh_device)
        prefill_start = time.perf_counter()
        signpost("PERF_MULTICHIP_PREFILL")
        prefill_out = decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
        ttnn.synchronize_device(mesh_device)
        signpost("PERF_MULTICHIP_PREFILL_END")
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        assert prefill_out is not None

        current_pos_host = torch.tensor([prefill_seq_len], dtype=torch.int32)
        current_pos_tt = _current_pos_to_tt(current_pos_host, mesh_device)
        rot_mats = DecodeRotaryHelper(rotary_emb, prefill_seq_len + 2, hf_config.head_dim, mesh_device).get_rot_mats(
            current_pos_host
        )
        decode_input_tt = _to_tt_decode(decode_hidden, decoder, mesh_device)

        decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        decode_out = decoder.decode_forward(
            decode_input_tt,
            current_pos=current_pos_tt,
            rot_mats=rot_mats,
            page_table=page_table_tt,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        decode_start = time.perf_counter()
        signpost("PERF_MULTICHIP_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost("PERF_MULTICHIP_DECODE_END")
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        ttnn.release_trace(mesh_device, trace_id)
        assert decode_out is not None

    optimized_provenance = ARTIFACT_DIR.parent / "optimized_decoder" / "perf" / "perf_provenance.json"
    before = json.loads(optimized_provenance.read_text()) if optimized_provenance.exists() else {}
    optimized_after = before.get("after_optimized", before)
    _write_json(
        ARTIFACT_DIR / "perf_trace_contract.json",
        {
            "mesh_shape": list(TARGET_MESH_SHAPE),
            "prefill_seq_len": prefill_seq_len,
            "decode_current_pos": prefill_seq_len,
            "prefill_signposts": ["PERF_MULTICHIP_PREFILL", "PERF_MULTICHIP_PREFILL_END"],
            "decode_signposts": ["PERF_MULTICHIP_DECODE", "PERF_MULTICHIP_DECODE_END"],
            "decode_measurement": "single warmed ttnn.execute_trace replay on 1x8 Ring",
            "host_wall_ms": {
                "multichip_prefill": prefill_ms,
                "multichip_traced_decode_replay": decode_ms,
            },
            "single_chip_baseline_device_us_from_existing_artifact": {
                "prefill": optimized_after.get("prefill_device_time_us"),
                "traced_decode_replay": optimized_after.get("traced_decode_replay_device_time_us"),
            },
            "status": "passed",
        },
    )
