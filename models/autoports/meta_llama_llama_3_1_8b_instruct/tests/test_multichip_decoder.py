# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    PAGE_BLOCK_SIZE,
    PCC_THRESHOLD,
    _assert_no_host_fallback,
    _assert_pcc,
    _decode_rot_mats,
    _hf_config,
    _hf_rotary,
    _page_table,
    _rope_setup,
    _synthetic_state_dict,
    _tt_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    TARGET_TOPOLOGY,
    MultiChipDecoder,
    MultiChipDecoderPolicy,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)
from models.common.auto_compose import to_torch_auto_compose
from models.common.auto_compose import extract_tensor_topology_info

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - only absent outside profiling runs

    def signpost(header: str) -> None:
        del header


FULL_CACHE_SEQ_LEN = 128 * 1024
MULTICHIP_TRACE_REGION_SIZE = 100_000_000
_DTYPE_ENV_MAP = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}
_MATH_FIDELITY_ENV_MAP = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def _optional_dtype_env(name: str) -> ttnn.DataType | None:
    value = os.environ.get(name)
    if not value:
        return None
    if value not in _DTYPE_ENV_MAP:
        raise ValueError(f"{name}={value!r} is unsupported; expected one of {sorted(_DTYPE_ENV_MAP)}")
    return _DTYPE_ENV_MAP[value]


def _multichip_policy_overrides() -> dict[str, object]:
    overrides: dict[str, object] = {}
    fidelity = os.environ.get("MULTICHIP_DECODER_MLP_MATH_FIDELITY")
    if fidelity:
        if fidelity not in _MATH_FIDELITY_ENV_MAP:
            raise ValueError(
                f"MULTICHIP_DECODER_MLP_MATH_FIDELITY={fidelity!r} is unsupported; "
                f"expected one of {sorted(_MATH_FIDELITY_ENV_MAP)}"
            )
        overrides["policy"] = MultiChipDecoderPolicy(mlp_math_fidelity=_MATH_FIDELITY_ENV_MAP[fidelity])

    env_to_kwarg = {
        "MULTICHIP_DECODER_ACTIVATION_DTYPE": "activation_dtype",
        "MULTICHIP_DECODER_ATTENTION_WEIGHT_DTYPE": "weight_dtype",
        "MULTICHIP_DECODER_KV_CACHE_DTYPE": "kv_cache_dtype",
        "MULTICHIP_DECODER_MLP_GATE_UP_DTYPE": "mlp_gate_up_dtype",
        "MULTICHIP_DECODER_MLP_DOWN_DTYPE": "mlp_down_dtype",
    }
    for env_name, kwarg_name in env_to_kwarg.items():
        dtype = _optional_dtype_env(env_name)
        if dtype is not None:
            overrides[kwarg_name] = dtype
    return overrides


@contextmanager
def _opened_mesh(
    shape: tuple[int, int],
    *,
    trace_region_size: int = 0,
    fabric_config: ttnn.FabricConfig | None = None,
):
    requested = shape[0] * shape[1]
    if requested > ttnn.get_num_devices():
        pytest.skip(f"requested {requested} devices, only {ttnn.get_num_devices()} available")

    if fabric_config is not None:
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
        )

    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*shape),
        trace_region_size=trace_region_size,
    )
    original_default_device = ttnn.GetDefaultDevice()
    ttnn.SetDefaultDevice(mesh)
    try:
        yield mesh
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)
        if fabric_config is not None:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _current_pos_tensor(mesh_device: ttnn.MeshDevice, current_pos_value: int, batch: int):
    current_pos_host = torch.full((batch,), current_pos_value, dtype=torch.int32)
    current_pos = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return current_pos_host, current_pos


def _run_optimized_baseline(
    state_dict: dict[str, torch.Tensor],
    *,
    prefill_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    seq_len: int,
    max_seq_len: int,
    max_num_blocks: int,
    emit_perf_signposts: bool = True,
    decode_replays: int = 4,
):
    hf_config = _hf_config()
    batch = prefill_hidden.shape[0]
    current_pos_value = seq_len

    with _opened_mesh((1, 1), trace_region_size=0) as mesh_device:
        rotary_emb = _hf_rotary(hf_config)
        decoder = OptimizedDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh_device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            page_block_size=PAGE_BLOCK_SIZE,
            max_num_blocks=max_num_blocks,
        )
        _, page_table_tt = _page_table(mesh_device, batch=batch, max_num_blocks=max_num_blocks)
        rope_setup = _rope_setup(mesh_device, hf_config, rotary_emb, max_seq_len + 1, batch)

        warm_prefill = decoder.prefill_forward(
            _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0)),
            rot_mats=tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len)),
            page_table=page_table_tt,
            user_id=0,
        )
        ttnn.synchronize_device(mesh_device)
        del warm_prefill

        if emit_perf_signposts:
            signpost(header="PERF_BASELINE_PREFILL")
        prefill_start = time.perf_counter()
        tt_prefill = decoder.prefill_forward(
            _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0)),
            rot_mats=tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len)),
            page_table=page_table_tt,
            user_id=0,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        if emit_perf_signposts:
            signpost(header="PERF_BASELINE_PREFILL_END")
        prefill_host = to_torch_auto_compose(tt_prefill)[:, 0, :seq_len, :].reshape(
            batch, seq_len, hf_config.hidden_size
        )

        current_pos_host, current_pos = _current_pos_tensor(mesh_device, current_pos_value, batch)
        decode_rot_mats = _decode_rot_mats(rope_setup, current_pos_host.to(torch.long))
        tt_decode_input = ttnn.to_memory_config(
            _tt_tensor(mesh_device, decode_hidden.unsqueeze(0)),
            decoder.decode_residual_memcfg,
        )

        warm = decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            rot_mats=decode_rot_mats,
            page_table=page_table_tt,
        )
        del warm

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_out = decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            rot_mats=decode_rot_mats,
            page_table=page_table_tt,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        decode_ms_samples = []
        for replay_idx in range(decode_replays):
            if emit_perf_signposts and replay_idx == 0:
                signpost(header="PERF_BASELINE_DECODE")
            decode_start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            decode_ms_samples.append((time.perf_counter() - decode_start) * 1000.0)
            if emit_perf_signposts and replay_idx == 0:
                signpost(header="PERF_BASELINE_DECODE_END")
        decode_host = to_torch_auto_compose(traced_out)[:, 0, :batch, :].reshape(
            batch, 1, hf_config.hidden_size
        )
        ttnn.release_trace(mesh_device, trace_id)

    return {
        "prefill": prefill_host,
        "decode": decode_host,
        "prefill_ms_e2e": prefill_ms,
        "decode_ms_e2e_samples": decode_ms_samples,
        "decode_ms_e2e_min": min(decode_ms_samples),
        "decode_ms_e2e_avg": sum(decode_ms_samples) / len(decode_ms_samples),
    }


def _run_multichip_case(
    state_dict: dict[str, torch.Tensor],
    *,
    prefill_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    seq_len: int,
    max_seq_len: int,
    max_num_blocks: int,
    emit_perf_signposts: bool = True,
    decode_replays: int = 4,
):
    hf_config = _hf_config()
    batch = prefill_hidden.shape[0]
    current_pos_value = seq_len

    with _opened_mesh(
        TARGET_MESH_SHAPE,
        trace_region_size=MULTICHIP_TRACE_REGION_SIZE,
        fabric_config=ttnn.FabricConfig.FABRIC_1D_RING,
    ) as mesh_device:
        rotary_emb = _hf_rotary(hf_config)
        decoder_overrides = _multichip_policy_overrides()
        decoder = MultiChipDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh_device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            page_block_size=PAGE_BLOCK_SIZE,
            max_num_blocks=max_num_blocks,
            **decoder_overrides,
        )
        page_table, page_table_tt = _page_table(mesh_device, batch=batch, max_num_blocks=max_num_blocks)
        assert page_table.shape == (batch, max_num_blocks)
        assert int(page_table[0, 0]) != 0 or int(page_table[0, 1]) != 1
        rope_setup = _rope_setup(mesh_device, hf_config, rotary_emb, max_seq_len + 1, batch)

        decoder.input_layernorm.load_device_weights()
        decoder.self_attn.load_device_weights()
        decoder.post_attention_layernorm.load_device_weights()
        decoder.mlp.load_device_weights()
        key_cache, value_cache = decoder.self_attn.kv_cache
        assert key_cache.dtype == decoder.policy.kv_cache_dtype
        assert value_cache.dtype == decoder.policy.kv_cache_dtype
        assert key_cache.shape[0] == max_num_blocks
        assert value_cache.shape[0] == max_num_blocks
        assert key_cache.shape[1] == 1
        assert value_cache.shape[1] == 1
        assert key_cache.shape[2] == PAGE_BLOCK_SIZE
        assert value_cache.shape[2] == PAGE_BLOCK_SIZE

        prefill_input = _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0))
        prefill_rot_mats = tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len))
        with (
            patch.object(FunctionalDecoder, "prefill_forward", side_effect=AssertionError("functional fallback")),
            patch.object(OptimizedDecoder, "prefill_forward", side_effect=AssertionError("optimized fallback")),
            _assert_no_host_fallback(),
        ):
            audit_prefill = decoder.prefill_forward(
                prefill_input,
                rot_mats=prefill_rot_mats,
                page_table=page_table_tt,
                user_id=0,
            )
        ttnn.synchronize_device(mesh_device)
        del audit_prefill

        if emit_perf_signposts:
            signpost(header="PERF_MULTICHIP_PREFILL")
        prefill_start = time.perf_counter()
        tt_prefill = decoder.prefill_forward(
            _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0)),
            rot_mats=tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len)),
            page_table=page_table_tt,
            user_id=0,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        if emit_perf_signposts:
            signpost(header="PERF_MULTICHIP_PREFILL_END")

        placements, _ = extract_tensor_topology_info(tt_prefill)
        assert all(isinstance(placement, ttnn.PlacementReplicate) for placement in placements)
        prefill_host = to_torch_auto_compose(tt_prefill)[:, 0, :seq_len, :].reshape(
            batch, seq_len, hf_config.hidden_size
        )

        current_pos_host, current_pos = _current_pos_tensor(mesh_device, current_pos_value, batch)
        decode_rot_mats = _decode_rot_mats(rope_setup, current_pos_host.to(torch.long))
        tt_decode_input = ttnn.to_memory_config(
            _tt_tensor(mesh_device, decode_hidden.unsqueeze(0)),
            decoder.decode_residual_memcfg,
        )

        tt_warm = decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            rot_mats=decode_rot_mats,
            page_table=page_table_tt,
        )
        with (
            patch.object(FunctionalDecoder, "decode_forward", side_effect=AssertionError("functional fallback")),
            patch.object(OptimizedDecoder, "decode_forward", side_effect=AssertionError("optimized fallback")),
            _assert_no_host_fallback(),
        ):
            tt_audit = decoder.decode_forward(
                tt_decode_input,
                current_pos=current_pos,
                rot_mats=decode_rot_mats,
                page_table=page_table_tt,
            )
        del tt_audit

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_out = decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            rot_mats=decode_rot_mats,
            page_table=page_table_tt,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first_replay = to_torch_auto_compose(traced_out)[:, 0, :batch, :].reshape(
            batch, 1, hf_config.hidden_size
        )

        decode_ms_samples = []
        replay_outputs = []
        for replay_idx in range(decode_replays):
            if emit_perf_signposts and replay_idx == 0:
                signpost(header="PERF_MULTICHIP_DECODE")
            decode_start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            decode_ms_samples.append((time.perf_counter() - decode_start) * 1000.0)
            if emit_perf_signposts and replay_idx == 0:
                signpost(header="PERF_MULTICHIP_DECODE_END")
            replay_outputs.append(
                to_torch_auto_compose(traced_out)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size)
            )
        ttnn.release_trace(mesh_device, trace_id)

        eager_decode = to_torch_auto_compose(tt_warm)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size)
        determinism_pcc = _assert_pcc(
            "multichip_decode_trace_repeated_input",
            first_replay,
            replay_outputs[-1],
            threshold=0.9999,
        )
        eager_trace_pcc = _assert_pcc(
            "multichip_decode_eager_vs_trace",
            eager_decode,
            first_replay,
            threshold=0.9999,
        )

    return {
        "prefill": prefill_host,
        "decode": first_replay,
        "prefill_ms_e2e": prefill_ms,
        "decode_ms_e2e_samples": decode_ms_samples,
        "decode_ms_e2e_min": min(decode_ms_samples),
        "decode_ms_e2e_avg": sum(decode_ms_samples) / len(decode_ms_samples),
        "determinism_pcc": determinism_pcc,
        "eager_trace_pcc": eager_trace_pcc,
        "runtime_fallback_audit": "multichip_prefill_decode_clean",
        "page_block_size": PAGE_BLOCK_SIZE,
        "max_num_blocks": max_num_blocks,
        "max_seq_len": max_seq_len,
        "mesh_shape": TARGET_MESH_SHAPE,
        "topology": str(TARGET_TOPOLOGY),
        "policy_name": decoder.policy.name,
        "activation_dtype": str(decoder.policy.activation_dtype),
        "attention_weight_dtype": str(decoder.policy.attention_weight_dtype),
        "mlp_gate_up_dtype": str(decoder.policy.mlp_gate_up_dtype),
        "mlp_down_dtype": str(decoder.policy.mlp_down_dtype),
        "kv_cache_dtype": str(decoder.policy.kv_cache_dtype),
        "mlp_math_fidelity": str(decoder.policy.mlp_math_fidelity),
    }


def test_multichip_decoder_contract_and_policy():
    assert Path("models/autoports/meta_llama_llama_3_1_8b_instruct/tt/multichip_decoder.py").exists()
    assert MultiChipDecoder.single_chip_baseline_cls is OptimizedDecoder
    assert "rot_mats" in inspect.signature(MultiChipDecoder.prefill_forward).parameters
    assert "page_table" in inspect.signature(MultiChipDecoder.prefill_forward).parameters
    assert "current_pos" in inspect.signature(MultiChipDecoder.decode_forward).parameters
    assert "page_table" in inspect.signature(MultiChipDecoder.decode_forward).parameters

    policy = MultiChipDecoderPolicy()
    optimized_policy = OptimizedDecoderPolicy()
    assert optimized_policy.attention_weight_dtype == ttnn.bfloat8_b
    assert policy.attention_weight_dtype == ttnn.bfloat4_b
    assert policy.mlp_gate_up_dtype == optimized_policy.mlp_gate_up_dtype == ttnn.bfloat4_b
    assert policy.mlp_down_dtype == optimized_policy.mlp_down_dtype == ttnn.bfloat4_b
    assert policy.kv_cache_dtype == optimized_policy.kv_cache_dtype == ttnn.bfloat8_b
    assert optimized_policy.activation_dtype == ttnn.bfloat16
    assert policy.activation_dtype == ttnn.bfloat8_b
    assert TARGET_MESH_SHAPE == (1, 8)
    assert TARGET_TOPOLOGY == ttnn.Topology.Ring


@pytest.mark.slow
def test_multichip_decoder_full_context_cache_contract():
    with _opened_mesh(
        TARGET_MESH_SHAPE,
        trace_region_size=MULTICHIP_TRACE_REGION_SIZE,
        fabric_config=ttnn.FabricConfig.FABRIC_1D_RING,
    ) as mesh_device:
        hf_config = _hf_config()
        decoder = MultiChipDecoder.from_state_dict(
            _synthetic_state_dict(),
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=FULL_CACHE_SEQ_LEN,
            page_block_size=PAGE_BLOCK_SIZE,
            max_num_blocks=FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE,
        )
        decoder.self_attn.load_device_weights()
        key_cache, value_cache = decoder.self_attn.kv_cache
        assert key_cache.shape[0] == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE
        assert value_cache.shape[0] == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE
        assert key_cache.shape[1] == hf_config.num_key_value_heads // 8
        assert value_cache.shape[1] == hf_config.num_key_value_heads // 8
        assert key_cache.shape[2] == PAGE_BLOCK_SIZE
        assert value_cache.shape[2] == PAGE_BLOCK_SIZE
        assert decoder.self_attn.config.max_seq_len == FULL_CACHE_SEQ_LEN
        assert decoder.self_attn.config.paged_attention_config.max_num_blocks == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE


@pytest.mark.slow
def test_multichip_decoder_synthetic_paged_prefill_decode_trace_against_optimized():
    seq_len = 128
    max_seq_len = 256
    max_num_blocks = max(2, (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE)
    decode_replays = int(os.environ.get("MULTICHIP_DECODER_DECODE_REPLAYS", "4"))
    hf_config = _hf_config()
    state_dict = _synthetic_state_dict()

    torch.manual_seed(123)
    prefill_hidden = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05

    baseline = _run_optimized_baseline(
        state_dict,
        prefill_hidden=prefill_hidden,
        decode_hidden=decode_hidden,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        max_num_blocks=max_num_blocks,
        decode_replays=decode_replays,
    )
    multichip = _run_multichip_case(
        state_dict,
        prefill_hidden=prefill_hidden,
        decode_hidden=decode_hidden,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        max_num_blocks=max_num_blocks,
        decode_replays=decode_replays,
    )

    prefill_pcc = _assert_pcc(
        "multichip_prefill_vs_optimized",
        baseline["prefill"],
        multichip["prefill"],
        threshold=PCC_THRESHOLD,
    )
    decode_pcc = _assert_pcc(
        "multichip_decode_trace_vs_optimized_trace",
        baseline["decode"],
        multichip["decode"],
        threshold=PCC_THRESHOLD,
    )

    speedup = baseline["decode_ms_e2e_min"] / multichip["decode_ms_e2e_min"]
    efficiency = speedup / TARGET_MESH_SHAPE[1]
    metrics = {
        "prefill_pcc_vs_optimized": prefill_pcc,
        "decode_pcc_vs_optimized": decode_pcc,
        "single_chip_prefill_ms_e2e": baseline["prefill_ms_e2e"],
        "single_chip_decode_ms_e2e_min": baseline["decode_ms_e2e_min"],
        "single_chip_decode_ms_e2e_avg": baseline["decode_ms_e2e_avg"],
        "single_chip_decode_ms_e2e_samples": baseline["decode_ms_e2e_samples"],
        "multichip_prefill_ms_e2e": multichip["prefill_ms_e2e"],
        "multichip_decode_ms_e2e_min": multichip["decode_ms_e2e_min"],
        "multichip_decode_ms_e2e_avg": multichip["decode_ms_e2e_avg"],
        "speedup": speedup,
        "efficiency": efficiency,
        **{k: v for k, v in multichip.items() if k not in {"prefill", "decode"}},
    }
    logger.info(f"multichip decoder metrics: {metrics}")


@pytest.mark.slow
def test_multichip_decoder_synthetic_paged_prefill_decode_trace_profile_only():
    if os.environ.get("MULTICHIP_DECODER_PROFILE_ONLY") != "1":
        pytest.skip("profile-only multichip run is enabled by MULTICHIP_DECODER_PROFILE_ONLY=1")

    seq_len = 128
    max_seq_len = 256
    max_num_blocks = max(2, (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE)
    decode_replays = int(os.environ.get("MULTICHIP_DECODER_DECODE_REPLAYS", "4"))
    hf_config = _hf_config()

    torch.manual_seed(123)
    prefill_hidden = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05

    multichip = _run_multichip_case(
        _synthetic_state_dict(),
        prefill_hidden=prefill_hidden,
        decode_hidden=decode_hidden,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        max_num_blocks=max_num_blocks,
        decode_replays=decode_replays,
    )
    assert multichip["runtime_fallback_audit"] == "multichip_prefill_decode_clean"
    assert multichip["mesh_shape"] == TARGET_MESH_SHAPE
    assert multichip["determinism_pcc"] >= 0.9999
    assert multichip["eager_trace_pcc"] >= 0.9999
    logger.info(f"profile-only multichip decoder metrics: {multichip}")
