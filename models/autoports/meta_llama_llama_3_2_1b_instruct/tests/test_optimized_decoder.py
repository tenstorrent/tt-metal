# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
import time
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
    _hf_and_meta_position_embeddings,
    _make_hf_layer_and_rotary,
    _make_page_table,
    _meta_rot_mats_prefill,
    _real_layer_state_dict,
    _run_decode_trace,
    _run_prefill,
    _synthetic_layer_state_dict,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPrecisionPolicy,
    _OptimizedLlamaMLP,
)


ARTIFACT_DIR = Path("models/autoports/meta_llama_llama_3_2_1b_instruct/doc/optimized_decoder")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _make_decoder(
    state_dict: dict[str, torch.Tensor],
    hf_config,
    mesh_device: ttnn.MeshDevice,
    *,
    page_block_size: int,
    max_seq_len: int,
    max_batch_size: int = 1,
    precision_policy: OptimizedDecoderPrecisionPolicy | None = None,
) -> OptimizedDecoder:
    return OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        precision_policy=precision_policy,
    )


def _config_summary(decoder: OptimizedDecoder) -> dict:
    attention_cfg = decoder.attention.config
    mlp_cfg = decoder.mlp.config
    prefill_qkv_cfg = attention_cfg.prefill_xqkv_prg_config(8192)
    prefill_wo_cfg = attention_cfg.prefill_wo_prg_config(8192)
    return {
        "precision_policy": decoder.precision_policy.to_dict(),
        "attention": {
            "wqkv_dtype": attention_cfg.wqkv.dtype.name,
            "wo_dtype": attention_cfg.wo.dtype.name,
            "kv_cache_dtype": attention_cfg.kv_cache_dtype.name,
            "activation_dtype": attention_cfg.activation_dtype.name,
            "decode_qkv_program_config": type(attention_cfg.decode_xqkv_prg_config).__name__,
            "decode_wo_program_config": type(attention_cfg.decode_attn_output_prg_config).__name__,
            "decode_sdpa_program_config": type(attention_cfg.decode_sdpa_prg_config).__name__,
            "prefill_qkv_program_config": type(attention_cfg.prefill_xqkv_prg_config(128)).__name__,
            "prefill_wo_program_config": type(attention_cfg.prefill_wo_prg_config(128)).__name__,
            "prefill_qkv_8192_in0_block_w": prefill_qkv_cfg.in0_block_w,
            "prefill_qkv_8192_out_subblock": [prefill_qkv_cfg.out_subblock_h, prefill_qkv_cfg.out_subblock_w],
            "prefill_wo_8192_in0_block_w": prefill_wo_cfg.in0_block_w,
            "prefill_wo_8192_out_subblock": [prefill_wo_cfg.out_subblock_h, prefill_wo_cfg.out_subblock_w],
            "decode_input_sharded": attention_cfg.decode_input_memcfg.is_sharded(),
            "decode_residual_sharded": attention_cfg.decode_residual_memcfg.is_sharded(),
            "uses_paged_attention": attention_cfg.paged_attention_config is not None,
        },
        "mlp": {
            "implementation": type(decoder.mlp).__name__,
            "gate_dtype": decoder.mlp.gate_weight_lazy.dtype.name,
            "up_dtype": decoder.mlp.up_weight_lazy.dtype.name,
            "down_dtype": decoder.mlp.down_weight_lazy.dtype.name,
            "decode_w1_w3_program_config": type(mlp_cfg.decode_w1_w3_prg_config).__name__,
            "decode_w2_program_config": type(mlp_cfg.decode_w2_prg_config).__name__,
            "decode_w1_w3_in0_block_w": mlp_cfg.decode_w1_w3_prg_config.in0_block_w,
            "decode_w2_in0_block_w": mlp_cfg.decode_w2_prg_config.in0_block_w,
            "decode_input_sharded": mlp_cfg.decode_input_memcfg.is_sharded(),
            "decode_w1_w3_output_sharded": mlp_cfg.decode_w1_w3_output_memcfg.is_sharded(),
            "decode_w2_input_sharded": mlp_cfg.decode_w2_input_memcfg.is_sharded(),
            "decode_residual_sharded": mlp_cfg.decode_residual_memcfg.is_sharded(),
            "prefill_w1_w3_program_config": type(mlp_cfg.prefill_w1_w3_prg_config(128)).__name__,
            "prefill_w2_program_config": type(mlp_cfg.prefill_w2_prg_config(128)).__name__,
        },
    }


def test_optimized_decoder_contract_and_runtime_fallback_audit():
    optimized_source = inspect.getsource(OptimizedDecoder.prefill_forward)
    optimized_source += inspect.getsource(OptimizedDecoder.decode_forward)
    optimized_source += inspect.getsource(OptimizedDecoder.kv_cache.fget)
    optimized_source += inspect.getsource(_OptimizedLlamaMLP.prefill_forward)
    optimized_source += inspect.getsource(_OptimizedLlamaMLP.decode_forward)

    for forbidden in ("torch", "from_torch", "to_torch", "cpu("):
        assert forbidden not in optimized_source

    assert "FunctionalDecoder" not in inspect.getsource(OptimizedDecoder)
    assert "_LlamaMLP" not in inspect.getsource(OptimizedDecoder)
    module_source = inspect.getsource(__import__(_OptimizedLlamaMLP.__module__, fromlist=["_"]))
    assert "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" in module_source


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_static_config_uses_optimized_path(mesh_device: ttnn.MeshDevice):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    decoder = _make_decoder(
        _synthetic_layer_state_dict(hf_config),
        hf_config,
        mesh_device,
        page_block_size=64,
        max_seq_len=256,
    )

    assert isinstance(decoder.mlp, _OptimizedLlamaMLP)
    assert decoder.precision_policy.attention_weight_dtype == ttnn.bfloat8_b
    assert decoder.precision_policy.mlp_ff1_ff3_weight_dtype == ttnn.bfloat4_b
    assert decoder.precision_policy.mlp_ff2_weight_dtype == ttnn.bfloat4_b
    assert decoder.precision_policy.kv_cache_dtype == ttnn.bfloat8_b
    assert decoder.attention.config.paged_attention_config is not None
    assert decoder.attention.config.decode_xqkv_prg_config.__class__.__name__.endswith("DRAMShardedProgramConfig")
    assert decoder.attention.config.decode_attn_output_prg_config.__class__.__name__.endswith("DRAMShardedProgramConfig")
    assert decoder.attention.config.prefill_xqkv_prg_config(8192).in0_block_w == 8
    assert decoder.attention.config.prefill_wo_prg_config(8192).in0_block_w == 8
    assert decoder.mlp.config.decode_w1_w3_prg_config.__class__.__name__.endswith("DRAMShardedProgramConfig")
    assert decoder.mlp.config.decode_w2_prg_config.__class__.__name__.endswith("DRAMShardedProgramConfig")
    assert decoder.mlp.config.decode_w1_w3_prg_config.in0_block_w >= 2
    assert decoder.mlp.config.decode_w2_prg_config.in0_block_w >= 2

    _write_json(ARTIFACT_DIR / "optimized_config_summary.json", _config_summary(decoder))


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_runtime_fallback_audit_measured_optimized_prefill_and_traced_decode(
    mesh_device: ttnn.MeshDevice, monkeypatch
):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256
    _, page_table_tt = _make_page_table(
        mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=71
    )
    decoder = _make_decoder(
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
    current_pos_tt = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = DecodeRotaryHelper(rotary_emb, prefill_seq_len + 2, hf_config.head_dim, mesh_device).get_rot_mats(
        current_pos_host
    )

    decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)

    def forbidden_host_bridge(*_args, **_kwargs):
        raise AssertionError("host fallback bridge called inside measured optimized TTNN pass")

    monkeypatch.setattr(ttnn, "from_torch", forbidden_host_bridge)
    monkeypatch.setattr(ttnn, "to_torch", forbidden_host_bridge, raising=False)

    prefill_out = decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decode_out = decoder.decode_forward(
        decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ttnn.release_trace(mesh_device, trace_id)

    assert prefill_out is not None
    assert decode_out is not None
    _write_json(
        ARTIFACT_DIR / "runtime_fallback_audit.json",
        {
            "prefill_seq_len": prefill_seq_len,
            "decode_current_pos": prefill_seq_len,
            "guarded_python_bridges": ["ttnn.from_torch", "ttnn.to_torch"],
            "measured_passes": ["prefill_forward", "decode_forward_trace_capture_and_replay"],
            "optimized_path": type(decoder.mlp).__name__,
            "status": "passed",
        },
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_synthetic_optimized_paged_prefill_decode_trace_and_determinism(mesh_device: ttnn.MeshDevice):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    hf_layer, rotary_emb = _make_hf_layer_and_rotary(hf_config, state_dict)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256
    page_table, page_table_tt = _make_page_table(
        mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=13
    )
    assert not torch.equal(page_table, torch.arange(page_table.numel(), dtype=torch.int32).reshape_as(page_table))

    decoder = _make_decoder(
        state_dict,
        hf_config,
        mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
    )

    torch.manual_seed(5)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_prefill, tt_prefill = _run_prefill(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        hidden_states=hidden_states,
    )
    prefill_pcc = _assert_pcc("optimized synthetic prefill", hf_prefill, tt_prefill)

    prefill_hf_pos_emb, _ = _hf_and_meta_position_embeddings(
        rotary_emb, hidden_states, torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0)
    )
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_decode, eager_decode, replay_decode_1, replay_decode_2 = _run_decode_trace(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        prefill_hidden_states=hidden_states,
        decode_hidden_states=decode_hidden,
        prefill_hf_pos_emb=prefill_hf_pos_emb,
        current_pos_value=prefill_seq_len,
    )
    eager_decode_pcc = _assert_pcc("optimized synthetic eager decode", hf_decode, eager_decode)
    replay_decode_pcc = _assert_pcc("optimized synthetic traced replay decode", hf_decode, replay_decode_1)
    repeated_pcc = _assert_pcc("optimized synthetic repeated traced decode", replay_decode_1, replay_decode_2, threshold=0.9999)

    _write_json(
        ARTIFACT_DIR / "synthetic_correctness.json",
        {
            "prefill_seq_len": prefill_seq_len,
            "decode_current_pos": prefill_seq_len,
            "page_block_size": page_block_size,
            "page_table": page_table.tolist(),
            "prefill_pcc": prefill_pcc,
            "eager_decode_pcc": eager_decode_pcc,
            "traced_decode_replay_pcc": replay_decode_pcc,
            "repeated_trace_replay_pcc": repeated_pcc,
            "threshold": PCC_THRESHOLD,
            "optimized_path": type(decoder.mlp).__name__,
        },
    )


@pytest.mark.real_weights
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_real_weights_optimized_paged_prefill_and_decode_trace(mesh_device: ttnn.MeshDevice):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _real_layer_state_dict()
    hf_layer, rotary_emb = _make_hf_layer_and_rotary(hf_config, state_dict)
    page_block_size = 64
    prefill_seq_len = int(os.getenv("OD_PREFILL_SEQ_LEN", "128"))
    max_seq_len = max(256, ((prefill_seq_len + page_block_size) + page_block_size - 1) // page_block_size * page_block_size)
    _, page_table_tt = _make_page_table(
        mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=29
    )
    decoder = _make_decoder(
        state_dict,
        hf_config,
        mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
    )

    torch.manual_seed(99)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_prefill, tt_prefill = _run_prefill(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        hidden_states=hidden_states,
    )
    prefill_pcc = _assert_pcc("optimized real-weight prefill", hf_prefill, tt_prefill)

    prefill_hf_pos_emb, _ = _hf_and_meta_position_embeddings(
        rotary_emb, hidden_states, torch.arange(prefill_seq_len, dtype=torch.long).unsqueeze(0)
    )
    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    hf_decode, _, replay_decode_1, replay_decode_2 = _run_decode_trace(
        decoder=decoder,
        hf_layer=hf_layer,
        rotary_emb=rotary_emb,
        mesh_device=mesh_device,
        page_table_tt=page_table_tt,
        prefill_hidden_states=hidden_states,
        decode_hidden_states=decode_hidden,
        prefill_hf_pos_emb=prefill_hf_pos_emb,
        current_pos_value=prefill_seq_len,
    )
    decode_pcc = _assert_pcc("optimized real-weight traced decode", hf_decode, replay_decode_1)
    repeated_pcc = _assert_pcc("optimized real-weight repeated traced decode", replay_decode_1, replay_decode_2, threshold=0.9999)

    artifact_name = (
        "real_weight_correctness.json"
        if prefill_seq_len == 128
        else f"real_weight_correctness_prefill_{prefill_seq_len}.json"
    )
    _write_json(
        ARTIFACT_DIR / artifact_name,
        {
            "prefill_seq_len": prefill_seq_len,
            "decode_current_pos": prefill_seq_len,
            "prefill_pcc": prefill_pcc,
            "traced_decode_replay_pcc": decode_pcc,
            "repeated_trace_replay_pcc": repeated_pcc,
            "threshold": PCC_THRESHOLD,
            "optimized_path": type(decoder.mlp).__name__,
            "precision_policy": decoder.precision_policy.to_dict(),
        },
    )


@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_repeated_run_stress(mesh_device: ttnn.MeshDevice):
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    hf_layer, rotary_emb = _make_hf_layer_and_rotary(hf_config, state_dict)
    page_block_size = 64
    prefill_seq_len = 128
    max_seq_len = 256
    _, page_table_tt = _make_page_table(
        mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=43
    )
    decoder = _make_decoder(
        state_dict,
        hf_config,
        mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
    )
    torch.manual_seed(11)
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15

    pccs = []
    for _ in range(int(os.getenv("OD_STRESS_ITERS", "3"))):
        hf_prefill, tt_prefill = _run_prefill(
            decoder=decoder,
            hf_layer=hf_layer,
            rotary_emb=rotary_emb,
            mesh_device=mesh_device,
            page_table_tt=page_table_tt,
            hidden_states=hidden_states,
        )
        pccs.append(_assert_pcc("optimized repeated prefill", hf_prefill, tt_prefill))

    _write_json(
        ARTIFACT_DIR / "stress_repeated_runs.json",
        {
            "iterations": len(pccs),
            "prefill_seq_len": prefill_seq_len,
            "prefill_pccs": pccs,
            "threshold": PCC_THRESHOLD,
            "status": "passed",
        },
    )


@pytest.mark.perf_artifact
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_perf_artifact_signposted_optimized_prefill_and_decode(mesh_device: ttnn.MeshDevice):
    from tracy import signpost

    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    state_dict = _synthetic_layer_state_dict(hf_config)
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    page_block_size = 64
    prefill_seq_len = int(os.getenv("OD_PERF_PREFILL_SEQ_LEN", "128"))
    max_seq_len = max(256, prefill_seq_len + page_block_size)
    _, page_table_tt = _make_page_table(mesh_device, batch=1, max_seq_len=max_seq_len, block_size=page_block_size, seed=53)
    decoder = _make_decoder(
        state_dict,
        hf_config,
        mesh_device,
        page_block_size=page_block_size,
        max_seq_len=max_seq_len,
    )
    hidden_states = torch.randn(1, prefill_seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    _, tt_rot_mats = _meta_rot_mats_prefill(rotary_emb, hidden_states, 0, prefill_seq_len, mesh_device)
    tt_prefill_input = _to_tt_prefill(hidden_states, mesh_device)

    decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    prefill_start = time.perf_counter()
    signpost("PERF_PREFILL")
    out = decoder.prefill_forward(tt_prefill_input, rot_mats=tt_rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_PREFILL_END")
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    assert out is not None

    decode_hidden = torch.randn(1, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.15
    current_pos_host = torch.tensor([prefill_seq_len], dtype=torch.int32)
    current_pos_tt = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = DecodeRotaryHelper(rotary_emb, prefill_seq_len + 2, hf_config.head_dim, mesh_device).get_rot_mats(
        current_pos_host
    )
    decode_input_tt = _to_tt_decode(decode_hidden, decoder, mesh_device)

    decoder.decode_forward(decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    decode_out = decoder.decode_forward(
        decode_input_tt, current_pos=current_pos_tt, rot_mats=rot_mats, page_table=page_table_tt
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    decode_start = time.perf_counter()
    signpost("PERF_DECODE")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost("PERF_DECODE_END")
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    ttnn.release_trace(mesh_device, trace_id)
    assert decode_out is not None

    functional_provenance = (
        Path("models/autoports/meta_llama_llama_3_2_1b_instruct/doc/functional_decoder/tracy/run_prefill_decode/perf_provenance.json")
    )
    before = json.loads(functional_provenance.read_text()) if functional_provenance.exists() else {}
    _write_json(
        ARTIFACT_DIR / "perf_trace_contract.json",
        {
            "prefill_seq_len": prefill_seq_len,
            "decode_current_pos": prefill_seq_len,
            "prefill_signposts": ["PERF_PREFILL", "PERF_PREFILL_END"],
            "decode_signposts": ["PERF_DECODE", "PERF_DECODE_END"],
            "decode_measurement": "single warmed ttnn.execute_trace replay",
            "host_wall_ms": {
                "optimized_prefill": prefill_ms,
                "optimized_traced_decode_replay": decode_ms,
            },
            "functional_before_device_us_from_existing_artifact": {
                "prefill": before.get("prefill_device_time_us"),
                "traced_decode_replay": before.get("decode_traced_replay_device_time_us"),
            },
            "optimized_path": type(decoder.mlp).__name__,
            "precision_policy": decoder.precision_policy.to_dict(),
        },
    )
