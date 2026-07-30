# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import time
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
    _real_state_dict,
    _reference_decode,
    _reference_layer,
    _reference_prefill,
    _rope_setup,
    _synthetic_state_dict,
    _tt_tensor,
    device_params,
    mesh_device,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)
from models.common.auto_compose import to_torch_auto_compose

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - only absent outside profiling runs

    def signpost(header: str) -> None:
        del header


FULL_CACHE_SEQ_LEN = 128 * 1024


def test_optimized_decoder_contract_and_policy():
    assert Path("models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py").exists()
    assert "rot_mats" in inspect.signature(OptimizedDecoder.prefill_forward).parameters
    assert "page_table" in inspect.signature(OptimizedDecoder.prefill_forward).parameters
    assert "current_pos" in inspect.signature(OptimizedDecoder.decode_forward).parameters
    assert "page_table" in inspect.signature(OptimizedDecoder.decode_forward).parameters

    policy = OptimizedDecoderPolicy()
    assert policy.attention_weight_dtype == ttnn.bfloat8_b
    assert policy.mlp_gate_up_dtype == ttnn.bfloat4_b
    assert policy.mlp_down_dtype == ttnn.bfloat4_b
    assert policy.kv_cache_dtype == ttnn.bfloat8_b
    assert policy.activation_dtype == ttnn.bfloat16
    assert policy.mlp_math_fidelity == ttnn.MathFidelity.LoFi
    assert not issubclass(OptimizedDecoder, FunctionalDecoder)


def _run_optimized_prefill_decode_trace_case(
    mesh_device: ttnn.MeshDevice,
    state_dict: dict[str, torch.Tensor],
    *,
    real_weights: bool,
    seq_len: int = 128,
    max_seq_len: int | None = None,
    max_num_blocks: int | None = None,
    emit_perf_signposts: bool = True,
    decode_replays: int = 2,
    policy: OptimizedDecoderPolicy | None = None,
    **decoder_kwargs,
):
    hf_config = _hf_config()
    batch = 1
    assert seq_len % 128 == 0
    max_seq_len = max_seq_len or max(seq_len + 1, 256)
    max_num_blocks = max_num_blocks or max(2, (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE)
    current_pos_value = seq_len

    torch.manual_seed(123)
    reference_layer = _reference_layer(hf_config, state_dict)
    rotary_emb = _hf_rotary(hf_config)
    tt_decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_batch_size=batch,
        max_seq_len=max_seq_len,
        page_block_size=PAGE_BLOCK_SIZE,
        max_num_blocks=max_num_blocks,
        policy=policy,
        **decoder_kwargs,
    )

    assert tt_decoder.policy.attention_weight_dtype != ttnn.bfloat16
    assert tt_decoder.input_layernorm.config.decode_in_sharded
    assert tt_decoder.input_layernorm.config.decode_out_sharded
    assert isinstance(tt_decoder.mlp.decode_gate_up_prg_config, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)
    assert isinstance(tt_decoder.mlp.decode_down_prg_config, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig)

    page_table, page_table_tt = _page_table(mesh_device, batch=batch, max_num_blocks=max_num_blocks)
    assert page_table.shape == (batch, max_num_blocks)
    assert int(page_table[0, 0]) != 0 or int(page_table[0, 1]) != 1

    rope_setup = _rope_setup(mesh_device, hf_config, rotary_emb, max_seq_len + 1, batch)

    prefill_hidden = torch.randn(batch, seq_len, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05
    ref_prefill, ref_cache = _reference_prefill(reference_layer, rotary_emb, prefill_hidden)
    tt_prefill = tt_decoder.prefill_forward(
        _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0)),
        rot_mats=tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len)),
        page_table=page_table_tt,
        user_id=0,
    )
    tt_prefill_host = to_torch_auto_compose(tt_prefill)[:, 0, :seq_len, :].reshape(batch, seq_len, hf_config.hidden_size)
    prefill_pcc = _assert_pcc("optimized_prefill", ref_prefill, tt_prefill_host, threshold=PCC_THRESHOLD)

    prefill_audit_input = _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0))
    prefill_audit_rot_mats = tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len))
    with (
        patch.object(FunctionalDecoder, "prefill_forward", side_effect=AssertionError("functional prefill fallback")),
        patch.object(FunctionalDecoder, "decode_forward", side_effect=AssertionError("functional decode fallback")),
        _assert_no_host_fallback(),
    ):
        tt_prefill_audit = tt_decoder.prefill_forward(
            prefill_audit_input,
            rot_mats=prefill_audit_rot_mats,
            page_table=page_table_tt,
            user_id=0,
        )
    ttnn.synchronize_device(mesh_device)
    del tt_prefill_audit

    tt_prefill_perf_input = _tt_tensor(mesh_device, prefill_hidden.unsqueeze(0))
    tt_prefill_perf_rot_mats = tuple(rope_setup.prefill_forward(start_pos=0, seq_len=seq_len))
    if emit_perf_signposts:
        signpost(header="PERF_PREFILL")
    prefill_start = time.perf_counter()
    tt_prefill_perf = tt_decoder.prefill_forward(
        tt_prefill_perf_input,
        rot_mats=tt_prefill_perf_rot_mats,
        page_table=page_table_tt,
        user_id=0,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    if emit_perf_signposts:
        signpost(header="PERF_PREFILL_END")
    del tt_prefill_perf

    decode_hidden = torch.randn(batch, 1, hf_config.hidden_size, dtype=torch.bfloat16) * 0.05
    ref_decode = _reference_decode(reference_layer, rotary_emb, ref_cache, decode_hidden, current_pos_value)
    current_pos_host = torch.full((batch,), current_pos_value, dtype=torch.int32)
    current_pos = ttnn.from_torch(
        current_pos_host,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    decode_rot_mats = _decode_rot_mats(rope_setup, current_pos_host.to(torch.long))
    tt_decode_input = ttnn.to_memory_config(
        _tt_tensor(mesh_device, decode_hidden.unsqueeze(0)), tt_decoder.decode_residual_memcfg
    )

    tt_warm = tt_decoder.decode_forward(
        tt_decode_input,
        current_pos=current_pos,
        rot_mats=decode_rot_mats,
        page_table=page_table_tt,
    )
    with (
        patch.object(FunctionalDecoder, "prefill_forward", side_effect=AssertionError("functional prefill fallback")),
        patch.object(FunctionalDecoder, "decode_forward", side_effect=AssertionError("functional decode fallback")),
        _assert_no_host_fallback(),
    ):
        tt_audit = tt_decoder.decode_forward(
            tt_decode_input,
            current_pos=current_pos,
            rot_mats=decode_rot_mats,
            page_table=page_table_tt,
        )
    del tt_audit

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_out = tt_decoder.decode_forward(
        tt_decode_input,
        current_pos=current_pos,
        rot_mats=decode_rot_mats,
        page_table=page_table_tt,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first_replay = to_torch_auto_compose(traced_out)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size)

    replay_outputs = []
    decode_ms_samples = []
    for replay_idx in range(decode_replays):
        if emit_perf_signposts and replay_idx == 0:
            signpost(header="PERF_DECODE")
        decode_start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        decode_ms_samples.append((time.perf_counter() - decode_start) * 1000.0)
        if emit_perf_signposts and replay_idx == 0:
            signpost(header="PERF_DECODE_END")
        replay_outputs.append(to_torch_auto_compose(traced_out)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size))
    ttnn.release_trace(mesh_device, trace_id)

    eager_decode = to_torch_auto_compose(tt_warm)[:, 0, :batch, :].reshape(batch, 1, hf_config.hidden_size)
    decode_pcc = _assert_pcc("optimized_decode_trace", ref_decode, first_replay)
    determinism_pcc = _assert_pcc("optimized_decode_trace_repeated_input", first_replay, replay_outputs[-1], threshold=0.9999)
    eager_trace_pcc = _assert_pcc("optimized_decode_eager_vs_trace", eager_decode, first_replay, threshold=0.9999)

    return {
        "real_weights": real_weights,
        "policy": tt_decoder.policy.name,
        "prefill_pcc": prefill_pcc,
        "decode_trace_pcc": decode_pcc,
        "determinism_pcc": determinism_pcc,
        "eager_trace_pcc": eager_trace_pcc,
        "seq_len": seq_len,
        "decode_context": current_pos_value + 1,
        "page_block_size": PAGE_BLOCK_SIZE,
        "max_num_blocks": max_num_blocks,
        "max_seq_len": max_seq_len,
        "prefill_ms_e2e": prefill_ms,
        "decode_ms_e2e_samples": decode_ms_samples,
        "decode_ms_e2e_min": min(decode_ms_samples),
        "decode_ms_e2e_avg": sum(decode_ms_samples) / len(decode_ms_samples),
        "runtime_fallback_audit": "optimized_prefill_decode_clean",
    }


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_decoder_full_context_cache_contract(mesh_device: ttnn.MeshDevice, device_params):
    hf_config = _hf_config()
    decoder = OptimizedDecoder.from_state_dict(
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
    assert key_cache.dtype == ttnn.bfloat8_b
    assert value_cache.dtype == ttnn.bfloat8_b
    assert key_cache.shape[0] == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE
    assert value_cache.shape[0] == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE
    assert key_cache.shape[2] == PAGE_BLOCK_SIZE
    assert value_cache.shape[2] == PAGE_BLOCK_SIZE
    assert decoder.self_attn.config.max_seq_len == FULL_CACHE_SEQ_LEN
    assert decoder.self_attn.config.paged_attention_config.max_num_blocks == FULL_CACHE_SEQ_LEN // PAGE_BLOCK_SIZE


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_decoder_synthetic_paged_prefill_decode_trace(mesh_device: ttnn.MeshDevice, device_params):
    metrics = _run_optimized_prefill_decode_trace_case(mesh_device, _synthetic_state_dict(), real_weights=False)
    logger.info(f"synthetic optimized decoder metrics: {metrics}")


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_decoder_repeated_trace_stress(mesh_device: ttnn.MeshDevice, device_params):
    decode_replays = int(os.environ.get("LLAMA31_8B_OPTIMIZED_DECODER_STRESS_REPLAYS", "8"))
    metrics = _run_optimized_prefill_decode_trace_case(
        mesh_device,
        _synthetic_state_dict(),
        real_weights=False,
        emit_perf_signposts=False,
        decode_replays=decode_replays,
    )
    logger.info(f"optimized decoder repeated-trace stress metrics: {metrics}")


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_decoder_synthetic_long_context_paged_prefill_decode_trace(
    mesh_device: ttnn.MeshDevice, device_params
):
    seq_len = int(os.environ.get("LLAMA31_8B_OPTIMIZED_DECODER_LONG_SEQ_LEN", "2048"))
    max_seq_len = int(os.environ.get("LLAMA31_8B_OPTIMIZED_DECODER_LONG_MAX_SEQ_LEN", str(seq_len + 128)))
    assert seq_len % 128 == 0
    assert max_seq_len > seq_len
    max_num_blocks = max(2, (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE)
    metrics = _run_optimized_prefill_decode_trace_case(
        mesh_device,
        _synthetic_state_dict(),
        real_weights=False,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        max_num_blocks=max_num_blocks,
        emit_perf_signposts=False,
    )
    logger.info(f"synthetic long-context optimized decoder metrics: {metrics}")


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_optimized_decoder_real_weights_paged_prefill_decode_trace(mesh_device: ttnn.MeshDevice, device_params):
    metrics = _run_optimized_prefill_decode_trace_case(mesh_device, _real_state_dict(), real_weights=True)
    logger.info(f"real-weight optimized decoder metrics: {metrics}")
