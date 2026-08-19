# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import os
import textwrap
import time

import pytest
import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import (
    BLOCK_SIZE,
    _assert_pcc,
    _causal_mask,
    _load_real_layer_state,
    _page_table,
    _randn,
    _rotary,
    _signpost,
    _state_for_perf,
    _synthetic_layer_state,
    _target_text_config,
    _to_torch,
    _torch_layer,
    _tt_bf16,
    _tt_int,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt import fused_decoder as fused_decoder_module
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder


def _prepare_fused_decode_after_prefill(device, cfg, tt_layer: FusedDecoder, layer_idx: int, seq_len: int):
    batch = 1
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=1800 + layer_idx + seq_len, scale=0.01)
    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    current_pos = _tt_int(torch.tensor([seq_len], dtype=torch.int32), device)

    if cfg.layer_types[layer_idx] == "full_attention":
        max_seq_len = max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=1600 + seq_len, scale=0.01)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)
        kv_cache = FusedDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
        )
        decode_position_embeddings = _rotary(cfg, decode_hidden, torch.tensor([[seq_len]], dtype=torch.long))
        kwargs = {
            "current_pos": current_pos,
            "position_embeddings": (
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            "page_table": _tt_int(page_table, device),
            "kv_cache": kv_cache,
        }
    else:
        state = FusedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=batch)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=1700 + seq_len, scale=0.01)
        tt_prefill = tt_layer.prefill_forward(_tt_bf16(hidden.unsqueeze(0), device), linear_state=state)
        kwargs = {"current_pos": current_pos, "linear_state": tt_prefill.linear_state}
    return decode_input, kwargs


def _run_fused_traced_decode(device, tt_layer: FusedDecoder, decode_input: ttnn.Tensor, decode_kwargs: dict):
    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(device, trace_id)
    ttnn.synchronize_device(device)
    return traced


def _run_fused_prefill_decode_parity(
    device,
    *,
    cfg,
    layer_idx: int,
    state: dict[str, torch.Tensor],
    seq_len: int,
    batch: int = 1,
    trace_decode: bool = False,
):
    max_seq_len = max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
    hidden = _randn((batch, seq_len, cfg.hidden_size), seed=1200 + batch + layer_idx + seq_len, scale=0.01)
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=1300 + batch + layer_idx + seq_len, scale=0.01)

    layer = _torch_layer(cfg, layer_idx, state)
    tt_layer = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    assert type(tt_layer).__name__ == "FusedDecoder"

    if cfg.layer_types[layer_idx] == "full_attention":
        cache = DynamicCache()
        pos = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, pos)
        with torch.no_grad():
            ref_prefill = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=_causal_mask(batch, seq_len),
                past_key_values=cache,
            )
        page_table = _page_table(batch, max_seq_len)
        kv_cache = FusedDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        tt_prefill = tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
        ).hidden_states
        prefill_msg = _assert_pcc("fused full_attention prefill", ref_prefill, _to_torch(tt_prefill).squeeze(0))

        current_pos = torch.full((batch,), seq_len, dtype=torch.int32)
        decode_pos = current_pos.to(torch.long).reshape(batch, 1)
        decode_position_embeddings = _rotary(cfg, decode_hidden, decode_pos)
        with torch.no_grad():
            ref_decode = layer(
                decode_hidden,
                position_embeddings=decode_position_embeddings,
                attention_mask=None,
                past_key_values=cache,
            )
        decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
        decode_kwargs = {
            "position_embeddings": (
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            "page_table": _tt_int(page_table, device),
            "kv_cache": kv_cache,
            "current_pos": _tt_int(current_pos, device),
        }
        tt_decode = (
            _run_fused_traced_decode(device, tt_layer, decode_input, decode_kwargs)
            if trace_decode
            else tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
        )
        decode_msg = _assert_pcc(
            "fused full_attention decode",
            ref_decode,
            _to_torch(tt_decode).squeeze(0).transpose(0, 1),
        )
        return prefill_msg, decode_msg

    linear_cache = None
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import _LinearCache

    linear_cache = _LinearCache(layer_idx)
    dummy_pos = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
    dummy_position_embeddings = _rotary(cfg, hidden, dummy_pos)
    with torch.no_grad():
        ref_prefill = layer(
            hidden,
            position_embeddings=dummy_position_embeddings,
            attention_mask=None,
            past_key_values=linear_cache,
        )
    linear_state = FusedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=batch)
    tt_prefill_result = tt_layer.prefill_forward(_tt_bf16(hidden.unsqueeze(0), device), linear_state=linear_state)
    prefill_msg = _assert_pcc(
        "fused linear_attention prefill", ref_prefill, _to_torch(tt_prefill_result.hidden_states).squeeze(0)
    )

    decode_position_embeddings = _rotary(cfg, decode_hidden, torch.full((batch, 1), seq_len, dtype=torch.long))
    with torch.no_grad():
        ref_decode = layer(
            decode_hidden,
            position_embeddings=decode_position_embeddings,
            attention_mask=None,
            past_key_values=linear_cache,
        )
    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    decode_kwargs = {
        "current_pos": _tt_int(torch.full((batch,), seq_len, dtype=torch.int32), device),
        "linear_state": tt_prefill_result.linear_state,
    }
    tt_decode = (
        _run_fused_traced_decode(device, tt_layer, decode_input, decode_kwargs)
        if trace_decode
        else tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    )
    decode_msg = _assert_pcc(
        "fused linear_attention decode",
        ref_decode,
        _to_torch(tt_decode).squeeze(0).transpose(0, 1),
    )
    return prefill_msg, decode_msg


def test_fused_decoder_graph_summary():
    summary = FusedDecoder.graph_summary
    assert summary.packed_attention_qkgv
    assert summary.packed_linear_attention_inputs
    assert summary.packed_shared_expert_gate_up
    assert summary.packed_routed_expert_gate_up
    assert summary.binary_activation_folding


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_seq5", "full_seq33"])
def test_synthetic_fused_decoder_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_fused_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"fused prefill {prefill_msg}")
    print(f"fused traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_batch2", "full_batch2"])
def test_synthetic_fused_decoder_batch2_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_fused_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        batch=2,
        trace_decode=True,
    )
    print(f"fused batch2 prefill {prefill_msg}")
    print(f"fused batch2 traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"), [(0, 65), (3, 33)], ids=["linear_non_aligned65", "full_non_aligned33"]
)
def test_synthetic_fused_decoder_non_aligned_lengths(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_fused_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"fused non-aligned prefill {prefill_msg}")
    print(f"fused non-aligned traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_repeat", "full_repeat"])
def test_synthetic_fused_decoder_repeated_input_determinism(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    tt_layer = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    decode_input, decode_kwargs = _prepare_fused_decode_after_prefill(device, cfg, tt_layer, layer_idx, seq_len)

    first = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    second = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    msg = _assert_pcc(
        f"fused {cfg.layer_types[layer_idx]} repeated decode", _to_torch(first), _to_torch(second), pcc=0.9999
    )
    print(f"fused repeated decode {msg}")


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_REAL_WEIGHTS") != "1", reason="set RUN_QWEN36_REAL_WEIGHTS=1 to load checkpoint weights"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(0, 1), (3, 1), (0, 5), (3, 5)],
    ids=["real_linear_layer0_seq1", "real_full_layer3_seq1", "real_linear_layer0_seq5", "real_full_layer3_seq5"],
)
def test_real_weight_fused_decoder_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _load_real_layer_state(layer_idx)
    prefill_msg, decode_msg = _run_fused_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"fused real prefill {prefill_msg}")
    print(f"fused real traced decode {decode_msg}")


def test_fused_runtime_fallback_audit_source():
    runtime_functions = {
        "_silu_mul_fused": fused_decoder_module._silu_mul_fused,
        "_partial_rope_fallback": fused_decoder_module._partial_rope_fallback,
        "_apply_partial_rope_fused": fused_decoder_module._apply_partial_rope_fused,
        "_FusedFullAttention._project_qkgv": fused_decoder_module._FusedFullAttention._project_qkgv,
        "_FusedFullAttention._reshape_prefill_heads": fused_decoder_module._FusedFullAttention._reshape_prefill_heads,
        "_FusedFullAttention._reshape_decode_heads": fused_decoder_module._FusedFullAttention._reshape_decode_heads,
        "_FusedFullAttention._norm_and_rope": fused_decoder_module._FusedFullAttention._norm_and_rope,
        "_FusedFullAttention._cache_update_tensor": fused_decoder_module._FusedFullAttention._cache_update_tensor,
        "_FusedFullAttention.prefill_forward": fused_decoder_module._FusedFullAttention.prefill_forward,
        "_FusedFullAttention.decode_forward": fused_decoder_module._FusedFullAttention.decode_forward,
        "_FusedLinearAttention._project_inputs": fused_decoder_module._FusedLinearAttention._project_inputs,
        "_FusedLinearAttention._log_g": fused_decoder_module._FusedLinearAttention._log_g,
        "_FusedLinearAttention._step": fused_decoder_module._FusedLinearAttention._step,
        "_FusedLinearAttention.prefill_forward": fused_decoder_module._FusedLinearAttention.prefill_forward,
        "_FusedQwenMoe._router_dense": fused_decoder_module._FusedQwenMoe._router_dense,
        "_FusedQwenMoe._shared": fused_decoder_module._FusedQwenMoe._shared,
        "_FusedQwenMoe._routed_decode": fused_decoder_module._FusedQwenMoe._routed_decode,
        "_FusedQwenMoe._routed_prefill_chunk": fused_decoder_module._FusedQwenMoe._routed_prefill_chunk,
        "_FusedQwenMoe.forward": fused_decoder_module._FusedQwenMoe.forward,
        "FusedDecoder.prefill_forward": fused_decoder_module.FusedDecoder.prefill_forward,
        "FusedDecoder.decode_forward": fused_decoder_module.FusedDecoder.decode_forward,
    }
    forbidden_names = {"torch"}
    forbidden_attrs = {"from_torch", "to_torch", "get_fallback_function"}
    violations = []
    for name, func in runtime_functions.items():
        source = textwrap.dedent(inspect.getsource(func))
        if "FunctionalDecoder.from_state_dict" in source or "FunctionalDecoder.prefill_forward" in source:
            violations.append(f"{name}: functional decoder fallback call")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{name}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                violations.append(f"{name}: attribute {node.attr}")
    assert not violations


def _run_fused_signposted_prefill(device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)

    if cfg.layer_types[layer_idx] == "full_attention":
        batch = 1
        max_seq_len = max(96, ((seq_len + 1 + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
        hidden = _randn((batch, seq_len, cfg.hidden_size), seed=2600 + seq_len, scale=0.01)
        position_ids = torch.arange(seq_len, dtype=torch.long).reshape(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, position_ids)
        page_table = _page_table(batch, max_seq_len)
        warm_cache = FusedDecoder.allocate_full_attention_cache(
            hf_config=cfg, mesh_device=device, max_batch_size=batch, max_seq_len=max_seq_len, block_size=BLOCK_SIZE
        )
        tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=warm_cache,
        )
        measure_cache = FusedDecoder.allocate_full_attention_cache(
            hf_config=cfg, mesh_device=device, max_batch_size=batch, max_seq_len=max_seq_len, block_size=BLOCK_SIZE
        )
        ttnn.synchronize_device(device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=measure_cache,
        ).hidden_states
    else:
        hidden = _randn((1, seq_len, cfg.hidden_size), seed=2700 + seq_len, scale=0.01)
        hidden_tt = _tt_bf16(hidden.unsqueeze(0), device)
        warm_state = FusedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
        tt_layer.prefill_forward(hidden_tt, linear_state=warm_state)
        measure_state = FusedDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
        ttnn.synchronize_device(device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(hidden_tt, linear_state=measure_state).hidden_states

    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} wall_ms={elapsed_ms:.3f} output_shape={tuple(out.shape)}")


def _run_fused_signposted_traced_decode(device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = FusedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    decode_input, decode_kwargs = _prepare_fused_decode_after_prefill(device, cfg, tt_layer, layer_idx, seq_len)

    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)

    _signpost(signpost_name)
    start = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} traced_wall_ms={elapsed_ms:.3f} output_shape={tuple(traced.shape)}")
    ttnn.release_trace(device, trace_id)


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FUSED_PERF") != "1", reason="set RUN_QWEN36_FUSED_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_fused_linear_prefill(device):
    _run_fused_signposted_prefill(device, layer_idx=0, seq_len=5, signpost_name="FUSED_LINEAR_PREFILL")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FUSED_PERF") != "1", reason="set RUN_QWEN36_FUSED_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_fused_full_prefill(device):
    _run_fused_signposted_prefill(device, layer_idx=3, seq_len=33, signpost_name="FUSED_FULL_PREFILL")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FUSED_PERF") != "1", reason="set RUN_QWEN36_FUSED_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_fused_linear_decode(device):
    _run_fused_signposted_traced_decode(device, layer_idx=0, seq_len=5, signpost_name="FUSED_LINEAR_DECODE")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FUSED_PERF") != "1", reason="set RUN_QWEN36_FUSED_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_fused_full_decode(device):
    _run_fused_signposted_traced_decode(device, layer_idx=3, seq_len=33, signpost_name="FUSED_FULL_DECODE")
