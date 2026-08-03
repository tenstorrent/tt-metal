# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed optimized-decoder profiler and policy-sweep entry points."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from tracy import signpost

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _page_table,
    _positions,
    _real_state,
    _reference_decode,
    _reference_decode_zero_prefix,
    _reference_prefill,
    _synthetic_state,
    _to_torch_decode,
    _to_torch_prefill,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy, OptimizedDecoder
from models.common.utility_functions import comp_pcc

RECORDED_ACTIVATIONS = Path(__file__).parents[1] / "doc/optimized_decoder/activations/layer0_inputs.safetensors"


def _input_activations(*, batch: int, mode: str, config):
    if os.environ.get("PHI35_REAL_WEIGHTS") == "1":
        if not RECORDED_ACTIVATIONS.is_file():
            raise FileNotFoundError(f"recorded target activations not found: {RECORDED_ACTIVATIONS}")
        recorded = load_file(RECORDED_ACTIVATIONS)
        if mode == "prefill":
            key = "prefill_128"
            hidden = recorded[key].unsqueeze(0).repeat(batch, 1, 1)
            selection = f"batch_repeat={batch}"
        else:
            key = "token_embeddings"
            hidden = recorded[key][127 : 127 + batch].unsqueeze(1)
            selection = f"matching_next_rows={batch}"
        print(f"ACTIVATION_SOURCE recorded_target path={RECORDED_ACTIVATIONS} key={key} {selection}")
        return hidden
    sequence = 128 if mode == "prefill" else 1
    return torch.randn(
        batch,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed((11 if mode == "prefill" else 20) + batch),
    ).to(torch.bfloat16)


def _recorded_decode_prefixes(batch: int):
    recorded = load_file(RECORDED_ACTIVATIONS)["token_embeddings"]
    return torch.stack([recorded[user : user + 127] for user in range(batch)])


def _fill_recorded_reference_cache(*, past, key_cache, value_cache, page_table, mesh_device, fused_rope):
    key_values = past[0]
    if fused_rope:
        head_dim = key_values.shape[-1]
        pair_index = torch.stack((torch.arange(head_dim // 2), torch.arange(head_dim // 2, head_dim)), dim=-1).flatten()
        key_values = key_values[..., pair_index]
    tt_key = ttnn.from_torch(
        key_values,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_value = ttnn.from_torch(
        past[1],
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    batch, heads, sequence, width = tuple(tt_key.shape)
    for user in range(batch):
        key = ttnn.slice(tt_key, [user, 0, 0, 0], [user + 1, heads, sequence, width])
        value = ttnn.slice(tt_value, [user, 0, 0, 0], [user + 1, heads, sequence, width])
        if key.dtype != key_cache.dtype:
            key = ttnn.typecast(key, key_cache.dtype)
            value = ttnn.typecast(value, value_cache.dtype)
        ttnn.experimental.paged_fill_cache(key_cache, key, page_table, batch_idx=user)
        ttnn.experimental.paged_fill_cache(value_cache, value, page_table, batch_idx=user)


def _policy():
    name = os.environ.get("PHI35_OPT_POLICY", "final")
    base = OptimizationPolicy()
    precision_base = replace(
        base,
        qkv_in0_block_w=4,
        o_proj_in0_block_w=4,
        gate_up_in0_block_w=4,
        down_in0_block_w=4,
        prefill_qkv_in0_block_w=1,
        prefill_o_proj_in0_block_w=1,
        prefill_gate_up_in0_block_w=1,
        prefill_down_in0_block_w=1,
    )
    policies = {
        "final": base,
        "final_prefill_b1": replace(
            base,
            prefill_qkv_in0_block_w=1,
            prefill_o_proj_in0_block_w=1,
            prefill_gate_up_in0_block_w=1,
            prefill_down_in0_block_w=1,
        ),
        "nonfused_cache": replace(base, fused_paged_cache_update=False),
        "separate_gate_up": replace(base, separate_gate_up_projections=True),
        "sharded_packed_split": replace(base, gate_up_split_interleaved=False),
        "default_sdpa": replace(base, explicit_decode_sdpa=False),
        "manual_rope": replace(base, fused_rope=False),
        "phase_split_prefill_rope": replace(base, fused_prefill_rope=True),
        "prefill_b2_default": replace(
            base,
            prefill_qkv_in0_block_w=2,
            prefill_o_proj_in0_block_w=2,
            prefill_gate_up_in0_block_w=2,
            prefill_down_in0_block_w=2,
        ),
        "prefill_b2_inner_m": replace(
            base,
            prefill_qkv_in0_block_w=2,
            prefill_o_proj_in0_block_w=2,
            prefill_gate_up_in0_block_w=2,
            prefill_down_in0_block_w=2,
            prefill_qkv_out_block_h=8,
            prefill_o_proj_out_block_h=8,
            prefill_gate_up_out_block_h=4,
            prefill_down_out_block_h=4,
        ),
        "prefill_inner_mn": replace(
            base,
            prefill_qkv_in0_block_w=12,
            prefill_o_proj_in0_block_w=12,
            prefill_gate_up_in0_block_w=8,
            prefill_down_in0_block_w=32,
            prefill_qkv_out_block_h=4,
            prefill_o_proj_out_block_h=4,
            prefill_gate_up_out_block_h=4,
            prefill_down_out_block_h=4,
            prefill_qkv_out_block_w=12,
            prefill_o_proj_out_block_w=12,
            prefill_gate_up_out_block_w=16,
            prefill_down_out_block_w=12,
        ),
        "prefill_grid_8x10": replace(
            base,
            prefill_core_grid=(8, 10),
            prefill_qkv_in0_block_w=8,
            prefill_o_proj_in0_block_w=12,
            prefill_gate_up_in0_block_w=8,
            prefill_down_in0_block_w=32,
            prefill_qkv_out_block_h=1,
            prefill_o_proj_out_block_h=1,
            prefill_gate_up_out_block_h=1,
            prefill_down_out_block_h=1,
            prefill_qkv_out_block_w=12,
            prefill_o_proj_out_block_w=12,
            prefill_gate_up_out_block_w=16,
            prefill_down_out_block_w=12,
        ),
        "bfp8_hifi2_kv16": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
        "bfp8_hifi2_kv16_manual_rope": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
            fused_rope=False,
        ),
        "bfp8_hifi2_kv16_nonfused_cache": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
            fused_paged_cache_update=False,
        ),
        "bfp8_hifi2_kv16_default_sdpa": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
            explicit_decode_sdpa=False,
        ),
        "bfp8_hifi2_kv16_functional_attention": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
            fused_rope=False,
            fused_paged_cache_update=False,
            explicit_decode_sdpa=False,
        ),
        "bfp8_hifi2_kv16_default_sdpa_manual_rope": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
            fused_rope=False,
            explicit_decode_sdpa=False,
        ),
        "bfp8_hifi2_kv16_default_sdpa_nonfused_cache": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
            fused_paged_cache_update=False,
            explicit_decode_sdpa=False,
        ),
        "bfp8_hifi2_kv16_manual_rope_nonfused_cache": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
            fused_rope=False,
            fused_paged_cache_update=False,
        ),
        "bf16_hifi4_kv16": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat16,
            gate_up_weight_dtype=ttnn.bfloat16,
            down_weight_dtype=ttnn.bfloat16,
            kv_cache_dtype=ttnn.bfloat16,
            attention_math_fidelity=ttnn.MathFidelity.HiFi4,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi4,
        ),
        "bfp8_lofi_kv16": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
        ),
        "attn4_lofi_kv16": replace(
            precision_base,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
        ),
        "gate4_lofi_kv16": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
        ),
        "down4_lofi_kv16": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            kv_cache_dtype=ttnn.bfloat16,
        ),
        "all4_lofi_kv16": replace(precision_base, kv_cache_dtype=ttnn.bfloat16),
        "bfp8_hifi2_kv8": replace(
            precision_base,
            attention_weight_dtype=ttnn.bfloat8_b,
            gate_up_weight_dtype=ttnn.bfloat8_b,
            down_weight_dtype=ttnn.bfloat8_b,
            attention_math_fidelity=ttnn.MathFidelity.HiFi2,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi2,
        ),
        "core8_base": replace(
            base,
            qkv_in0_block_w=4,
            o_proj_in0_block_w=4,
            gate_up_in0_block_w=4,
            down_in0_block_w=4,
        ),
        "qkv6": replace(base, qkv_in0_block_w=6),
        "o6": replace(base, o_proj_in0_block_w=6),
        "gate4": replace(base, gate_up_in0_block_w=4),
        "down8": replace(base, down_in0_block_w=8),
        "down32": replace(base, down_in0_block_w=32),
        "core16": replace(
            base,
            decode_core_grid=(8, 2),
            qkv_in0_block_w=6,
            o_proj_in0_block_w=6,
            gate_up_in0_block_w=6,
            down_in0_block_w=16,
        ),
        "core32": replace(
            base,
            decode_core_grid=(8, 4),
            qkv_in0_block_w=3,
            o_proj_in0_block_w=3,
            gate_up_in0_block_w=3,
            down_in0_block_w=8,
        ),
        "core8": replace(
            base,
            qkv_in0_block_w=12,
            o_proj_in0_block_w=12,
            gate_up_in0_block_w=6,
            down_in0_block_w=16,
        ),
    }
    if name not in policies:
        raise ValueError(f"unknown PHI35_OPT_POLICY={name!r}; choose one of {tuple(policies)}")
    print(f"OPT_POLICY name={name} values={policies[name]}")
    return name, policies[name]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_warmed_prefill(mesh_device, batch):
    name, policy = _policy()
    config = _config()
    state = _real_state() if os.environ.get("PHI35_REAL_WEIGHTS") == "1" else _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        policy=policy,
    )
    hidden = _input_activations(batch=batch, mode="prefill", config=config)
    tt_hidden = _to_tt_prefill(hidden, mesh_device)
    page_table = _page_table(
        batch,
        128,
        mesh_device,
        permute=os.environ.get("PHI35_IDENTITY_PAGE_TABLE") != "1",
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    signpost(f"OPT_PREFILL_B{batch}")
    start = time.perf_counter()
    output = decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    signpost(f"OPT_PREFILL_B{batch}_END")
    reference, _ = _reference_prefill(config, state, hidden)
    _assert_pcc(f"optimized-perf-prefill-{name}-b{batch}", reference, _to_torch_prefill(output))
    print(f"PERF_RESULT policy={name} mode=prefill batch={batch} sequence=128 warmed_ms={elapsed_ms:.6f}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_traced_decode(mesh_device, batch):
    name, policy = _policy()
    config = _config()
    state = _real_state() if os.environ.get("PHI35_REAL_WEIGHTS") == "1" else _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        policy=policy,
    )
    hidden = _input_activations(batch=batch, mode="decode", config=config)
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(
        batch,
        128,
        mesh_device,
        permute=os.environ.get("PHI35_IDENTITY_PAGE_TABLE") != "1",
    )
    positions = [127] * batch
    current_positions = _positions(positions, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    recorded_past = None
    if os.environ.get("PHI35_REAL_WEIGHTS") == "1":
        prefixes = _recorded_decode_prefixes(batch)
        _, recorded_past = _reference_prefill(config, state, prefixes)
        if os.environ.get("PHI35_CACHE_SOURCE") == "optimized_prefill":
            decoder.prefill_forward(
                _to_tt_prefill(prefixes, mesh_device),
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
            )
        else:
            _fill_recorded_reference_cache(
                past=recorded_past,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
                mesh_device=mesh_device,
                fused_rope=policy.fused_rope,
            )
        ttnn.synchronize_device(mesh_device)
        print(
            "CACHE_SOURCE "
            f"{os.environ.get('PHI35_CACHE_SOURCE', 'reference_fill')} "
            f"recorded_target_prefix prefix_length=127 batch={batch}"
        )

    def decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=False,
        )

    decode()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples = []
        signpost(f"OPT_DECODE_B{batch}")
        for _ in range(10):
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append(1000 * (time.perf_counter() - start))
        signpost(f"OPT_DECODE_B{batch}_END")
        actual = _to_torch_decode(output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    reference = (
        _reference_decode(config, state, hidden, 127, recorded_past)
        if recorded_past is not None
        else _reference_decode_zero_prefix(config, state, hidden, positions, use_long=False)
    )
    if os.environ.get("PHI35_DIAGNOSTIC_PCC") == "1":
        for user in range(batch):
            _, user_pcc = comp_pcc(reference[user].float(), actual[user].float(), 0.995)
            print(
                f"PCC_DIAGNOSTIC policy={name} user={user} {user_pcc} "
                f"reference_norm={reference[user].float().norm().item():.8f} "
                f"actual_norm={actual[user].float().norm().item():.8f}"
            )
    else:
        _assert_pcc(f"optimized-perf-decode-{name}-b{batch}", reference, actual)
    print(
        f"PERF_RESULT policy={name} mode=decode batch={batch} context=128 trace_replays=10 "
        f"mean_ms={sum(samples) / len(samples):.6f} min_ms={min(samples):.6f}"
    )
