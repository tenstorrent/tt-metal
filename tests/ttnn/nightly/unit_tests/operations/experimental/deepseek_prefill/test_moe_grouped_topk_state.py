# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fidelity coverage for moe_grouped_topk

In production, Tensix cores' destination registers, SFPU configuration and constant registers hold
whatever the previous operation left. uint16 indices travel through 32-bit DEST with an undefined
high half, and nothing clears DEST between programs.

These tests recreate that: many distinct calls back to back, DEST deliberately polluted with
adversarial bit patterns immediately before the gate, padding boundaries inside a tile and inside
a face, more height tiles than cores, a top-k straddling zero, and trace capture with replay.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_index_domain,
    assert_indices_exact,
    build_padding_config,
    distinct_logits,
    grouped_gate_golden_act,
)
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP

EPSILON = 1e-20
KIMI = (1, 384, 1, 1, 8, 2.827)
TOTAL_EXPERTS, K = KIMI[1], KIMI[4]
SCORE_FUNC = "sigmoid"
INPUT_DTYPE = ttnn.bfloat16

# Relative key gap below which the sort datapath cannot separate two keys (measured ~8e-4 on
# Blackhole).
SORT_RESOLUTION = 2e-3
MIN_STRICT_FRACTION = 0.5

ENGINES = pytest.mark.parametrize("stable_sort", [True, False], ids=["stable", "unstable"])


def realistic_inputs(tokens, seed):
    """The broad matrix's distribution, bf16-quantised: distinct logits plus a normal bias."""
    torch.manual_seed(seed)
    logits = distinct_logits((1, 1, tokens, TOTAL_EXPERTS), dtype=torch.bfloat16).float()
    bias = torch.randn(1, 1, tokens, TOTAL_EXPERTS).to(torch.bfloat16).float()
    return logits, bias


def golden(logits, bias, stable_sort):
    n_groups, _, summed, topk_groups, k, route_scale = KIMI
    indices, _ = grouped_gate_golden_act(
        logits, bias, route_scale, EPSILON, n_groups, summed, topk_groups, k, SCORE_FUNC, stable=stable_sort
    )
    return indices


def strict_rows(biased):
    """Rows whose top k+1 keys are pairwise separated beyond SORT_RESOLUTION, from the device's own
    biased scores so the predicate describes what the hardware actually compared."""
    s = biased.reshape(-1, TOTAL_EXPERTS).sort(dim=-1, descending=True).values[:, : K + 1]
    gaps = s[:, :-1] - s[:, 1:]
    scale = torch.maximum(s[:, :-1].abs(), s[:, 1:].abs()).clamp_min(1e-6)
    return (gaps / scale > SORT_RESOLUTION).all(dim=-1)


def assert_exact_where_strict(indices, ref, biased, context=""):
    mask = strict_rows(biased)
    frac = mask.float().mean().item()
    logger.info(f"{context}strict rows: {int(mask.sum())}/{mask.numel()} ({frac:.1%})")
    assert frac >= MIN_STRICT_FRACTION, f"{context}only {frac:.1%} of rows are strict; test would be vacuous"
    assert_indices_exact(indices.reshape(-1, K)[mask], ref.reshape(-1, K)[mask], K, context=context)


def to_device(t, dtype=INPUT_DTYPE, device=None):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def run_gate(device, dev_logits, dev_bias, stable_sort, num_real=None, tokens=None):
    """One call on already-uploaded inputs. Returns (indices, biased) trimmed to shape."""
    n_groups, total_experts, summed, topk_groups, k, route_scale = KIMI
    tokens = tokens or dev_logits.shape[-2]
    dev_biased = to_device(torch.zeros(1, 1, tokens, total_experts), ttnn.float32, device)
    padding_config = build_padding_config(device, num_real) if num_real is not None else None

    _, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
        dev_logits,
        dev_bias,
        n_groups=n_groups,
        summed_experts_per_group=summed,
        topk_groups=topk_groups,
        n_activated_experts=k,
        route_scale=route_scale,
        stable_sort=stable_sort,
        epsilon=EPSILON,
        score_func=SCORE_FUNC,
        padding_config=padding_config,
        biased_scores=dev_biased,
    )
    return (
        ttnn.to_torch(indices)[:1, :1, :tokens, :k],
        ttnn.to_torch(dev_biased)[:1, :1, :tokens, :total_experts].float(),
    )


def gate_from_host(device, logits, bias, stable_sort, num_real=None):
    return run_gate(device, to_device(logits, device=device), to_device(bias, device=device), stable_sort, num_real)


# --------------------------------------------------------------------------------------------


@ENGINES
def test_sequential_distinct_inputs(device, stable_sort):
    """Production cadence: many calls, each with new data, program cache warm, each exact."""
    calls = 24
    for seed in range(calls):
        logits, bias = realistic_inputs(32, seed)
        indices, biased = gate_from_host(device, logits, bias, stable_sort)
        assert_exact_where_strict(indices, golden(logits, bias, stable_sort), biased, context=f"[call {seed}] ")


# uint32 bit patterns whose HIGH half is the adversarial part: if any path reads the high half of a
# uint16 index word in 32-bit DEST, these are the values it would see.
POLLUTION = {
    "all_ones": 0xFFFFFFFF,
    "high_ones": 0xFFFF0000,
    "neg_zero": 0x80000000,
    "neg_one": 0xBF800000,
    "pos_inf": 0x7F800000,
    "low_ones": 0x0000FFFF,
}


def pollute_dest(device, pattern):
    """Write `pattern` through the FPU on every core so DEST holds it when the gate starts.
    Two tiles per core across the 11x10 grid; nothing runs between this and the gate."""
    grid = device.compute_with_storage_grid_size()
    tiles = 2 * grid.x * grid.y
    signed = pattern - (1 << 32) if pattern >= (1 << 31) else pattern  # torch.int32 wants two's complement
    raw = torch.full((1, 1, 32 * tiles, 32), signed, dtype=torch.int32).view(torch.float32)
    t = to_device(raw, ttnn.float32, device)
    out = ttnn.multiply(t, 1.0)
    ttnn.synchronize_device(device)
    ttnn.deallocate(out)
    ttnn.deallocate(t)


@ENGINES
@pytest.mark.parametrize("pattern", list(POLLUTION), ids=list(POLLUTION))
def test_dest_pollution_before_gate(device, stable_sort, pattern):
    """Adversarial DEST residue immediately before the gate must not reach the indices.
    Uses the production token count so 20 cores run the gate, all of them polluted."""
    tokens = PREFILL_CHUNK_TOKENS_PER_CHIP
    logits, bias = realistic_inputs(tokens, seed=7)
    dev_logits, dev_bias = to_device(logits, device=device), to_device(bias, device=device)

    pollute_dest(device, POLLUTION[pattern])
    indices, biased = run_gate(device, dev_logits, dev_bias, stable_sort)

    assert_index_domain(indices, K, TOTAL_EXPERTS)
    assert_exact_where_strict(indices, golden(logits, bias, stable_sort), biased, context=f"[{pattern}] ")


# Face boundary at row 16 and tile boundary at row 32: the writer patches sentinels with per-face
# offset arithmetic, so a boundary that lands inside a tile or inside a face is where an off-by-one
# would show. Existing padding coverage only ever uses tile-aligned boundaries.
PADDING_BOUNDARIES = [1, 15, 16, 17, 31, 32, 33, 47, 48, 63]


@pytest.mark.parametrize("num_real", PADDING_BOUNDARIES)
def test_padding_boundary_inside_tile(device, num_real):
    tokens = 64
    logits, bias = realistic_inputs(tokens, seed=11)
    indices, biased = gate_from_host(device, logits, bias, stable_sort=True, num_real=num_real)

    assert_index_domain(indices, K, TOTAL_EXPERTS, num_real=num_real, apply_padding=True)
    real = slice(0, num_real)
    assert_exact_where_strict(
        indices[:, :, real],
        golden(logits, bias, True)[:, :, real],
        biased[:, :, real],
        context=f"[num_real={num_real}] ",
    )


@ENGINES
def test_production_token_count(device, stable_sort):
    """One chip's share of one prefill chunk: 640 tokens, 20 height tiles, one per core."""
    tokens = PREFILL_CHUNK_TOKENS_PER_CHIP
    logits, bias = realistic_inputs(tokens, seed=3)
    indices, biased = gate_from_host(device, logits, bias, stable_sort)
    assert_index_domain(indices, K, TOTAL_EXPERTS)
    assert_exact_where_strict(indices, golden(logits, bias, stable_sort), biased, context="[640] ")


def test_more_tiles_than_cores(device):
    """128 height tiles on a 110-core grid forces the per-core loop to run twice on some cores,
    reusing DEST across tiles inside one kernel. Not the production shape, but a live code path."""
    grid = device.compute_with_storage_grid_size()
    tokens = 32 * (grid.x * grid.y + 18)
    logits, bias = realistic_inputs(tokens, seed=5)
    indices, biased = gate_from_host(device, logits, bias, stable_sort=True)
    assert_index_domain(indices, K, TOTAL_EXPERTS)
    assert_exact_where_strict(indices, golden(logits, bias, True), biased, context=f"[{tokens} tok] ")


def test_mixed_sign_topk(device):
    """A top-k straddling zero. The SFPU compare-exchange orders in sign-magnitude space, so a
    descending sort across the sign boundary is where a sign-handling defect would show."""
    tokens = 128
    logits, bias = realistic_inputs(tokens, seed=13)
    # Centre each row's top-k on zero: shift by that row's k/2-th largest biased score, then
    # re-quantise so the golden sees exactly the bf16 bias the device receives.
    host_biased = torch.sigmoid(logits) + bias
    shift = host_biased.topk(K, dim=-1).values[..., K // 2 - 1 : K // 2]
    bias = (bias - shift).to(torch.bfloat16).float()
    indices, biased = gate_from_host(device, logits, bias, stable_sort=True)

    top = biased.reshape(-1, TOTAL_EXPERTS).topk(K, dim=-1).values
    straddling = ((top < 0).any(dim=-1) & (top > 0).any(dim=-1)).float().mean().item()
    logger.info(f"[mixed_sign] rows whose top-{K} straddles zero: {straddling:.1%}")
    assert straddling >= 0.3, f"only {straddling:.1%} of rows straddle zero; input needs re-centring"

    assert_index_domain(indices, K, TOTAL_EXPERTS)
    assert_exact_where_strict(indices, golden(logits, bias, True), biased, context="[mixed_sign] ")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
def test_trace_replay(device):
    """The traced prefill legs capture the gate once and replay it per chunk with new inputs
    copied into persistent tensors. Replay must be exact on every input, not just the captured one."""
    tokens = PREFILL_CHUNK_TOKENS_PER_CHIP
    n_groups, total_experts, summed, topk_groups, k, route_scale = KIMI

    logits0, bias0 = realistic_inputs(tokens, seed=100)
    dev_logits, dev_bias = to_device(logits0, device=device), to_device(bias0, device=device)
    dev_biased = to_device(torch.zeros(1, 1, tokens, total_experts), ttnn.float32, device)

    def call():
        return ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            dev_logits,
            dev_bias,
            n_groups=n_groups,
            summed_experts_per_group=summed,
            topk_groups=topk_groups,
            n_activated_experts=k,
            route_scale=route_scale,
            stable_sort=True,
            epsilon=EPSILON,
            score_func=SCORE_FUNC,
            padding_config=None,
            biased_scores=dev_biased,
        )

    call()  # compile
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    _, dev_indices = call()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    try:
        for seed in range(101, 109):
            logits, bias = realistic_inputs(tokens, seed)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(logits, dtype=INPUT_DTYPE, layout=ttnn.TILE_LAYOUT), dev_logits
            )
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(bias, dtype=INPUT_DTYPE, layout=ttnn.TILE_LAYOUT), dev_bias)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)

            indices = ttnn.to_torch(dev_indices)[:1, :1, :tokens, :k]
            biased = ttnn.to_torch(dev_biased)[:1, :1, :tokens, :total_experts].float()
            assert_index_domain(indices, K, TOTAL_EXPERTS)
            assert_exact_where_strict(indices, golden(logits, bias, True), biased, context=f"[replay {seed}] ")
    finally:
        ttnn.release_trace(device, tid)
