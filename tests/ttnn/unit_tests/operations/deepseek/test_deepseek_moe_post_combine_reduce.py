# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for post_combine_reduce fused kernel.

Validates correctness against:
- PyTorch reference (weighted sum across experts)
- Old implementation from tt_moe.py (to_layout + mul + sum)

Tests structured data, random data, sparse weights, and non-local expert skipping.
Shape: [1, 3200, 8, 7168] - DeepSeek-V3 dimensions.
"""

import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import run_for_blackhole

from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config

NUM_TOKENS = 3200
NUM_EXPERTS = 8
EMB_DIM = DeepSeekV3Config.EMB_SIZE
EXPERT_DIM = 2
PCC_THRESHOLD = 0.999
NUM_ROUTED_EXPERTS = DeepSeekV3Config.NUM_ROUTED_EXPERTS


def pytorch_reference(combine, weights):
    """PyTorch reference: weighted sum across experts."""
    return (combine * weights.expand(-1, -1, -1, combine.shape[-1])).sum(dim=EXPERT_DIM)


def old_implementation(combine_tt, weights_tt):
    """Old implementation as used in tt_moe.py: to_layout(TILE) + mul + sum."""
    combine_tiled = ttnn.to_layout(combine_tt, ttnn.TILE_LAYOUT)
    weights_tiled = ttnn.to_layout(weights_tt, ttnn.TILE_LAYOUT)
    weighted = ttnn.mul(combine_tiled, weights_tiled)
    return ttnn.sum(weighted, dim=EXPERT_DIM)


def make_dispatch_table_all_local(device):
    """Create dispatch table where all experts are local (single device test)."""
    # All experts map to chip 0 (local)
    table = torch.zeros(NUM_ROUTED_EXPERTS, dtype=torch.int32)
    return ttnn.from_torch(table, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_indices(num_tokens, num_experts, device):
    """Create indices tensor with random global expert IDs."""
    # Each token routes to num_experts random experts out of NUM_ROUTED_EXPERTS
    indices = torch.stack([torch.randperm(NUM_ROUTED_EXPERTS)[:num_experts] for _ in range(num_tokens)])
    indices = indices.unsqueeze(0).to(torch.int32)  # [1, num_tokens, num_experts]
    return indices, ttnn.from_torch(
        indices, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def new_implementation(combine_tt, weights_tt, indices_tt, dispatch_table_tt):
    """Fused kernel: reads ROW_MAJOR, produces TILE output, skips non-local experts."""
    return ttnn.experimental.deepseek_prefill.post_combine_reduce(
        combine_tt,
        weights_tt,
        indices_tt,
        dispatch_table_tt,
        expert_dim=EXPERT_DIM,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def compute_pcc(a, b):
    """Compute PCC between two tensors."""
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def to_device(tensor, device):
    return ttnn.from_torch(tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def assert_pcc(result, expected, threshold=PCC_THRESHOLD, label=""):
    nan_count = torch.isnan(result).sum().item()
    assert nan_count == 0, f"{label}: got {nan_count} NaN elements"
    pcc = compute_pcc(result, expected)
    logger.info(f"  {label}: PCC={pcc:.6f}")
    assert pcc > threshold, f"{label}: PCC {pcc:.6f} below {threshold}"
    return pcc


# ============================================================================
# Structured data test
# ============================================================================


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
def test_structured_data(device):
    """Constant-per-tile activations with sequential weights [1..8].
    This pattern is easy to verify manually and catches tile ordering bugs."""
    torch.manual_seed(42)
    tile_width = 1024
    num_tiles = EMB_DIM // tile_width

    tile_values = 0.1 * torch.arange(
        1,
        NUM_TOKENS * NUM_EXPERTS * num_tiles + 1,
        dtype=torch.float32,
    )
    combine = (
        tile_values.view(1, NUM_TOKENS, NUM_EXPERTS, num_tiles, 1)
        .expand(1, NUM_TOKENS, NUM_EXPERTS, num_tiles, tile_width)
        .reshape(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM)
        .to(torch.bfloat16)
    )

    weights = (
        torch.arange(1, NUM_EXPERTS + 1, dtype=torch.float32)
        .view(1, 1, NUM_EXPERTS, 1)
        .expand(1, NUM_TOKENS, NUM_EXPERTS, 1)
        .to(torch.bfloat16)
    )

    dispatch_table_tt = make_dispatch_table_all_local(device)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    assert_pcc(result, ref, threshold=0.998, label="structured")


# ============================================================================
# Random data tests
# ============================================================================


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
def test_random_data(device):
    """Random activations and weights, compared to PyTorch reference."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    assert_pcc(result, ref, label="random")


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
def test_vs_old_implementation(device):
    """Fused kernel vs old implementation (to_layout + mul + sum) with random data."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    ref = pytorch_reference(combine, weights)
    combine_tt = to_device(combine, device)
    weights_tt = to_device(weights, device)

    old_result = ttnn.to_torch(old_implementation(combine_tt, weights_tt))
    new_result = ttnn.to_torch(new_implementation(combine_tt, weights_tt, indices_tt, dispatch_table_tt))

    assert_pcc(old_result, ref, label="old_vs_ref")
    assert_pcc(new_result, ref, label="new_vs_ref")
    assert_pcc(old_result, new_result, label="old_vs_new")


# ============================================================================
# Sparse weight tests
# ============================================================================


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
@pytest.mark.parametrize("k_active", [6, 4, 2, 1])
def test_sparse_weights(device, k_active):
    """Fused kernel with sparse weights (k_active out of 8 experts non-zero per token)."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.zeros(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)
    for t in range(NUM_TOKENS):
        active = torch.randperm(NUM_EXPERTS)[:k_active]
        weights[0, t, active, 0] = torch.randn(k_active, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    ref = pytorch_reference(combine, weights)
    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    assert_pcc(result, ref, label=f"sparse_{k_active}/{NUM_EXPERTS}")


# ============================================================================
# Non-local expert skip test
# ============================================================================


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
def test_skip_nonlocal_experts(device):
    """Verify that marking experts as non-local (-1 in dispatch table) produces
    the same result when those experts' combine_output is zero (as in real MoE)."""
    torch.manual_seed(42)

    # Create indices: each token routes to 8 random experts
    indices_torch, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    # Build dispatch table where only experts 0-63 are local (column 0 of TP4)
    local_expert_end = 64
    table = torch.full((NUM_ROUTED_EXPERTS,), -1, dtype=torch.int32)
    for i in range(local_expert_end):
        table[i] = i // 8  # map to chip within dispatch group
    dispatch_table_tt = ttnn.from_torch(
        table, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    # Reference: only sum local experts (non-local should be skipped by kernel)
    ref_combine = combine.clone()
    for t in range(NUM_TOKENS):
        for k in range(NUM_EXPERTS):
            expert_id = indices_torch[0, t, k].item()
            if expert_id >= local_expert_end:
                ref_combine[0, t, k, :] = 0.0
    ref = pytorch_reference(ref_combine, weights)

    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )
    # Use the standard PCC threshold so this test validates non-local expert skipping.
    assert_pcc(result, ref, threshold=PCC_THRESHOLD, label="skip_nonlocal_no_init_zeros")


# ============================================================================
# Output format test
# ============================================================================


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
def test_output_layout(device):
    """Verify output is TILE layout with correct shape."""
    torch.manual_seed(42)
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    dispatch_table_tt = make_dispatch_table_all_local(device)
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    result_tt = new_implementation(
        to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt
    )
    assert result_tt.layout == ttnn.TILE_LAYOUT, f"Expected TILE_LAYOUT, got {result_tt.layout}"
    assert list(result_tt.shape) == [1, NUM_TOKENS, EMB_DIM], f"Wrong shape: {result_tt.shape}"


# ============================================================================
# Garbage-in-nonlocal-slots tests (repro for issue #42999)
# ============================================================================
#
# In real MoE prefill, when combine runs with init_zeros=False, DRAM slots for
# non-local experts are left uninitialized. Those bytes can decode to NaN/Inf
# in bfloat16. The post_combine_reduce kernel is supposed to skip non-local
# experts, but when a token has *all* non-local experts, the writer forces the
# last expert's weight to 0 and compute takes the `must_zero_init` path — it
# multiplies the uninitialized combine_input tile by 0. In IEEE-754,
# NaN*0 = NaN and Inf*0 = NaN, so garbage leaks through and poisons the sum.


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
@pytest.mark.parametrize("garbage", ["nan", "inf"], ids=["nan", "inf"])
def test_all_nonlocal_garbage_in_combine(device, garbage):
    """All experts non-local + garbage (NaN/Inf) in combine_input.

    Every token hits the `must_zero_init` path. Expected output: all zeros.
    If the kernel does mul(garbage, 0) the IEEE-754 result is NaN and the
    output is polluted — reproducing the test_prefill_block PCC collapse.
    """
    torch.manual_seed(42)
    fill = float("nan") if garbage == "nan" else float("inf")
    combine = torch.full((1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM), fill, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    # Dispatch table: all -1 → every expert is non-local for this chip.
    table = torch.full((NUM_ROUTED_EXPERTS,), -1, dtype=torch.int32)
    dispatch_table_tt = ttnn.from_torch(
        table, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )

    nan_count = torch.isnan(result).sum().item()
    inf_count = torch.isinf(result).sum().item()
    max_abs = result.abs().max().item() if result.numel() > 0 else 0.0
    logger.info(f"  all_nonlocal_{garbage}: NaN={nan_count} Inf={inf_count} max|x|={max_abs}")
    assert nan_count == 0, f"got {nan_count} NaN in output (NaN*0=NaN leaked through must_zero_init)"
    assert inf_count == 0, f"got {inf_count} Inf in output"
    assert max_abs == 0.0, f"expected all-zero output, got max|x|={max_abs}"


@run_for_blackhole(reason_str="DeepSeek-V3 dimensions require 100 cores (3200 tokens / 32 per core); WH has only 72")
@pytest.mark.parametrize("garbage", ["nan", "inf"], ids=["nan", "inf"])
def test_mixed_nonlocal_garbage_in_combine(device, garbage):
    """Partial non-local (TP4-like) + garbage only in non-local slots.

    Mimics real combine behaviour: local slots hold valid data, non-local
    slots hold uninitialised DRAM (simulated as NaN/Inf). With topk=8 and
    75% non-local, ~10% of tokens fall through the `must_zero_init` path
    and are the ones that can poison the output.
    """
    torch.manual_seed(42)
    fill = float("nan") if garbage == "nan" else float("inf")
    local_expert_end = 64  # TP4: 64/256 experts local per chip
    combine = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, EMB_DIM, dtype=torch.bfloat16)
    weights = torch.randn(1, NUM_TOKENS, NUM_EXPERTS, 1, dtype=torch.bfloat16)

    table = torch.full((NUM_ROUTED_EXPERTS,), -1, dtype=torch.int32)
    for i in range(local_expert_end):
        table[i] = i // 8
    dispatch_table_tt = ttnn.from_torch(
        table, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    indices_torch, indices_tt = make_indices(NUM_TOKENS, NUM_EXPERTS, device)

    # Poison non-local slots in combine + reference: only local experts contribute.
    ref_combine = combine.clone()
    all_nonlocal_tokens = 0
    for t in range(NUM_TOKENS):
        any_local = False
        for k in range(NUM_EXPERTS):
            expert_id = indices_torch[0, t, k].item()
            if expert_id >= local_expert_end:
                combine[0, t, k, :] = fill
                ref_combine[0, t, k, :] = 0.0
            else:
                any_local = True
        if not any_local:
            all_nonlocal_tokens += 1
    ref = pytorch_reference(ref_combine, weights)
    logger.info(f"  mixed_{garbage}: {all_nonlocal_tokens}/{NUM_TOKENS} tokens hit must_zero_init")

    result = ttnn.to_torch(
        new_implementation(to_device(combine, device), to_device(weights, device), indices_tt, dispatch_table_tt)
    )

    nan_count = torch.isnan(result).sum().item()
    inf_count = torch.isinf(result).sum().item()
    logger.info(f"  mixed_{garbage}: NaN={nan_count} Inf={inf_count} in output")
    assert nan_count == 0, f"got {nan_count} NaN in output"
    assert inf_count == 0, f"got {inf_count} Inf in output"
    assert_pcc(result, ref, label=f"mixed_nonlocal_{garbage}")
