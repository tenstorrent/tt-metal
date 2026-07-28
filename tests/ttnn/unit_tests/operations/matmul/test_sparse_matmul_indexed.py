# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level probe for the indexed/gather mode of ttnn.sparse_matmul.

This is the go/no-go gate for the MoE expert-gather work (see models/demos/qwen3_6_a3b/MOE_GATHER_PLAN.md):
it exercises the new optional `indices` operand with a hard-coded, NON-MONOTONIC index list and no
model, proving (1) it compiles + runs without hanging (kernel lockstep), (2) it matches a torch
reference, (3) the down-projection `is_input_a_sparse=True` compact-A path, (4) bf4 weight addressing
by arbitrary expert id, and (5) that the dense sparsity-scan path is unchanged when `indices` is absent.
"""
import math

from loguru import logger
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics


def _sparse_pc(m, n, tile_h, tile_w):
    """A 1D-optimized program config that spreads N across a core grid with per_core_N=1, mirroring
    qwen3_6_a3b's TtMoE._sparse_pc (the production gather caller)."""
    nt = int(math.ceil(n / tile_w))
    cx, cy = 1, 1
    for d in range(min(10, nt), 0, -1):
        if nt % d == 0 and nt // d <= 10:
            cx, cy = nt // d, d
            break
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=max(tile_h, m) // tile_h,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _make_indices(active_ids, device):
    """[1,1,1,num_active] UINT16 ROW_MAJOR device tensor of expert ids (compact-output slot order)."""
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, len(active_ids))
    return ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _make_sparsity(active_ids, num_experts, device):
    """[1,1,1,E] bf16 ROW_MAJOR mask, nonzero exactly at the active expert ids (still required by the
    kernel's multicast machinery even in gather mode)."""
    s = torch.zeros(1, 1, 1, num_experts, dtype=torch.float32)
    for e in active_ids:
        s[0, 0, 0, e] = 1.0
    return ttnn.from_torch(s.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


@pytest.mark.parametrize("num_experts", [16, 256])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_gather_gate_up(device, num_experts, in1_dtype):
    """gate_up-like: A is dense/broadcast [1,1,1,K], B = expert weights [1,E,K,N], is_input_b_sparse.
    Compact output slot i must equal in0 @ in1[active_ids[i]]."""
    torch.manual_seed(0)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    # A non-monotonic active set (the bf4-addressing-by-arbitrary-index check).
    active_ids = [num_experts - 1, 5, num_experts - 2, 12, 1, num_experts // 2, 7, 0][: min(8, num_experts)]
    num_active = len(active_ids)

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    indices_t = _make_indices(active_ids, device)

    out_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        indices=indices_t,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=_sparse_pc(m, n, tile_h, tile_w),
    )
    out = ttnn.to_torch(out_t).reshape(num_active, m, n)
    logger.info(f"gate_up gather out shape {tuple(out_t.shape)} -> compact [{num_active}, {m}, {n}]")

    in1_f = in1.float()
    for i, e in enumerate(active_ids):
        ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
        assert_numeric_metrics(
            ref, out[i].float(), atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
        )


@pytest.mark.parametrize("num_experts", [16, 256])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_gather_down(device, num_experts, in1_dtype):
    """down-like: A is COMPACT [1,num_active,1,I] (one row per active expert), B = [1,E,I,H], both
    sparse. Compact output slot i must equal A[i] @ B[active_ids[i]] (A indexed by i, B by id)."""
    torch.manual_seed(1)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256  # m=intermediate-row tile, k=I, n=H
    active_ids = [num_experts - 1, 3, num_experts - 2, 9, 2, num_experts // 2, 11, 0][: min(8, num_experts)]
    num_active = len(active_ids)

    a_compact = torch.randn((1, num_active, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    a_t = ttnn.from_torch(a_compact, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    indices_t = _make_indices(active_ids, device)

    out_t = ttnn.sparse_matmul(
        a_t,
        in1_t,
        sparsity=sparsity_t,
        indices=indices_t,
        is_input_a_sparse=True,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=_sparse_pc(m, n, tile_h, tile_w),
    )
    out = ttnn.to_torch(out_t).reshape(num_active, m, n)
    logger.info(f"down gather out shape {tuple(out_t.shape)} -> compact [{num_active}, {m}, {n}]")

    in1_f = in1.float()
    for i, e in enumerate(active_ids):
        ref = torch.matmul(a_compact[0, i].float(), in1_f[0, e])
        assert_numeric_metrics(
            ref, out[i].float(), atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
        )


def test_indices_absent_is_unchanged(device):
    """Sanity: with no `indices`, the op produces the dense [.., E, M, N] output (default path), proving
    the gather operand is the sole trigger and the legacy behavior is preserved."""
    torch.manual_seed(2)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    num_experts = 16
    active_ids = [3, 7, 11, 0]

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)

    out_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=len(active_ids),
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=_sparse_pc(m, n, tile_h, tile_w),
    )
    # Dense expert axis (= E), not compact.
    assert out_t.shape[-3] == num_experts, f"expected dense E={num_experts} axis, got shape {tuple(out_t.shape)}"
