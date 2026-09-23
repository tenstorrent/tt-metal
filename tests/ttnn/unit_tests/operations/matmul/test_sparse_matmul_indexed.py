# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Op-level tests for the indexed/gather mode of ttnn.sparse_matmul.

The new optional `indices` operand is exercised with a hard-coded, NON-MONOTONIC id list, proving
(1) it compiles + runs without hanging (kernel lockstep), (2) it matches a torch reference, (3) the
`is_input_a_sparse=True` compact-A path, (4) bf4 weight addressing by arbitrary group id, (5) that a
program-cache hit re-dispatches a *different* indices buffer correctly, (6) that the dense
sparsity-scan path is value-identical when `indices` is absent, and (7) that the operand's host-side
contract is enforced.
"""

import math

from loguru import logger
import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics


def _sparse_pc(m, n, tile_h, tile_w):
    """A 1D-optimized program config that spreads N across a core grid with per_core_N=1, mirroring
    the production gather caller (an MoE expert projection)."""
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
    """[1,1,1,num_active] UINT16 ROW_MAJOR device tensor of group ids (compact-output slot order)."""
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, len(active_ids))
    return ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _make_sparsity(active_ids, num_experts, device):
    """[1,1,1,E] bf16 ROW_MAJOR mask, nonzero exactly at the active group ids.

    In indexed mode neither kernel reads this tensor -- the indexed loop visits only active groups, so
    there is no validity scan -- but `sparsity` remains a required positional operand of the op.
    """
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


def test_gather_program_cache_reuses_with_new_indices(device):
    """A cached program must re-dispatch a *different* indices buffer.

    The indices address is patched in override_runtime_arguments; if that patch were missing or wrong,
    the second call would silently gather the first call's groups. Run the same configuration twice
    with two equal-shaped, separately allocated, non-monotonic id lists and require (a) no new program
    cache entry and (b) each output to match its own reference."""
    torch.manual_seed(3)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    num_experts = 16
    active_ids_a = [15, 2, 9, 0, 13, 4, 7, 11]
    active_ids_b = [1, 14, 6, 12, 3, 10, 5, 8]
    num_active = len(active_ids_a)
    assert len(active_ids_b) == num_active
    assert active_ids_a != active_ids_b

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = _sparse_pc(m, n, tile_h, tile_w)

    # Allocate both id lists (and both masks) up front so they land at distinct addresses.
    sparsity_a = _make_sparsity(active_ids_a, num_experts, device)
    indices_a = _make_indices(active_ids_a, device)
    sparsity_b = _make_sparsity(active_ids_b, num_experts, device)
    indices_b = _make_indices(active_ids_b, device)
    assert indices_a.buffer_address() != indices_b.buffer_address()

    def run(sparsity_t, indices_t):
        out_t = ttnn.sparse_matmul(
            in0_t,
            in1_t,
            sparsity=sparsity_t,
            indices=indices_t,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program_config,
        )
        return ttnn.to_torch(out_t).reshape(num_active, m, n)

    out_a = run(sparsity_a, indices_a)
    cache_entries_after_first = device.num_program_cache_entries()
    out_b = run(sparsity_b, indices_b)
    cache_entries_after_second = device.num_program_cache_entries()
    assert (
        cache_entries_after_second == cache_entries_after_first
    ), "the second indexed call should reuse the cached program"

    in1_f = in1.float()
    for out, active_ids, tag in ((out_a, active_ids_a, "first"), (out_b, active_ids_b, "second")):
        for i, e in enumerate(active_ids):
            ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
            logger.info(f"{tag} call, compact slot {i} -> group {e}")
            assert_numeric_metrics(
                ref, out[i].float(), atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
            )


def test_indices_absent_is_unchanged(device):
    """With no `indices`, the op must produce the unchanged dense [.., E, M, N] result: the active
    slots hold their products and the skipped slots stay zero-filled. Guards the claim that the
    gather operand is the sole trigger and the legacy path is untouched."""
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

    out = ttnn.to_torch(out_t).reshape(num_experts, m, n).float()
    in1_f = in1.float()
    for e in range(num_experts):
        if e in active_ids:
            ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
            assert_numeric_metrics(
                ref, out[e], atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
            )
        else:
            assert torch.count_nonzero(out[e]) == 0, f"inactive expert slot {e} must stay zero-filled"


####################################################################################################
# Host-side contract on the `indices` operand
####################################################################################################


def _contract_inputs(device, num_experts=16, m=32, k=128, n=256):
    torch.manual_seed(4)
    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    active_ids = [3, 7, 11, 0]
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    return in0_t, in1_t, sparsity_t, active_ids, _sparse_pc(m, n, 32, 32), (m, k, n, num_experts)


def _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc, **kwargs):
    return ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        indices=indices_t,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        **kwargs,
    )


def test_indices_rejects_multi_stick_shape(device, expect_error):
    """The id list is fetched with a single page-0 read, so it must live in one ROW_MAJOR stick.
    A same-volume [1,1,N,1] tensor is N separate one-element pages."""
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, len(active_ids), 1)
    indices_t = ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, "single ROW_MAJOR stick"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_host_tensor(device, expect_error):
    """A host id list must never reach the program factory, which dispatches its buffer address.

    is_allocated() alone is true for host tensors, so the op validates device storage explicitly; the
    generic device-operation launch guard happens to reject this case one layer earlier, which is why
    the message below is the framework's rather than sparse_matmul's. The op's own check still covers
    what the framework does not: an indices tensor resident on a *different* device than the inputs.
    """
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, len(active_ids))
    indices_t = ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)  # no device=

    with expect_error(RuntimeError, "Device Operations expect device tensors as inputs"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_wrong_dtype(device, expect_error):
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    t = torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, len(active_ids))
    indices_t = ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with expect_error(RuntimeError, "indices must be UINT16 dtype"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_more_than_num_groups(device, expect_error):
    """num_active must be <= the number of sparse groups in B, which is B's group axis (not the
    product of A's and B's batch lengths)."""
    in0_t, in1_t, sparsity_t, _, pc, dims = _contract_inputs(device)
    num_experts = dims[3]
    indices_t = _make_indices(list(range(num_experts)) + [0], device)

    with expect_error(RuntimeError, "must be <= the number of sparse groups"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc)


def test_indices_rejects_nnz(device, expect_error):
    """nnz would be silently ignored in indexed mode (the loop count is num_active)."""
    in0_t, in1_t, sparsity_t, active_ids, pc, _ = _contract_inputs(device)
    indices_t = _make_indices(active_ids, device)

    with expect_error(RuntimeError, "must not be supplied together with indices"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc, nnz=len(active_ids))


def test_indices_requires_input_b_sparse(device, expect_error):
    """The ids address B's sparse-group axis, so B must be the sparse operand."""
    torch.manual_seed(6)
    m, k, n = 32, 128, 256
    # A-sparse-only mode: the sparsity length is A's batch length, which is 1 for [1,1,M,K].
    in0_t = ttnn.from_torch(
        torch.randn((1, 1, m, k), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    in1_t = ttnn.from_torch(
        torch.randn((1, 1, k, n), dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    sparsity_t = ttnn.from_torch(
        torch.ones((1, 1, 1, 1), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    indices_t = _make_indices([0], device)

    with expect_error(RuntimeError, "requires is_input_b_sparse"):
        ttnn.sparse_matmul(
            in0_t,
            in1_t,
            sparsity=sparsity_t,
            indices=indices_t,
            is_input_a_sparse=True,
            is_input_b_sparse=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_sparse_pc(m, n, 32, 32),
        )


def test_indices_rejects_mismatched_optional_output(device, expect_error):
    """A preallocated output must have the indexed (compact) shape: a full-E tensor would be left with
    holes, and an undersized one would be written out of bounds."""
    in0_t, in1_t, sparsity_t, active_ids, pc, dims = _contract_inputs(device)
    m, _, n, num_experts = dims
    indices_t = _make_indices(active_ids, device)
    full_e_output = ttnn.from_torch(
        torch.zeros((1, 1, 1, num_experts, m, n), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    with expect_error(RuntimeError, "must match the indexed output shape"):
        _run_indexed(in0_t, in1_t, sparsity_t, indices_t, pc, optional_output_tensor=full_e_output)


def test_indexed_optional_output(device):
    """The indexed-shaped preallocated output is accepted and receives the gathered results."""
    torch.manual_seed(5)
    tile_h, tile_w = 32, 32
    m, k, n = 32, 128, 256
    num_experts = 16
    active_ids = [13, 1, 8, 4]
    num_active = len(active_ids)

    in0 = torch.randn((1, 1, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    sparsity_t = _make_sparsity(active_ids, num_experts, device)
    indices_t = _make_indices(active_ids, device)
    preallocated = ttnn.from_torch(
        torch.full((1, 1, 1, num_active, m, n), 99.0, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    out_t = _run_indexed(
        in0_t, in1_t, sparsity_t, indices_t, _sparse_pc(m, n, tile_h, tile_w), optional_output_tensor=preallocated
    )
    out = ttnn.to_torch(out_t).reshape(num_active, m, n).float()

    in1_f = in1.float()
    for i, e in enumerate(active_ids):
        ref = torch.matmul(in0[0, 0].float(), in1_f[0, e])
        assert_numeric_metrics(
            ref, out[i], atol=0.05 * k, rtol=10.0 * k, frobenius_threshold=0.01 * k, pcc_threshold=0.99
        )
