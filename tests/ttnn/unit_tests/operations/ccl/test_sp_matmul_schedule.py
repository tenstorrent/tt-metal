# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Standalone check of the sequence-parallel "slice schedule" mechanism in the 2D-mcast matmul.

`ttnn.experimental.sp_matmul_schedule_test` runs the matmul with a MatmulFusedOpSignaler(SP_REDUCE_SCATTER) that
carries a caller-provided permutation of the sub-batches (no CCL, no waits). With the same program config
(fuse_batch=False) the result must be bitwise identical to `ttnn.matmul` on the [B*T,1,Ms,K] view, whatever the order
in which the sub-batches are processed.
"""

import random

import pytest
import torch

import ttnn


def _pack(order, ag_mode=False):
    """order: list of (in0_idx, out_idx) per matmul iteration.

    rs mode: wait/local bits stay 0 (no CCL). ag mode: alternate is_local (bit 17, read through the alternate in0
    accessor) and wait_dir (bit 16) so both direction semaphores get a (trivially satisfied, wait_count=0) wait_min.
    """
    words = []
    for j, (in0_idx, out_idx) in enumerate(order):
        w = (in0_idx & 0xFF) | ((out_idx & 0xFF) << 8)
        if ag_mode:
            is_local = j % 2 == 0
            wait_dir = (j // 2) % 2
            w |= (int(is_local) << 17) | (wait_dir << 16)
        words.append(w)
    return words


def _schedules(n, seed):
    identity = [(i, i) for i in range(n)]
    reversed_ = [(n - 1 - i, n - 1 - i) for i in range(n)]
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    # random: process sub-batch perm[j] at iteration j (in0_idx == out_idx, i.e. a re-ordering of the work)
    random_same = [(p, p) for p in perm]
    # random with in0 and out permuted independently: output slot out_idx receives in0 slot in0_idx. Reference is a
    # matmul whose rows have been permuted accordingly, built below from the plain result.
    perm2 = list(range(n))
    rng.shuffle(perm2)
    random_mixed = [(perm[j], perm2[j]) for j in range(n)]
    return {
        "identity": identity,
        "reversed": reversed_,
        "random": random_same,
        "random_mixed": random_mixed,
    }


def _compute_kernel_config(fp32_acc):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


@pytest.mark.parametrize("bt", [4, 8])
@pytest.mark.parametrize("ms", [64, 512])
@pytest.mark.parametrize("k", [128, 4096])
@pytest.mark.parametrize("n", [128, 1536])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("fp32_acc", [False, True])
@pytest.mark.parametrize("mode", ["rs", "ag"])
def test_sp_matmul_schedule(device, bt, ms, k, n, transpose_b, fp32_acc, mode):
    torch.manual_seed(0)
    if bt * ms * n * 2 > 256 * 1024 * 1024:
        pytest.skip("output too large for a quick check")
    if mode == "ag" and fp32_acc:
        pytest.skip("the AG wait/alt-address path does not interact with the accumulation format")
    ag_mode = mode == "ag"

    in0 = torch.randn(bt, 1, ms, k, dtype=torch.bfloat16)
    in1 = (
        torch.randn(1, 1, n, k, dtype=torch.bfloat16) if transpose_b else torch.randn(1, 1, k, n, dtype=torch.bfloat16)
    )

    mem = ttnn.DRAM_MEMORY_CONFIG
    tt_in0 = ttnn.from_torch(in0, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tt_in1 = ttnn.from_torch(in1, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)

    ckc = _compute_kernel_config(fp32_acc)
    grid_size = device.compute_with_storage_grid_size()
    # leave one column free so the test op has a dummy core for the RS-style signal
    grid = ttnn.CoreCoord(min(grid_size.x - 1, 8), min(grid_size.y, 8))
    program_config = ttnn.experimental.sp_matmul_program_config(tt_in0, tt_in1, grid, transpose_b, ckc)
    assert program_config.fuse_batch is False

    ref = ttnn.matmul(
        tt_in0,
        tt_in1,
        transpose_b=transpose_b,
        program_config=program_config,
        compute_kernel_config=ckc,
        memory_config=mem,
    )
    ref_torch = ttnn.to_torch(ref)
    assert ref_torch.shape == (bt, 1, ms, n)
    # sanity: the reference itself is a correct matmul. Only a PCC guard: with fp32_dest_acc_en=False the plain
    # matmul accumulates its K-block partials in bf16 (up to 32 blocks at K=4096), which is far from allclose but is
    # exactly what the scheduled run must reproduce bitwise below.
    golden = in0.float() @ (in1.transpose(-1, -2) if transpose_b else in1).float()
    assert _pcc(ref_torch.float(), golden) > 0.99

    for name, order in _schedules(bt, seed=bt * 1000 + ms + k + n + int(transpose_b)).items():
        # twice: the second call is a program-cache hit (override_runtime_arguments, incl. the AG alt address)
        for _ in range(2):
            out = ttnn.experimental.sp_matmul_schedule_test(
                tt_in0,
                tt_in1,
                _pack(order, ag_mode),
                transpose_b,
                program_config,
                compute_kernel_config=ckc,
                memory_config=mem,
                ag_mode=ag_mode,
            )
            out_torch = ttnn.to_torch(out)
            expected = torch.empty_like(ref_torch)
            for in0_idx, out_idx in order:
                expected[out_idx] = ref_torch[in0_idx]
            assert torch.equal(out_torch, expected), f"schedule {name} ({order}) differs from ttnn.matmul bitwise"


@pytest.mark.parametrize("b, t, s, k", [(1, 4, 256, 128), (2, 4, 512, 4096), (3, 2, 64, 256)])
def test_sub_batched_view(device, expect_error, b, t, s, k):
    """sub_batched_view is the [B,1,S,X] -> [B*T,1,S/T,X] view both fused ops feed to the scheduled matmul."""
    torch.manual_seed(1)
    x = torch.randn(b, 1, s, k, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    view = ttnn.experimental.sp_sub_batched_view(tt_x, t)
    assert list(view.shape) == [b * t, 1, s // t, k]
    assert view.buffer_address() == tt_x.buffer_address()
    assert torch.equal(ttnn.to_torch(view), x.reshape(b * t, 1, s // t, k))

    # and the view drives the scheduled matmul like a plain [B*T,1,S/T,K] input
    w = torch.randn(1, 1, k, 128, dtype=torch.bfloat16)
    tt_w = ttnn.from_torch(w, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ckc = _compute_kernel_config(False)
    program_config = ttnn.experimental.sp_matmul_program_config(view, tt_w, ttnn.CoreCoord(4, 4), False, ckc)
    ref = ttnn.to_torch(ttnn.matmul(view, tt_w, program_config=program_config, compute_kernel_config=ckc))
    order = [(i, b * t - 1 - i) for i in range(b * t)]
    out = ttnn.to_torch(
        ttnn.experimental.sp_matmul_schedule_test(
            view, tt_w, _pack(order), False, program_config, compute_kernel_config=ckc
        )
    )
    assert torch.equal(out, ref.flip(0))
    with expect_error(RuntimeError, "must be a multiple of num_slices"):
        ttnn.experimental.sp_sub_batched_view(tt_x, t * 64)  # S not divisible by num_slices*32


def test_sp_matmul_schedule_rejects_non_permutation(device, expect_error):
    in0 = torch.randn(4, 1, 64, 128, dtype=torch.bfloat16)
    in1 = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
    tt_in0 = ttnn.from_torch(in0, layout=ttnn.TILE_LAYOUT, device=device)
    tt_in1 = ttnn.from_torch(in1, layout=ttnn.TILE_LAYOUT, device=device)
    ckc = _compute_kernel_config(False)
    program_config = ttnn.experimental.sp_matmul_program_config(tt_in0, tt_in1, ttnn.CoreCoord(2, 2), False, ckc)
    with expect_error(RuntimeError, "in0_idx .* not a permutation"):
        ttnn.experimental.sp_matmul_schedule_test(
            tt_in0, tt_in1, _pack([(0, 0), (0, 1), (2, 2), (3, 3)]), False, program_config
        )
    with expect_error(RuntimeError, "schedule words for .* sub-batches"):
        ttnn.experimental.sp_matmul_schedule_test(
            tt_in0, tt_in1, _pack([(0, 0), (1, 1), (2, 2)]), False, program_config
        )
    # rs mode rejects wait/local bits; ag mode rejects a non-zero wait_count (would hang without an all-gather)
    with expect_error(RuntimeError, "wait/local bits must be 0"):
        ttnn.experimental.sp_matmul_schedule_test(
            tt_in0, tt_in1, _pack([(0, 0), (1, 1), (2, 2), (3, 3)], ag_mode=True), False, program_config
        )
    with expect_error(RuntimeError, "wait_count must be 0"):
        words = _pack([(0, 0), (1, 1), (2, 2), (3, 3)], ag_mode=True)
        words[1] |= 1 << 24
        ttnn.experimental.sp_matmul_schedule_test(tt_in0, tt_in1, words, False, program_config, ag_mode=True)
