# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark for the one number that decides the Q-block rewrite's shape.

Q1-Q3 move the SDPA compute unit from a Q chunk (256-320 rows, 8-10 tile rows) to a Q block (64 rows, 2 tile rows),
because selection happens at 64 and the OR up to the chunk throws the sparsity away. Under random sparsity no two
Q blocks share a page, so M cannot be batched back up -- M = 2 tile rows is forced. If the matmul is badly inefficient
there, Q2's shape is wrong and the rewrite needs a different decomposition.

Everything here runs dense on ONE device. Total FLOPs are identical across a sweep, so a time difference is the
per-shape efficiency difference and nothing else.

  A. q_chunk sweep  -> the small-M penalty.          Is M = 64 rows affordable?
  B. k_chunk sweep  -> the fixed per-visit cost.     Independently measures the ~8 us/visit the plan rests on.

Run:   pytest models/tt_dit/tests/models/minimax_h3/test_sol_qblock_microbench_minimax_h3.py -s
       (-s so the tables print; they are the point of the file)
Profile: each case issues ITERS back-to-back identical ops, so a device-profiler capture shows clean per-op spans.
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn

B, NH, D = 1, 8, 128
SEQ = 4096  # one device, no ring: big enough for the chunk sweeps to be meaningful, small enough to iterate
WARMUP, ITERS = 3, 20

Q_CHUNKS = [64, 128, 256, 320]  # 2, 4, 8, 10 tile rows. 64 is what Q2 would use; 256/320 are production today.
K_CHUNKS = [128, 256, 512, 1024]  # fixed math, 4x spread in visit count -> slope gives cost per visit


def _time_sdpa(device, q_chunk, k_chunk):
    """Median wall-clock ms for one dense SDPA at this blocking. Same FLOPs for every (q_chunk, k_chunk)."""
    torch.manual_seed(0)
    shape = (B, NH, SEQ, D)
    tt = [
        ttnn.from_torch(torch.randn(shape) * 0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for _ in range(3)
    ]
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
    )

    def once():
        out = ttnn.transformer.scaled_dot_product_attention(
            *tt, is_causal=False, scale=D**-0.5, program_config=program_config, compute_kernel_config=ckc
        )
        ttnn.deallocate(out)

    for _ in range(WARMUP):
        once()
    ttnn.synchronize_device(device)

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        once()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    for t in tt:
        ttnn.deallocate(t)
    return sorted(samples)[len(samples) // 2]


def _report(title, rows, header):
    logger.info(f"\n=== {title} ===\n{header}\n" + "\n".join(rows))


def test_q_chunk_sweep(device):
    """A. The small-M penalty. Identical FLOPs at every q_chunk; time ratio is pure efficiency."""
    ms = {q: _time_sdpa(device, q, 512) for q in Q_CHUNKS}
    base = ms[256]
    rows = [
        f"  q_chunk {q:>4} ({q // 32:>2} tile rows)   {ms[q]:7.3f} ms   {base / ms[q]:5.2f}x of q256" for q in Q_CHUNKS
    ]
    _report("A. q_chunk sweep (k_chunk 512, dense, 1 device)", rows, "  setting                        time   relative")
    logger.info(f"  => M=64 costs {ms[64] / base:.2f}x the time of M=256 for the same math")


def test_m64_is_affordable(device):
    """The decision gate for Q2's shape.

    Q1-Q3 predicts ~3.4x less math. If M = 64 runs slower than ~1.7x of M = 256 per FLOP, half the predicted gain is
    eaten by matmul inefficiency alone and the Q block is the wrong compute unit -- revisit before building the plan
    format around it. The threshold is deliberately loose: this is a go/no-go on the shape, not a perf gate.
    """
    slow = _time_sdpa(device, 64, 512)
    fast = _time_sdpa(device, 256, 512)
    ratio = slow / fast
    logger.info(f"[gate] M=64 vs M=256 for identical math: {ratio:.2f}x")
    assert ratio < 1.7, (
        f"M=64 rows costs {ratio:.2f}x the time of M=256 for the same FLOPs. Q2 assumes a 64-row compute unit "
        f"is affordable; at this ratio it is not, and the Q-block decomposition needs rethinking."
    )


def test_k_chunk_sweep(device):
    """B. Cost per visit, measured rather than taken from the profile.

    Math is fixed; halving k_chunk doubles the number of loop iterations. Fitting time against visit count gives the
    fixed per-visit cost directly -- the ~8 us/visit figure that the whole 'visits do not scale' argument rests on.
    """
    ms = {k: _time_sdpa(device, 256, k) for k in K_CHUNKS}
    visits = {k: (SEQ // k) for k in K_CHUNKS}  # per Q chunk, per head
    rows = [f"  k_chunk {k:>5}   {visits[k]:>4} visits   {ms[k]:7.3f} ms" for k in K_CHUNKS]
    _report("B. k_chunk sweep (q_chunk 256, dense, 1 device)", rows, "  setting                            time")

    lo, hi = K_CHUNKS[-1], K_CHUNKS[0]  # fewest visits vs most visits
    extra_visits = (visits[hi] - visits[lo]) * (SEQ // 256) * NH * B
    per_visit_us = (ms[hi] - ms[lo]) * 1e3 / max(extra_visits, 1)
    logger.info(f"  => ~{per_visit_us:.2f} us per visit ({extra_visits} extra visits cost {ms[hi] - ms[lo]:.3f} ms)")


@pytest.mark.parametrize("n_tiles", [12, 14, 16])
def test_awkward_n_factorization(device, n_tiles):
    """Q5: the kernel picks its sub-block width as largest_factor_le(N, max_sbw).

    After compaction N = 2 x kept tiles, so kept = 7 gives N = 14, whose largest factor <= 4 is 2 -- half the usual
    sub-block width. This drives SDPA at k_chunk = N*32 so the awkward widths are exercised end to end; compare 14
    against its neighbours 12 and 16 to see what the collapse actually costs before Q5 pads it away.
    """
    ms = _time_sdpa(device, 256, n_tiles * 32)
    logger.info(f"[sbw] N = {n_tiles} tiles (k_chunk {n_tiles * 32}): {ms:.3f} ms")
