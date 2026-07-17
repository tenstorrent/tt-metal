# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reader-NoC ablation: confirm the flagged shape is COMPUTE-bound, not DM-bound.

The decisive /perf-measure classify-the-bound experiment for the flagged
1x10x9472x128 shape. FlashAttention-2 re-reads all of K/V once per q-chunk, so this
shape moves ~n_q_chunks(37) x 2 x 24 MB ~= 1.8 GB of DRAM reads — enough that, at a
plausible ~340 GB/s, DRAM *could* by itself explain ~5 ms. So DM-bound is a live
alternative that must be ruled out by direct measurement, not assumed.

Method (measurement-only compile-time gates, byte-identical at their defaults):

  SDPA_ABLATE_READER=1  -> reader NoC stub: every noc_async_read_tile + barrier skipped,
                           cb_reserve/push kept -> zero READ bytes, CB counts unchanged.
  SDPA_ABLATE_WRITER=1  -> writer NoC stub: every noc_async_write_tile + barrier skipped,
                           cb_wait/pop kept -> zero WRITE bytes, CB counts unchanged.

Four same-session variants isolate each half of the data movement:
  baseline / reader-stub / writer-stub / reader+writer-stub.

If a stubbed variant runs at the same wall-time as baseline, that traffic was fully
HIDDEN behind compute (compute-bound). If it is much faster, that traffic was on the
critical path (DM-bound). All variants run SAME-SESSION (one process, steady AICLK) to
defeat the ~1.8x clock drift between fresh pytest invocations that invalidates cross-run
ns A/B. Correctness is NOT asserted (the stubs feed compute garbage) — this is perf only.

Measure:
  scripts/run_safe_pytest.sh --profile \
    tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_scaled_dot_product_attention_reader_ablation.py

Then take, per variant, the median of the warm rows (drop the first) of the SDPA op's
DEVICE KERNEL DURATION [ns] column in the emitted ops_perf_results CSV.
"""

from __future__ import annotations

import math
import os

import torch
import ttnn
from loguru import logger

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

PERF_SHAPE = (1, 10, 9472, 128)
ITERS = 6  # 1 warm-up + 5 measured per variant


def _fa_rand(*shape):
    n1 = torch.randn(shape)
    n2 = torch.randn(shape) * 10
    b = torch.bernoulli(torch.full(shape, 0.001))
    return n1 + n2 * b


def test_sdpa_reader_ablation(device):
    torch.manual_seed(1234)
    B, H, S, D = PERF_SHAPE
    Q, K, V = _fa_rand(B, H, S, D), _fa_rand(B, H, S, D), _fa_rand(B, H, S, D)
    scale = 1.0 / math.sqrt(D)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    def run_variant(label, reader, writer):
        for key, val in (("SDPA_ABLATE_READER", reader), ("SDPA_ABLATE_WRITER", writer)):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(val)
        out = None
        for _ in range(ITERS):
            out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
        ttnn.to_torch(out)  # force completion
        logger.info(f"dm-ablation [{label}] done ({ITERS} iters)")

    # Same process => shared AICLK. Four variants isolate the reader (K/V re-reads),
    # the writer (output write-back), and both (total DRAM traffic) against the baseline.
    run_variant("baseline (all DRAM traffic)", None, None)
    run_variant("reader NoC stub (zero read bytes)", 1, None)
    run_variant("writer NoC stub (zero write bytes)", None, 1)
    run_variant("reader+writer stub (zero DRAM bytes)", 1, 1)
    os.environ.pop("SDPA_ABLATE_READER", None)
    os.environ.pop("SDPA_ABLATE_WRITER", None)
