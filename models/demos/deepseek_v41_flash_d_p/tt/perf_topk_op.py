# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.13 (INDEX_TOPK_OP, DS41-151): device time of the indexers' top-k, fork (ttnn.bringup.topk_large_indices) vs stock
(ttnn.experimental.topk_large_indices), one card, at the Galaxy 8x4 per-chip calls (160 query rows a chip after P.8 /
P.10): the ladder's last chunk [51200, 56320) as P.12's op table captured them (index r2 160 x 28160 k 512, candidate
blocks 160 x 7040 k 2048, index r1 160 x 56320 k 512) and the same calls at 1M (522240 / 130560 / 1044480).
Both ops run on the same input in one process, REPS times each after a warm call. Run under the profiler:

    scripts/run_safe_pytest.sh --device auto --profile --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_topk_op.py

and read DEVICE KERNEL DURATION of the ops (the op name tells fork from stock) in the printed CSV."""

import pytest
import torch

import ttnn

REPS = 5

CALLS = [
    ("r2-28160", 160, 28160, 512),
    ("cand-7040", 160, 7040, 2048),
    ("r1-56320", 160, 56320, 512),
    ("1M-r2-522240", 160, 522240, 512),
    ("1M-cand-130560", 160, 130560, 2048),
    ("1M-r1-1044480", 160, 1044480, 512),
]


@pytest.mark.parametrize("sid, sq, n, k", CALLS, ids=[c[0] for c in CALLS])
def test_topk_fork_vs_stock(device, sid, sq, n, k):
    x = torch.randn(1, 1, sq, n, generator=torch.Generator().manual_seed(n + k)).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    for op in (ttnn.experimental.topk_large_indices, ttnn.bringup.topk_large_indices):
        for _ in range(1 + REPS):
            ttnn.deallocate(op(t, k=k, valid_length=n))
        ttnn.synchronize_device(device)
