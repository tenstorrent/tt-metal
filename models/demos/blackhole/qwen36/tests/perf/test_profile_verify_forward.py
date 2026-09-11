# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op device profile of ONE DFlash verify forward — what is the ~450 ms actually made of?

The verify forward is now traced (1.96x end to end, 3.68 -> 7.21 tok/s), and a step is ~973 ms of
which ~853 ms is target: ~450 ms per forward, against production traced decode at 56 ms/tok. Two
questions decide what to do next, and one capture answers both:

1. **How much dispatch is left?** Compare this capture's total device time against the ~450 ms of
   wall per forward. A small gap means the forward is now device-bound and no further tracing or
   dispatch work helps it.
2. **Row-proportional or weight-bound?** This decides whether shrinking the bucket from 128 rows to
   32 is worth ~180 ms/step or ~20. The forward verifies a 16-token block using 128 rows. If its
   cost is dominated by the 64 layers' weight streaming, fewer rows buys nothing -- the same 27B
   weights stream either way, and at M=128 vs M=32 both are weight-bound. If it is dominated by
   work that scales with sequence length (SDPA over 128 positions, the GDN chunk scan, the norms),
   a 32-row bucket removes ~3/4 of it.

   The ratio hints at the answer -- 450 ms for a 128-row forward against 56 ms/tok for M=1 decode
   is 8x, which is hard to explain by weights alone -- but a ratio is not a profile.

This profiles the EAGER masked forward on purpose. Device time per op is the same whether the op is
dispatched eagerly or replayed from a trace; tracing removes host gaps, not kernel time. So this is
the floor the traced path is converging on, and it avoids asking whether the profiler attributes
ops correctly inside a replay.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      python -m tracy -p --op-support-count 100000 -r -v -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_verify_forward.py"

    D=<new report dir>
    tt-perf-report generated/profiler/reports/$D/ops_perf_results_$D.csv \\
      --start-signpost start --end-signpost stop --no-color
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
BLOCK = 16


def _tracy_signpost():
    try:
        from tracy import signpost

        return signpost
    except ImportError:  # pragma: no cover
        return None


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_profile_verify_forward(mesh_device, device_params, reset_seeds, ensure_gc):
    """One masked verify forward between signposts; weight upload and warm-up sit outside."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    bucket = model._mask_bucket_for(BLOCK)

    g = torch.Generator().manual_seed(3)
    ids = torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.int32)

    # Warm OUTSIDE the window: an un-warmed capture measures JIT, not the forward.
    for _ in range(2):
        model.prefill_block_all_logits(ids, page_table, actual_len=BLOCK, chunk_start=0)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"warmed: verifying a {BLOCK}-token block through a {bucket}-row bucket, 64 layers")

    signpost = _tracy_signpost()
    if signpost:
        signpost("start")
    out = model.prefill_block_all_logits(ids, page_table, actual_len=BLOCK, chunk_start=0)
    ttnn.synchronize_device(mesh_device)
    if signpost:
        signpost("stop")

    assert out.shape[1] == BLOCK, f"expected all-row logits for {BLOCK} rows, got {tuple(out.shape)}"
    logger.info(f"verify forward produced logits {tuple(out.shape)}")
