# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does a BIGGER speculative block buy throughput, now that the verify forward is traced?

With the verify traced and its readback narrowed, the loop runs at 18.82 tok/s (53 ms/tok) at
``block_size=16``, committing 7.000 tokens per target forward. Every other lever on the table saves
a FIXED amount per step; this one is a MULTIPLIER -- if a 32-slot block still commits ~2x the
tokens, tok/s scales with it and every later saving compounds.

Why it might be free on the cost side:

* The drafter's matmuls run at M = q_len padded to a 32-row tile, so a 32-token block costs the
  drafter the SAME device time as a 16-token one (see DFLASH_DRAFTER_OP_MAPPING.md, lever 1:
  "block_size = 32 would cost the same device time as 16").
* The verify bucket is already 128 rows, so 32 slots still fits one forward, and ONE trace serves
  any valid_len below the bucket (measured in test_dflash_target_trace_replay.py).

Why it might not pay:

* The drafter was TRAINED at block_size 16 -- "1 anchor + 15 drafted". Asking it for 31 drafted
  slots is asking it to predict twice as far ahead as it was trained to. Acceptance past slot 15
  may collapse, in which case the extra slots are wasted drafting and the step gets slower, not
  faster. That is the whole question, and it is empirical.

Greedy, so every block size must emit IDENTICAL tokens -- the target verifies each slot and accepts
only its own argmax. Block size changes how many tokens a step commits, never which ones.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_block_size_sweep.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.drafters import TtDrafter
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
MAX_NEW_TOKENS = 64
BLOCK_SIZES = (16, 24, 32)
PROMPT = "The capital of France is"


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_block_size_sweep(mesh_device, device_params, reset_seeds, ensure_gc):
    """tok/s and acceptance at several block sizes, traced verify, same prompt."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    from transformers import AutoTokenizer

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(drafter_path)), target)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids

    # EAGER verify on purpose. Acceptance vs block size is a model-quality question -- how far the
    # drafter can usefully predict -- and is independent of whether the verify is traced. Running it
    # eagerly also avoids a real hazard: capturing the trace first and then changing block size makes
    # the drafter allocate new q_len-shaped tensors while a trace is parked, which Metal warns about
    # ("Allocating device buffers is potentially unsafe due to the existence of an active trace")
    # and which killed the first version of this test after its bs=16 run.
    #
    # So the tok/s column here is the EAGER rate: use it for the RATIO between block sizes, not as a
    # throughput figure. The traced rate at bs=16 is 18.82 tok/s.
    logger.info(f"drafter's native block_size is {cfg.block_size} (1 anchor + {cfg.block_size - 1} drafted)")

    rows, baseline_ids = [], None
    for bs in BLOCK_SIZES:
        t0 = time.perf_counter()
        stats = dflash_generate(
            drafter, target, prompt, max_new_tokens=MAX_NEW_TOKENS, block_size=bs, return_stats=True
        )
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        rows.append((bs, n / dt, dt * 1000 / n, stats.mean_acceptance_length, len(stats.acceptance_lengths)))
        logger.info(
            f"block_size={bs:2d}: {n / dt:6.2f} tok/s ({dt * 1000 / n:5.1f} ms/tok), "
            f"acceptance {stats.mean_acceptance_length:.3f}/step over {len(stats.acceptance_lengths)} steps"
        )
        if baseline_ids is None:
            baseline_ids = stats.output_ids
        else:
            # Greedy: block size changes how MANY tokens commit per step, never WHICH.
            assert torch.equal(baseline_ids, stats.output_ids), (
                f"block_size={bs} produced different tokens than block_size={BLOCK_SIZES[0]}; greedy "
                "speculation is exact, so this is a correctness bug, not a tuning effect"
            )

    logger.info("=" * 76)
    for bs, tps, mspt, acc, steps in rows:
        logger.info(f"  block {bs:2d}: {tps:6.2f} tok/s  {mspt:5.1f} ms/tok  acceptance {acc:5.2f}  {steps:2d} steps")
    logger.info("  (eager verify: compare the RATIO across rows; traced bs=16 is 18.82 tok/s)")
    logger.info("=" * 76)
    print("\n>>> " + " | ".join(f"bs{bs}: {tps:.2f} tok/s acc {acc:.2f}" for bs, tps, _, acc, _ in rows) + "\n")
