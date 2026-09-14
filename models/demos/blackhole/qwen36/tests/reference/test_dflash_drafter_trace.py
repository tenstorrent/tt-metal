# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 3: does a REPLAYED drafter step emit the same tokens as an eagerly dispatched one?

MEASURED 2026-09-14 (T3K, 64 tokens, eager verify, C=512): **yes -- and it is exactly 1.00x.**

    eager  drafter  4.12 tok/s  acceptance 6.300
    TRACED drafter  4.10 tok/s  acceptance 6.300   tokens identical

The capture is CORRECT and buys nothing yet. Per-phase timing
(tests/perf/test_dflash_drafter_trace_breakdown.py) says why, and it is not subtle: of a 75 ms
traced step the replay is 10 ms, while staging (14.5 ms) and commit_staged_context (17.2 ms) add
back roughly what the forward's ~174 eager dispatches cost. A trace can hold neither -- staging is
host I/O by definition, and the commit's offset advances with the accept count.

    lm_head + readback  33.3 ms  (44 %)   <- device-side argmax would take most of this
    commit              17.2 ms  (23 %)   <- 20 dispatches a capture cannot hold
    replay              10.1 ms  (13 %)
    stage_step           8.9 ms  (12 %)   <- two [1,1,16,C+32] masks, scales with C not context
    stage_taps           4.8 ms
    stage_tokens         0.8 ms

So the remaining work is ordered by that table, not by intuition: argmax, then the commit, then
capacity. Do not read the trace as a failure -- it is the precondition, and it is now exact.

Two measurement traps hit while producing those numbers, both mine:

* Running the traced arm FIRST (correct for state, see below) charged it with JIT compilation for
  programs both arms share, which read as 0.85x. Each arm now runs a short throwaway generation
  before timing.
* The "drafter is ~120 ms of dispatch a trace erases" figure came from the LEGACY growing-history
  drafter. The fixed-capacity forward is cheaper in dispatch terms, so there was less to reclaim
  than the lever was scoped on.

The drafter is host-dispatch-bound -- ~9 % device-utilized at ~180 dispatches per step -- so a trace
is worth more to it than any device-time change. Stages 1 and 2 made a step capturable: fixed shapes
and stable addresses (``ctx_capacity``), then staged inputs so nothing is built or uploaded mid-step.
This checks the capture is CORRECT, which is a separate claim from it being fast.

ORDERING IS DELIBERATE: the traced generation runs FIRST. An earlier equivalence test in this repo
ran the eager reference first and passed for the wrong reason -- the eager run left model state
valid, so the traced path's failure to re-establish that state was invisible until it was measured
end to end. Running traced-first means any state the capture forgets to set is state nobody set.

Greedy, so the two runs must emit BIT-IDENTICAL ids: a trace replays recorded ops against recorded
addresses, and if it is right at all it is right exactly. Acceptance must match too -- a drafter
whose replay is subtly wrong still produces valid tokens (the target verifies every slot) and shows
up only as fewer of them committed per step, which is precisely the failure a token check alone
would miss.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_drafter_trace.py
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
TRACE_REGION = 250_000_000  # must hold the verify trace AND the drafter trace
MAX_NEW_TOKENS = 64
CAPACITY = 512
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
def test_traced_drafter_matches_eager(mesh_device, device_params, reset_seeds, ensure_gc):
    """Traced drafter vs eager drafter: identical tokens, identical acceptance."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    from transformers import AutoTokenizer

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    state_dict = load_drafter_state_dict(drafter_path)

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids

    def run(traced):
        tt = TtDFlashDrafter(mesh_device, cfg, state_dict, tt_ccl=model.tt_ccl, ctx_capacity=CAPACITY)
        drafter = TtDrafter(tt, target)
        if traced:
            drafter.enable_traced_draft(q_len=cfg.block_size, ctx_pad=16)
        # WARM BEFORE TIMING. This test runs the traced arm FIRST on purpose (see the module
        # docstring), which is right for correctness and wrong for timing: the first generation
        # pays JIT compilation for programs BOTH arms share, so whichever runs first is charged for
        # the other. Measured that way the trace looked like 0.85x while a warmed per-phase
        # breakdown put a whole traced step at 75 ms, which cannot be slower than ~174 eager
        # dispatches. One short throwaway generation per arm removes the bias.
        dflash_generate(drafter, target, prompt, max_new_tokens=8)
        t0 = time.perf_counter()
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=MAX_NEW_TOKENS, return_stats=True)
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        logger.info(
            f"[{'TRACED' if traced else 'eager '} drafter] {n} tokens in {dt:.2f}s = {n / dt:.2f} tok/s "
            f"({dt * 1000 / n:.0f} ms/tok), acceptance {stats.mean_acceptance_length:.3f} over "
            f"{len(stats.acceptance_lengths)} steps"
        )
        if traced:
            drafter.release_draft_trace()
        return stats, n / dt

    # TRACED FIRST -- see the module docstring.
    traced_stats, traced_tps = run(True)
    eager_stats, eager_tps = run(False)

    logger.info("=" * 78)
    logger.info(f"  eager  drafter {eager_tps:6.2f} tok/s   acceptance {eager_stats.mean_acceptance_length:.3f}")
    logger.info(f"  TRACED drafter {traced_tps:6.2f} tok/s   acceptance {traced_stats.mean_acceptance_length:.3f}")
    logger.info(f"  {traced_tps / max(eager_tps, 1e-9):.2f}x")
    logger.info("=" * 78)
    print(
        f"\n>>> drafter trace: {eager_tps:.2f} -> {traced_tps:.2f} tok/s ({traced_tps / max(eager_tps, 1e-9):.2f}x)\n"
    )

    assert torch.equal(traced_stats.output_ids, eager_stats.output_ids), (
        "the traced drafter emitted different tokens than the eager one. Greedy speculation is "
        "exact, so this is a replay bug -- most likely an input the capture reads from a buffer "
        "that staging did not refill"
    )
    assert traced_stats.acceptance_lengths == eager_stats.acceptance_lengths, (
        f"same tokens but different acceptance ({traced_stats.mean_acceptance_length:.3f} vs "
        f"{eager_stats.mean_acceptance_length:.3f}): the replay is drafting differently, which the "
        "target then corrects. The tokens hide it; the step count does not"
    )
