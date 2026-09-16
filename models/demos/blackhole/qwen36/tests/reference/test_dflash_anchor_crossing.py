# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does acceptance survive crossing the ANCHOR boundary?

Every HEALTHY number in this work comes from a run that never crosses the 128-token anchor, and
every SICK one comes from a run that does:

    test_dflash_traced_throughput.py   5 + 64  = 69   never crosses   acceptance 7.000
    demo spec_128                    128 + 100 = 228  crosses          acceptance 1.138

That correlation has never been tested directly, because the two differ in prompt content and length
as well. This file removes both: the prompt is the SAME five tokens in every arm, and only
``max_new_tokens`` varies, so the single difference between arms is whether ``start`` walks past
``TtTarget.ANCHOR``.

Crossing is not a no-op in this design. ``TtTarget.forward`` runs whole buckets in a loop --

    while end - self._anchor > self.ANCHOR:
        lg, tp = self._run(self._anchor, self._anchor + self.ANCHOR)
        self._anchor += self.ANCHOR
        self._anchor_gdn = self.model.save_gdn_state(into=self._anchor_gdn)

-- so crossing re-anchors the GDN snapshot mid-generation, and a whole bucket is ``length ==
ANCHOR``, which ``_run`` deliberately sends down the EAGER fallback rather than the trace
(``gdn/tp.py::_normalize_valid_len`` turns ``valid_len >= T`` into None, compiling different
programs than the capture holds). So a crossing run mixes traced and eager verifies and rewrites the
anchor snapshot, and a non-crossing run does neither.

WHAT THIS RULES IN OR OUT. Hypotheses already tested and killed, so they are not retried here:

* shared ``model.tt_ccl`` with the drafter -- no effect on acceptance;
* ``reset()`` allocating a fresh GDN snapshot per generation -- passing ``into=`` did NOT restore
  acceptance (3.875) and SIGBUSed the drafter on the next generation; that allocation is
  load-bearing because it follows ``_reset_gdn_state_for_new_sequence()``;
* generation index / parity -- acceptance alternates 7.000 / 1.500 / 7.000 with traced generation
  number in the non-crossing configuration (test_dflash_warmup_position.py), but the demo sits at
  1.138 on BOTH parities, so parity is not what ails it.

The demo's 1.138 is invariant under every one of those, which is why the anchor is what is left.

READ THE PER-STEP LIST, NOT JUST THE MEAN. Each arm logs the accepted length of every step next to
the absolute ``start`` it ran at. If acceptance is healthy up to the step that crosses 128 and
collapses at or after it, the boundary is the cause and the mean would have hidden it.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_anchor_crossing.py
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
PROMPT = "The capital of France is"

# The prompt is 5 tokens, so `end = 5 + max_new_tokens` and ANCHOR is 128:
#   64  -> end  69   no crossing   (this is exactly test_dflash_traced_throughput.py's control)
#  110  -> end 115   no crossing   (long, but still inside the first bucket)
#  130  -> end 135   ONE crossing
#  200  -> end 205   ONE crossing, then a long tail past it
BUDGETS = [64, 110, 130, 200]


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("max_new_tokens", BUDGETS, ids=lambda n: f"gen{n}")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_acceptance_across_the_anchor(mesh_device, device_params, max_new_tokens, reset_seeds, ensure_gc):
    """Same prompt, longer span. Only the anchor crossing differs between arms."""
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
    n_in = prompt.shape[1]

    # One eager generation, at THIS arm's budget, so every shape the measured run needs is compiled
    # before the trace is parked -- otherwise the first novel width hangs the process.
    dflash_generate(drafter, target, prompt, max_new_tokens=max_new_tokens)
    target.enable_traced_verify()

    t0 = time.perf_counter()
    stats = dflash_generate(drafter, target, prompt, max_new_tokens=max_new_tokens, return_stats=True)
    dt = time.perf_counter() - t0

    # Walk the per-step accepted lengths back into absolute positions, and mark the step that
    # crosses the anchor. `start` begins at the prompt length and advances by each step's commit.
    anchor = TtTarget.ANCHOR
    start, rows, crossed_at = n_in, [], None
    for i, produced in enumerate(stats.acceptance_lengths):
        crosses = (start + produced) > anchor >= start
        if crosses and crossed_at is None:
            crossed_at = i
        rows.append((i, start, produced, crosses))
        start += produced

    n = stats.num_output_tokens
    logger.info(
        f">>>>> gen{max_new_tokens}: {n} tok in {dt:.2f}s = {n / dt:5.2f} tok/s, "
        f"acceptance {stats.mean_acceptance_length:.3f} over {len(rows)} steps, "
        f"{'CROSSES at step ' + str(crossed_at) if crossed_at is not None else 'no crossing'}"
    )
    logger.info("      step  start  accepted")
    for i, s, p, crosses in rows:
        logger.info(f"      {i:4d}  {s:5d}  {p:8d}{'   <-- crosses ANCHOR' if crosses else ''}")

    # Mean acceptance before vs after the boundary is the number this file exists to produce.
    if crossed_at is not None:
        before = [p for _, _, p, _ in rows[:crossed_at]]
        after = [p for _, _, p, _ in rows[crossed_at:]]
        mb = sum(before) / max(len(before), 1)
        ma = sum(after) / max(len(after), 1)
        logger.info(f"      acceptance before the anchor {mb:.3f} ({len(before)} steps), after {ma:.3f} ({len(after)})")
        print(
            f"\n>>> gen{max_new_tokens}: before anchor {mb:.3f}, after {ma:.3f}, overall "
            f"{stats.mean_acceptance_length:.3f}"
        )
    else:
        print(f"\n>>> gen{max_new_tokens}: no crossing, acceptance {stats.mean_acceptance_length:.3f}")

    text = tokenizer.decode(stats.output_ids[0, n_in:], skip_special_tokens=True)
    print(f">>> output: {text[:160]!r}\n")

    # Greedy decoding pins the tokens, so every arm must still start the same way. This is also the
    # gate _assert_output_quality does NOT provide: it only detects repetition, and the corruption
    # this path produces is multilingual token soup, which is not repetitive.
    assert text.startswith(" Paris."), f"gen{max_new_tokens} emitted {text[:80]!r}"
    # Non-ASCII is the soup's fingerprint. The reference continuation for this prompt is English
    # prose and a <think> block; a healthy run has essentially none.
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    logger.info(f"      non-ascii chars in output: {non_ascii} / {len(text)}")
