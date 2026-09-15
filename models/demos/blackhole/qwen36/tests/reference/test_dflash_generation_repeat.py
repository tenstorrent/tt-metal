# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does acceptance DECAY across repeated generations on a reused drafter/target?

Greedy verification fixes the emitted tokens: the target accepts only its own argmax, so every
generation from the same prompt MUST emit the same text. Acceptance is not similarly pinned -- it
is how many of those tokens the drafter guessed right -- so the same prompt producing the same text
at a DIFFERENT acceptance means the drafter entered the run with different state. That is a leak,
and it contaminates every A/B in this suite that reuses a drafter, which is most of them.

WHY THIS FILE EXISTS (measured 2026-09-15, T3K, this machine):

    test_dflash_traced_throughput.py   "The capital of France is"   acceptance 7.000
    test_dflash_prose_throughput.py    "The capital of France is"   acceptance 2.611

Same prompt, same greedy loop, same traced verify, and both emit the identical continuation
(' Paris.\\n\\n<think>\\nHere's a thinking process:...'). The measured difference between them is
WHERE IN THE SEQUENCE the measured run falls:

    traced_throughput:  eager gen(64) -> capture -> [MEASURED traced gen(64)]      2nd generation
    prose_throughput:   eager gen(8)  -> capture ->  traced gen(8) -> [MEASURED]   3rd generation

Hypotheses already eliminated, so they are not retried here:

* the drafter's TT_CCL instance -- prose passes `tt_ccl=model.tt_ccl` and traced_throughput lets the
  drafter build its own. Sharing puts the drafter's tap all-gather on the same round-robin
  semaphore pool as the target's traced collectives, which looked structural. Giving the drafter a
  fresh TT_CCL leaves the control at 2.611, unchanged to three decimals. Not it.
* the drafter's block width on the final step -- see test_drafter_block_width.py. Widths 16/8/4/3/2/1
  all pass standalone, including the exact hung state (new_ctx=1, q_len=2, start=110, 109 rows of
  history) in 0.20 s. Not it.

So this isolates generation INDEX with everything else held still: one drafter, one target, one
capture, the same prompt and the same token budget, N times in a row. If acceptance is flat, the
leak is not in repetition and the prose test's control differs for some other reason. If it decays,
the decay curve names the leak's rate and every reused-drafter measurement in this suite needs
re-reading.

Both arms run so the capture is not a confound:

* ``eager``  -- N generations, no trace. Isolates the loop's own state handling.
* ``traced`` -- one eager warm-up, capture, then N generations. The shipping configuration.

MEASURED 2026-09-15 (T3K, NEW_TOKENS=32, this prompt):

    eager   gen 1-5   6.200  6.200  6.200  6.200  6.200     flat to three decimals
    traced  gen 1-3   4.429  3.875  1.409                   collapsing

The eager arm is EXACTLY stable over five generations, so the leak is not in ``dflash_generate``'s
reset / snapshot / restore bookkeeping -- that path reuses a drafter five times with no drift at
all. It appears only once ``enable_traced_verify()`` has run, and it compounds per generation.

The most likely mechanism is already recorded as a landmine but was never connected to this:
``dflash_generate`` calls ``target.reset()`` at the top of EVERY generation, and ``reset()``
reallocates the anchor's GDN snapshot, which Metal warns about as *"Allocating device buffers is
potentially unsafe due to the existence of an active trace"*. The first traced generation is fine
because the capture has just happened; each later one re-does that unsafe reallocation underneath a
parked trace. (That warning does appear in these runs' logs.) NOT YET PROVEN -- it is the next thing
to test, by making ``reset()`` a no-op for the buffers the trace baked in, or by releasing and
re-capturing the trace per generation and checking whether the decay disappears.

THE DECAY TRIGGERS THE HANG, which is why these two bugs were entangled. At acceptance 1.409 the
accept counts are nothing like a healthy run's, so ``new_ctx`` takes values a warm-up never
produced, and ``kv_seq = new_ctx + q_len`` becomes a shape that was never compiled. Compiling it
with a trace parked hangs, exactly as in test_drafter_block_width.py's header. This file hung on its
own traced arm at ``layer_idx=3, start=27, new_ctx=1, q_len=10, kv_seq=11`` -- a novel width reached
only because acceptance had already collapsed. So a run of this test may hang before it can assert;
that hang is a RESULT, not a flake, and it needs `tt-smi -r` afterwards.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_generation_repeat.py
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
TRACE_REGION = 250_000_000
# 32 keeps `start` well below 127, away from the verify_size==1 bug at start % ANCHOR == 127, so a
# short run cannot be mistaken for the decay this is looking for.
NEW_TOKENS = 32
N_REPEATS = 5
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
def test_acceptance_across_repeats(mesh_device, device_params, reset_seeds, ensure_gc):
    """Same prompt, same budget, N times. Acceptance should be constant; the text certainly must be."""
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
    # The drafter builds its own TT_CCL: sharing the model's was tested and changes nothing, and an
    # unshared one is what the known-good test_dflash_traced_throughput.py uses.
    drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(drafter_path)), target)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids

    def run(tag):
        t0 = time.perf_counter()
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=NEW_TOKENS, return_stats=True)
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
        logger.info(
            f">>>>> {tag}: {n} tok in {dt:.2f}s = {n / dt:5.2f} tok/s, "
            f"acceptance {stats.mean_acceptance_length:.3f} over {len(stats.acceptance_lengths)} steps, "
            f"rollbacks {stats.num_rollbacks}"
        )
        return stats.mean_acceptance_length, text, stats.acceptance_lengths

    # ---- arm 1: eager, no capture ----
    eager = [run(f"eager gen {i + 1}") for i in range(N_REPEATS)]

    # ---- arm 2: traced. One eager generation already happened above, which is the precondition
    # enable_traced_verify() asserts, so the capture is legal here.
    target.enable_traced_verify()
    traced = [run(f"traced gen {i + 1}") for i in range(N_REPEATS)]

    def report(label, rows):
        accs = [a for a, _, _ in rows]
        logger.info(f"=== {label}: " + ", ".join(f"{a:.3f}" for a in accs) + " ===")
        logger.info(f"    per-step: " + " | ".join(str(l) for _, _, l in rows))
        return accs

    eager_accs = report("eager acceptance by generation", eager)
    traced_accs = report("traced acceptance by generation", traced)
    print(f"\n>>> eager  {['%.3f' % a for a in eager_accs]}")
    print(f">>> traced {['%.3f' % a for a in traced_accs]}\n")

    # The text is the hard invariant: greedy decoding from a fixed prompt cannot legally vary. If
    # THIS trips, the problem is worse than an acceptance leak -- the emitted tokens are wrong.
    for label, rows in (("eager", eager), ("traced", traced)):
        for i, (_, text, _) in enumerate(rows):
            assert text == rows[0][1], (
                f"{label} generation {i + 1} emitted DIFFERENT TEXT from generation 1 -- greedy "
                f"decoding from a fixed prompt must be deterministic.\n  gen1: {rows[0][1]!r}\n"
                f"  gen{i + 1}: {text!r}"
            )

    # Acceptance is the actual subject. Reported either way; asserted so a decay cannot pass quietly.
    for label, accs in (("eager", eager_accs), ("traced", traced_accs)):
        assert max(accs) - min(accs) < 0.5, (
            f"{label} acceptance is not stable across repeats: {['%.3f' % a for a in accs]}. "
            "Identical text at differing acceptance means the drafter entered these runs with "
            "different state -- a leak across generations on a reused drafter."
        )
