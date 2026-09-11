# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ACCEPTANCE RATE of the device DFlash drafter — the drafter's real metric, priced per weight dtype.

WHY THIS EXISTS SEPARATELY FROM PCC
-----------------------------------
Every other drafter test measures PCC against a host reference. PCC is a proxy, and for a
*speculative* drafter it is the wrong one twice over:

1. **Quality does not depend on the drafter at all.** Greedy speculation is exact — the target
   verifies every drafted slot and accepts only its own argmax — so a worse drafter changes the
   output not at all. ``test_tt_drafter_end_to_end`` pins that against autoregressive decoding;
   this file adds the sharper form of the same claim: two drafters of DIFFERENT PRECISION must emit
   **bit-identical token ids**. If they ever do not, something in the accept path is broken, not the
   drafter.
2. **Speed does depend on the drafter, and PCC does not predict how much.** What a weaker drafter
   costs is *acceptance length* — tokens committed per target forward. That is the number a weight
   dtype trades against, and 0.9942-vs-0.9978 PCC says nothing quantitative about it.

So: MLP gate/up went bf4 for -17 % device time (see ``tt/dflash/weights.py`` ``MLP_DTYPE``), and the
question this answers is what that did to tokens/step. A speculative step costs one target forward
regardless of how many slots are accepted, so end-to-end speedup scales with acceptance length —
a 5 % acceptance loss would eat a third of the 17 %.

HOW
---
One target load, two drafters. The 27B is ~150 GB of checkpoint and dominates the runtime, so
building it twice would both waste minutes and expose the comparison to run-to-run variance; the
drafters differ only in the ``mlp_dtype`` handed to ``load_drafter_weights``, and the target,
prompts and RNG are shared. Greedy (``temperature=0``) so the whole thing is deterministic and the
per-step acceptance sequence is directly comparable.

MEASURED (full 27B on T3K, greedy, 3 prompts x 96 new tokens each)

    variant                  pooled          per prompt                    rollbacks
    all bf8                  5.089 tok/step  5.278 / 7.308 / 3.800         50 of 56 steps
    gate/up bf4 (shipped)    5.089 tok/step  5.000 / 7.917 / 3.800         50 of 56 steps

So the bf4 MLP costs **nothing measurable** in acceptance, and the output ids are bit-identical.

Read those numbers with two caveats:

* **Resolution.** Total emitted tokens is capped by ``max_new_tokens``, so the pooled mean is
  essentially ``tokens / steps`` and one step either way is a ~5 % swing on a single prompt. bf4
  needed one MORE step on prompt 1 and one FEWER on prompt 2; the pooled tie at 5.089 is partly
  that cancellation, not a guarantee of exact equality. The honest claim is "no loss detectable at
  56 steps", not "identical".
* **Greedy only.** The bit-identical-output claim holds *because* greedy accepts only the target's
  own argmax. Under sampling (``temperature > 0``) the rejection sampler reads the draft
  *probability* of each token, not just its rank, so a less precise drafter can move both the
  acceptance length and the realised output. This file does not cover that.

Also visible here and worth its own work: **50 of 56 steps take the rollback path** (partial
acceptance), and ``generate.py`` pays a second target forward on each of those. That is a far bigger
end-to-end lever than anything left inside the drafter, and it is independent of weight precision.

Run (needs the real 27B and the drafter checkpoint)::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_acceptance.py
"""

from __future__ import annotations

import os

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
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
MAX_NEW_TOKENS = 96

#: Prompts, chosen for a spread of predictability: a factual completion the drafter should nail, a
#: list continuation, and free-form prose where it should do worst. One prompt is not a measurement.
PROMPTS = [
    "The capital of France is",
    "Here are the first ten prime numbers, in order: 2, 3, 5,",
    "Explain, in your own words, why speculative decoding makes a language model faster:",
]

#: (label, kwargs to load_drafter_weights). bf8 first so it reads as baseline -> shipped.
VARIANTS = [
    ("all bf8", dict(mlp_dtype=ttnn.bfloat8_b, mlp_down_dtype=ttnn.bfloat8_b)),
    ("gate/up bf4 (shipped)", dict(mlp_dtype=ttnn.bfloat4_b, mlp_down_dtype=ttnn.bfloat8_b)),
]

#: How much acceptance loss is tolerable for the bf4 MLP. The dtype buys -17 % device time, and
#: since a step is one target forward either way, tokens/step scales the end-to-end win directly --
#: so a >5 % relative acceptance loss would mean the trade is not obviously worth it and wants a
#: human decision, which is exactly what this assert escalates.
MAX_ACCEPTANCE_LOSS = 0.05


def _page_table():
    return torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)


@pytest.mark.timeout(0)
@torch.no_grad()
@parametrize_mesh_tp()
def test_acceptance_rate_per_mlp_dtype(mesh_device, reset_seeds, ensure_gc):
    """Acceptance length for each MLP weight dtype, plus bit-identical outputs across both."""
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B target")

    from transformers import AutoTokenizer

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    state_dict = load_drafter_state_dict(drafter_path)

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    assert len(model.layers) == cfg.num_target_layers, "taps address real checkpoint layer indices"

    target = TtTarget(model, cfg.target_layer_ids, _page_table(), device_taps=True)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompts = [tokenizer(p, return_tensors="pt").input_ids for p in PROMPTS]

    results: dict[str, dict] = {}
    for label, dtypes in VARIANTS:
        tt = TtDFlashDrafter(mesh_device, cfg, state_dict, tt_ccl=model.tt_ccl, **dtypes)
        drafter = TtDrafter(tt, target)
        per_prompt = []
        for name, ids in zip(PROMPTS, prompts):
            stats = dflash_generate(drafter, target, ids, max_new_tokens=MAX_NEW_TOKENS, return_stats=True)
            per_prompt.append(stats)
            logger.info(
                f"[{label}] {name[:42]!r:46} {stats.mean_acceptance_length:.3f} tok/step over "
                f"{len(stats.acceptance_lengths):2d} steps, {stats.num_rollbacks} rollbacks"
            )
        # Pooled over prompts, not a mean of means: a step is a step regardless of which prompt.
        lengths = [n for s in per_prompt for n in s.acceptance_lengths]
        results[label] = {
            "mean": sum(lengths) / len(lengths),
            "steps": len(lengths),
            "outputs": [s.output_ids for s in per_prompt],
            "per_step": lengths,
        }
        logger.info(f"[{label}] POOLED {results[label]['mean']:.3f} tok/step over {len(lengths)} steps")

    base, ship = (results[VARIANTS[0][0]], results[VARIANTS[1][0]])
    loss = (base["mean"] - ship["mean"]) / base["mean"]
    logger.info(
        f"=== acceptance: bf8 {base['mean']:.3f} -> bf4 {ship['mean']:.3f} tok/step "
        f"({-loss * 100:+.1f} %) over {base['steps']}/{ship['steps']} steps ==="
    )

    # 1. Quality: greedy + a verifying target means precision CANNOT move the output.
    for name, a, b in zip(PROMPTS, base["outputs"], ship["outputs"]):
        assert torch.equal(a, b), (
            f"bf4 and bf8 drafters produced DIFFERENT tokens for {name!r} — greedy speculation is "
            "exact, so this is a bug in the accept/rollback path, not a precision effect:\n"
            f"  bf8 {a.tolist()}\n  bf4 {b.tolist()}"
        )

    # 2. Speed: acceptance is what the dtype actually trades.
    assert ship["mean"] > 1.0, f"bf4 drafter accepted {ship['mean']:.3f} tok/step — speculation bought nothing"
    assert loss <= MAX_ACCEPTANCE_LOSS, (
        f"bf4 MLP cost {loss * 100:.1f} % of acceptance length ({base['mean']:.3f} -> "
        f"{ship['mean']:.3f} tok/step), above the {MAX_ACCEPTANCE_LOSS * 100:.0f} % budget. The dtype "
        "buys -17 % device time; if acceptance drops by more than that is worth, put MLP_DTYPE back "
        "to PROJ_DTYPE in tt/dflash/weights.py."
    )
