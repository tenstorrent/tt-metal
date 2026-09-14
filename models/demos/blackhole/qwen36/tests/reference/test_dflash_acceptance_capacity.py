# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What does the SDPA tile-padding dilution COST in acceptance? Priced in the only unit that matters.

ANSWER, MEASURED 2026-09-14 (full 27B on T3K, greedy, 3 prompts x 96 new tokens): **NOTHING.**

    variant                      pooled          per prompt              tokens
    growing history (shipped)    5.182 tok/step  5.278 / 7.917 / 3.800   identical
    fixed capacity 512           5.182 tok/step  5.278 / 7.917 / 3.800   identical

    delta +0.000 tok/step (+0.0 %)

Identical on EVERY prompt, not merely pooled. The dilution is real -- against the fp32 host oracle
the growing-history path measures 0.9798 where the fixed path measures 0.9940
(tests/unit/test_drafter_fixed_capacity.py) -- and it costs zero acceptance.

The reason is RMSNorm, and it is worth stating because it predicts where else this will and will not
matter. Tile-pad dilution is close to a PER-ROW SCALE FACTOR: a query row keeps ``r / (r + p)`` of
its magnitude. RMSNorm is per-row scale-INVARIANT, every sublayer output is added to a residual and
normed again, and the drafter's last op before the LM head is another RMSNorm. So the architecture
cancels most of the defect on its own, and argmax -- the only thing acceptance depends on -- never
sees it. PCC does see it, because PCC compares raw magnitudes.

The consequence for planning: fixing the dilution is NOT a throughput lever, and the fixed-capacity
KV scheme should be judged purely on what it was built for, which is making the drafter traceable.

tests/unit/test_sdpa_tile_padding.py established that ttnn SDPA attends a key sequence's tile
padding when the logical length is not a multiple of 32: the additive mask is itself padded with
ZEROS, and zero means visible. The drafter's key length is ``hist_len + new_ctx + q_len`` -- 16, 23,
39, 42, 53 in a normal run -- so it has been diluting its attention on nearly every step, worst
where the causal mask is tightest (the first query row of a 16-key step keeps 6 % of its magnitude).

The fixed-capacity path (``ctx_capacity=C``, built for tracing) is immune by construction: ``C +
ctx_pad + q_len`` is always tile-aligned and every non-real column carries an explicit mask entry.
Against the fp32 host oracle it is strictly closer wherever padding exists and IDENTICAL where none
does (tests/unit/test_drafter_fixed_capacity.py).

PCC is not the unit anyone cares about, though. A drafter cannot make the output wrong -- greedy
speculation lets the target verify every slot, so both variants here must emit BIT-IDENTICAL tokens,
which is asserted. What a weaker drafter costs is ACCEPTANCE LENGTH, tokens committed per target
forward, and since a step costs one target forward either way, acceptance scales throughput
directly. So the question this answers is: was the drafter leaving tokens/step on the table?

Both drafters share one target load and one state_dict, differing only in ``ctx_capacity`` -- the
same single-variable design as test_dflash_acceptance.py's dtype comparison, and for the same
reason: the 27B dominates the runtime and building it twice would price the comparison against
run-to-run variance instead of against the change.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_acceptance_capacity.py
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
CAPACITY = 512  # >= prompt + MAX_NEW_TOKENS for every prompt below

PROMPTS = [
    "The capital of France is",
    "Here are the first ten prime numbers, in order: 2, 3, 5,",
    "Explain, in your own words, why speculative decoding makes a language model faster:",
]

#: (label, ctx_capacity). None is the shipped growing-history path, i.e. the diluted one.
VARIANTS = [("growing history (shipped)", None), (f"fixed capacity {CAPACITY}", CAPACITY)]


def _page_table():
    return torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)


@pytest.mark.timeout(0)
@torch.no_grad()
@parametrize_mesh_tp()
def test_acceptance_per_kv_scheme(mesh_device, reset_seeds, ensure_gc):
    """Acceptance length for each KV scheme, plus bit-identical outputs across both."""
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B target")

    from transformers import AutoTokenizer

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    state_dict = load_drafter_state_dict(drafter_path)

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    target = TtTarget(model, cfg.target_layer_ids, _page_table(), device_taps=True)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompts = [tokenizer(p, return_tensors="pt").input_ids for p in PROMPTS]

    results: dict[str, dict] = {}
    for label, cap in VARIANTS:
        tt = TtDFlashDrafter(mesh_device, cfg, state_dict, tt_ccl=model.tt_ccl, ctx_capacity=cap)
        drafter = TtDrafter(tt, target)
        per_prompt, ids_out = [], []
        for name, ids in zip(PROMPTS, prompts):
            stats = dflash_generate(drafter, target, ids, max_new_tokens=MAX_NEW_TOKENS, return_stats=True)
            per_prompt.append(stats)
            ids_out.append(stats.output_ids)
            logger.info(
                f"[{label}] {name[:42]!r:46} {stats.mean_acceptance_length:.3f} tok/step over "
                f"{len(stats.acceptance_lengths):2d} steps, {stats.num_rollbacks} rollbacks"
            )
        # Pooled over prompts, not a mean of means: a step is a step regardless of which prompt.
        steps = sum(len(s.acceptance_lengths) for s in per_prompt)
        tokens = sum(sum(s.acceptance_lengths) for s in per_prompt)
        results[label] = {"pooled": tokens / max(steps, 1), "steps": steps, "ids": ids_out}
        logger.info(f"[{label}] POOLED {tokens / max(steps, 1):.3f} tok/step over {steps} steps")

    base, new = (results[v[0]] for v in VARIANTS)

    # Greedy speculation is exact: the target verifies every slot and accepts only its own argmax,
    # so a better OR worse drafter changes tokens/step and nothing else. Different ids here would
    # mean a broken accept path, not a better drafter.
    for i, (a, b) in enumerate(zip(base["ids"], new["ids"])):
        assert torch.equal(a, b), f"prompt {i}: the two KV schemes emitted different tokens"

    delta = new["pooled"] - base["pooled"]
    logger.info("=" * 78)
    for label, r in results.items():
        logger.info(f"  {label:28} {r['pooled']:.3f} tok/step over {r['steps']:2d} steps")
    logger.info(f"  delta {delta:+.3f} tok/step ({100 * delta / base['pooled']:+.1f} %), tokens identical")
    logger.info("=" * 78)
    print(f"\n>>> growing {base['pooled']:.3f} -> fixed {new['pooled']:.3f} tok/step ({delta:+.3f})\n")
