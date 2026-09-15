# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does a 64-row anchor change the TOKENS? The PCC gate was a proxy; this is the real criterion.

ANSWER, MEASURED 2026-09-15: **the tokens are IDENTICAL, and it still does not pay.**

    ANCHOR=128   acceptance 7.000/step over  9 steps
    ANCHOR= 64   acceptance 6.300/step over 10 steps   tokens IDENTICAL

Two reversals in one measurement. The exactness rejection was WRONG -- pcc 0.9666 flipped no argmax
and the generated text is character-identical. But the lever still does not pay, for a different
reason: acceptance falls 10 %, because a drifted target rejects more drafts.

    current      7.0 / 315 ms  -> 22.2 tok/s
    ANCHOR=64    6.3 / ~280 ms -> ~22.5 tok/s   (~33-38 ms/step saved: row-proportional device
                                                 work, smaller staged masks, smaller logits source)

A wash. And the two effects are CAUSALLY COUPLED, which is the part worth remembering: shrinking the
bucket shrinks the context the anchor re-runs, that shortened context is exactly what increases the
drift (the bisection showed the error is worst where least context exists to dilute it), and the
drift is what costs acceptance. The device saving and the accuracy loss are the same phenomenon
measured twice. There is no version of this that takes one without the other.

ANCHOR=64 was rejected on exactness: pcc 0.9666 / 0.9988 / 0.9916 at chunk_start 64 / 128 / 192
against a one-shot prefill (tests/reference/test_dflash_anchor64.py). It is worth ~34 ms/step --
~43 % of the verify's device time is row-proportional and the bucket computes 128 rows to verify at
most 16 tokens -- so it is the last lever with real headroom.

THE REJECTION USED THE WRONG BAR. Chaining is slightly lossy at BOTH bucket sizes: the shipped
ANCHOR=128 path measures **0.9995**, not 1.0, against a one-shot prefill at start 128/130
(README-DFLASH.md). Its own test asserts > 0.99, not equality. And the component bisection
(tests/unit/test_bucket64_carry_bisect.py) found no broken component -- conv_carry, conv_states and
rec_state all drift together and amplify through the recurrent scan, worst when the least context
exists to dilute them. chunk_start=64 is simply the shortest-context case there is.

So the question is not "is it exact" -- nothing here is -- but "does it change what the model says".
Greedy speculation makes that precise: the target's own argmax decides every token, so a target
whose logits drift can emit DIFFERENT text. This runs the same prompt at both anchors and compares
the ids, and it also reports acceptance, because a drifting target rejects more drafts and would
show up as fewer tokens per step even when the text survives.

Both arms run EAGER verify: the trace serves one bucket size, and the question is about the anchor,
not the capture.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_anchor64_tokens.py
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
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
MAX_NEW_TOKENS = 64
PROMPT = "The capital of France is"


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
def test_anchor64_token_identity(mesh_device, device_params, monkeypatch, reset_seeds, ensure_gc):
    """Same prompt at ANCHOR 128 and 64: do the emitted tokens and the acceptance survive?"""
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
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids

    def run(anchor):
        monkeypatch.setattr(TtTarget, "ANCHOR", anchor)
        target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
        drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, state_dict, tt_ccl=model.tt_ccl), target)
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=MAX_NEW_TOKENS, return_stats=True)
        text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
        logger.info(
            f"[ANCHOR={anchor:3d}] acceptance {stats.mean_acceptance_length:.3f}/step over "
            f"{len(stats.acceptance_lengths)} steps"
        )
        logger.info(f"[ANCHOR={anchor:3d}] -> {text!r}")
        return stats, text

    s128, t128 = run(128)
    s64, t64 = run(64)

    same = torch.equal(s128.output_ids, s64.output_ids)
    n = min(s128.output_ids.shape[1], s64.output_ids.shape[1])
    first_diff = next(
        (i for i in range(n) if s128.output_ids[0, i] != s64.output_ids[0, i]),
        None,
    )
    logger.info("=" * 78)
    logger.info(f"  tokens identical: {same}")
    if not same:
        logger.info(f"  first differing position: {first_diff} (prompt is {s128.num_input_tokens} tokens)")
    logger.info(f"  acceptance  128: {s128.mean_acceptance_length:.3f}   64: {s64.mean_acceptance_length:.3f}")
    logger.info("=" * 78)
    print(
        f"\n>>> ANCHOR 64 vs 128: tokens {'IDENTICAL' if same else 'DIFFER'}, "
        f"acceptance {s128.mean_acceptance_length:.3f} -> {s64.mean_acceptance_length:.3f}\n"
    )

    # Diagnostic, not a gate. A difference does NOT by itself condemn ANCHOR=64: the shipped path is
    # 0.9995 against a one-shot prefill, not exact, so both anchors are approximations and neither
    # is the "true" continuation. What matters is whether the text stays good and acceptance holds.
    # Recording the answer is the point; the judgement is a human one.
    assert s64.mean_acceptance_length > 0, "ANCHOR=64 produced no accepted tokens at all"
