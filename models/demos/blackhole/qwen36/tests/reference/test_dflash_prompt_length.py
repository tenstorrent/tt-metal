# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does DFlash still work with a REAL prompt? Every measurement in this work used five tokens.

MEASURED 2026-09-15, and it found a bug that fails this test at length 120 BY DESIGN:

    len   8   acceptance eager 2.300  traced 2.300  tokens SAME  text coherent
    len  64   acceptance eager 1.917  traced 2.091  tokens SAME  text coherent
    len 120   AssertionError: "drafter context is at 126 + 1 new rows but the block starts at 128"

THE BUG. ``TtTarget.max_block(start) = ANCHOR - (start % ANCHOR)`` returns 1 when
``start % 128 == 127``. generate.py then computes ``verify_size = 1`` and its ``if verify_size > 1``
guard SKIPS ``drafter.propose`` entirely -- so the drafter never receives that step's taps, its
``_ctx_len`` does not advance, and the next step trips the drafter's own invariant.

Nothing caught it because every test in this work used "The capital of France is": 5 tokens with
acceptance 7.000 puts ``start`` at 5+7k, and 5+7k = 127 has no integer solution. The benchmark
prompt's arithmetic never lands on the one position that triggers it.

THE OTHER FINDING, and the more consequential one: acceptance on ordinary prose is ~2.0-2.3 against
7.000 on the benchmark prompt. Throughput is acceptance / step_time and step time is near-constant,
so the "1.19x of production" headline in this work probably holds only for a prompt whose
continuation the drafter can nail. That is measured here for acceptance but NOT yet for throughput
(tests/reference/test_dflash_prose_throughput.py exists but its control disagrees with
test_dflash_traced_throughput.py by 4x on the control's own prompt, so its numbers are not
trustworthy yet).

Opt-in via DFLASH_PROBE_PROMPT_LEN=1: it fails until the verify_size==1 bug is fixed, at which
point it should be promoted to a normal test -- the token-identity assert it carries is the
invariant the suite only ever checked at length 5.

The demo (demo/dflash_demo.py) ran the pipeline on a 128-token chat prompt and produced garbage --
coherent for ~15 tokens, then multilingual token soup -- with acceptance 1.138 against the 7.000
that "The capital of France is" yields. Every throughput number, every acceptance figure and every
negative result in this work was measured on that five-token prompt, so if长 prompts are broken then
none of those conclusions generalise. That is what this establishes.

It bisects prompt length and, at each length, runs BOTH verify paths:

* eager and traced AGREE and both look wrong  -> the target is broken at that length; tracing is
  innocent and the bug predates every trace in this work.
* they DISAGREE                                -> the trace is implicated, and the existing
  token-identity gates missed it because they only ever ran at length 5.

Lengths straddle 128 deliberately. ``TtTarget.ANCHOR`` is 128, and a prompt of exactly 128 is the
one case that takes the whole-bucket EAGER fallback (``_run`` uses the trace only when
``length < ANCHOR``), so 120 / 128 / 136 separate "long prompt" from "prompt on the anchor
boundary".

Acceptance is the quantitative signal and the text is the qualitative one. Both are printed: the
repetition checks in text_demo's _assert_output_quality passed the garbage above, so coherence here
is a human judgement, not an assert.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_prompt_length.py
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

#: Opt-in: this REPRODUCES A KNOWN BUG and fails by design at length 120. See the module docstring.
pytestmark = pytest.mark.skipif(
    os.environ.get("DFLASH_PROBE_PROMPT_LEN") != "1",
    reason="reproduces the verify_size==1 drafter-context bug; set DFLASH_PROBE_PROMPT_LEN=1",
)

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 250_000_000
NEW_TOKENS = 24
#: 8 is near the five-token case everything was measured on; 120/128/136 straddle ANCHOR.
LENGTHS = (8, 64, 120, 128, 136)

#: Real prose, so a truncation at any length is still a well-formed prompt rather than a fragment of
#: chat markup. Long enough to cut 136 tokens from.
BASE_TEXT = (
    "The history of computing hardware spans more than a century, beginning with mechanical "
    "calculators and progressing through relays, vacuum tubes, discrete transistors, and finally "
    "integrated circuits. Each transition reduced the cost of a single logical operation by orders "
    "of magnitude, and each one was driven less by a single invention than by the accumulation of "
    "manufacturing technique. The earliest machines were built one at a time by hand, and their "
    "designers could describe every wire; modern processors contain billions of devices that no "
    "individual has examined. What changed was not only scale but the nature of the design problem, "
    "which became a question of managing complexity rather than of assembling components. "
    "Abstraction layers multiplied in response: register transfer descriptions replaced schematics, "
    "synthesis tools replaced hand placement, and verification grew into a discipline larger than "
    "design itself. A modern chip is specified, simulated and proven long before any silicon is "
    "committed, and the cost of a mistake discovered late is measured in months rather than in "
    "rework. That economics, more than any particular circuit technique, explains why the industry "
    "organised itself the way it did. "
)


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
def test_acceptance_and_text_vs_prompt_length(mesh_device, device_params, reset_seeds, ensure_gc):
    """Acceptance and generated text at several prompt lengths, eager verify and traced."""
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
    drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, state_dict, tt_ccl=model.tt_ccl), target)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    base_ids = tokenizer(BASE_TEXT, return_tensors="pt", add_special_tokens=False).input_ids
    assert base_ids.shape[1] >= max(LENGTHS), f"BASE_TEXT is only {base_ids.shape[1]} tokens"

    def run(prompt, traced):
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=NEW_TOKENS, return_stats=True)
        text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
        return stats, text

    rows = []
    for length in LENGTHS:
        prompt = base_ids[:, :length]
        eager_stats, eager_text = run(prompt, traced=False)
        # One capture serves every valid_len below the bucket, so enable it once and reuse.
        if getattr(target, "_traced_verify", False) is False:
            target.enable_traced_verify()
        traced_stats, traced_text = run(prompt, traced=True)

        same = torch.equal(eager_stats.output_ids, traced_stats.output_ids)
        rows.append((length, eager_stats.mean_acceptance_length, traced_stats.mean_acceptance_length, same))
        logger.info("-" * 78)
        logger.info(
            f"len {length:4d}  acceptance eager {eager_stats.mean_acceptance_length:.3f}  "
            f"traced {traced_stats.mean_acceptance_length:.3f}  tokens {'SAME' if same else 'DIFFER'}"
        )
        logger.info(f"len {length:4d}  eager  -> {eager_text[:220]!r}")
        logger.info(f"len {length:4d}  traced -> {traced_text[:220]!r}")

    logger.info("=" * 78)
    for length, ea, ta, same in rows:
        logger.info(f"  len {length:4d}  eager {ea:5.3f}  traced {ta:5.3f}  {'SAME' if same else 'DIFFER'}")
    logger.info("=" * 78)
    print("\n>>> " + " | ".join(f"len{l}: e{ea:.2f}/t{ta:.2f}{'' if s else ' DIFFER'}" for l, ea, ta, s in rows) + "\n")

    # The load-bearing invariant, and the one the existing suite only ever checked at length 5:
    # greedy speculation is exact, so the two verify paths must agree at EVERY prompt length.
    bad = [l for l, _, _, same in rows if not same]
    assert not bad, f"traced and eager verify emitted different tokens at prompt lengths {bad}"
