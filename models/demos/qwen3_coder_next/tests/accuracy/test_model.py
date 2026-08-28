# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Correctness gate for `Qwen/Qwen3-Coder-Next` — the anti-cheat guard for optimizers.

    python_env/bin/python -m pytest models/demos/qwen3_coder_next/tests/accuracy/test_model.py -s

WHY THIS EXISTS SEPARATELY FROM tests/e2e/
An optimizer needs a gate that fails on NUMERICAL damage and on nothing else. The e2e
gate also asserts native-ttnn stub sources, exact TP shard widths, exact per-module
invocation counts and zero host aten ops. Those are the right assertions for bring-up
and the WRONG ones for optimization: fusing two modules, changing a shard width or
adding a host-side prep step can all be legitimate speedups that leave PCC untouched,
yet every one of them trips a structural assertion. Pointing an optimizer at the e2e
gate therefore rejects real wins while claiming to check correctness.

So this file asserts exactly one thing: the pipeline's logits still track the
HuggingFace reference. It is deliberately blind to HOW the answer was computed.

It prints `PCC: <float>` on its own line because that is the form the surrounding
tooling parses to decide whether an edit is verified; a gate that emits no PCC has
every edit rejected as unverified.

Depth, capacity and prompt follow the same env knobs as the rest of the package
(TT_QWEN3_LAYERS, TT_QWEN3_CAPACITY, TT_QWEN3_MAX_NEW_TOKENS, TT_QWEN3_PROMPT), so the
gate measures the same build the optimizer profiles.
"""
from __future__ import annotations

import os

from models.demos.qwen3_coder_next.tt.pipeline import DEFAULT_PROMPT, _pcc

# The threshold. Declared as a module-level constant because discovery reads it
# statically to confirm this file really is a PCC gate.
PCC_GATE = float(os.environ.get("TT_QWEN3_PCC_GATE", 0.95))

MAX_NEW_TOKENS = int(os.environ.get("TT_QWEN3_MAX_NEW_TOKENS", 8))


def test_model_accuracy(pipeline):
    """The pipeline's logits must still track the HF reference along one trajectory."""
    tokenizer = pipeline._tokenizer
    prompt = os.environ.get("TT_QWEN3_PROMPT", DEFAULT_PROMPT)

    result = pipeline.run_text_generation(
        tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS, collect_logits=True
    )

    # Score the TT trajectory under the reference. `_hf_score_sequence` conditions the
    # reference on the tokens TT actually produced, so a single early divergence cannot
    # cascade into a meaningless comparison the way free-running generation can.
    golden_steps = pipeline._hf_score_sequence(result["prompt_ids"], result["tokens"])
    steps = len(result["tokens"])
    assert steps > 0, "the pipeline generated nothing"

    per_step = [_pcc(golden_steps[i], result["logits"][i]) for i in range(steps)]
    achieved = min(per_step)  # the worst step, not the mean: one bad step is damage

    print()
    print(f"PROMPT     : {prompt}")
    print(f"TT output  : {result['text']!r}")
    for i, value in enumerate(per_step):
        print(f"  step {i}: PCC={value:.6f}")
    print(f"PCC: {achieved:.6f}")
    print(f"layers={pipeline.depth} steps={steps} gate={PCC_GATE}")

    assert achieved >= PCC_GATE, f"accuracy gate FAILED: worst-step PCC {achieved} < {PCC_GATE}"
