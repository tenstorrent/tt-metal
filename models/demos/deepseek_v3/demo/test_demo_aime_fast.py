# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fast AIME-24 regression smoke test for the DeepSeek-V3 demo.

Runs a curated, 3-prompt subset of the lm-eval ``r1_aime24`` task that, on the
ds-rc1 baseline, produces correct boxed answers within 8K generated tokens. The
test scores generations with a self-contained boxed-answer matcher (so the
runtime venv does not need ``lm_eval``) and fails only on catastrophic
accuracy regressions, so it is safe to run on stochastic device sampling.

The subset is intentionally tiny so the test fits inside a single quad/dual
decode batch and completes in roughly the time of an 8K-token decode loop
(~10-15 minutes), which makes it suitable for the local "all needed" CI bundle.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import run_demo

MODEL_PATH = Path(
    os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked")
)
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))

PROMPTS_FILE = Path(__file__).resolve().parent / "aime_under_8k_prompts.json"

# Minimum number of correct generations required for the test to pass.
# The curated subset has 3 AIME-24 prompts that converged correctly within 8K
# tokens on the ds-rc1 baseline run. Stochastic sampling means we cannot
# guarantee all three pass on every run; require at least one as a
# regression-only smoke threshold.
DEFAULT_MIN_CORRECT = int(os.getenv("DEEPSEEK_AIME_FAST_MIN_CORRECT", "1"))


_SUPPORTED_TASKS = {"r1_aime24", "aime24"}
_BOXED_INT_RE = re.compile(r"\\boxed\s*\{\s*(-?\d+)\s*\}")


def _extract_boxed_int_answer(pred: str) -> str | None:
    """Return the last \\boxed{<int>} content as a normalized integer string, or None."""
    matches = _BOXED_INT_RE.findall(pred or "")
    if not matches:
        return None
    try:
        return str(int(matches[-1]))
    except ValueError:
        return None


def _aime_exact_match(doc: dict, pred: str) -> float:
    expected_raw = doc.get("answer")
    if expected_raw is None:
        return 0.0
    try:
        expected = str(int(str(expected_raw).strip()))
    except (TypeError, ValueError):
        expected = str(expected_raw).strip()
    predicted = _extract_boxed_int_answer(pred)
    return 1.0 if predicted is not None and predicted == expected else 0.0


def _load_prompts() -> list[dict]:
    if not PROMPTS_FILE.exists():
        pytest.fail(f"AIME fast eval prompts fixture missing: {PROMPTS_FILE}")
    with open(PROMPTS_FILE, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list) or not items:
        pytest.fail(f"AIME fast eval prompts fixture is empty or malformed: {PROMPTS_FILE}")
    for item in items:
        if "prompt" not in item or "doc" not in item or "task" not in item:
            pytest.fail(
                f"AIME fast eval prompt is missing required keys (prompt/doc/task): {item}"
            )
        if item["task"] not in _SUPPORTED_TASKS:
            pytest.fail(
                f"AIME fast eval prompt task '{item['task']}' is not supported by the inline "
                f"scorer. Supported tasks: {sorted(_SUPPORTED_TASKS)}."
            )
    return items


def _score_generations(prompt_items: list[dict], generations: list[dict]) -> list[dict]:
    scored: list[dict] = []
    for idx, (item, gen_result) in enumerate(zip(prompt_items, generations), start=1):
        pred = gen_result.get("text") or ""
        scored.append(
            {
                "index": idx,
                "task": item["task"],
                "exact_match": _aime_exact_match(item["doc"], pred),
                "answer": str(item["doc"].get("answer", "")),
                "predicted": _extract_boxed_int_answer(pred),
                "generated_tokens": len(gen_result.get("tokens", [])),
            }
        )
    return scored


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "max_users_per_row": 8,
                "max_new_tokens": 8192,
                "enable_trace": True,
                "sample_on_device": True,
            },
            id="quad_aime_under_8k_fast",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(2400)],
        ),
    ],
)
def test_demo_aime_under_8k_fast(case: dict):
    prompt_items = _load_prompts()
    prompts = [item["prompt"] for item in prompt_items]

    results = run_demo(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=case["max_new_tokens"],
        max_users_per_row=case["max_users_per_row"],
        repeat_batches=1,
        enable_trace=case["enable_trace"],
        sample_on_device=case["sample_on_device"],
        stop_at_eos=True,
        signpost=True,
    )

    generations = results.get("generations") or []
    assert len(generations) == len(prompts), (
        f"Expected {len(prompts)} generations, got {len(generations)}"
    )

    scored = _score_generations(prompt_items, generations)
    correct = sum(1 for s in scored if s["exact_match"] == 1.0)
    total = len(scored)

    summary_lines = [f"AIME-under-8K results: {correct}/{total} correct"]
    for s in scored:
        summary_lines.append(
            "  index={index} task={task} expected={answer} predicted={predicted} "
            "generated_tokens={generated_tokens} exact_match={exact_match}".format(**s)
        )
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    min_correct = max(DEFAULT_MIN_CORRECT, 0)
    assert correct >= min_correct, (
        f"AIME-under-8K regression: only {correct}/{total} correct, "
        f"expected >= {min_correct}. Details:\n{summary}"
    )
