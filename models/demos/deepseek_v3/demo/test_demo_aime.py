# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""AIME-24 regression tests for the DeepSeek-V3 demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import _is_primary_artifact_writer, run_demo
from models.demos.deepseek_v3.demo.score_lmeval_outputs import (
    extract_aime_boxed_integer,
    normalize_aime_integer,
    score_aime24,
)

MODEL_PATH = Path(
    os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked")
)
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))

FAST_PROMPTS_FILE = Path(__file__).resolve().parent / "aime_under_8k_prompts.json"
FULL_PROMPTS_FILE = Path(os.getenv("DEEPSEEK_AIME24_PROMPTS_FILE", "/data/deepseek/prompts/demo_aime24_full.json"))
ARTIFACT_DIR = Path(os.getenv("DEEPSEEK_AIME24_ARTIFACT_DIR", "generated/artifacts"))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _load_prompt_items(path: Path) -> list[dict]:
    if not path.exists():
        pytest.fail(f"AIME prompt file missing: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list) or not items:
        pytest.fail(f"AIME prompt file is empty or malformed: {path}")
    for item in items:
        if not {"prompt", "doc", "task"}.issubset(item):
            pytest.fail(f"AIME prompt is missing required keys (prompt/doc/task): {item}")
        if item["task"] not in {"aime24", "r1_aime24"}:
            pytest.fail(f"Unsupported AIME task in {path}: {item['task']}")
    return items


def _score_generations(prompt_items: list[dict], generations: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for idx, (item, generation) in enumerate(zip(prompt_items, generations), start=1):
        text = generation.get("text") or ""
        result = score_aime24(item["doc"], text)
        predicted = extract_aime_boxed_integer(text)
        target = normalize_aime_integer(item["doc"].get("answer"))
        rows.append(
            {
                "index": idx,
                "aime_id": item["doc"].get("id"),
                "target": target,
                "predicted": predicted,
                "correct": bool(result["exact_match"]),
                "no_boxed": bool(result["no_boxed"]),
                "boxed_wrong": bool(result["boxed_wrong"]),
                "generated_tokens": len(generation.get("tokens", [])),
            }
        )

    summary = {
        "correct": sum(1 for row in rows if row["correct"]),
        "no_boxed": sum(1 for row in rows if row["no_boxed"]),
        "boxed_wrong": sum(1 for row in rows if row["boxed_wrong"]),
        "total": len(rows),
    }
    return rows, summary


def _print_and_write_summary(case_id: str, rows: list[dict], summary: dict) -> None:
    lines = [
        (
            f"AIME24 {case_id}: {summary['correct']}/{summary['total']} correct, "
            f"{summary['no_boxed']}/{summary['total']} no-boxed, "
            f"{summary['boxed_wrong']}/{summary['total']} boxed-wrong"
        )
    ]
    for row in rows:
        lines.append(
            "  index={index} aime_id={aime_id} target={target} predicted={predicted} "
            "tokens={generated_tokens} correct={correct} no_boxed={no_boxed}".format(**row)
        )
    print("\n" + "\n".join(lines))

    if _is_primary_artifact_writer():
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"case": case_id, "summary": summary, "rows": rows}
        (ARTIFACT_DIR / f"{case_id}_score.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            {
                "id": "quad_aime_fast",
                "prompts_file": FAST_PROMPTS_FILE,
                "max_new_tokens": 8192,
                "max_seq_len": None,
                "max_users_per_row": 8,
                "min_correct": _env_int("DEEPSEEK_AIME_FAST_MIN_CORRECT", 3),
            },
            id="quad_aime_fast",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(3600)],
        ),
        pytest.param(
            {
                "id": "quad_aime_full",
                "prompts_file": FULL_PROMPTS_FILE,
                "max_new_tokens": 32365,
                "max_seq_len": 32768,
                "max_users_per_row": 8,
                "min_correct": _env_int("DEEPSEEK_AIME_FULL_MIN_CORRECT", 24),
            },
            id="quad_aime_full",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(10800)],
        ),
    ],
)
def test_demo_aime24(case: dict):
    prompt_items = _load_prompt_items(Path(case["prompts_file"]))
    prompts = [item["prompt"] for item in prompt_items]

    run_kwargs = {
        "prompts": prompts,
        "model_path": MODEL_PATH,
        "cache_dir": CACHE_DIR,
        "random_weights": False,
        "max_new_tokens": case["max_new_tokens"],
        "max_users_per_row": case["max_users_per_row"],
        "repeat_batches": 1,
        "enable_trace": True,
        "sample_on_device": True,
        "stop_at_eos": True,
        "sampling_temperature": 0.6,
        "sampling_top_k": 32,
        "sampling_top_p": 0.95,
        "signpost": True,
    }
    if case["max_seq_len"] is not None:
        run_kwargs["max_seq_len"] = case["max_seq_len"]

    results = run_demo(**run_kwargs)
    generations = results.get("generations") or []
    assert len(generations) == len(prompts), f"Expected {len(prompts)} generations, got {len(generations)}"

    rows, summary = _score_generations(prompt_items, generations)
    _print_and_write_summary(case["id"], rows, summary)

    assert summary["correct"] >= case["min_correct"], (
        f"AIME24 {case['id']} regression: {summary['correct']}/{summary['total']} correct, "
        f"expected >= {case['min_correct']}; no_boxed={summary['no_boxed']}, "
        f"boxed_wrong={summary['boxed_wrong']}"
    )
