# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-CPU completion checks for performance evidence; no accuracy gate is inferred."""

import math
import re

from models.demos.llama_3p1_8b_d_p.tests.performance.book_observation import (
    METADATA_KEYS,
    OBSERVATION_SCOPE,
    final_position,
    validate_prompts,
)
from models.demos.llama_3p1_8b_d_p.tests.performance.long_context_performance_helpers import (
    aggregate,
    chunk_plan,
    request_schedule,
)


def report_reasons(report, expected_tokens):
    reasons = []
    plan = chunk_plan(expected_tokens)
    expected = dict(
        schema_version=4,
        run_mode="execution_and_performance",
        execution_verified=False,
        status="performance_complete_no_accuracy_claim",
        full_model_accepted=False,
        execution_scope="error_free_full32_prefill",
        cache_dtype="bfloat8_b",
        weights_dtype="bfloat16",
        activations_dtype="bfloat16",
        num_layers=32,
        mesh=[4, 8],
        slots=2,
        prompt_tokens=expected_tokens,
        chunk_tokens=1024,
        include_final_norm_lm_head=True,
        traced=False,
    )
    if any(report.get(k) != v or type(report.get(k)) is not type(v) for k, v in expected.items()):
        reasons.append("Performance scope/precision/geometry mismatch")
    try:
        if any(
            type(sample["tokens"]) is not int or sample["tokens"] != expected_tokens for sample in report["samples"]
        ):
            reasons.append("Timing sample context differs from requested context")
        if aggregate(report["samples"]) != report["summary"]:
            reasons.append("Summary differs from raw sample clocks")
        for previous, current in zip(report["samples"], report["samples"][1:]):
            if previous["clock_timestamps"]["prompt_end"] > current["clock_timestamps"]["prompt_start"]:
                reasons.append("Full request timing intervals overlap or are out of order")
                break
        validate_prompts(report["prompts"], expected_tokens)
        validate_observations(report, expected_tokens)
        checks = report["output_checks"]
        expected_keys = {
            (s["slot"], s["phase"], s["repetition"], start, chip)
            for s in report["samples"]
            for start, _ in plan
            for chip in range(32)
        }
        keys = [(r["slot"], r["phase"], r["repetition"], r["start"], r["chip"]) for r in checks]
        if len(keys) != len(expected_keys) or len(set(keys)) != len(expected_keys) or set(keys) != expected_keys:
            reasons.append("Missing/duplicate all32 logits evidence")
        baseline = {(r["slot"], r["start"], r["chip"]): r["sha256"] for r in checks if r["phase"] == "warmup"}
        for r in checks:
            if (
                r["finite"] is not True
                or r["repeat_equal"] is not True
                or r["shape"] != [1, 1, 256, 16032]
                or r["elements"] != 4104192
                or r["end"] != r["start"] + 1024
                or type(r.get("max_abs")) not in (int, float)
                or not math.isfinite(r["max_abs"])
                or r["max_abs"] < 0
                or r["sp"] != r["chip"] // 8
                or r["tp"] != r["chip"] % 8
                or not re.fullmatch("[0-9a-f]{64}", r["sha256"])
                or r["sha256"] != baseline[(r["slot"], r["start"], r["chip"])]
            ):
                reasons.append("Logits finite/shape/repeat digest failure")
                break
        if len(report["prompts"]) != 2 or {p["slot"] for p in report["prompts"]} != {0, 1}:
            reasons.append("Two distinct slot prompts required")
        prompts = sorted(report["prompts"], key=lambda p: p["slot"])
        if (
            any(len(p["token_ids"]) != expected_tokens for p in prompts)
            or prompts[0]["token_ids"] == prompts[1]["token_ids"]
        ):
            reasons.append("Distinct exact-length prompts required")
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        reasons.append("Malformed performance evidence: " + str(error))
    return reasons


def capture_reasons(directory, pins, sha):
    import json

    reasons = []
    for label in ("before", "after"):
        if json.loads((directory / (label + "-source-pins.json")).read_text()) != pins:
            reasons.append(label + " source pins differ")
    return reasons


def validate_observations(report, tokens):
    schedule = request_schedule()
    observations = report["next_token_observations"]
    keys = [(r["slot"], r["phase"], r["repetition"]) for r in observations]
    if keys != schedule:
        raise ValueError("Exactly one ordered final-token observation per request is required")
    position = final_position(tokens)
    baseline = {}
    for r in observations:
        expected = dict(
            final_prompt_position=position["position"],
            chunk_start=position["chunk_start"],
            sp=position["sp"],
            local_row=position["local_row"],
            tp_order=list(range(8)),
            vocab_size=128256,
            scope=OBSERVATION_SCOPE,
        )
        if any(r.get(k) != v or type(r.get(k)) is not type(v) for k, v in expected.items()):
            raise ValueError("Final-token row/TP/vocabulary mapping differs")
        meta = report["prompts"][r["slot"]]["metadata"]
        if any(r[k] != meta[k] for k in METADATA_KEYS):
            raise ValueError("Book observation metadata differs from bound prompt")
        top = r["top5"]
        if len(top) != 5 or len({x["token_id"] for x in top}) != 5:
            raise ValueError("Exactly five distinct vocabulary predictions are required")
        for x in top:
            if type(x["token_id"]) is not int or not 0 <= x["token_id"] < 128256 or not isinstance(x["piece"], str):
                raise ValueError("Invalid decoded vocabulary prediction")
            if type(x["logit"]) not in (int, float) or not math.isfinite(x["logit"]):
                raise ValueError("Prediction logit must be finite")
            if (
                type(x["probability"]) not in (int, float)
                or not math.isfinite(x["probability"])
                or not 0 <= x["probability"] <= 1
            ):
                raise ValueError("Invalid predicted-token probability")
        if top != sorted(top, key=lambda x: (-x["logit"], x["token_id"])) or r["argmax_token_id"] != top[0]["token_id"]:
            raise ValueError("Top-five order or argmax differs")
        if sum(x["probability"] for x in top) > 1 + 1e-12 or any(
            a["probability"] < b["probability"] for a, b in zip(top, top[1:])
        ):
            raise ValueError("Top-five probabilities are inconsistent")
        if type(r["expected_next_token_rank"]) is not int or not 1 <= r["expected_next_token_rank"] <= 128256:
            raise ValueError("Continuation rank outside vocabulary")
        probability = r["expected_next_token_probability"]
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Continuation probability must be finite")
        if not re.fullmatch("[0-9a-f]{64}", r["final_logits_float32_sha256"]):
            raise ValueError("Final full-vocabulary digest is required")
        record = {k: v for k, v in r.items() if k not in ("phase", "repetition")}
        if r["phase"] == "warmup":
            baseline[r["slot"]] = record
        elif record != baseline[r["slot"]]:
            raise ValueError("Final-token observation changed from same-slot warmup")
    readbacks = report["readback_hash_wall_seconds"]
    if [(r["slot"], r["phase"], r["repetition"]) for r in readbacks] != schedule:
        raise ValueError("Outside-timer readback costs must cover every request")
    if any(
        type(r["seconds"]) not in (int, float) or not math.isfinite(r["seconds"]) or r["seconds"] <= 0
        for r in readbacks
    ):
        raise ValueError("Readback costs must be finite positive seconds")
