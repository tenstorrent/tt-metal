# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Audit saved GPQA outputs without exporting text or changing their scores."""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def repetition_metrics(text):
    # Exact duplication is a diagnostic, not a semantic reasoning-quality test.
    words = text.split()[-4096:]
    grams = Counter(tuple(words[i : i + 20]) for i in range(max(0, len(words) - 19)))
    paragraphs = Counter(p.strip() for p in text.split("\n\n") if len(p.strip()) > 80)
    return {
        "tail_words": len(words),
        "tail_duplicate_20gram_fraction": 1 - len(grams) / sum(grams.values()) if grams else 0.0,
        "max_tail_20gram_count": max(grams.values(), default=0),
        "max_paragraph_repeats": max(paragraphs.values(), default=0),
    }


def audit(receipts, raw_directory, *, count, max_output_tokens, max_model_len):
    if count < 1 or not 0 < max_output_tokens < max_model_len:
        raise ValueError("Expected a nonempty selection and output budget below the model context limit")
    receipt_bytes = receipts.read_bytes()
    public = [json.loads(line) for line in receipt_bytes.splitlines() if line.strip()]
    if sorted(r["id"] for r in public) != list(range(count)):
        raise ValueError("Receipts must contain every selected question exactly once")
    expected_files = {f"{i:03d}.json" for i in range(count)}
    if {p.name for p in raw_directory.glob("*.json")} != expected_files:
        raise ValueError("Private responses must match the complete selected set")
    rows = []
    for p in sorted(public, key=lambda r: r["id"]):
        raw_bytes = (raw_directory / f"{p['id']:03d}.json").read_bytes()
        raw = json.loads(raw_bytes)
        if not isinstance(p["correct"], int) or p["correct"] not in (0, 1):
            raise ValueError(f"Invalid recorded score for row {p['id']}")
        for field, digest in (("text", "final_answer_sha256"), ("reasoning", "reasoning_sha256")):
            if sha256(raw[field].encode()) != p[digest]:
                raise ValueError(f"Private response hash differs from the scored receipt for row {p['id']}")
        if raw["finish_reason"] != p["finish_reason"]:
            raise ValueError(f"Finish reason differs from the scored receipt for row {p['id']}")
        for field in ("prompt_tokens", "completion_tokens"):
            if raw["usage"][field] != p["usage"][field]:
                raise ValueError(f"Token usage differs from the scored receipt for row {p['id']}")
        prompt, output = [raw["usage"][field] for field in ("prompt_tokens", "completion_tokens")]
        if not all(isinstance(n, int) and n >= 0 for n in (prompt, output)):
            raise ValueError(f"Invalid token counts for row {p['id']}")
        if output > max_output_tokens or prompt + output > max_model_len:
            raise ValueError(f"Token counts exceed the declared configuration for row {p['id']}")
        finish = raw["finish_reason"]
        # A length finish alone does not identify which bound stopped generation.
        limit = "not_length_limited"
        if finish == "length":
            at_output = output == max_output_tokens
            at_context = prompt + output == max_model_len
            limit = (
                "both"
                if at_output and at_context
                else "output_budget"
                if at_output
                else "model_context"
                if at_context
                else "unresolved"
            )
        rows.append(
            {
                "id": p["id"],
                "correct": p["correct"],
                "finish_reason": finish if finish in ("stop", "length") else "other",
                "limit": limit,
                "prompt_tokens": prompt,
                "completion_tokens": output,
                "remaining_model_context_tokens": max_model_len - prompt - output,
                "final_chars": len(raw["text"]),
                "reasoning_chars": len(raw["reasoning"]),
                "raw_sha256": sha256(raw_bytes),
                **repetition_metrics(raw["reasoning"] + "\n" + raw["text"]),
            }
        )

    def group(selected):
        return {
            "count": len(selected),
            "correct": sum(r["correct"] for r in selected),
            "empty_final": sum(r["final_chars"] == 0 for r in selected),
            "median_output_tokens": statistics.median(r["completion_tokens"] for r in selected) if selected else None,
            "median_tail_duplicate_20gram_fraction": (
                statistics.median(r["tail_duplicate_20gram_fraction"] for r in selected) if selected else None
            ),
            "max_tail_duplicate_20gram_fraction": max(
                (r["tail_duplicate_20gram_fraction"] for r in selected), default=None
            ),
        }

    stopped = [r for r in rows if r["finish_reason"] == "stop"]
    return {
        "schema_version": 1,
        "receipt_sha256": sha256(receipt_bytes),
        "selected_count": count,
        "max_output_tokens": max_output_tokens,
        "max_model_len": max_model_len,
        "all_private_responses_match_scored_receipts": True,
        "full_score": {"correct": sum(r["correct"] for r in rows), "total": count},
        "limits": dict(Counter(r["limit"] for r in rows)),
        "groups": {
            "length_limited": group([r for r in rows if r["finish_reason"] == "length"]),
            "natural_stop": group(stopped),
            "wrong_natural_stop": group([r for r in stopped if not r["correct"]]),
            "correct": group([r for r in rows if r["correct"]]),
        },
        "interpretation": (
            "All selected rows remain in the full-score denominator. Subset metrics are diagnostics, not an "
            "alternative benchmark score. Lack of exact repetition does not establish correct reasoning. "
            "Declared limits must be checked against the original launch configuration."
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--private-responses", type=Path, required=True)
    parser.add_argument("--count", type=int, default=198)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.receipts,
        args.private_responses,
        count=args.count,
        max_output_tokens=args.max_output_tokens,
        max_model_len=args.max_model_len,
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
