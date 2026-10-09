# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream OpenBench GPQA, one bounded epoch with explicit differences."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

REVISION = "d03922ccbbd9004e802c556ef9c2a40ed870659c"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main(args):
    os.environ["HF_HOME"] = str(args.root / "hf")
    os.environ["HF_HUB_OFFLINE"] = "0" if args.prepare else "1"
    os.environ["HF_DATASETS_OFFLINE"] = "0" if args.prepare else "1"
    from inspect_ai import eval as evaluate
    from inspect_ai.model import get_model
    from openbench.evals.gpqa_diamond import gpqa_diamond

    revision = subprocess.check_output(
        ["git", "-C", str(args.root / "harness"), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("OpenBench source revision changed")
    task = gpqa_diamond()
    samples = [dict(input=s.input, target=s.target, id=s.id) for s in task.dataset]
    if len(samples) != 198:
        raise ValueError("GPQA Diamond must contain 198 questions")
    protocol_path = args.root / "protocol.json"
    if args.prepare:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        data_revision = HfApi().dataset_info("nmayorga7/gpqa_diamond").sha
        data = load_dataset("nmayorga7/gpqa_diamond", split="train", revision=data_revision)
        columns = ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]

        def canonical(rows):
            return sorted(digest({k: row[k] for k in columns}) for row in rows)

        with args.csv.open(newline="") as stream:
            local = list(csv.DictReader(stream))
        if canonical(local) != canonical(data):
            raise ValueError("OpenBench mirror differs from the pinned original GPQA Diamond records")
        protocol = dict(
            harness_revision=revision,
            dataset_revision=data_revision,
            dataset_records=198,
            dataset_matches_original=True,
            samples_sha256=digest(samples),
            original_csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest(),
            upstream_temperature=0.5,
            upstream_epochs=10,
            epochs=1,
            max_tokens=65536,
            max_connections=128,
            max_samples=128,
            differences=[
                "One epoch instead of ten to fit the eight-hour queue",
                "Explicit 65536-token output bound",
                "Local endpoint; OpenRouter private overrides are unknown",
            ],
            prompt_and_scorer="Unmodified pinned OpenBench",
            score_selection="Single predeclared run; no retries or best-of selection",
        )
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
        print(json.dumps(protocol), flush=True)
        return
    protocol = json.loads(protocol_path.read_text())
    if protocol["samples_sha256"] != digest(samples):
        raise ValueError("Prepared OpenBench task changed")
    output = args.root / "evaluation"
    output.mkdir(exist_ok=False)
    model = get_model(
        "openai-api/local/Qwen/Qwen3.8-27B",
        base_url="http://127.0.0.1:8078/v1",
        api_key="local-unused",
        responses_api=False,
    )
    logs = evaluate(
        task,
        model=model,
        epochs=1,
        max_tokens=65536,
        max_connections=128,
        max_samples=128,
        max_retries=0,
        retry_on_error=0,
        fail_on_error=False,
        timeout=7000,
        log_dir=str(output),
        display="plain",
        metadata=protocol,
    )
    log = logs[0]
    samples = log.samples or []
    stop_reasons = Counter(choice.stop_reason for sample in samples for choice in sample.output.choices)
    summary = dict(
        status=log.status,
        samples=len(samples),
        sample_errors=sum(s.error is not None for s in samples),
        stop_reasons=dict(stop_reasons),
        truncated=stop_reasons.get("max_tokens", 0),
        results=log.results.model_dump(mode="json") if log.results else None,
        protocol=protocol,
        log_location=log.location,
        complete=log.status == "success" and len(samples) == 198 and not any(s.error for s in samples),
    )
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary["complete"]:
        raise RuntimeError("OpenBench did not complete all 198 samples cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    main(parser.parse_args())
