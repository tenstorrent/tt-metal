# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process CPU tokenizer timings; no model, TTNN, or device initialization."""

import argparse
import json
import os
import time
from pathlib import Path

# Match tt/model.py without importing its accelerator dependencies.
MODEL_ID = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promptsfile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.promptsfile.read_text())
    if source.get("model") != MODEL_ID or source.get("revision") != REVISION:
        parser.error("Prompt artifact must use the pinned model and revision")
    prompts = source["prompts"]
    if len(prompts) != 10 or any(not isinstance(p.get("text"), str) or not p.get("token_ids") for p in prompts):
        parser.error("Expected exactly 10 text prompts with saved token_ids")

    # Set before importing libraries that initialize native CPU thread pools.
    # OMP does not govern the tokenizer's Rust/Rayon pool; record its settings
    # without changing them, so this tests the serving environment's defaults.
    os.environ["OMP_NUM_THREADS"] = "8"
    import tokenizers
    import transformers
    from transformers import AutoTokenizer

    start = time.perf_counter_ns()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    report = {
        "model": MODEL_ID,
        "revision": REVISION,
        "promptsfile": str(args.promptsfile.resolve()),
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "is_fast": tokenizer.is_fast,
        "threads": {
            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM")
        },
        "load_ms": (time.perf_counter_ns() - start) / 1e6,
        "protocol": "encode(add_special_tokens=True): prompt0 x10, prompts0..9 once, prompts0..9 again; no async microbatch wait",
        "calls": [],
    }

    def measure(phase, iteration, index):
        prompt = prompts[index]
        started = time.perf_counter_ns()
        ids = tokenizer.encode(prompt["text"], add_special_tokens=True)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        result = {
            "phase": phase,
            "iteration": iteration,
            "prompt_index": index,
            "ms": elapsed_ms,
            "characters": len(prompt["text"]),
            "utf8_bytes": len(prompt["text"].encode("utf-8")),
            "length": len(ids),
            "expected_length": len(prompt["token_ids"]),
            "matches_expected": ids == prompt["token_ids"],
        }
        if not result["matches_expected"]:
            result["actual_token_ids"] = ids
        report["calls"].append(result)

    for iteration in range(10):
        measure("warm_prompt0", iteration, 0)
    for phase in ("first_pass", "repeat_pass"):
        for index in range(10):
            measure(phase, index, index)
    report["passed"] = all(call["matches_expected"] for call in report["calls"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, separators=(",", ":"))
    args.output.write_text(serialized + "\n")
    print(serialized)
    if not report["passed"]:
        raise SystemExit("Token IDs differ from the saved prompt artifact; timings are not a matched-input control")


if __name__ == "__main__":
    main()
