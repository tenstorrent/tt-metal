# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the snapshot ``inference.py`` text example on one Tenstorrent device and print JSON.

The example asks two questions about one support message: a yes/no "urgency" decision and a
three-way team "routing" choice. Output has the same shape as the reference app's
``Decider.predict``.

Usage::

    python models/demos/pplx_decider_v1_27b/demo/demo.py
    # optional: compare with the HF bf16 answers (reference/hf_demo_reference.py)
    python models/demos/pplx_decider_v1_27b/demo/demo.py --compare-hf <demo_hf_reference.json>
    # your own decision
    python models/demos/pplx_decider_v1_27b/demo/demo.py --state "..." --question '{"type": "noul", "instructions": "..."}'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import DEMO_STATE, TTDecider, demo_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default=DEMO_STATE)
    parser.add_argument("--question", type=json.loads, help="one question as JSON (default: the two demo questions)")
    parser.add_argument("--compare-hf", type=Path, help="HF answers JSON from reference/hf_demo_reference.py")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    questions = {"question": args.question} if args.question else demo_questions()
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        start = time.perf_counter()
        decider = TTDecider.from_pretrained(device)
        load_s = time.perf_counter() - start
        results, timings = {}, {}
        for name, question in questions.items():
            start = time.perf_counter()
            results[name] = decider.predict(args.state, question)
            timings[name] = time.perf_counter() - start
        print(json.dumps(results, indent=2))
        print(
            f"# model load {load_s:.1f} s; first predict per question (includes program compile): "
            + ", ".join(f"{k} {v:.2f} s" for k, v in timings.items())
        )
        if args.compare_hf:
            hf = json.loads(args.compare_hf.read_text())["results"]
            for name, answer in results.items():
                if name not in hf:
                    continue
                ref = hf[name]["answer"]
                if answer["type"] == "noul":
                    diff, same = abs(answer["noul"] - ref["noul"]), (answer["noul"] > 0.5) == (ref["noul"] > 0.5)
                else:
                    keys = answer["probabilities"]
                    diff = max(abs(keys[k] - ref["probabilities"][k]) for k in keys)
                    same = answer.get("choice", answer.get("score")) == ref.get("choice", ref.get("score"))
                    if answer["type"] == "score":
                        same = max(keys, key=keys.get) == max(ref["probabilities"], key=ref["probabilities"].get)
                print(f"# vs HF bf16: {name}: same decision {same}, max |prob diff| {diff:.4f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
