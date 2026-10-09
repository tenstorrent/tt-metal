# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the snapshot ``inference.py`` examples on one Tenstorrent device and print JSON.

Text (default): two questions about one support message, a yes/no "urgency" decision and a
three-way team "routing" choice. Image (``--image PATH``): the ``inference.py --image`` example,
"What is the dominant color?" (red / green / blue / other) about a local image. Output has the
same shape as the reference app's ``Decider.predict``.

Usage::

    python models/demos/pplx_decider_v1_27b/demo/demo.py
    # the inference.py image example on a local image
    python models/demos/pplx_decider_v1_27b/demo/demo.py --image photo.png
    # an image row of the stage-12 HF golden (its PNG and question), compared with the HF answer
    python models/demos/pplx_decider_v1_27b/demo/demo.py --golden-row v02_count_circles
    # optional: compare the text example with the HF bf16 answers (reference/hf_demo_reference.py)
    python models/demos/pplx_decider_v1_27b/demo/demo.py --compare-hf <demo_hf_reference.json>
    # your own decision (add --image for an image question)
    python models/demos/pplx_decider_v1_27b/demo/demo.py --state "..." --question '{"type": "noul", "instructions": "..."}'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import (
    DEMO_STATE,
    IMAGE_STATE,
    TTDecider,
    demo_questions,
    image_question,
)

IMAGE_GOLDEN = Path("/local/ttuser/gtobar/artifacts/pplx_decider/goldens/vision/e2e")


def same_decision(answer: dict, ref: dict) -> tuple[bool, float]:
    if answer["type"] == "noul":
        return (answer["noul"] > 0.5) == (ref["noul"] > 0.5), abs(answer["noul"] - ref["noul"])
    keys = answer["probabilities"]
    diff = max(abs(keys[k] - ref["probabilities"][k]) for k in keys)
    same = max(keys, key=keys.get) == max(ref["probabilities"], key=ref["probabilities"].get)
    return same, diff


def golden_image_row(row_id: str, golden_dir: Path) -> tuple[dict, dict]:
    rows = {r["id"]: r for r in map(json.loads, (golden_dir / "prompts.jsonl").read_text().splitlines())}
    if row_id not in rows:
        raise SystemExit(f"--golden-row must be one of {sorted(rows)}")
    summary = json.loads((golden_dir / "summary_bf16.json").read_text())
    return rows[row_id]["row"], next(r for r in summary["rows"] if r["id"] == row_id)["answer"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help=f"default: {DEMO_STATE!r} (text) / {IMAGE_STATE!r} (--image)")
    parser.add_argument("--question", type=json.loads, help="one question as JSON (default: the demo questions)")
    parser.add_argument("--image", type=Path, help="run the inference.py image example on a local image")
    parser.add_argument("--golden-row", help="run one row of the stage-12 HF image golden and compare")
    parser.add_argument("--golden-dir", type=Path, default=IMAGE_GOLDEN)
    parser.add_argument("--compare-hf", type=Path, help="HF answers JSON from reference/hf_demo_reference.py")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    if args.image is not None and not args.image.is_file():
        parser.error("--image must point to an existing image file")

    hf_reference = None
    if args.golden_row:
        row, hf_answer = golden_image_row(args.golden_row, args.golden_dir)
        state, images, questions = row["state"], row["images"], {args.golden_row: row["question"]}
        hf_reference = {args.golden_row: hf_answer}
    elif args.image is not None:
        state, images = args.state or IMAGE_STATE, [str(args.image)]
        questions = {"question": args.question} if args.question else {"dominant_color": image_question()}
    else:
        state, images = args.state or DEMO_STATE, []
        questions = {"question": args.question} if args.question else demo_questions()
        if args.compare_hf:
            hf_reference = {k: v["answer"] for k, v in json.loads(args.compare_hf.read_text())["results"].items()}

    device = ttnn.open_device(device_id=args.device_id, l1_small_size=24576)
    try:
        start = time.perf_counter()
        decider = TTDecider.from_pretrained(device, vision=bool(images))
        load_s = time.perf_counter() - start
        results, timings = {}, {}
        for name, question in questions.items():
            start = time.perf_counter()
            results[name] = decider.predict(state, question, images=images)
            timings[name] = time.perf_counter() - start
        print(json.dumps(results, indent=2))
        print(
            f"# model load {load_s:.1f} s (vision tower {'on' if images else 'off'}); first predict per question "
            "(includes program compile): " + ", ".join(f"{k} {v:.2f} s" for k, v in timings.items())
        )
        for name, answer in results.items():
            if hf_reference and name in hf_reference:
                same, diff = same_decision(answer, hf_reference[name])
                print(f"# vs HF bf16: {name}: same decision {same}, max |prob diff| {diff:.4f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
