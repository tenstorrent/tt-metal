# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Score an existing mp4 with extended no-reference VQA metrics: DOVER (VQA_A/VQA_T),
FAST-VQA, MaxVQA (16 antonym-pair dimensions, including clear-motion/blurry-motion),
Q-Align, and an optical-flow-based Flow Score. Standalone reporting only, not wired
into pytest and not gated -- run it by hand against an artifact from ~/h3_t2va_artifacts.

None of these metrics need a golden/reference video; each scores the mp4 alone (or,
for MaxVQA/Q-Align, alone via a fixed prompt template) -- see MiniMaxH3.md.

These tools do not fit `~/vbench_env`: DOVER/FAST-VQA/MaxVQA pin `torch~=1.13`
(numpy<2), while the pyiqa release with Q-Align needs `transformers~=5.x`, which
conflicts with VBench's `transformers==4.33.2` pin. Two more interpreters are needed:

    uv venv --python 3.10 ~/ext_vqa_legacy_env
    uv pip install --python ~/ext_vqa_legacy_env/bin/python \
        "torch==1.13.1" "torchvision==0.14.1" "numpy<2" decord opencv-python timm \
        einops scipy tqdm matplotlib scikit-video "thop==0.0.31-2005241907" onnx
    uv pip install --python ~/ext_vqa_legacy_env/bin/python -e ~/ext_vqa_repos/DOVER --no-deps
    uv pip install --python ~/ext_vqa_legacy_env/bin/python -r ~/ext_vqa_repos/FAST-VQA-and-FasterVQA/requirements.txt
    uv pip install --python ~/ext_vqa_legacy_env/bin/python -e ~/ext_vqa_repos/FAST-VQA-and-FasterVQA --no-deps
    uv pip install --python ~/ext_vqa_legacy_env/bin/python open_clip_torch --no-deps
    uv pip install --python ~/ext_vqa_legacy_env/bin/python "torch==1.13.1" "torchvision==0.14.1" "numpy<2" \
        ftfy regex huggingface_hub sentencepiece  # open_clip_torch drags torch 2.x back in; re-pin after

    uv venv --python 3.10 ~/ext_vqa_modern_env
    uv pip install --python ~/ext_vqa_modern_env/bin/python pyiqa

`~/ext_vqa_repos/{DOVER,FAST-VQA-and-FasterVQA,ExplainableVQA}` are `git clone`s of
VQAssessment/DOVER, timothyhtimothy/FAST-VQA-and-FasterVQA, VQAssessment/ExplainableVQA,
laid out side by side (MaxVQA's code loads DOVER's checkpoint via a `../DOVER` relative
path). Fetch DOVER's and FAST-VQA-B's pretrained weights per each repo's README; MaxVQA's
own `maxvqa_maxwell.pt` ships checked into the ExplainableVQA repo.

Usage:
    python ext_vqa_score.py --video <path.mp4> [--metrics dover,fastvqa,maxvqa,qalign,flow]
                             [--device cpu] [--output <file.json>]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LEGACY_PYTHON = Path.home() / "ext_vqa_legacy_env" / "bin" / "python"
MODERN_PYTHON = Path.home() / "ext_vqa_modern_env" / "bin" / "python"
LEGACY_METRICS = {"dover", "fastvqa", "maxvqa", "flow"}
MODERN_METRICS = {"qalign"}
TOOLS_DIR = Path(__file__).parent


def _run_worker(interpreter: Path, worker: str, video: Path, metrics: set[str], device: str) -> dict:
    if not interpreter.is_file():
        return {m: {"skipped": f"no interpreter at {interpreter}"} for m in metrics}
    result = subprocess.run(
        [
            str(interpreter),
            str(TOOLS_DIR / worker),
            "--video",
            str(video),
            "--metrics",
            ",".join(sorted(metrics)),
            "--device",
            device,
        ],
        capture_output=True,
        text=True,
        timeout=5400,
    )
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {
            m: {"error": f"worker produced no JSON (exit {result.returncode})", "stderr_tail": result.stderr[-2000:]}
            for m in metrics
        }


def score(video: Path, metrics: set[str], device: str) -> dict:
    results = {}
    if legacy := metrics & LEGACY_METRICS:
        results.update(_run_worker(LEGACY_PYTHON, "_ext_vqa_legacy_worker.py", video, legacy, device))
    if modern := metrics & MODERN_METRICS:
        results.update(_run_worker(MODERN_PYTHON, "_ext_vqa_modern_worker.py", video, modern, device))
    unknown = metrics - LEGACY_METRICS - MODERN_METRICS
    for m in unknown:
        results[m] = {"error": f"unknown metric '{m}'"}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--metrics", default="dover,fastvqa,maxvqa,qalign,flow")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    results = score(args.video.resolve(), metrics, args.device)

    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
