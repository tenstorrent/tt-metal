# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice HF reference demo.

Invokes reference/run_inference.py (PyTorch baseline) as a subprocess.
This file stays outside tt/ — no TTNN ops here.

Usage:
    python models/experimental/vibevoice/demo_hf.py \
        --text "Hello, world!" \
        --voice resources/voices/en-Alice_woman.wav \
        --output_dir /tmp/vv_out
"""

import argparse
import subprocess
import sys
from pathlib import Path

from models.experimental.vibevoice.common.config import (
    DEFAULT_TXT_PATH,
    MODEL_PATH,
    REFERENCE_DIR,
    VOICES_DIR,
)
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.common.resource_utils import ensure_demo_resources


def main():
    parser = argparse.ArgumentParser(description="VibeVoice HF reference inference")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to synthesize. If omitted, reads from --text_file.",
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=str(DEFAULT_TXT_PATH),
        help="Path to text file.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=str(VOICES_DIR / "en-Alice_woman.wav"),
        help="Reference voice WAV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/tmp/vv_hf_out",
        help="Output directory for generated WAV files.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to VibeVoice-1.5B checkpoint.",
    )
    args = parser.parse_args()

    try:
        ensure_demo_resources()
        args.model_path = str(ensure_model_weights(args.model_path))
    except Exception as exc:
        print(f"[demo_hf] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    run_inference_py = REFERENCE_DIR / "run_inference.py"
    if not run_inference_py.exists():
        print(f"[demo_hf] ERROR: reference script not found at {run_inference_py}", file=sys.stderr)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(run_inference_py),
        "--model_path",
        args.model_path,
        "--voice",
        args.voice,
        "--output_dir",
        args.output_dir,
    ]
    if args.text:
        cmd += ["--text", args.text]
    else:
        cmd += ["--text_file", args.text_file]

    print(f"[demo_hf] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[demo_hf] Reference inference exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[demo_hf] Output written to: {args.output_dir}")


if __name__ == "__main__":
    main()
