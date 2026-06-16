# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice HF reference demo.

Invokes reference/run_inference.py (PyTorch baseline) as a subprocess.
This file stays outside tt/ — no TTNN ops here.

Usage:
    python models/experimental/vibevoice/demo_hf.py \
        --text_file resources/text/1p_vibevoice.txt \
        --voice resources/voices/en-Alice_woman.wav \
        --output_dir /tmp/vv_out \
        --seed 0
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from models.experimental.vibevoice.common.config import (
    DEFAULT_TXT_PATH,
    MODEL_PATH,
    REFERENCE_DIR,
    VOICES_DIR,
)
from models.experimental.vibevoice.common.model_utils import ensure_model_weights
from models.experimental.vibevoice.common.resource_utils import ensure_demo_resources, normalize_script


def main():
    parser = argparse.ArgumentParser(description="VibeVoice HF reference inference")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Inline script text. If omitted, reads from --text_file.",
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
    parser.add_argument("--seed", type=int, default=0, help="Diffusion RNG seed (default: 0)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--cfg_scale", type=float, default=1.3)
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

    txt_path = args.text_file
    temp_txt: Path | None = None
    if args.text:
        script = normalize_script(args.text)
        fd, temp_path = tempfile.mkstemp(suffix=".txt", text=True)
        os.close(fd)
        temp_txt = Path(temp_path)
        temp_txt.write_text(script + "\n", encoding="utf-8")
        txt_path = str(temp_txt)

    cmd = [
        sys.executable,
        str(run_inference_py),
        "--model_path",
        args.model_path,
        "--txt_path",
        txt_path,
        "--voice_path",
        args.voice,
        "--output_dir",
        args.output_dir,
        "--speaker_names",
        "Alice",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--cfg_scale",
        str(args.cfg_scale),
    ]

    print(f"[demo_hf] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
    finally:
        if temp_txt is not None:
            temp_txt.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"[demo_hf] Reference inference exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[demo_hf] Output written to: {args.output_dir}")


if __name__ == "__main__":
    main()
