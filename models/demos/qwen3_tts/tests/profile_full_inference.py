# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Profile the full Qwen3-TTS inference run with tracy.

Wrapper around demo_full_ttnn_tts.run_full_ttnn_tts that hardcodes args so we
don't have to pass strings-with-spaces through tracy's arg parser.

Usage:
    python -m tracy -p -v -r --op-support-count 20000 --dump-device-data-mid-run \
        models/demos/qwen3_tts/tests/profile_full_inference.py

Captures: speaker encoder + ICL + Talker prefill + Talker decode trace + CP
prefill trace + 14×CP decode traces × N generated frames. Use the companion
parser to split the resulting CSV into per-sub-block sheets.
"""
import sys
from pathlib import Path

# Default to short generation so the tracy CSV stays manageable.
NUM_FRAMES = 4

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import run_full_ttnn_tts  # noqa: E402


def main():
    demo_dir = REPO_ROOT / "models" / "demos" / "qwen3_tts" / "demo"
    run_full_ttnn_tts(
        text="Hello there.",
        ref_audio=str(demo_dir / "jim_reference.wav"),
        ref_text="Jason, can we take a look at the review slides",
        output_path="/tmp/profile_full_inference.wav",
        max_new_tokens=NUM_FRAMES,
        device_id=0,
        seed=42,
        use_2cq=False,
    )


if __name__ == "__main__":
    main()
