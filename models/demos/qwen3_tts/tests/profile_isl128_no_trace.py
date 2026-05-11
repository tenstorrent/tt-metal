# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile qwen3_tts full inference path at ISL ~128, NO trace, NO warmup-pad.

Same wrapper pattern as profile_full_inference.py but passes use_trace=False
so prefill + decode run untraced and we get raw per-op timings.

Usage:
    python -m tracy -p -v -r --op-support-count 20000 --dump-device-data-mid-run \
        models/demos/qwen3_tts/tests/profile_isl128_no_trace.py
"""
import sys
from pathlib import Path

# 4 decode steps keeps the captured CSV small while exercising the full path.
NUM_FRAMES = 2

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import run_full_ttnn_tts  # noqa: E402


def main():
    demo_dir = REPO_ROOT / "models" / "demos" / "qwen3_tts" / "demo"
    # Text crafted so the ICL embedding ends up around 128 tokens after
    # role + ref_text + ref_codes + target_text. Bucket get_padded_prefill_len
    # rounds to 128 for inputs in (64, 128].
    run_full_ttnn_tts(
        text="This is a test of the inference profiling path at sequence length one twenty eight tokens long.",
        ref_audio=str(demo_dir / "jim_reference.wav"),
        ref_text="Jason, can we take a look at the review slides",
        output_path="/tmp/profile_isl128_no_trace.wav",
        max_new_tokens=NUM_FRAMES,
        device_id=0,
        seed=42,
        use_2cq=False,
        use_trace=False,
    )


if __name__ == "__main__":
    main()
