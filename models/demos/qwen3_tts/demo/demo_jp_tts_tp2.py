# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Japanese Qwen3-TTS: 1.7B and 0.6B on N300 TP=2.

Requires an N300 (or 2-chip mesh). Set MESH_DEVICE=N300.

    MESH_DEVICE=N300 PYTHONPATH=$PWD python models/demos/qwen3_tts/demo/demo_jp_tts_tp2.py

Writes:
    qwen3-tts-1.7B-tp2.wav
    qwen3-tts-0.6B-tp2.wav
"""

import argparse
import os
from pathlib import Path

from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
    _load_ref_text_for,
    get_default_reference_path,
    run_full_ttnn_tts,
)

MODELS = (
    ("1.7B", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "qwen3-tts-1.7B-tp2.wav"),
    ("0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "qwen3-tts-0.6B-tp2.wav"),
)


def _load_default_text() -> str:
    p = Path(__file__).with_name("sample_ja.txt")
    return p.read_text(encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description="Japanese Qwen3-TTS 1.7B and 0.6B, N300 TP=2")
    parser.add_argument("--text", type=str, default=None, help="Japanese text (default: sample_ja.txt)")
    parser.add_argument("--output-dir", type=str, default="/tmp", help="Directory for wavs")
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MESH_DEVICE", "N300")
    text = args.text if args.text else _load_default_text()
    ref_audio = args.ref_audio if args.ref_audio else get_default_reference_path()
    ref_text = args.ref_text if args.ref_text else _load_ref_text_for(ref_audio)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, hf_id, wav_name in MODELS:
        output_path = str(out_dir / wav_name)
        print("\n" + "#" * 80)
        print(f"# {label}  {hf_id}  →  {output_path}")
        print("#" * 80)
        result = run_full_ttnn_tts(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            output_path=output_path,
            language="japanese",
            seed=args.seed,
            device_id=args.device_id,
            hf_id=hf_id,
            use_2cq=True,
        )
        rows.append((label, result))

    print("\n" + "=" * 80)
    print("1.7B vs 0.6B  (N300 TP=2)")
    print("=" * 80)
    print(f"{'Model':<8} {'Prefill':>10} {'Steady AR':>14} {'fps':>8} {'RTF':>8}  wav")
    print("-" * 80)
    for label, r in rows:
        print(
            f"{label:<8} {r['prefill_ms']:>8.1f} ms {r['steady_ms_per_frame']:>8.1f} ms/fr "
            f"{r['steady_frames_per_sec']:>8.2f} {r['rtf']:>8.2f}  {r['output_wav']}"
        )
    print("=" * 80)
    print("RTF vs 80 ms/frame (12 Hz). <1 is faster than real-time.")
    for label, r in rows:
        print(f"  {label}: {r['output_wav']}")


if __name__ == "__main__":
    main()
