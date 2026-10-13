# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-shot CLI: speak a sentence in one of the shipped voices.

    python -m models.experimental.voxtral_tts.demo.demo \
        "Hello from Tenstorrent." --voice neutral_male --out hello.wav --seed 0

`--voice` takes any preset name; `--list-voices` prints them. `--max-frames` caps the utterance at
80 ms per frame (by default it scales with the text; generation normally stops on [END_AUDIO]).
Checkpoint: `--ckpt` > $VOXTRAL_CKPT > download mistralai/Voxtral-4B-TTS-2603 (CC BY-NC 4.0).
"""

import argparse
import sys
import wave

from models.experimental.voxtral_tts import frontend
from models.experimental.voxtral_tts.reference.voxtral_paths import resolve_model_dir
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import FRAME_RATE, TtVoxtralPipeline

SAMPLE_RATE = 24000


def write_wav(path, wav):
    """wav [1,1,T] float in [-1,1] -> 16-bit PCM."""
    a = wav.reshape(-1).clamp(-1.0, 1.0).cpu().numpy()
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes((a * 32767.0).astype("<i2").tobytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="what to say")
    ap.add_argument("--voice", default="neutral_male")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None, help="80 ms of audio each")
    ap.add_argument("--ckpt", default=None, help="model directory (default: $VOXTRAL_CKPT, else HF hub)")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args(argv)

    model_dir = resolve_model_dir(a.ckpt)
    presets = frontend.voices(model_dir)
    if a.list_voices:
        print("\n".join(presets))
        return 0
    if not a.text:
        ap.error("give some text, or --list-voices")
    if a.voice not in presets:
        ap.error(f"unknown voice {a.voice!r}; --list-voices to see the {len(presets)} presets")

    pipe = TtVoxtralPipeline(ckpt_path=model_dir)
    try:
        # Every prefill shape, every codec bucket, one trace capture; verbose so the wait does
        # not look like a hang.
        pipe.warmup(verbose=True)
        wav = pipe.synthesize(a.text, a.voice, seed=a.seed, max_frames=a.max_frames)
        write_wav(a.out, wav)
        t = pipe.last_timings
        audio_s = t["frames"] / FRAME_RATE
        total = t["prefill_s"] + t["decode_s"] + t.get("codec_s", 0.0)
        print(
            f"{a.out}: {audio_s:.1f}s of audio in {total:.2f}s "
            f"({audio_s / max(total, 1e-9):.2f}x real time), "
            f"{t['frames']} frames at {t['decode_ms_per_frame']:.1f} ms/frame"
        )
    finally:
        pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
