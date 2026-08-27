# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One-shot CLI: speak a sentence in one of the shipped voices.

    python -m models.experimental.voxtral_tts.demo.demo \
        "Hello from Tenstorrent." --voice neutral_male --out hello.wav --seed 0

`--voice` takes any preset name; `--list-voices` prints them. A frame is 80 ms of audio, and
`--max-frames` caps the utterance (the default is generous -- generation normally stops itself on
[END_AUDIO]).
"""

import argparse
import sys
import wave

import numpy as np
import ttnn

from models.experimental.voxtral_tts import frontend
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (
    FRAME_RATE,
    TtVoxtralPipeline,
    open_device,
)

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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="what to say")
    ap.add_argument("--voice", default="neutral_male")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=600, help="80 ms of audio each")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args(argv)

    if a.list_voices:
        print("\n".join(frontend.voices()))
        return 0
    if not a.text:
        ap.error("give some text, or --list-voices")
    if a.voice not in frontend.voices():
        ap.error(f"unknown voice {a.voice!r}; --list-voices to see the {len(frontend.voices())} presets")

    device = open_device()
    try:
        pipe = TtVoxtralPipeline(device)
        pipe.warmup()
        embeds = frontend.build_prompt_embeds(a.text, a.voice, pipe.wb)
        frames, _, _ = pipe.generate(embeds, max_frames=a.max_frames, seed=a.seed, verbose=False)
        wav = pipe.decode(frames)
        write_wav(a.out, wav)
        t = pipe.last_timings
        audio_s = frames.shape[0] / FRAME_RATE
        total = t["prefill_s"] + t["decode_s"] + t.get("codec_s", 0.0)
        print(f"{a.out}: {audio_s:.1f}s of audio in {total:.2f}s "
              f"({audio_s / max(total, 1e-9):.2f}x real time), "
              f"{frames.shape[0]} frames at {t['decode_ms_per_frame']:.1f} ms/frame")
        pipe.close()
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
