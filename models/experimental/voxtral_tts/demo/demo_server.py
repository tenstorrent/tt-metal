# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Interactive REPL: load and warm once, then one wav per typed line.

    python -m models.experimental.voxtral_tts.demo.demo_server --voice neutral_male

Commands: `\\voice NAME` switches preset, `\\voices` lists them, `\\seed N` fixes the seed,
`\\out PATH` sets the next output path, `\\quit` exits. Anything else is spoken.

Warm-up and the checkpoint load are paid once, so the second request onward is what the
Performance table in the README describes. `--ckpt` > $VOXTRAL_CKPT > HF hub download.
"""

import argparse
import os
import sys

from models.experimental.voxtral_tts import frontend
from models.experimental.voxtral_tts.demo.demo import write_wav
from models.experimental.voxtral_tts.reference.voxtral_paths import resolve_model_dir
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import FRAME_RATE, TtVoxtralPipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="neutral_male")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="model directory (default: $VOXTRAL_CKPT, else HF hub)")
    a = ap.parse_args(argv)

    model_dir = resolve_model_dir(a.ckpt)
    presets = frontend.voices(model_dir)
    if a.voice not in presets:
        ap.error(f"unknown voice {a.voice!r}; one of: {', '.join(presets)}")
    pipe = TtVoxtralPipeline(ckpt_path=model_dir)
    try:
        print("warming up: every prefill shape, every codec bucket, one trace capture ...", flush=True)
        pipe.warmup(verbose=True)
        voice, seed, out, n = a.voice, a.seed, "out.wav", 0
        print(f"ready. voice={voice} seed={seed}. \\quit to exit.", flush=True)
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in (r"\quit", r"\q"):
                break
            if line == r"\voices":
                print(", ".join(presets))
                continue
            if line.startswith(r"\voice "):
                cand = line.split(None, 1)[1].strip()
                if cand not in presets:
                    print(f"unknown voice {cand!r}")
                else:
                    voice = cand
                    print(f"voice={voice}")
                continue
            if line.startswith(r"\seed "):
                seed = int(line.split()[1])
                print(f"seed={seed}")
                continue
            if line.startswith(r"\out "):
                out, n = line.split(None, 1)[1].strip(), 0  # the next utterance goes to exactly this path
                print(f"out={out}")
                continue

            wav = pipe.synthesize(line, voice, seed=seed, max_frames=a.max_frames)
            root, ext = os.path.splitext(out)
            path = out if n == 0 else f"{root}_{n}{ext}"  # out.wav, out_1.wav, ...; any suffix, or none
            write_wav(path, wav)
            t = pipe.last_timings
            audio_s = t["frames"] / FRAME_RATE
            total = t["prefill_s"] + t["decode_s"] + t.get("codec_s", 0.0)
            print(f"  {path}  {audio_s:.1f}s in {total:.2f}s ({audio_s / max(total, 1e-9):.2f}x real time)")
            n += 1
    finally:
        pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
