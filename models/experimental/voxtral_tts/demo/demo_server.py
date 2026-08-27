# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Interactive REPL: load and warm once, then one wav per typed line.

    python -m models.experimental.voxtral_tts.demo.demo_server --voice neutral_male

Commands: `\\voice NAME` switches preset, `\\voices` lists them, `\\seed N` fixes the seed,
`\\out PATH` sets the next output path, `\\quit` exits. Anything else is spoken.

Warm-up and the checkpoint load are paid once, so the second request onward is what the
Performance table in the README describes.
"""

import argparse
import sys

import ttnn

from models.experimental.voxtral_tts import frontend
from models.experimental.voxtral_tts.demo.demo import write_wav
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import (
    FRAME_RATE,
    TtVoxtralPipeline,
    open_device,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="neutral_male")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=600)
    a = ap.parse_args(argv)

    device = open_device()
    try:
        pipe = TtVoxtralPipeline(device)
        print("warming up: every prefill shape, every codec bucket, one trace "
              "capture (~74 s) ...", flush=True)
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
                print(", ".join(frontend.voices()))
                continue
            if line.startswith(r"\voice "):
                cand = line.split(None, 1)[1].strip()
                if cand not in frontend.voices():
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
                out = line.split(None, 1)[1].strip()
                print(f"out={out}")
                continue

            # Each request is independent: reset so the previous utterance's KV cache cannot
            # influence this one (see tests/test_request_path_repeatability.py).
            pipe.backbone.reset()
            embeds = frontend.build_prompt_embeds(line, voice, pipe.wb)
            frames, _, _ = pipe.generate(embeds, max_frames=a.max_frames, seed=seed, verbose=False)
            wav = pipe.decode(frames)
            path = out if n == 0 else out.replace(".wav", f"_{n}.wav")
            write_wav(path, wav)
            t = pipe.last_timings
            audio_s = frames.shape[0] / FRAME_RATE
            total = t["prefill_s"] + t["decode_s"] + t.get("codec_s", 0.0)
            print(f"  {path}  {audio_s:.1f}s in {total:.2f}s "
                  f"({audio_s / max(total, 1e-9):.2f}x real time)")
            n += 1
        pipe.close()
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
