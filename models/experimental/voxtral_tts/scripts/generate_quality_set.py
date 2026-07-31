# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate the device audio set used for end-to-end quality scoring.

FREE-RUNNING, not teacher-forced -- the model is fed its OWN codes each step, which is what
serving does and what no per-block PCC or `compare_codes()` can tell you. Everything that only
appears after tens of autoregressive steps (drift, repetition, early or absent [END_AUDIO])
is visible here and nowhere else.

    HF_MODEL=<export dir> python .../scripts/generate_quality_set.py
    ... --cases 2,3            # just those fixture cases
    ... --max-frames 700       # override the per-case budget
    ... --out <dir>            # default is voxtral_tts/generated/

Writes one WAV per case plus `results.json`, which `score_quality_set.py` consumes. Output defaults
to `voxtral_tts/generated/`, which is gitignored -- the clips are large and derive from CC BY-NC
weights, so they stay local.

THE FRAME BUDGET IS A MEASUREMENT TRAP. Frames are 12.5 Hz and speech runs ~18 chars/s, so the
676-char fixture paragraphs need ~460 frames. Capping at 200 truncates them mid-sentence and the
run reports "no natural stop" -- which reads exactly like a model that will not terminate. It cost
real time; the budget below is therefore derived from the text length, not fixed.
"""

import argparse
import json
import math
import os
import time
import wave

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import FRAME_RATE, TtVoxtralPipeline

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(_HERE, "tests", "prompt_fixture.json")
# `generated/` is gitignored (voxtral_tts/.gitignore) -- audio is large and derives from CC BY-NC
# weights, so it must never be committed. Default here so the clips are easy to find and play.
DEFAULT_OUT = os.path.join(_HERE, "generated")


def frame_budget(text):
    """~18 chars/s of speech at 12.5 frames/s, x2.2 margin, floor 320. See the docstring.

    The margin is generous on purpose, and the budget is a CAP rather than a cost -- generation
    stops on [END_AUDIO] regardless, so over-provisioning is free and under-provisioning silently
    fakes a non-terminating model.

    THE FLOOR IS WHAT MATTERS, and 200 was not enough. Character count only predicts duration for
    prose: fixture case 10 is 55 characters of "Numbers 1234567890 and symbols !@#$%^&*()", which
    the model VOCALISES at 240 frames (fp32) / 267 (bf16) -- 4.4 frames per character against the
    1.53 this formula assumes. A 200-frame floor truncated it and looked like a bf16 regression
    until the fp32 run was checked. 320 covers the measured worst case with room.

    Measured usage otherwise: 68-102 frames for the 100-char prompts, 458-490 for the 676-char
    ones, so an expressive voice can want 1.5x what a flat one does at the same length.
    """
    return max(320, int(math.ceil(len(text) / 18.0 * FRAME_RATE * 2.2)))


def save_wav(wav, path, sr=24000):
    a = (wav.detach().reshape(-1).clamp(-1, 1).numpy() * 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(a.tobytes())


def artifacts(wav):
    """Objective defect checks. These do NOT replace listening; they catch the obvious and they
    are the only part of audio quality that regresses detectably in CI."""
    x = wav.reshape(-1).float()
    return {
        "peak": x.abs().max().item(),
        "rms": x.pow(2).mean().sqrt().item(),
        "clipped_%": (x.abs() >= 0.999).float().mean().item() * 100,
        "dc_offset": x.mean().item(),
        "silent_%": (x.abs() < 1e-4).float().mean().item() * 100,
        # crude discontinuity detector: a 24 kHz speech signal should never step by 0.5
        "click_count": int(((x[1:] - x[:-1]).abs() > 0.5).sum()),
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cases", default="all", help='"all" or e.g. "0,1,2"')
    ap.add_argument("--max-frames", type=int, default=0, help="0 = derive from text length")
    ap.add_argument("--tag", default="", help="suffix for WAV/results filenames")
    args = ap.parse_args()

    fx = json.load(open(FIXTURE))
    cases = (list(range(len(fx["cases"]))) if args.cases == "all"
             else [int(c) for c in args.cases.split(",")])
    os.makedirs(args.out, exist_ok=True)

    dev = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
        results = []
        for ci in cases:
            case = fx["cases"][ci]
            ids = torch.tensor(case["ids"], dtype=torch.long)
            embeds = pref.build_inputs_embeds(ids, pref.load_voice(case["voice"]), pipe.wb)
            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            budget = args.max_frames or frame_budget(case["text"])
            print(f"\n=== case {ci}: voice={case['voice']} P={len(ids)} budget={budget} ===")
            print(f"    text: {case['text'][:70]!r}")
            t0 = time.perf_counter()
            frames, t_pre, t_gen = pipe.generate(embeds, max_frames=budget, verbose=True)
            wav = pipe.decode(frames)
            total = time.perf_counter() - t0
            audio_s = frames.shape[0] / FRAME_RATE
            path = os.path.join(args.out, f"case{ci}_{case['voice']}{args.tag}.wav")
            save_wav(wav, path)
            a = artifacts(wav)
            terminated = frames.shape[0] < budget
            print(f"    frames {frames.shape[0]}  audio {audio_s:.1f}s  "
                  f"{'TERMINATED on [END_AUDIO]' if terminated else 'HIT the budget'}")
            print(f"    wall {total:.1f}s  RTF {total/audio_s:.2f}  "
                  f"(prefill {t_pre:.2f}s, {t_gen/max(frames.shape[0],1):.2f}s/frame)")
            print(f"    artifacts: peak {a['peak']:.3f} rms {a['rms']:.4f} "
                  f"clipped {a['clipped_%']:.3f}% dc {a['dc_offset']:+.5f} "
                  f"silent {a['silent_%']:.1f}% clicks {a['click_count']}")
            results.append({"case": ci, "voice": case["voice"], "text": case["text"],
                            "frames": int(frames.shape[0]), "audio_s": audio_s,
                            "terminated": terminated, "rtf": total / audio_s,
                            "wav": path, **a})
            json.dump(results, open(os.path.join(args.out, f"results{args.tag}.json"), "w"),
                      indent=2)
        print(f"\n  {len(results)} case(s) -> {args.out}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
