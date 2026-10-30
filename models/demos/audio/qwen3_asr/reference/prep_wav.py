# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side preprocessing for the raw-wav, language-auto TT demo.

For each clip: build the Qwen3-ASR prompt with NO forced language (auto-detect), run
the processor to get input_ids + mel, and also run the full CPU Qwen3-ASR (the existing
solution) to get the baseline transcription + detected language. Saves a per-clip npz
the TT demo consumes; the TT path then runs MY encoder+decoder on the same input_ids+mel.

Run in the CPU-reference virtualenv (``requirements-reference.txt``) — it needs the full
``qwen_asr`` package and its transformers pin, which must not be installed into the
tt-metal environment. The default clip list is an in-repo 16 kHz mono wav so this works
from a clean checkout; pass ``--clip name=path[,start,dur]`` (repeatable) for your own audio.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

AUDIO_TOKEN_ID = 151676

# Default clip: an in-repo 16 kHz mono speech wav, so the demo prep runs from a clean
# checkout (this used to point at machine-local audio).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DEFAULT_CLIPS = [  # (name, wav, start_s, dur_s)
    (
        "en_sample",
        os.path.join(
            REPO_ROOT,
            "models",
            "demos",
            "audio",
            "whisper",
            "demo",
            "dataset",
            "conditional_generation",
            "17646385371758249908.wav",
        ),
        0.0,
        20.0,
    ),
]


def parse_clip(spec):
    """``name=path[,start,dur]`` -> (name, path, start, dur)."""
    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise argparse.ArgumentTypeError("expected name=path[,start,dur]")
    parts = rest.split(",")
    path = parts[0]
    start = float(parts[1]) if len(parts) > 1 else 0.0
    dur = float(parts[2]) if len(parts) > 2 else 20.0
    return (name, path, start, dur)


def load_slice(path, start, dur, sr=16000):
    w, file_sr = sf.read(path, dtype="float32")
    assert file_sr == sr
    if w.ndim > 1:
        w = w.mean(1)
    a = int(start * sr)
    b = min(len(w), a + int(dur * sr))
    return w[a:b].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("QWEN3ASR_WAV_DIR", "/tmp/qwen3_asr_wav"))
    ap.add_argument(
        "--clip",
        type=parse_clip,
        action="append",
        dest="clips",
        metavar="NAME=PATH[,START,DUR]",
        help="clip to prepare (repeatable); defaults to the in-repo sample wav",
    )
    args = ap.parse_args()
    clips = args.clips or DEFAULT_CLIPS
    os.makedirs(args.out, exist_ok=True)

    from qwen_asr import Qwen3ASRModel

    wrap = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B", dtype=torch.float32, device_map="cpu", max_inference_batch_size=1, max_new_tokens=128
    )
    summary = {}
    for name, path, start, dur in clips:
        wav = load_slice(path, start, dur)
        # processor inputs with NO forced language (auto-detect)
        prompt = wrap._build_text_prompt(context="", force_language=None)
        inputs = wrap.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"][0].cpu().numpy().astype(np.int64)
        feats = inputs["input_features"]
        mel = feats[0].float().cpu().numpy() if feats.dim() == 3 else feats.float().cpu().numpy()
        n_audio = int((input_ids == AUDIO_TOKEN_ID).sum())

        # existing solution: full CPU Qwen3-ASR, auto-detect
        res = wrap.transcribe(audio=[(wav, 16000)], language=None)[0]
        cpu_text, cpu_lang = res.text.strip(), (res.language or "")

        np.savez(os.path.join(args.out, f"{name}.npz"), input_ids=input_ids, mel=mel, prompt_len=len(input_ids))
        summary[name] = {
            "wav": path,
            "start": start,
            "dur": dur,
            "n_audio_tokens": n_audio,
            "prompt_len": int(len(input_ids)),
            "mel_shape": list(mel.shape),
            "cpu_text": cpu_text,
            "cpu_lang": cpu_lang,
        }
        print(
            f"[{name}] mel={mel.shape} ids={input_ids.shape} audio_tok={n_audio} "
            f"lang={cpu_lang!r}\n   CPU: {cpu_text!r}"
        )
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
