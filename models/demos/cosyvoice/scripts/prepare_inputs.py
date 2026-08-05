# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run CosyVoice's front-end once and emit everything the TTNN pipeline needs.

The front-end is not a neural network this bring-up ports. It is a text
normaliser, a Whisper-family tokenizer, an **ONNX** speech tokenizer
(`speech_tokenizer_v1.onnx`) and an **ONNX** speaker encoder (`campplus.onnx`),
plus a mel filterbank. Three of those four are ONNX graphs shipped as blobs, and
none of them is on the bounty's critical path -- the bounty is the LLM, the flow
decoder and the vocoder.

So the boundary is drawn here, and it is the same two-environment boundary as
`export_weights.py`: this script runs **once in the CosyVoice venv**, and writes a
flat `.npz` the TTNN side loads without importing cosyvoice, onnxruntime, or the
reference's torch pin.

    PYTHONPATH=/root/tt/CosyVoice:/root/tt/CosyVoice/third_party/Matcha-TTS \
    /root/tt/cosyvoice_env/bin/python prepare_inputs.py \
        --mode zero_shot --text "..." --prompt-text "..." --prompt-wav a.wav \
        --out inputs.npz

The four modes differ only in which fields are populated -- see
`tt/pipeline.py::describe_mode`. Note the instruct trap documented there: the
instruction goes in the **prompt_text slot**, and CosyVoice-1 wants a style
*description*, not a directive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

DEFAULT_ROOT = os.environ.get("COSYVOICE_ROOT", "/root/tt/CosyVoice")
MODES = ("sft", "zero_shot", "cross_lingual", "instruct")


def build(args) -> dict:
    from cosyvoice.cli.cosyvoice import CosyVoice

    model_dir = os.path.join(args.cosyvoice_root, args.checkpoint)
    cosy = CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=False)
    fe = cosy.frontend

    if args.mode == "sft":
        payload = fe.frontend_sft(args.text, args.speaker)
    elif args.mode == "zero_shot":
        speech = load_wav(args.prompt_wav, fe)
        payload = fe.frontend_zero_shot(args.text, args.prompt_text, speech, cosy.sample_rate)
    elif args.mode == "cross_lingual":
        speech = load_wav(args.prompt_wav, fe)
        payload = fe.frontend_cross_lingual(args.text, speech, cosy.sample_rate)
    else:
        # the instruction lands in prompt_text; a *description*, not a directive
        payload = fe.frontend_instruct(args.text, args.speaker, args.instruct)

    out = {}
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        elif isinstance(v, (int, float)):
            out[k] = np.asarray(v)
    return out


def load_wav(path: str, fe):
    """torchaudio rather than `whisper.load_audio`, which shells out to ffmpeg --
    one fewer system dependency for a container to have to carry."""
    import torchaudio

    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    return wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice-root", default=os.path.join(DEFAULT_ROOT, "pretrained_models"))
    ap.add_argument("--checkpoint", default="CosyVoice-300M")
    ap.add_argument("--mode", default="zero_shot", choices=MODES)
    ap.add_argument("--text", required=True)
    ap.add_argument("--prompt-text", default="")
    ap.add_argument("--prompt-wav", default=None)
    ap.add_argument("--speaker", default="中文女", help="sft / instruct: a name from the speaker table")
    ap.add_argument("--instruct", default="", help="instruct mode: a character or style DESCRIPTION")
    ap.add_argument("--out", default="inputs.npz")
    args = ap.parse_args()

    sys.path.insert(0, DEFAULT_ROOT)
    sys.path.insert(0, os.path.join(DEFAULT_ROOT, "third_party", "Matcha-TTS"))

    if args.mode in ("zero_shot", "cross_lingual") and not args.prompt_wav:
        raise SystemExit(f"--prompt-wav is required for {args.mode}")

    arrays = build(args)
    arrays["__meta__"] = np.frombuffer(
        json.dumps({"mode": args.mode, "text": args.text, "checkpoint": args.checkpoint}).encode(), dtype=np.uint8
    )
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {args.out}")
    for k, v in arrays.items():
        if k != "__meta__":
            print(f"  {k:28s} {v.shape} {v.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
