# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run CosyVoice's front-end over the 4-mode x 5-language sweep and emit device inputs.

The front-end is not a neural network this bring-up ports. It is a text
normaliser, a Whisper-family tokenizer, an **ONNX** speech tokenizer
(`speech_tokenizer_v1.onnx`) and an **ONNX** speaker encoder (`campplus.onnx`),
plus a mel filterbank. Three of those four are ONNX graphs shipped as blobs, and
none is on the bounty's critical path -- the bounty is the LLM, the flow decoder
and the vocoder.

So the boundary is drawn here, the same two-environment boundary as
`export_weights.py`: this runs **once in the CosyVoice venv** and writes a flat
`.npz` per case that the TTNN side loads without importing cosyvoice or
onnxruntime.

    PYTHONPATH=/mnt/CosyVoice:/mnt/CosyVoice/third_party/Matcha-TTS \
    /mnt/cosyvoice_env/bin/python prepare_inputs.py --out-dir /tmp/sweep

The texts, speakers and instruct descriptions are imported from
`run_reference.py` rather than duplicated, so the TTNN sweep and the PyTorch
baseline are provably the same corpus -- which is the whole point of comparing
them.

Three mode-specific shapes, all of them easy to get wrong:

* **cross_lingual** deletes `prompt_text` and `llm_prompt_speech_token`, so the
  LLM sees only the target text while the *flow* still gets the prompt speech
  tokens and mel. The text carries a `<|zh|>`-style language tag instead.
* **instruct** deletes `llm_embedding` -- the LLM runs with no speaker vector at
  all -- and puts the instruction in the `prompt_text` slot. It must be a style
  *description*, not a directive; see `run_reference.py` for what that cost.
* **sft** and **instruct** have no prompt audio, so the flow has no
  `prompt_speech_feat` and the generated mel starts at frame 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

DEFAULT_ROOT = os.environ.get("COSYVOICE_ROOT", "/mnt/CosyVoice")
MODES = ("sft", "zero_shot", "cross_lingual", "instruct")


def resolve_speaker(cosy, prefs):
    """First available speaker from the per-language preference list."""
    available = list(cosy.list_available_spks())
    for name in prefs:
        if name in available:
            return name
    return available[0] if available else ""


def build_case(cosy, mode: str, lang: str, texts, xling_tag, spk_pref, instructs, prompt_text, prompt_wav, xling_wav):
    fe = cosy.frontend
    text = texts[lang]
    if mode == "sft":
        return fe.frontend_sft(text, resolve_speaker(cosy, spk_pref[lang])), text
    if mode == "zero_shot":
        return fe.frontend_zero_shot(text, prompt_text, prompt_wav, cosy.sample_rate, ""), text
    if mode == "cross_lingual":
        tagged = xling_tag[lang] + text
        return fe.frontend_cross_lingual(tagged, xling_wav, cosy.sample_rate, ""), text
    return fe.frontend_instruct(text, resolve_speaker(cosy, spk_pref[lang]), instructs[lang]), text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice-root", default=DEFAULT_ROOT)
    ap.add_argument("--checkpoint", default="CosyVoice-300M")
    ap.add_argument("--out-dir", default="/tmp/cosy_sweep")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--langs", default="zh,en,ja,yue,ko")
    args = ap.parse_args()

    sys.path.insert(0, args.cosyvoice_root)
    sys.path.insert(0, os.path.join(args.cosyvoice_root, "third_party", "Matcha-TTS"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from cosyvoice.cli.cosyvoice import CosyVoice

    # the corpus itself comes from run_reference.py, never duplicated
    from run_reference import INSTRUCTS, SPK_PREF, TEXTS, XLING_TAG, ZERO_SHOT_PROMPT_TEXT

    # Paths, not loaded tensors: `_extract_speech_feat` calls load_wav(path, 24000)
    # itself while `_extract_speech_token` resamples to 16 kHz separately, so the
    # frontend needs the file and does its own two resamples.
    asset = os.path.join(args.cosyvoice_root, "asset")
    prompt_wav = os.path.join(asset, "zero_shot_prompt.wav")
    xling_wav = os.path.join(asset, "cross_lingual_prompt.wav")

    os.makedirs(args.out_dir, exist_ok=True)
    index = []
    for mode in args.modes.split(","):
        # instruct needs the -Instruct checkpoint, sft the -SFT one; both share the
        # architecture, so this is a weight swap and not a different graph.
        ckpt = {"sft": "-SFT", "instruct": "-Instruct"}.get(mode, "")
        model_dir = os.path.join(args.cosyvoice_root, "pretrained_models", args.checkpoint + ckpt)
        if not os.path.exists(model_dir):
            print(f"  skip {mode}: {model_dir} not present")
            continue
        cosy = CosyVoice(model_dir, load_jit=False, load_trt=False, fp16=False)
        for lang in args.langs.split(","):
            payload, text = build_case(
                cosy, mode, lang, TEXTS, XLING_TAG, SPK_PREF, INSTRUCTS, ZERO_SHOT_PROMPT_TEXT, prompt_wav, xling_wav
            )
            arrays = {k: v.detach().cpu().numpy() for k, v in payload.items() if isinstance(v, torch.Tensor)}
            meta = {
                "mode": mode,
                "lang": lang,
                "text": text,
                "checkpoint": os.path.basename(model_dir),
                "keys": sorted(arrays),
            }
            arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
            name = f"{mode}_{lang}.npz"
            np.savez_compressed(os.path.join(args.out_dir, name), **arrays)
            index.append({"file": name, **{k: meta[k] for k in ("mode", "lang", "text", "checkpoint")}})
            shapes = " ".join(f"{k}{list(v.shape)}" for k, v in arrays.items() if k != "__meta__")
            print(f"  {mode:<14} {lang:<4} {shapes}")
        del cosy

    with open(os.path.join(args.out_dir, "index.json"), "w") as fh:
        json.dump({"cases": index}, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(index)} cases -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
