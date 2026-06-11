# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""
Phase 1 baseline capture.

Runs the upstream native PyTorch HiggsAudioV2 model (transformers >= 5.3) on CPU
bf16 with greedy decoding, and dumps the prompt + teacher-forcing audio trajectory to JSON
fixture. The TTNN accuracy harness (tests/test_accuracy_native.py) loads this fixture
and compares per-step argmax tokens.

Requires:
  - transformers >= 5.3 (native HiggsAudioV2ForConditionalGeneration)
  - torch >= 2.7 (torch.float8_e8m0fnu dtype, required by transformers 5.x)
  - HF_HUB_OFFLINE=1  (no network on the remote box)
  - processor_config.json patched to point `audio_tokenizer_name_or_path` at the
    local tokenizer directory.

Output:
  tests/fixtures/baseline_tts_short.json
"""
import json
import argparse
import os
import pathlib
import time

import numpy as np
import torch


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


MODEL_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/data/hf_cache/higgs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--out", default=str(MODEL_DIR / "tests" / "fixtures" / "baseline_tts_short.json"))
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration

    print(f"Loading processor from {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)

    print(f"Loading model from {args.model} (bf16, CPU)", flush=True)
    t0 = time.time()
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16
    ).to("cpu")
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "Generate speech in the style of a calm neutral male voice."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello, this is a baseline test."}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_text_tokens = inputs["input_ids"][0].cpu().numpy()
    print(f"  prompt input_ids shape: {inputs['input_ids'].shape}", flush=True)

    print(f"Generating up to {args.max_new_tokens} new tokens (greedy)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
        )
    dt = time.time() - t0
    print(f"  generation took {dt:.1f}s", flush=True)

    text_seq = out.sequences[0].cpu().numpy()
    audio_seqs = getattr(out, "audio_sequences", None)
    if audio_seqs is None or len(audio_seqs) == 0:
        audio_arr = np.zeros((0, 0), dtype=np.int64)
    else:
        audio_arr = np.asarray(audio_seqs[0].cpu().numpy(), dtype=np.int64)

    print(f"  generated text seq shape: {text_seq.shape}", flush=True)
    print(f"  audio_sequences shape: {audio_arr.shape}", flush=True)

    # The accuracy gate (tests/test_accuracy_native.py) only needs the prompt and
    # the teacher-forcing audio trajectory; write them as reviewable JSON.
    fixture = {
        "prompt_text_tokens": prompt_text_tokens.astype(int).tolist(),
        "audio_tokens": audio_arr.astype(int).tolist(),  # [T, num_codebooks]
    }
    with open(out_path, "w") as fh:
        json.dump(fixture, fh, indent=0)
    print(f"  wrote fixture: {out_path}", flush=True)


if __name__ == "__main__":
    main()
