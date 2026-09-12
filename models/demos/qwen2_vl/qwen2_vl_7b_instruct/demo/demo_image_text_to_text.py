# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runnable demo: Qwen2-VL-7B-Instruct image-text-to-text on Tenstorrent.

Loads a real image + text prompt through the HF Qwen2VLProcessor, runs the ONE
shared chained TTNN pipeline (`tt/pipeline.py`, the same code the e2e test
exercises), and prints the generated answer text.

Run:
    ./python_env/bin/python -m models.demos.qwen2_vl.qwen2_vl_7b_instruct.demo.demo_image_text_to_text \
        --prompt "Describe the colors in this image." --max-new-tokens 24
"""

from __future__ import annotations

import argparse
import sys

import torch
from PIL import Image

import ttnn

from ..tt.pipeline import build_pipeline

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def _default_image():
    """Deterministic gradient image (same one the golden was captured from)."""
    img = Image.new("RGB", (112, 112))
    px = img.load()
    for y in range(112):
        for x in range(112):
            px[x, y] = ((x * 2) % 256, (y * 2) % 256, (x + y) % 256)
    return img


def run_demo(image_path=None, prompt="Describe the colors in this image.", max_new_tokens=24):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    image = Image.open(image_path).convert("RGB") if image_path else _default_image()

    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[text], images=[image], return_tensors="pt")
    inputs = {
        "input_ids": enc.input_ids,
        "attention_mask": enc.attention_mask,
        "pixel_values": enc.pixel_values,
        "image_grid_thw": enc.image_grid_thw,
    }

    model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        pipe = build_pipeline(device, model)
        # Use the fixed-capacity KV-cache decode (prefill once + O(1) seq=1
        # steps): same tokens as the full-seq recompute path but ~2x faster and
        # flat with context. The cache capacity must cover prompt + new tokens
        # and be a multiple of 32 (KV-cache write op requirement).
        S = int(inputs["input_ids"].shape[1])
        pipe.capacity = ((S + max_new_tokens + 31) // 32) * 32
        tokens, _ = pipe.generate_kv(inputs, max_new_tokens=max_new_tokens)
        answer = proc.batch_decode(torch.tensor([tokens]), skip_special_tokens=True)[0]
    finally:
        ttnn.close_device(device)

    print("=" * 60)
    print(f"PROMPT: {prompt}")
    print(f"ANSWER: {answer}")
    print("=" * 60)
    print(f"generated token ids: {tokens}")
    return answer


def main():
    ap = argparse.ArgumentParser(description="Qwen2-VL-7B-Instruct image-text-to-text TTNN demo")
    ap.add_argument("--image", default=None, help="path to an image (default: built-in gradient image)")
    ap.add_argument("--prompt", default="Describe the colors in this image.")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()
    run_demo(args.image, args.prompt, args.max_new_tokens)


if __name__ == "__main__":
    sys.exit(main())
