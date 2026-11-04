# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Record the bf16 CPU HuggingFace reference (logits, vision tensors, greedy text) the gates score against.

Usage::

    python models/demos/blackhole/paddleocr_vl/tests/generate_goldens.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from PIL import Image

from models.demos.blackhole.paddleocr_vl.tests.ocr_corpus import CORPUS, build_corpus

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
OCR_PROMPT = "OCR:"

# One image per bucket, chosen for dense text so a numerical fault has somewhere
# to show up. Keep this list short: each entry writes ~40 MB of tensors.
INTERMEDIATE_SAMPLES = ("label_shipping", "receipt_grocery", "table_quarterly", "page_table_large")

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.abspath(os.path.join(HERE, "..", "demo"))
IMAGES_DIR = os.path.join(DEMO, "sample_images")
GOLDEN_DIR = os.path.join(DEMO, "golden")


def build_inputs(processor, image: Image.Image, prompt: str = OCR_PROMPT):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return processor(text=[text], images=[image], return_tensors="pt")


@torch.no_grad()
def capture_intermediates(model, inputs) -> dict:
    """Tower and projector outputs via get_image_features, the same path generation uses."""
    vision_out = model.model.get_image_features(
        pixel_values=inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    )
    return {
        "pixel_values": inputs["pixel_values"].to(torch.float32),
        "image_grid_thw": inputs["image_grid_thw"],
        "tower_out": vision_out.last_hidden_state.to(torch.float32),
        "projector_out": vision_out.pooler_output.to(torch.float32),
    }


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process the first N samples")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--skip-intermediates", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"[goldens] rendering corpus -> {IMAGES_DIR}")
    records = {r["name"]: r for r in build_corpus(IMAGES_DIR)}

    print(f"[goldens] loading {MODEL_ID} (bf16, cpu)")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()
    print(f"[goldens] loaded in {time.time() - t0:.1f}s")

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    samples = CORPUS[: args.limit] if args.limit else CORPUS
    out = []

    def write_manifest():
        """Rewritten after every sample so an interrupted CPU pass keeps its progress."""
        manifest = {
            "model_id": MODEL_ID,
            "prompt": OCR_PROMPT,
            "dtype": "bfloat16",
            "device": "cpu",
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
            "complete": len(out) == len(samples),
            "n_expected": len(samples),
            "samples": out,
        }
        with open(os.path.join(GOLDEN_DIR, "hf_goldens.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    for i, s in enumerate(samples, 1):
        rec = records[s.name]
        img = Image.open(rec["path"]).convert("RGB")
        inputs = build_inputs(processor, img)

        n_img_tok = int((inputs["image_grid_thw"].prod(dim=-1) // 4).sum())
        t0 = time.time()
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        dt = time.time() - t0

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = gen[0][prompt_len:]
        text = processor.batch_decode([new_tokens], skip_special_tokens=True)[0]

        out.append(
            {
                "name": s.name,
                "bucket": s.bucket,
                "kind": s.kind,
                "image": os.path.relpath(rec["path"], DEMO),
                "image_size": rec["size"],
                "image_grid_thw": inputs["image_grid_thw"][0].tolist(),
                "n_image_tokens": n_img_tok,
                "prompt_len": prompt_len,
                "n_generated": int(new_tokens.numel()),
                "seconds": round(dt, 2),
                "ground_truth": s.ground_truth,
                "hf_output": text,
            }
        )
        print(
            f"[goldens] {i:2d}/{len(samples)} {s.name:20s} bucket={s.bucket:5d} "
            f"imgtok={n_img_tok:5d} gen={int(new_tokens.numel()):4d} {dt:6.1f}s"
        )

        if not args.skip_intermediates and s.name in INTERMEDIATE_SAMPLES:
            tensors = capture_intermediates(model, inputs)
            path = os.path.join(GOLDEN_DIR, f"intermediates_{s.name}.pt")
            torch.save(tensors, path)
            shapes = {k: list(v.shape) for k, v in tensors.items()}
            print(f"[goldens]      intermediates -> {os.path.basename(path)} {shapes}")

        write_manifest()

    print(f"[goldens] wrote {os.path.join(GOLDEN_DIR, 'hf_goldens.json')} ({len(out)}/{len(samples)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
