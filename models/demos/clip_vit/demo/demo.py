# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import requests
import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, CLIPModel

import ttnn
from models.demos.clip_vit.tt.tt_clip_model import TtCLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"

DEMO_IMAGES = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
]

DEMO_LABELS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car",
    "a person riding a bicycle",
    "a group of people playing sports",
    "food on a dining table",
    "a scenic landscape with mountains",
]


def run_clip_demo(device, image_urls, labels):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    hf_model = CLIPModel.from_pretrained(MODEL_NAME).eval()

    pil_images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
    proc = processor(
        labels,
        pil_images,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )

    with torch.no_grad():
        ref_probs = hf_model(**proc).logits_per_image.softmax(dim=-1)

    ttnn_model = TtCLIPModel(hf_model.config, hf_model, device)
    ttnn_input_ids = ttnn.from_torch(
        proc["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_pixel_values = ttnn.from_torch(
        proc["pixel_values"],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    seq_len = proc["input_ids"].shape[1]
    ttnn_position_ids = ttnn.from_torch(
        torch.arange(seq_len).unsqueeze(0).expand(proc["input_ids"].shape[0], -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    logits_per_image, _ = ttnn_model(
        input_ids=ttnn_input_ids,
        pixel_values=ttnn_pixel_values,
        position_ids=ttnn_position_ids,
    )
    tt_probs = ttnn.to_torch(logits_per_image).to(torch.float32).softmax(dim=-1)

    logger.info(f"Ran CLIP demo on {len(image_urls)} image(s) × {len(labels)} label(s)")
    mismatches = 0
    for i, url in enumerate(image_urls):
        ref_top = labels[int(ref_probs[i].argmax())]
        tt_top = labels[int(tt_probs[i].argmax())]
        mark = "PASS" if ref_top == tt_top else "DIFF"
        logger.info(f"  [{mark}] {url}")
        logger.info(f"         HF   top-1: {ref_top!r}")
        logger.info(f"         TTNN top-1: {tt_top!r}")
        if ref_top != tt_top:
            mismatches += 1

    assert mismatches == 0, f"TTNN top-1 disagreed with HF on {mismatches}/{len(image_urls)} images"
    logger.info("CLIP demo completed successfully")


def test_clip_demo(device):
    run_clip_demo(device, DEMO_IMAGES, DEMO_LABELS)


def test_clip_demo_dp(mesh_device):
    run_clip_demo(mesh_device, DEMO_IMAGES, DEMO_LABELS)
