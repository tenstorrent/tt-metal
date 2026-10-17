# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, CLIPModel

import ttnn
from models.demos.clip_vit.tt.tt_clip_model import TtCLIPModel
from models.demos.clip_vit.tt.tt_clip_model_optimized import TtCLIPModelOptimized

MODEL_NAME = "openai/clip-vit-base-patch32"

BATCH_SIZE = 7

DEMO_IMAGES = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
    "http://images.cocodataset.org/val2017/000000087038.jpg",
    "http://images.cocodataset.org/val2017/000000174482.jpg",
    "http://images.cocodataset.org/val2017/000000403385.jpg",
]

DEMO_LABELS = [
    "a photo of a cat",
    "a photo of a dog",
    "a person on a beach",
    "a train on tracks",
    "a bowl of food",
    "a living room with furniture",
    "a street scene in a city",
]

assert len(DEMO_IMAGES) == BATCH_SIZE
assert len(DEMO_LABELS) == BATCH_SIZE


def _build_ttnn_model(hf_model, device, use_opt: bool, dtype):
    if use_opt:
        return TtCLIPModelOptimized(
            hf_model.config,
            hf_model,
            device,
            vision_batch=BATCH_SIZE,
            text_batch=BATCH_SIZE,
            dtype=dtype,
        )
    return TtCLIPModel(hf_model.config, hf_model, device)


def _resolve_config(pytestconfig):
    use_opt = pytestconfig.getoption("--opt")
    b8 = pytestconfig.getoption("--b8")
    b16 = pytestconfig.getoption("--b16")

    if b8 and b16:
        pytest.fail("Pass only one of --b8 or --b16.")
    if (b8 or b16) and not use_opt:
        pytest.fail("--b8/--b16 only apply to the optimized model; add --opt.")

    dtype = ttnn.bfloat8_b if b8 else ttnn.bfloat16
    return use_opt, dtype


def run_clip_demo(device, pytestconfig):
    use_opt, dtype = _resolve_config(pytestconfig)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    hf_model = CLIPModel.from_pretrained(MODEL_NAME).eval()

    pil_images = [Image.open(requests.get(url, stream=True).raw) for url in DEMO_IMAGES]
    proc = processor(
        DEMO_LABELS,
        pil_images,
        padding="max_length",
        max_length=77,
        return_tensors="pt",
    )

    with torch.no_grad():
        ref_probs = hf_model(**proc).logits_per_image.softmax(dim=-1)

    ttnn_model = _build_ttnn_model(hf_model, device, use_opt, dtype)

    ttnn_input_ids = ttnn.from_torch(
        proc["input_ids"],
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    pixel_layout = ttnn.ROW_MAJOR_LAYOUT if use_opt else ttnn.TILE_LAYOUT
    ttnn_pixel_values = ttnn.from_torch(
        proc["pixel_values"],
        dtype=ttnn.bfloat16,
        layout=pixel_layout,
        device=device,
    )

    if use_opt:
        logits_per_image, _ = ttnn_model(
            input_ids=ttnn_input_ids,
            pixel_values=ttnn_pixel_values,
        )
    else:
        seq_len = proc["input_ids"].shape[1]
        ttnn_position_ids = ttnn.from_torch(
            torch.arange(seq_len).unsqueeze(0).expand(BATCH_SIZE, -1),
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

    model_tag = "optimized" if use_opt else "base"
    dtype_tag = "bfloat8_b" if dtype == ttnn.bfloat8_b else "bfloat16"
    logger.info(f"CLIP demo: {model_tag} model, dtype={dtype_tag}, " f"{BATCH_SIZE} image(s) × {BATCH_SIZE} label(s)")

    mismatches = 0
    for i, url in enumerate(DEMO_IMAGES):
        ref_top = DEMO_LABELS[int(ref_probs[i].argmax())]
        tt_top = DEMO_LABELS[int(tt_probs[i].argmax())]
        mark = "PASS" if ref_top == tt_top else "DIFF"
        logger.info(f"  [{mark}] {url}")
        logger.info(f"         HF   top-1: {ref_top!r}")
        logger.info(f"         TTNN top-1: {tt_top!r}")
        if ref_top != tt_top:
            mismatches += 1

    logger.info(f"CLIP demo completed: {BATCH_SIZE - mismatches}/{BATCH_SIZE} images matched HF top-1")


def test_clip_demo(device, pytestconfig):
    run_clip_demo(device, pytestconfig)
