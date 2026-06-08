# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end OCR test for the full dots.ocr model in TP4 (Blackhole P150x4).

Runs the entire image->text pipeline through the integrated ``DotsOCRModelTP4``:
vision tower (TP4 BH) -> host scatter-merge into the text stream -> replicated-
hidden prefill (fills the paged KV cache) -> paged single-token decode.

The vision TP4 BH kernels are hardware-swept for the S=11264 vision bucket
(grid 88x128 -> 2816 merged image tokens), so the demo image is resized to land
on exactly that grid; the test skips if the processor produces a different grid.

Run::

    MESH_DEVICE=P150x4 pytest -s \\
        models/experimental/dots_ocr_tp4/tests/test_dots_ocr_ocr.py
"""

import json
import os

import pytest
import torch

import ttnn

from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape
from models.experimental.dots_ocr_tp4.tt.dots_ocr_model import DotsOCRModelTP4, VISION_GRID_THW

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
IMAGE_TOKEN = "<|imgpad|>"
IMAGE_TOKEN_ID = 151665

# Force the vision bucket: grid (t=1, h=88, w=128) -> 11264 patches.
_T, _GH, _GW = VISION_GRID_THW
TARGET_W = _GW * PATCH_SIZE  # 1792
TARGET_H = _GH * PATCH_SIZE  # 1232
MAX_PIXELS = _GH * _GW * PATCH_SIZE * PATCH_SIZE  # 11264 * 14 * 14 = 2_207_744

DEMO_IMAGE = "https://raw.githubusercontent.com/rednote-hilab/dots.ocr/master/demo/demo_image1.jpg"


def _resolve_model_path():
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    from huggingface_hub import snapshot_download

    return snapshot_download(DOTS_OCR_MODEL_ID)


def _build_processor(model_path):
    from transformers import AutoImageProcessor, AutoTokenizer, AutoVideoProcessor, Qwen2_5_VLProcessor

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    video_processor = AutoVideoProcessor.from_pretrained(model_path)
    with open(os.path.join(model_path, "chat_template.json")) as f:
        chat_template = json.load(f)["chat_template"]
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, video_processor, chat_template=chat_template)
    processor.image_token = IMAGE_TOKEN
    processor.image_token_id = IMAGE_TOKEN_ID
    # Pin the resize pixel budget so smart-resize keeps the (pre-resized) image at
    # its 11264-patch grid instead of down/up-scaling it to a different bucket.
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = MAX_PIXELS
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min(int(getattr(image_processor, "min_pixels", 0) or 0), MAX_PIXELS)
    return processor, tokenizer


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("max_new_tokens", [int(os.environ.get("DOTS_OCR_TP4_OCR_TOKENS", "128"))])
def test_dots_ocr_tp4_ocr(mesh_device, max_new_tokens):
    from PIL import Image
    import requests
    from transformers import AutoModelForCausalLM

    # The vision TP4 BH kernels require Blackhole and the 4-chip TP4 mesh.
    if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
        pytest.skip(f"vision TP4 BH requires Blackhole, got {mesh_device.arch().name}")
    if int(mesh_device.get_num_devices()) != 4:
        pytest.skip(f"dots.ocr TP4 requires a 4-device mesh (P150x4), got {mesh_device.get_num_devices()}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    model_path = _resolve_model_path()
    processor, tokenizer = _build_processor(model_path)

    # --- Demo image, resized to land on the 88x128 vision bucket. ---
    image = Image.open(requests.get(DEMO_IMAGE, stream=True).raw).convert("RGB")
    image = image.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract the text from this image."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # ``process_vision_info`` (qwen_vl_utils) just extracts the message images; for
    # a single already-loaded PIL image it is equivalent to ``[image]``. Fall back
    # to that when the optional package is absent so the test stays self-contained.
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
    except ImportError:
        image_inputs, video_inputs = [image], None
    inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    grid = inputs["image_grid_thw"]
    if grid.tolist() != [VISION_GRID_THW]:
        pytest.skip(
            f"demo image produced image_grid_thw={grid.tolist()}, but the vision TP4 BH "
            f"kernels are swept for [[{_T}, {_GH}, {_GW}]] (S=11264). Adjust the resize "
            f"target or the processor pixel budget to hit that grid."
        )

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)

    # --- Build the integrated TP4 model and generate. ---
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    model = DotsOCRModelTP4.from_hf(mesh_device, hf_model)

    generated_ids = model.generate(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=grid,
        max_new_tokens=max_new_tokens,
        stop_on_eos=True,
    )

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "=" * 70)
    print(
        f"[dots_ocr_tp4 OCR] grid={grid.tolist()}  prompt_len={int(input_ids.shape[1])}  "
        f"generated={len(generated_ids)} tokens"
    )
    print(f"OCR OUTPUT:\n{text}")
    print("=" * 70)

    assert len(text.strip()) > 0, "OCR output should not be empty"
