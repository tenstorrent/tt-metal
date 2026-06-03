# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 contract: ``Qwen25VlMultimodalPreprocessor`` vs ``Qwen2VLProcessor``.

``Qwen25VlTokenizerEncoderPair.encode_with_images`` consumes exactly:

- ``input_ids`` (int64/int32 batch)
- ``attention_mask`` (same batch layout as tokenizer)
- ``pixel_values`` (bf16/float32 patch stream for the vision encoder)
- ``image_grid_thw`` (per-image ``(T, H, W)`` token grids)

This test locks those tensors to the HuggingFace ``Qwen2VLProcessor`` reference on
CPU (no mesh / no TT device required).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, Qwen2Tokenizer, Qwen2VLProcessor

from models.tt_dit.encoders.qwen25vl.multimodal_preprocess import Qwen25VlMultimodalPreprocessor


@pytest.fixture(scope="module", name="hf_processor")
def _hf_processor():
    return Qwen2VLProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


@pytest.fixture(scope="module", name="edit_tokenizer")
def _edit_tokenizer():
    return Qwen2Tokenizer.from_pretrained("Qwen/Qwen-Image-Edit-2511", subfolder="tokenizer")


@pytest.fixture(scope="module", name="ours")
def _ours(edit_tokenizer):
    ip = AutoImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    return Qwen25VlMultimodalPreprocessor(tokenizer=edit_tokenizer, image_processor=ip)


def test_multimodal_preprocess_matches_hf_single_image(hf_processor, ours):
    rng = np.random.default_rng(42)
    img = Image.fromarray(rng.integers(0, 255, size=(384, 384, 3), dtype=np.uint8), mode="RGB")
    text = [
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> describe<|im_end|>\n" "<|im_start|>assistant\n"
    ]
    kw = dict(text=text, images=[img], padding=True, return_tensors="pt")
    ref = hf_processor(**kw)
    got = ours(**kw)
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        assert torch.equal(ref[key], got[key]), key


def test_multimodal_preprocess_matches_hf_batch_two_prompts(hf_processor, ours):
    rng = np.random.default_rng(7)
    images = [
        Image.fromarray(rng.integers(0, 255, size=(256, 320, 3), dtype=np.uint8), mode="RGB"),
        Image.fromarray(rng.integers(0, 255, size=(320, 256, 3), dtype=np.uint8), mode="RGB"),
    ]
    text = [
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> one<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|> two<|im_end|>\n<|im_start|>assistant\n",
    ]
    kw = dict(text=text, images=images, padding=True, return_tensors="pt")
    ref = hf_processor(**kw)
    got = ours(**kw)
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        assert torch.equal(ref[key], got[key]), key
