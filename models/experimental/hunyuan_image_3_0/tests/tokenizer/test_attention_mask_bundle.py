# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-span attention mask from host bundle (prepare_full_attn_slices path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    build_attention_mask_for_bundle,
    build_full_attn_slices,
    bundle_to_denoise_cond,
    enrich_bundle_attention,
    prepare_gen_image_inputs,
    prepare_i2i_inputs,
)

PROMPT = "a cat on a mat"
IMAGE_SIZE = 1024
CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (1024, 1024), color=(128, 64, 32))


def _bidirectional_block(mask: torch.Tensor, span: slice) -> torch.Tensor:
    block = mask[0, 0, span, span]
    return torch.all(block)


def test_t2i_full_attn_single_gen_span(hunyuan_tokenizer, processor):
    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=1)
    spans = build_full_attn_slices(bundle, processor)
    assert len(spans) == 1
    assert len(spans[0]) == 1
    gen_slice = bundle.gen_image_slices[0][0]
    assert spans[0][0] == gen_slice

    mask = build_attention_mask_for_bundle(bundle, processor)
    assert mask.shape == (1, 1, bundle.seq_len, bundle.seq_len)
    assert _bidirectional_block(mask, gen_slice)


def test_i2i_full_attn_joint_plus_gen(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)

    spans = build_full_attn_slices(bundle, processor)
    assert len(spans) == 1
    assert len(spans[0]) == 2  # joint cond + gen

    joint_slice = bundle.joint_image_slices[0][0]
    gen_slice = bundle.gen_image_slices[0][0]
    assert spans[0][0] == joint_slice
    assert spans[0][1] == gen_slice

    mask = build_attention_mask_for_bundle(bundle, processor)
    assert _bidirectional_block(mask, joint_slice)
    assert _bidirectional_block(mask, gen_slice)

    # Distinct spans must not be mutually bidirectional.
    cross = mask[0, 0, joint_slice.start, gen_slice.start]
    assert not bool(cross)


def test_enrich_bundle_attention_populates_fields(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)
    assert bundle.full_attn_slices is None
    assert bundle.attention_mask is None

    enrich_bundle_attention(bundle, processor)
    assert bundle.full_attn_slices is not None
    assert bundle.attention_mask is not None
    assert bundle.attention_mask.dtype == torch.bool


def test_bundle_to_denoise_cond_i2i_shape(hunyuan_tokenizer, processor, rgb_image):
    from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_tensors

    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)
    wte = load_tensors(MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    cond_dict = bundle_to_denoise_cond(bundle, wte, processor)

    assert "text_pre" in cond_dict
    assert cond_dict["text_post"] is None or cond_dict["text_post"].shape[1] >= 0
    assert cond_dict["attention_mask"].shape == (1, 1, bundle.seq_len, bundle.seq_len)
    assert len(cond_dict["image_infos"][0]) == 3
    assert cond_dict["gen_slice"].stop - cond_dict["gen_slice"].start == 4096


def test_bundle_to_denoise_cond_matches_manual_build(hunyuan_tokenizer, processor):
    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=1)
    spans = build_full_attn_slices(bundle, processor)
    manual = build_attention_mask(bundle.seq_len, spans, bsz=1)
    auto = build_attention_mask_for_bundle(bundle, processor)
    assert torch.equal(manual, auto)
