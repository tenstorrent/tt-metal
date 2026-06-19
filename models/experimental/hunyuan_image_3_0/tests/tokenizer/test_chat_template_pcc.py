# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer tests for T2I / I2I chat template and host preprocess bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import prepare_gen_image_inputs, prepare_i2i_inputs

PROMPT = "a cat on a mat"
IMAGE_SIZE = 1024
CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (1024, 1024), color=(128, 64, 32))


def test_cfg_uncond_replaces_text_not_image(hunyuan_tokenizer):
    out = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)["output"]
    assert out.tokens.shape[0] == 2
    cfg_id = hunyuan_tokenizer.special.cfg_token_id
    boi_id = hunyuan_tokenizer.special.boi_token_id

    cond, uncond = out.tokens[0], out.tokens[1]
    boi_idx = (cond == boi_id).nonzero(as_tuple=False)[0].item()
    assert (uncond[:boi_idx] == cfg_id).any()
    assert torch.equal(cond[boi_idx:], uncond[boi_idx:])
    assert torch.equal(out.gen_image_mask[0], out.gen_image_mask[1])
    assert torch.equal(out.gen_timestep_scatter_index[0], out.gen_timestep_scatter_index[1])


def test_host_preprocess_bundle(hunyuan_tokenizer):
    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)
    assert bundle.input_ids.shape == (2, bundle.seq_len)
    assert bundle.position_ids.shape == bundle.input_ids.shape
    assert bundle.rope_image_info is not None
    assert len(bundle.rope_image_info) == 2
    slice_i, (th, tw) = bundle.rope_image_info[0][0]
    assert th * tw == 4096
    assert slice_i.stop - slice_i.start == 4096
    assert bundle.vae_image_mask is None
    assert bundle.cond_timestep_scatter_index is None


def test_i2i_cond_joint_layout(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    t2i = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cfg_factor=1)["output"]
    i2i = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cond_images=cond, cfg_factor=1)["output"]

    assert i2i.tokens.shape[1] > t2i.tokens.shape[1]
    assert i2i.vae_image_mask is not None
    assert i2i.vit_image_mask is not None
    assert i2i.joint_image_slices is not None
    assert len(i2i.joint_image_slices[0]) == 1
    assert i2i.cond_timestep_scatter_index is not None
    assert len(i2i.cond_timestep_scatter_index[0]) == 1

    vae_len = int(i2i.vae_image_mask[0].sum())
    vit_len = int(i2i.vit_image_mask[0].sum())
    joint_len = i2i.joint_image_slices[0][0].stop - i2i.joint_image_slices[0][0].start
    assert joint_len == vae_len + vit_len + 1  # includes <joint_img_sep>

    sep_id = hunyuan_tokenizer.special.joint_img_sep_token_id
    tokens = i2i.tokens[0].tolist()
    assert tokens.count(sep_id) == 1


def test_i2i_host_preprocess_bundle(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=1)

    assert bundle.batch_cond_images is not None
    assert len(bundle.batch_cond_images) == 1
    assert bundle.vae_image_mask is not None
    assert bundle.vit_image_mask is not None
    assert bundle.cond_timestep_scatter_index is not None
    assert bundle.rope_image_info is not None
    # cond_joint (vae + vit) + gen_image
    assert len(bundle.rope_image_info[0]) == 3


def test_i2i_cfg_preserves_cond_and_gen_blocks(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    out = hunyuan_tokenizer.apply_chat_template(PROMPT, image_size=IMAGE_SIZE, cond_images=cond, cfg_factor=2)["output"]
    assert out.tokens.shape[0] == 2
    cfg_id = hunyuan_tokenizer.special.cfg_token_id
    cond_row, uncond_row = out.tokens[0], out.tokens[1]
    assert (uncond_row == cfg_id).any()
    vae_slice = out.vae_image_slices[0][0]
    vit_slice = out.vit_image_slices[0][0]
    gen_slice = out.gen_image_slices[0][0]
    assert torch.equal(cond_row[vae_slice], uncond_row[vae_slice])
    assert torch.equal(cond_row[vit_slice], uncond_row[vit_slice])
    assert torch.equal(cond_row[gen_slice], uncond_row[gen_slice])
    assert len((cond_row == hunyuan_tokenizer.special.boi_token_id).nonzero(as_tuple=False)) == 2
    assert torch.equal(out.vae_image_mask[0], out.vae_image_mask[1])
    assert torch.equal(out.vit_image_mask[0], out.vit_image_mask[1])
    assert torch.equal(out.gen_image_mask[0], out.gen_image_mask[1])
    assert torch.equal(out.cond_timestep_scatter_index[0], out.cond_timestep_scatter_index[1])
