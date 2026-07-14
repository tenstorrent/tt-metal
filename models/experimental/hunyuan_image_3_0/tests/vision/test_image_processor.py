# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from models.experimental.hunyuan_image_3_0.ref.image_processor import (
    CondImage,
    HunyuanImage3ImageProcessor,
    resize_and_crop,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer import load_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (800, 600), color=(128, 64, 32))


def test_init(processor):
    assert processor.vae_reso_group.base_size == 1024
    assert processor.vit_info.max_token_length == 1024
    assert processor.cond_image_type == "vae_vit"


def test_build_gen_image_info(processor):
    info = processor.build_gen_image_info((1024, 1024))
    assert info.token_width == info.token_height == 64
    assert info.image_width == info.image_height == 1024


def test_vae_process_image(processor, rgb_image):
    target = processor.vae_reso_group.get_target_size(*rgb_image.size)
    out = processor.vae_process_image(rgb_image, target)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 3
    assert out.shape[-2] % processor.vae_info.h_factor == 0
    assert out.i.image_type == "vae"
    assert out.section_type == "cond_vae_image"


def test_vit_process_image(processor, rgb_image):
    out = processor.vit_process_image(rgb_image)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 2
    assert out.i.image_type == "siglip2"
    assert "spatial_shapes" in out.vision_encoder_kwargs
    assert "pixel_attention_mask" in out.vision_encoder_kwargs


def test_get_image_with_size_vae_vit(processor, rgb_image):
    cond, ok = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    assert ok
    assert isinstance(cond, CondImage)
    assert cond.vae_image.i.image_type == "vae"
    assert cond.vit_image.i.image_type == "siglip2"
    assert cond.section_type == "cond_joint_image"


def test_build_cond_images_from_list(processor, rgb_image, tmp_path):
    path = tmp_path / "input.png"
    rgb_image.save(path)
    conds = processor.build_cond_images(image_list=[str(path)])
    assert len(conds) == 1
    assert isinstance(conds[0], CondImage)


def test_resize_and_crop_center():
    img = Image.new("RGB", (400, 200), color=(255, 0, 0))
    out = resize_and_crop(img, (256, 256), crop_type="center")
    assert out.size == (256, 256)


def test_build_gen_image_info_matches_standalone(processor):
    cfg = load_config(CONFIG_PATH)
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.image_info import build_gen_image_info

    standalone = build_gen_image_info(
        image_size=1024,
        image_base_size=cfg.image_base_size,
        vae_downsample_factor=cfg.vae_downsample_factor,
        vae_patch_size=cfg.vae_patch_size,
    )
    via_proc = processor.build_gen_image_info((1024, 1024))
    assert standalone.token_width == via_proc.token_width
    assert standalone.token_height == via_proc.token_height
