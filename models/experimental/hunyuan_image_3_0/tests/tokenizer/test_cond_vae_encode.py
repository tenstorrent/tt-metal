# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for I2I cond VAE/ViT encode + scatter into inputs_embeds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from models.experimental.hunyuan_image_3_0.ref.cond_vae_encode import encode_cond_images, vae_encode_image
from models.experimental.hunyuan_image_3_0.ref.cond_vit_encode import encode_cond_vit_images
from models.experimental.hunyuan_image_3_0.ref.image_gen.input_instantiate import (
    instantiate_continuous_tokens,
    instantiate_vae_image_tokens,
    instantiate_vit_image_tokens,
)
from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
    load_aligner,
    load_patch_embed,
    load_siglip2_vision,
    load_timestep_embedder,
)
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    build_i2i_inputs_embeds,
    prepare_i2i_inputs,
)
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import load_encoder
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_tensors

CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"
PROMPT = "a cat on a mat"
HAS_WEIGHTS = (MODEL_DIR / "model.safetensors.index.json").is_file()


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (1024, 1024), color=(128, 64, 32))


@pytest.mark.skipif(not HAS_WEIGHTS, reason="Hunyuan checkpoint not available")
def test_vae_encode_cond_image_shape(processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    encoder = load_encoder(MODEL_DIR, dtype=torch.float32)
    t, latents = vae_encode_image(encoder, cond.vae_image, scaling_factor=0.562679178327931)
    assert t.shape == (1,)
    assert torch.all(t == 0)
    assert latents.shape == (1, 32, 64, 64)


@pytest.mark.skipif(not HAS_WEIGHTS, reason="Hunyuan checkpoint not available")
def test_encode_cond_images_batched(processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    encoder = load_encoder(MODEL_DIR, dtype=torch.float32)
    out = encode_cond_images([[cond]], encoder, cfg_factor=1, scaling_factor=0.562679178327931)
    assert isinstance(out.cond_vae_images, torch.Tensor)
    assert out.cond_vae_images.shape == (1, 32, 64, 64)
    assert out.cond_timesteps.shape == (1,)


def test_instantiate_vae_image_tokens_synthetic():
    torch.manual_seed(0)
    bsz, seqlen, hidden = 1, 20, 32
    from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown
    from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder

    patch_embed = UNetDown(1, 32, hidden, hidden, hidden)
    time_embed = TimestepEmbedder(hidden_size=hidden)
    hidden_states = torch.randn(bsz, seqlen, hidden)
    base = hidden_states.clone()
    images = torch.randn(bsz, 32, 2, 2)
    timesteps = torch.zeros(bsz)
    mask = torch.zeros(bsz, seqlen, dtype=torch.bool)
    mask[0, 5:9] = True

    out = instantiate_vae_image_tokens(hidden_states, timesteps, images, mask, patch_embed, time_embed)
    assert not torch.equal(out, base)
    assert torch.equal(out[~mask], base[~mask])
    assert not torch.equal(out[mask], base[mask])


@pytest.mark.skipif(not HAS_WEIGHTS, reason="Hunyuan checkpoint not available")
def test_encode_cond_vit_images_packed(processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    out = encode_cond_vit_images([[cond]], cfg_factor=1)
    assert isinstance(out.cond_vit_images, list)
    assert len(out.cond_vit_images) == 1
    assert out.cond_vit_images[0].ndim == 3
    assert out.cond_vit_image_kwargs is not None
    assert "spatial_shapes" in out.cond_vit_image_kwargs
    assert "attention_mask" in out.cond_vit_image_kwargs
    assert out.cond_vit_image_kwargs["spatial_shapes"][0].ndim == 2


@pytest.mark.skipif(not HAS_WEIGHTS, reason="Hunyuan checkpoint not available")
def test_instantiate_vit_image_tokens_checkpoint(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=1024, cfg_factor=1)
    vit_enc = encode_cond_vit_images([[cond]], cfg_factor=1)
    vision_model = load_siglip2_vision(MODEL_DIR, num_layers=1)
    aligner = load_aligner(MODEL_DIR)

    hidden = torch.randn(1, bundle.seq_len, 4096)
    base = hidden.clone()
    mask = bundle.vit_image_mask

    out = instantiate_vit_image_tokens(
        hidden,
        vit_enc.cond_vit_images,
        mask,
        vit_enc.cond_vit_image_kwargs,
        vision_model,
        aligner,
    )
    assert torch.equal(out[~mask], base[~mask])
    assert not torch.equal(out[mask], base[mask])


@pytest.mark.skipif(not HAS_WEIGHTS, reason="Hunyuan checkpoint not available")
def test_build_i2i_inputs_embeds(hunyuan_tokenizer, processor, rgb_image):
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=1024, cfg_factor=1)
    wte = load_tensors(MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    patch_embed = load_patch_embed(MODEL_DIR)
    time_embed = load_timestep_embedder("time_embed", MODEL_DIR)
    timestep_emb = load_timestep_embedder("timestep_emb", MODEL_DIR)

    gen = torch.Generator().manual_seed(42)
    bundle = build_i2i_inputs_embeds(
        bundle,
        wte,
        patch_embed=patch_embed,
        time_embed=time_embed,
        timestep_emb=timestep_emb,
        vision_model=load_siglip2_vision(MODEL_DIR, num_layers=1),
        aligner=load_aligner(MODEL_DIR),
        generator=gen,
        vit_num_layers=1,
    )

    assert bundle.inputs_embeds is not None
    assert bundle.inputs_embeds.shape == (1, bundle.seq_len, wte.shape[1])
    assert bundle.cond_vae_images is not None
    assert bundle.cond_timesteps is not None
    assert bundle.cond_vit_images is not None
    assert bundle.cond_vit_image_kwargs is not None

    vae_mask = bundle.vae_image_mask[0]
    vit_mask = bundle.vit_image_mask[0]
    wte_only = torch.nn.functional.embedding(bundle.input_ids[0], wte.float())
    assert not torch.allclose(bundle.inputs_embeds[0, vae_mask], wte_only[vae_mask])
    assert not torch.allclose(bundle.inputs_embeds[0, vit_mask], wte_only[vit_mask])

    ts_idx = bundle.cond_timestep_scatter_index[0, 0].item()
    assert not torch.allclose(
        bundle.inputs_embeds[0, ts_idx],
        wte_only[ts_idx],
        atol=1e-4,
    )

    out = instantiate_continuous_tokens(
        wte_only.unsqueeze(0).clone(),
        bundle.cond_timesteps,
        bundle.cond_timestep_scatter_index,
        timestep_emb,
    )
    assert torch.allclose(out[0, ts_idx], bundle.inputs_embeds[0, ts_idx], atol=1e-4)
