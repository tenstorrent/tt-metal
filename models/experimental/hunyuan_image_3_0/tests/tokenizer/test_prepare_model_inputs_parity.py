# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parity vs upstream ``prepare_model_inputs`` / attention-mask path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors import safe_open

from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    build_i2i_inputs_embeds,
    enrich_bundle_attention,
    prepare_gen_image_inputs,
    prepare_i2i_inputs,
)
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR, load_tensors

PROMPT = "a cat on a mat"
IMAGE_SIZE = 1024
HF_IMAGE_SIZE = "1024x1024"
MAX_LENGTH = 10000
CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"
UPSTREAM = Path("/home/iguser/ign-tt/hunyan_instruct")
HAS_WEIGHTS = (MODEL_DIR / "model.safetensors.index.json").is_file()
HAS_UPSTREAM = UPSTREAM.is_dir()

_ENCODE_PREFIXES = (
    "vae.",
    "vision_model.",
    "vision_aligner.",
    "patch_embed.",
    "time_embed.",
    "timestep_emb.",
    "model.wte.",
)


def _rope_pairs(info_row):
    return [(sl.start, sl.stop, hw) for sl, hw in info_row]


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (1024, 1024), color=(128, 64, 32))


@pytest.fixture(scope="module")
def hf_preprocess_model():
    """HF model shell: image_processor + tokenizer only (no backbone weights)."""
    if not HAS_UPSTREAM:
        pytest.skip("HF upstream repo not available")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM

    config = HunyuanImage3Config.from_pretrained(str(MODEL_DIR))
    model = HunyuanImage3ForCausalMM(config, skip_load_module={"all"})
    model.load_tokenizer(str(MODEL_DIR))
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_encode_model():
    """HF model with VAE/ViT/patch/timestep modules loaded (backbone skipped)."""
    if not HAS_WEIGHTS or not HAS_UPSTREAM:
        pytest.skip("HF upstream repo or Hunyuan checkpoint not available")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM

    config = HunyuanImage3Config.from_pretrained(str(MODEL_DIR))
    model = HunyuanImage3ForCausalMM(config, skip_load_module={"transformers"})
    model.load_tokenizer(str(MODEL_DIR))

    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    keys = [k for k in weight_map if k.startswith(_ENCODE_PREFIXES)]
    open_shards: dict[str, object] = {}
    state: dict[str, torch.Tensor] = {}
    for key in keys:
        shard_name = weight_map[key]
        if shard_name not in open_shards:
            open_shards[shard_name] = safe_open(MODEL_DIR / shard_name, framework="pt")
        state[key] = open_shards[shard_name].get_tensor(key)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _upstream_attention_mask(model, tokenizer_output, seq_len, bsz):
    batch_full_attn_slices = [model.image_processor.prepare_full_attn_slices(tokenizer_output, i) for i in range(bsz)]
    attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0).repeat(bsz, 1, 1)
    for i in range(bsz):
        for image_slice in batch_full_attn_slices[i]:
            attention_mask[i, image_slice, image_slice] = True
    return attention_mask.unsqueeze(1), batch_full_attn_slices


@pytest.mark.skipif(not HAS_UPSTREAM, reason="HF upstream repo required")
def test_t2i_prepare_model_inputs_tokenizer_parity(hf_preprocess_model, hunyuan_tokenizer, processor):
    hf_out = hf_preprocess_model.preprocess_inputs(
        prompt=PROMPT,
        mode="gen_image",
        image_size=HF_IMAGE_SIZE,
        cfg_factor=2,
        max_length=MAX_LENGTH,
    )
    hf_output = hf_out["output"]

    bundle = prepare_gen_image_inputs(hunyuan_tokenizer, PROMPT, image_size=IMAGE_SIZE, cfg_factor=2)
    enrich_bundle_attention(bundle, processor)

    assert torch.equal(bundle.input_ids, hf_output.tokens)
    assert torch.equal(bundle.gen_image_mask, hf_output.gen_image_mask)

    hf_rope = hf_preprocess_model.build_batch_rope_image_info(hf_output, hf_out["sections"])
    assert _rope_pairs(bundle.rope_image_info[0]) == _rope_pairs(hf_rope[0])

    hf_mask, hf_spans = _upstream_attention_mask(hf_preprocess_model, hf_output, bundle.seq_len, 2)
    assert torch.equal(bundle.attention_mask, hf_mask)
    assert bundle.full_attn_slices[0] == hf_spans[0]


@pytest.mark.skipif(not HAS_UPSTREAM, reason="HF upstream repo required")
def test_i2i_prepare_model_inputs_tokenizer_parity(hf_preprocess_model, hunyuan_tokenizer, processor, rgb_image):
    hf_out = hf_preprocess_model.preprocess_inputs(
        prompt=PROMPT,
        image=rgb_image,
        mode="gen_image",
        image_size=HF_IMAGE_SIZE,
        cfg_factor=2,
        max_length=MAX_LENGTH,
    )
    hf_output = hf_out["output"]

    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=2)
    enrich_bundle_attention(bundle, processor)

    assert torch.equal(bundle.input_ids, hf_output.tokens)
    assert torch.equal(bundle.vae_image_mask, hf_output.vae_image_mask)
    assert torch.equal(bundle.vit_image_mask, hf_output.vit_image_mask)
    assert torch.equal(bundle.cond_timestep_scatter_index, hf_output.cond_timestep_scatter_index)

    hf_rope = hf_preprocess_model.build_batch_rope_image_info(hf_output, hf_out["sections"])
    assert _rope_pairs(bundle.rope_image_info[0]) == _rope_pairs(hf_rope[0])

    hf_mask, hf_spans = _upstream_attention_mask(hf_preprocess_model, hf_output, bundle.seq_len, 2)
    assert torch.equal(bundle.attention_mask, hf_mask)
    assert bundle.full_attn_slices[0] == hf_spans[0]


@pytest.mark.skipif(not HAS_WEIGHTS or not HAS_UPSTREAM, reason="HF upstream + checkpoint required")
def test_i2i_prepare_model_inputs_encode_parity(hf_encode_model, hunyuan_tokenizer, processor, rgb_image):
    from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
        load_aligner,
        load_patch_embed,
        load_siglip2_vision,
        load_timestep_embedder,
    )

    hf_inputs = hf_encode_model.prepare_model_inputs(
        prompt=PROMPT,
        image=rgb_image,
        mode="gen_image",
        image_size=HF_IMAGE_SIZE,
        max_length=MAX_LENGTH,
    )
    hf_output = hf_inputs["tokenizer_output"]

    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    bundle = prepare_i2i_inputs(hunyuan_tokenizer, PROMPT, cond, image_size=IMAGE_SIZE, cfg_factor=2)
    wte = load_tensors(MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    gen = torch.Generator().manual_seed(int(hf_inputs["generator"][0].initial_seed()))

    bundle = build_i2i_inputs_embeds(
        bundle,
        wte,
        patch_embed=load_patch_embed(MODEL_DIR),
        time_embed=load_timestep_embedder("time_embed", MODEL_DIR),
        timestep_emb=load_timestep_embedder("timestep_emb", MODEL_DIR),
        vision_model=load_siglip2_vision(MODEL_DIR, num_layers=27),
        aligner=load_aligner(MODEL_DIR),
        generator=gen,
    )
    enrich_bundle_attention(bundle, processor)

    assert torch.equal(bundle.input_ids, hf_inputs["input_ids"].cpu())
    assert torch.equal(bundle.position_ids, hf_inputs["position_ids"].cpu())

    hf_mask, _ = _upstream_attention_mask(
        hf_encode_model, hf_output, bundle.seq_len, int(hf_inputs["input_ids"].shape[0])
    )
    assert torch.equal(bundle.attention_mask, hf_mask.cpu())

    assert isinstance(bundle.cond_vae_images, torch.Tensor)
    assert bundle.cond_vae_images.shape == hf_inputs["cond_vae_images"].shape
    assert torch.allclose(
        bundle.cond_vae_images.float(),
        hf_inputs["cond_vae_images"].cpu().float(),
        atol=1e-4,
        rtol=1e-4,
    )

    with torch.no_grad():
        hf_hidden = torch.nn.functional.embedding(hf_inputs["input_ids"], wte.float())
        hf_hidden = hf_encode_model.instantiate_vae_image_tokens(
            hf_hidden,
            hf_inputs["cond_timesteps"],
            hf_inputs["cond_vae_images"],
            hf_inputs["cond_vae_image_mask"],
        )
        hf_hidden = hf_encode_model.instantiate_continuous_tokens(
            hf_hidden,
            hf_inputs["cond_timesteps"],
            hf_inputs["cond_timesteps_index"],
        )
        hf_hidden = hf_encode_model.instantiate_vit_image_tokens(
            hf_hidden,
            hf_inputs["cond_vit_images"],
            hf_inputs["cond_vit_image_mask"],
            hf_inputs["cond_vit_image_kwargs"],
        )

    assert torch.allclose(
        bundle.inputs_embeds.float(),
        hf_hidden.cpu().float(),
        atol=1e-3,
        rtol=1e-3,
    )
