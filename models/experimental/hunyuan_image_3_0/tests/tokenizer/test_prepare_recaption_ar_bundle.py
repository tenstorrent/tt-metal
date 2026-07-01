# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parity for ``prepare_recaption_ar_bundle`` (template + cond encode + mask)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
    load_aligner,
    load_patch_embed,
    load_siglip2_vision,
    load_timestep_embedder,
)
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    enrich_bundle_attention,
    prepare_recaption_ar_bundle,
    prepare_recaption_inputs,
    print_recaption_inputs_report,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import build_i2i_inputs_embeds
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors

PROMPT = "make the sky more dramatic at sunset"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"
UPSTREAM = Path("/home/iguser/ign-tt/hunyan_instruct")
HAS_INSTRUCT = (INSTRUCT_MODEL_DIR / "model.safetensors.index.json").is_file()
HAS_UPSTREAM = UPSTREAM.is_dir()
VIT_LAYERS = 1

if HAS_UPSTREAM and str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))


@pytest.fixture(scope="module")
def instruct_tok():
    if not HAS_INSTRUCT:
        pytest.skip(f"Instruct checkpoint not found at {INSTRUCT_MODEL_DIR}")
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture(scope="module")
def wte():
    if not HAS_INSTRUCT:
        pytest.skip("Instruct checkpoint not found")
    return load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (1024, 1024), color=(128, 64, 32))


def _manual_ar_bundle(
    tok, prompt, proc, wte, *, cond_images=None, bot_task="recaption", system_prompt=None, generator=None
):
    bundle = prepare_recaption_inputs(
        tok,
        prompt,
        cond_images=cond_images,
        bot_task=bot_task.split("_")[0] if bot_task == "think_recaption" else bot_task,
        system_prompt=system_prompt,
        sequence_template=None,
    )
    if cond_images is not None:
        bundle = build_i2i_inputs_embeds(
            bundle,
            wte,
            patch_embed=load_patch_embed(INSTRUCT_MODEL_DIR),
            time_embed=load_timestep_embedder("time_embed", INSTRUCT_MODEL_DIR),
            timestep_emb=load_timestep_embedder("timestep_emb", INSTRUCT_MODEL_DIR),
            vision_model=load_siglip2_vision(INSTRUCT_MODEL_DIR, num_layers=VIT_LAYERS),
            aligner=load_aligner(INSTRUCT_MODEL_DIR),
            model_dir=INSTRUCT_MODEL_DIR,
            generator=generator,
            vit_num_layers=VIT_LAYERS,
        )
    else:
        import torch.nn.functional as F

        bundle.inputs_embeds = F.embedding(bundle.input_ids, wte.float())
    bundle.bot_task = bot_task
    return enrich_bundle_attention(bundle, proc)


@pytest.mark.skipif(not HAS_INSTRUCT, reason="Instruct checkpoint required")
def test_recaption_ar_bundle_text_only_matches_manual(instruct_tok, processor, wte):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_recaption", "recaption")
    manual = _manual_ar_bundle(instruct_tok, PROMPT, processor, wte, bot_task="recaption", system_prompt=system_prompt)
    bundle = prepare_recaption_ar_bundle(
        instruct_tok,
        PROMPT,
        processor,
        wte,
        bot_task="recaption",
        system_prompt=system_prompt,
        sequence_template=None,
    )
    assert torch.equal(bundle.input_ids, manual.input_ids)
    assert torch.allclose(bundle.inputs_embeds, manual.inputs_embeds, atol=1e-4, rtol=1e-4)
    assert bundle.full_attn_slices == manual.full_attn_slices
    assert torch.equal(bundle.attention_mask, manual.attention_mask)


@pytest.mark.skipif(not HAS_INSTRUCT, reason="Instruct checkpoint required")
def test_recaption_ar_bundle_i2i_cond_matches_manual(instruct_tok, processor, wte, rgb_image):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_unified", "image")
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    gen_manual = torch.Generator().manual_seed(42)
    gen_bundle = torch.Generator().manual_seed(42)
    manual = _manual_ar_bundle(
        instruct_tok,
        PROMPT,
        processor,
        wte,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
        generator=gen_manual,
    )
    bundle = prepare_recaption_ar_bundle(
        instruct_tok,
        PROMPT,
        processor,
        wte,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
        sequence_template=None,
        patch_embed=load_patch_embed(INSTRUCT_MODEL_DIR),
        time_embed=load_timestep_embedder("time_embed", INSTRUCT_MODEL_DIR),
        timestep_emb=load_timestep_embedder("timestep_emb", INSTRUCT_MODEL_DIR),
        vision_model=load_siglip2_vision(INSTRUCT_MODEL_DIR, num_layers=VIT_LAYERS),
        aligner=load_aligner(INSTRUCT_MODEL_DIR),
        model_dir=INSTRUCT_MODEL_DIR,
        generator=gen_bundle,
    )
    report = print_recaption_inputs_report(bundle, instruct_tok, label="ar_bundle_i2i")
    assert report["vit_placeholder_count"] > 0
    assert report["vae_placeholder_count"] > 0
    assert torch.equal(bundle.input_ids, manual.input_ids)
    assert torch.allclose(bundle.inputs_embeds, manual.inputs_embeds, atol=1e-3, rtol=1e-3)
    assert bundle.rope_image_info == manual.rope_image_info
    assert torch.equal(bundle.attention_mask, manual.attention_mask)


@pytest.mark.skipif(not (HAS_INSTRUCT and HAS_UPSTREAM), reason="Instruct + upstream required")
def test_recaption_ar_bundle_ids_match_upstream_preprocess(instruct_tok, processor, wte, rgb_image):
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_unified", "image")
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")
    config = HunyuanImage3Config.from_pretrained(str(INSTRUCT_MODEL_DIR))
    hf_model = HunyuanImage3ForCausalMM(config, skip_load_module={"all"})
    hf_model.load_tokenizer(str(INSTRUCT_MODEL_DIR))
    upstream_ids = hf_model.preprocess_inputs(
        prompt=PROMPT,
        image=rgb_image,
        mode="gen_text",
        bot_task="recaption",
        system_prompt=system_prompt,
        cfg_factor=1,
        max_length=10000,
    )["output"].tokens

    bundle = prepare_recaption_ar_bundle(
        instruct_tok,
        PROMPT,
        processor,
        wte,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
        sequence_template=None,
        patch_embed=load_patch_embed(INSTRUCT_MODEL_DIR),
        time_embed=load_timestep_embedder("time_embed", INSTRUCT_MODEL_DIR),
        timestep_emb=load_timestep_embedder("timestep_emb", INSTRUCT_MODEL_DIR),
        vision_model=load_siglip2_vision(INSTRUCT_MODEL_DIR, num_layers=VIT_LAYERS),
        aligner=load_aligner(INSTRUCT_MODEL_DIR),
        model_dir=INSTRUCT_MODEL_DIR,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(bundle.input_ids[0], upstream_ids[0])
    assert bundle.inputs_embeds is not None
    assert bundle.full_attn_slices is not None
