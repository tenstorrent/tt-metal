# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parity for ``prepare_recaption_inputs`` vs upstream ``mode='gen_text'`` template."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    prepare_recaption_inputs,
    print_recaption_inputs_report,
)
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR

PROMPT = "make the sky more dramatic at sunset"
MAX_LENGTH = 10000
CONFIG_PATH = Path(__file__).resolve().parents[2] / "ref/tokenizer/assets/config.json"
UPSTREAM = Path("/home/iguser/ign-tt/hunyan_instruct")
HAS_UPSTREAM = UPSTREAM.is_dir()
HAS_INSTRUCT = (MODEL_DIR / "model.safetensors.index.json").is_file()


@pytest.fixture(scope="module")
def instruct_tok():
    if not HAS_INSTRUCT:
        pytest.skip(f"Instruct checkpoint not found at {MODEL_DIR}")
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_model_dir(MODEL_DIR, sequence_template="instruct")


@pytest.fixture(scope="module")
def bundled_tok():
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_pretrained(sequence_template="instruct")


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture(scope="module")
def hf_preprocess_model():
    if not HAS_UPSTREAM or not HAS_INSTRUCT:
        pytest.skip("HF upstream repo or Instruct checkpoint not available")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM

    config = HunyuanImage3Config.from_pretrained(str(MODEL_DIR))
    model = HunyuanImage3ForCausalMM(config, skip_load_module={"all"})
    model.load_tokenizer(str(MODEL_DIR))
    return model


def _upstream_recaption_ids(hf_model, *, prompt, system_prompt, image=None, bot_task="recaption"):
    out = hf_model.preprocess_inputs(
        prompt=prompt,
        image=image,
        mode="gen_text",
        bot_task=bot_task,
        system_prompt=system_prompt,
        cfg_factor=1,
        max_length=MAX_LENGTH,
    )
    return out["output"].tokens


def test_recaption_text_only_parity(bundled_tok, hf_preprocess_model):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_recaption", "recaption")
    bundle = prepare_recaption_inputs(
        bundled_tok,
        PROMPT,
        bot_task="recaption",
        system_prompt=system_prompt,
    )
    upstream = _upstream_recaption_ids(
        hf_preprocess_model,
        prompt=PROMPT,
        system_prompt=system_prompt,
        bot_task="recaption",
    )
    report = print_recaption_inputs_report(bundle, bundled_tok, upstream_ids=upstream, label="text_only_recaption")
    assert torch.equal(bundle.input_ids[0], upstream[0]), "input_ids mismatch vs upstream"
    assert report["recaption_inputs_ok"]
    assert bundle.mode == "gen_text"
    assert bundle.bot_task == "recaption"
    assert bundle.input_ids[0, -1].item() == bundled_tok.special.recaption_token_id


def test_recaption_i2i_with_cond_parity(instruct_tok, hf_preprocess_model, processor, rgb_image):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_unified", "image")
    cond, _ = processor.get_image_with_size(rgb_image, return_type="vae_vit")

    bundle = prepare_recaption_inputs(
        instruct_tok,
        PROMPT,
        cond_images=cond,
        bot_task="recaption",
        system_prompt=system_prompt,
    )
    upstream = _upstream_recaption_ids(
        hf_preprocess_model,
        prompt=PROMPT,
        system_prompt=system_prompt,
        image=rgb_image,
        bot_task="recaption",
    )
    report = print_recaption_inputs_report(bundle, instruct_tok, upstream_ids=upstream, label="i2i_recaption")
    assert torch.equal(bundle.input_ids[0], upstream[0])
    assert report["recaption_inputs_ok"]
    assert report["vit_placeholder_count"] > 0
    assert report["vae_placeholder_count"] > 0


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (1024, 1024), color=(128, 64, 32))


def test_think_prefix_ends_with_think_token(bundled_tok, hf_preprocess_model):
    from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

    system_prompt = get_system_prompt("en_think_recaption", "think_recaption")
    bundle = prepare_recaption_inputs(
        bundled_tok,
        PROMPT,
        bot_task="think",
        system_prompt=system_prompt,
    )
    upstream = _upstream_recaption_ids(
        hf_preprocess_model,
        prompt=PROMPT,
        system_prompt=system_prompt,
        bot_task="think",
    )
    report = print_recaption_inputs_report(bundle, bundled_tok, upstream_ids=upstream, label="think_prefix")
    assert torch.equal(bundle.input_ids[0], upstream[0])
    assert bundle.input_ids[0, -1].item() == bundled_tok.special.think_token_id
    assert report["recaption_inputs_ok"]
