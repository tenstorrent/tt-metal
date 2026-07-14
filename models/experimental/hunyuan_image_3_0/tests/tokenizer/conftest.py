# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

TOKENIZER_DIR = Path(__file__).resolve().parent
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))

from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, MODEL_DIR
from tokenizer_helpers import CONFIG_PATH, HAS_INSTRUCT, HAS_UPSTREAM, HAS_WEIGHTS, ensure_upstream_in_path


@pytest.fixture(scope="session")
def hunyuan_tokenizer():
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_pretrained()


@pytest.fixture(scope="module")
def processor():
    return HunyuanImage3ImageProcessor(json.load(open(CONFIG_PATH)))


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (1024, 1024), color=(128, 64, 32))


@pytest.fixture(scope="module")
def instruct_tok():
    if not HAS_INSTRUCT:
        pytest.skip(f"Instruct checkpoint not found at {INSTRUCT_MODEL_DIR}")
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")


@pytest.fixture(scope="module")
def bundled_tok():
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_pretrained(sequence_template="instruct")


@pytest.fixture(scope="module")
def recaption_tok():
    from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer

    return HunyuanTokenizer.from_pretrained(sequence_template="instruct")


@pytest.fixture(scope="module")
def wte():
    if not HAS_INSTRUCT:
        pytest.skip("Instruct checkpoint not found")
    from models.experimental.hunyuan_image_3_0.ref.weights import load_tensors

    return load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]


@pytest.fixture(scope="module")
def hf_preprocess_model():
    if not HAS_UPSTREAM or not HAS_INSTRUCT:
        pytest.skip("HF upstream repo (HUNYUAN_UPSTREAM) or Instruct checkpoint not available")
    ensure_upstream_in_path()
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM

    config = HunyuanImage3Config.from_pretrained(str(INSTRUCT_MODEL_DIR))
    model = HunyuanImage3ForCausalMM(config, skip_load_module={"all"})
    model.load_tokenizer(str(INSTRUCT_MODEL_DIR))
    return model


@pytest.fixture(scope="module")
def hf_encode_model():
    if not HAS_WEIGHTS or not HAS_UPSTREAM:
        pytest.skip("HF upstream repo or Hunyuan checkpoint not available")
    ensure_upstream_in_path()
    import torch
    from safetensors import safe_open

    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
    from tokenizer_helpers import _ENCODE_PREFIXES

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
