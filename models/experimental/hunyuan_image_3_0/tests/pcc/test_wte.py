# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PCC: HunyuanTtWte vs torch F.embedding on a real prompt bundle.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_wte.py -v -s

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ttnn
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import prepare_gen_image_inputs
from models.experimental.hunyuan_image_3_0.ref.weights import MODEL_DIR
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte

PROMPT = "a photo of a cat"
PCC_THR = 0.999


def _load_wte(model_dir: Path) -> torch.Tensor:
    index = json.load(open(model_dir / "model.safetensors.index.json", encoding="utf-8"))["weight_map"]
    key = "model.wte.weight"
    with safe_open(model_dir / index[key], framework="pt") as f:
        return f.get_tensor(key)


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


@pytest.fixture(scope="function")
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_hunyuan_tt_wte_vs_torch(mesh_device):
    if not (MODEL_DIR / "model.safetensors.index.json").is_file():
        pytest.skip(f"weights not found at {MODEL_DIR}")

    mesh_device.enable_program_cache()
    tok = HunyuanTokenizer.from_pretrained()
    bundle = prepare_gen_image_inputs(tok, PROMPT, image_size=1024)
    input_ids = bundle.input_ids

    wte_w = _load_wte(MODEL_DIR)
    ref = F.embedding(input_ids, wte_w.float())

    wte_tt = HunyuanTtWte(
        mesh_device,
        wte_w,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = wte_tt.embedding_torch(input_ids)

    pcc = _pcc(ref, out)
    logger.info(f"HunyuanTtWte PCC={pcc:.6f} shape={tuple(out.shape)}")
    assert pcc >= PCC_THR, f"PCC {pcc:.6f} < {PCC_THR}"
