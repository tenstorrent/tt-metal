# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

from models.experimental.dpt_large.config import DPTLargeConfig
from models.experimental.dpt_large.fallback import DPTFallbackPipeline
from models.experimental.dpt_large.vit_backbone import DPTViTBackboneTTNN
from models.experimental.dpt_large.reassembly import DPTReassembly
from models.experimental.dpt_large.fusion_head import DPTFusionHead

transformers = pytest.importorskip("transformers")

RUN_PARITY = os.getenv("DPT_RUN_NECK_HEAD_PARITY", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_PARITY,
    reason="Set DPT_RUN_NECK_HEAD_PARITY=1 to enable DPT neck/head CPU parity test.",
)


def _make_dummy_image(path, size=96):
    arr = np.linspace(0, 255, num=size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    Image.fromarray(arr).save(path)


def test_neck_head_cpu_parity_vs_hf(tmp_path):
    """Compare HF DPT pipeline vs HF backbone + DPTReassembly + DPTFusionHead on CPU."""

    img_path = tmp_path / "dummy.png"
    _make_dummy_image(img_path)

    cfg = DPTLargeConfig(
        image_size=96,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )

    # HF reference pipeline (full DPT: backbone + neck + head).
    fallback = DPTFallbackPipeline(config=cfg, pretrained=False, device="cpu")
    depth_ref = fallback.run_depth_cpu(str(img_path), normalize=True)  # [1,1,H,W]

    # Our reimplementation path: HF backbone + DPTReassembly + DPTFusionHead (CPU only).
    backbone = DPTViTBackboneTTNN(config=cfg, hf_model=fallback._model, pretrained=False, device="cpu")
    # Ensure we don't open a TT device in this test.
    backbone.tt_device = None
    backbone.to("cpu").eval()

    neck = DPTReassembly(config=cfg, tt_device=None, layer_cfg=None).to("cpu").eval()
    head = DPTFusionHead(config=cfg, tt_device=None, layer_cfg=None).to("cpu").eval()

    state_dict = fallback._model.state_dict()
    if hasattr(neck, "load_from_hf_state_dict"):
        neck.load_from_hf_state_dict(state_dict)
    if hasattr(head, "load_from_hf_state_dict"):
        head.load_from_hf_state_dict(state_dict)

    with torch.no_grad():
        pixel_values = fallback._prepare(str(img_path))
        feats = backbone(pixel_values, return_tt=False)
        pyramid = neck(feats)
        depth_our = head(pyramid)
        depth_our = fallback._normalize_depth(torch.as_tensor(depth_our).float())

    cpu_flat = depth_ref.flatten()
    our_flat = depth_our.cpu().numpy().flatten()

    pcc = np.corrcoef(cpu_flat, our_flat)[0, 1]
    mae = np.mean(np.abs(cpu_flat - our_flat))
    rmse = np.sqrt(np.mean((cpu_flat - our_flat) ** 2))

    # Encode strict parity: HF neck/head and our implementation must be
    # numerically identical up to tiny numerical noise.
    assert pcc > 0.99, f"PCC too low vs HF: {pcc}, MAE={mae}, RMSE={rmse}"
    assert mae < 1e-4, f"MAE too high vs HF: {mae}, PCC={pcc}, RMSE={rmse}"
    assert rmse < 1e-4, f"RMSE too high vs HF: {rmse}, PCC={pcc}, MAE={mae}"
