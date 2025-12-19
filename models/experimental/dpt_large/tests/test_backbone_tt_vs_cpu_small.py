# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

from models.experimental.dpt_large.tt.config import DPTLargeConfig
from models.experimental.dpt_large.tt.fallback import DPTFallbackPipeline
from models.experimental.dpt_large.tt.vit_backbone import DPTViTBackboneTTNN

transformers = pytest.importorskip("transformers")

RUN_TT_BACKBONE = os.getenv("DPT_RUN_TT_BACKBONE_PARITY", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_TT_BACKBONE, reason="Set DPT_RUN_TT_BACKBONE_PARITY=1 to enable TT backbone parity test."
)

try:
    import ttnn  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ttnn = None


def _make_dummy_image(path, size=96):
    arr = np.linspace(0, 255, num=size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    Image.fromarray(arr).save(path)


def test_backbone_tt_vs_cpu_small(tmp_path):
    if ttnn is None:
        pytest.skip("TTNN is not available; skipping TT backbone parity test.")

    img_path = tmp_path / "dummy.png"
    _make_dummy_image(img_path, size=96)

    # Small DPT config that matches the TT encoder path.
    cfg_cpu = DPTLargeConfig(
        image_size=96,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    cfg_tt = DPTLargeConfig(
        image_size=96,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        device="wormhole_n300",
        enable_tt_device=True,
    )

    # Shared HF model for both CPU and TT backbones.
    fallback = DPTFallbackPipeline(config=cfg_cpu, pretrained=False, device="cpu")
    pixel_values = fallback._prepare(str(img_path))

    backbone_cpu = DPTViTBackboneTTNN(config=cfg_cpu, hf_model=fallback._model, pretrained=False, device="cpu")
    backbone_cpu.tt_device = None  # force CPU path

    backbone_tt = DPTViTBackboneTTNN(config=cfg_tt, hf_model=fallback._model, pretrained=False, device="cpu")
    if getattr(backbone_tt, "tt_device", None) is None:
        pytest.skip("TT device not available; skipping TT backbone parity test.")

    out_cpu = backbone_cpu(pixel_values, return_tt=False)
    out_tt = backbone_tt(pixel_values, return_tt=False)

    # Ensure we actually exercised the TT encoder path.
    assert (
        getattr(backbone_tt, "used_tt_encoder_last_forward", False) is True
    ), "TT encoder path was not used in backbone_tt."

    # Compare per-layer backbone feature maps using the keys produced by the
    # backbone implementation (1-based indices matching HF hidden_states).
    for layer_idx, feat_cpu in out_cpu.features.items():
        feat_tt = out_tt.features[layer_idx]  # [B, C, H, W] torch (converted from TT internally)

        cpu_flat = feat_cpu.detach().float().cpu().flatten().numpy()
        tt_flat = feat_tt.detach().float().cpu().flatten().numpy()

        pcc = np.corrcoef(cpu_flat, tt_flat)[0, 1]
        mae = np.mean(np.abs(cpu_flat - tt_flat))
        rmse = np.sqrt(np.mean((cpu_flat - tt_flat) ** 2))

        assert pcc > 0.99, f"Backbone PCC too low at layer {layer_idx}: {pcc}, MAE={mae}, RMSE={rmse}"
