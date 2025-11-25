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
from models.experimental.dpt_large.pipeline import DPTTTPipeline

transformers = pytest.importorskip("transformers")

RUN_TT_LARGE = os.getenv("DPT_RUN_TT_LARGE_PARITY", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_TT_LARGE, reason="Set DPT_RUN_TT_LARGE_PARITY=1 to enable large-config TT parity tests."
)

try:
    import ttnn  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ttnn = None


def _make_dummy_image(path, size=384):
    # Simple deterministic gradient image
    arr = np.linspace(0, 255, num=size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    Image.fromarray(arr).save(path)


def test_vit_large_backbone_tt_vs_cpu_features(tmp_path):
    if ttnn is None:
        pytest.skip("TTNN is not available; skipping TT large backbone parity test.")

    img_path = tmp_path / "dummy_384.png"
    _make_dummy_image(img_path, size=384)

    # Full DPT-Large config
    cfg_cpu = DPTLargeConfig(
        image_size=384,
        patch_size=16,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
    )
    cfg_tt = DPTLargeConfig(
        image_size=384,
        patch_size=16,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        device="wormhole_n300",
        enable_tt_device=True,
        allow_cpu_fallback=False,
    )

    # Shared HF model for CPU and TT backbones.
    fallback = DPTFallbackPipeline(config=cfg_cpu, pretrained=False, device="cpu")
    pixel_values = fallback._prepare(str(img_path))

    backbone_cpu = DPTViTBackboneTTNN(config=cfg_cpu, hf_model=fallback._model, pretrained=False, device="cpu")
    backbone_cpu.tt_device = None  # ensure CPU path

    backbone_tt = DPTViTBackboneTTNN(config=cfg_tt, hf_model=fallback._model, pretrained=False, device="cpu")
    if getattr(backbone_tt, "tt_device", None) is None:
        pytest.skip("TT device not available; skipping TT large backbone parity test.")

    out_cpu = backbone_cpu(pixel_values, return_tt=False)
    out_tt = backbone_tt(pixel_values, return_tt=False)

    # Ensure TT encoder path exercised.
    assert getattr(backbone_tt, "used_tt_encoder_last_forward", False) is True

    for layer_idx, feat_cpu in out_cpu.features.items():
        feat_tt = out_tt.features[layer_idx]
        cpu_flat = feat_cpu.detach().float().cpu().flatten().numpy()
        tt_flat = feat_tt.detach().float().cpu().flatten().numpy()
        pcc = np.corrcoef(cpu_flat, tt_flat)[0, 1]
        mae = np.mean(np.abs(cpu_flat - tt_flat))
        rmse = np.sqrt(np.mean((cpu_flat - tt_flat) ** 2))
        assert pcc > 0.99, f"Backbone PCC too low at layer {layer_idx}: {pcc}, MAE={mae}, RMSE={rmse}"


def test_vit_large_e2e_tt_vs_cpu_depth(tmp_path):
    if ttnn is None:
        pytest.skip("TTNN is not available; skipping TT large e2e parity test.")

    img_path = tmp_path / "dummy_384.png"
    _make_dummy_image(img_path, size=384)

    cfg_tt = DPTLargeConfig(
        image_size=384,
        patch_size=16,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        device="wormhole_n300",
        enable_tt_device=True,
        allow_cpu_fallback=False,
    )

    pipeline = DPTTTPipeline(config=cfg_tt, pretrained=False, device="cpu")
    if getattr(pipeline.backbone, "tt_device", None) is None:
        pytest.skip("TT device not available; skipping TT large e2e parity test.")

    # CPU reference via the same HF model used by the TT pipeline.
    cpu_depth = pipeline.fallback.run_depth_cpu(str(img_path), normalize=True)
    tt_depth = pipeline.forward(str(img_path), normalize=True)

    cpu_flat = cpu_depth.flatten()
    tt_flat = tt_depth.flatten()
    pcc = np.corrcoef(cpu_flat, tt_flat)[0, 1]
    mae = np.mean(np.abs(cpu_flat - tt_flat))
    rmse = np.sqrt(np.mean((cpu_flat - tt_flat) ** 2))

    assert pcc > 0.99, f"E2E PCC too low: {pcc}, MAE={mae}, RMSE={rmse}"
