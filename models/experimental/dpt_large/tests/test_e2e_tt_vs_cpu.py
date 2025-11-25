# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

from models.experimental.dpt_large.config import DPTLargeConfig
from models.experimental.dpt_large.pipeline import DPTTTPipeline

transformers = pytest.importorskip("transformers")

RUN_TT = os.getenv("DPT_RUN_TT_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(not RUN_TT, reason="Set DPT_RUN_TT_TESTS=1 to enable TT tests.")

try:
    import ttnn  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    ttnn = None


def _make_dummy_image(path, size=96):
    arr = np.linspace(0, 255, num=size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    Image.fromarray(arr).save(path)


def test_tt_vs_cpu_pcc(tmp_path):
    if ttnn is None:
        pytest.skip("TTNN is not available; skipping TT vs CPU parity test.")

    img_path = tmp_path / "dummy.png"
    _make_dummy_image(img_path)

    # Small DPT config for fast TT vs CPU parity.
    cfg = DPTLargeConfig(
        image_size=96,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        device="wormhole_n300",
        enable_tt_device=True,
        allow_cpu_fallback=False,
    )
    pipeline = DPTTTPipeline(config=cfg, pretrained=False, device="cpu")

    # If we failed to acquire a TT device, skip rather than silently falling back.
    if getattr(pipeline.backbone, "tt_device", None) is None:
        pytest.skip("TT device not available; skipping TT vs CPU parity test.")

    cpu_depth = pipeline.fallback.run_depth_cpu(str(img_path), normalize=True)
    tt_depth = pipeline.forward(str(img_path), normalize=True)

    # Flatten to [H*W] for correlation metrics.
    cpu_flat = cpu_depth.flatten()
    tt_flat = tt_depth.flatten()

    pcc = np.corrcoef(cpu_flat, tt_flat)[0, 1]
    mae = np.mean(np.abs(cpu_flat - tt_flat))
    rmse = np.sqrt(np.mean((cpu_flat - tt_flat) ** 2))

    # Primary correctness guard with zero-variance handling (corrcoef returns NaN if both tensors are constant).
    if np.isnan(pcc) and mae < 1e-6 and rmse < 1e-6:
        pcc = 1.0
    assert pcc > 0.99, f"PCC too low: {pcc}, MAE={mae}, RMSE={rmse}"
