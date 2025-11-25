# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image

from models.experimental.dpt_large.config import DPTLargeConfig
from models.experimental.dpt_large.fallback import DPTFallbackPipeline

transformers = pytest.importorskip("transformers")


def _make_dummy_image(path, size=96):
    arr = np.linspace(0, 255, num=size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    Image.fromarray(arr).save(path)


def test_fallback_depth(tmp_path):
    img_path = tmp_path / "dummy.png"
    _make_dummy_image(img_path)
    cfg = DPTLargeConfig(
        image_size=96, hidden_size=256, intermediate_size=512, num_hidden_layers=2, num_attention_heads=4
    )
    pipe = DPTFallbackPipeline(config=cfg, pretrained=False, device="cpu")
    depth = pipe.run_depth_cpu(str(img_path))
    assert depth.shape == (1, 1, cfg.image_size, cfg.image_size)
    assert np.isfinite(depth).all()
