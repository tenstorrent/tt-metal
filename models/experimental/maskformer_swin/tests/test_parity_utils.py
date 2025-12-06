# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.experimental.maskformer_swin import parity


def test_ensure_pil_accepts_numpy_rgb():
    array = (np.random.rand(8, 6, 3) * 255).astype(np.float32)
    img = parity._ensure_pil(array)
    assert isinstance(img, Image.Image)
    assert img.size == (6, 8)


def test_ensure_pil_rejects_invalid_shapes():
    array = np.zeros((8, 6), dtype=np.uint8)
    with pytest.raises(ValueError):
        parity._ensure_pil(array)


def test_expand_image_paths_scans_directories(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"dummy")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.png").write_bytes(b"dummy")
    (nested / "ignore.txt").write_text("nope")

    paths = parity._expand_image_paths([str(root)])
    assert sorted(Path(p).name for p in paths) == ["a.jpg", "b.png"]


def test_compare_tensors_handles_zero_variance():
    ref = np.ones((2, 2), dtype=np.float32)
    test = np.ones((2, 2), dtype=np.float32)
    pcc, max_abs = parity.compare_tensors(ref, test, pcc_threshold=0.5, max_abs_threshold=1e-3)
    assert pcc == 1.0
    assert max_abs == 0.0


def test_compare_with_golden(tmp_path):
    golden_dir = tmp_path / "goldens"
    golden_dir.mkdir()
    sample = np.random.rand(2, 2).astype(np.float32)
    np.save(golden_dir / "tap.npy", sample)

    loaded = parity.load_golden_tensors(golden_dir)
    assert "tap" in loaded

    metrics = parity.compare_with_golden({"tap": sample.copy()}, loaded, config=parity.ParityConfig())
    assert "tap" in metrics
    pcc, max_abs = metrics["tap"]
    assert pytest.approx(pcc, rel=1e-5) == 1.0
    assert max_abs == 0.0


def test_compare_with_golden_missing_tap(tmp_path):
    golden_dir = tmp_path / "goldens"
    golden_dir.mkdir()
    np.save(golden_dir / "tap.npy", np.zeros((1,), dtype=np.float32))
    loaded = parity.load_golden_tensors(golden_dir)

    with pytest.raises(KeyError):
        parity.compare_with_golden({}, loaded, config=parity.ParityConfig())
