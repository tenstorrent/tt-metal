# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the RMS helpers in torch_impl/vc/pipeline.py.

These guard `_rms_numpy` against the reduction-axis regression (a previous
version averaged over n_frames instead of frame_length, returning the wrong
shape) by comparing it to `librosa.feature.rms`, and confirm `change_rms`
preserves the target-audio shape. Pure CPU/NumPy — no TTNN device needed,
so this file is safe to run on its own or alongside the device tests.
"""

import numpy as np
import pytest
import torch

librosa = pytest.importorskip("librosa")

from models.demos.rvc.torch_impl.vc.pipeline import _rms_numpy, change_rms


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("n_samples", [4096, 9000])
def test_rms_numpy_matches_librosa(batch, n_samples):
    """_rms_numpy must match librosa.feature.rms in both shape and value."""
    rng = np.random.default_rng(0)
    y = rng.standard_normal((batch, n_samples)).astype(np.float32)
    frame_length, hop_length = 2048, 512

    ours = _rms_numpy(y, frame_length=frame_length, hop_length=hop_length)

    ref = np.stack(
        [
            librosa.feature.rms(
                y=y[i], frame_length=frame_length, hop_length=hop_length
            )[0]
            for i in range(batch)
        ],
        axis=0,
    )

    # Shape must be (batch, n_frames) — the regression returned (batch, frame_length).
    assert ours.shape == ref.shape, f"shape mismatch: {ours.shape} vs {ref.shape}"
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-4)


def test_change_rms_preserves_shape():
    """change_rms must return audio with the same shape as the target input."""
    rng = np.random.default_rng(1)
    src = torch.from_numpy(rng.standard_normal((1, 16000)).astype(np.float32))
    tgt = torch.from_numpy(rng.standard_normal((1, 24000)).astype(np.float32))

    # change_rms scales target_audio in place; pass a clone so tgt stays a
    # reference for the shape assertion.
    out = change_rms(src, 16000, tgt.clone(), 24000, rate=0.5)

    assert out.shape == tgt.shape, f"shape changed: {out.shape} vs {tgt.shape}"
    assert torch.isfinite(out).all(), "non-finite values in change_rms output"
