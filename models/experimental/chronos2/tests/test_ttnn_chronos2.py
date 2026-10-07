# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""TTNN backend vs the FP32 reference on synthetic series (needs a TT device).

  CHRONOS2_CHECKPOINT=/path/to/chronos-2 CHRONOS2_TEST_DEVICE=0 \
      pytest models/experimental/chronos2/tests/test_ttnn_chronos2.py
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# About twice the worst relative error measured for each precision.
REL_TOL = {"fp32": 5.0e-3, "bf16": 2.0e-2, "bfp8_b": 3.0e-2}
HORIZON = 64


def _series(B, T, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32)
    v = np.stack([10 + 3 * np.sin(2 * np.pi * t / 24 + b) + 0.3 * rng.standard_normal(T) for b in range(B)])
    return v.astype(np.float32), np.ones_like(v, dtype=np.float32)


@pytest.fixture(scope="module")
def reference(checkpoint_dir):
    from models.experimental.chronos2.reference.chronos2_reference import Chronos2ReferenceFP32

    return Chronos2ReferenceFP32(checkpoint_dir)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "bfp8_b"])
def test_forecast_matches_reference_and_repeats(precision, checkpoint_dir, tt_device, reference):
    import ttnn

    from models.experimental.chronos2.tt import create_backend

    backend = create_backend(checkpoint_dir, None, tt_device, precision=precision)
    try:
        for B, T, masked_tail in ((2, 512, 0), (1, 65, 0), (1, 512, 16)):
            v, m = _series(B, T, seed=T)
            if masked_tail:
                m[:, -masked_tail:] = 0.0
                v[:, -masked_tail:] = 0.0
            q1 = backend.forecast(v, m, HORIZON)["quantiles"]
            ttnn.synchronize_device(tt_device)
            q2 = backend.forecast(v, m, HORIZON)["quantiles"]
            ttnn.synchronize_device(tt_device)
            assert q1.shape == (B, HORIZON, backend.cfg.num_quantiles) and q1.dtype == np.float32
            assert np.isfinite(q1).all()
            assert np.array_equal(q1, q2), "repeated forecast must be bit-identical"
            ref = reference.predict(v, m, HORIZON)["quantiles"]
            rel = float(np.linalg.norm(q1 - ref) / max(np.linalg.norm(ref), 1e-30))
            assert rel < REL_TOL[precision], (B, T, masked_tail, rel)
        assert backend.executor.trace_stats["eager"] == 0
    finally:
        backend.release()
