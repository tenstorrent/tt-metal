# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal profiler test: single TTNN estimator forward (stays under 1000-op device buffer)."""
import sys
from pathlib import Path

import pytest
import torch

DEMO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden"


@pytest.fixture(scope="module")
def estimator_device():
    import ttnn

    device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)

    sys.path.insert(0, str(DEMO_ROOT))
    from models.demos.cosyvoice.tt.flow.estimator_ttnn import UNetEstimatorTtnn
    from models.demos.cosyvoice.tt.flow.weights import load_flow_weights

    fw = load_flow_weights(CKPT_DIR / "flow.pt")
    est = UNetEstimatorTtnn(fw["decoder"], device)
    yield est, device
    ttnn.close_device(device)


def test_estimator_profiled(estimator_device):
    est, device = estimator_device

    golden = torch.load(str(GOLDEN_DIR / "flow" / "zero_shot.pt"), map_location="cpu", weights_only=True)
    mu, mask, spks, cond = golden["mu"], golden["mask"], golden["spks"], golden["cond"]

    x_in = torch.zeros([2, 80, mu.shape[2]])
    x_in[:] = torch.randn_like(mu)
    mask_in = torch.zeros([2, 1, mu.shape[2]])
    mask_in[:] = mask
    mu_in = torch.zeros([2, 80, mu.shape[2]])
    mu_in[0] = mu
    spks_in = torch.zeros([2, 80])
    spks_in[0] = spks
    cond_in = torch.zeros([2, 80, mu.shape[2]])
    cond_in[0] = cond
    t = torch.tensor([0.0, 0.0])

    out = est.forward(x_in, mask_in, mu_in, t, spks_in, cond_in, streaming=False)
    assert out.shape == (2, 80, mu.shape[2])
