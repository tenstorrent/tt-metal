import numpy as np
import torch
from models.experimental.opt_transfer.references.seamless_m4t_v2 import SeamlessBlock
from models.experimental.opt_transfer.verify import drift_metrics, golden_outputs, pcc


def test_pcc_identity_is_one():
    a = torch.randn(4, 4)
    assert pcc(a, a.clone()) > 0.999


def test_golden_outputs_runs_reference():
    blk = SeamlessBlock(1024, 16)
    x = torch.randn(1, 8, 1024)
    out = golden_outputs(blk, (x,))
    assert out.shape == (1, 8, 1024)


def test_drift_detects_late_divergence():
    T, V = 100, 50
    g = np.random.randn(T, V)
    f = g.copy()
    f[80:] = np.random.randn(20, V)
    m = drift_metrics(g, f)
    assert 70 <= m["first_divergence_step"] <= 85
    assert m["token_match_rate"] < 1.0


def test_drift_identical_trajectories():
    g = np.random.randn(40, 30)
    m = drift_metrics(g, g.copy())
    assert m["token_match_rate"] == 1.0
    assert m["first_divergence_step"] == 40
