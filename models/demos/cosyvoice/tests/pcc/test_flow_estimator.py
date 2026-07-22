"""PCC tests for CosyVoice2 flow estimator + CFM (Phase 2b).

Tests:
  1. Estimator dphi_dt PCC ≥ 0.99 (per-step, step 0)
  2. CFM Euler loop mel PCC ≥ 0.99 (final output)

Run:
  pytest models/demos/cosyvoice/tests/pcc/test_flow_estimator.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

DEMO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden" / "flow"
FLOW_PT = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B" / "flow.pt"

_COSYVOICE_SRC = str(DEMO_ROOT / "model_data" / "CosyVoice_src")
_MATCHA = str(DEMO_ROOT / "model_data" / "CosyVoice_src" / "third_party" / "Matcha-TTS")
if _COSYVOICE_SRC not in sys.path:
    sys.path.insert(0, _COSYVOICE_SRC)
if _MATCHA not in sys.path:
    sys.path.append(_MATCHA)

from models.common.utility_functions import comp_pcc

MODES = ["zero_shot", "cross_lingual", "instruct2", "sft"]
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def estimator():
    from models.demos.cosyvoice.tt.flow.unet_estimator import UNetEstimator
    from models.demos.cosyvoice.tt.flow.weights import load_flow_weights

    components = load_flow_weights(str(FLOW_PT))
    return UNetEstimator(components["decoder"])


@pytest.fixture(scope="module")
def cfm(estimator):
    from models.demos.cosyvoice.tt.flow.cfm import CausalConditionalCFM

    return CausalConditionalCFM(estimator)


@pytest.mark.parametrize("mode", MODES)
def test_estimator_dphi_dt_pcc(estimator, mode):
    """Estimator dphi_dt PCC ≥ 0.99 at step 0."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=True)
    x_init = g["x_init"]
    mu = g["mu"]
    mask = g["mask"]
    spks = g["spks"]
    cond = g["cond"]
    t_span = g["t_span"]
    golden_dphi_dt = g["dphi_dt"]

    t = t_span[0].unsqueeze(0)
    dt = t_span[1] - t_span[0]

    x_in = torch.zeros([2, 80, x_init.size(2)], dtype=spks.dtype)
    mask_in = torch.zeros([2, 1, x_init.size(2)], dtype=spks.dtype)
    mu_in = torch.zeros([2, 80, x_init.size(2)], dtype=spks.dtype)
    t_in = torch.zeros([2], dtype=spks.dtype)
    spks_in = torch.zeros([2, 80], dtype=spks.dtype)
    cond_in = torch.zeros([2, 80, x_init.size(2)], dtype=spks.dtype)

    x_in[:] = x_init
    mask_in[:] = mask
    mu_in[0] = mu
    t_in[:] = t.unsqueeze(0)
    spks_in[0] = spks
    cond_in[0] = cond

    with torch.no_grad():
        dphi_dt = estimator.forward(x_in, mask_in, mu_in, t_in, spks_in, cond_in)

    passing, msg = comp_pcc(golden_dphi_dt[0], dphi_dt, PCC_THRESHOLD)
    assert passing, f"[{mode}] dphi_dt[0] PCC failed: {msg}"


@pytest.mark.parametrize("mode", MODES)
def test_cfm_mel_pcc(cfm, mode):
    """CFM Euler loop mel PCC ≥ 0.99."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=True)

    with torch.no_grad():
        mel = cfm.solve_euler(g["x_init"], g["t_span"], g["mu"], g["mask"], g["spks"], g["cond"])

    passing, msg = comp_pcc(g["mel"], mel, PCC_THRESHOLD)
    assert passing, f"[{mode}] mel PCC failed: {msg}"
