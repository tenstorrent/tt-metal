"""PCC tests for CosyVoice2 flow encoder (Phase 2b).

Tests:
  1. Encoder mu PCC ≥ 0.99 vs golden fixtures (per-mode)
  2. spks PCC ≥ 0.99
  3. cond PCC ≥ 0.99

Run:
  pytest models/demos/cosyvoice/tests/pcc/test_flow_encoder.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.common.utility_functions import comp_pcc

DEMO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = DEMO_ROOT / "model_data" / "golden" / "flow"
FLOW_PT = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B" / "flow.pt"

MODES = ["zero_shot", "cross_lingual", "instruct2", "sft"]
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def flow_model():
    from models.demos.cosyvoice.tt.flow.flow_matching import FlowEncoderModel
    from models.demos.cosyvoice.tt.flow.weights import load_flow_weights

    components = load_flow_weights(str(FLOW_PT))
    model = FlowEncoderModel(components)
    model.eval()
    return model


@pytest.mark.parametrize("mode", MODES)
def test_flow_encoder_mu_pcc(flow_model, mode):
    """Encoder mu PCC ≥ 0.99 vs golden."""
    golden_path = GOLDEN_DIR / f"{mode}.pt"
    if not golden_path.exists():
        pytest.skip(f"Golden fixture not found: {golden_path}")

    g = torch.load(str(golden_path), map_location="cpu", weights_only=True)

    with torch.no_grad():
        mu, spks, conds = flow_model(
            g["token"],
            g["token_len"],
            g["prompt_token"],
            g["prompt_token_len"],
            g["prompt_feat"],
            g["prompt_feat_len"],
            g["embedding"],
        )

    passing, msg = comp_pcc(g["mu"], mu, PCC_THRESHOLD)
    assert passing, f"[{mode}] mu PCC failed: {msg}"

    passing, msg = comp_pcc(g["spks"], spks, PCC_THRESHOLD)
    assert passing, f"[{mode}] spks PCC failed: {msg}"

    passing, msg = comp_pcc(g["cond"], conds, PCC_THRESHOLD)
    assert passing, f"[{mode}] cond PCC failed: {msg}"
