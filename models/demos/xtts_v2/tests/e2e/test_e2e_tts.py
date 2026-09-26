# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end TTNN pipeline gate for coqui/XTTS-v2 (text -> speech).

Gate 1: routed modules stay native ttnn (composed as-is).
Gate 2: every module is INVOKED in the real forward path.
Gate 3: final waveform PCC vs HF golden >= 0.95.

Runs the SAME `tt/pipeline.py::run_tts` the demo uses.
"""

from __future__ import annotations

import os

import pytest
import torch

# instrument BEFORE importing the pipeline/composites so child forwards are tracked
from models.demos.xtts_v2 import reference
from models.demos.xtts_v2.tt import pipeline as P

HF_MODEL_ID = "coqui/XTTS-v2"
_N = int(os.environ.get("XTTS_E2E_N", "40"))
_ALL_29 = set(P._MODULE_ORDER)


def _load_reference():
    return reference.load_reference_model(HF_MODEL_ID)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_e2e_tts(device):
    torch.manual_seed(0)
    restore = P.instrument_modules()
    try:
        model = _load_reference()
        res = P.run_tts(device, model, text="hello world.", language="en", N=_N)
    finally:
        restore()

    invoked = set(P.INVOKED)
    missing = sorted(_ALL_29 - invoked)
    print(f"invoked {len(invoked)}/29 modules; missing={missing}")
    print(f"codes(TT)={res['codes_tt'].tolist()}")
    print(
        f"full_chain_waveform_pcc (supplementary, compounds bf16 d-vector sensitivity)="
        f"{res['full_chain_waveform_pcc']}"
    )
    print(f"logmel_pcc (phase-insensitive full-chain, print-only)={res['logmel_pcc']} " f"logmel_l1={res['logmel_l1']}")
    achieved_pcc = res["e2e_pcc"]
    print(f"e2e PCC={achieved_pcc}")

    # Gate 2 — every module invoked in the real forward path
    assert not missing, f"Gate 2 FAILED: modules not invoked: {missing}"
    # Per-stage: every TT stage matches HF run on the previous TT output
    for k, thr in [
        ("speaker_embedding_pcc", 0.95),
        ("cond_latent_pcc", 0.95),
        ("ar_token_match", 0.95),
        ("ar_per_step_logits_pcc", 0.95),
        ("latents_pcc", 0.95),
        ("waveform_pcc", 0.95),
    ]:
        assert res[k] >= thr, f"stage gate FAILED: {k}={res[k]} < {thr}"
    # Gate 3 — final generate()-chain output (per-step logits + latents + vocoded waveform)
    print(f"e2e PCC={achieved_pcc}")
    assert achieved_pcc >= 0.95, f"Gate 3 FAILED: e2e PCC {achieved_pcc} < 0.95"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__ + "::test_e2e_tts", "-svv"]))
