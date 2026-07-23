# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end TTNN pipeline gate for microsoft/VibeVoice-1.5B (text -> speech).

Gate 1: routed graduated stubs stay native ttnn (composed as-is).
Gate 2: every one of the 19 graduated modules is INVOKED in the real forward path.
Gate 3: final waveform PCC vs the HF golden (faithful reimplemented generate() chain,
        both capped to N diffusion frames / S ddpm steps) >= 0.95.

Runs the SAME `tt/pipeline.py::run_tts` the demo uses.
"""

from __future__ import annotations

import os

import pytest
import torch

# instrument BEFORE importing composites so child forwards are tracked
from models.demos.vibevoice_1_5b.tt import pipeline as P
from models.demos.vibevoice_1_5b.tt._golden import reference as R

_N = int(os.environ.get("VIBEVOICE_E2E_N", "6"))
_S = int(os.environ.get("VIBEVOICE_E2E_S", "5"))
_ALL_19 = set(P.GRADUATED)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_e2e_tts(device):
    torch.manual_seed(0)
    model = R.load_reference_model()
    processor = R.build_processor()
    inputs = R.make_inputs(processor, "Speaker 0: Hello there, this is a test.", R.default_voice_sample())
    inputs = dict(inputs)
    inputs["noises"] = R.make_noises(_N + 2, int(model.config.acoustic_vae_dim))
    golden = R.hf_reference_tts(model, processor, inputs, N=_N, S=_S, noises=inputs["noises"])

    restore = P.instrument_stubs()
    try:
        res = P.run_tts(device, model, processor, inputs=inputs, N=_N, S=_S, golden=golden)
    finally:
        restore()

    invoked = set(P.INVOKED)
    missing = sorted(_ALL_19 - invoked)
    print(f"invoked {len(invoked)}/19 graduated stubs; missing={missing}")
    print(f"diffusion frames TT={res['diff_count']} HF={len(golden['audio'])}")
    achieved_pcc = res["e2e_pcc"]
    print(f"e2e PCC={achieved_pcc}")

    # Gate 2 — every graduated module invoked in the real forward path
    assert not missing, f"Gate 2 FAILED: graduated stubs not invoked: {missing}"
    # Gate 3 — final waveform PCC
    assert achieved_pcc >= 0.95, f"Gate 3 FAILED: e2e PCC {achieved_pcc} < 0.95"


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "trace_region_size": 400000000, "num_command_queues": 2}],
    indirect=True,
)
def test_e2e_tts_traced(device):
    """Same gate for the TRACED + 2CQ generation path (the customer default when a trace region and
    2 command queues are available). Replays the per-frame loop from captured traces; must invoke
    all 19 stubs and match the HF golden waveform PCC >= 0.95, identical to the eager gate."""
    torch.manual_seed(0)
    model = R.load_reference_model()
    processor = R.build_processor()
    inputs = R.make_inputs(processor, "Speaker 0: Hello there, this is a test.", R.default_voice_sample())
    inputs = dict(inputs)
    inputs["noises"] = R.make_noises(_N + 2, int(model.config.acoustic_vae_dim))
    golden = R.hf_reference_tts(model, processor, inputs, N=_N, S=_S, noises=inputs["noises"])

    restore = P.instrument_stubs()
    try:
        res = P.run_tts(device, model, processor, inputs=inputs, N=_N, S=_S, golden=golden, use_trace=True, two_cq=True)
    finally:
        restore()

    invoked = set(P.INVOKED)
    missing = sorted(_ALL_19 - invoked)
    print(f"[traced+2cq] invoked {len(invoked)}/19 stubs; missing={missing}")
    print(f"[traced+2cq] diffusion frames TT={res['diff_count']} HF={len(golden['audio'])}")
    achieved_pcc = res["e2e_pcc"]
    print(f"[traced+2cq] e2e PCC={achieved_pcc}")

    assert not missing, f"Gate 2 FAILED (traced): graduated stubs not invoked: {missing}"
    assert achieved_pcc >= 0.95, f"Gate 3 FAILED (traced): e2e PCC {achieved_pcc} < 0.95"
