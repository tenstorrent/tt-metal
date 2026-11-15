# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 11: short and medium prefill (T = 32 to 512) at the model geometry, per regime. `default` is the
regime the descriptor picks; `prep` forces R-PREP (PREP_MIN_TILE_ROWS = 1), `prep_h1` R-PREP with one value head per
prep group, `local` the local chunk step.
A correctness gate plus a fixed launch count for `run_safe_pytest.sh --profile`."""

import os

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_perf import LAUNCHES, _to_device


EXPERIMENT = {}  # knob overrides for A/B measurement


@pytest.mark.parametrize("regime", ["default", "prep", "prep_h1", "local"])
@pytest.mark.parametrize("tokens", [32, 64, 96, 128, 256, 512, 1024, 2048, 4096])
def test_short_prefill(device, monkeypatch, tokens, regime):
    # Subsets for --profile runs, e.g. GDN_TOKENS=128,4096 GDN_REGIMES=prep.
    for env, value in (("GDN_TOKENS", str(tokens)), ("GDN_REGIMES", regime)):
        if os.environ.get(env) and value not in os.environ[env].split(","):
            pytest.skip(f"not selected by {env}")
    if regime.startswith("prep"):
        monkeypatch.setattr(gdn_pd, "PREP_MIN_TILE_ROWS", 1)
    if regime == "prep_h1":
        monkeypatch.setattr(gdn_pd, "PREP_HEADS", 1)
    elif regime == "local":
        monkeypatch.setattr(gdn_pd, "PREP_OFFLOAD", False)
    for knob, value in EXPERIMENT.items():
        monkeypatch.setattr(gdn_pd, knob, value)
    torch.manual_seed(0)
    tensors = make_inputs(((1, tokens, 2560), (1, 12, 128, 128)))
    for name in ORDER:
        if name != "recurrent_state":
            tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    device_inputs = _to_device(tensors, device, ttnn.bfloat16)
    for _ in range(LAUNCHES):
        out, cs, rs = gated_delta_net(*device_inputs, compute_kernel_config=compute_config())
    out_t, cs_t, rs_t = (ttnn.to_torch(t).to(torch.float32) for t in (out, cs, rs))
    assert torch.equal(cs_t, expected[1]), "conv_state_out not bit-exact"
    for got, ref, label, pcc_min, rms_max in (
        (out_t, expected[0], "output", 0.995, 0.04),
        (rs_t, expected[2], "recurrent_state_out", 0.999, 0.01),
    ):
        pcc, rel_rms = _rel_metrics(got, ref)
        assert pcc >= pcc_min and rel_rms <= rms_max, f"{label}: PCC {pcc:.6f}, rel RMS {rel_rms:.6f}"
