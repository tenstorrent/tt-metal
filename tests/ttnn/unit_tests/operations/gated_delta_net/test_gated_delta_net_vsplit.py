# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests: R-VSPLIT (state columns of a pair split over Vt / V_BLOCK_TILES cores, partial
sum o^2 all-gathered per tile row) vs R-STEP (whole pair per core). The host predicate picks the regime;
forcing V_BLOCK_TILES checks every split on the same inputs, including multi-tile-row prefill (several
combine rounds) and inf padding."""

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
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics, _to_device

GRID_CORES = 110


@pytest.mark.parametrize(
    "num_pairs, Vt, expected",
    [
        (12, 4, 4),  # perf_decode_b1: 48 cores
        (24, 4, 4),  # decode B=2 / batched B=2: 96 cores
        (96, 4, 1),  # decode B=8: pairs already fill the grid -> R-STEP
        (192, 4, 1),  # decode B=16 -> R-STEP
        (3, 2, 2),  # small geometry
        (120, 2, 1),  # small geometry decode B=40 -> R-STEP
    ],
)
def test_regime_predicate(num_pairs, Vt, expected):
    assert gdn_pd._value_pieces(num_pairs, Vt, GRID_CORES) == expected


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="decode_b1_chip"),
        pytest.param(((1, 70, 2560), (1, 12, 128, 128)), id="prefill_t70_chip"),
        pytest.param(((1, 10, 256), (2, 3, 32, 64)), id="batched_b2_t5_small"),
    ],
)
@pytest.mark.parametrize("v_block_tiles", [1, 2, None], ids=["vb1", "vb2", "auto"])
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_forced_split(device, monkeypatch, inputs, v_block_tiles, poison):
    Vt = inputs[1][3] // 32
    if v_block_tiles is not None and Vt % v_block_tiles != 0:
        pytest.skip("block must divide Vt")
    monkeypatch.setattr(gdn_pd, "V_BLOCK_TILES", v_block_tiles)
    torch.manual_seed(11)
    tensors = make_inputs(inputs)
    for name in ORDER:
        if name != "recurrent_state":
            tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    out, cs, rs = gated_delta_net(*_to_device(tensors, device, poison), compute_kernel_config=compute_config())
    out_t, cs_t, rs_t = (ttnn.to_torch(t).to(torch.float32) for t in (out, cs, rs))
    assert torch.equal(cs_t, expected[1]), "conv_state_out not bit-exact"
    for got, ref, label, pcc_min, rms_max in (
        (out_t, expected[0], "output", 0.995, 0.04),
        (rs_t, expected[2], "recurrent_state_out", 0.999, 0.01),
    ):
        pcc, rel_rms = _rel_metrics(got, ref)
        assert pcc >= pcc_min and rel_rms <= rms_max, f"{label}: PCC {pcc:.6f}, rel RMS {rel_rms:.6f}"
