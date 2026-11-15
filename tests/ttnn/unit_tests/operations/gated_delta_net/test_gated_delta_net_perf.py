# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf_decode_b1 (feature_spec.LOOSE_CASES[0]) and one shape per kernel path (fp32 / bf16 decode, decode
with shared tile rows, prefill over several tile rows, batched prefill): a correctness gate plus a fixed
launch count for `run_safe_pytest.sh --profile` device-kernel-time measurement."""

import ast
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

LAUNCHES = int(os.environ.get("GDN_LAUNCHES", "3"))  # e.g. 12: the median of launches 2..12
# Knob overrides for A/B measurement, e.g. GDN_KNOBS="STATE_SPLIT=0".
EXPERIMENT = {
    name: ast.literal_eval(value)
    for name, value in (item.split("=") for item in os.environ.get("GDN_KNOBS", "").split(",") if item)
}


def _to_device(tensors, device, token_dtype):
    return [
        ttnn.from_torch(
            tensors[name],
            dtype=ttnn.float32 if name == "recurrent_state" else token_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for name in ORDER
    ]


@pytest.mark.parametrize(
    "inputs, token_dtype",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), ttnn.bfloat16, id="perf_decode_b1"),
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), ttnn.bfloat16, id="model_decode_b8"),
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), ttnn.bfloat16, id="model_decode_b16"),
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), ttnn.float32, id="decode_b1_f32"),
        pytest.param(((1, 96, 2560), (1, 12, 128, 128)), ttnn.bfloat16, id="prefill_t96"),
        pytest.param(((1, 80, 2560), (2, 12, 128, 128)), ttnn.bfloat16, id="batched_b2_t40"),
        pytest.param(((1, 4096, 2560), (1, 12, 128, 128)), ttnn.bfloat16, id="model_prefill_4096"),
    ],
)
def test_perf_decode(device, monkeypatch, inputs, token_dtype):
    for name, value in EXPERIMENT.items():
        monkeypatch.setattr(gdn_pd, name, value)
    gdn_pd.clear_program_cache()
    torch.manual_seed(0)
    tensors = make_inputs(inputs)
    if token_dtype == ttnn.bfloat16:
        for name in ORDER:
            if name != "recurrent_state":
                tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    device_inputs = _to_device(tensors, device, token_dtype)
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


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="perf_decode_b1"),
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), id="model_decode_b16"),
    ],
)
def test_perf_in_place(device, inputs):
    """Refinement 10: the same calls updating conv_state / recurrent_state in place (first call checked)."""
    torch.manual_seed(0)
    tensors = make_inputs(inputs)
    for name in ORDER:
        if name != "recurrent_state":
            tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    device_inputs = _to_device(tensors, device, ttnn.bfloat16)
    conv_state, recurrent_state = (
        device_inputs[ORDER.index("conv_state")],
        device_inputs[ORDER.index("recurrent_state")],
    )
    for launch in range(LAUNCHES):
        out, _, _ = gated_delta_net(
            *device_inputs,
            compute_kernel_config=compute_config(),
            conv_state_out=conv_state,
            recurrent_state_out=recurrent_state,
        )
        if launch == 0:
            out_t, cs_t, rs_t = (ttnn.to_torch(t).to(torch.float32) for t in (out, conv_state, recurrent_state))
            assert torch.equal(cs_t, expected[1]), "conv_state not bit-exact"
            for got, ref, label in ((out_t, expected[0], "output"), (rs_t, expected[2], "recurrent_state")):
                pcc, rel_rms = _rel_metrics(got, ref)
                assert pcc >= 0.995 and rel_rms <= 0.04, f"{label}: PCC {pcc:.6f}, rel RMS {rel_rms:.6f}"
