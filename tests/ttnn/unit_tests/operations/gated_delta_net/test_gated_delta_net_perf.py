# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""perf_decode_b1 config (feature_spec.LOOSE_CASES[0]): correctness gate plus a fixed launch count
for `run_safe_pytest.sh --profile` device-kernel-time measurement."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics, _to_device

LAUNCHES = 3


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="perf_decode_b1"),
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), id="model_decode_b8"),
    ],
)
def test_perf_decode(device, inputs):
    torch.manual_seed(0)
    tensors = make_inputs(inputs)
    for name in ORDER:
        if name != "recurrent_state":
            tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    device_inputs = _to_device(tensors, device, None)
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
