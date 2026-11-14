# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug tests for gated_delta_net: report per-output / per-pair error instead of asserting the first one.

Kept as a record of the debugging process; DO NOT DELETE.
"""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
    to_device,
)


def _metrics(actual, expected):
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    finite = bool(torch.isfinite(actual).all())
    pcc = torch.corrcoef(torch.stack([actual, expected]))[0, 1].item() if finite else float("nan")
    rms = torch.sqrt(((actual - expected) ** 2).mean()).item()
    scale = expected.std().item()
    return pcc, rms / scale if scale > 1e-12 else rms, (actual - expected).abs().max().item()


def _report(inputs, tensors, device):
    expected = reference(*(tensors[name] for name in ORDER))
    out, cs, rs = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    out, cs, rs = (ttnn.to_torch(t).float() for t in (out, cs, rs))
    (_, rows, channels), (batch, value_heads, key_dim, value_dim) = inputs
    print()
    print("conv_state_out exact:", torch.equal(cs, expected[1]), "max diff", (cs - expected[1]).abs().max().item())
    print("output    pcc / rel_rms / max:", _metrics(out, expected[0]))
    print("state     pcc / rel_rms / max:", _metrics(rs, expected[2]))
    o = out.reshape(rows, value_heads, value_dim)
    e = expected[0].reshape(rows, value_heads, value_dim)
    for h in range(value_heads):
        print(f"  head {h}: output {_metrics(o[:, h], e[:, h])}  state {_metrics(rs[:, h], expected[2][:, h])}")
    return out, cs, rs, expected


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 256), (1, 3, 32, 64)), id="decode_b1_small"),
        pytest.param(((1, 5, 256), (1, 3, 32, 64)), id="prefill_t5"),
    ],
)
def test_report_random(device, inputs):
    torch.manual_seed(42)
    _report(inputs, make_inputs(inputs), device)


def test_zero_state_unit_gates(device):
    """S0 = 0, decode: S1 = kn u^T with u = beta v, o = (qn.kn) beta v. Isolates conv/gates/norm from the read."""
    inputs = ((1, 1, 256), (1, 3, 32, 64))
    torch.manual_seed(42)
    tensors = make_inputs(inputs)
    tensors["recurrent_state"].zero_()
    _report(inputs, tensors, device)


def test_identity_like_state(device):
    """Nonzero S0 with all-zero x (so kn = qn = 0, u = 0): S1 = alpha S0 exactly; output 0."""
    inputs = ((1, 1, 256), (1, 3, 32, 64))
    torch.manual_seed(42)
    tensors = make_inputs(inputs)
    tensors["mixed_qkv"].zero_()
    tensors["conv_state"].zero_()
    _report(inputs, tensors, device)
