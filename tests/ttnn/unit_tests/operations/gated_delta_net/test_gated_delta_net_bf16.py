# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""bfloat16 token tensors (fp32 state): 32 B face rows read through aligned staging and written
by several cores into one output / conv_state_out tile; repeated calls must be bit-identical."""

import pytest
import torch

import ttnn
from eval.metrics import default_tolerance
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)

REPEATS = 8


def _rel_metrics(actual, expected):
    actual, expected = actual.double().flatten(), expected.double().flatten()
    pcc = torch.corrcoef(torch.stack([actual, expected]))[0, 1].item()
    rms = torch.sqrt(((actual - expected) ** 2).mean()).item()
    scale = expected.std().item()
    return pcc, rms / scale if scale > 1e-12 else rms


def _to_device(tensors, device, poison):
    out = []
    for name in ORDER:
        dtype = ttnn.float32 if name == "recurrent_state" else ttnn.bfloat16
        t = ttnn.from_torch(
            tensors[name], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if poison is not None and name != "recurrent_state":
            t = ttnn.fill_implicit_tile_padding(t, poison)
        out.append(t)
    return out


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 256), (1, 3, 32, 64)), id="decode_b1_small"),
        pytest.param(((1, 3, 256), (3, 3, 32, 64)), id="decode_b3_shared_rows"),
        pytest.param(((1, 40, 256), (40, 3, 32, 64)), id="decode_b40_two_tile_rows"),
        pytest.param(((1, 10, 256), (2, 3, 32, 64)), id="batched_b2_t5_odd_boundary"),
        pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="prefill_t33"),
    ],
)
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_bf16_tokens(device, inputs, poison):
    torch.manual_seed(7)
    tensors = make_inputs(inputs)
    for name in ORDER:
        if name != "recurrent_state":
            tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    device_inputs = _to_device(tensors, device, poison)

    first = None
    for _ in range(REPEATS):
        out, cs, rs = gated_delta_net(*device_inputs, compute_kernel_config=compute_config())
        assert out.dtype == ttnn.bfloat16 and cs.dtype == ttnn.bfloat16 and rs.dtype == ttnn.float32
        result = tuple(ttnn.to_torch(t).to(torch.float32) for t in (out, cs, rs))
        if first is None:
            first = result
        else:
            for got, ref, label in zip(result, first, ("output", "conv_state_out", "recurrent_state_out")):
                assert torch.equal(got, ref), f"{label} differs between repeated calls"

    out_t, cs_t, rs_t = first
    assert torch.equal(cs_t, expected[1]), "conv_state_out not bit-exact"
    pcc_min, rms_max = default_tolerance(ttnn.bfloat16, True)
    for got, ref, label in ((out_t, expected[0], "output"), (rs_t, expected[2], "recurrent_state_out")):
        assert torch.isfinite(got).all(), f"{label}: non-finite values"
        pcc, rel_rms = _rel_metrics(got, ref)
        assert pcc >= pcc_min, f"{label}: PCC {pcc:.6f} < {pcc_min}"
        assert rel_rms <= rms_max, f"{label}: relative RMS {rel_rms:.6f} > {rms_max}"
