# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests: R-QKCORE (decode R-STEP with every sequence in one tile row: per key head a q core and a k
core prep qn / kn of all sequences at once and multicast them to the key head's pair cores) against the same call
with the q/k cores off (every pair core preps its own q/k). With the normalized hand-off (QK_RAW_HANDOFF off) the q/k
cores run the same per-row conv and L2 norm on rows 0..B-1, so all three outputs must match bit for bit; the default
raw hand-off is checked against the reference (and against the normalized one in test_gated_delta_net_bcast)."""

import pytest
import torch

from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bcast import _to_device
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics

import ttnn


def test_predicate():
    # model geometry (Hk 4, Hv 12, Kt = Vt = 4) on 110 cores
    assert gdn_pd._qk_core_units(8, 1, 12, 4, 4, 4, 1, 110) == 1  # 96 pair cores + 8 q/k cores
    assert gdn_pd._qk_core_units(16, 1, 12, 4, 4, 4, 1, 110) == 2  # 96 pair cores, 2 sequences each
    assert gdn_pd._qk_core_units(8, 2, 12, 4, 4, 4, 1, 110) is None  # not decode
    assert gdn_pd._qk_core_units(8, 1, 12, 4, 4, 4, 4, 110) is None  # R-VSPLIT (R-BCAST shares instead)
    assert gdn_pd._qk_core_units(40, 1, 12, 4, 4, 4, 1, 110) is None  # sequences over two tile rows


@pytest.mark.parametrize(
    "inputs, token_dtype",
    [
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), ttnn.bfloat16, id="chip_b8_bf16"),  # 1 sequence per core
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), ttnn.bfloat16, id="chip_b16_bf16"),  # 2 per core
        pytest.param(((1, 5, 2560), (5, 12, 128, 128)), ttnn.float32, id="chip_b5_f32"),  # SFPU conv, 2 bands
        pytest.param(((1, 27, 2560), (27, 12, 128, 128)), ttnn.bfloat16, id="chip_b27_bf16"),  # ragged last core
        pytest.param(((1, 32, 256), (32, 3, 32, 64)), ttnn.float32, id="small_b32_f32"),  # K != V, full tile row
    ],
)
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_qk_cores_match_own(device, monkeypatch, inputs, token_dtype, poison):
    batch, value_heads, key_dim, value_dim = inputs[1]
    key_heads = (inputs[0][2] - value_heads * value_dim) // (2 * key_dim)
    grid = device.compute_with_storage_grid_size()
    assert gdn_pd._qk_core_units(
        batch, 1, value_heads, key_heads, key_dim // 32, value_dim // 32, 1, grid.x * grid.y
    ), "case must exercise R-QKCORE"
    torch.manual_seed(7)
    tensors = make_inputs(inputs)
    if token_dtype == ttnn.bfloat16:
        for name in ORDER:
            if name != "recurrent_state":
                tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    results = []
    for share, raw in ((True, False), (False, False), (True, True)):
        monkeypatch.setattr(gdn_pd, "SHARE_QK_CORES", share)
        monkeypatch.setattr(gdn_pd, "QK_RAW_HANDOFF", raw)
        device_inputs = _to_device(tensors, device, poison, token_dtype)
        outs = gated_delta_net(*device_inputs, compute_kernel_config=compute_config())
        results.append([ttnn.to_torch(t).to(torch.float32) for t in outs])
    for shared, own, label in zip(*results[:2], ("output", "conv_state_out", "recurrent_state_out")):
        assert torch.equal(shared, own), f"{label}: q/k cores differ from the per-pair q/k prep"
    for result in (results[0], results[2]):
        assert torch.equal(result[1], expected[1]), "conv_state_out not bit-exact"
        for got, ref, label, pcc_min, rms_max in (
            (result[0], expected[0], "output", 0.995, 0.04),
            (result[2], expected[2], "recurrent_state_out", 0.999, 0.01),
        ):
            pcc, rel_rms = _rel_metrics(got, ref)
            assert pcc >= pcc_min and rel_rms <= rms_max, f"{label}: PCC {pcc:.6f}, rel RMS {rel_rms:.6f}"
