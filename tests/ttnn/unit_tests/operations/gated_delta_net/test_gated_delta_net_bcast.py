# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Regime-pinned tests: R-BCAST (decode R-VSPLIT: per key head, one core preps qn and one kn, multicast to the
key head's other cores) against the same call with the sharing off (every core preps its own q/k). The shared
rows are exact fp32 copies of what the unshared path computes, so all three outputs must match bit for bit."""

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net import gated_delta_net
from ttnn.operations.gated_delta_net import gated_delta_net_program_descriptor as gdn_pd
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs


def _to_device(tensors, device, poison, token_dtype):
    out = []
    for name in ORDER:
        dtype = ttnn.float32 if name == "recurrent_state" else token_dtype
        t = ttnn.from_torch(
            tensors[name], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if poison is not None and name != "recurrent_state":
            t = ttnn.fill_implicit_tile_padding(t, poison)
        out.append(t)
    return out


def test_predicate():
    assert gdn_pd._share_qk(pieces=4, tokens=1)
    assert not gdn_pd._share_qk(pieces=1, tokens=1)  # R-STEP: no one-unit-per-core lockstep
    assert not gdn_pd._share_qk(pieces=4, tokens=2)  # several tile rows / chunked


@pytest.mark.parametrize(
    "inputs, token_dtype",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), ttnn.bfloat16, id="chip_b1_bf16"),  # 4 groups of 12
        pytest.param(((1, 2, 2560), (2, 12, 128, 128)), ttnn.bfloat16, id="chip_b2_bf16"),  # 8 groups of 12
        pytest.param(((1, 1, 10240), (1, 48, 128, 128)), ttnn.float32, id="h48_b1_f32"),  # 16 groups of 6
        pytest.param(((1, 3, 512), (3, 4, 32, 64)), ttnn.float32, id="small_r1_f32"),  # Hv = Hk: groups of 2
    ],
)
@pytest.mark.parametrize("poison", [None, float("inf")], ids=["clean", "inf_padding"])
def test_shared_matches_own(device, monkeypatch, inputs, token_dtype, poison):
    torch.manual_seed(5)
    tensors = make_inputs(inputs)
    if token_dtype == ttnn.bfloat16:
        for name in ORDER:
            if name != "recurrent_state":
                tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    results = []
    # q / k cores (QK_CORES_VSPLIT, where the rectangles fit), R-BCAST injectors, every core its own q / k.
    for qk_cores, share in ((True, True), (False, True), (False, False)):
        monkeypatch.setattr(gdn_pd, "QK_CORES_VSPLIT", qk_cores)
        monkeypatch.setattr(gdn_pd, "SHARE_QK_PREP", share)
        gdn_pd.clear_program_cache()
        device_inputs = _to_device(tensors, device, poison, token_dtype)
        outs = gated_delta_net(*device_inputs, compute_kernel_config=compute_config())
        results.append([ttnn.to_torch(t).to(torch.float32) for t in outs])
    for label, *variants in zip(("output", "conv_state_out", "recurrent_state_out"), *results):
        for variant in variants[:-1]:
            assert torch.equal(variant, variants[-1]), f"{label}: shared q/k prep differs from the unshared path"
