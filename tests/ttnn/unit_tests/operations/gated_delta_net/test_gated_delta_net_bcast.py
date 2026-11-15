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
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
)
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net_bf16 import _rel_metrics


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
    # q / k cores (QK_CORES_VSPLIT, where the rectangles fit; normalized hand-off), R-BCAST injectors, every core its
    # own q / k.
    monkeypatch.setattr(gdn_pd, "QK_RAW_HANDOFF", False)
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


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="chip_b1"),
        pytest.param(((1, 3, 512), (3, 4, 32, 64)), id="small_r1"),
        pytest.param(((1, 8, 2560), (8, 12, 128, 128)), id="chip_b8"),  # R-QKCORE pair cores, 1 sequence each
        pytest.param(((1, 16, 2560), (16, 12, 128, 128)), id="chip_b16"),  # 2 sequences each (held y, inverse norms)
    ],
)
@pytest.mark.parametrize("token_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
def test_raw_handoff_accuracy(device, monkeypatch, inputs, token_dtype):
    """QK_RAW_HANDOFF scales by the inverse norms after the state read instead of before: not bit-identical to the
    normalized hand-off, but as close to the reference (conv_state_out exact)."""
    torch.manual_seed(1)
    tensors = make_inputs(inputs)
    if token_dtype == ttnn.bfloat16:
        for name in ORDER:
            if name != "recurrent_state":
                tensors[name] = tensors[name].to(torch.bfloat16).to(torch.float32)
    expected = reference(*(tensors[name] for name in ORDER))
    errors = []
    for raw in (False, True):
        monkeypatch.setattr(gdn_pd, "QK_RAW_HANDOFF", raw)
        gdn_pd.clear_program_cache()
        outs = gated_delta_net(*_to_device(tensors, device, None, token_dtype), compute_kernel_config=compute_config())
        out, conv_state_out, state = (ttnn.to_torch(t).to(torch.float32) for t in outs)
        assert torch.equal(conv_state_out, expected[1]), "conv_state_out not bit-exact"
        errors.append([_rel_metrics(out, expected[0]), _rel_metrics(state, expected[2])])
    for label, (normalized, raw) in zip(("output", "recurrent_state_out"), zip(*errors)):
        assert raw[0] >= normalized[0] - 1e-6, f"{label}: PCC {raw[0]} vs {normalized[0]} normalized"
        assert raw[1] <= 1.25 * normalized[1], f"{label}: relative RMS {raw[1]} vs {normalized[1]} normalized"
