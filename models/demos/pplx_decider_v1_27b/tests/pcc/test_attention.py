# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated full attention (self_attn) vs HF, real weights, real-prompt inputs."""

import pytest

from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    FULL_LAYERS,
    SEQ_LENS,
    build_rotary,
    build_tt_layer,
    check_pcc,
    hf_layer_goldens,
    seq_ids,
    to_device,
    to_host,
)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=seq_ids())
@pytest.mark.parametrize("layer_idx", FULL_LAYERS, ids=[f"L{i}" for i in FULL_LAYERS])
def test_gated_attention(device, layer_idx, seq_len):
    golden = hf_layer_goldens(layer_idx, seq_len)
    layer = build_tt_layer(device, layer_idx)
    assert layer.kind == "full_attention"
    out = layer.mixer_prefill(to_device(golden["input_norm"], device), build_rotary(device))
    check_pcc(
        golden["mixer"],
        to_host(out, golden["mixer"].shape),
        module="gated_attention",
        layer=layer_idx,
        seq_len=seq_len,
    )
