# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (linear_attn) vs HF, real weights, real-prompt inputs."""

import pytest

from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DELTA_LAYERS,
    DEVICE_PARAMS,
    SEQ_LENS,
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
@pytest.mark.parametrize("layer_idx", DELTA_LAYERS, ids=[f"L{i}" for i in DELTA_LAYERS])
def test_gated_deltanet(device, layer_idx, seq_len):
    golden = hf_layer_goldens(layer_idx, seq_len)
    layer = build_tt_layer(device, layer_idx)
    assert layer.kind == "linear_attention"
    out = layer.mixer_prefill(to_device(golden["input_norm"], device))
    check_pcc(
        golden["mixer"],
        to_host(out, golden["mixer"].shape),
        module="gated_deltanet",
        layer=layer_idx,
        seq_len=seq_len,
    )
