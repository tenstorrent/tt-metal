# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole decoder layer (both kinds, two indices each) vs HF, real weights, real-prompt inputs."""

import pytest

from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    ALL_LAYERS,
    DEVICE_PARAMS,
    SEQ_LENS,
    build_rotary,
    build_tt_layer,
    check_pcc,
    hf_layer_goldens,
    model_args,
    seq_ids,
    to_device,
    to_host,
)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=seq_ids())
@pytest.mark.parametrize("layer_idx", ALL_LAYERS, ids=[f"L{i}" for i in ALL_LAYERS])
def test_decoder_layer(device, layer_idx, seq_len):
    golden = hf_layer_goldens(layer_idx, seq_len)
    layer = build_tt_layer(device, layer_idx)
    rotary = build_rotary(device) if model_args().layer_kind(layer_idx) == "full_attention" else None
    out = layer(to_device(golden["input"], device), rotary)
    check_pcc(
        golden["layer"],
        to_host(out, golden["layer"].shape),
        module=f"decoder_layer_{layer.kind}",
        layer=layer_idx,
        seq_len=seq_len,
    )
