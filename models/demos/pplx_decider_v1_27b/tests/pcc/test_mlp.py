# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP vs HF, real weights, real-prompt inputs (the HF post-attention-normed hidden state)."""

import pytest

from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    ALL_LAYERS,
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
@pytest.mark.parametrize("layer_idx", ALL_LAYERS, ids=[f"L{i}" for i in ALL_LAYERS])
def test_mlp(device, layer_idx, seq_len):
    golden = hf_layer_goldens(layer_idx, seq_len)
    layer = build_tt_layer(device, layer_idx)
    out = layer.mlp(to_device(golden["post_norm"], device))
    check_pcc(golden["mlp"], to_host(out, golden["mlp"].shape), module="mlp", layer=layer_idx, seq_len=seq_len)
