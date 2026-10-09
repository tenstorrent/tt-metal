# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zero-centred RMSNorm (input_layernorm and post_attention_layernorm) vs HF, real weights and inputs."""

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
@pytest.mark.parametrize("which", ["input_norm", "post_norm"])
def test_rmsnorm(device, which, layer_idx, seq_len):
    golden = hf_layer_goldens(layer_idx, seq_len)
    layer = build_tt_layer(device, layer_idx)
    if which == "input_norm":
        x, norm = golden["input"], layer.input_norm
    else:
        x, norm = golden["input"] + golden["mixer"], layer.post_norm  # HF h = x + mixer
    expected = golden[which]
    out = norm(to_device(x, device))
    check_pcc(expected, to_host(out, expected.shape), module=f"rmsnorm_{which}", layer=layer_idx, seq_len=seq_len)
