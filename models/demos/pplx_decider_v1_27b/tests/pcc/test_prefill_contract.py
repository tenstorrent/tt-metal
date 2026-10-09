# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked-prefill contract on one loaded layer instance (both kinds).

Bucket-length requests run back to back on the SAME instance, so stale K/V pages or DeltaNet
state from a previous request would show up as a PCC drop: 8192 (4 chunks of 2048) -> 128 (one
short chunk that reuses pages and state slots the 8192 request filled) -> 2048 (exactly one chunk)
-> 4096 (two chunks, cross-chunk carry).
The last request is repeated and must be bit-identical (determinism).
"""

import pytest
import torch

from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    build_rotary,
    build_tt_layer,
    check_pcc,
    hf_layer_goldens,
    model_args,
    to_device,
    to_host,
)

REQUEST_LENGTHS = [8192, 128, 2048, 4096]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["L0_linear", "L3_full"])
def test_prefill_contract(device, layer_idx):
    layer = build_tt_layer(device, layer_idx)
    assert layer.config.optimizations.prefill_chunk == 2048
    rotary = build_rotary(device) if model_args().layer_kind(layer_idx) == "full_attention" else None
    last = None
    for seq_len in REQUEST_LENGTHS:
        golden = hf_layer_goldens(layer_idx, seq_len)
        x = to_device(golden["input"], device)
        out = to_host(layer(x, rotary), golden["layer"].shape)
        check_pcc(golden["layer"], out, module=f"contract_{layer.kind}", layer=layer_idx, seq_len=seq_len)
        last = (x, out)
    x, out = last
    again = to_host(layer(x, rotary), out.shape)
    assert torch.equal(out, again), "Repeated identical prefill must be bit-identical"
