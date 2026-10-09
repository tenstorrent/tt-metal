# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runtime fallback audit: a measured prefill pass issues no host conversions, round trips or torch ops.

After setup (weights loaded, input uploaded, one warm-up pass), one decoder-layer prefill of each
kind runs inside ``count_host_calls``; the counter must stay empty. S=128 is the smallest bucket
(one short chunk); S=4096 is two 2048-token chunks with cross-chunk K/V and DeltaNet state carry. A positive
control proves the counter sees ``ttnn.to_torch`` and torch ops.
"""

import pytest
import torch

import ttnn
from models.demos.pplx_decider_v1_27b.tests.runtime_audit import count_host_calls
from models.demos.pplx_decider_v1_27b.tests.test_utils import (
    DEVICE_PARAMS,
    bf16_round,
    build_rotary,
    build_tt_layer,
    golden_tensor,
    model_args,
    to_device,
)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("seq_len", [128, 4096], ids=["S128", "S4096"])
@pytest.mark.parametrize("layer_idx", [0, 3], ids=["L0_linear", "L3_full"])
def test_prefill_pass_stays_on_device(device, layer_idx, seq_len):
    layer = build_tt_layer(device, layer_idx)
    rotary = build_rotary(device) if model_args().layer_kind(layer_idx) == "full_attention" else None
    x = to_device(bf16_round(golden_tensor(f"L{layer_idx}_input", seq_len)), device)
    ttnn.deallocate(layer(x, rotary))  # warm-up: lazy weight upload and program compile happen here
    ttnn.synchronize_device(device)

    with count_host_calls() as counts:
        out = layer(x, rotary)
        ttnn.synchronize_device(device)
    assert ttnn.is_tensor_storage_on_device(out)
    assert tuple(out.shape) == (1, seq_len, model_args().hidden_size)
    assert not counts, f"Host calls inside the measured prefill pass: {dict(counts)}"

    # Positive control: the counter does see host conversions and torch ops.
    with count_host_calls() as control:
        torch.ones(2).sum()
        ttnn.to_torch(out)
    assert control["ttnn.to_torch"] == 1 and any(k.startswith("torch_function:") for k in control), dict(control)
