# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Scaffolding test for ttnn.experimental.gated_delta_prefill_query.

The op is currently SCAFFOLDING: the placeholder kernels wire the full data path
end-to-end but do not yet implement the gated delta-rule recurrence. This test
therefore pins the *interface* (shapes / layouts / dtypes) and the one guarantee
the placeholder makes: state' is an exact passthrough copy of the input state.
"""
import pytest
import torch

import ttnn


@pytest.mark.parametrize(
    "Nk, Nv, S, d",
    [
        (2, 4, 32, 64),  # small, fast
        (16, 48, 64, 128),  # Qwen3.6-27B-shaped
    ],
)
def test_gated_delta_prefill_query_scaffold(device, Nk, Nv, S, d):
    torch.manual_seed(0)

    q_t = torch.randn(1, 1, Nk, d, dtype=torch.bfloat16)
    k_t = torch.randn(1, Nk, S, d, dtype=torch.bfloat16)
    v_t = torch.randn(1, Nv, S, d, dtype=torch.bfloat16)
    gate_t = torch.rand(1, Nv, 1, 1, dtype=torch.float32)  # beta (write strength)
    decay_t = -torch.rand(1, Nv, 1, 1, dtype=torch.float32)  # g (log-space decay)
    state_t = torch.randn(1, Nv, d, d, dtype=torch.float32)

    q = ttnn.from_torch(q_t, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    k = ttnn.from_torch(k_t, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(v_t, layout=ttnn.TILE_LAYOUT, device=device)
    gate = ttnn.from_torch(gate_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    decay = ttnn.from_torch(decay_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    state = ttnn.from_torch(state_t, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)

    o, state_out = ttnn.experimental.gated_delta_prefill_query(q, k, v, gate, decay, state)

    # ---- interface guarantees ----
    assert list(o.shape) == [1, 1, Nv, d]
    assert list(state_out.shape) == [1, Nv, d, d]
    assert o.dtype == ttnn.bfloat16
    assert state_out.dtype == ttnn.float32

    # ---- placeholder guarantee: state' ~= state (passthrough) ----
    # copy_tile routes fp32 through the unpacker's TF32 path (10-bit mantissa) unless
    # UnpackToDestFp32 is enabled, so the passthrough is close but not bit-exact.
    state_out_torch = ttnn.to_torch(state_out)
    assert torch.allclose(state_out_torch, state_t, rtol=1e-2, atol=1e-2), "state passthrough mismatch"
