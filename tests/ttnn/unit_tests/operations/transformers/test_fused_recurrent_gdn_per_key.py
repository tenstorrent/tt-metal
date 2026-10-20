# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests of ttnn.transformer.fused_recurrent_gated_delta_rule's per-key-channel decay (the KDA recurrence:
g [B, T, HV, K], state row k decays by exp(g[k])) and of the round-robin head mapping (B*HV heads beyond the core
count), against the op's registered golden.

    pytest tests/ttnn/unit_tests/operations/transformers/test_fused_recurrent_gdn_per_key.py -s
"""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import make_inputs
from tests.ttnn.utils_for_testing import assert_with_pcc

OP = ttnn.transformer.fused_recurrent_gated_delta_rule
K = V = 128
PCC = 0.99999
# The FPU carries fp32 operands at reduced mantissa precision, so single elements of an O(1) state differ from the
# fp32 golden by up to ~3e-3 (the same on the scalar-decay path); PCC is the gate, the max-error bound is a sanity net.
ATOL = RTOL = 1e-2


def _golden():
    return ttnn.get_golden_function(OP)


def _dev(device, t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _per_key_inputs(B, T, H, seed):
    """GDN-style inputs (q, k L2-normed) with a KDA decay: g = -5 sigmoid(.) per key channel, [B, T, H, K]."""
    x = make_inputs(B, T, H, H, K, V, seed=seed)
    gen = torch.Generator().manual_seed(seed + 100)
    x["g"] = -5.0 * torch.sigmoid(torch.randn(B, T, H, K, generator=gen, dtype=torch.float32))
    return x


def _run(device, x, **kw):
    return OP(
        _dev(device, x["q"]),
        _dev(device, x["k"]),
        _dev(device, x["v"]),
        _dev(device, x["g"]),
        _dev(device, x["beta"]),
        initial_state=_dev(device, x["initial_state"]),
        **kw,
    )


def _check(o, s, o_ref, s_ref):
    o, s = ttnn.to_torch(o), ttnn.to_torch(s)
    assert o.shape == o_ref.shape and s.shape == s_ref.shape, (o.shape, o_ref.shape, s.shape, s_ref.shape)
    assert_with_pcc(o_ref, o, PCC)
    assert_with_pcc(s_ref, s, PCC)
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s, s_ref, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "B, T, H",
    [
        (1, 1, 64),  # GLM-5.3 KDA on one chip, 1 user (64 heads)
        (2, 1, 64),  # 128 heads: still one head per core on Blackhole
        (4, 1, 64),  # 256 heads: more heads than cores
        (32, 1, 16),  # GLM-5.3 KDA at TP4, 32 users per row
        (1, 4, 16),  # multi-token verify with per-token states
        (3, 3, 64),  # multi-token, many heads per core
    ],
)
def test_per_key_decay_vs_golden(device, B, T, H):
    x = _per_key_inputs(B, T, H, seed=B * 7 + T)
    per_token = T > 1
    o_ref, s_ref = _golden()(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=not per_token,
        output_per_token_state=per_token,
    )
    o, s = _run(device, x, output_final_state=not per_token, output_per_token_state=per_token)
    _check(o, s, o_ref, s_ref)


@pytest.mark.parametrize("B, H", [(1, 64), (8, 64)])
def test_per_key_constant_rows_equal_scalar_decay(device, B, H):
    """A rank-4 g constant over K is the scalar-decay op: both are one fp32 FPU multiply per element."""
    x = make_inputs(B, 1, H, H, K, V, seed=11)
    o3, s3 = _run(device, x, output_final_state=True)
    x4 = dict(x, g=x["g"].unsqueeze(-1).expand(-1, -1, -1, K).contiguous())
    o4, s4 = _run(device, x4, output_final_state=True)
    torch.testing.assert_close(ttnn.to_torch(o4), ttnn.to_torch(o3), rtol=0, atol=0)
    torch.testing.assert_close(ttnn.to_torch(s4), ttnn.to_torch(s3), rtol=0, atol=0)


@pytest.mark.parametrize("B, T, H", [(8, 1, 64), (2, 4, 64)])
def test_scalar_decay_many_heads_per_core(device, B, T, H):
    """The original scalar-decay op with B*HV beyond the core count (round-robin heads), final and per-token
    state."""
    x = make_inputs(B, T, H, H, K, V, seed=5)
    per_token = T > 1
    o_ref, s_ref = _golden()(
        x["q"],
        x["k"],
        x["v"],
        x["g"],
        x["beta"],
        initial_state=x["initial_state"],
        output_final_state=not per_token,
        output_per_token_state=per_token,
    )
    o, s = _run(device, x, output_final_state=not per_token, output_per_token_state=per_token)
    _check(o, s, o_ref, s_ref)


def test_per_key_ring_in_place(device):
    """Ring mode at T=1 with idx[h] = h: the state is read from and written back into the caller's [B*H, K, V]
    buffer (the GLM KDA decode runs on its persistent state this way), with a unit query scale folded by the
    caller."""
    B, H = 4, 64
    x = _per_key_inputs(B, 1, H, seed=3)
    q_scaled = x["q"] * K**-0.5
    o_ref, s_ref = _golden()(
        x["q"], x["k"], x["v"], x["g"], x["beta"], initial_state=x["initial_state"], output_final_state=True
    )
    ring = _dev(device, x["initial_state"].reshape(B * H, K, V))
    idx = _dev(device, torch.arange(B * H, dtype=torch.int32).reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    o, st = OP(
        _dev(device, q_scaled),
        _dev(device, x["k"]),
        _dev(device, x["v"]),
        _dev(device, x["g"]),
        _dev(device, x["beta"]),
        scale=1.0,
        initial_state=ring,
        initial_state_block_idx=idx,
        output_per_token_state=True,
    )
    assert st.buffer_address() == ring.buffer_address(), "ring mode must return the caller's buffer"
    s = ttnn.to_torch(ring).reshape(B, H, K, V)
    o = ttnn.to_torch(o)
    assert_with_pcc(o_ref, o, PCC)
    assert_with_pcc(s_ref, s, PCC)
    torch.testing.assert_close(o, o_ref, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(s, s_ref, rtol=RTOL, atol=ATOL)
