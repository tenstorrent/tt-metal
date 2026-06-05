# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mechanical pytest tests for ``Attention.update`` (no forward pass).

Two tests, both side-step the attention formula entirely:

1. **Read-back.** Push HF-format constant-valued ``q_proj`` / ``k_proj``
   / ``v_proj`` / ``o_proj`` tensors through ``Attention.update`` and
   read the live internal ``wqkv`` / ``wo`` buffers back to torch. Using
   the same constant for Q/K/V means the internal fused
   ``[Q | K | V]`` wqkv must come back uniformly equal to that
   constant.

2. **Buffer-address preservation.** Snapshot ``buffer_address()`` before
   and after a second update; ``ttnn.copy`` must overwrite the existing
   device buffer rather than allocate a new one (otherwise captured
   traces and the DRAM prefetcher's stashed addresses would silently
   stop pointing at live data).

Uses ``dummy_weights=True`` -- no HF auth required.
"""

from __future__ import annotations

import pytest
import torch

from _completer_utils import as_update_input, open_completer

WQKV_CONST_1 = 0.5
WO_CONST_1 = 0.25
WQKV_CONST_2 = 0.125
WO_CONST_2 = 0.0625


@pytest.fixture(scope="module")
def attn():
    with open_completer(dummy_weights=True) as completer:
        yield completer.model.layers[0].attention


def _push_constants(attn, wqkv_const: float, wo_const: float) -> None:
    """Build constant HF-shape ``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``o_proj`` tensors and push them through ``Attention.update``.

    Using one constant for all of Q/K/V keeps the readback assertion
    simple: the internal fused wqkv must come back uniformly equal to
    ``wqkv_const``.
    """
    H = attn.hidden_size
    D = attn.head_dim
    n_q = attn.n_heads * D
    n_kv = attn.n_kv_heads * D

    q_hf = torch.full((n_q, H), wqkv_const, dtype=torch.bfloat16)
    k_hf = torch.full((n_kv, H), wqkv_const, dtype=torch.bfloat16)
    v_hf = torch.full((n_kv, H), wqkv_const, dtype=torch.bfloat16)
    o_hf = torch.full((H, n_q), wo_const, dtype=torch.bfloat16)

    attn.update(
        q_proj=as_update_input(q_hf, attn.mesh_device),
        k_proj=as_update_input(k_hf, attn.mesh_device),
        v_proj=as_update_input(v_hf, attn.mesh_device),
        o_proj=as_update_input(o_hf, attn.mesh_device),
    )


def _buffer_addr(t):
    """``buffer_address()`` that also works for multi-device tensors."""
    import ttnn

    try:
        return t.buffer_address()
    except Exception:
        return tuple(x.buffer_address() for x in ttnn.get_device_tensors(t))


def _all_equal_constant(got, value):
    expected = torch.full(tuple(got.shape), float(value), dtype=got.dtype)
    return torch.equal(got, expected)


def test_readback(attn):
    """Constants pushed through ``update`` survive the round trip exactly."""
    import ttnn

    _push_constants(attn, WQKV_CONST_1, WO_CONST_1)

    got_wqkv = ttnn.to_torch(attn.wqkv)
    got_wo = ttnn.to_torch(attn.wo)

    assert _all_equal_constant(got_wqkv, WQKV_CONST_1), (
        f"wqkv read-back differs from written constant {WQKV_CONST_1}: "
        f"max|diff|={float((got_wqkv.float() - WQKV_CONST_1).abs().max()):.6g}"
    )
    assert _all_equal_constant(got_wo, WO_CONST_1), (
        f"wo read-back differs from written constant {WO_CONST_1}: "
        f"max|diff|={float((got_wo.float() - WO_CONST_1).abs().max()):.6g}"
    )

    wo_ring = getattr(attn, "wo_sharded_ring", None)
    if wo_ring is not None:
        got_ring = ttnn.to_torch(wo_ring)
        assert _all_equal_constant(got_ring, WO_CONST_1), (
            "wo_sharded_ring prefetcher mirror was not updated alongside wo "
            f"(max|diff|={float((got_ring.float() - WO_CONST_1).abs().max()):.6g})"
        )


def test_addresses_preserved(attn):
    """A second ``update`` must not reallocate any of the live buffers."""
    addrs_before = {
        "wqkv": _buffer_addr(attn.wqkv),
        "wo": _buffer_addr(attn.wo),
    }
    wo_ring = getattr(attn, "wo_sharded_ring", None)
    if wo_ring is not None:
        addrs_before["wo_sharded_ring"] = _buffer_addr(wo_ring)

    _push_constants(attn, WQKV_CONST_2, WO_CONST_2)

    addrs_after = {
        "wqkv": _buffer_addr(attn.wqkv),
        "wo": _buffer_addr(attn.wo),
    }
    if wo_ring is not None:
        addrs_after["wo_sharded_ring"] = _buffer_addr(wo_ring)

    assert addrs_before == addrs_after, (
        "buffer_address() changed after update -- ttnn.copy reallocated, which "
        "would silently invalidate captured traces and prefetcher addresses. "
        f"before={addrs_before}, after={addrs_after}"
    )
