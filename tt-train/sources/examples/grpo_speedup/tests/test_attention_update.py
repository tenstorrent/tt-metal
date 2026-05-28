# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mechanical pytest tests for ``Attention.update`` (no forward pass).

Two tests, both side-step the attention formula entirely:

1. **Read-back.** Build a known constant-valued tensor matching the live
   ``wqkv`` / ``wo`` buffers (shape, dtype, layout, memory config, mesh
   sharding). Push it through ``Attention.update``, read the live
   buffers back to torch, and assert byte equality.

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

from _completer_utils import build_completer, teardown_completer

WQKV_CONST_1 = 0.5
WO_CONST_1 = 0.25
WQKV_CONST_2 = 0.125
WO_CONST_2 = 0.0625


@pytest.fixture(scope="module")
def attn():
    completer = build_completer(dummy_weights=True)
    try:
        yield completer.model.layers[0].attention
    finally:
        teardown_completer(completer)


def _wqkv_mesh_mapper(a):
    """Mirror the mesh mapping used in ``Attention.__init__`` for ``self.wqkv``."""
    import ttnn

    return ttnn.ShardTensor2dMesh(
        a.mesh_device,
        dims=(3, 2) if a.TG else (2, 3),
        mesh_shape=a.args.cluster_shape,
    )


def _wo_mesh_mapper(a):
    """Mirror the mesh mapping used in ``Attention.__init__`` for ``self.wo``."""
    import ttnn

    if a.use_fused_all_gather_matmul or a.TG:
        return ttnn.ShardTensor2dMesh(a.mesh_device, dims=(2, 3), mesh_shape=a.args.cluster_shape)
    return ttnn.ShardTensorToMesh(a.mesh_device, dim=2)


def _build_constant_like(a, template, value, mapper):
    """Construct a constant ``ttnn.Tensor`` matching ``template``'s
    shape / dtype / layout / memory config and sharded with ``mapper``."""
    import ttnn

    torch_t = torch.full(tuple(template.shape), float(value), dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_t,
        dtype=template.dtype,
        layout=template.layout,
        device=a.mesh_device,
        memory_config=template.memory_config(),
        mesh_mapper=mapper,
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

    target_wqkv = _build_constant_like(attn, attn.wqkv, WQKV_CONST_1, _wqkv_mesh_mapper(attn))
    target_wo = _build_constant_like(attn, attn.wo, WO_CONST_1, _wo_mesh_mapper(attn))
    attn.update(wqkv=target_wqkv, wo=target_wo)

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

    target_wqkv = _build_constant_like(attn, attn.wqkv, WQKV_CONST_2, _wqkv_mesh_mapper(attn))
    target_wo = _build_constant_like(attn, attn.wo, WO_CONST_2, _wo_mesh_mapper(attn))
    attn.update(wqkv=target_wqkv, wo=target_wo)

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
