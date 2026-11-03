# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for ring_joint_scaled_dot_product_attention (Phase 0 contract). Immutable: the implementer must not
edit this file.

Covers the Phase 0 SUPPORTED rectangle: bf16 Q/K/V, fp32 DEST (the default config), TILE, tile-aligned, self-attention,
MHA, non-causal, no joint / replicated tile-aligned joint, exact / padded sequence, Linear on both mesh axes, Ring on
the box's ring stand-in (or a real torus axis). Every device's valid rows are compared against a pure-torch fp32
reference of the group's global attention; a second call must be bitwise identical (deterministic, program-cache
path with fresh tensors).

The mesh comes from the golden suite's cached fabric-mesh fixture (opened with the production router payload and the
fabric config each case pins): a multi-device fabric op cannot run on the single-device `device` fixture.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

# mesh_device / _fabric_mesh_cache come from this directory's conftest.py (one session-wide fabric-mesh cache).
from eval.golden_tests.ring_joint_scaled_dot_product_attention.topology import (
    ring_mock_meshes,
    system_mesh_shape,
    wrap_axes,
)
from ttnn.operations.ring_joint_scaled_dot_product_attention import ring_joint_scaled_dot_product_attention

SEED = 42
SEQ_PAD = 40
PCC = {ttnn.bfloat16: 0.995, ttnn.float32: 0.999, ttnn.bfloat8_b: 0.99}

# (id, per-device Q/K/V shape (b, h, n, d), padded?, joint L or None)
SHAPES = [
    ("single_tile", (1, 1, 32, 32), False, None),
    ("multi_tile_padded", (1, 2, 128, 64), True, None),
    ("non_square_dh128", (1, 4, 256, 128), False, None),
    ("multi_batch_padded", (2, 2, 128, 64), True, None),
    ("joint_replicated_L64", (1, 2, 128, 64), True, 64),
]


# --- placements -------------------------------------------------------------------------------------------------


def _placements():
    """(id, mesh_shape, fabric_config, cluster_axis, topology) this cluster can run."""
    rows, cols = system_mesh_shape()
    out = []
    for axis in (0, 1):
        if (rows, cols)[axis] >= 2:
            out.append((f"linear_axis{axis}", (rows, cols), ttnn.FabricConfig.FABRIC_2D, axis, ttnn.Topology.Linear))
    for axis in wrap_axes():  # a real torus axis (Galaxy)
        if (rows, cols)[axis] >= 3:
            cfg = ttnn.FabricConfig.FABRIC_2D_TORUS_Y if axis == 0 else ttnn.FabricConfig.FABRIC_2D_TORUS_X
            out.append((f"ring_axis{axis}", (rows, cols), cfg, axis, ttnn.Topology.Ring))
    for mesh_shape, axis, configs in ring_mock_meshes():  # the box's snake ring as 1xN (wrap-less boxes)
        if axis == 1 and ttnn.FabricConfig.FABRIC_2D_TORUS_XY in configs and mesh_shape[1] >= 3:
            out.append(("ring_mock_1xN", mesh_shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY, axis, ttnn.Topology.Ring))
    return out


PLACEMENTS = _placements()


# --- tensors and reference --------------------------------------------------------------------------------------


def _stacked(mesh_shape, shape, gen):
    rows, cols = mesh_shape
    return torch.randn((rows, cols, *shape), generator=gen, dtype=torch.float32).to(torch.bfloat16)


def _replicate_along(stacked, cluster_axis):
    out = stacked.clone()
    if cluster_axis == 1:
        out[:, :] = stacked[:, :1]
    else:
        out[:, :] = stacked[:1, :]
    return out


def _to_mesh(stacked, mesh):
    if stacked is None:
        return None
    rows, cols = stacked.shape[0], stacked.shape[1]
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    return ttnn.from_torch(
        glob,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )


def _members(r, c, cluster_axis, rows, cols):
    if cluster_axis == 1:
        return [(r, j) for j in range(cols)], c
    return [(i, c) for i in range(rows)], r


def _attend(q, k, v, key_valid, scale):
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    s[..., ~key_valid] = float("-inf")
    return torch.matmul(torch.softmax(s, dim=-1), v)


def torch_reference(q, k, v, jq, jk, jv, *, cluster_axis, logical_n):
    """Stacked (R, C, b, h, n, d) inputs -> {(r, c): (out, joint_out | None)} in fp32. Contiguous layout, non-causal,
    replicated joint (every member holds the same joint rows)."""
    rows, cols = q.shape[0], q.shape[1]
    scale = q.shape[-1] ** -0.5
    res = {}
    for r in range(rows):
        for c in range(cols):
            members, _ = _members(r, c, cluster_axis, rows, cols)
            n_kv = k.shape[-2]
            k_all = torch.cat([k[m].float() for m in members], dim=-2)
            v_all = torch.cat([v[m].float() for m in members], dim=-2)
            valid = torch.arange(len(members) * n_kv) < logical_n
            if jk is not None:
                k_all = torch.cat([k_all, jk[r, c].float()], dim=-2)
                v_all = torch.cat([v_all, jv[r, c].float()], dim=-2)
                valid = torch.cat([valid, torch.ones(jk.shape[-2], dtype=torch.bool)])
            out = _attend(q[r, c].float(), k_all, v_all, valid, scale)
            jout = _attend(jq[r, c].float(), k_all, v_all, valid, scale) if jq is not None else None
            res[(r, c)] = (out, jout)
    return res


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return 1.0 if den < 1e-30 else float((a * b).sum() / den)


def _device_tensors(t):
    return [ttnn.to_torch(x).float() for x in ttnn.get_device_tensors(t)]


# --- the test ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape_id,shape,padded,joint_len", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize(
    "placement_id,mesh_shape,fabric_config,cluster_axis,topology",
    PLACEMENTS,
    ids=[p[0] for p in PLACEMENTS],
)
def test_ring_joint_scaled_dot_product_attention(
    placement_id, mesh_shape, fabric_config, cluster_axis, topology, shape_id, shape, padded, joint_len, mesh_device
):
    assert tuple(mesh_device.shape) == tuple(mesh_shape)
    rows, cols = mesh_shape
    G = mesh_shape[cluster_axis]
    b, h, n, d = shape
    logical_n = G * n - (SEQ_PAD if padded else 0)

    gen = torch.Generator().manual_seed(SEED)
    q = _stacked(mesh_shape, shape, gen)
    k = _stacked(mesh_shape, shape, gen)
    v = _stacked(mesh_shape, shape, gen)
    jq = jk = jv = None
    if joint_len is not None:
        jshape = (b, h, joint_len, d)
        jq, jk, jv = (_replicate_along(_stacked(mesh_shape, jshape, gen), cluster_axis) for _ in range(3))

    def run():
        return ring_joint_scaled_dot_product_attention(
            _to_mesh(q, mesh_device),
            _to_mesh(k, mesh_device),
            _to_mesh(v, mesh_device),
            _to_mesh(jq, mesh_device),
            _to_mesh(jk, mesh_device),
            _to_mesh(jv, mesh_device),
            logical_n=logical_n,
            cluster_axis=cluster_axis,
            topology=topology,
        )

    out, joint_out = run()

    assert list(out.shape) == [b, h, n, d]
    assert out.dtype == ttnn.bfloat16 and out.layout == ttnn.TILE_LAYOUT
    if joint_len is None:
        assert joint_out is None
    else:
        assert joint_out is not None and list(joint_out.shape) == [b, h, joint_len, d]
        assert joint_out.dtype == ttnn.bfloat16 and joint_out.layout == ttnn.TILE_LAYOUT

    ref = torch_reference(q, k, v, jq, jk, jv, cluster_axis=cluster_axis, logical_n=logical_n)
    outs = _device_tensors(out)
    jouts = _device_tensors(joint_out) if joint_out is not None else None
    threshold = PCC[ttnn.bfloat16]
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        _, p = _members(r, c, cluster_axis, rows, cols)
        valid_rows = (p * n + torch.arange(n)) < logical_n  # padded query rows are unspecified
        exp_out, exp_jout = ref[(r, c)]
        got = outs[idx][:, :, valid_rows]
        assert torch.isfinite(got).all(), f"device ({r},{c}): non-finite output"
        pcc = _pcc(got, exp_out[:, :, valid_rows])
        assert pcc >= threshold, f"device ({r},{c}) output pcc {pcc:.6f} < {threshold}"
        if jouts is not None:
            assert torch.isfinite(jouts[idx]).all(), f"device ({r},{c}): non-finite joint output"
            jpcc = _pcc(jouts[idx], exp_jout)
            assert jpcc >= threshold, f"device ({r},{c}) joint pcc {jpcc:.6f} < {threshold}"

    # Second call: fresh tensors (new addresses -> runtime-arg patch on the cached program); bitwise identical.
    out2, joint_out2 = run()
    for idx, (a, b2) in enumerate(zip(outs, _device_tensors(out2))):
        assert torch.equal(a, b2), f"device {divmod(idx, cols)}: output differs between calls (non-deterministic)"
    if jouts is not None:
        for idx, (a, b2) in enumerate(zip(jouts, _device_tensors(joint_out2))):
            assert torch.equal(a, b2), f"device {divmod(idx, cols)}: joint output differs between calls"
