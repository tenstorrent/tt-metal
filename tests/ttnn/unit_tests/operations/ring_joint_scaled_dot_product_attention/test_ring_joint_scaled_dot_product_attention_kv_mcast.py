# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 pins: K/V row multicast (design regime R2 `qstationary_kv_row_mcast`).

The multicast only changes WHERE a member's K/V block comes from (the row injector's NoC multicast instead of its
own DRAM read), never the data or the compute, so the R2 output must be bit-identical to the R1 output
(`KV_MCAST_GROUP = 1`) — and correct against torch. Every case asserts the row groups were really used (whole
units per core >= 1, several q blocks per head), covering local / remote / joint key segments, the replicated
and sharded joint kinds, GQA stacked heads, both DEST widths, ring and line.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.ring_joint_scaled_dot_product_attention import ring_joint_scaled_dot_product_attention
from ttnn.operations.ring_joint_scaled_dot_product_attention import (
    ring_joint_scaled_dot_product_attention_program_descriptor as pdmod,
)
from eval.golden_tests.ring_joint_scaled_dot_product_attention.topology import ring_mock_meshes


def _ring():
    for mesh_shape, axis, configs in ring_mock_meshes():
        if axis == 1 and ttnn.FabricConfig.FABRIC_2D_TORUS_XY in configs and mesh_shape[1] >= 3:
            return [(mesh_shape, ttnn.FabricConfig.FABRIC_2D_TORUS_XY)]
    return []


# (id, b, hq, hk, n per device, d, joint L, joint mode, fp32 DEST, topology)
CASES = [
    ("mha_bf16", 1, 10, 10, 1024, 128, 0, "none", False, ttnn.Topology.Ring),
    ("mha_fp32_linear", 1, 8, 8, 1024, 128, 0, "none", True, ttnn.Topology.Linear),
    ("gqa_bf16", 1, 16, 4, 1024, 64, 0, "none", False, ttnn.Topology.Ring),
    ("joint_repl_L128", 1, 6, 6, 1024, 128, 128, "replicated", False, ttnn.Topology.Ring),
    ("joint_shard_L333", 1, 6, 6, 1024, 128, 333, "sharded", True, ttnn.Topology.Ring),
]


def _run(mesh_device, mesh_shape, q, k, v, jq, jk, jv, *, logical_n, logical_l, fp32, topology):
    G = mesh_shape[1]
    shard = lambda t: ttnn.from_torch(  # noqa: E731
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_shape),
    )
    repl = lambda t: ttnn.from_torch(  # noqa: E731
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    joint = [None, None, None]
    if jq is not None:
        joint = [shard(t) for t in (jq, jk, jv)] if logical_l else [repl(t) for t in (jq, jk, jv)]
    out, jout = ring_joint_scaled_dot_product_attention(
        shard(q),
        shard(k),
        shard(v),
        *joint,
        logical_n=logical_n,
        logical_l=logical_l,
        cluster_axis=1,
        topology=topology,
        num_links=2 if topology == ttnn.Topology.Ring else 1,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32
        ),
    )
    to = lambda t: ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=mesh_shape)
    )  # noqa: E731,E501
    return to(out), (to(jout) if jout is not None else None)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("mesh_shape,fabric_config", _ring())
def test_kv_mcast_bit_identical(mesh_shape, fabric_config, case, mesh_device, monkeypatch):
    _, b, hq, hk, n, d, L, jmode, fp32, topology = case
    G = mesh_shape[1]
    torch.manual_seed(0)
    q = torch.randn((b, hq, G * n, d), dtype=torch.bfloat16)
    k = torch.randn((b, hk, G * n, d), dtype=torch.bfloat16)
    v = torch.randn((b, hk, G * n, d), dtype=torch.bfloat16)
    jq = jk = jv = None
    logical_l = 0
    if jmode != "none":
        Lp = -(-L // (32 * G)) * 32 * G if jmode == "sharded" else L
        jq = torch.randn((b, hq, Lp, d), dtype=torch.bfloat16)
        jk = torch.randn((b, hk, Lp, d), dtype=torch.bfloat16)
        jv = torch.randn((b, hk, Lp, d), dtype=torch.bfloat16)
        logical_l = L if jmode == "sharded" else 0
    logical_n = G * n - 8
    kw = dict(logical_n=logical_n, logical_l=logical_l, fp32=fp32, topology=topology)

    monkeypatch.setattr(pdmod, "KV_MCAST_GROUP", 1)
    ref_out, ref_j = _run(mesh_device, mesh_shape, q, k, v, jq, jk, jv, **kw)
    assert pdmod.LAST_KV_GROUPS == {}, "KV_MCAST_GROUP = 1 must build no row group"
    monkeypatch.setattr(pdmod, "KV_MCAST_GROUP", None)
    out, jout = _run(mesh_device, mesh_shape, q, k, v, jq, jk, jv, **kw)
    used = pdmod.LAST_KV_GROUPS
    assert used and all(len(g) >= 2 for gs in used.values() for g in gs), f"row groups not used: {used}"

    assert torch.equal(out, ref_out), f"R2 output differs from R1: max {(out.float() - ref_out.float()).abs().max()}"
    if jout is not None:
        assert torch.equal(jout, ref_j), "R2 joint output differs from R1"

    # and correct against torch (one device's slice is enough: every device ran the same scheme)
    kk = k.repeat_interleave(hq // hk, dim=1)
    vv = v.repeat_interleave(hq // hk, dim=1)
    keys, vals = kk[:, :, :logical_n].float(), vv[:, :, :logical_n].float()
    qq = q.float()
    if jq is not None:
        jl = logical_l or L
        keys = torch.cat([keys, jk.repeat_interleave(hq // hk, dim=1)[:, :, :jl].float()], dim=2)
        vals = torch.cat([vals, jv.repeat_interleave(hq // hk, dim=1)[:, :, :jl].float()], dim=2)
    p = torch.softmax(qq[:, :, :n] @ keys.transpose(-1, -2) * d**-0.5, dim=-1)
    exp = p @ vals
    got = out[:b, :, :n].float()
    pcc = torch.corrcoef(torch.stack([got.flatten(), exp.flatten()]))[0, 1].item()
    assert pcc > (0.999 if fp32 else 0.995), f"PCC {pcc}"
