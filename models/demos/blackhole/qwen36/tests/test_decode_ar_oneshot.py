# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test for QWEN36_DECODE_AR_ONESHOT (all_reduce_async one_shot through DecodeResidualAllReduce).

test_decode_ar_oneshot_matches_default: the model's own DecodeResidualAllReduce.all_reduce, flag off vs on, on fresh
DRAM partials of R = 1, 4, 8, 16, 32 rows (the decode / verify bucket heights): the RES40 results are byte-identical on
every device and close to the torch sum; with fuse_residual the armed residual + sum is byte-identical as well. The
flag-on instance must actually pass one_shot=True (checked on the wrapper's kwargs).

Run (Galaxy column, ring fabric, 8 KB fabric packets as served):
    QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=0,1,2,3,27,26,25,24 \
        pytest models/demos/blackhole/qwen36/tests/test_decode_ar_oneshot.py -v -s
"""

import os
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

DIM = 5120  # Qwen3.8-27B hidden size


def _device_params():
    params = {
        # QWEN36_FABRIC_RING=1: a Galaxy column's Ring CCLs need the ring (torus) fabric.
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING
        if os.environ.get("QWEN36_FABRIC_RING") == "1"
        else ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 8 << 20,
    }
    router = tpc.fabric_router_config()
    if router is not None:
        params["fabric_router_config"] = router
    return params


def _topology():
    return ttnn.Topology.Ring if os.environ.get("QWEN36_FABRIC_RING") == "1" else ttnn.Topology.Linear


def _mesh_shape():
    from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape

    return _resolve_mesh_shape(8)


def _dar(mesh_device, one_shot, fuse_residual):
    from models.tt_transformers.tt.ccl import TT_CCL

    args = types.SimpleNamespace(dim=DIM, ccl_topology=_topology)
    return tpc.DecodeResidualAllReduce(
        mesh_device, args, TT_CCL(mesh_device), fuse_residual=fuse_residual, one_shot=one_shot
    )


def _to_mesh(mesh_device, t, memory_config, shard):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1) if shard else ttnn.ReplicateTensorToMesh(mesh_device),
    )


@torch.no_grad()
@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_mesh_shape(), id="mesh")], indirect=True)
def test_decode_ar_oneshot_matches_default(mesh_device, reset_seeds, ensure_gc):
    tp = list(mesh_device.shape)[1]
    if tp < 2 or 40 % tp:
        pytest.skip(f"needs a (1, N) mesh with N >= 2 dividing 40, got {list(mesh_device.shape)}")
    seen = []
    orig = ttnn.experimental.all_reduce_async

    def spy(*a, **kw):
        seen.append(bool(kw.get("one_shot", False)))
        return orig(*a, **kw)

    checked = 0
    for fuse in (False, True):
        dars = {"off": _dar(mesh_device, False, fuse), "on": _dar(mesh_device, True, fuse)}
        assert dars["on"].one_shot and not dars["off"].one_shot
        for rows in (1, 4, 8, 16, 32):
            g = torch.Generator().manual_seed(rows + 50 * fuse)
            full = torch.randn(1, tp, rows, DIM, generator=g).to(torch.bfloat16)
            res_t = torch.randn(1, 1, rows, DIM, generator=g).to(torch.bfloat16)
            outs = {}
            for name, dar in dars.items():
                partial = _to_mesh(mesh_device, full, ttnn.DRAM_MEMORY_CONFIG, shard=True)
                residual = _to_mesh(mesh_device, res_t, dar.res_mem, shard=False)
                if fuse:
                    dar.arm_residual(residual)
                seen.clear()
                ttnn.experimental.all_reduce_async = spy
                try:
                    out = dar.all_reduce(partial)
                finally:
                    ttnn.experimental.all_reduce_async = orig
                assert seen == [name == "on"], f"{name}: all_reduce_async one_shot kwargs {seen}"
                if fuse:
                    out = dar.add_residual(residual, out)
                ttnn.synchronize_device(mesh_device)
                outs[name] = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(out)]
                if out is not residual:
                    ttnn.deallocate(out)
                ttnn.deallocate(residual)
            ref = full.float().sum(dim=1, keepdim=True) + (res_t.float() if fuse else 0)
            for dev, (o_off, o_on) in enumerate(zip(outs["off"], outs["on"])):
                assert o_on.shape[-2] == rows, f"rows={rows}: output rows {o_on.shape[-2]}"
                assert torch.equal(o_off, o_on), (
                    f"fuse={fuse} rows={rows} device {dev}: one_shot differs from default "
                    f"(max|d|={(o_off.float() - o_on.float()).abs().max().item():.3e})"
                )
                err = (o_on.float() - ref).abs().max().item()
                assert err <= 0.05 * ref.abs().max().item(), f"rows={rows} device {dev}: |out - torch| = {err:.3e}"
                checked += 1
            logger.info(f"fuse_residual={fuse} rows={rows}: one_shot byte-identical to default on {tp} devices")
    assert checked == 2 * 5 * tp
