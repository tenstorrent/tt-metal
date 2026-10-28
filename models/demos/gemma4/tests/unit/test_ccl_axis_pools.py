# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The CCLManager's per-axis semaphore pools: the lane engine's owner path runs a lane-axis (row) all-reduce
immediately followed by a TP-axis (column) one on the same worker cores. With one shared barrier set the two
axes' reduce-scatters raced on the same barrier word and, under row skew, the writer's reset-after-wait wiped
the other axis's increment (the served Galaxy hangs; tt-triage 2026-10-25 21:07, 2026-10-26 01:14). The
regression gate: many back-to-back alternating-axis all-reduces through one manager on a lane-engine mesh,
each result equal to the sum over every device, and distinct pools per axis."""

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import lanes_fabric_config, parametrize_mesh_with_fabric
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allreduce

MESH_PARAMS = dict(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={"fabric_config": lanes_fabric_config(), "l1_small_size": 65536},
)
H = 2816  # the 26B hidden size
ROWS = 32


def _engine_mesh_config(mesh_device):
    tp, ep = mesh_device.shape[0], mesh_device.shape[1]
    cfg = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tp, ep=ep),
        prefill=ModeConfig(tp=tp, ep=ep),
        tp_axis=0,
        weight_fracture=True,
    )
    cfg.lane_sharded = True
    return cfg


def test_pools_are_distinct_per_axis():
    """Host-only view of the contract: the accessors return different semaphore objects per axis."""
    mgr = object.__new__(CCLManager)
    mgr._sem_pools = {
        key: {
            "rs": [[f"rs{key}"] * 3],
            "ag": [[f"ag{key}"] * 3],
            "barrier": [f"b{key}"],
            "rs_idx": 0,
            "ag_idx": 0,
            "barrier_idx": 0,
        }
        for key in (None, 0, 1)
    }
    assert mgr.get_barrier_semaphore(axis=0) == "b0" and mgr.get_barrier_semaphore(axis=1) == "b1"
    assert mgr.get_barrier_semaphore() == "bNone"
    assert mgr.get_rs_semaphore(axis=1)[0] == "rs1" and mgr.get_ag_semaphore(axis=0)[0] == "ag0"
    barriers, rs, ag = mgr.get_all_reduce_semaphores(1)
    assert barriers == ["b1"] and rs[0] == "rs1" and ag == ["ag1", "ag1"]
    assert mgr.get_all_reduce_semaphores(7)[0] == ["bNone"]  # an axis without a pool falls back


@parametrize_mesh_with_fabric(**MESH_PARAMS)
@pytest.mark.parametrize("iterations", [200])
def test_alternating_axis_all_reduces(mesh_device, iterations):
    """``iterations`` x (lane-axis all-reduce, TP-axis all-reduce) through one manager: every result is the
    sum over all devices (the two axes compose to a full reduction), none of the pairs hangs."""
    shape = tuple(int(x) for x in mesh_device.shape)
    if shape not in ((2, 4), (8, 4)):
        pytest.skip(f"the lane engine's shapes are (2,4) and (8,4); got {shape}")
    cfg = _engine_mesh_config(mesh_device)
    mgr = CCLManager(mesh_device)
    assert mgr._pool(cfg.ep_axis) is not mgr._pool(cfg.tp_axis), "the two axes must not share a semaphore pool"
    assert mgr._pool(cfg.ep_axis)["barrier"][0] is not mgr._pool(cfg.tp_axis)["barrier"][0]
    n_dev = shape[0] * shape[1]
    torch.manual_seed(0)
    per_device = torch.randn(n_dev, 1, ROWS, H, dtype=torch.bfloat16)
    full_sum = per_device.float().sum(0, keepdim=True)  # [1, 1, ROWS, H]
    x = ttnn.from_torch(
        per_device,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    checks = {0, 1, 2, iterations // 2, iterations - 1}
    worst = 0.0
    for i in range(iterations):
        # ccl_allreduce consumes its input (deallocates it after the collective): hand it a fresh copy each time
        xi = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        y = ccl_allreduce(xi, cfg, mgr, axis=cfg.ep_axis)  # the lane (row) axis
        z = ccl_allreduce(y, cfg, mgr, axis=cfg.tp_axis)  # the TP (column) axis
        if i in checks:
            out = ttnn.to_torch(z, mesh_composer=composer).float()  # [n_dev, 1, ROWS, H]
            err = (out - full_sum).abs().max().item() / full_sum.abs().max().item()
            worst = max(worst, err)
            assert err < 2e-2, f"iteration {i}: relative error {err:.4f} vs the full-mesh sum"
        z.deallocate(True)  # y was consumed by the second all-reduce
    ttnn.synchronize_device(mesh_device)
    print(f"RESULT ccl_axis_pools {shape} iterations={iterations} worst_rel_err={worst:.4f}")
