# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""high_bw_all_gather vs ttnn.all_gather over cluster_axis 0 on the MiMo 2x2 FABRIC_2D mesh, at MoE shapes:
x (S, 4096) bf16 row-major / tile, topk indices (S, 8) uint16 and weights (S, 8) bf16 row-major.

    scripts/run_safe_pytest.sh --profile models/demos/mimo_v2_d_p/tests/perf/test_hbw_ag_probe.py
"""

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

SEQS = [int(s) for s in os.environ.get("MIMO_CCL_SEQ", "640,2048").split(",")]
LINKS = [int(s) for s in os.environ.get("MIMO_HBW_LINKS", "1,2,4").split(",")]
ITERS = int(os.environ.get("MIMO_CCL_ITERS", "3"))


def _timed(dev, tag, fn):
    fn()
    ttnn.synchronize_device(dev)
    for _ in range(ITERS):
        signpost(f"{tag}_start")
        fn()
        ttnn.synchronize_device(dev)
        signpost(f"{tag}_end")


@MESH_PARAMS
def test_hbw_ag_probe(mesh_device, device_params):
    rows, cols = tuple(mesh_device.shape)
    sp_topo, _ = per_axis_topology()
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    for S in SEQS:
        cases = [
            ("x_rm", torch.randn(1, 1, rows * S, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            ("x_tile", torch.randn(1, 1, rows * S, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            ("idx", torch.randint(0, 256, (1, 1, rows * S, 8)), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
            ("w", torch.rand(1, 1, rows * S, 8), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ]
        for name, t, dt, lay in cases:
            x = ttnn.from_torch(
                t, device=mesh_device, layout=lay, dtype=dt, mesh_mapper=shard, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = ttnn.from_torch(
                torch.zeros_like(t),
                device=mesh_device,
                layout=lay,
                dtype=dt,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ref = t if dt != ttnn.bfloat16 else t.bfloat16()
            for L in LINKS:
                try:
                    _timed(
                        mesh_device,
                        f"hbw_{name}_S{S}_l{L}",
                        lambda: ttnn.experimental.high_bw_all_gather(
                            x, dim=2, output_tensor=out, cluster_axis=0, num_links=L
                        ),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"HBW_FAIL {name} S{S} L{L}: {str(e).splitlines()[0][:300]}")
                    continue
                for d in ttnn.get_device_tensors(out):
                    got = ttnn.to_torch(d)
                    assert torch.equal(got.to(ref.dtype), ref), f"hbw {name} S{S} L{L} mismatch"
                print(f"HBW_OK {name} S{S} L{L}")
            _timed(
                mesh_device,
                f"ag_{name}_S{S}",
                lambda: ttnn.deallocate(ttnn.all_gather(x, dim=2, cluster_axis=0, topology=sp_topo)),
            )
            ttnn.deallocate(x)
            ttnn.deallocate(out)
