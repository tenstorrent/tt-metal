# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.reduce_scatter of the MoE send-back partials over the SP axis (cluster_axis 0), at the MiMo shape
[1, 1, T, H] bf16 TILE (T = rows x S tokens of a mesh column), against its hyperparameters; exactness vs the torch sum.
Run with --profile; MIMO_RS_T (4096), MIMO_RS_ITERS (5)."""

import itertools
import os

import torch

import ttnn
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

T = int(os.environ.get("MIMO_RS_T", "4096"))
ITERS = int(os.environ.get("MIMO_RS_ITERS", "5"))


@MESH_PARAMS
def test_rs_probe(mesh_device, device_params):
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    rows, cols = tuple(mesh_device.shape)
    sp_topo = per_axis_topology()[0]
    H = 4096
    torch.manual_seed(0)
    parts = torch.randn(rows, cols, T, H).bfloat16()
    x = ttnn.from_torch(
        parts.reshape(rows, cols, T, H).view(rows * cols, 1, T, H).reshape(rows, cols, T, H),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    x = ttnn.reshape(x, (1, 1, T, H))
    ref = parts.float().sum(0)  # [cols, T, H]: column c's sum over the rows
    cases = (
        [dict()]
        + [
            dict(num_workers_per_link=w, num_buffers_per_channel=b)
            for w, b in itertools.product((1, 2, 4, 8), (2, 4, 8))
        ]
        + [dict(num_links=1), dict(num_links=2)]
    )
    if os.environ.get("MIMO_RS_ONLY_DS"):
        cases = []
    # DeepSeek's MoE reduce-scatter: a list of the per-destination slices
    S_ = T // rows
    for mc_name, mc in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
        name = f"ds_moe_rs_{mc_name}"
        try:
            slices = ttnn.split(x, S_, dim=2)
            f = lambda: ttnn.experimental.deepseek_moe_reduce_scatter(
                slices, output_memory_config=mc, dim=2, cluster_axis=0, topology=sp_topo
            )
            o = f()
            ttnn.synchronize_device(mesh_device)
            err = 0.0
            for d, t in enumerate(ttnn.get_device_tensors(o)):
                r, c = divmod(d, cols)
                err = max(err, (ttnn.to_torch(t).float()[0, 0] - ref[c, r * S_ : (r + 1) * S_]).abs().max().item())
            o.deallocate(True)
            for _ in range(ITERS):
                signpost(f"{name}_start")
                f().deallocate(True)
                ttnn.synchronize_device(mesh_device)
                signpost(f"{name}_end")
            print(f"RS_OK {name}: max abs err {err:.3g}")
        except Exception as e:  # noqa: BLE001
            print(f"RS_FAIL {name}: {str(e).splitlines()[0][:300]}")
    for kw in cases:
        name = "rs_" + ("_".join(f"{k.replace('num_', '')}{v}" for k, v in kw.items()) or "default")
        f = lambda: ttnn.reduce_scatter(
            x, dim=2, cluster_axis=0, topology=sp_topo, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw
        )
        try:
            o = f()
            ttnn.synchronize_device(mesh_device)
        except Exception as e:  # noqa: BLE001
            print(f"RS_FAIL {name}: {str(e).splitlines()[0][:200]}")
            continue
        S = T // rows
        err = 0.0
        for d, t in enumerate(ttnn.get_device_tensors(o)):
            r, c = divmod(d, cols)
            got = ttnn.to_torch(t).float()[0, 0]
            err = max(err, (got - ref[c, r * S : (r + 1) * S]).abs().max().item())
        o.deallocate(True)
        for _ in range(ITERS):
            signpost(f"{name}_start")
            f().deallocate(True)
            ttnn.synchronize_device(mesh_device)
            signpost(f"{name}_end")
        print(f"RS_OK {name}: max abs err {err:.3g}")
