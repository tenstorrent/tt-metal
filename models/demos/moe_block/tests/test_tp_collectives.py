# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The MoE block's TP collectives on the LoudBox 2x4 (cluster_axis 1, a line of 4), per candidate implementation:
device time (slowest chip, median) and agreement with ttnn's. Shapes per chip at chunk 5120 (2560 rows a mesh row,
640 owned a chip) for GLM-5.3-Flash (H 4096) and K3 (H 7168, latent 3584). Needs TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 for the times."""

import os
import statistics

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.tests.test_moe_block import per_chip_programs

R, C = 2, 4
ITERS = 5


def _dev(mesh, shape, seed, dtype=ttnn.bfloat16):
    t = torch.randn(R * shape[0], C * shape[1], generator=torch.Generator().manual_seed(seed))
    return t, ttnn.from_torch(
        t.view(1, 1, R * shape[0], C * shape[1]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(2, 3)),
    )


def _time(mesh, f):
    """Median over ITERS of the slowest chip's summed program time of one f() call (us); None without the profiler."""
    out = f()
    ttnn.synchronize_device(mesh)
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return None, out
    per_chip_programs(mesh)
    ts = []
    for _ in range(ITERS):
        if out is not None and getattr(out, "is_allocated", lambda: False)():
            ttnn.deallocate(out)
        out = f()
        ttnn.synchronize_device(mesh)
        progs = [r for r in per_chip_programs(mesh) if r]
        ts.append(max(sum(r) for r in progs))
    return statistics.median(ts), out


def _host(mesh, t, dims):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(R, C), dims=dims)).float()


CASES = {
    # name: (rows, cols) per chip of the input, op kind, gather / scatter dim
    "glm_x_ag": ((2560, 1024), "ag", 3),
    "k3_x_ag": ((2560, 1792), "ag", 3),
    "glm_tail_ag": ((640, 4096), "ag", 2),
    "k3_tail_ag": ((640, 3584), "ag", 2),
    "glm_se_rs": ((2560, 4096), "rs", 3),
    "k3_se_rs": ((2560, 7168), "rs", 3),
    "glm_se_rs_rows": ((2560, 4096), "rs", 2),
    "glm_logits_rs": ((2560, 288), "rs32", 2),
    "k3_logits_rs": ((2560, 448), "rs32", 2),
    # all-to-alls: (in_dim, out_dim) - K3's latent slices to the owners; the non-latent tail back to the width split
    "k3_lat_a2a": ((2560, 896), "a2a", (2, 3)),
    "glm_tail_a2a": ((640, 4096), "a2a", (3, 2)),
    "k3_se_rs_rows": ((2560, 7168), "rs", 2),
}


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_tp_collectives(mesh_device, device_params):
    sel = os.environ.get("TPC_CASES")
    for name, (shape, kind, dim) in CASES.items():
        if sel and name not in sel.split(","):
            continue
        host, x = _dev(mesh_device, shape, 1, ttnn.float32 if kind == "rs32" else ttnn.bfloat16)
        if kind == "rs32":
            kind = "rs"
        rows, cols = shape
        G = C
        res, ref = {}, None
        if kind == "ag":
            out_shape = [1, 1, rows * (G if dim == 2 else 1), cols * (G if dim == 3 else 1)]
            cands = {
                "ttnn": lambda: ttnn.all_gather(x, dim=dim, cluster_axis=1),
            }
            pre = ttnn.allocate_tensor_on_device(
                ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
            cands["high_bw"] = lambda: ttnn.experimental.high_bw_all_gather(x, dim, pre, cluster_axis=1, num_links=2)
            cands["fabric_ag"] = lambda: ttnn.bringup.fabric_all_gather(x, dim, pre, cluster_axis=1, num_links=2)
            if dim == 2:
                from ttnn.bringup.combine_rf.fabric_all_gather_py import fabric_all_gather as fag_py

                cands["fabric_ag_py"] = lambda: fag_py(
                    x, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=2, dim=2
                )
        elif kind == "a2a":
            ind, outd = dim
            cands = {
                "ref_ag_part": lambda: ttnn.mesh_partition(
                    ttnn.all_gather(x, dim=outd, cluster_axis=1), dim=ind, cluster_axis=1
                ),
                "a2a_generic": lambda: ttnn.experimental.all_to_all_async_generic(
                    x, in_dim=ind, out_dim=outd, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear
                ),
                "a2a_generic_swap": lambda: ttnn.experimental.all_to_all_async_generic(
                    x, in_dim=outd, out_dim=ind, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear
                ),
            }
        else:
            cands = {
                "ttnn": lambda: ttnn.reduce_scatter(x, dim=dim, cluster_axis=1),
                "ttnn_l2": lambda: ttnn.reduce_scatter(x, dim=dim, cluster_axis=1, num_links=2),
            }
            if dim == 2:
                from ttnn.bringup.fabric_reduce_scatter_ttnn.fabric_reduce_scatter import fabric_reduce_scatter

                cands["fabric_rs"] = lambda: fabric_reduce_scatter(
                    x, cluster_axis=1, topology=ttnn.Topology.Linear, num_links=2
                )
        for cname, f in cands.items():
            try:
                us, out = _time(mesh_device, f)
                logger.info(f"TPC {name} {cname}: out per chip {list(out.shape)}")
                y = _host(mesh_device, out, (2, 3))
                if cname in ("ttnn", "ref_ag_part"):
                    ref = y
                err = (y - ref).abs().max().item() if ref is not None else float("nan")
                res[cname] = (us, err)
                if cname not in ("high_bw", "fabric_ag"):
                    ttnn.deallocate(out)
            except Exception as e:  # (report and go on: some don't support the dim / layout)
                res[cname] = (None, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
        mb = rows * cols * 2 / 1e6
        logger.info(
            f"TPC {name:16s} {kind} dim {dim} shard {mb:.1f} MB: "
            + "; ".join(f"{k} {v[0] if v[0] is None else round(v[0], 1)} us (max err {v[1]})" for k, v in res.items())
        )
        ttnn.deallocate(x)
