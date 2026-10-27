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


CHECK_CALLS = 3  # (calls without the profiler: each on its own inputs, each checked)


def ncalls():
    """The calls a checked test makes: 1 + ITERS with the profiler (call 0 untimed), else CHECK_CALLS."""
    return 1 + ITERS if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1" else CHECK_CALLS


def gen(seed, k):
    """Call k's input generator (every call its own inputs: stale data from the previous call cannot pass)."""
    return torch.Generator().manual_seed(1000 * seed + k)


def _calls(mesh, run, check, free=True):
    """ncalls() calls, call k on its own inputs (run(k) -> out) and checked against its own reference (check(k, out) ->
    its result; out deallocated after unless not free: a persistent buffer). Returns (median us of the slowest chip's summed program time over calls 1 ..,
    None without the profiler; the checks' results)."""
    prof = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    ts, res = [], []
    for k in range(ncalls()):
        out = run(k)
        ttnn.synchronize_device(mesh)
        if prof:
            progs = [r for r in per_chip_programs(mesh) if r]
            if k > 0:
                ts.append(max(sum(r) for r in progs))
                if os.environ.get("TIME_SPLIT"):  # (each program's slowest chip)
                    logger.info(f"TIME_SPLIT {[round(max(r[i] for r in progs), 1) for i in range(len(progs[0]))]}")
        res.append(check(k, out))
        if free and out is not None and getattr(out, "is_allocated", lambda: False)():
            ttnn.deallocate(out)
    return (statistics.median(ts) if ts else None), res


def _shard(xh, shape, r, c):
    rows, cols = shape
    return xh[r * rows : (r + 1) * rows, c * cols : (c + 1) * cols]


def _coll_ref(xh, shape, kind, dim):
    """The collective's every chip's output from the host x [R rows, C cols] (ag / rs / a2a over the TP axis), laid
    out as _host(.., (2, 3)) lays the device's: chips' outputs concatenated by mesh row (dim 2) and column (dim 3)."""
    d = {2: 0, 3: 1}
    out = []
    for r in range(R):
        sh = [_shard(xh, shape, r, c).float() for c in range(C)]
        if kind == "ag":
            per = [torch.cat(sh, dim=d[dim])] * C
        elif kind == "a2a":
            ind, outd = dim
            g = torch.cat(sh, dim=d[outd])
            per = list(torch.chunk(g, C, dim=d[ind]))
        else:
            per = list(torch.chunk(sum(sh), C, dim=d[dim]))
        out.append(torch.cat(per, dim=1))
    return torch.cat(out, dim=0)


def _host(mesh, t, dims):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(R, C), dims=dims)).float()


CASES = {
    # name: (rows, cols) per chip of the input, op kind, gather / scatter dim
    "glm_x_ag": ((2560, 1024), "ag", 3),
    "k3_x_ag": ((2560, 1792), "ag", 3),
    "glm_x_ag640": ((640, 1024), "ag", 3),  # (Galaxy per-chip rows)
    "k3_x_ag640": ((640, 1792), "ag", 3),
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
    "glm_se_rs_rows640": ((640, 4096), "rs", 2),  # (the SE reduce-scatter at Galaxy per-chip rows)
    "k3_se_rs640": ((640, 7168), "rs", 3),
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
        xdt = ttnn.float32 if kind == "rs32" else ttnn.bfloat16
        xk = [_dev(mesh_device, shape, 1000 + k, xdt) for k in range(ncalls())]  # (every call its own x)
        host, x = xk[0]
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
                # (all_to_all_async_generic's in / out dims: the other way round from mesh_partition(all_gather))
                "a2a_generic": lambda: ttnn.experimental.all_to_all_async_generic(
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
        refs = [
            _coll_ref(h_.to(torch.bfloat16).float() if xdt == ttnn.bfloat16 else h_, shape, kind, dim) for h_, _ in xk
        ]

        # (call k vs the host's collective of its own x: moves exact; sums within bf16 rounding of the largest)
        tol = 0.0 if kind in ("ag", "a2a") else 2.0**-6

        def check(k, out):
            err = (_host(mesh_device, out, (2, 3)) - refs[k]).abs().max().item()
            assert err <= tol * refs[k].abs().max().item(), (name, k, err)
            return err

        for cname, f in cands.items():

            def run(k, f=f):
                nonlocal x
                x = xk[k][1]  # (the candidates' lambdas read x)
                return f()

            us, errs = _calls(mesh_device, run, check, free=cname not in ("high_bw", "fabric_ag"))
            res[cname] = (us, max(errs))
        mb = rows * cols * 2 / 1e6
        logger.info(
            f"TPC {name:16s} {kind} dim {dim} shard {mb:.1f} MB: "
            + "; ".join(
                f"{k} {v[0] if v[0] is None else round(v[0], 1)} us (max err vs host, worst call {v[1]})"
                for k, v in res.items()
            )
        )
        for _, t in xk:
            ttnn.deallocate(t)


AGMM_CASES = {
    # name: (rows, H / TP per chip, gate / up columns per chip (2 I_s / TP))
    "glm_x_gu": (2560, 1024, 1024),
    "k3_x_gu": (2560, 1792, 3072),
}


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_agmm_bar(mesh_device, device_params):
    """The bar for the fused head's x all-gather -> shared expert gate / up: ttnn.all_gather + the block's mm2d linear
    (HiFi2, bf8 weights). (all_gather_minimal_matmul_async, the fused gather-over-K matmul, throws at these shapes in
    its program build - dropped.)"""
    import models.demos.glm53_flash_d_p.tt.mm_configs as mmc

    cfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    for name, (rows, kl, n) in AGMM_CASES.items():
        K = kl * C
        xk = [_dev(mesh_device, (rows, kl), 1000 + k) for k in range(ncalls())]  # (every call its own x)
        x = xk[0][1]
        wt = torch.randn(K, n * C, generator=torch.Generator().manual_seed(2)) * 0.02
        res = {}
        for wdt in (ttnn.bfloat8_b,):
            w = ttnn.from_torch(
                wt.view(1, 1, K, n * C),
                dtype=wdt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(None, 3)),
            )

            def sep():
                xf = ttnn.all_gather(x, dim=3, cluster_axis=1)
                y = ttnn.linear(
                    xf,
                    w,
                    compute_kernel_config=cfg,
                    dtype=ttnn.bfloat16,
                    program_config=mmc.linear_config(xf, w, ttnn.bfloat16),
                )
                ttnn.deallocate(xf)
                return y

            variants = {"ag+mm2d": sep}
            # (call k vs the host's: each mesh row's gathered x [rows, K] @ the chip's W columns)
            refs = [
                torch.cat(
                    [
                        torch.cat(
                            [
                                h_[r * rows : (r + 1) * rows].to(torch.bfloat16).float() @ wt[:, c * n : (c + 1) * n]
                                for c in range(C)
                            ],
                            1,
                        )
                        for r in range(R)
                    ],
                    0,
                )
                for h_, _ in xk
            ]

            def check(k, out):
                y = _host(mesh_device, out, (2, 3))
                p = torch.corrcoef(torch.stack([y.flatten(), refs[k].flatten()]))[0, 1].item()
                assert p > 0.999, (name, k, p)
                return p

            for vn, f in variants.items():

                def run(k, f=f):
                    nonlocal x
                    x = xk[k][1]  # (the variants read x)
                    return f()

                us, ps = _calls(mesh_device, run, check)
                res[vn] = (us, round(min(ps), 5))
            ttnn.deallocate(w)
        logger.info(
            f"AGMM {name} rows {rows} K {K} N/chip {n}: "
            + "; ".join(
                f"{k} {v[0] if v[0] is None else round(v[0], 1)} us (PCC vs host, worst call {v[1]})"
                for k, v in res.items()
            )
        )
        for _, t in xk:
            ttnn.deallocate(t)
