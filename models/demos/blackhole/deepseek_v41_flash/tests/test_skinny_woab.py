# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test + trace-replay timing of the attention-tail chain (tt/skinny.py SkinnyWoAB: wo_a -> wo_b on the 8 DRAM-bank cores) against torch (dequantised bfp8 weights) and against the stock chain
(ttnn.linear x2 with the model's program configs).  Env: SK_T (rows, default 8), SK_ARMS (comma list of 'skinny', 'stock'; default both), SK_REP (ops per trace, 4), SK_TREPS (3), FD_OUT (output dir).
Output: <FD_OUT>/layer_arms.json (arm -> trace_id, PCC), optab.py reads it with prof/.logs/cpp_device_perf_report.csv."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import attention as A
from models.demos.blackhole.deepseek_v41_flash.tt import skinny as SK

OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/skinnymm/run_woab")
T = int(os.environ.get("SK_T", "8"))
REP = int(os.environ.get("SK_REP", "4"))
TREPS = int(os.environ.get("SK_TREPS", "3"))
ARMS = os.environ.get("SK_ARMS", "skinny,stock").split(",")


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.timeout(2400)
@torch.no_grad()
def test_skinny_woab(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    rows, cols = tuple(md.shape)
    torch.manual_seed(3)
    wo_a = (torch.randn(4096, cols * 1024) * 0.03).bfloat16().float()  # [4096, 8192]
    wo_b = (torch.randn(cols * 1024, 5120) * 0.03).bfloat16().float()  # [8192, 5120]
    c_host = torch.randn(1, 1, 32, 4608).bfloat16().float() * 0.5
    c_host[:, :, T:, :] = 0
    mk_c = lambda: ttnn.from_torch(
        c_host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    c = mk_c()
    arms = {}
    # reference with the dequantised weights
    deq = lambda w, mapper_dim: ttnn.to_torch(
        ttnn.from_torch(
            w.reshape(1, 1, *w.shape),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, mapper_dim), mesh_shape=(rows, cols)),
        ),
        mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, mapper_dim), mesh_shape=(rows, cols)),
    ).float()
    wa_d = deq(wo_a, 3)[:1]  # [1,1,4096,8192] (row 0 of the mesh)
    wb_d = deq(wo_b, 2)[:1]
    xx = c_host[0, 0, :T, 512:]  # [T, 4096]
    pccs = {}

    def check(part, name):
        outs = ttnn.get_device_tensors(part)
        res = []
        for chip in (0, 5, 13, 31):
            col = chip % cols
            got = ttnn.to_torch(outs[chip]).float().reshape(32, 5120)[:T]
            y = (xx @ wa_d.reshape(4096, cols * 1024)[:, col * 1024 : (col + 1) * 1024]).bfloat16().float()
            ref = y @ wb_d.reshape(cols * 1024, 5120)[col * 1024 : (col + 1) * 1024]
            res.append(_pcc(got, ref))
        pccs[name] = res
        print("PCC", name, res, flush=True)

    skin = SK.SkinnyWoAB(md, wo_a, wo_b)
    stock = None
    stock_fid = {}
    if "stock" in ARMS:
        # stock chain with the zero rows in front like the model
        wa_full = torch.cat([torch.zeros(512, cols * 1024), wo_a]).reshape(1, 1, 4608, cols * 1024)
        wa_s = ttnn.from_torch(
            wa_full,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
        )
        wb_s = ttnn.from_torch(
            wo_b.reshape(1, 1, cols * 1024, 5120),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=(rows, cols)),
        )
        ckc = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        pa = A._proj_kwargs("OA", 4608, 1024)
        pb = A._proj_kwargs("OB", 1024, 5120)
        stock = lambda: ttnn.linear(
            ttnn.linear(c, wa_s, compute_kernel_config=ckc, **pa), wb_s, compute_kernel_config=ckc, **pb
        )
        stock_fid.clear()
        for nm, fid in (("stock_lofi", ttnn.MathFidelity.LoFi), ("stock_h2", ttnn.MathFidelity.HiFi2)):
            ck = ttnn.init_device_compute_kernel_config(
                md.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
            )
            stock_fid[nm] = lambda ck=ck: ttnn.linear(
                ttnn.linear(c, wa_s, compute_kernel_config=ck, **pa), wb_s, compute_kernel_config=ck, **pb
            )

    fns = {}
    for nc in (8, 16):
        if f"ds{nc}" not in ARMS:
            continue
        # stock DRAM-sharded matmuls (weights width-sharded over the 8 banks, in0 L1 width-sharded over nc cores), model shapes (K = 4608 with the zero rows)
        dram_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
        )
        dmem = lambda K, N: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_crs, (K, N // 8), ttnn.ShardOrientation.ROW_MAJOR),
        )
        wa_full = torch.cat([torch.zeros(512, cols * 1024), wo_a]).reshape(1, 1, 4608, cols * 1024)
        wa_ds = ttnn.from_torch(
            wa_full,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=dmem(4608, 1024),
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=(rows, cols)),
        )
        wb_ds = ttnn.from_torch(
            wo_b.reshape(1, 1, cols * 1024, 5120),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=dmem(1024, 5120),
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=(rows, cols)),
        )
        crs = ttnn.num_cores_to_corerangeset(nc, md.compute_with_storage_grid_size(), row_wise=True)
        sh = lambda width: ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, (32, width // nc), ttnn.ShardOrientation.ROW_MAJOR),
        )
        pcf = lambda K, N: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=max(d for d in range(1, K // 32 // nc + 1) if (K // 32 // nc) % d == 0 and d <= 16),
            per_core_M=1,
            per_core_N=-(-N // 32 // nc),
            fused_activation=None,
        )
        ckc2 = ttnn.init_device_compute_kernel_config(
            md.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def mk(nc=nc, wa_ds=wa_ds, wb_ds=wb_ds, sh=sh, pcf=pcf, ckc2=ckc2):
            def f():
                cs = ttnn.to_memory_config(c, sh(4608))
                y = ttnn.linear(
                    cs,
                    wa_ds,
                    program_config=pcf(4608, 1024),
                    compute_kernel_config=ckc2,
                    memory_config=sh(1024),
                    dtype=ttnn.bfloat16,
                )
                o = ttnn.linear(
                    y,
                    wb_ds,
                    program_config=pcf(1024, 5120),
                    compute_kernel_config=ckc2,
                    memory_config=sh(5120),
                    dtype=ttnn.bfloat16,
                )
                return ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)

            return f

        fns[f"ds{nc}"] = mk()
    if "skinny" in ARMS:
        fns["skinny"] = lambda: skin(c, T)
    if "skinny_c" in ARMS:
        # compact x: tiny 8 x 32 tiles (faces 0 / 1 of 8 rows) of row tile m at tile m * 160, 16 KiB pages
        mt = 1 if T <= 8 else 2
        buf = torch.zeros(mt, 160, 2, 8, 16)
        for m in range(mt):
            for kt in range(144):
                blk = c_host[0, 0, m * 8 : m * 8 + 8, kt * 32 : kt * 32 + 32]  # [8, 32]
                buf[m, kt, 0] = blk[:, :16]
                buf[m, kt, 1] = blk[:, 16:]
        cc = ttnn.from_torch(
            buf.reshape(81920 * mt // SK.AFPB(), SK.AFPB() // 2),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=md,
            memory_config=ttnn.L1_MEMORY_CONFIG
            if os.environ.get("DSV41_SKINNY_XL1") == "1"
            else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        fns["skinny_c"] = lambda: skin(cc, T)
    if stock is not None:
        fns["stock"] = stock
        for nm_, fn_ in stock_fid.items():
            if nm_ in ARMS:
                fns[nm_] = fn_
    for name, fn in fns.items():
        part = fn()
        ttnn.synchronize_device(md)
        check(part, name)
        ttnn.ReadDeviceProfiler(md)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(REP):
            fn()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        for _ in range(TREPS):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
        arms[name] = dict(trace_id=int(tid), pcc=pccs.get(name))
        json.dump(dict(arms=arms, layer=-1), open(os.path.join(OUT, "layer_arms.json"), "w"))
    print("SK_DONE " + json.dumps(arms), flush=True)
