# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Scratch probe: matmul program configs for the K3 block's dense ops at 320 rows per chip (device time, slowest
chip, vs the DRAM floor of the weight). Shapes: latent down / up, shared gate / up / down, router."""
import os

import pytest
import torch
from loguru import logger

import models.demos.glm53_flash_d_p.tt.mm_configs as mmc
import ttnn
from models.demos.deepseek_v3.utils.config_helpers import dram_sharded_weight_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params

M = 320
SHAPES = {  # name: (K, N, weight dtype)
    "lat_down": (7168, 3584, ttnn.bfloat8_b),
    "lat_up": (3584, 7168, ttnn.bfloat8_b),
    "uni": (7168, 3072, ttnn.bfloat8_b),  # (lin2d debug: 8 tiles a column)
    "se_gu2": (7168, 12288, ttnn.bfloat8_b),  # (shared gate + up as one linear)
    "se_gu": (7168, 6144, ttnn.bfloat8_b),
    "se_down": (6144, 7168, ttnn.bfloat8_b),
    "router": (7168, int(os.environ.get("MM_E", "448")), ttnn.bfloat16),
}
_PARAMS = [pytest.param((2, 4), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="2x4")]
_KEY = "DEVICE KERNEL DURATION [ns]"


def _times(mesh_device):
    ttnn.ReadDeviceProfiler(mesh_device)
    rows = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        rows.append(
            [
                float(p.program_analyses_results[_KEY].duration) / 1e3
                for p in programs
                if _KEY in (p.program_analyses_results or {})
            ]
        )
    return rows


def _configs(Kt, Nt, gx, gy):
    """(label, program_config) candidates."""
    out = [("default", None)]
    n = gx * gy
    for bw in (2, 4, 8):
        if Kt % bw:
            continue
        pn = -(-Nt // n)
        sw = max(d for d in (1, 2, 4, 8) if pn % d == 0 and d <= 8)
        out.append(
            (
                f"1d_in0 bw{bw} pn{pn}",
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=bw,
                    out_subblock_h=1,
                    out_subblock_w=sw,
                    out_block_h=10,
                    out_block_w=pn,
                    per_core_M=10,
                    per_core_N=pn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                ),
            )
        )
    for bw in (4, 8, 16):  # 1D in0 multicast, every core reads its own weight columns, taller output subblocks
        if Kt % bw:
            continue
        pn = -(-Nt // n)
        for sh in (2, 5, 10):
            sw = max(d for d in (1, 2, 4, 8) if pn % d == 0 and d * sh <= 8) if sh <= 8 else 0
            if sw == 0 or 10 % sh:
                continue
            out.append(
                (
                    f"1d_in0 bw{bw} pn{pn} sh{sh}",
                    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                        in0_block_w=bw,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        out_block_h=10,
                        out_block_w=pn,
                        per_core_M=10,
                        per_core_N=pn,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    ),
                )
            )
    for gyy in (5, 10):
        pm = -(-10 // gyy)
        pn = -(-Nt // gx)
        for bw in (2, 4, 8):
            if Kt % bw:
                continue
            sw = max(d for d in (1, 2, 4, 8) if pn % d == 0 and d * pm <= 8)
            out.append(
                (
                    f"2d {gx}x{gyy} bw{bw} pm{pm} pn{pn}",
                    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gyy),
                        in0_block_w=bw,
                        out_subblock_h=1,
                        out_subblock_w=sw,
                        out_block_h=pm,
                        out_block_w=pn,
                        per_core_M=pm,
                        per_core_N=pn,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=True,
                    ),
                )
            )
    return out


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_dense_mm(mesh_device, device_params):
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    g = mesh_device.compute_with_storage_grid_size()
    names = os.environ.get("MM_SHAPES", ",".join(SHAPES)).split(",")
    fids = [ttnn.MathFidelity[f] for f in os.environ.get("MM_FID", "HiFi2,LoFi").split(",")]
    for name in names:
        Kd, Nd, wdt = SHAPES[name]
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = (torch.randn(Kd, Nd) * 0.02).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wts = {"il": wt}
        try:
            wts["ds"] = ttnn.from_torch(
                w,
                dtype=wdt,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=rep,
                memory_config=dram_sharded_weight_config(Kd, Nd, mesh_device.dram_grid_size()),
            )
        except Exception as e:  # noqa: BLE001
            logger.info(f"MM {name} dram-sharded weight FAILED {str(e).splitlines()[0][:150]}")
        mmc.ENABLED = True
        mm2d = mmc.mm2d(mesh_device, M, Kd, Nd, ttnn.bfloat16, wdt, ttnn.bfloat16)
        ref = x.float() @ w.float()
        wbytes = Kd * Nd * (1.0625 if wdt == ttnn.bfloat8_b else 2)
        floor = wbytes / 512e3
        best = None
        for fid in fids:
            cfg = ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            cands = [(f"il {l}", pc, "il") for l, pc in _configs(Kd // 32, Nd // 32, g.x, g.y)]
            cands += [("il mm2d", mm2d, "il")] + ([("ds mm2d", mm2d, "ds")] if "ds" in wts else [])
            # DRAM-sharded weights: one core column per bank, its column block = the bank's shard (N / 8 tile columns)
            nb = mesh_device.dram_grid_size().x
            if "ds" in wts and (Nd // 32) % nb == 0:
                pn = (Nd // 32) // nb
                for gy, bw in ((10, 8), (10, 16), (5, 8), (5, 16)):
                    pm = -(-10 // gy)
                    if (Kd // 32) % bw:
                        continue
                    sw = max(d_ for d_ in (1, 2, 4, 8) if pn % d_ == 0 and d_ * pm <= 8)
                    cands.append(
                        (
                            f"ds 2d {nb}x{gy} bw{bw} pn{pn}",
                            ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                                compute_with_storage_grid_size=ttnn.CoreCoord(nb, gy),
                                in0_block_w=bw,
                                out_subblock_h=1,
                                out_subblock_w=sw,
                                out_block_h=pm,
                                out_block_w=pn,
                                per_core_M=pm,
                                per_core_N=pn,
                                transpose_mcast=False,
                                fused_activation=None,
                                fuse_batch=True,
                            ),
                            "ds",
                        )
                    )
            for label, pc, wk in cands:
                try:
                    _times(mesh_device)
                    us = []
                    for _ in range(3):
                        y = ttnn.linear(xt, wts[wk], compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc)
                        ttnn.synchronize_device(mesh_device)
                        us.append(max(r[-1] for r in _times(mesh_device)))
                        yt = y
                    got = ttnn.to_torch(ttnn.get_device_tensors(yt)[0]).float()
                    p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                    ttnn.deallocate(yt)
                    t = min(us)
                    logger.info(
                        f"MM {name:8s} {str(fid).split('.')[-1]:5s} {label:28s} {t:8.1f} us  ({floor / t * 100:5.1f}% of DRAM floor {floor:.1f})  pcc {p:.5f}"
                    )
                    if best is None or t < best[0]:
                        best = (t, str(fid), label)
                except Exception as e:  # noqa: BLE001
                    logger.info(
                        f"MM {name:8s} {str(fid).split('.')[-1]:5s} {label:28s} FAILED {str(e).splitlines()[0][:150]}"
                    )
        logger.info(f"MM_BEST {name}: {best}  floor {floor:.1f}")
        ttnn.deallocate(xt)
        for t in wts.values():
            ttnn.deallocate(t)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_minimal_mm(mesh_device, device_params):
    """ttnn.experimental.minimal_matmul (in0 must match the weight's dtype: bf8 x, plus the typecast)."""
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    g = mesh_device.compute_with_storage_grid_size()
    fid = ttnn.MathFidelity[os.environ.get("MM_FID", "HiFi2")]
    cfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    for name in os.environ.get("MM_SHAPES", "lat_down,lat_up").split(","):
        Kd, Nd, wdt = SHAPES[name]
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = (torch.randn(Kd, Nd) * 0.02).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        ref = x.float() @ w.float()
        floor = Kd * Nd * 1.0625 / 512e3

        def timed(label, f):
            try:
                _times(mesh_device)
                best, y = None, None
                for _ in range(3):
                    y = f()
                    ttnn.synchronize_device(mesh_device)
                    rows = _times(mesh_device)
                    t = max(sum(r) for r in rows)
                    progs = [round(max(r[i] for r in rows), 1) for i in range(min(len(r) for r in rows))]
                    best = (t, progs) if best is None or t < best[0] else best
                got = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
                p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                ttnn.deallocate(y)
                logger.info(
                    f"MMIN {name:8s} {label:34s} {best[0]:8.1f} us {best[1]}  ({floor / best[0] * 100:5.1f}% of floor {floor:.1f})  pcc {p:.5f}"
                )
            except Exception as e:  # noqa: BLE001
                logger.info(f"MMIN {name:8s} {label:34s} FAILED {str(e).splitlines()[0][:160]}")

        mmc.ENABLED = True
        timed(
            "linear mm2d (bf16 x)",
            lambda: ttnn.linear(
                xt,
                wt,
                compute_kernel_config=cfg,
                dtype=ttnn.bfloat16,
                program_config=mmc.mm2d(mesh_device, M, Kd, Nd, ttnn.bfloat16, wdt, ttnn.bfloat16),
            ),
        )

        def mm(c):
            def f():
                x8 = ttnn.typecast(xt, ttnn.bfloat8_b)
                y = ttnn.experimental.minimal_matmul(x8, wt, config=c, dtype=ttnn.bfloat16, compute_kernel_config=cfg)
                ttnn.deallocate(x8)
                return y

            return f

        timed("minimal default", mm(None))
        Mt, Kt, Nt = M // 32, Kd // 32, Nd // 32
        for mb in (1, 2, 5, 10):
            for nb in (1, 2, 4, 8):
                for kb in (4, 8, 16):
                    for sh, sw in ((1, 1), (1, 2), (1, 4), (2, 2), (2, 1), (5, 1)):
                        if mb % sh or nb % sw or sh * sw > 4:
                            continue
                        if (sh, sw) not in (
                            (1, max(d for d in (1, 2, 4) if nb % d == 0)),
                            (min(mb, 2) if mb % 2 == 0 else (5 if mb % 5 == 0 else 1), 1),
                        ):
                            continue
                        c = ttnn.MinimalMatmulConfig(
                            M_block_size=mb,
                            K_block_size=kb,
                            N_block_size=nb,
                            subblock_h=sh,
                            subblock_w=sw,
                            compute_with_storage_grid_size=g,
                        )
                        timed(f"minimal M{mb} K{kb} N{nb} sb{sh}x{sw}", mm(c))
        ttnn.deallocate(xt)
        ttnn.deallocate(wt)


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
def test_sharded_in0_mm(mesh_device, device_params):
    """in0 width-sharded in L1 (no single-core in0 DRAM read): 1D mcast-in0 over n cores (each reads its own weight
    columns), and the DRAM-sharded config (weights width-sharded over the banks); plus the in0 resharding cost."""
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    g = mesh_device.compute_with_storage_grid_size()
    fid = ttnn.MathFidelity[os.environ.get("MM_FID", "HiFi2")]
    cfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    nbanks = mesh_device.dram_grid_size().x
    for name in os.environ.get("MM_SHAPES", "lat_down,lat_up").split(","):
        Kd, Nd, wdt = SHAPES[name]
        Kt, Nt = Kd // 32, Nd // 32
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = (torch.randn(Kd, Nd) * 0.02).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
        wds = ttnn.from_torch(
            w,
            dtype=wdt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
            memory_config=dram_sharded_weight_config(Kd, Nd, mesh_device.dram_grid_size()),
        )
        ref = x.float() @ w.float()
        floor = Kd * Nd * 1.0625 / 512e3

        def timed(label, f):
            try:
                _times(mesh_device)
                best, y = None, None
                for _ in range(3):
                    y = f()
                    ttnn.synchronize_device(mesh_device)
                    rows = _times(mesh_device)
                    t = max(sum(r) for r in rows)
                    progs = [round(max(r[i] for r in rows), 1) for i in range(min(len(r) for r in rows))]
                    best = (t, progs) if best is None or t < best[0] else best
                got = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
                p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                ttnn.deallocate(y)
                logger.info(
                    f"MMSH {name:8s} {label:40s} {best[0]:8.1f} us {best[1]}  ({floor / best[0] * 100:5.1f}% of floor {floor:.1f})  pcc {p:.5f}"
                )
            except Exception as e:  # noqa: BLE001
                logger.info(f"MMSH {name:8s} {label:40s} FAILED {str(e).splitlines()[0][:200]}")

        def in0_mc(n):
            crs = ttnn.num_cores_to_corerangeset(n, g, row_wise=True)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs, [M, Kd // n], ttnn.ShardOrientation.ROW_MAJOR),
            )

        for n in [c for c in (8, 14, 16, 28, 56, 112) if Kt % c == 0 and c <= g.x * g.y]:
            mc = in0_mc(n)
            kpc = Kt // n
            # 1D mcast in0, the output columns over all cores that divide N
            for nout in [c for c in (56, 112, 28) if Nt % c == 0]:
                pn = Nt // nout
                for bw in sorted({d for d in (1, 2, 4, 8) if kpc % d == 0}):
                    sw = max(d for d in (1, 2, 4) if pn % d == 0)
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=g,
                        in0_block_w=bw,
                        out_subblock_h=1,
                        out_subblock_w=sw,
                        out_block_h=10,
                        out_block_w=pn,
                        per_core_M=10,
                        per_core_N=pn,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    )

                    def f(mc=mc, pc=pc):
                        xs = ttnn.to_memory_config(xt, mc)
                        y = ttnn.linear(xs, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc)
                        ttnn.deallocate(xs)
                        return y

                    timed(f"1d in0 sh{n} out{nout} bw{bw} pn{pn}", f)
            # DRAM sharded: in0 on n cores, out on N / per_core_N cores (L1 width sharded)
            for pn in [d for d in (Nt // nbanks, Nt // 16, Nt // 28, Nt // 56) if d > 0 and Nt % d == 0]:
                for bw in sorted({d for d in (1, 2, 4, 8) if kpc % d == 0}):
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=bw, per_core_M=10, per_core_N=pn, fused_activation=None
                    )

                    def f(mc=mc, pc=pc):
                        xs = ttnn.to_memory_config(xt, mc)
                        y = ttnn.linear(
                            xs,
                            wds,
                            compute_kernel_config=cfg,
                            dtype=ttnn.bfloat16,
                            program_config=pc,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                        )
                        ttnn.deallocate(xs)
                        return y

                    timed(f"ds in0 sh{n} bw{bw} pn{pn}", f)
        ttnn.deallocate(xt)
        ttnn.deallocate(wt)
        ttnn.deallocate(wds)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_stream_linear(mesh_device):
    """combine_rf/stream_linear (every core streams its weight columns, x multicast in K blocks) vs ttnn mm2d."""
    from ttnn.bringup.combine_rf.stream_linear import stream_linear_prepare, upload_stacked

    g = mesh_device.compute_with_storage_grid_size()
    variants = [
        tuple(int(t) for t in v.split(":")) for v in os.environ.get("SL_VARIANTS", "8:3,4:3,8:2,16:2").split(",")
    ]
    fids = os.environ.get("MM_FID", "HiFi2,LoFi").split(",")
    for name, pn in (("lat_down", 1), ("lat_up", 2)):
        if name not in os.environ.get("MM_SHAPES", "lat_down,lat_up"):
            continue
        Kd, Nd, wdt = SHAPES[name]
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = (torch.randn(Kd, Nd) * 0.02).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        ws = None
        ref = x.float() @ w.float()
        floor = Kd * Nd * 1.0625 / 512e3

        def timed(label, f):
            try:
                _times(mesh_device)
                best, y = None, None
                for _ in range(3):
                    y = f()
                    ttnn.synchronize_device(mesh_device)
                    rows = _times(mesh_device)
                    t = max(sum(r) for r in rows)
                    best = t if best is None or t < best else best
                got = ttnn.to_torch(y).float()
                p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                ttnn.deallocate(y)
                logger.info(
                    f"MMSL {name:8s} {label:30s} {best:8.1f} us  ({floor / best * 100:5.1f}% of floor {floor:.1f})  pcc {p:.5f}"
                )
                if not p > 0.999 and os.environ.get("SL_DEBUG"):  # (per output tile: relative error, rows x columns)
                    e = (got - ref).abs().nan_to_num(1e9).reshape(M // 32, 32, Nd // 32, 32).amax(
                        (1, 3)
                    ) / ref.abs().reshape(M // 32, 32, Nd // 32, 32).amax((1, 3))
                    bad = e > 0.05
                    logger.info(
                        f"MMSL bad tiles {int(bad.sum())} / {bad.numel()}; per row {bad.sum(1).tolist()}; per col {bad.sum(0).tolist()}"
                    )
            except Exception as e:  # noqa: BLE001
                logger.info(f"MMSL {name:8s} {label:30s} FAILED {str(e).splitlines()[0][:200]}")

        for fid in fids:
            cfg = ttnn.types.BlackholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity[fid], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            )
            mmc.ENABLED = True
            pc = mmc.mm2d(mesh_device, M, Kd, Nd, ttnn.bfloat16, wdt, ttnn.bfloat16)
            timed(
                f"{fid} ttnn mm2d",
                lambda: ttnn.linear(xt, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc),
            )
            for kb, nb in variants:
                if (Kd // 32) % kb:
                    continue
                run, pns = stream_linear_prepare(
                    mesh_device, M=M, K=Kd, N=Nd, kb=kb, nb=nb, fidelity=ttnn.MathFidelity[fid]
                )
                if ws is None:
                    ws = upload_stacked(mesh_device, w, pns)
                timed(f"{fid} stream kb{kb} nb{nb}", lambda: run(xt, ws))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mm2d_big_blocks(mesh_device):
    """ttnn 2D multicast matmul at in0_block_w beyond mm_configs' L1 budget, and output sub-block shapes."""
    g = mesh_device.compute_with_storage_grid_size()
    for name in os.environ.get("MM_SHAPES", "lat_down,lat_up").split(","):
        Kd, Nd, wdt = SHAPES[name]
        Kt, Nt = Kd // 32, Nd // 32
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = (torch.randn(Kd, Nd) * 0.02).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        ref = x.float() @ w.float()
        pn = -(-Nt // g.x)
        gx = -(-Nt // pn)
        for fid in ("HiFi2",):
            for full in (False, True):
                cfg = ttnn.types.BlackholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity[fid],
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                    dst_full_sync_en=full,
                )
                for bw in [b for b in (8, 16, 28, 32, 56) if Kt % b == 0]:
                    for sw in [d for d in (1, 2, 4, 5, 8) if pn % d == 0 and d <= (8 if full else 4)]:
                        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=ttnn.CoreCoord(gx, 10),
                            in0_block_w=bw,
                            out_subblock_h=1,
                            out_subblock_w=sw,
                            out_block_h=1,
                            out_block_w=pn,
                            per_core_M=1,
                            per_core_N=pn,
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=True,
                        )
                        try:
                            _times(mesh_device)
                            best = None
                            for _ in range(3):
                                y = ttnn.linear(
                                    xt, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc
                                )
                                ttnn.synchronize_device(mesh_device)
                                t = max(sum(r) for r in _times(mesh_device))
                                best = t if best is None or t < best else best
                            got = ttnn.to_torch(y).float()
                            p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
                            ttnn.deallocate(y)
                            logger.info(
                                f"MMBIG {name:8s} {fid} full {int(full)} bw {bw:3d} sw {sw}  {best:8.1f} us  pcc {p:.5f}"
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.info(
                                f"MMBIG {name:8s} {fid} full {int(full)} bw {bw:3d} sw {sw}  FAILED {str(e).splitlines()[0][:120]}"
                            )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_lin2d(mesh_device):
    """combine_rf/lin2d (fixed per-column / per-row senders, NOC0 + NOC1 alternating) vs ttnn mm2d."""
    from ttnn.bringup.combine_rf.lin2d import lin2d_prepare, upload_stacked

    variants = [tuple(int(t) for t in v.split(":")) for v in os.environ.get("L2D_VARIANTS", "8:4,4:4,16:2").split(",")]
    fids = os.environ.get("MM_FID", "HiFi2,LoFi").split(",")
    for name in os.environ.get("MM_SHAPES", "lat_down,lat_up").split(","):
        Kd, Nd, wdt = SHAPES[name]
        x = torch.randn(M, Kd).to(torch.bfloat16)
        w = (torch.randn(Kd, Nd) * 0.02).to(torch.bfloat16)
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        ref = x.float() @ w.float()
        floor = Kd * Nd * 1.0625 / 512e3
        ws = {}
        for fid in fids:
            if os.environ.get("L2D_MM2D", "1") == "1":
                wt = ttnn.from_torch(w, dtype=wdt, layout=ttnn.TILE_LAYOUT, device=mesh_device)
                cfg = ttnn.types.BlackholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity[fid],
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
                mmc.ENABLED = True
                pc = mmc.mm2d(mesh_device, M, Kd, Nd, ttnn.bfloat16, wdt, ttnn.bfloat16)
                _timed_pcc(
                    mesh_device,
                    f"L2D {name:8s} {fid} ttnn mm2d",
                    ref,
                    floor,
                    lambda: ttnn.linear(xt, wt, compute_kernel_config=cfg, dtype=ttnn.bfloat16, program_config=pc),
                )
                ttnn.deallocate(wt)
            for kb, nb in variants:
                if (Kd // 32) % kb:
                    continue
                try:
                    run, layout = lin2d_prepare(
                        mesh_device, M=M, K=Kd, N=Nd, kb=kb, nb=nb, fidelity=ttnn.MathFidelity[fid]
                    )
                except Exception as e:  # noqa: BLE001
                    logger.info(f"L2D {name:8s} {fid} kb{kb} nb{nb} prepare FAILED {str(e).splitlines()[0][:200]}")
                    continue
                key = (tuple(layout["pns"]), layout["NP"])
                if key not in ws:
                    ws[key] = upload_stacked(mesh_device, w, layout, dtype=wdt)
                _timed_pcc(
                    mesh_device,
                    f"L2D {name:8s} {fid} lin2d kb{kb} nb{nb} NP{layout['NP']}",
                    ref,
                    floor,
                    lambda: run(xt, ws[key]),
                )
                if int(os.environ.get("L2D_PROBE", "0")) & 512 and getattr(run, "dbg", None) is not None:
                    _lin2d_trace(run.dbg, layout, nb, Kd // 32 // kb)


def _timed_pcc(mesh_device, label, ref, floor, f, n=3):
    try:
        _times(mesh_device)
        best, y = None, None
        for _ in range(n):
            y = f()
            ttnn.synchronize_device(mesh_device)
            t = max(sum(r) for r in _times(mesh_device))
            best = t if best is None or t < best else best
        got = ttnn.to_torch(y).float()
        p = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1].item()
        ttnn.deallocate(y)
        logger.info(f"{label:44s} {best:8.1f} us  ({floor / best * 100:5.1f}% of floor {floor:.1f})  pcc {p:.5f}")
        if not p > 0.999 and os.environ.get("L2D_DEBUG"):
            Mt, Nt = ref.shape[0] // 32, ref.shape[1] // 32
            e = (got - ref).abs().nan_to_num(1e9).reshape(Mt, 32, Nt, 32).amax((1, 3)) / ref.abs().reshape(
                Mt, 32, Nt, 32
            ).amax((1, 3))
            bad = e > 0.05
            logger.info(
                f"L2DBAD {int(bad.sum())} / {bad.numel()}; per row {bad.sum(1).tolist()}; per col {bad.sum(0).tolist()}"
            )
    except Exception as e:  # noqa: BLE001
        logger.info(f"{label:44s} FAILED {str(e).splitlines()[0][:200]}")


def _lin2d_trace(dbg, layout, nb, nblk, clk_ghz=1.35):
    """lin2d PROBE 512 events per core: W push per step, x push, credits, sends (top / left: BRISC; bottom / right:
    NCRISC) -> per-step timeline (us from the first core's start)."""
    d = ttnn.to_torch(dbg).to(torch.int64) & 0xFFFFFFFF
    GX = len(layout["pns"])
    GY = d.shape[0] // GX
    NS = layout["NP"] * nblk
    t0 = int(d[:, 255].min())
    us = lambda v: (v - t0) / (clk_ghz * 1e3)  # noqa: E731
    core = lambda cx, cy: cx + cy * GX  # noqa: E731
    logger.info(f"L2DTR start spread {us(int(d[:, 255].max())):.2f} us; NS {NS}")
    for cy in range(GY):  # x: per row, the senders' times (left BRISC even, right NCRISC odd) and the latest push
        xs = []
        for j in range(min(nblk, 8)):
            v = int(d[core(0, cy), 192 + j // 2]) if j % 2 == 0 else int(d[core(GX - 1, cy), 256 + 192 + j // 2])
            xs.append(f"{us(v):6.2f}/{us(int(d[core(0, cy):core(0, cy) + GX, 64 + j].max())):6.2f}")
        logger.info(f"L2DTR x row {cy}: send/last push " + " ".join(xs))
    for cx in range(GX):  # W: per column, the senders' first sends
        ws = [
            f"{us(int(d[core(cx, 0) if r % 2 == 0 else core(cx, GY - 1), (160 if r % 2 == 0 else 416) + r // 2])):6.2f}"
            for r in range(8)
        ]
        logger.info(f"L2DTR W col {cx}: sends " + " ".join(ws))
    for r in range(NS):
        push = d[:, r]
        q, k = r & 1, r >> 1
        snd = [int(d[core(cx, 0 if q == 0 else GY - 1), (160 if q == 0 else 256 + 160) + k]) for cx in range(GX)]
        cred = d[:, 96 + r - nb] if r >= nb else None
        xp = d[:, 64 + r] if r < nblk else None
        logger.info(
            f"L2DTR step {r:3d} send {us(min(snd)):7.2f}..{us(max(snd)):7.2f}  push {us(int(push.min())):7.2f}..{us(int(push.max())):7.2f}"
            + (f"  x push ..{us(int(xp.max())):7.2f}" if xp is not None else "")
            + (f"  credit(for this step) ..{us(int(cred.max())):7.2f}" if cred is not None else "")
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_lin2d_se(mesh_device):
    """The K3 shared expert as two lin2d calls: gate / up with the SiTU-GLU fused (glu=True), then down; vs fp32."""
    from ttnn.bringup.combine_rf.lin2d import glu_weight, lin2d_prepare, upload_stacked

    H, I = 7168, 6144
    g = torch.Generator().manual_seed(0)
    x = torch.randn(M, H, generator=g).to(torch.bfloat16)
    wg, wu, wd = (
        torch.randn(H, I, generator=g) * 0.02,
        torch.randn(H, I, generator=g) * 0.02,
        torch.randn(I, H, generator=g) * 0.02,
    )
    gs, us_ = x.float() @ wg, x.float() @ wu
    aref = 4 * torch.tanh(gs / 4) * torch.sigmoid(gs) * (25 * torch.tanh(us_ / 25))
    yref = aref @ wd
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    fid = ttnn.MathFidelity[os.environ.get("MM_FID", "HiFi2")]
    gu_run, lay_gu = lin2d_prepare(mesh_device, M=M, K=H, N=2 * I, fidelity=fid, glu=True)
    w_gu = upload_stacked(mesh_device, glu_weight(wg, wu), lay_gu)
    d_run, lay_d = lin2d_prepare(mesh_device, M=M, K=I, N=H, fidelity=fid)
    w_d = upload_stacked(mesh_device, wd, lay_d)
    floor_gu, floor_d = H * 2 * I * 1.0625 / 512e3, I * H * 1.0625 / 512e3
    a_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("L2D_A_L1", "1") == "1" else None
    _timed_pcc(mesh_device, f"L2DSE gate/up + SiTU (NP {lay_gu['NP']})", aref, floor_gu, lambda: gu_run(xt, w_gu, a_mc))
    a = gu_run(xt, w_gu, a_mc)
    _timed_pcc(mesh_device, f"L2DSE down (NP {lay_d['NP']})", yref, floor_d, lambda: d_run(a, w_d))

    def both():
        a_ = gu_run(xt, w_gu, a_mc)
        y_ = d_run(a_, w_d)
        ttnn.deallocate(a_)
        return y_

    _timed_pcc(mesh_device, "L2DSE shared expert (both)", yref, floor_gu + floor_d, both)
    # down with the residual fused (the block's routed output yr): y = yr + a @ wd
    r = torch.randn(M, H, generator=g).to(torch.bfloat16)
    rt = ttnn.from_torch(
        r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    dr_run, _ = lin2d_prepare(mesh_device, M=M, K=I, N=H, fidelity=fid, residual=True)
    _timed_pcc(mesh_device, "L2DSE down + residual", yref + r.float(), floor_d, lambda: dr_run(a, w_d, None, rt))
