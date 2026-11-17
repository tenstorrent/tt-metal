# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated bench of genify_matmul's matmul K-walk (PERF FOCUS 1 block: 6x28 tiles, 1x4 subblocks,
K = 160 tiles as 14 K-blocks of 12 / last 4; A bf16 -> SrcB, B bfp4_b -> SrcA; LoFi, fp32 DEST,
half-sync; fp32 partials with packer L1 accumulation, last K-block -> partials (Interm)).

A and B live in L1 shards holding exactly two K-block slots per core; the reader only reserves /
pushes, so every K-block re-presents slot kb % 2 -- no NoC traffic, compute only. Kernels live in
ttnn/ttnn/operations/genify_matmul/perf_experiments/matmul_kwalk_efficiency/kernels.

    scripts/run_safe_pytest.sh --run-all <this file>::test_kwalk_check     # correctness (PCC / rel-RMS)
    scripts/run_safe_pytest.sh <this file>::test_kwalk_perf                # DEVICE KERNEL DURATION per variant

Env: KW_VARIANTS (comma list, see VARIANTS), KW_BLOCK="bm,bn,sbh,sbw,bk,K", KW_GRID="gx,gy",
KW_FIDELITY (LoFi|HiFi2|HiFi4), KW_FP32 (1|0).
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

ROOT = Path(__file__).resolve().parents[7]
KDIR = ROOT / "ttnn/ttnn/operations/genify_matmul/perf_experiments/matmul_kwalk_efficiency/kernels"
TILE = 32
CB_ACT, CB_WEIGHT, CB_OUT, CB_PARTIALS = 0, 1, 16, 24
CLOCK_MHZ = 1350.0

# name -> (KW_VARIANT, extra defines)
VARIANTS = {
    "base": (0, []),
    "base_skip": (0, [("SKIP_COMPUTE", "1")]),
    "own": (1, []),
    "own_skip": (1, [("SKIP_COMPUTE", "1")]),
    "raw": (2, []),
    "raw_skip": (2, [("SKIP_COMPUTE", "1")]),
    "raw_nopack": (2, [("ABL_NOPACK", "1")]),
    "raw_nomop": (2, [("MM_FLAVOR", "1")]),
    "raw_nothrottle": (2, [("MM_FLAVOR", "2")]),
    "diag_one_setc": (2, [("MM_FLAVOR", "3")]),
    "dest_adv": (2, [("MM_FLAVOR", "4")]),
    "hdr": (3, []),
    "hdr_skip": (3, [("SKIP_COMPUTE", "1")]),
    "dest_adv_clrab": (2, [("MM_FLAVOR", "4"), ("KW_CLR_AB", "1")]),
}


def _block():
    bm, bn, sbh, sbw, bk, k = (int(x) for x in os.environ.get("KW_BLOCK", "6,28,1,4,12,160").split(","))
    nkb = -(-k // bk)
    last = k - (nkb - 1) * bk
    return bm, bn, sbh, sbw, bk, k, nkb, last


def _grid(device):
    g = os.environ.get("KW_GRID")
    if g:
        gx, gy = (int(x) for x in g.split(","))
    else:
        cg = device.compute_with_storage_grid_size()
        gx, gy = cg.x, cg.y
    return gx, gy


def _cfg():
    fid = getattr(ttnn.MathFidelity, os.environ.get("KW_FIDELITY", "LoFi"))
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=fid, fp32_dest_acc_en=os.environ.get("KW_FP32", "1") == "1", math_approx_mode=False
    )


def _make_inputs(device, gx, gy):
    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    ncores = gx * gy
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    torch.manual_seed(0)
    a = torch.randn(2 * bm * TILE, bk * TILE)  # slot 0 = rows [0, bm), slot 1 = rows [bm, 2bm) (tiles)
    b = torch.randn(2 * bk * TILE, bn * TILE) / (k * TILE) ** 0.5

    def shard(t, dtype):
        h, w = t.shape
        mc = ttnn.create_sharded_memory_config(
            shape=(h, w),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.from_torch(
            t.repeat(ncores, 1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )

    ta = shard(a, getattr(ttnn, os.environ.get("KW_ADTYPE", "bfloat16")))
    tb = shard(b, getattr(ttnn, os.environ.get("KW_BDTYPE", "bfloat4_b")))
    return ta, tb, grid, ncores


def _reference(ta, tb):
    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    aq = ttnn.to_torch(ta).float()[: 2 * bm * TILE]
    bq = ttnn.to_torch(tb).float()[: 2 * bk * TILE]
    acc = torch.zeros(bm * TILE, bn * TILE, dtype=torch.float64)
    for kb in range(nkb):
        s = kb % 2
        kk = (last if kb == nkb - 1 else bk) * TILE
        A = aq[s * bm * TILE : (s + 1) * bm * TILE, :kk].double()
        B = bq[s * bk * TILE : s * bk * TILE + kk, :].double()
        acc += A @ B
    return acc.float()


def _descriptor(ta, tb, out, grid, gx, gy, variant, check):
    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    kv, defs = VARIANTS[variant]
    defs = [("KW_VARIANT", str(kv)), *defs] + ([("CHECK_OUT", "1")] if check else [])
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_ACT, ta),
        ttnn.cb_descriptor_from_sharded_tensor(CB_WEIGHT, tb),
        ttnn.CBDescriptor(
            total_size=bm * bn * 4096,
            core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_PARTIALS, data_format=ttnn.float32, page_size=4096)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=4 * 4096,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.float32, page_size=4096)],
        ),
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "reader.cpp"),
        core_ranges=grid,
        compile_time_args=[CB_ACT, CB_WEIGHT, bm * bk, bk * bn, nkb],
        runtime_args=ttnn.RuntimeArgs(),
        config=ttnn.ReaderConfigDescriptor(),
    )
    wrt = ttnn.RuntimeArgs()
    for y in range(gy):
        for x in range(gx):
            wrt[x][y] = [out.buffer_address(), y * gx + x]
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "writer.cpp"),
        core_ranges=grid,
        compile_time_args=[CB_OUT, bm * bn, 4096, *ttnn.TensorAccessorArgs(out).get_compile_time_args()],
        runtime_args=wrt,
        defines=[("CHECK_OUT", "1")] if check else [],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "compute.cpp"),
        core_ranges=grid,
        compile_time_args=[CB_ACT, CB_WEIGHT, CB_OUT, CB_PARTIALS, bm // sbh, bn // sbw, sbh, sbw, bk, nkb, last],
        runtime_args=ttnn.RuntimeArgs(),
        defines=defs,
        config=_cfg(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def _out_tensor(device, ncores):
    bm, bn, *_ = _block()
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([ncores * bm * TILE, bn * TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    vals = []
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if entry is not None:
                vals.append(float(entry.duration))
    return max(vals) if vals else None


def _variant_list(default):
    return [v for v in os.environ.get("KW_VARIANTS", default).split(",") if v]


def test_kwalk_check(device):
    gx, gy = _grid(device)
    ta, tb, grid, ncores = _make_inputs(device, gx, gy)
    ref = _reference(ta, tb)
    bm, bn, *_ = _block()
    outs = {}
    for v in _variant_list("base,own,raw,dest_adv,hdr"):
        out = _out_tensor(device, ncores)
        ttnn.generic_op([ta, tb, out], _descriptor(ta, tb, out, grid, gx, gy, v, check=True))
        got = ttnn.to_torch(out).float().reshape(ncores, bm * TILE, bn * TILE)
        pcc = torch.corrcoef(torch.stack([got[0].flatten(), ref.flatten()]))[0, 1].item()
        rms = ((got[0] - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        spread = (got - got[0:1]).abs().max().item()
        outs[v] = got[0]
        logger.info(f"KW_CHECK {v:<12} pcc {pcc:.7f} rel_rms {rms:.3e} core_spread {spread:.3e}")
        # LoFi on a 7-bit-mantissa SrcA (bfp8 B) is a few % rel-RMS by design; the hard gate is
        # bit-identity with the baseline helper (below) plus a sane PCC.
        assert pcc > 0.999 and spread == 0.0, v
    if "base" in outs:
        for v, o in outs.items():
            d = (o - outs["base"]).abs().max().item()
            logger.info(f"KW_CHECK {v:<12} max|diff vs base| {d:.3e}")
            assert d == 0.0 or v in ("own_skip", "raw_skip", "diag_one_setc", "dest_adv_clrab"), v


def test_kwalk_perf(device):
    gx, gy = _grid(device)
    ta, tb, grid, ncores = _make_inputs(device, gx, gy)
    bm, bn, sbh, sbw, bk, k, nkb, last = _block()
    out = _out_tensor(device, 1)
    floor_ns = bm * bn * k * 16 / CLOCK_MHZ * 1000.0
    if os.environ.get("KW_FIDELITY", "LoFi") != "LoFi":
        floor_ns *= {"HiFi2": 2, "HiFi3": 3, "HiFi4": 4}[os.environ["KW_FIDELITY"]]
    _read_kernel_ns(device)
    rows = []
    for v in _variant_list("base,hdr"):
        ttnn.generic_op([ta, tb, out], _descriptor(ta, tb, out, grid, gx, gy, v, check=False))
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
        passes = nkb * (bm // sbh) * (bn // sbw)
        cyc_tile = ns * CLOCK_MHZ / 1000.0 / (bm * bn * k)
        rows.append(
            f"KW_PERF {v:<12} {ns:12.0f} ns  {cyc_tile:6.2f} cyc/tile-K (floor 16 x fid)  "
            f"over_floor {(ns - floor_ns) * CLOCK_MHZ / 1000.0 / passes:7.1f} cyc/pass"
        )
    logger.info(
        f"\nKW block={bm}x{bn} sb={sbh}x{sbw} bk={bk} K={k} nkb={nkb} last={last} grid={gx}x{gy} "
        f"A={os.environ.get('KW_ADTYPE', 'bfloat16')} B={os.environ.get('KW_BDTYPE', 'bfloat4_b')} fid={os.environ.get('KW_FIDELITY', 'LoFi')} fp32={os.environ.get('KW_FP32', '1')} floor={floor_ns:.0f} ns\n"
        + "\n".join(rows)
    )
