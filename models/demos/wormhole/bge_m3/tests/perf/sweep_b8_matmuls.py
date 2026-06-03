# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep — B8/S512 QKV, AttnOut, MLP-wo matmul program configs.

Same class of win as the MLP-wi sweep (which found 3.9x): these three B8
matmuls also fall through to the default ttnn.linear routing (their tuned
program configs gate on batch 1/32). All are M=4096.

  - QKV     : K=1024 N=3072  (no act)
  - AttnOut : K=1024 N=1024  (no act)
  - MLPwo   : K=4096 N=1024  (no act)

PCC baseline = default ttnn.linear (LoFi, what B8 ships post exp 1-3).

Full sweep:
    TT_VISIBLE_DEVICES=0 pytest \
      models/demos/wormhole/bge_m3/tests/perf/sweep_b8_matmuls.py::test_b8_mm_sweep -sv
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from tools.tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

BATCH = 8
SEQ = 512
M = BATCH * SEQ  # 4096
TILE = 32

DEVICE_PARAMS = {"trace_region_size": 50_000_000, "num_command_queues": 1}
MATMUL_OP_CODE = "MatmulDeviceOperation"
PROFILER_SUBDIR = "bge_m3_b8_mm_sweep"
PCC_THRESHOLD = 0.95


SHAPES = {
    "QKV": (1024, 3072),
    "AttnOut": (1024, 1024),
    "MLPwo": (4096, 1024),
}


@dataclass(frozen=True)
class Variant:
    shape: str
    kind: str  # "default" | "mm2d"
    grid_x: int = 11
    grid_y: int = 10
    in0_block_w: int = 4
    out_subblock_h: int = 1
    out_subblock_w: int = 2

    @property
    def name(self):
        if self.kind == "default":
            return f"{self.shape}_default"
        return f"{self.shape}_g{self.grid_x}x{self.grid_y}_ibw{self.in0_block_w}_sub{self.out_subblock_h}x{self.out_subblock_w}"


# Sweep multiple grids: the 11x10 grid forces per_core_M=13/per_core_N=9or3 ->
# only sub=1x1 possible. Smaller/even grids (8x8, 10x8, 8x10) give per_core dims
# divisible by 2/4, unlocking larger subblocks (more math per CB pass).
_GRIDS = [(11, 10), (10, 8), (8, 10), (8, 8)]


def _build_variants():
    out = []
    for shape, (K, N) in SHAPES.items():
        out.append(Variant(shape=shape, kind="default"))
        K_tiles, N_tiles, M_tiles = K // TILE, N // TILE, M // TILE
        for gx, gy in _GRIDS:
            pc_M = (M_tiles + gy - 1) // gy
            pc_N = (N_tiles + gx - 1) // gx
            for ibw in [2, 4, 8]:
                if K_tiles % ibw != 0:
                    continue
                for osh in [1, 2, 4]:
                    for osw in [1, 2, 4]:
                        if osh * osw > 8:
                            continue
                        if pc_M % osh != 0 or pc_N % osw != 0:
                            continue
                        out.append(
                            Variant(
                                shape=shape,
                                kind="mm2d",
                                grid_x=gx,
                                grid_y=gy,
                                in0_block_w=ibw,
                                out_subblock_h=osh,
                                out_subblock_w=osw,
                            )
                        )
    return out


VARIANT_LIMIT = int(os.environ.get("BGE_M3_B8MM_VARIANT_LIMIT", "0"))


def _all_variants():
    vs = _build_variants()
    return vs[:VARIANT_LIMIT] if VARIANT_LIMIT > 0 else vs


def _vid(v: Variant) -> str:
    if v.kind == "default":
        return f"{v.shape}|default"
    return f"{v.shape}|mm2d|{v.grid_x}|{v.grid_y}|{v.in0_block_w}|{v.out_subblock_h}|{v.out_subblock_w}"


def _from_vid(cid: str) -> Variant:
    parts = cid.split("|")
    if parts[1] == "default":
        return Variant(shape=parts[0], kind="default")
    _, _, gx, gy, ibw, osh, osw = parts
    return Variant(
        shape=parts[0],
        kind="mm2d",
        grid_x=int(gx),
        grid_y=int(gy),
        in0_block_w=int(ibw),
        out_subblock_h=int(osh),
        out_subblock_w=int(osw),
    )


def _inputs(mesh_device, K, N, *, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(1, 1, K, N, dtype=torch.bfloat16) * 0.05
    b = torch.randn(1, 1, 1, N, dtype=torch.bfloat16) * 0.05
    a_d = ttnn.from_torch(
        a, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    w_d = ttnn.from_torch(
        w, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_d = ttnn.from_torch(
        b, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return a_d, w_d, b_d


def _ck(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _run(a, w, b, v: Variant, mesh_device, *, ck):
    K, N = SHAPES[v.shape]
    if v.kind == "default":
        return ttnn.linear(
            a,
            w,
            bias=b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ck,
            core_grid=ttnn.CoreGrid(x=11, y=10),
        )
    M_tiles, N_tiles = M // TILE, N // TILE
    pcfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(v.grid_x, v.grid_y),
        in0_block_w=v.in0_block_w,
        out_subblock_h=v.out_subblock_h,
        out_subblock_w=v.out_subblock_w,
        per_core_M=(M_tiles + v.grid_y - 1) // v.grid_y,
        per_core_N=(N_tiles + v.grid_x - 1) // v.grid_x,
        transpose_mcast=False,
        fused_activation=None,
    )
    return ttnn.linear(
        a,
        w,
        bias=b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ck,
        program_config=pcfg,
    )


def _pcc(ref, out):
    return float(torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0)).item())


@pytest.mark.parametrize("variant", [v for v in _all_variants() if v.kind != "default"], ids=lambda v: v.name)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_b8_mm_pcc(mesh_device, variant):
    K, N = SHAPES[variant.shape]
    a, w, b = _inputs(mesh_device, K, N, seed=11)
    ck = _ck(mesh_device)
    ref = _run(a, w, b, Variant(shape=variant.shape, kind="default"), mesh_device, ck=ck)
    ttnn.synchronize_device(mesh_device)
    ref_h = ttnn.to_torch(ref).float()
    ttnn.deallocate(ref)
    try:
        out = _run(a, w, b, variant, mesh_device, ck=ck)
    except Exception as e:
        for t in (a, w, b):
            ttnn.deallocate(t)
        pytest.fail(f"{variant.name}: {type(e).__name__}: {e}")
    ttnn.synchronize_device(mesh_device)
    out_h = ttnn.to_torch(out).float()
    for t in (out, a, w, b):
        ttnn.deallocate(t)
    pcc = _pcc(ref_h, out_h)
    logger.info(f"{variant.name:<40} PCC={pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"{variant.name} PCC {pcc} < {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_b8_mm_worker(mesh_device):
    v = _from_vid(os.environ["BGE_M3_B8MM_CASE"])
    iters = int(os.environ.get("BGE_M3_B8MM_WORKER_ITERS", "20"))
    K, N = SHAPES[v.shape]
    a, w, b = _inputs(mesh_device, K, N, seed=0)
    ck = _ck(mesh_device)
    for _ in range(2):
        out = _run(a, w, b, v, mesh_device, ck=ck)
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    for _ in range(iters):
        out = _run(a, w, b, v, mesh_device, ck=ck)
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh_device)
    for t in (a, w, b):
        ttnn.deallocate(t)


def _min_ns(subdir):
    import pandas as pd

    fn = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(fn)
    df = df[df["OP CODE"] == MATMUL_OP_CODE]
    if df.empty:
        raise RuntimeError("no matmul rows")
    dur = pd.to_numeric(
        df[df["DEVICE KERNEL DURATION [ns]"] != "-"]["DEVICE KERNEL DURATION [ns]"], errors="coerce"
    ).dropna()
    return float(dur.min())


@pytest.mark.timeout(0)
def test_b8_mm_sweep():
    worker = "models/demos/wormhole/bge_m3/tests/perf/sweep_b8_matmuls.py::test_b8_mm_worker"
    csv_path = Path(__file__).resolve().parent / "sweep_results" / "b8_matmuls_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for v in _all_variants():
        cid = _vid(v)
        os.environ["BGE_M3_B8MM_CASE"] = cid
        logger.info(f"[sweep] profiling {cid}")
        try:
            run_device_profiler(
                f"pytest {worker} -sv", PROFILER_SUBDIR, device_analysis_types=["device_kernel_duration"]
            )
            us = _min_ns(PROFILER_SUBDIR) / 1000.0
            status = "ok"
        except Exception as e:  # noqa: BLE001
            us = float("nan")
            status = f"err:{type(e).__name__}"
            logger.error(f"[sweep] {cid} failed: {e}")
        finally:
            os.environ.pop("BGE_M3_B8MM_CASE", None)
        results.append(dict(shape=v.shape, variant=v.name, us=us, status=status, is_default=(v.kind == "default")))
        _write_csv(results, csv_path)
    logger.info(f"[sweep] CSV: {csv_path}")
    _print_table(results)


def _write_csv(results, path):
    import csv

    cols = ["shape", "variant", "us", "status", "is_default"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})


def _print_table(results):
    by_shape = {}
    for r in results:
        by_shape.setdefault(r["shape"], []).append(r)
    logger.info("")
    logger.info("=" * 80)
    logger.info("  B8 matmul sweep (M=4096) — per shape, baseline = default ttnn.linear")
    logger.info("=" * 80)
    for shape, rows in by_shape.items():
        base = next((r for r in rows if r["is_default"]), None)
        base_us = base["us"] if base and base["status"] == "ok" else float("nan")
        logger.info(f"\n── {shape}  (default {base_us:.3f} µs)")
        for r in sorted(rows, key=lambda r: (r["status"] != "ok", r["us"])):
            vs = f"{base_us / r['us']:.3f}x" if (r["status"] == "ok" and base_us == base_us and base_us > 0) else "-"
            tag = "  <= DEFAULT" if r["is_default"] else ""
            logger.info(f"  {r['variant']:<42}{r['us']:>10.3f}{vs:>9}  {r['status']}{tag}")
        ok = [r for r in rows if r["status"] == "ok" and not r["is_default"]]
        if ok and base_us == base_us and base_us > 0:
            best = min(ok, key=lambda r: r["us"])
            if best["us"] < base_us:
                logger.info(
                    f"  >>> BEST {shape}: {best['variant']} = {best['us']:.3f} µs ({base_us / best['us']:.3f}x)"
                )
    logger.info("=" * 80)
