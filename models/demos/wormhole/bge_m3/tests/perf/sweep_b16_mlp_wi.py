# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep — B16/S512 MLP-wi matmul program configs.

Tracy shows MLP-wi (M=4096, K=1024, N=4096, fused GELU) is the single biggest
matmul at B16: ~404 µs/call × 24 = 9.7 ms (>half of all matmul time). It runs
with the default ttnn.linear routing (no program config) because the tuned
configs gate on batch 1 / 32 only.

This sweep tunes a MatmulMultiCoreReuseMultiCastProgramConfig on the 11×10
P150 grid (the default the model already uses for B16 matmuls), varying
in0_block_w / out_subblock / per_core tiling. The LoFi + fp32-off compute
kernel matches the B16 production policy (exp 1).

PCC baseline = the default ttnn.linear (what B16 ships today, post exp 1).

PCC-only:
    TT_VISIBLE_DEVICES=0 pytest \
      models/demos/wormhole/bge_m3/tests/perf/sweep_b16_mlp_wi.py::test_b16_mlp_wi_pcc -sv

Full sweep:
    TT_VISIBLE_DEVICES=0 pytest \
      models/demos/wormhole/bge_m3/tests/perf/sweep_b16_mlp_wi.py::test_b16_mlp_wi_sweep -sv
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

BATCH = 16
SEQ = 512
M = BATCH * SEQ  # 4096
K = 1024
N = 4096
TILE = 32

DEVICE_PARAMS = {"trace_region_size": 50_000_000, "num_command_queues": 1}
MATMUL_OP_CODE = "MatmulDeviceOperation"
PROFILER_SUBDIR = "bge_m3_b16_mlp_wi_sweep"
# LoFi-vs-LoFi reorderings: ibw=1 accumulates worst (~0.967). 0.95 flags truly
# broken tiling while allowing accumulation-order drift; model gate is 0.94.
PCC_THRESHOLD = 0.95
GELU = (ttnn.UnaryOpType.GELU, True)


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str  # "default" | "mm2d"
    grid_x: int = 11
    grid_y: int = 10
    in0_block_w: int = 4
    out_subblock_h: int = 1
    out_subblock_w: int = 2


def _build_variants() -> list[Variant]:
    variants = [Variant(name="default", kind="default")]
    M_tiles = M // TILE  # 128
    N_tiles = N // TILE  # 128
    K_tiles = K // TILE  # 32

    # Grids available on P150 (11x10=110). Also try 8x8 (64) for less dispatch.
    grids = [(11, 10), (8, 8)]
    for gx, gy in grids:
        pc_M = (M_tiles + gy - 1) // gy
        pc_N = (N_tiles + gx - 1) // gx
        for ibw in [1, 2, 4, 8]:
            if K_tiles % ibw != 0:
                continue
            for osh in [1, 2, 4]:
                for osw in [1, 2, 4]:
                    if osh * osw > 8:  # fp32_dest_acc_en=False cap
                        continue
                    if pc_M % osh != 0 or pc_N % osw != 0:
                        continue
                    variants.append(
                        Variant(
                            name=f"mm2d_g{gx}x{gy}_ibw{ibw}_sub{osh}x{osw}",
                            kind="mm2d",
                            grid_x=gx,
                            grid_y=gy,
                            in0_block_w=ibw,
                            out_subblock_h=osh,
                            out_subblock_w=osw,
                        )
                    )
    return variants


VARIANT_LIMIT = int(os.environ.get("BGE_M3_B16MLP_VARIANT_LIMIT", "0"))


def _all_variants():
    vs = _build_variants()
    return vs[:VARIANT_LIMIT] if VARIANT_LIMIT > 0 else vs


def _variant_to_id(v: Variant) -> str:
    if v.kind == "default":
        return "default"
    return f"mm2d-{v.grid_x}-{v.grid_y}-{v.in0_block_w}-{v.out_subblock_h}-{v.out_subblock_w}"


def _id_to_variant(cid: str) -> Variant:
    if cid == "default":
        return Variant(name="default", kind="default")
    _, gx, gy, ibw, osh, osw = cid.split("-")
    return Variant(
        name=cid,
        kind="mm2d",
        grid_x=int(gx),
        grid_y=int(gy),
        in0_block_w=int(ibw),
        out_subblock_h=int(osh),
        out_subblock_w=int(osw),
    )


def _inputs(mesh_device, *, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(1, 1, M, K, dtype=torch.bfloat16) * 0.05
    w = torch.randn(1, 1, K, N, dtype=torch.bfloat16) * 0.05
    b = torch.randn(1, 1, 1, N, dtype=torch.bfloat16) * 0.05
    a_d = ttnn.from_torch(
        a, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    w_d = ttnn.from_torch(
        w, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_d = ttnn.from_torch(
        b, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return a_d, w_d, b_d


def _ck(mesh_device):
    # LoFi + fp32_dest_acc_en=False = B16 production MLP-wi policy (exp 1).
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _run(a, w, b, v: Variant, mesh_device, *, ck):
    if v.kind == "default":
        return ttnn.linear(
            a,
            w,
            bias=b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ck,
            core_grid=ttnn.CoreGrid(x=11, y=10),
            activation="gelu",
        )
    M_tiles = M // TILE
    N_tiles = N // TILE
    pcfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(v.grid_x, v.grid_y),
        in0_block_w=v.in0_block_w,
        out_subblock_h=v.out_subblock_h,
        out_subblock_w=v.out_subblock_w,
        per_core_M=(M_tiles + v.grid_y - 1) // v.grid_y,
        per_core_N=(N_tiles + v.grid_x - 1) // v.grid_x,
        transpose_mcast=False,
        fused_activation=GELU,
    )
    return ttnn.linear(
        a,
        w,
        bias=b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=ck,
        program_config=pcfg,
    )


def _pcc(ref, out):
    return float(torch.nn.functional.cosine_similarity(ref.flatten().unsqueeze(0), out.flatten().unsqueeze(0)).item())


@pytest.mark.parametrize("variant", _all_variants(), ids=lambda v: v.name)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_b16_mlp_wi_pcc(mesh_device, variant):
    if variant.name == "default":
        pytest.skip("control")
    a, w, b = _inputs(mesh_device, seed=11)
    ck = _ck(mesh_device)
    ref = _run(a, w, b, _id_to_variant("default"), mesh_device, ck=ck)
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
    logger.info(f"B16-MLPwi {variant.name:<32} PCC={pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"{variant.name} PCC {pcc} < {PCC_THRESHOLD}"


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_b16_mlp_wi_worker(mesh_device):
    v = _id_to_variant(os.environ["BGE_M3_B16MLP_CASE"])
    iters = int(os.environ.get("BGE_M3_B16MLP_WORKER_ITERS", "20"))
    a, w, b = _inputs(mesh_device, seed=0)
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


def _post_process_min_kernel_ns(subdir):
    import pandas as pd

    fn = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(fn)
    df = df[df["OP CODE"] == MATMUL_OP_CODE]
    if df.empty:
        raise RuntimeError(f"no matmul rows in {fn}")
    dur = pd.to_numeric(
        df[df["DEVICE KERNEL DURATION [ns]"] != "-"]["DEVICE KERNEL DURATION [ns]"], errors="coerce"
    ).dropna()
    cores = pd.to_numeric(df[df["CORE COUNT"] != "-"]["CORE COUNT"], errors="coerce").dropna()
    return float(dur.min()), int(cores.iloc[0]) if not cores.empty else -1


@pytest.mark.timeout(0)
def test_b16_mlp_wi_sweep():
    worker = "models/demos/wormhole/bge_m3/tests/perf/sweep_b16_mlp_wi.py::test_b16_mlp_wi_worker"
    csv_path = Path(__file__).resolve().parent / "sweep_results" / "b16_mlp_wi_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for v in _all_variants():
        cid = _variant_to_id(v)
        os.environ["BGE_M3_B16MLP_CASE"] = cid
        logger.info(f"[sweep] profiling {cid}")
        try:
            run_device_profiler(
                f"pytest {worker} -sv", PROFILER_SUBDIR, device_analysis_types=["device_kernel_duration"]
            )
            ns, cores = _post_process_min_kernel_ns(PROFILER_SUBDIR)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            ns, cores = float("nan"), -1
            status = f"err:{type(e).__name__}"
            logger.error(f"[sweep] {cid} failed: {e}")
        finally:
            os.environ.pop("BGE_M3_B16MLP_CASE", None)
        results.append(
            dict(
                variant=v.name,
                case_id=cid,
                us=ns / 1000.0,
                cores=cores,
                status=status,
                is_default=(v.name == "default"),
            )
        )
        _write_csv(results, csv_path)
    logger.info(f"[sweep] CSV: {csv_path}")
    _print_table(results)


def _write_csv(results, path):
    import csv

    cols = ["variant", "case_id", "us", "cores", "status", "is_default"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})


def _print_table(results):
    base = next((r for r in results if r["is_default"]), None)
    base_us = base["us"] if base and base["status"] == "ok" else float("nan")
    logger.info("")
    logger.info("=" * 92)
    logger.info(f"  B16 MLP-wi sweep (M={M} K={K} N={N} GELU) — baseline default: {base_us:.3f} µs")
    logger.info("=" * 92)
    logger.info(f"  {'variant':<34}{'us':>10}{'vs_base':>9}{'cores':>7}  status")
    for r in sorted(results, key=lambda r: (r["status"] != "ok", r["us"])):
        vs = f"{base_us / r['us']:.3f}x" if (r["status"] == "ok" and base_us == base_us and base_us > 0) else "-"
        tag = "  <= DEFAULT" if r["is_default"] else ""
        logger.info(f"  {r['variant']:<34}{r['us']:>10.3f}{vs:>9}{r['cores']:>7}  {r['status']}{tag}")
    ok = [r for r in results if r["status"] == "ok" and not r["is_default"]]
    if ok and base_us == base_us and base_us > 0:
        best = min(ok, key=lambda r: r["us"])
        if best["us"] < base_us:
            logger.info(
                f"  >>> BEST: {best['variant']} = {best['us']:.3f} µs ({base_us / best['us']:.3f}x vs default {base_us:.3f})"
            )
        else:
            logger.info(f"  >>> No config beat default ({base_us:.3f} µs).")
    logger.info("=" * 92)
