# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Brute-force matmul config sweep with in-process device-kernel-time readback.

Flow per config:
  1. build the FULL logical weight + push to the mesh with the model's real
     mesh mapper (so each chip computes its true per-device shard);
  2. build the matching activation (replicated, or K-sharded for row-parallel);
  3. warm up (JIT compile + cache), then run the matmul `iters` times, reading
     the device kernel duration after each via the profiler getters;
  4. per chip take the min over iters (denoise); across chips take max (the
     bottleneck) for the matmul, avg for the optional collective (CCL rule).

No env vars in the interface, no CSV: `sweeps/__init__` set the profiler flags at
import time; durations come back as Python ints.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from . import profiler_setup as prof
from .targets import MatmulTarget, SweepAxis

import ttnn

_DTYPE = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
_FIDELITY = {
    "lofi": ttnn.MathFidelity.LoFi,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "hifi3": ttnn.MathFidelity.HiFi3,
    "hifi4": ttnn.MathFidelity.HiFi4,
}
_TILE = 32


# --------------------------------------------------------------------------- #
# mesh mappers matching tt/layer_weights.py                                    #
# --------------------------------------------------------------------------- #
def _weight_mapper(mesh, shard: str):
    ms = list(mesh.shape)
    if shard == "replicate":
        return ttnn.ReplicateTensorToMesh(mesh)
    if shard == "col":  # shard N (out) across TP cols
        return ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=ms)
    if shard == "row":  # shard K (in) across TP cols
        return ttnn.ShardTensor2dMesh(mesh, dims=(None, 2), mesh_shape=ms)
    raise NotImplementedError(f"weight mapper for shard={shard!r} (phase 2)")


def _act_mapper(mesh, shard: str):
    ms = list(mesh.shape)
    if shard in ("replicate", "col"):  # full-K activation, replicated
        return ttnn.ReplicateTensorToMesh(mesh)
    if shard == "row":  # activation K-sharded to match weight K-shard
        return ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=ms)
    raise NotImplementedError(f"act mapper for shard={shard!r} (phase 2)")


def _per_device_kn(mesh, t: MatmulTarget) -> tuple[int, int]:
    tp = int(mesh.shape[1])
    if t.shard == "col":
        return t.k_full, t.n_full // tp
    if t.shard == "row":
        return t.k_full // tp, t.n_full
    return t.k_full, t.n_full


# --------------------------------------------------------------------------- #
# program / memory config builders                                            #
# --------------------------------------------------------------------------- #
def _ckc(fidelity: str, fp32_acc: bool, packer: bool):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=_FIDELITY[fidelity],
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=packer,
    )


def _out_mem_config(mesh, out_mem: str, m: int, n_perdev: int):
    if out_mem == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if out_mem == "l1":
        return ttnn.L1_MEMORY_CONFIG
    if out_mem == "l1_width_sharded":
        # shard N across as many cores as evenly divide n_tiles (<= device grid)
        n_tiles = max(1, n_perdev // _TILE)
        grid = mesh.compute_with_storage_grid_size()
        max_cores = int(grid.x) * int(grid.y)
        cores = 1
        for c in range(min(max_cores, n_tiles), 0, -1):
            if n_tiles % c == 0:
                cores = c
                break
        gx = min(int(grid.x), cores)
        gy = max(1, cores // gx)
        return ttnn.create_sharded_memory_config(
            shape=(1, 1, max(m, _TILE), n_perdev),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    raise ValueError(f"unknown out_mem {out_mem!r}")


def _prog_config(mesh, prog, m: int, k_perdev: int, n_perdev: int):
    """Return (program_config_or_None, label). Grids are clamped to the device grid."""
    if prog == "auto":
        return None, "auto"
    if not isinstance(prog, dict):
        raise ValueError(f"prog must be 'auto' or dict, got {prog!r}")

    grid = mesh.compute_with_storage_grid_size()
    gx = min(int(prog.get("grid_x", grid.x)), int(grid.x))
    gy = min(int(prog.get("grid_y", grid.y)), int(grid.y))
    kt = max(1, k_perdev // _TILE)
    ibw = int(prog.get("in0_block_w", min(4, kt)))
    mt = max(1, (m + _TILE - 1) // _TILE)
    pcm = int(prog.get("per_core_M", mt))
    pcn = int(prog.get("per_core_N", max(1, (n_perdev // _TILE) // max(1, gx * gy))))
    osh = int(prog.get("out_subblock_h", 1))
    osw = int(prog.get("out_subblock_w", min(2, pcn)))
    label = f"{prog.get('type', '1d')} {gx}x{gy} ibw{ibw} pcM{pcm} pcN{pcn}"

    if prog.get("type", "1d") == "2d":
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=osh,
            out_subblock_w=osw,
            per_core_M=pcm,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            transpose_mcast=False,
        )
    else:
        cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=osh,
            out_subblock_w=osw,
            per_core_M=pcm,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    return cfg, label


# --------------------------------------------------------------------------- #
# measurement                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    name: str
    shard: str
    phase: str
    batch: int
    seq_len: int
    m: int
    in_dtype: str
    wt_dtype: str
    fidelity: str
    fp32_acc: bool
    packer: bool
    out_mem: str
    prog: str
    per_dev_k: int
    per_dev_n: int
    matmul_ns: int | None = None
    collective_ns: int | None = None
    status: str = "ok"


def _agg_max_over_chips_min_over_iters(iter_readings: list[dict[int, int]]) -> int:
    """iter_readings: list per iter of {chip: dominant_ns}. Return max_chip(min_iter)."""
    chips = set().union(*[set(r) for r in iter_readings]) if iter_readings else set()
    per_chip_min = {c: min(r.get(c, 0) for r in iter_readings) for c in chips}
    return max(per_chip_min.values()) if per_chip_min else 0


def measure_one(
    mesh,
    t: MatmulTarget,
    *,
    m,
    phase,
    batch,
    seq_len,
    in_dtype,
    wt_dtype,
    fidelity,
    fp32_acc,
    packer,
    out_mem,
    prog,
    iters,
    warmup,
) -> Result:
    import torch

    k_pd, n_pd = _per_device_kn(mesh, t)
    r = Result(
        t.name,
        t.shard,
        phase,
        batch,
        seq_len,
        m,
        in_dtype,
        wt_dtype,
        fidelity,
        fp32_acc,
        packer,
        out_mem,
        prog if isinstance(prog, str) else "custom",
        k_pd,
        n_pd,
    )

    a_dev = b_dev = None
    try:
        # weight [1,1,K_full,N_full] -> sharded to per-device via model mapper
        w = torch.randn(1, 1, t.k_full, t.n_full, dtype=torch.float32) * 0.02
        b_dev = ttnn.from_torch(
            w,
            dtype=_DTYPE[wt_dtype],
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=_weight_mapper(mesh, t.shard),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # activation [1,1,M,K_full] -> replicated (col/replicate) or K-sharded (row)
        a = torch.randn(1, 1, max(m, 1), t.k_full, dtype=torch.float32) * 0.02
        a_dev = ttnn.from_torch(
            a,
            dtype=_DTYPE[in_dtype],
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=_act_mapper(mesh, t.shard),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ckc = _ckc(fidelity, fp32_acc, packer)
        out_mc = _out_mem_config(mesh, out_mem, m, n_pd)
        prog_cfg, r.prog = _prog_config(mesh, prog, m, k_pd, n_pd)

        kw = dict(memory_config=out_mc, compute_kernel_config=ckc)
        if prog_cfg is not None:
            kw["program_config"] = prog_cfg

        # warmup
        for _ in range(warmup):
            o = ttnn.linear(a_dev, b_dev, **kw)
            ttnn.deallocate(o)
        prof.read_latest_kernel_ns(mesh)  # flush warmup out of "latest"

        readings = []
        last_out = None
        for _ in range(iters):
            o = ttnn.linear(a_dev, b_dev, **kw)
            readings.append(prof.dominant_kernel_ns(mesh))  # sync+read inside
            if last_out is not None:
                ttnn.deallocate(last_out)
            last_out = o
        r.matmul_ns = _agg_max_over_chips_min_over_iters(readings)

        # optional collective on the matmul output
        if t.collective != "none" and last_out is not None:
            r.collective_ns = _measure_collective(mesh, last_out, t.collective)
        if last_out is not None:
            ttnn.deallocate(last_out)

    except Exception as e:  # brute force: record and move on
        r.status = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
    finally:
        for x in (a_dev, b_dev):
            try:
                if x is not None:
                    ttnn.deallocate(x)
            except Exception:
                pass
    return r


def _measure_collective(mesh, tensor, kind: str) -> int | None:
    """Time the CCL following a matmul. Aggregated avg across chips (CCL rule)."""
    try:
        prof.read_latest_kernel_ns(mesh)  # clear window
        if kind == "all_reduce":
            _ = ttnn.all_reduce(tensor, num_links=1, topology=ttnn.Topology.Linear, cluster_axis=1)
        elif kind == "all_gather":
            _ = ttnn.all_gather(tensor, dim=3, num_links=1, topology=ttnn.Topology.Linear, cluster_axis=1)
        elif kind == "reduce_scatter":
            _ = ttnn.reduce_scatter(tensor, dim=3, num_links=1, topology=ttnn.Topology.Linear, cluster_axis=1)
        else:
            return None
        per_chip = prof.sum_kernel_ns(mesh)  # sum: a CCL may launch several kernels
        vals = [v for v in per_chip.values() if v > 0]
        return int(sum(vals) / len(vals)) if vals else 0
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# sweep driver                                                                 #
# --------------------------------------------------------------------------- #
def run_sweep(mesh, targets: list[MatmulTarget], axis: SweepAxis) -> list[Result]:
    grid = mesh.compute_with_storage_grid_size()
    print(
        f"[sweep] mesh={tuple(mesh.shape)} device_grid=({grid.x},{grid.y}) " f"profiler_ready={prof.profiler_ready()}",
        flush=True,
    )

    cfg_combos = list(
        itertools.product(
            axis.in_dtype,
            axis.wt_dtype,
            axis.fidelity,
            axis.fp32_acc,
            axis.packer_l1_acc,
            axis.out_mem,
            axis.prog,
        )
    )
    total = len(targets) * len(axis.phases) * len(cfg_combos)
    print(
        f"[sweep] {len(targets)} targets x {len(axis.phases)} phase-specs x {len(cfg_combos)} "
        f"configs = {total} runs (prefill_chunk={axis.prefill_chunk} dp_rows={axis.dp_rows})",
        flush=True,
    )

    results: list[Result] = []
    n = 0
    for t in targets:
        for ps in axis.phases:
            m = ps.m_tokens(prefill_chunk=axis.prefill_chunk, dp_rows=axis.dp_rows, dp_split=t.dp_split)
            for ind, wtd, fid, fp32, pk, omem, prog in cfg_combos:
                n += 1
                t0 = time.perf_counter()
                r = measure_one(
                    mesh,
                    t,
                    m=m,
                    phase=ps.phase,
                    batch=ps.batch,
                    seq_len=ps.seq_len,
                    in_dtype=ind,
                    wt_dtype=wtd,
                    fidelity=fid,
                    fp32_acc=fp32,
                    packer=pk,
                    out_mem=omem,
                    prog=prog,
                    iters=axis.iters,
                    warmup=axis.warmup,
                )
                results.append(r)
                mm = f"{r.matmul_ns}ns" if r.matmul_ns is not None else "-"
                cc = f" ccl={r.collective_ns}ns" if r.collective_ns is not None else ""
                tag = r.status if r.status != "ok" else f"mm={mm}{cc}"
                print(
                    f"[{n}/{total}] {t.name} {ps.phase} b{ps.batch} s{ps.seq_len} M{m} "
                    f"{ind}/{wtd} {fid} acc{int(fp32)} pk{int(pk)} {omem} {r.prog} -> {tag} "
                    f"({time.perf_counter()-t0:.2f}s)",
                    flush=True,
                )
    return results


def print_table(results: list[Result]) -> None:
    ok = [r for r in results if r.status == "ok" and r.matmul_ns is not None]
    ok.sort(key=lambda r: (r.name, r.phase, r.batch, r.matmul_ns))
    print("\n=== sweep results (per-device kernel time, sorted fastest-first per target/phase/batch) ===")
    hdr = (
        f"{'target':18} {'phase':>7} {'b':>3} {'seq':>4} {'M':>5} {'K/dev':>6} {'N/dev':>6} "
        f"{'in/wt':>9} {'fid':>6} {'acc':>3} {'mem':>16} {'prog':>16} {'matmul_ns':>10} {'ccl_ns':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        print(
            f"{r.name:18} {r.phase:>7} {r.batch:>3} {r.seq_len:>4} {r.m:>5} {r.per_dev_k:>6} {r.per_dev_n:>6} "
            f"{r.in_dtype+'/'+r.wt_dtype:>9} {r.fidelity:>6} {int(r.fp32_acc):>3} "
            f"{r.out_mem:>16} {r.prog:>16} {r.matmul_ns:>10} "
            f"{(r.collective_ns if r.collective_ns is not None else ''):>8}"
        )
    fails = [r for r in results if r.status != "ok"]
    if fails:
        print(f"\n{len(fails)} failed configs (first 15):")
        for r in fails[:15]:
            print(f"  {r.name} m{r.m} {r.in_dtype}/{r.wt_dtype} {r.fidelity} {r.out_mem} {r.prog}: {r.status}")
