# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark of ``ttnn.sparse_matmul`` at the GLM-5.3-Flash per-chip routed-expert shapes (ONE chip).

Why: on the 8x4 Galaxy the hybrid placement gives every chip W13 [1, 72, 4096, 512] and W2 [1, 72, 256, 4096]
(bfp8) and ``tt/moe.py`` streams them with the 1D mcast_in0 sparse matmul, whose only parallel axis is the
output width (cores = Nt / per_core_N: 16 cores at N = 512) and whose per-K-block pipeline (sender memory read ->
semaphore round trip -> multicast; per-core in1 read -> barrier) costs a memory latency per block. This sweep
measures what the kernel reaches per (shape, program config, active experts) and what the alternatives cost.

Stages (``GLM53_EXP_STAGE``, comma list; default ``w13,w2,active,tokens,place``):
  w13     : x [1,1,32,4096] bf16 x W13 [1,72,4096,N] bfp8 (b_sparse), N in {512,1024,2048,4096}, 32 active:
            core grid x in0_block_w x out_subblock_w sweep
  w2      : h [1,72,32,I] bf16 x W2 [1,72,I,4096] bfp8 (a+b sparse), I in {256,512,1024,2048}, 32 active
  active  : 8 / 32 / 72 active experts, nnz None / exact / indices (gather) mode, in0 in L1, at the baseline
            (moe.py today) and the best configs of the sweep
  tokens  : M in {32,64,128,256} in one call vs 32-token blocks at the per-chip shapes (N=512, I=256)
  place   : the placement alternatives at 8 / 64 / 256 tokens: hybrid EP4xTP8 (72 x [4096,512]), EP8xTP4
            (36 x [4096,1024]), EP32 (9 x [4096,4096]) and the expert-grouped hybrid (72/g x [4096, 512g]),
            active counts from a Monte-Carlo of uniform top-8 routing (max over the chips)
  dense   : dense alternatives over all 72 experts: transposed tall-skinny matmul (W13^T as in0, the expert x
            width axis split over 72-128 cores, x^T multicast) and the batched bmm for W2^T
  dense2  : dense MoE over all 72 experts as ONE matmul each (W13 [4096, 72x512], W2 [72x256, 4096]; the expert
            sum is the K reduction): auto / 1D mcast_in0 / 2D mcast configs at M = 32..256

Collect (device profiler build, no watcher):
    TT_VISIBLE_DEVICES=2 python -m tracy -r -p -v -m pytest models/demos/glm53_flash/tests/test_expert_kernel_perf.py -s
    python models/demos/glm53_flash/tests/test_expert_kernel_perf.py --report <ops_perf_results.csv> --cases <cases.json>
Each timed case is one ``C<id>`` .. ``C<id>_END`` signpost window holding ``GLM53_EXP_REPS`` identical calls;
the report divides the window's matmul device kernel time by the repetitions (what ``tt-perf-report
--start-signpost C<id> --end-signpost C<id>_END`` prints as Device Time) and reports us per call, GB/s of expert
weight bytes (bfp8 tile bytes: 1088 B per 32x32 tile = 1.0625 B/element) and % of the 512 GB/s DRAM peak.
Every case is checked against a torch reference (bfp8-rounded weights) or against the baseline configuration's
output of the same case. ``GLM53_EXP_CASES`` names the sidecar JSON (default ``_logs/optexperts_cases.json``).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.utils import pcc

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy module ships with the source tree

    def signpost(header, message=None):
        pass


TILE = 32
E_GLOBAL, TOPK, H = 288, 8, 4096
BFP8_TILE = 1088  # bytes per 32x32 bfp8_b tile (1024 B mantissas + 64 B exponents)
BF16_TILE = 2048
PEAK_GBS = 512.0
L1_DFB_BUDGET = 1_150_000  # bytes of dataflow buffers per core we allow (1.46 MB L1 minus base + l1_small)

STAGES = os.environ.get("GLM53_EXP_STAGE", "w13,w2,active,tokens,place").split(",")
REPS = int(os.environ.get("GLM53_EXP_REPS", "4"))
CASES_JSON = Path(os.environ.get("GLM53_EXP_CASES", "/home/ttuser/ashai/glm53/_logs/optexperts_cases.json"))


# ----------------------------------------------------------------------------------------------- helpers
def _read_profiler(device):
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:  # not built with the profiler
        pass


def _compute(device, fidelity=ttnn.MathFidelity.HiFi2):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def rect_for(cores: int, grid: tuple[int, int]) -> tuple[int, int] | None:
    """Squarest (x, y) rectangle of exactly ``cores`` cores inside ``grid``; None if impossible."""
    best = None
    for y in range(1, grid[1] + 1):
        if cores % y:
            continue
        x = cores // y
        if x > grid[0]:
            continue
        if best is None or abs(x - y) < abs(best[0] - best[1]):
            best = (x, y)
    return best


def core_counts(Nt: int, grid: tuple[int, int], top: int = 2) -> list[int]:
    """Core counts c | Nt with a rectangle in the grid, largest first."""
    cs = [c for c in range(1, grid[0] * grid[1] + 1) if Nt % c == 0 and rect_for(c, grid) is not None]
    return sorted(cs, reverse=True)[:top]


def l1_dfb_bytes(Mt: int, in0_block_w: int, per_core_N: int, in0_tile: int = BF16_TILE) -> int:
    in0 = 2 * Mt * in0_block_w * in0_tile
    in1 = 2 * in0_block_w * per_core_N * BFP8_TILE
    out = Mt * per_core_N * BF16_TILE
    interm = Mt * per_core_N * 4096  # fp32 partials (separate from the bf16 output)
    return in0 + in1 + out + interm


def mk_config(
    cores_xy: tuple[int, int],
    m: int,
    n: int,
    k: int,
    in0_block_w: int,
    out_subblock_w: int = 1,
    out_subblock_h: int = 1,
    per_core_N: int | None = None,
):
    cx, cy = cores_xy
    Mt, Nt, Kt = math.ceil(m / TILE), math.ceil(n / TILE), math.ceil(k / TILE)
    if per_core_N is None:  # a grid larger than Nt / per_core_N = batch groups (the kernel's own tiling)
        per_core_N = max(1, math.ceil(Nt / (cx * cy)))
    assert Kt % in0_block_w == 0, (Kt, in0_block_w)
    assert per_core_N % out_subblock_w == 0 and Mt % out_subblock_h == 0, (per_core_N, out_subblock_w, Mt)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        out_block_h=Mt,
        out_block_w=per_core_N,
        per_core_M=Mt,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def cfg_candidates(m: int, n: int, k: int, grid: tuple[int, int], ibws=(4, 16, 32, 64, 128), subs=(1, 2, 4), top=2):
    """(cores_xy, in0_block_w, out_subblock_w) triples that divide, fit L1 and respect the dest limit."""
    Mt, Nt, Kt = math.ceil(m / TILE), math.ceil(n / TILE), math.ceil(k / TILE)
    out = []
    for cores in core_counts(Nt, grid, top=top):
        xy = rect_for(cores, grid)
        per_core_N = Nt // cores
        for ibw in ibws:
            if Kt % ibw or l1_dfb_bytes(Mt, ibw, per_core_N) > L1_DFB_BUDGET:
                continue
            for sw in subs:
                if per_core_N % sw or sw > 4:
                    continue
                out.append((xy, ibw, sw))
    return out


def mc_active(T: int, E_local: int, trials: int = 200, seed: int = 0) -> tuple[float, float]:
    """Distinct active experts per chip for T tokens of uniform top-8 routing over 288 experts, when a chip
    holds a contiguous group of ``E_local`` experts: (mean over groups, mean over trials of the max over groups)."""
    g = torch.Generator().manual_seed(seed)
    n_groups = E_GLOBAL // E_local
    means, maxes = [], []
    for _ in range(trials):
        picks = torch.stack([torch.randperm(E_GLOBAL, generator=g)[:TOPK] for _ in range(T)]).flatten()
        counts = [
            len(torch.unique(picks[(picks >= lo) & (picks < lo + E_local)])) for lo in range(0, E_GLOBAL, E_local)
        ]
        means.append(sum(counts) / n_groups)
        maxes.append(max(counts))
    return sum(means) / trials, sum(maxes) / trials


def bfp8_round(w: torch.Tensor) -> torch.Tensor:
    """Host bfp8_b round trip (the weight values the device actually multiplies)."""
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


class Bench:
    """Timed cases with signpost windows; a sidecar JSON maps the window label to the case description."""

    def __init__(self, device):
        self.device = device
        self.cases: dict[str, dict] = {}
        self.n = 0
        grid = device.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        self.compute = _compute(device)
        # earlier runs' report CSVs (``--report --out``): their device-time winners seed ``best``
        self.prior: list[dict] = []
        for f in filter(None, os.environ.get("GLM53_EXP_PRIOR", "").split(",")):
            import pandas as pd

            for r in pd.read_csv(f).to_dict("records"):
                if isinstance(r.get("name"), str) and r.get("device_us") == r.get("device_us"):
                    self.prior.append(r)
        logger.info(
            f"device grid {self.grid} ({grid.x * grid.y} cores), REPS={REPS}, stages {STAGES}, "
            f"{len(self.prior)} prior cases"
        )

    def run(self, meta: dict, fn, check=None):
        """Compile + check once, then REPS signposted calls. Returns the checked output (torch) or None."""
        label = f"C{self.n:04d}"
        self.n += 1
        meta = dict(meta, label=label, reps=REPS)
        try:
            out = fn()
            ttnn.synchronize_device(self.device)
        except Exception as e:  # host-side rejection (validation / L1 allocation): recorded, not fatal
            msg = str(e).strip().splitlines()[-1][:300] if str(e).strip() else repr(e)
            meta["error"] = msg
            self.cases[label] = meta
            logger.warning(f"{label} INVALID {meta.get('name')}: {msg}")
            self._dump()
            return None
        res = None
        if check is not None:
            res = check(out)
            if isinstance(res, dict):
                meta.update(res)
        ttnn.deallocate(out)
        _read_profiler(self.device)
        signpost(label)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(REPS)]
        ttnn.synchronize_device(self.device)
        wall = (time.perf_counter() - t0) / REPS
        signpost(f"{label}_END")
        _read_profiler(self.device)
        for o in outs:
            ttnn.deallocate(o)
        meta["wall_us"] = wall * 1e6
        gbs = meta["bytes"] / wall / 1e9 if meta.get("bytes") else 0.0
        self.cases[label] = meta
        logger.info(
            f"{label} {meta.get('name')}: wall {wall*1e6:8.1f} us/call ({gbs:6.1f} GB/s wall-based) "
            f"{ {k: v for k, v in meta.items() if k in ('pcc', 'maxdiff', 'pcc_vs_base')} }"
        )
        self._dump()
        return res

    def _dump(self):
        CASES_JSON.parent.mkdir(parents=True, exist_ok=True)
        CASES_JSON.write_text(json.dumps(self.cases, indent=1))

    def best(self, name_prefix: str) -> dict | None:
        """The fastest valid case whose name starts with the prefix: device time from the prior report CSVs when
        present, else the wall time of this run."""

        def plain(r):  # a sweep point, not a mode variant of one
            return not any(str(r.get(k, "")) in ("True", "exact", "L1") for k in ("indices", "in0_l1", "out_l1", "nnz"))

        cands = [(float(r["device_us"]), r) for r in self.prior if r["name"].startswith(name_prefix) and plain(r)]
        if not cands:
            cands = [
                (m["wall_us"], m)
                for m in self.cases.values()
                if "wall_us" in m and m["name"].startswith(name_prefix) and plain(m)
            ]
        return min(cands, key=lambda c: c[0])[1] if cands else None


# ----------------------------------------------------------------------------------------------- shapes
class ExpertShape:
    """Random expert weights of one per-chip layout on the device: W13 [1,E,K,N] and/or W2 [1,E,I,H]."""

    def __init__(self, device, E: int, n13: int | None, i2: int | None, seed: int = 0, w13_t: bool = False):
        self.device, self.E, self.n13, self.i2 = device, E, n13, i2
        g = torch.Generator().manual_seed(seed)
        self.w13_host = self.w2_host = None
        if n13 is not None:
            self.w13_host = (torch.randn(1, E, H, n13, generator=g) * 0.02).bfloat16()
            t0 = time.perf_counter()
            self.w13 = ttnn.from_torch(self.w13_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
            logger.info(f"W13 [1,{E},{H},{n13}] bfp8 uploaded in {time.perf_counter()-t0:.1f} s")
            self.w13_t = None
            if w13_t:
                self.w13_t = ttnn.from_torch(
                    self.w13_host.transpose(-1, -2).contiguous(),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
        if i2 is not None:
            self.w2_host = (torch.randn(1, E, i2, H, generator=g) * 0.02).bfloat16()
            t0 = time.perf_counter()
            self.w2 = ttnn.from_torch(self.w2_host, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
            logger.info(f"W2 [1,{E},{i2},{H}] bfp8 uploaded in {time.perf_counter()-t0:.1f} s")
        self._ref13: dict[int, torch.Tensor] = {}
        self._ref2: dict[int, torch.Tensor] = {}

    def ref13(self, e: int) -> torch.Tensor:
        if e not in self._ref13:
            self._ref13[e] = bfp8_round(self.w13_host[0, e])
        return self._ref13[e]

    def ref2(self, e: int) -> torch.Tensor:
        if e not in self._ref2:
            self._ref2[e] = bfp8_round(self.w2_host[0, e])
        return self._ref2[e]

    def bytes13(self, n_active: int) -> int:
        return n_active * (H // TILE) * (self.n13 // TILE) * BFP8_TILE

    def bytes2(self, n_active: int) -> int:
        return n_active * (self.i2 // TILE) * (H // TILE) * BFP8_TILE

    def free(self):
        for t in ("w13", "w2", "w13_t"):
            if getattr(self, t, None) is not None:
                ttnn.deallocate(getattr(self, t))


def active_ids(E: int, n: int, seed: int = 1) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(E, generator=g)[:n].tolist())


def sparsity_tensor(device, E: int, ids: list[int]):
    s = torch.zeros(1, 1, 1, E, dtype=torch.bfloat16)
    s[..., ids] = 1.0
    return ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def indices_tensor(device, ids: list[int]):
    t = torch.tensor(ids, dtype=torch.int32).reshape(1, 1, 1, len(ids))
    return ttnn.from_torch(t, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


# ----------------------------------------------------------------------------------------------- cases
class Sweep:
    def __init__(self, bench: Bench):
        self.b = bench
        self.dev = bench.device
        self.base_out: dict[tuple, torch.Tensor] = {}  # (kind, E, N/I, M, n_active) -> baseline output

    # ---- W13: x [1,1,M,H] x W13 [1,E,H,N] (b_sparse) -> [1,1,1,E,M,N]
    def w13_case(
        self,
        sh: ExpertShape,
        ids,
        M,
        cfg_tuple,
        *,
        name,
        nnz=None,
        indices=False,
        in0_l1=False,
        out_l1=False,
        in0_dtype=ttnn.bfloat16,
        check_torch=True,
        extra=None,
        zero_fill=True,
        transposed=False,
    ):
        xy, ibw, sw = cfg_tuple
        x_host = torch.randn(1, 1, M, H, generator=torch.Generator().manual_seed(7)).bfloat16()
        x = ttnn.from_torch(
            x_host,
            dtype=in0_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.L1_MEMORY_CONFIG if in0_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        sp = sparsity_tensor(self.dev, sh.E, ids)
        idx = indices_tensor(self.dev, ids) if indices else None
        pc = mk_config(xy, M, sh.n13, H, ibw, sw)
        kw = dict(
            sparsity=sp,
            nnz=(len(ids) if nnz == "exact" else None),
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            program_config=pc,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.b.compute,
            memory_config=ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        if idx is not None:
            kw["indices"] = idx
        if not zero_fill:
            kw["zero_fill_output"] = False
        if transposed:  # the [1, E, N, K] copy of the weights (transposed tiles)
            kw["in1_transposed"] = True
        w13 = sh.w13_t if transposed else sh.w13

        def fn():
            return ttnn.sparse_matmul(x, w13, **kw)

        key = ("w13", sh.E, sh.n13, M, len(ids))

        def check(out):
            y = ttnn.to_torch(out).float()
            y = y.reshape(-1, M, sh.n13) if indices else y.reshape(sh.E, M, sh.n13)
            got = y if indices else y[ids]
            res = {}
            if check_torch:
                ref = torch.stack([x_host[0, 0].float() @ sh.ref13(e) for e in ids])
                res["pcc"] = pcc(ref, got)
                res["maxdiff"] = float((ref - got).abs().max())
                if not indices and zero_fill:
                    inactive = [e for e in range(sh.E) if e not in ids]
                    res["inactive_max"] = float(y[inactive].abs().max()) if inactive else 0.0
            if key in self.base_out:
                res["pcc_vs_base"] = pcc(self.base_out[key], got)
            else:
                self.base_out[key] = got
            bar = 0.99 if in0_dtype != ttnn.bfloat16 else 0.999
            assert res.get("pcc", 1.0) > bar and res.get("pcc_vs_base", 1.0) > bar, (name, res)
            return res

        meta = dict(
            name=name,
            kind="w13",
            E=sh.E,
            K=H,
            N=sh.n13,
            M=M,
            active=len(ids),
            cores=xy[0] * xy[1],
            grid=list(xy),
            in0_block_w=ibw,
            out_subblock_w=sw,
            per_core_N=math.ceil(sh.n13 // TILE / (xy[0] * xy[1])),
            nnz=nnz or "none",
            indices=indices,
            in0_l1=in0_l1,
            out_l1=out_l1,
            in0_dtype=str(in0_dtype).split(".")[-1],
            bytes=sh.bytes13(len(ids)),
            ops_per_call=2 if zero_fill else 1,
            zero_fill=zero_fill,
            transposed=transposed,
        )
        if extra:
            meta.update(extra)
        res = self.b.run(meta, fn, check)
        for t in (x, sp, idx):
            if t is not None:
                ttnn.deallocate(t)
        return res

    # ---- W2: h [1,E,M,I] x W2 [1,E,I,H] (a+b sparse) -> [1,E,M,H]
    def w2_case(
        self,
        sh: ExpertShape,
        ids,
        M,
        cfg_tuple,
        *,
        name,
        nnz=None,
        indices=False,
        out_l1=False,
        check_torch=True,
        extra=None,
        accumulate=False,
    ):
        xy, ibw, sw = cfg_tuple
        h_host = torch.randn(1, sh.E, M, sh.i2, generator=torch.Generator().manual_seed(8)).bfloat16()
        if indices:  # compact A: rows in slot order of the ids
            h_dev = torch.stack([h_host[0, e] for e in ids]).unsqueeze(0)
        else:
            h_dev = h_host
        hd = ttnn.from_torch(h_dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
        sp = sparsity_tensor(self.dev, sh.E, ids)
        idx = indices_tensor(self.dev, ids) if indices else None
        pc = mk_config(xy, M, H, sh.i2, ibw, sw, per_core_N=4 if xy[0] * xy[1] > 32 and accumulate else None)
        kw = dict(
            sparsity=sp,
            nnz=(len(ids) if nnz == "exact" else None),
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=pc,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.b.compute,
            memory_config=ttnn.L1_MEMORY_CONFIG if out_l1 else ttnn.DRAM_MEMORY_CONFIG,
        )
        if idx is not None:
            kw["indices"] = idx
        if accumulate:
            kw["accumulate_batch"] = True

        def fn():
            return ttnn.sparse_matmul(hd, sh.w2, **kw)

        key = ("w2", sh.E, sh.i2, M, len(ids))

        def check(out):
            res = {}
            if accumulate:  # [1, G, M, H]: the sum over the active experts (per group), compared as the total
                y = ttnn.to_torch(out).float().sum(dim=1).reshape(M, H)
                res["groups"] = int(out.shape[1])
                if check_torch:
                    ref = sum((h_host[0, e].float() @ sh.ref2(e) for e in ids), torch.zeros(M, H))
                    res["pcc"] = pcc(ref, y)
                    res["maxdiff"] = float((ref - y).abs().max())
                if key in self.base_out:
                    res["pcc_vs_base"] = pcc(self.base_out[key].sum(dim=0), y)
                assert res.get("pcc", 1.0) > 0.999 and res.get("pcc_vs_base", 1.0) > 0.9999, (name, res)
                return res
            y = ttnn.to_torch(out).float().reshape(-1, M, H)
            got = y if indices else y[ids]
            if check_torch:
                ref = torch.stack([h_host[0, e].float() @ sh.ref2(e) for e in ids])
                res["pcc"] = pcc(ref, got)
                res["maxdiff"] = float((ref - got).abs().max())
            if key in self.base_out:
                res["pcc_vs_base"] = pcc(self.base_out[key], got)
            else:
                self.base_out[key] = got
            assert res.get("pcc", 1.0) > 0.999 and res.get("pcc_vs_base", 1.0) > 0.999, (name, res)
            return res

        meta = dict(
            name=name,
            kind="w2",
            E=sh.E,
            K=sh.i2,
            N=H,
            M=M,
            active=len(ids),
            cores=xy[0] * xy[1],
            grid=list(xy),
            in0_block_w=ibw,
            out_subblock_w=sw,
            per_core_N=math.ceil(H // TILE / (xy[0] * xy[1])),
            nnz=nnz or "none",
            indices=indices,
            out_l1=out_l1,
            bytes=sh.bytes2(len(ids)),
            ops_per_call=2,
            accumulate=accumulate,
        )
        if extra:
            meta.update(extra)
        res = self.b.run(meta, fn, check)
        for t in (hd, sp, idx):
            if t is not None:
                ttnn.deallocate(t)
        return res

    # ---- dense transposed: W13^T [1,E,N,H] (in0, fused batch) x x^T [1,1,H,M] -> [1,E,N,M]; mcast_in0=False
    def dense_t_case(self, sh: ExpertShape, M, cores: int, per_core_M: int, ibw: int, sub_hw, *, name, extra=None):
        x_host = torch.randn(1, 1, M, H, generator=torch.Generator().manual_seed(7)).bfloat16()
        xt = ttnn.from_torch(
            x_host.transpose(-1, -2).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev
        )
        Mt_x = math.ceil(M / TILE)
        xy = rect_for(cores, self.b.grid) or self.b.grid
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=xy,
            in0_block_w=ibw,
            out_subblock_h=sub_hw[0],
            out_subblock_w=sub_hw[1],
            out_block_h=per_core_M,
            out_block_w=Mt_x,
            per_core_M=per_core_M,
            per_core_N=Mt_x,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )

        def fn():
            return ttnn.matmul(
                sh.w13_t,
                xt,
                program_config=pc,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.b.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def check(out):
            y = ttnn.to_torch(out).float().reshape(sh.E, sh.n13, M).transpose(-1, -2)  # [E, M, N]
            ids = list(range(0, sh.E, max(1, sh.E // 8)))[:8]
            ref = torch.stack([x_host[0, 0].float() @ sh.ref13(e) for e in ids])
            return {"pcc": pcc(ref, y[ids]), "maxdiff": float((ref - y[ids]).abs().max())}

        meta = dict(
            name=name,
            kind="dense_t13",
            E=sh.E,
            K=H,
            N=sh.n13,
            M=M,
            active=sh.E,
            cores=cores,
            grid=list(xy),
            in0_block_w=ibw,
            per_core_M=per_core_M,
            sub=list(sub_hw),
            bytes=sh.bytes13(sh.E),
            ops_per_call=1,
        )
        if extra:
            meta.update(extra)
        res = self.b.run(meta, fn, check)
        ttnn.deallocate(xt)
        return res

    # ---- dense bmm for W2^T: W2^T [1,E,H,I] x h^T [1,E,I,M] -> [1,E,H,M] (one expert per core)
    def dense_bmm2_case(self, sh: ExpertShape, M, ibw: int, sub_hw, *, name, extra=None):
        w2t = ttnn.from_torch(
            sh.w2_host.transpose(-1, -2).contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=self.dev
        )
        h_host = torch.randn(1, sh.E, M, sh.i2, generator=torch.Generator().manual_seed(8)).bfloat16()
        ht = ttnn.from_torch(
            h_host.transpose(-1, -2).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev
        )
        Mt_x = math.ceil(M / TILE)
        xy = rect_for(sh.E, self.b.grid)
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=xy,
            in0_block_w=ibw,
            out_subblock_h=sub_hw[0],
            out_subblock_w=sub_hw[1],
            per_core_M=H // TILE,
            per_core_N=Mt_x,
        )

        def fn():
            return ttnn.matmul(
                w2t,
                ht,
                program_config=pc,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.b.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def check(out):
            y = ttnn.to_torch(out).float().reshape(sh.E, H, M).transpose(-1, -2)  # [E, M, H]
            ids = list(range(0, sh.E, max(1, sh.E // 8)))[:8]
            ref = torch.stack([h_host[0, e].float() @ sh.ref2(e) for e in ids])
            return {"pcc": pcc(ref, y[ids]), "maxdiff": float((ref - y[ids]).abs().max())}

        meta = dict(
            name=name,
            kind="dense_bmm2",
            E=sh.E,
            K=sh.i2,
            N=H,
            M=M,
            active=sh.E,
            cores=sh.E,
            grid=list(xy),
            in0_block_w=ibw,
            sub=list(sub_hw),
            bytes=sh.bytes2(sh.E),
            ops_per_call=1,
        )
        if extra:
            meta.update(extra)
        res = self.b.run(meta, fn, check)
        ttnn.deallocate(ht)
        ttnn.deallocate(w2t)
        return res


# ----------------------------------------------------------------------------------------------- stages
BASE13 = lambda grid, n: (rect_for(core_counts(n // TILE, grid, 1)[0], grid), 4, 1)  # noqa: E731  moe.py today
BASE2 = lambda grid, i: (rect_for(core_counts(H // TILE, grid, 1)[0], grid), min(4, i // TILE), 1)  # noqa: E731


def stage_w13(sw: Sweep, shapes: dict):
    for n in (512, 1024, 2048, 4096):
        sh = shapes[("w13", 72, n)]
        ids = active_ids(72, 32)
        base = BASE13(sw.b.grid, n)
        sw.w13_case(sh, ids, 32, base, name=f"w13 N={n} a=32 BASE c={base[0][0]*base[0][1]} ibw=4 sw=1")
        for xy, ibw, s in cfg_candidates(32, n, H, sw.b.grid):
            if (xy, ibw, s) == base:
                continue
            sw.w13_case(
                sh, ids, 32, (xy, ibw, s), name=f"w13 N={n} a=32 c={xy[0]*xy[1]} ibw={ibw} sw={s}", check_torch=False
            )
        if n == 512:  # grid shape at the same core count; in0 (x) in L1 and as bfp8 (the sender's in0 read + multicast
            # per expert is on the critical path at 16 cores; bfp8 in0 is NOT exact, measured for the byte model only)
            sw.w13_case(sh, ids, 32, ((8, 2), 128, 1), name="w13 N=512 a=32 c=16(8x2) ibw=128 sw=1", check_torch=False)
            sw.w13_case(sh, ids, 32, ((2, 8), 128, 1), name="w13 N=512 a=32 c=16(2x8) ibw=128 sw=1", check_torch=False)
            for ibw in (4, 16, 32, 64, 128):
                sw.w13_case(
                    sh,
                    ids,
                    32,
                    ((4, 4), ibw, 1),
                    name=f"w13 N=512 a=32 in0=L1 c=16 ibw={ibw}",
                    in0_l1=True,
                    check_torch=False,
                )
            for ibw in (32, 64):
                sw.w13_case(
                    sh,
                    ids,
                    32,
                    ((4, 4), ibw, 1),
                    name=f"w13 N=512 a=32 in0=bfp8 c=16 ibw={ibw}",
                    in0_dtype=ttnn.bfloat8_b,
                    check_torch=False,
                )
                sw.w13_case(
                    sh,
                    ids,
                    32,
                    ((4, 4), ibw, 1),
                    name=f"w13 N=512 a=32 in0=bfp8+L1 c=16 ibw={ibw}",
                    in0_dtype=ttnn.bfloat8_b,
                    in0_l1=True,
                    check_torch=False,
                )


def stage_w2(sw: Sweep, shapes: dict):
    for i in (256, 512, 1024, 2048):
        sh = shapes[("w2", 72, i)]
        ids = active_ids(72, 32)
        base = BASE2(sw.b.grid, i)
        sw.w2_case(sh, ids, 32, base, name=f"w2 I={i} a=32 BASE c={base[0][0]*base[0][1]} ibw={base[1]} sw=1")
        for xy, ibw, s in cfg_candidates(32, H, i, sw.b.grid, ibws=(2, 4, 8, 16, 32, 64)):
            if (xy, ibw, s) == base:
                continue
            sw.w2_case(
                sh, ids, 32, (xy, ibw, s), name=f"w2 I={i} a=32 c={xy[0]*xy[1]} ibw={ibw} sw={s}", check_torch=False
            )


def best_cfg(sw: Sweep, prefix: str, fallback):
    m = sw.b.best(prefix)
    if m is None:
        logger.info(f"best_cfg {prefix!r}: no measurement, fallback {fallback}")
        return fallback
    if m.get("indices") in (True, "True"):  # never seed from an indexed-mode case
        pass
    grid = m.get("grid")
    if isinstance(grid, str):
        grid = json.loads(grid.replace("(", "[").replace(")", "]"))
    if not isinstance(grid, list):  # older report CSVs carry only the core count
        grid = list(rect_for(int(m["cores"]), sw.b.grid))
    cfg = (tuple(int(v) for v in grid), int(m["in0_block_w"]), int(m["out_subblock_w"]))
    logger.info(f"best_cfg {prefix!r}: {m['name']} -> {cfg}")
    return cfg


def stage_active(sw: Sweep, shapes: dict):
    for n in (512, 1024, 2048, 4096):
        sh = shapes[("w13", 72, n)]
        base = BASE13(sw.b.grid, n)
        Kt = H // TILE
        fb = (
            rect_for(core_counts(n // TILE, sw.b.grid, 1)[0], sw.b.grid),
            128
            if l1_dfb_bytes(1, 128, max(1, n // TILE // core_counts(n // TILE, sw.b.grid, 1)[0])) <= L1_DFB_BUDGET
            else 32,
            1,
        )
        best = best_cfg(sw, f"w13 N={n} a=32 c=", fb)
        for a in (8, 32, 72):
            ids = active_ids(72, a)
            tag = f"c={best[0][0]*best[0][1]} ibw={best[1]} sw={best[2]}"
            sw.w13_case(sh, ids, 32, base, name=f"w13 N={n} a={a} BASE", check_torch=(a == 8))
            sw.w13_case(sh, ids, 32, best, name=f"w13 N={n} a={a} BEST {tag}", check_torch=(a == 8))
            sw.w13_case(sh, ids, 32, best, name=f"w13 N={n} a={a} BEST nnz=exact", nnz="exact", check_torch=False)
            sw.w13_case(sh, ids, 32, best, name=f"w13 N={n} a={a} BEST indices", indices=True, check_torch=(a == 8))
            if a == 32:
                sw.w13_case(sh, ids, 32, best, name=f"w13 N={n} a={a} BEST in0=L1", in0_l1=True, check_torch=False)
                sw.w13_case(sh, ids, 32, best, name=f"w13 N={n} a={a} BEST out=L1", out_l1=True, check_torch=False)
    for i in (256, 512, 1024, 2048):
        sh = shapes[("w2", 72, i)]
        base = BASE2(sw.b.grid, i)
        fb = (rect_for(64, sw.b.grid), i // TILE if i <= 1024 else 32, 1)
        best = best_cfg(sw, f"w2 I={i} a=32 c=", fb)
        for a in (8, 32, 72):
            ids = active_ids(72, a)
            tag = f"c={best[0][0]*best[0][1]} ibw={best[1]} sw={best[2]}"
            sw.w2_case(sh, ids, 32, base, name=f"w2 I={i} a={a} BASE", check_torch=(a == 8))
            sw.w2_case(sh, ids, 32, best, name=f"w2 I={i} a={a} BEST {tag}", check_torch=(a == 8))
            sw.w2_case(sh, ids, 32, best, name=f"w2 I={i} a={a} BEST nnz=exact", nnz="exact", check_torch=False)
            sw.w2_case(sh, ids, 32, best, name=f"w2 I={i} a={a} BEST indices", indices=True, check_torch=(a == 8))


def stage_tokens(sw: Sweep, shapes: dict):
    """(a) one call over M gathered tokens vs 32-token blocks (the per-block cost x M/32 is the blocked cost),
    (b) the indexed mode (only the active experts, compact output) and in0 in L1 at the best config."""
    sh13, sh2 = shapes[("w13", 72, 512)], shapes[("w2", 72, 256)]
    for a in (32, 72):
        ids = active_ids(72, a)
        for M in (32, 64, 128, 256):
            p13 = f"tokens w13 N=512 a={a} M={M} c="
            for xy, ibw, s in cfg_candidates(M, 512, H, sw.b.grid, ibws=(4, 16, 32, 64, 128), subs=(1,), top=1):
                sw.w13_case(sh13, ids, M, (xy, ibw, s), name=f"{p13}{xy[0]*xy[1]} ibw={ibw}", check_torch=(ibw == 4))
            best = best_cfg(sw, p13, ((4, 4), 16, 1))
            tag = f"c={best[0][0]*best[0][1]} ibw={best[1]}"
            sw.w13_case(
                sh13,
                ids,
                M,
                best,
                name=f"tokens w13 N=512 a={a} M={M} indices {tag}",
                indices=True,
                check_torch=(M == 32),
            )
            sw.w13_case(
                sh13, ids, M, best, name=f"tokens w13 N=512 a={a} M={M} in0=L1 {tag}", in0_l1=True, check_torch=False
            )
            p2 = f"tokens w2 I=256 a={a} M={M} c="
            for xy, ibw, s in cfg_candidates(M, H, 256, sw.b.grid, ibws=(4, 8), subs=(1, 2), top=1):
                sw.w2_case(
                    sh2,
                    ids,
                    M,
                    (xy, ibw, s),
                    name=f"{p2}{xy[0]*xy[1]} ibw={ibw} sw={s}",
                    check_torch=(ibw == 4 and s == 1),
                )
            best = best_cfg(sw, p2, ((8, 8), 8, 1))
            tag = f"c={best[0][0]*best[0][1]} ibw={best[1]} sw={best[2]}"
            sw.w2_case(
                sh2,
                ids,
                M,
                best,
                name=f"tokens w2 I=256 a={a} M={M} indices {tag}",
                indices=True,
                check_torch=(M == 32),
            )


PLACEMENTS = {  # name -> (E_local, N13 per chip, I per chip); the same 240 MB of expert weights per chip in all
    "hybrid EP4xTP8": (72, 512, 256),
    "hybrid g=2 (36 x 1024)": (36, 1024, 512),
    "hybrid g=4 (18 x 2048)": (18, 2048, 1024),
    "hybrid g=8 (9 x 4096)": (9, 4096, 2048),
    "EP8xTP4": (36, 1024, 512),
    "EP32": (9, 4096, 2048),  # Nt = 128 has no rectangle on the 12x10 grid: 64 cores x 2 tiles is the most
}
PLACE_TOKENS = (8, 32, 64, 256)


def stage_place(sw: Sweep, shapes: dict):
    """Per placement and token count: the active-expert count (Monte-Carlo, max over chips), a mini-sweep of
    the W13 + W2 calls at that (shape, active, M = all gathered tokens in one call) and the indexed mode at the
    mini-sweep's best config."""
    mc = {}
    for name, (E_local, n13, i2) in PLACEMENTS.items():
        for T in PLACE_TOKENS:
            if name.startswith("hybrid g="):
                g = 72 // E_local  # a group is active when any of its g experts is
                mean, mx = mc_active_grouped(T, g)
            else:
                mean, mx = mc_active(T, E_local)
            mc[(name, T)] = (mean, mx)
            logger.info(f"MC active experts {name} T={T}: mean {mean:.1f} max-over-chips {mx:.1f} of {E_local}")
    for name, (E_local, n13, i2) in PLACEMENTS.items():
        key13, key2 = ("w13", E_local, n13), ("w2", E_local, i2)
        if key13 not in shapes:
            shapes[key13] = ExpertShape(sw.dev, E_local, n13, None, seed=3)
        if key2 not in shapes:
            shapes[key2] = ExpertShape(sw.dev, E_local, None, i2, seed=4)
        sh13, sh2 = shapes[key13], shapes[key2]
        for T in PLACE_TOKENS:
            a = max(1, min(E_local, round(mc[(name, T)][1])))
            ids = active_ids(E_local, a, seed=T)
            M = max(32, T)
            extra = dict(placement=name, tokens=T, mc_mean=mc[(name, T)][0], mc_max=mc[(name, T)][1])
            p13 = f"place {name} T={T} w13 a={a} M={M} c="
            for xy, ibw, s in cfg_candidates(M, n13, H, sw.b.grid, ibws=(16, 32, 64, 128), subs=(1,), top=2):
                sw.w13_case(
                    sh13, ids, M, (xy, ibw, s), name=f"{p13}{xy[0]*xy[1]} ibw={ibw}", check_torch=False, extra=extra
                )
            best = best_cfg(sw, p13, None)
            if best is not None:
                sw.w13_case(
                    sh13,
                    ids,
                    M,
                    best,
                    name=f"place {name} T={T} w13 a={a} M={M} indices c={best[0][0]*best[0][1]} ibw={best[1]}",
                    indices=True,
                    check_torch=False,
                    extra=extra,
                )
            p2 = f"place {name} T={T} w2 a={a} M={M} c="
            for xy, ibw, s in cfg_candidates(M, H, i2, sw.b.grid, ibws=(8, 16, 32, 64), subs=(1,), top=2):
                sw.w2_case(
                    sh2, ids, M, (xy, ibw, s), name=f"{p2}{xy[0]*xy[1]} ibw={ibw}", check_torch=False, extra=extra
                )
            best = best_cfg(sw, p2, None)
            if best is not None:
                sw.w2_case(
                    sh2,
                    ids,
                    M,
                    best,
                    name=f"place {name} T={T} w2 a={a} M={M} indices c={best[0][0]*best[0][1]} ibw={best[1]}",
                    indices=True,
                    check_torch=False,
                    extra=extra,
                )
        if E_local != 72:
            sh13.free()
            sh2.free()
            del shapes[key13], shapes[key2]


def mc_active_grouped(T: int, g: int, trials: int = 200, seed: int = 0) -> tuple[float, float]:
    """Hybrid placement with the 72 column experts packed into groups of g: a group is streamed when any of
    its experts is active. Returns (mean, max over the 4 columns) of active groups."""
    gen = torch.Generator().manual_seed(seed)
    means, maxes = [], []
    for _ in range(trials):
        picks = torch.stack([torch.randperm(E_GLOBAL, generator=gen)[:TOPK] for _ in range(T)]).flatten()
        counts = []
        for lo in range(0, E_GLOBAL, 72):
            local = picks[(picks >= lo) & (picks < lo + 72)] - lo
            counts.append(len(torch.unique(local // g)))
        means.append(sum(counts) / 4)
        maxes.append(max(counts))
    return sum(means) / trials, sum(maxes) / trials


def stage_dense(sw: Sweep, shapes: dict):
    sh = shapes[("w13", 72, 512)]
    if sh.w13_t is None:
        sh.w13_t = ttnn.from_torch(
            sh.w13_host.transpose(-1, -2).contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=sw.dev
        )
    for M in (32, 256):
        Mt = M // TILE
        subs = [(4, 1)] if Mt == 1 else [(1, 4), (2, 2)]
        for cores, pcm in ((72, 16), (96, 12), (128, 9)):
            for ibw in (8, 16, 32):
                for sub in subs:
                    if pcm % sub[0]:
                        continue
                    sw.dense_t_case(
                        sh,
                        M,
                        cores,
                        pcm,
                        ibw,
                        sub,
                        name=f"dense w13T M={M} c={cores} pcM={pcm} ibw={ibw} sub={sub[0]}x{sub[1]}",
                    )
    sh2 = shapes[("w2", 72, 256)]
    for M in (32, 256):
        Mt = M // TILE
        subs = [(4, 1)] if Mt == 1 else [(1, 4), (2, 2)]
        for ibw in (4, 8):
            for sub in subs:
                sw.dense_bmm2_case(sh2, M, ibw, sub, name=f"dense w2T bmm M={M} c=72 ibw={ibw} sub={sub[0]}x{sub[1]}")


def stage_dense2(sw: Sweep, shapes: dict):
    """Dense MoE over ALL 72 local experts, no sparsity (gpt_oss ``decode_forward_dense`` style): W13 as one
    [4096, 72 x 512] matmul and W2 as one [72 x 256, 4096] matmul (the sum over experts is the K reduction), so
    every expert is read exactly once per call for any M, with no output fill, mask or expert-sum op."""
    sh13, sh2 = shapes[("w13", 72, 512)], shapes[("w2", 72, 256)]
    dev, cmp = sw.dev, sw.b.compute
    E, N13, I2 = 72, 512, 256
    w13d = ttnn.from_torch(
        sh13.w13_host[0].permute(1, 0, 2).reshape(1, 1, H, E * N13).contiguous(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
    )
    w2d = ttnn.from_torch(
        sh2.w2_host.reshape(1, 1, E * I2, H), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev
    )
    grid = sw.b.grid

    def cfg1d(cores, Mt, Nt, Kt, ibw):
        xy = rect_for(cores, grid)
        pcN = math.ceil(Nt / cores)
        if Kt % ibw or l1_dfb_bytes(Mt, ibw, pcN) > L1_DFB_BUDGET:
            return None
        sw_ = max(w for w in (1, 2, 4) if pcN % w == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=xy,
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=sw_,
            out_block_h=Mt,
            out_block_w=pcN,
            per_core_M=Mt,
            per_core_N=pcN,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def cfg2d(cols, Mt, Nt, Kt, ibw):  # Mt rows of cores (one M tile each), N over the columns
        pcN = math.ceil(Nt / cols)
        if Kt % ibw or Mt > grid[1] or cols > grid[0] or l1_dfb_bytes(1, ibw, pcN) > L1_DFB_BUDGET:
            return None
        sw_ = max(w for w in (1, 2, 4) if pcN % w == 0)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(cols, Mt),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=sw_,
            out_block_h=1,
            out_block_w=pcN,
            per_core_M=1,
            per_core_N=pcN,
            transpose_mcast=False,
            fused_activation=None,
        )

    for M in (32, 64, 128, 256):
        Mt = M // TILE
        x_host = torch.randn(1, 1, M, H, generator=torch.Generator().manual_seed(7)).bfloat16()
        h_host = torch.randn(1, 1, M, E * I2, generator=torch.Generator().manual_seed(8)).bfloat16()
        x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        hd = ttnn.from_torch(h_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        def chk13(out):
            y = ttnn.to_torch(out).float().reshape(M, E * N13)
            ids = list(range(0, E, 9))
            ref = torch.cat([x_host[0, 0].float() @ sh13.ref13(e) for e in ids], dim=-1)
            got = torch.cat([y[:, e * N13 : (e + 1) * N13] for e in ids], dim=-1)
            return {"pcc": pcc(ref, got), "maxdiff": float((ref - got).abs().max())}

        def chk2(out):
            y = ttnn.to_torch(out).float().reshape(M, H)
            ref = sum(h_host[0, 0, :, e * I2 : (e + 1) * I2].float() @ sh2.ref2(e) for e in range(E))
            return {"pcc": pcc(ref, y), "maxdiff": float((ref - y).abs().max())}

        variants13 = [("auto", None)]
        variants13 += [
            (f"1d c={c} ibw={ibw}", cfg1d(c, Mt, E * N13 // TILE, H // TILE, ibw))
            for c in (120, 96, 64)
            for ibw in (4, 8, 16)
        ]
        if Mt > 1:
            variants13 += [
                (f"2d {cols}x{Mt} ibw={ibw}", cfg2d(cols, Mt, E * N13 // TILE, H // TILE, ibw))
                for cols in (12, 8)
                for ibw in (4, 8)
            ]
        for tag, pc in variants13:
            if tag != "auto" and pc is None:
                continue
            meta = dict(
                name=f"dense2 w13 M={M} {tag}",
                kind="dense2_13",
                E=E,
                K=H,
                N=E * N13,
                M=M,
                active=E,
                cores=0,
                in0_block_w=0,
                out_subblock_w=0,
                bytes=sh13.bytes13(E),
                ops_per_call=1,
            )
            sw.b.run(
                meta,
                lambda pc=pc: ttnn.matmul(
                    x,
                    w13d,
                    program_config=pc,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=cmp,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                chk13,
            )
        variants2 = [("auto", None)]
        variants2 += [
            (f"1d c={c} ibw={ibw}", cfg1d(c, Mt, H // TILE, E * I2 // TILE, ibw))
            for c in (64, 32)
            for ibw in (8, 16, 32, 64)
        ]
        if Mt > 1:
            variants2 += [
                (f"2d {cols}x{Mt} ibw={ibw}", cfg2d(cols, Mt, H // TILE, E * I2 // TILE, ibw))
                for cols in (8, 4)
                for ibw in (8, 16, 32)
            ]
        for tag, pc in variants2:
            if tag != "auto" and pc is None:
                continue
            meta = dict(
                name=f"dense2 w2 M={M} {tag}",
                kind="dense2_2",
                E=E,
                K=E * I2,
                N=H,
                M=M,
                active=E,
                cores=0,
                in0_block_w=0,
                out_subblock_w=0,
                bytes=sh2.bytes2(E),
                ops_per_call=1,
            )
            sw.b.run(
                meta,
                lambda pc=pc: ttnn.matmul(
                    hd,
                    w2d,
                    program_config=pc,
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=cmp,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                chk2,
            )
        ttnn.deallocate(x)
        ttnn.deallocate(hd)
    ttnn.deallocate(w13d)
    ttnn.deallocate(w2d)


def stage_kernel(sw: Sweep, shapes: dict):
    """The sparse kernel changes (own build): batch groups (a grid of G x Nt cores streams the experts G-way,
    in0 reused across the experts when it fits), the accumulate-over-experts down mode ([1, G, M, H], no slab,
    no reduce) and zero_fill_output=False, at the hybrid per-chip shapes with 28 (real routing) / 72 active.
    EVIDENCE ONLY: those sparse_matmul kwargs (`zero_fill_output`, `in1_transposed`, `accumulate_batch`) existed
    only in the dropped local build (doc/mesh/expert_kernel_sweep.md section 8); the stage is skipped on the main
    build (`ttnn.sparse_matmul` has no such kwargs) and the custom op in custom_ops/expert_stream replaces it."""
    if "accumulate_batch" not in str(ttnn.sparse_matmul.__doc__):
        logger.warning("stage kernel skipped: this build's sparse_matmul has none of the section-8 kernel features")
        return
    sh13, sh2 = shapes[("w13", 72, 512)], shapes[("w2", 72, 256)]
    grid = sw.b.grid
    if sh13.w13_t is None:
        sh13.w13_t = ttnn.from_torch(
            sh13.w13_host.transpose(-1, -2).contiguous(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=sw.dev
        )
    for a in (28, 72):
        ids = active_ids(72, a, seed=5)
        # W13 N=512 with the weights stored transposed ([1, E, N, K]: a core's column = contiguous tiles)
        for G, xy in ((1, (4, 4)), (6, (12, 8))):
            if xy[0] > grid[0] or xy[1] > grid[1]:
                continue
            for ibw in (16, 64):
                sw.w13_case(
                    sh13,
                    ids,
                    32,
                    (xy, ibw, 1),
                    name=f"kernel w13T N=512 a={a} G={G} c={xy[0]*xy[1]} ibw={ibw}",
                    check_torch=True,
                    transposed=True,
                )
        # W13 N=512: Nt = 16 blocks; G groups = 16 G cores
        for G, xy in ((1, (4, 4)), (4, (8, 8)), (6, (12, 8))):
            if xy[0] > grid[0] or xy[1] > grid[1]:
                continue
            for ibw in (16, 32, 64):
                sw.w13_case(
                    sh13,
                    ids,
                    32,
                    (xy, ibw, 1),
                    name=f"kernel w13 N=512 a={a} G={G} c={xy[0]*xy[1]} ibw={ibw}",
                    check_torch=(ibw == 64),
                )
            sw.w13_case(
                sh13,
                ids,
                32,
                (xy, 16, 1),
                name=f"kernel w13 N=512 a={a} G={G} c={xy[0]*xy[1]} ibw=16 nofill",
                check_torch=False,
                zero_fill=False,
            )
        # W2 I=256 (K=256, Nt=128): 32 cores x 4 tiles per group; G = 1 / 2 / 3 groups; accumulate mode
        for G, xy in ((1, (8, 4)), (2, (8, 8)), (3, (12, 8))):
            if xy[0] > grid[0] or xy[1] > grid[1]:
                continue
            for ibw in (4, 8):
                cfg = (xy, ibw, 4)
                if G == 1:
                    sw.w2_case(
                        sh2, ids, 32, cfg, name=f"kernel w2 I=256 a={a} G=1 c=32 ibw={ibw} slab", check_torch=(ibw == 8)
                    )
                sw.w2_case(
                    sh2,
                    ids,
                    32,
                    cfg,
                    name=f"kernel w2 I=256 a={a} G={G} c={xy[0]*xy[1]} ibw={ibw} accumulate",
                    check_torch=(ibw == 8),
                    accumulate=True,
                )


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_expert_kernel_sweep(mesh_device):
    bench = Bench(mesh_device)
    sw = Sweep(bench)
    shapes: dict = {}
    need13 = {512, 1024, 2048, 4096} if any(s in STAGES for s in ("w13", "active")) else set()
    need2 = {256, 512, 1024, 2048} if any(s in STAGES for s in ("w2", "active")) else set()
    if any(s in STAGES for s in ("tokens", "place", "dense", "dense2", "kernel")):
        need13.add(512)
        need2.add(256)
    for n in sorted(need13):
        shapes[("w13", 72, n)] = ExpertShape(mesh_device, 72, n, None, seed=1)
    for i in sorted(need2):
        shapes[("w2", 72, i)] = ExpertShape(mesh_device, 72, None, i, seed=2)
    for stage in STAGES:
        logger.info(f"===== stage {stage} =====")
        {
            "w13": stage_w13,
            "w2": stage_w2,
            "active": stage_active,
            "tokens": stage_tokens,
            "place": stage_place,
            "dense": stage_dense,
            "dense2": stage_dense2,
            "kernel": stage_kernel,
        }[stage](sw, shapes)
    logger.info(f"{len(bench.cases)} cases; sidecar {CASES_JSON}")
    # wall-time summary (device times come from the profiler CSV via --report)
    rows = sorted((m["name"], m.get("wall_us", float("nan"))) for m in bench.cases.values())
    for name, w in rows:
        logger.info(f"WALL {w:9.1f} us  {name}")


# ----------------------------------------------------------------------------------------------- report
def report(csv: Path, cases: Path, out: Path | None, perf_report_tool: str | None):
    import pandas as pd

    df = pd.read_csv(csv, low_memory=False)
    meta = json.loads(cases.read_text())
    sp = df[df["OP TYPE"] == "signpost"]
    dev = df[df["OP TYPE"] == "tt_dnn_device"]
    dev = dev[dev["OP CODE"].astype(str).str.contains("Matmul")]
    rows = []
    for label, m in meta.items():
        if "error" in m:
            rows.append(dict(m, device_us=float("nan"), gbs=float("nan"), pct=float("nan"), cores_csv=""))
            continue
        t0 = sp[sp["OP CODE"] == label]["HOST START TS"].min()
        t1 = sp[sp["OP CODE"] == f"{label}_END"]["HOST START TS"].min()
        if pd.isna(t0) or pd.isna(t1):
            rows.append(dict(m, device_us=float("nan"), gbs=float("nan"), pct=float("nan"), cores_csv=""))
            continue
        win = dev[(dev["HOST START TS"] >= t0) & (dev["HOST START TS"] <= t1)]
        us = win["DEVICE KERNEL DURATION [ns]"].sum() / 1e3 / m["reps"]
        gbs = m["bytes"] / (us * 1e-6) / 1e9 if us > 0 else float("nan")
        cores = ",".join(str(int(c)) for c in win["CORE COUNT"].unique()) if len(win) else ""
        rows.append(dict(m, device_us=us, gbs=gbs, pct=100.0 * gbs / PEAK_GBS, cores_csv=cores, n_ops=len(win)))
    tab = pd.DataFrame(rows)
    cols = [
        "label",
        "name",
        "kind",
        "E",
        "K",
        "N",
        "M",
        "active",
        "cores",
        "in0_block_w",
        "out_subblock_w",
        "nnz",
        "indices",
        "in0_l1",
        "out_l1",
        "bytes",
        "device_us",
        "gbs",
        "pct",
        "wall_us",
        "pcc",
        "pcc_vs_base",
        "maxdiff",
        "error",
        "placement",
        "tokens",
        "mc_mean",
        "mc_max",
        "cores_csv",
        "n_ops",
    ]
    cols = [c for c in cols if c in tab.columns]
    tab = tab[cols]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 10000)
    pd.set_option("display.max_columns", 50)
    text = tab.to_string(index=False, float_format=lambda v: f"{v:.3f}")
    print(text)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(out.with_suffix(".csv"), index=False)
        out.write_text(text + "\n")
        print(f"written {out} and {out.with_suffix('.csv')}")
    if perf_report_tool:  # cross-check one window with tt-perf-report
        import subprocess

        ok = tab[tab["device_us"].notna()]
        if len(ok):
            lab = ok.iloc[0]["label"]
            res = subprocess.run(
                [
                    perf_report_tool,
                    str(csv),
                    "--start-signpost",
                    lab,
                    "--end-signpost",
                    f"{lab}_END",
                    "--no-advice",
                    "--no-color",
                    "--no-summary",
                ],
                capture_output=True,
                text=True,
            )
            print(
                f"tt-perf-report window {lab} (expect Device Time sum = {ok.iloc[0]['device_us'] * ok.iloc[0]['reps']:.1f} us):\n{res.stdout[-3000:]}"
            )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="report mode: join the Tracy ops CSV with the cases sidecar")
    ap.add_argument("--report", type=Path, required=True, help="ops_perf_results_*.csv of the tracy run")
    ap.add_argument("--cases", type=Path, default=CASES_JSON)
    ap.add_argument("--out", type=Path, default=None, help="table (.txt) + .csv output path")
    ap.add_argument("--perf-report", default=None, help="tt-perf-report binary for a one-window cross-check")
    a = ap.parse_args()
    report(a.report, a.cases, a.out, a.perf_report)
