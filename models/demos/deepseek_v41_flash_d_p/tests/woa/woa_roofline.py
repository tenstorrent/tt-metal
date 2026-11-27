# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.1 (CPU only): wo_a's DRAM floor, its roofline under a bank-local weight layout, and that layout itself.

Shapes come from the checkpoint the run's spec names (inference config: n_heads, head_dim, o_groups, o_lora_rank;
the safetensors header: wo_a's dtype), the mesh from the spec's run box, the core cap from tt/settings.py
(MAX_WORKER_CORES). Measured times are read from this ledger's results where a metric exists (PB.MM256.json) and cited
where they exist only in prose (P.81: grouped_matmul CHANGELOG / BREADCRUMBS.0004 / DS41-320).

    python3 -m models.demos.deepseek_v41_flash_d_p.tests.woa.woa_roofline     # prints the tables
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[2]
LEDGER = MODEL_DIR / "bringup" / "ledgers" / "galaxy_8x4-full"
RESULTS = LEDGER / "results"

# ---- hardware (Blackhole, a Galaxy chip; X.7_roofline.md "Hardware")
CLOCK_HZ = 1.35e9  # AICLK (tech_reports/GEMM_FLOPS: BH 1.35 GHz; known_issues_detail DS41-249 run: AICLK 1350 MHz)
TILE_CYCLES = {"LoFi": 16, "HiFi2": 32}  # a 32x32x32 tile product (8x16 @ 16x16 a cycle; GEMM_FLOPS.md)
DRAM_BANKS = 8  # GDDR6 channels = DRAM banks of the allocator (flat plan.cpp: dram_grid_size().x)
DRAM_BANK_BPS = 64e9  # a bank (CCL_bandwidth.md, measured 64 GB/s a bank)
DRAM_PEAK = DRAM_BANKS * DRAM_BANK_BPS  # 512 GB/s; DS41-249: bank-local expert reads fit this rate (rmse lowest)
DRAM_PLAN_EFF = 0.80  # 410 GB/s, X.7's planning fraction (P.81's floors)
MM_REAL_EFF = 0.80  # "realistic" compute: DS41-299, the flat expert cores at 84 % of LoFi peak
GRID = (12, 10)  # the Galaxy chip's worker grid for the model (owner 2026-10-23: 12 x 10)

# ---- measured (device us a call, slowest chip)
TODAY_160_US = 133.0  # P.81 attempt 2 op probe (8x4, rt profiler, slowest chip, 20 iters), bw 4 depth 4 = shipped
TODAY_160_SRC = "P.81 op probe (grouped_matmul CHANGELOG 'block_w 4 0.133'; BREADCRUMBS.0004 P.81 attempt 2; DS41-320)"
TODAY_256_METRIC = "mm256_confirm_us_attn_wo_a_gmm_256x4096x1024_on_tagentic_cc"
CHUNK_DEPLOY, CHUNK_BATCH = 5120, 8192


def bfp8_tile_bytes() -> int:
    """bfloat8_b tile: 1024 one-byte mantissas + one shared exponent byte a 16 values (= ttnn.tile_size: 1088)."""
    return 32 * 32 + 32 * 32 // 16


@dataclass(frozen=True)
class WoA:
    G: int  # groups a chip (ATTN_SP32_PROJ: every group, replicated)
    K: int  # hpg * head_dim
    N: int  # o_lora_rank
    w_dtype: str  # checkpoint dtype
    mesh: tuple
    cap: int  # MAX_WORKER_CORES

    @property
    def KT(self):
        return self.K // 32

    @property
    def NT(self):
        return self.N // 32

    def rows(self, chunk: int) -> int:
        return chunk // (self.mesh[0] * self.mesh[1])

    def w_bytes(self) -> int:
        return self.G * self.KT * self.NT * bfp8_tile_bytes()

    def x_bytes(self, rows: int) -> int:
        return self.G * (rows // 32) * self.KT * bfp8_tile_bytes()

    def y_bytes(self, rows: int) -> int:  # wo_a writes bfloat8_b (GEMM_INPUTS / BF16_ACTIVATIONS: wo_b's input)
        return self.G * (rows // 32) * self.NT * bfp8_tile_bytes()


def load() -> tuple[WoA, dict]:
    from models.demos.common.bringup.core.spec import Spec
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
    from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    spec = Spec.load(os.environ.get("BRINGUP_SPEC") or LEDGER / "spec.yaml")
    path = hf_path(spec)
    cfg = Config.from_dir(path)
    dt, shape, _, _ = Checkpoint(path).info("layers.0.attn.wo_a.weight")
    hpg = cfg.n_heads // cfg.o_groups
    assert list(shape) == [cfg.o_groups * cfg.o_lora_rank, hpg * cfg.head_dim], shape
    box = spec.data["run"]["box"]
    mesh = tuple(spec.data["boxes"][box]["mesh"])
    assert settings.get("ATTN_SP32_PROJ") == "on"  # every group on every chip (replicated wo_a)
    w = WoA(cfg.o_groups, hpg * cfg.head_dim, cfg.o_lora_rank, str(dt), mesh, int(settings.get("MAX_WORKER_CORES")))
    return w, {"spec": spec, "cfg": cfg, "deploy_chunk": spec.data["deploy"]["chunk"]}


def measured() -> dict:
    """Today's device us a call at 160 / 256 rows and wo_a calls a chunk (layers 0-20, DECODER_SKIP=decode)."""
    mm = json.loads((RESULTS / "PB.MM256.json").read_text())["metrics"]
    t256 = float(mm[TODAY_256_METRIC]["value"])
    lad = (RESULTS / "PB.MM256_ladder.md").read_text()
    m = re.search(r"\| bringup\.grouped_matmul attn\.wo_a \|[^|]*\|[^|]*\| (\d+) \| ([\d.]+) \|", lad)
    assert m, "PB.MM256_ladder.md: no wo_a inventory row"
    calls, off_us_chunk = int(m.group(1)), float(m.group(2))
    return {
        160: TODAY_160_US,
        256: t256,
        "calls": calls,
        "off256_inmodel_us": off_us_chunk / calls,  # bw 4 depth 4, the in-model op profile (MM_TUNED256 off)
        "src160": TODAY_160_SRC,
        "src256": f"results/PB.MM256.json {TODAY_256_METRIC} (gmm bw 2 depth 2, mesh confirm, every chip)",
    }


# ---- the bank-local layout (design section 3) ---------------------------------------------------------------------


def column_split(nt: int, n: int) -> list[int]:
    """As grouped_matmul.column_split: column tiles per core, as even as possible, the wider first."""
    base, rem = divmod(nt, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


@dataclass(frozen=True)
class CoreSlot:
    bank: int
    core: int  # index in the bank's cluster
    g: int
    n0: int
    pn: int
    tile0: int  # first tile of its region in the bank (K-major: tile (k, j) at tile0 + k * pn + j)


def bank_plan(G: int, NT: int, KT: int, banks: int = DRAM_BANKS, cores_per_bank: int = 12) -> list[CoreSlot]:
    """The (g, n) columns of every group in banks: the G * NT columns in order, an equal run a bank (G == banks: one
    group a bank); a bank's columns of one group split over its cores (column_split); each core's region in its bank
    is its pn columns K-major, the cores' regions back to back. No zero padding anywhere (OV-3)."""
    units = G * NT
    assert units % banks == 0, (G, NT, banks)
    per_bank = units // banks
    out = []
    for b in range(banks):
        cols = [(u // NT, u % NT) for u in range(b * per_bank, (b + 1) * per_bank)]
        by_g: dict[int, list[int]] = {}
        for g, n in cols:
            by_g.setdefault(g, []).append(n)
        # cores shared out over the bank's groups in proportion to their columns (one group a bank on the 8x4)
        ci, t0 = 0, 0
        groups = sorted(by_g)
        for gi, g in enumerate(groups):
            ns = by_g[g]
            nc = cores_per_bank - ci if gi == len(groups) - 1 else max(1, round(cores_per_bank * len(ns) / per_bank))
            n0 = ns[0]
            for pn in column_split(len(ns), nc):
                if pn == 0:
                    continue
                out.append(CoreSlot(b, ci, g, n0, pn, t0))
                ci, n0, t0 = ci + 1, n0 + pn, t0 + KT * pn
    return out


def bank_regions(w: torch.Tensor, plan: list[CoreSlot], banks: int = DRAM_BANKS) -> list[torch.Tensor]:
    """w [G, K, N] (x @ w orientation) -> one [tiles, 32, 32] region a bank in the plan's order (whole tiles moved,
    each in its own orientation: a bfp8 tile's bytes, exponents included, are the same wherever it lies)."""
    G, K, N = w.shape
    KT, NT = K // 32, N // 32
    t = w.reshape(G, KT, 32, NT, 32).permute(0, 1, 3, 2, 4)  # [G, KT, NT, 32, 32]
    per = [[] for _ in range(banks)]
    for s in plan:
        per[s.bank].append(t[s.g, :, s.n0 : s.n0 + s.pn].reshape(-1, 32, 32))
    return [torch.cat(p) for p in per]


def grouped_mm_from_banks(x: torch.Tensor, regions, plan, KT: int, NT: int, bw: int) -> torch.Tensor:
    """y [G, M, N] = x [G, M, K] @ w, every core reading only its bank: per K block kb, ONE contiguous read of bw * pn
    tiles at tile0 + kb * bw * pn (the reader's single noc_async_read), K blocks in today's order."""
    G, M, _ = x.shape
    y = torch.zeros(G, M, NT * 32, dtype=x.dtype)
    for s in plan:
        reg = regions[s.bank]
        acc = torch.zeros(M, s.pn * 32, dtype=x.dtype)
        for kb in range(KT // bw):
            a = s.tile0 + kb * bw * s.pn
            blk = reg[a : a + bw * s.pn].reshape(bw, s.pn, 32, 32).permute(0, 2, 1, 3).reshape(bw * 32, s.pn * 32)
            acc += x[s.g, :, kb * bw * 32 : (kb + 1) * bw * 32] @ blk
        y[s.g, :, s.n0 * 32 : (s.n0 + s.pn) * 32] = acc
    return y


# ---- floors and bounds (design sections 1 and 5) ------------------------------------------------------------------


def analyse(w: WoA, rows: int, today_us: float, calls: int, cores_per_bank: int = 12) -> dict:
    plan = bank_plan(w.G, w.NT, w.KT, DRAM_BANKS, cores_per_bank)
    units_max = max(s.pn for s in plan)
    MT = rows // 32
    wb, xb, yb = w.w_bytes(), w.x_bytes(rows), w.y_bytes(rows)
    floor_w = wb / DRAM_PEAK * 1e6
    floor_w410 = wb / (DRAM_PEAK * DRAM_PLAN_EFF) * 1e6
    dram_all = (wb + xb + yb) / DRAM_PEAK * 1e6
    dram_all410 = (wb + xb + yb) / (DRAM_PEAK * DRAM_PLAN_EFF) * 1e6
    comp = units_max * MT * w.KT * TILE_CYCLES["HiFi2"] / CLOCK_HZ * 1e6  # the busiest core at the HiFi2 peak
    best = max(dram_all, comp)
    real = max(dram_all410, comp / MM_REAL_EFF)
    flops = 2 * w.G * rows * w.K * w.N
    return {
        "rows": rows,
        "cores": len(plan),
        "units_max": units_max,
        "w_MB": wb / 1e6,
        "x_MB": xb / 1e6,
        "y_MB": yb / 1e6,
        "floor_w_us": floor_w,
        "floor_w410_us": floor_w410,
        "dram_all_us": dram_all,
        "dram_all410_us": dram_all410,
        "compute_us": comp,
        "best_us": best,
        "realistic_us": real,
        "today_us": today_us,
        "gap_w": today_us / floor_w,
        "today_GBps_all": (wb + xb + yb) / today_us / 1e3,
        "today_TFps_core": flops / today_us / 1e6 / 88,  # today's 88 cores (8 rows x 11)
        "upper_ms": calls * max(0.0, today_us - best) / 1e3,
        "realistic_ms": calls * max(0.0, today_us - real) / 1e3,
    }


def table(res: dict) -> str:
    rows = [
        ("rows a chip", "rows", "{:d}"),
        ("weights MB / x MB / y MB", None, None),
        ("floor: weights at 512 GB/s (us)", "floor_w_us", "{:.1f}"),
        ("weights at 410 GB/s (us)", "floor_w410_us", "{:.1f}"),
        ("all bytes at 512 GB/s (us)", "dram_all_us", "{:.1f}"),
        ("all bytes at 410 GB/s (us)", "dram_all410_us", "{:.1f}"),
        ("compute, busiest core at HiFi2 peak (us)", "compute_us", "{:.1f}"),
        ("best case = max(all bytes 512, compute) (us)", "best_us", "{:.1f}"),
        ("realistic = max(all bytes 410, compute / 0.8) (us)", "realistic_us", "{:.1f}"),
        ("today (us)", "today_us", "{:.1f}"),
        ("today / weights floor", "gap_w", "{:.2f}x"),
        ("upper bound ms a chunk (x calls)", "upper_ms", "{:.2f}"),
        ("realistic ms a chunk", "realistic_ms", "{:.2f}"),
    ]
    cols = sorted(res)
    out = ["| | " + " | ".join(f"{res[c]['rows']} rows" for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for label, key, fmt in rows[1:]:
        if key is None:
            vals = [f"{res[c]['w_MB']:.2f} / {res[c]['x_MB']:.2f} / {res[c]['y_MB']:.2f}" for c in cols]
        else:
            vals = [fmt.format(res[c][key]) for c in cols]
        out.append(f"| {label} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def run() -> tuple[WoA, dict, dict]:
    w, ctx = load()
    meas = measured()
    res = {c: analyse(w, w.rows(c), meas[w.rows(c)], meas["calls"]) for c in (ctx["deploy_chunk"], CHUNK_BATCH)}
    return w, meas, res


if __name__ == "__main__":
    w, meas, res = run()
    print(w, meas)
    print(table(res))
    for c, r in res.items():
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print("min cores at 3 columns a core:", math.ceil(w.G * w.NT / 3))
