# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.5 (OWNER 2026-11-27, CPU only, no device, no build): would wo_a's bank-local layout help wq_b and wo_b?

An ESTIMATE from measured numbers only:
- wo_a, measured (WA.2 / WA.3 op probes, OV-79 table; WA.4 test_woa_offon in the model): today vs the bank path on the
  model's 11 x 10 matmul grid, at 256 rows (8192 chunk) and 160 rows (5120 chunk).
- wq_b / wo_b today: results/PB.MM256.json (256 rows, the tuned configs, mesh confirm) and P.81 / mm_configs (160 rows).
- Shapes from the checkpoint's inference config (woa_roofline.load), dtypes as the model holds them: x and w bfloat8_b,
  output bfloat16 (attention.py _q_proj / _out_wo: wq_b dtype bfloat16, wo_b act_dtype() = bfloat16).

Two estimates of the bank path's us a call, each calibrated on wo_a (the only bank path measured):
  A "gap fraction": the bank path closes the fraction f of (today - realistic roofline) that it closed for wo_a
    (realistic = max(all bytes at 410 GB/s, busiest core's HiFi2 compute / 0.8), woa_roofline's model).
  B "analog": max(all bytes at the bank path's measured byte rate on wo_a, busiest core's tile products at the bank
    path's measured us a tile product on wo_a), the bank cores <= MAX_WORKER_CORES (13 a bank on 8 banks).
Best estimate = the mean of A and B; range = [0, today - realistic].

    python3 -m models.demos.deepseek_v41_flash_d_p.tests.woa.woa_estimate
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from models.demos.deepseek_v41_flash_d_p.tests.woa import woa_roofline as rl

BF16_TILE = 2048
CALLS = 20  # a chunk, layers 0-20 (DECODER_SKIP=decode): PB.MM256_ladder.md inventory, wq_b / wo_b / wo_a 20 each

# wo_a, measured us a call (slowest chip): OV-79 table of results/WA.3_offon.md (tests/woa/test_woa_layout_probe.py,
# the model's 11 x 10 grid, one kernel group: what the model runs)
WOA_TODAY = {256: 151.1, 160: 131.7}
WOA_BANK = {256: 130.3, 160: 114.2}
WOA_BANK_CORES = 80  # band 10: 10 cores a bank, busiest core 4 column tiles (32 = 8 x 3 + 2 x 4)

# today's wq_b / wo_b us a call
TODAY = {
    ("wq_b", 256): ("mm256_confirm_us_attn_wq_b_256x1280x32768_on_tagentic_cc", None),  # PB.MM256.json
    ("wo_b", 256): ("mm256_confirm_us_attn_wo_b_256x8192x5120_on_tagentic_cc", None),
    ("wq_b", 160): (None, 182.0),  # P.81 ROOFLINE (1D ob 5x2 bw20, shipped), DS41-306
    ("wo_b", 160): (None, 142.3),  # P.81 probe, shipped (DS41-306; mm_configs GALAXY 0.1364 in the sweep)
}
# P.81 (160 rows, op probe): the stock DRAM-sharded matmul (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig,
# use_multicore_pipeline, 8 / 12 / 13 cores a bank): matmul us, the output copy L1 -> DRAM us (wq_b only)
STOCK_DS = {"wq_b": (153.0, 174.0, 37.0), "wo_b": (280.0, 650.0, 0.0)}


@dataclass(frozen=True)
class Proj:
    name: str
    K: int
    N: int

    @property
    def KT(self):
        return self.K // 32

    @property
    def NT(self):
        return self.N // 32

    def bytes(self, rows: int) -> dict:
        mt = rows // 32
        t8 = rl.bfp8_tile_bytes()
        return {"w": self.KT * self.NT * t8, "x": mt * self.KT * t8, "y": mt * self.NT * BF16_TILE}

    def busiest_products(self, rows: int, cores_per_bank: int) -> int:
        """Tile products of the busiest core: a bank's NT / 8 columns over cores_per_bank cores."""
        cols = math.ceil(self.NT / rl.DRAM_BANKS / cores_per_bank)
        return cols * (rows // 32) * self.KT


def projections() -> list[Proj]:
    w, ctx = rl.load()
    cfg = ctx["cfg"]
    return [
        Proj("wq_b", cfg.q_lora_rank, cfg.n_heads * cfg.head_dim),  # [256, 1280] @ [1280, 32768] (every head, SP32)
        Proj("wo_b", cfg.o_groups * cfg.o_lora_rank, cfg.dim),  # [256, 8192] @ [8192, 5120]
    ]


def today_us(name: str, rows: int) -> float:
    key, val = TODAY[(name, rows)]
    if key is None:
        return val
    return float(json.loads((rl.RESULTS / "PB.MM256.json").read_text())["metrics"][key]["value"])


def _us_per_product() -> float:
    return rl.TILE_CYCLES["HiFi2"] / rl.CLOCK_HZ * 1e6


def woa_calibration(rows: int) -> dict:
    """wo_a's measured bank path against its roofline: f (gap fraction closed), the byte rate and us a tile product."""
    w, _ = rl.load()
    b = {"w": w.w_bytes(), "x": w.x_bytes(rows), "y": w.y_bytes(rows)}
    total = sum(b.values())
    realistic = max(total / (rl.DRAM_PEAK * rl.DRAM_PLAN_EFF) * 1e6, _woa_products(w, rows) * _us_per_product() / 0.8)
    f = (WOA_TODAY[rows] - WOA_BANK[rows]) / (WOA_TODAY[rows] - realistic)
    return {
        "realistic_us": realistic,
        "f": f,
        "rate_gbs": total / WOA_BANK[rows] / 1e3,  # bytes / us -> GB/s
        "us_per_product": WOA_BANK[rows] / _woa_products(w, rows),
        "bytes_mb": total / 1e6,
    }


def _woa_products(w, rows: int) -> int:
    cpb = WOA_BANK_CORES // rl.DRAM_BANKS
    return math.ceil(w.NT / cpb) * (rows // 32) * w.KT


def estimate(p: Proj, rows: int, cores_per_bank: int | None = None) -> dict:
    cap = rl.load()[0].cap
    cpb = cores_per_bank or cap // rl.DRAM_BANKS  # 13 a bank: 104 cores <= 110 (rule 18)
    b = p.bytes(rows)
    total = sum(b.values())
    prod = p.busiest_products(rows, cpb)
    compute_peak = prod * _us_per_product()
    plan_bps = rl.DRAM_PEAK * rl.DRAM_PLAN_EFF
    realistic = max(total / plan_bps * 1e6, compute_peak / 0.8)
    t = today_us(p.name, rows)
    cal = woa_calibration(rows)
    gap = max(0.0, t - realistic)
    a_us = t - cal["f"] * gap
    b_us = max(total / (cal["rate_gbs"] * 1e3), prod * cal["us_per_product"])
    best_us = (a_us + b_us) / 2
    # wo_a's clusters each need only their own group's x; a plain matmul's 8 clusters each need ALL of x: read by
    # each cluster (independent clusters, as wo_a) = 7 more copies of x, or one grid-wide multicast (lockstep)
    b_x8_us = max((total + 7 * b["x"]) / (cal["rate_gbs"] * 1e3), prod * cal["us_per_product"])
    return {
        "name": p.name,
        "rows": rows,
        "w_mb": b["w"] / 1e6,
        "x_mb": b["x"] / 1e6,
        "y_mb": b["y"] / 1e6,
        "w_floor512_us": b["w"] / rl.DRAM_PEAK * 1e6,
        "w_floor410_us": b["w"] / plan_bps * 1e6,
        "all410_us": total / plan_bps * 1e6,
        "compute_peak_us": compute_peak,
        "cores": cpb * rl.DRAM_BANKS,
        "busiest_products": prod,
        "realistic_us": realistic,
        "today_us": t,
        "gap_us": gap,
        "a_us": a_us,
        "b_us": b_us,
        "b_x8_us": b_x8_us,
        "best_us": best_us,
        "gain_us": max(0.0, t - best_us),
        "gain_ms_chunk": max(0.0, t - best_us) * CALLS / 1e3,
        "gain_hi_ms_chunk": gap * CALLS / 1e3,
        "today_ms_chunk": t * CALLS / 1e3,
        "f": cal["f"],
    }


def run() -> dict:
    out = {}
    for p in projections():
        for rows in (256, 160):
            out[(p.name, rows)] = estimate(p, rows)
    return out


def table(res: dict) -> str:
    rows = [
        "| proj, rows a chip | w / x / y MB | w at 512 / 410 GB/s us | all bytes 410 us | compute peak us (cores) "
        "| realistic us | today us | est A / B us | best est us | gain ms a chunk best (range) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res.values():
        rows.append(
            f"| {r['name']}, {r['rows']} | {r['w_mb']:.2f} / {r['x_mb']:.2f} / {r['y_mb']:.2f} | "
            f"{r['w_floor512_us']:.1f} / {r['w_floor410_us']:.1f} | {r['all410_us']:.1f} | "
            f"{r['compute_peak_us']:.1f} ({r['cores']}) | {r['realistic_us']:.1f} | {r['today_us']:.1f} | "
            f"{r['a_us']:.1f} / {r['b_us']:.1f} | {r['best_us']:.1f} | {r['gain_ms_chunk']:.2f} (0-{r['gain_hi_ms_chunk']:.2f}) |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    res = run()
    for rows in (256, 160):
        c = woa_calibration(rows)
        print(
            f"wo_a {rows} rows: realistic {c['realistic_us']:.1f} us, f {c['f']:.3f}, bank rate {c['rate_gbs']:.0f} GB/s, "
            f"{c['us_per_product'] * 1e3:.2f} ns a tile product, {c['bytes_mb']:.2f} MB"
        )
    print(table(res))
