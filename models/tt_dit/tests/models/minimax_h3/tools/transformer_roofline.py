# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Roofline of the MiniMax-H3 transformer block on the Wormhole 4x8 Galaxy, 15 s / 768P / 16:9.

Two views of one block, per device. (1) The matmul-class ops from the shared registry `minimax_h3_ops.py`
(to_qkv, to_out, ff1 are AGMMs, ff2 is a plain matmul + reduce-scatter), each given compute / DRAM / fabric
bounds from its shape; `--ops agmm` (default) selects the three AGMMs, `--ops all` or a comma list selects any
subset. (2) Whole-block mode: every op in the Tracy per-op breakdown, rooflined by class from the profile CSV.
50 blocks (+2 token-refiner blocks at M=64) per forward. Not a test; pytest leaves it alone. Host-only: numpy +
matplotlib, no ttnn/torch.

    python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --dump
    python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --figs all --measured-csv sweep_results_mm.csv
    python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --ops all --dump
    python models/tt_dit/tests/models/minimax_h3/tools/transformer_roofline.py --selftest

AGMM = `ttnn.experimental.all_gather_minimal_matmul_async`, the fused TP all-gather + matmul every
column-parallel linear in the H3 DiT runs on (`models/tt_dit/layers/linear.py`). Ring-matmul model (ported
unchanged from `origin/cglagovich/agmm_analysis:agmm/roofline_lib.py`, the source of the "AGMM Regime & Block
Tuning" and "AGMM shape database" artifacts), per device, bf16, ring R = TP, L links per direction,
K_gathered = K_local * R:

    FLOPs          = 2 * M * K_gathered * N
    bytes_dram     = 2 * (M * K_gathered + K_gathered * N)     gathered in0 + resident in1
    bytes_per_link = (R - 1) * (2 * M * K_local) / (2 * L)     bidirectional ring split over L links
    peak_flops     = cores * (4096 / fidelity_cycles) * clock  4096 FLOP/cycle/core at LoFi, HiFi2 = 2048
    ideal          = max(FLOPs/peak, bytes_dram/DRAM_BW, bytes_per_link/LINK_BW), limiter = argmax
    N*             = peak / (2 * LINK_BW) * (R - 1) / (R * L)  compute<->fabric crossover, M-independent

A plain matmul + reduce-scatter (ff2) runs on the full grid and moves (R - 1) * (2 * M * N / R) / (2 * L) per link.

Every constant in those artifacts is Blackhole Galaxy (12x9 = 108-core ring-matmul grid at 1.35 GHz =
298.6 TFLOP/s HiFi2, 512 GB/s DRAM, 2 links x 25 GB/s). The Wormhole Galaxy the model ships on
(`MESH_4X8_RING_WH` = `4x8nl4`, TP=4 / SP=8, Ring, 4 links) is a different part on every axis; the
`WH` constants below are each traced to the repo file that defines them, and the `BH` constants are
kept verbatim, labelled as inherited, for the side-by-side. Note the two fabrics coincide on aggregate
ingress (4 x 12.5 = 2 x 25 = 50 GB/s per direction), so the fabric bars are identical across arches.

Whole-block mode (default when the Tracy profile exists; `--no-block` restores the selected-ops-only output): reads
the fsdp1 15 s block profile the perf doc's per-op table came from (`--profile-csv`), merges the warm iteration
across the 32 devices like `project_block_perf.py`, and gives every op a roofline by class -- these are
judgement calls built on the models already in the repo (the ring-matmul analysis, `OpPerformanceModelGeneral`,
`roofline_utils.py`, `estimate_fabric_transfer_cycles`) and are printed on every block figure:
  compute-bound (SDPA, matmuls): 2*FLOPs / (cores_of_the_op * 2048 FLOP/cycle * 1.0 GHz at HiFi2)
  DRAM-bound (embeddings, RMSNorm, tilize/untilize, concat): bytes in + out of DRAM tensors / 288 GB/s
  fabric-bound (all-gather, reduce-scatter, broadcast): (R-1) * shard / (2 * links) / 12.5 GB/s per link
  ideal = max of the terms that apply; ops under ~1% of the block are grouped as "other" (measured only).

Per-hop fabric latency (~0.7 us on Wormhole 1D, `ccl_common.cpp`) is ~2 us over the 3-hop ring against
~1 ms of transfer at this M and is not modelled. Colours: resources use the dataviz reference
categorical slots 1-3 (compute blue, DRAM orange, fabric aqua), pre-validated for CVD separation;
ops are told apart by marker shape and direct labels, never by colour.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass, replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimax_h3_ops import (  # noqa: E402
    AGMM_OPS,
    M_15S_768P_16_9,
    MEASURED_US_WH_15S,
    OPS_BY_NAME,
    SWEEP_USE_CASE_TO_OP,
    OpSpec,
    select_ops,
)

Op = OpSpec  # the roofline's op record is the registry's

FIDELITY_CYCLES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
LOFI_FLOP_PER_CYCLE_PER_CORE = 4096  # 8x16 x 16x16 per cycle, 2*8*16*16 -- matmul_device_operation.cpp:2791
BF16_BYTES = 2


@dataclass(frozen=True)
class Arch:
    name: str
    short: str
    clock_hz: float
    ring_matmul_cores: int  # worker grid of the ring all-gather matmul (one row/column reserved for the CCL muxes)
    ring_matmul_grid: tuple[int, int]
    full_cores: int  # full compute grid, what a plain matmul (ff2) runs on
    dram_bw: float  # bytes/s
    link_bw: float  # bytes/s per unidirectional ethernet link
    num_links: int  # links per direction the model's mesh config uses
    ring_size: int  # TP factor
    l1_bytes: int
    source: str

    def peak_flops(self, fidelity: str = "HiFi2", cores: int | None = None) -> float:
        cores = self.ring_matmul_cores if cores is None else cores
        return cores * (LOFI_FLOP_PER_CYCLE_PER_CORE / FIDELITY_CYCLES[fidelity]) * self.clock_hz

    def n_star(self, fidelity: str = "HiFi2", num_links: int | None = None) -> float:
        """Output width above which a ring all-gather matmul (AGMM) is compute-bound rather than fabric-bound (any M)."""
        L = self.num_links if num_links is None else num_links
        R = self.ring_size
        return self.peak_flops(fidelity) / (2 * self.link_bw) * (R - 1) / (R * L)


# Wormhole DRAM: the repo carries several numbers for the 12 x 1 GiB GDDR6 part. "spec" is the
# 12 Gbps datasheet figure (tech_reports/Saturating_DRAM_bandwidth:64-66), "measured" the u-benchmark
# from the same report, "perf_model" what ttnn's OpPerformanceModel uses (ttnn/core/operation.cpp:34).
WH_DRAM_BW = {"spec": 288e9, "measured": 267e9, "perf_model": 258e9}

WH = Arch(
    name="Wormhole Galaxy 4x8 (4x8nl4: TP=4 / SP=8, Ring, 4 links)",
    short="WH",
    clock_hz=1.0e9,  # tests/nightly/sdpa_perf_utils.py:31
    ring_matmul_cores=64,  # 8x8: agmm_worker_grid reserves the bottom row of the 8x9 grid (utils/matmul.py:407)
    ring_matmul_grid=(8, 8),
    full_cores=72,  # 8x9 -- sdpa_perf_utils.py:70 WH_GALAXY_GRID
    dram_bw=WH_DRAM_BW["spec"],
    link_bw=12.5e9,  # 100 Gbps per link -- ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:2130
    num_links=4,  # MESH_4X8_RING_WH -- tests/models/minimax_h3/common.py:134
    ring_size=4,
    l1_bytes=1464 * 1024,  # tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml:132
    source="sdpa_perf_utils.py, utils/matmul.py:407, ccl_common.cpp:2130, common.py:134, wormhole_b0_80_arch.yaml",
)

BH = Arch(
    name="Blackhole Galaxy 4x8 (constants inherited from cglagovich/agmm_analysis: 2 links)",
    short="BH",
    clock_hz=1.35e9,
    ring_matmul_cores=108,  # 12x9
    ring_matmul_grid=(12, 9),
    full_cores=120,  # 12x10 -- sdpa_perf_utils.py:69 GALAXY_GRID
    dram_bw=512e9,
    link_bw=25e9,  # 400 Gbps nominal, "currently limited to half BW" -- ccl_common.cpp:2132
    num_links=2,  # BH_GALAXY channels count 2 -- tools/scaleout/generate_mgd/generate_mgd.cpp
    ring_size=4,
    l1_bytes=1536 * 1024,  # blackhole_140_arch.yaml:109 (the artifacts budget 1400 KB of it for CBs)
    source="origin/cglagovich/agmm_analysis:agmm/roofline_lib.py (298.6 TFLOP/s, 512 GB/s, 25 GB/s/link)",
)


# The ops themselves (shapes, fusion, blocking, perf-doc baselines, colours) live in `minimax_h3_ops.py`.
# 15 s / 768P / 16:9: 1344x768, 362 frames -> 107 latent frames x 24x42 patches = 107856 video rows
# + 603 audio latents x 2 channels + 39 text tokens = 109101, padded to SP*TILE*... = 109312
# (`packing.padded_sequence_length`), 13664 rows per device at SP=8 (MiniMaxH3_wormhole_perf.md:129).
M_REFINER = 64  # the 2 token-refiner blocks run the same (K, N) shapes over the 39-token text stream
BLOCKS_PER_FORWARD = 50

RESOURCES = ("compute", "dram", "fabric")
RESOURCE_LABEL = {"compute": "compute", "dram": "DRAM", "fabric": "fabric"}
RESOURCE_COLOR = {"compute": "#2a78d6", "dram": "#eb6834", "fabric": "#1baf7a"}
INK, INK_2, INK_MUTED, GRID = "#1a1a19", "#52514e", "#8a8983", "#e6e5e0"


@dataclass(frozen=True)
class Roofline:
    op: Op
    arch: Arch
    M: int
    fidelity: str
    num_links: int
    dram_bw: float
    peak_flops: float
    flops: float
    bytes_dram: float
    bytes_per_link: float
    t_compute: float
    t_dram: float
    t_fabric: float
    measured: float | None  # seconds

    @property
    def times(self) -> dict[str, float]:
        return {"compute": self.t_compute, "dram": self.t_dram, "fabric": self.t_fabric}

    @property
    def limiter(self) -> str:
        return max(self.times, key=self.times.get)

    @property
    def ideal(self) -> float:
        return self.times[self.limiter]

    @property
    def ai_dram(self) -> float:
        return self.flops / self.bytes_dram

    @property
    def ai_fabric(self) -> float:
        return self.flops / self.bytes_per_link

    @property
    def headroom(self) -> float | None:
        return None if self.measured is None else self.measured / self.ideal

    def util(self, resource: str) -> float | None:
        """Achieved fraction of that resource's peak at the measured time (= t_resource / measured)."""
        return None if self.measured is None else self.times[resource] / self.measured

    def attained_tflops(self) -> float | None:
        return None if self.measured is None else self.flops / self.measured / 1e12


def roofline(
    M: int,
    op: Op,
    arch: Arch,
    fidelity: str = "HiFi2",
    num_links: int | None = None,
    dram_bw: float | None = None,
    measured_us: float | None = None,
) -> Roofline:
    L = arch.num_links if num_links is None else num_links
    R = arch.ring_size
    dram_bw = arch.dram_bw if dram_bw is None else dram_bw
    flops = 2.0 * M * op.K * op.N
    bytes_dram = BF16_BYTES * (M * op.K + op.K * op.N)
    if op.kind == "agmm":
        cores = arch.ring_matmul_cores
        shard = BF16_BYTES * M * (op.K / R)  # this device's in0 shard, gathered to the other R-1
    else:  # plain matmul on the full grid, then reduce-scatter of the [M, N] partial into N/R per device
        cores = arch.full_cores
        shard = BF16_BYTES * M * (op.N / R)
    bytes_per_link = (R - 1) * shard / (2 * L)
    peak = arch.peak_flops(fidelity, cores)
    return Roofline(
        op=op,
        arch=arch,
        M=M,
        fidelity=fidelity,
        num_links=L,
        dram_bw=dram_bw,
        peak_flops=peak,
        flops=flops,
        bytes_dram=bytes_dram,
        bytes_per_link=bytes_per_link,
        t_compute=flops / peak,
        t_dram=bytes_dram / dram_bw,
        t_fabric=bytes_per_link / arch.link_bw,
        measured=None if measured_us is None else measured_us * 1e-6,
    )


def load_measured_csv(path: str, M: int, ops: list[Op]) -> dict[str, float]:
    """Best `OK` device_kernel_duration per op from a `sweep_mm_block_sizes.py` results CSV, in us."""
    by_kn = {(op.K, op.N): op.name for op in ops}
    best: dict[str, float] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "OK" or int(row["M"]) != M:
                continue
            name = by_kn.get((int(row["K"]), int(row["N"]))) or SWEEP_USE_CASE_TO_OP.get(row.get("use_case", ""))
            if name is None:
                continue
            us = int(row["device_kernel_duration_ns"]) / 1e3
            best[name] = min(us, best.get(name, math.inf))
    return best


# ----------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------


def _fmt_us(seconds: float) -> str:
    return f"{seconds * 1e6:,.1f}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def dump_table(rows: list[Roofline], title: str) -> str:
    out = [f"### {title}", ""]
    out.append(
        "| op | M | K_g | N | fusion | GFLOP | MB DRAM | MB/link | t_compute us | t_dram us | t_fabric us | ideal us | limiter | measured us | headroom | FPU util | DRAM util | fabric util |"
    )
    out.append("|" + "---|" * 18)
    for r in rows:
        out.append(
            f"| {r.op.name} | {r.M} | {r.op.K} | {r.op.N} | {r.op.fusion} | {r.flops / 1e9:,.0f} | {r.bytes_dram / 1e6:,.1f} | "
            f"{r.bytes_per_link / 1e6:,.1f} | {_fmt_us(r.t_compute)} | {_fmt_us(r.t_dram)} | {_fmt_us(r.t_fabric)} | "
            f"**{_fmt_us(r.ideal)}** | {r.limiter} | {'—' if r.measured is None else _fmt_us(r.measured)} | "
            f"{'—' if r.headroom is None else f'{r.headroom:.2f}x'} | {_pct(r.util('compute'))} | {_pct(r.util('dram'))} | {_pct(r.util('fabric'))} |"
        )
    block = [r for r in rows if r.M != M_REFINER]
    if block:
        ideal_block = sum(r.ideal for r in block)
        names = ", ".join(r.op.name for r in block)
        line = f"Ideal per block for {names}: {ideal_block * 1e3:.2f} ms -> {ideal_block * BLOCKS_PER_FORWARD * 1e3:.0f} ms per forward ({BLOCKS_PER_FORWARD} blocks)"
        if all(r.measured is not None for r in block):
            meas_block = sum(r.measured for r in block)
            line += f"; measured {meas_block * 1e3:.2f} ms per block, {meas_block / ideal_block:.2f}x headroom"
        out += ["", line]
    return "\n".join(out)


def constants_table(arches: list[Arch], fidelity: str) -> str:
    out = [f"### Architecture constants ({fidelity} speed of light)", ""]
    out.append("| constant | " + " | ".join(a.short for a in arches) + " |")
    out.append("|---|" + "---|" * len(arches))
    rows = [
        ("AI clock", lambda a: f"{a.clock_hz / 1e9:.2f} GHz"),
        (f"FLOP/cycle/core @ {fidelity}", lambda a: f"{LOFI_FLOP_PER_CYCLE_PER_CORE // FIDELITY_CYCLES[fidelity]}"),
        (
            "ring-matmul (AGMM) grid",
            lambda a: f"{a.ring_matmul_grid[0]}x{a.ring_matmul_grid[1]} = {a.ring_matmul_cores} cores (full {a.full_cores})",
        ),
        (f"peak @ {fidelity}, ring-matmul grid", lambda a: f"{a.peak_flops(fidelity) / 1e12:.1f} TFLOP/s"),
        ("peak @ LoFi, ring-matmul grid", lambda a: f"{a.peak_flops('LoFi') / 1e12:.1f} TFLOP/s"),
        ("DRAM bandwidth", lambda a: f"{a.dram_bw / 1e9:.0f} GB/s"),
        ("eth link, unidirectional", lambda a: f"{a.link_bw / 1e9:.1f} GB/s"),
        ("links per direction", lambda a: f"{a.num_links}"),
        ("ring ingress, aggregate", lambda a: f"{a.num_links * a.link_bw / 1e9:.0f} GB/s"),
        ("ring size (TP)", lambda a: f"{a.ring_size}"),
        ("L1 per core", lambda a: f"{a.l1_bytes // 1024} KB"),
        ("N* at shipped links", lambda a: f"{a.n_star(fidelity):,.0f}"),
        ("N* at 1 / 2 / 3 / 4 links", lambda a: " / ".join(f"{a.n_star(fidelity, L):,.0f}" for L in (1, 2, 3, 4))),
        ("source", lambda a: a.source),
    ]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(fn(a) for a in arches) + " |")
    return "\n".join(out)


def write_csv(rows: list[Roofline], path: str) -> None:
    fields = [
        "arch",
        "op",
        "M",
        "K_gathered",
        "N",
        "fusion",
        "fidelity",
        "num_links",
        "dram_GBps",
        "peak_TFLOPs",
        "GFLOP",
        "MB_dram",
        "MB_per_link",
        "ai_dram_flop_per_byte",
        "ai_fabric_flop_per_byte",
        "t_compute_us",
        "t_dram_us",
        "t_fabric_us",
        "ideal_us",
        "limiter",
        "measured_us",
        "headroom",
        "attained_TFLOPs",
        "flop_util",
        "dram_util",
        "fabric_util",
    ]
    with open(path, "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "arch": r.arch.short,
                    "op": r.op.name,
                    "M": r.M,
                    "K_gathered": r.op.K,
                    "N": r.op.N,
                    "fusion": r.op.fusion,
                    "fidelity": r.fidelity,
                    "num_links": r.num_links,
                    "dram_GBps": r.dram_bw / 1e9,
                    "peak_TFLOPs": round(r.peak_flops / 1e12, 3),
                    "GFLOP": round(r.flops / 1e9, 3),
                    "MB_dram": round(r.bytes_dram / 1e6, 3),
                    "MB_per_link": round(r.bytes_per_link / 1e6, 3),
                    "ai_dram_flop_per_byte": round(r.ai_dram, 1),
                    "ai_fabric_flop_per_byte": round(r.ai_fabric, 1),
                    "t_compute_us": round(r.t_compute * 1e6, 3),
                    "t_dram_us": round(r.t_dram * 1e6, 3),
                    "t_fabric_us": round(r.t_fabric * 1e6, 3),
                    "ideal_us": round(r.ideal * 1e6, 3),
                    "limiter": r.limiter,
                    "measured_us": "" if r.measured is None else round(r.measured * 1e6, 1),
                    "headroom": "" if r.headroom is None else round(r.headroom, 3),
                    "attained_TFLOPs": "" if r.measured is None else round(r.attained_tflops(), 2),
                    "flop_util": "" if r.measured is None else round(r.util("compute"), 4),
                    "dram_util": "" if r.measured is None else round(r.util("dram"), 4),
                    "fabric_util": "" if r.measured is None else round(r.util("fabric"), 4),
                }
            )


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------


def _style(ax, grid_axis: str = "both") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _constants_line(arch: Arch, fidelity: str) -> str:
    return (
        f"{arch.short}: {arch.ring_matmul_cores} cores x {LOFI_FLOP_PER_CYCLE_PER_CORE // FIDELITY_CYCLES[fidelity]} FLOP/cycle x "
        f"{arch.clock_hz / 1e9:.2f} GHz = {arch.peak_flops(fidelity) / 1e12:.1f} TFLOP/s {fidelity}  ·  "
        f"DRAM {arch.dram_bw / 1e9:.0f} GB/s  ·  {arch.num_links} x {arch.link_bw / 1e9:.1f} GB/s links  ·  ring {arch.ring_size}"
    )


def fig_roofline(rows: list[Roofline], arch: Arch, fidelity: str, title: str, measured_note: str):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    peak_t = arch.peak_flops(fidelity) / 1e12
    lofi_t = arch.peak_flops("LoFi") / 1e12
    panels = (
        (
            "dram",
            "arithmetic intensity vs DRAM  (FLOP per byte of gathered in0 + resident in1)",
            arch.dram_bw,
            lambda r: r.ai_dram,
        ),
        (
            "fabric",
            "arithmetic intensity vs fabric  (FLOP per byte moved on one ethernet link)",
            arch.link_bw,
            lambda r: r.ai_fabric,
        ),
    )
    for ax, (res, xlabel, bw, ai_of) in zip(axes, panels):
        _style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ridge = peak_t * 1e12 / bw
        ais = [ai_of(r) for r in rows]
        xlo, xhi = ridge / 60, max(max(ais) * 4, ridge * 4)
        x = np.logspace(math.log10(xlo), math.log10(xhi), 400)
        slope = bw * x / 1e12
        # bandwidth slope (its own colour) up to the ridge, compute ceiling (compute colour) past it
        ax.plot(x[x <= ridge], slope[x <= ridge], color=RESOURCE_COLOR[res], lw=2.2, solid_capstyle="round")
        ax.plot(x[x <= ridge], np.full((x <= ridge).sum(), peak_t), color=RESOURCE_COLOR["compute"], lw=1.0, alpha=0.35)
        ax.plot(x[x >= ridge], np.full((x >= ridge).sum(), peak_t), color=RESOURCE_COLOR["compute"], lw=2.2)
        ax.plot(x, np.full_like(x, lofi_t), color=RESOURCE_COLOR["compute"], lw=1.0, ls=(0, (4, 2, 1, 2)), alpha=0.8)
        ax.plot([ridge], [peak_t], marker="|", color=INK_2, ms=10, mew=1.2)
        ax.annotate(
            f"ridge {ridge:,.0f} FLOP/B",
            (ridge, peak_t),
            xytext=(5, -12),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=INK_2,
        )
        ax.text(
            xlo * 1.25,
            peak_t * 1.04,
            f"{fidelity} speed of light {peak_t:.1f} TFLOP/s",
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=RESOURCE_COLOR["compute"],
        )
        ax.text(
            xlo * 1.25,
            lofi_t * 1.04,
            f"LoFi ceiling {lofi_t:.1f} TFLOP/s (not the operating point)",
            ha="left",
            va="bottom",
            fontsize=8,
            color=RESOURCE_COLOR["compute"],
            alpha=0.85,
        )
        # slope label sits beside the slope at mid-height (rotation would depend on the axes aspect)
        y_lab = math.sqrt(peak_t / 8 * peak_t) / 1.6
        ax.annotate(
            f"{RESOURCE_LABEL[res]} slope\n{bw / 1e9:.1f} GB/s" + (" per link" if res == "fabric" else ""),
            (y_lab * 1e12 / bw, y_lab),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8.5,
            color=RESOURCE_COLOR[res],
            ha="left",
            va="center",
        )
        handles = []
        for r in rows:
            xi = ai_of(r)
            y_ideal = r.flops / r.ideal / 1e12
            ax.plot([xi], [y_ideal], marker=r.op.marker, ms=8, color=INK, mec=INK, mfc=INK, zorder=5)
            label = f"{r.op.name}: ideal {r.ideal * 1e3:.2f} ms, {RESOURCE_LABEL[r.limiter]}-bound"
            if r.measured is not None:
                y_meas = r.attained_tflops()
                ax.plot([xi, xi], [y_meas, y_ideal], color=INK_MUTED, lw=1.0, zorder=4)
                ax.plot([xi], [y_meas], marker=r.op.marker, ms=8, mfc="white", mec=INK, mew=1.6, zorder=6)
                label += f"; measured {r.measured * 1e3:.2f} ms = {y_meas:.0f} TFLOP/s, {r.headroom:.2f}x to ideal"
            handles.append(Line2D([], [], marker=r.op.marker, ls="", ms=7, mfc="white", mec=INK, mew=1.4, label=label))
        ax.legend(
            handles=handles,
            loc="lower right",
            frameon=True,
            framealpha=0.92,
            edgecolor="none",
            fontsize=8,
            handletextpad=0.4,
            borderaxespad=0.2,
        )
        ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
        ax.set_xlim(xlo, xhi)
    ymin = min(min(r.attained_tflops() or peak_t for r in rows), peak_t / 8) * 0.25
    axes[0].set_ylim(ymin, lofi_t * 2.2)
    axes[0].set_ylabel("attainable TFLOP/s per device", fontsize=9, color=INK_2)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.945, _constants_line(arch, fidelity), fontsize=8.5, color=INK_2)
    fig.text(
        0.01,
        0.915,
        f"Filled marker = roofline ideal (100% of the binding resource), hollow = {measured_note}; the drop line is the headroom. "
        "Per device, bf16.",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_time_bars(rows_by_arch: dict[str, list[Roofline]], fidelity: str, title: str, measured_note: str):
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mtransforms
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    arches = list(rows_by_arch)
    ingress = [rows[0].arch.num_links * rows[0].arch.link_bw for rows in rows_by_arch.values()]
    fig, axes = plt.subplots(len(arches), 1, figsize=(11, 3.9 * len(arches)), sharey=True, squeeze=False)
    ymax = 0.0
    width, gap = 0.24, 0.03
    for ax, key in zip(axes[:, 0], arches):
        rows = rows_by_arch[key]
        arch = rows[0].arch
        _style(ax, grid_axis="y")
        xs = np.arange(len(rows)) * 1.3
        tick_pos, tick_lab = [], []
        for i, res in enumerate(RESOURCES):
            vals = [r.times[res] * 1e3 for r in rows]
            pos = xs + (i - 1) * (width + gap)
            ax.bar(pos, vals, width, color=RESOURCE_COLOR[res], zorder=3)
            for x, v, r in zip(pos, vals, rows):
                if r.limiter == res:
                    continue  # the limiter's value is the ideal, stated in the group note above the tick
                ax.annotate(
                    f"{v:,.2f}",
                    (x, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=INK_2,
                )
            tick_pos += list(pos)
            tick_lab += [RESOURCE_LABEL[res]] * len(rows)
        for xi, r in zip(xs, rows):
            ideal = r.ideal * 1e3
            lim_x = xi + (RESOURCES.index(r.limiter) - 1) * (width + gap)
            ax.hlines(ideal, lim_x - 0.75 * width, lim_x + 0.75 * width, color=INK, lw=1.8, zorder=5)
            note = f"ideal {ideal:,.2f} ms = the {RESOURCE_LABEL[r.limiter]} bar ({RESOURCE_LABEL[r.limiter]}-bound)"
            top = ideal
            if r.measured is not None:
                meas = r.measured * 1e3
                ax.plot([xi], [meas], marker="D", ms=8, mfc="white", mec=INK, mew=1.6, zorder=6)
                ax.vlines(xi, ideal, meas, color=INK_MUTED, lw=1.0, ls=(0, (2, 2)), zorder=4)
                note += f"\nmeasured {meas:,.2f} ms = {r.headroom:.2f}x ideal, FPU {_pct(r.util('compute'))}"
                top = max(top, meas)
            ax.annotate(
                note,
                (xi, top),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_2,
            )
            ymax = max(ymax, top)
            # op group label under the three resource tick labels
            ax.text(
                xi,
                -0.16,
                f"{r.op.name}   (K {r.op.K:,} · N {r.op.N:,} · {r.op.fusion})",
                transform=mtransforms.blended_transform_factory(ax.transData, ax.transAxes),
                ha="center",
                va="top",
                fontsize=9,
                color=INK,
                fontweight="bold",
            )
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=8, color=INK_2)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
        ax.set_ylabel("best-case time per call, ms", fontsize=9, color=INK_2)
        ax.text(
            0.0, 1.02, _constants_line(arch, fidelity), transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom"
        )
        handles = [
            Patch(color=RESOURCE_COLOR[res], label=f"{RESOURCE_LABEL[res]} time at 100% of peak") for res in RESOURCES
        ]
        handles.append(Line2D([], [], color=INK, lw=1.6, label="ideal = tallest bar"))
        if any(r.measured is not None for r in rows):
            handles.append(Line2D([], [], marker="D", ls="", ms=7, mfc="white", mec=INK, mew=1.4, label="measured"))
        ax.legend(
            handles=handles,
            loc="upper left",
            frameon=False,
            fontsize=8,
            ncol=len(handles),
            handlelength=1.4,
            columnspacing=1.2,
        )
    axes[0, 0].set_ylim(0, ymax * 1.5)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(
        0.01,
        0.005,
        "Each group is one op; its three bars are the time that op would take if it were limited only by compute, only by DRAM\n"
        "bandwidth, or only by fabric (all-gather) bandwidth. The tallest bar is the roofline ideal (black tick).\n"
        f"Hollow diamond = {measured_note}."
        + (
            f" Fabric bars coincide across arches: every ring ingests {ingress[0] / 1e9:.0f} GB/s per direction."
            if len(set(ingress)) == 1 and len(ingress) > 1
            else ""
        ),
        fontsize=8,
        color=INK_MUTED,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=3.0)
    return fig


def fig_stacked(rows_by_arch: dict[str, list[Roofline]], fidelity: str, title: str, measured_note: str):
    """One stacked bar per arch: the selected ops' ideal times summed, colour-coded by op, with the summed
    measured wall time marked on top (and, where measured, a second stack of the measured times)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    _style(ax, grid_axis="y")
    width = 0.55
    columns = []  # (x, label, [(op, seconds)], hatch, sublabel)
    op_color: dict[str, str] = {}
    x = 0.0
    for key, rows in rows_by_arch.items():
        block = [r for r in rows if r.M != M_REFINER]
        arch = block[0].arch
        op_color.update({r.op.name: r.op.color for r in block})
        columns.append(
            (x, f"{arch.short} ideal", [(r.op.name, r.ideal) for r in block], None, _constants_line(arch, fidelity))
        )
        x += 1.0
        if all(r.measured is not None for r in block):
            columns.append((x, f"{arch.short} measured", [(r.op.name, r.measured) for r in block], "//", None))
            x += 1.0
        x += 0.35
    ymax = 0.0
    totals: dict[str, float] = {}
    for cx, label, parts, hatch, _ in columns:
        bottom = 0.0
        for op_name, secs in parts:
            ms = secs * 1e3
            ax.bar(
                cx,
                ms,
                width,
                bottom=bottom,
                color=op_color[op_name],
                hatch=hatch,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            if ms > 1.2:
                ax.text(
                    cx,
                    bottom + ms / 2,
                    f"{op_name}  {ms:,.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                )
            bottom += ms
        totals[label] = bottom
        ax.text(
            cx,
            bottom,
            f"{bottom:,.2f} ms\n×{BLOCKS_PER_FORWARD} blocks = {bottom * BLOCKS_PER_FORWARD:,.0f} ms / forward",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
            linespacing=1.3,
        )
        ymax = max(ymax, bottom)
    # the summed measured wall time sitting on top of the ideal stack it is compared against
    for cx, label, parts, hatch, _ in columns:
        meas_label = label.replace("ideal", "measured")
        if hatch is None and meas_label in totals:
            ideal, meas = totals[label], totals[meas_label]
            ax.hlines(meas, cx - width / 2, cx + 1.0 + width / 2, color=INK, lw=1.4, ls=(0, (3, 2)), zorder=5)
            ax.plot([cx], [meas], marker="D", ms=9, mfc="white", mec=INK, mew=1.6, zorder=6)
            ax.annotate(
                f"measured sum {meas:,.2f} ms\n{meas / ideal:.2f}x the ideal sum",
                (cx, meas),
                xytext=(0, 34),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=INK_2,
            )
            ymax = max(ymax, meas)
    ax.set_xticks([c[0] for c in columns])
    ax.set_xticklabels([c[1] for c in columns], fontsize=9.5, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.7, columns[-1][0] + 0.7)
    ax.set_ylim(0, ymax * 1.28)
    ax.set_ylabel(
        f"time per transformer block, ms ({len(op_color)} calls: {', '.join(op_color)})", fontsize=9, color=INK_2
    )
    handles = [Patch(color=color, label=name) for name, color in op_color.items()]
    handles.append(Patch(facecolor="white", edgecolor=INK_2, hatch="//", label="measured stack"))
    handles.append(
        Line2D([], [], marker="D", ls=(0, (3, 2)), color=INK, ms=7, mfc="white", mec=INK, label="measured sum")
    )
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(
        0.01, 0.925, "\n".join(c[4] for c in columns if c[4]), fontsize=8.5, color=INK_2, va="top", linespacing=1.4
    )
    fig.text(
        0.01,
        0.005,
        "Ideal = each op's roofline lower bound (its binding resource at 100% of peak), summed over the selected ops of one "
        f"block.\nMeasured = {measured_note}.",
        fontsize=8,
        color=INK_MUTED,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


def fig_nstar(arches: list[Arch], ops: list[Op], fidelity: str, title: str):
    """N* (compute/fabric crossover width) vs link count. Only the ring all-gather matmuls have an N*, so only
    ops of the agmm family are drawn as reference lines."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ops = [op for op in ops if op.kind == "agmm"]
    links = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _style(ax)
    ax.set_yscale("log")
    values = [op.N for op in ops] + [a.n_star(fidelity, L) for a in arches for L in links]
    lo, hi = min(values) / 1.25, max(values) * 1.15
    ticks = [
        t for t in (200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000) if lo <= t <= hi
    ]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(lo, hi)
    styles = [
        dict(color=INK, ls="-", marker="o"),
        dict(color=INK_MUTED, ls=(0, (4, 2)), marker="s"),
        dict(color=INK_2, ls=":", marker="^"),
    ]
    for i, arch in enumerate(arches):
        ns = [arch.n_star(fidelity, L) for L in links]
        st = styles[i % len(styles)]
        ax.plot(
            links,
            ns,
            lw=2,
            ms=6,
            mfc="white",
            mew=1.6,
            **st,
            label=f"N* {arch.short}: {arch.peak_flops(fidelity) / 1e12:.0f} TFLOP/s, {arch.link_bw / 1e9:.1f} GB/s/link",
        )
        L = arch.num_links
        ax.plot([L], [arch.n_star(fidelity, L)], marker=st["marker"], ms=9, color=st["color"], zorder=6)
        ax.annotate(
            f"{arch.short} shipped: {L} links, N* = {arch.n_star(fidelity, L):,.0f}",
            (L, arch.n_star(fidelity, L)),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=st["color"],
        )
    for op in ops:
        ax.axhline(op.N, color=GRID, lw=1.2, zorder=1)
        ax.axhline(op.N, color=INK_2, lw=0.8, ls=(0, (1, 3)), zorder=2)
        ax.text(
            4.02,
            op.N,
            f"{op.name}  N = {op.N:,}",
            fontsize=8.5,
            color=INK_2,
            va="center",
            ha="left",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"),
            zorder=7,
        )
    ax.set_xticks(links)
    ax.set_xlim(0.8, 4.9)
    ax.set_xlabel("ethernet links per ring direction", fontsize=9, color=INK_2)
    ax.set_ylabel("N* — output width at the compute/fabric crossover (elements)", fontsize=9, color=INK_2)
    ax.text(
        0.98,
        0.56,
        "op N above its arch's line: compute-bound at any M\nbelow it: fabric-bound at any M",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(
        0.01,
        0.93,
        f"N* = peak / (2 · link BW) · (R−1) / (R · L), R = 4, {fidelity}. M cancels: the regime is fixed by N, ring and links.",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


# ----------------------------------------------------------------------------------------------
# self-test against the artifact numbers
# ----------------------------------------------------------------------------------------------


def selftest() -> None:
    def close(a, b, tol):
        assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"

    # The BH shape database row ltx_m1216_k4096_n8 (ring 4, 2 links): t_compute 0.3, t_dram 19.6, t_fabric 74.7 us.
    r = roofline(1216, Op.adhoc("ltx", 4096, 8, "—"), BH, "HiFi2")
    close(r.t_fabric * 1e6, 74.7, 0.05)
    close(r.t_dram * 1e6, 19.6, 0.05)
    close(r.t_compute * 1e6, 0.27, 0.01)
    # ltx_m4864_k4096_n3072_c3: t_compute 409.9, t_dram 127.0, t_fabric 298.8 -> compute.
    r = roofline(4864, Op.adhoc("ltx", 4096, 3072, "chunks3"), BH, "HiFi2")
    close(r.t_compute * 1e6, 409.9, 0.1)
    close(r.t_dram * 1e6, 127.0, 0.1)
    close(r.t_fabric * 1e6, 298.8, 0.1)
    assert r.limiter == "compute"
    close(BH.peak_flops("HiFi2") / 1e12, 298.6, 0.05)
    close(BH.n_star("HiFi2", 1), 4479, 1)  # artifact N* table, ring 4 / 1 link
    # Wormhole anchors (MiniMaxH3_wormhole_perf.md roofline section: ff1 1.05 T, 8.0 ms at 64 cores).
    close(WH.peak_flops("HiFi2") / 1e12, 131.07, 0.01)
    r = roofline(M_15S_768P_16_9, OPS_BY_NAME["ff1"], WH, "HiFi2", measured_us=MEASURED_US_WH_15S["ff1"])
    close(r.flops / 1e12, 1.053, 0.001)
    close(r.t_compute * 1e6, 8034.4, 0.5)
    close(r.t_dram * 1e6, 777.7, 0.5)
    close(r.t_fabric * 1e6, 1101.9, 0.5)
    assert r.limiter == "compute"
    close(r.util("compute"), 0.511, 0.002)
    close(WH.n_star("HiFi2"), 983, 1)
    # to_out: compute-bound on WH at 4 links, fabric-bound on BH at 2 links.
    assert roofline(M_15S_768P_16_9, OPS_BY_NAME["to_out"], WH).limiter == "compute"
    assert roofline(M_15S_768P_16_9, OPS_BY_NAME["to_out"], BH).limiter == "fabric"
    print("selftest OK")


# ----------------------------------------------------------------------------------------------
# whole transformer block from a Tracy ops_perf_results CSV
# ----------------------------------------------------------------------------------------------

DEFAULT_PROFILE_CSV = "generated/profiler/reports/2026_09_17_21_33_20/ops_perf_results_2026_09_17_21_33_20.csv"
DTYPE_BYTES = {
    "BFLOAT16": 2,
    "FLOAT32": 4,
    "UINT32": 4,
    "INT32": 4,
    "UINT16": 2,
    "BFLOAT8_B": 1.0625,
    "BFLOAT4_B": 0.5625,
    "UINT8": 1,
}
SDPA_COMPUTE_CORES = 63  # 7x9: CORE COUNT reads 71 because it includes the fused CCL workers (perf doc)
OTHER_SHARE = 0.01  # ops below this share of the block are grouped as "other"
CLASS_COLOR = {
    "compute": RESOURCE_COLOR["compute"],
    "dram": RESOURCE_COLOR["dram"],
    "fabric": RESOURCE_COLOR["fabric"],
    "other": "#8a8983",
}
CLASS_LABEL = {"compute": "compute-bound", "dram": "DRAM-bound", "fabric": "fabric-bound", "other": "measured only"}
BOUND_NOTE = (
    "Bound models (per op class, stated assumptions): compute = 2·FLOPs / (op's core count × 2048 FLOP/cycle × 1.0 GHz, HiFi2); "
    "DRAM = bytes in + out of DRAM-resident tensors / 288 GB/s;\n"
    "fabric = ring volume (R−1)·shard / (2·links) / 12.5 GB/s per link; ideal = max of the terms that apply. "
    "\nSDPA FLOPs = 4·S_local·S_total·d·heads (full joint attention) on 63 compute cores. Ops under 1% of the block are 'other' (measured only)."
)


@dataclass
class BlockOp:
    name: str
    op_code: str
    calls: int
    measured: float  # seconds, summed over calls, merged across devices
    klass: str  # compute | dram | fabric | other
    t_compute: float = 0.0
    t_dram: float = 0.0
    t_fabric: float = 0.0
    formula: str = ""

    @property
    def ideal(self) -> float | None:
        if self.klass == "other":
            return None
        return max(self.t_compute, self.t_dram, self.t_fabric)

    @property
    def limiter(self) -> str:
        if self.klass == "other":
            return "other"
        return max(
            {"compute": self.t_compute, "dram": self.t_dram, "fabric": self.t_fabric}.items(), key=lambda kv: kv[1]
        )[0]


def _tensors(row: dict, prefix: str) -> list[tuple[tuple[int, ...], str, bool]]:
    """[(padded shape, dtype, is_dram)] for INPUT_i / OUTPUT_i columns of one Tracy row."""
    out = []
    for i in range(8):
        w = row.get(f"{prefix}_{i}_W_PAD[LOGICAL]", "")
        if not w:
            break
        shape = tuple(int(row[f"{prefix}_{i}_{d}_PAD[LOGICAL]"].split("[")[0]) for d in "WZYX")
        out.append((shape, row.get(f"{prefix}_{i}_DATATYPE", ""), "DRAM" in row.get(f"{prefix}_{i}_MEMORY", "")))
    return out


def _bytes(tensors, only_dram: bool = True) -> float:
    total = 0.0
    for shape, dtype, is_dram in tensors:
        if only_dram and not is_dram:
            continue
        n = 1
        for d in shape:
            n *= d
        total += n * DTYPE_BYTES.get(dtype, 2)
    return total


def _attr(row: dict, key: str, default):
    m = re.search(rf"'{key}': '([^']*)'", row.get("ATTRIBUTES", ""))
    if not m or not m.group(1).isdigit():
        return default
    return int(m.group(1))


def load_block_profile(path: str, arch: Arch, fidelity: str = "HiFi2") -> list[BlockOp]:
    """Warm-iteration ops of one transformer block, one BlockOp per (op, shape) group, rooflined by class."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    start = stop = None
    for i, r in enumerate(rows):
        if r.get("OP TYPE") != "signpost":
            continue
        if r["OP CODE"] == "start" and start is None:
            start = i
        elif r["OP CODE"] == "stop" and start is not None and stop is None:
            stop = i
    warm = rows[start + 1 : stop] if start is not None else rows
    by_dev: dict[str, list[dict]] = {}
    for r in warm:
        if r.get("OP TYPE") == "signpost":
            continue
        by_dev.setdefault(r["DEVICE ID"], []).append(r)
    devices = list(by_dev.values())
    n_ops = min(len(d) for d in devices)
    per_cycle = LOFI_FLOP_PER_CYCLE_PER_CORE / FIDELITY_CYCLES[fidelity]
    dram_bw, link_bw = arch.dram_bw, arch.link_bw
    groups: dict[str, BlockOp] = {}
    for i in range(n_ops):
        r0 = devices[0][i]
        code = r0["OP CODE"]
        durs = [int(d[i]["DEVICE KERNEL DURATION [ns]"]) * 1e-9 for d in devices if d[i]["DEVICE KERNEL DURATION [ns]"]]
        # tt-perf-report convention (project_block_perf._per_op): mean for collectives, max otherwise
        dur = (
            (sum(durs) / len(durs))
            if any(k in code.lower() for k in ("allgather", "reducescatter", "broadcast"))
            else max(durs)
        )
        ins, outs = _tensors(r0, "INPUT"), _tensors(r0, "OUTPUT")
        cores = int(r0.get("CORE COUNT") or 0) or arch.full_cores
        R, L = _attr(r0, "ring_size", arch.ring_size), _attr(r0, "num_links", arch.num_links)
        name, klass, tc, td, tf, formula = (
            code.replace("DeviceOperation", "").replace("Op", ""),
            "other",
            0.0,
            0.0,
            0.0,
            "",
        )
        if code == "AllGatherMinimalMatmulAsyncOp":
            m_rows, k_g, n = ins[0][0][2], ins[1][0][2], ins[1][0][3]
            op = {s.N: s for s in AGMM_OPS}.get(n, Op.adhoc(f"agmm N{n}", k_g, n, "?"))
            rl = roofline(m_rows, op, arch, fidelity, num_links=L)
            name, klass, tc, td, tf = f"AGMM {op.name}", rl.limiter, rl.t_compute, rl.t_dram, rl.t_fabric
            formula = f"AGMM roofline: 2·{m_rows}·{k_g}·{n} FLOP on {arch.ring_matmul_cores} cores; gather (R−1)·M·K_local·2B/(2·{L})"
        elif code == "RingJointSDPADeviceOperation":
            heads, s_local, d = ins[0][0][1], ins[0][0][2], ins[0][0][3]
            s_total = ins[3][0][2] if len(ins) > 3 else s_local
            flops = 4.0 * s_local * s_total * d * heads
            tc = flops / (SDPA_COMPUTE_CORES * per_cycle * arch.clock_hz)
            td = _bytes(ins + outs) / dram_bw
            name, klass = "RingJointSDPA", "compute"
            formula = f"4·{s_local}·{s_total}·{d}·{heads} = {flops / 1e12:.2f} TFLOP on {SDPA_COMPUTE_CORES} cores"
        elif code == "MinimalMatmulDeviceOperation":
            m_rows, k, n = ins[0][0][2], ins[1][0][2], ins[1][0][3]
            tc = 2.0 * m_rows * k * n / (cores * per_cycle * arch.clock_hz)
            td = _bytes(ins + outs) / dram_bw
            name = f"MinimalMatmul {OPS_BY_NAME['ff2'].name}" if m_rows > 32 else f"MinimalMatmul M={m_rows} (adaLN)"
            klass = "compute" if tc >= td else "dram"
            formula = f"2·{m_rows}·{k}·{n} FLOP on {cores} cores; {_bytes(ins + outs) / 1e6:.0f} MB DRAM"
        elif code == "ReduceScatterMinimalAsyncDeviceOperation":
            b = _bytes(ins[:1])
            tf = (R - 1) * (b / R) / (2 * L) / link_bw
            td = (b + b / R) / dram_bw
            name, klass = "ReduceScatter (ff2)", "fabric"
            formula = f"(R−1)·(B/R)/(2·L): B={b / 1e6:.0f} MB, R={R}, L={L}"
        elif code == "AllGatherAsyncDeviceOperation":
            b = _bytes(ins[:1])
            tf = (R - 1) * b / (2 * L) / link_bw
            td = (b + R * b) / dram_bw
            name, klass = "AllGatherAsync (FSDP weights)", "fabric"
            formula = f"(R−1)·shard/(2·L): shard={b / 1e6:.1f} MB, R={R}, L={L}"
        elif code == "AllBroadcastDeviceOperation":
            b, n_out = _bytes(ins[:1]), max(1, len(outs))
            tf = (n_out - 1) * b / (2 * L) / link_bw
            td = (b + n_out * b) / dram_bw
            name, klass = "AllBroadcast (FSDP)", "fabric"
            formula = f"(n_out−1)·B/(2·L): B={b / 1e6:.1f} MB, n_out={n_out}, L={L}"
        elif code in (
            "EmbeddingsDeviceOperation",
            "DitFusedDistributedRmsnormDeviceOperation",
            "UntilizeWithUnpaddingDeviceOperation",
            "TilizeWithValPaddingDeviceOperation",
            "ConcatDeviceOperation",
        ):
            td = _bytes(ins + outs) / dram_bw
            klass = "dram"
            name = {
                "EmbeddingsDeviceOperation": "Embeddings (adaLN tables)",
                "DitFusedDistributedRmsnormDeviceOperation": "DistributedRMSNorm",
                "UntilizeWithUnpaddingDeviceOperation": "UntilizeWithUnpadding",
                "TilizeWithValPaddingDeviceOperation": "TilizeWithValPadding",
                "ConcatDeviceOperation": "Concat",
            }[code]
            formula = f"{_bytes(ins + outs) / 1e6:.0f} MB in+out / {dram_bw / 1e9:.0f} GB/s"
        g = groups.get(name)
        if g is None:
            groups[name] = BlockOp(name, code, 1, dur, klass, tc, td, tf, formula)
        else:
            g.calls += 1
            g.measured += dur
            g.t_compute += tc
            g.t_dram += td
            g.t_fabric += tf
    ops = list(groups.values())
    total = sum(o.measured for o in ops)
    keep, other = [], BlockOp("other (small ops)", "", 0, 0.0, "other")
    for o in sorted(ops, key=lambda o: -o.measured):
        if o.measured < OTHER_SHARE * total or o.klass == "other":
            other.calls += o.calls
            other.measured += o.measured
        else:
            keep.append(o)
    if other.calls:
        other.formula = "below 1% of the block each"
        keep.append(other)
    return keep


def dump_block_table(ops: list[BlockOp], title: str) -> str:
    out = [
        f"### {title}",
        "",
        "| op | calls | measured ms | ideal ms | limiter | util | headroom | bound model |",
        "|---|---|---|---|---|---|---|---|",
    ]
    tm = ti = 0.0
    for o in ops:
        tm += o.measured
        if o.ideal is None:
            out.append(f"| {o.name} | {o.calls} | {o.measured * 1e3:.2f} | — | measured only | — | — | {o.formula} |")
            continue
        ti += o.ideal
        out.append(
            f"| {o.name} | {o.calls} | {o.measured * 1e3:.2f} | {o.ideal * 1e3:.2f} | {o.limiter} | {100 * o.ideal / o.measured:.0f}% | "
            f"{o.measured / o.ideal:.2f}x | {o.formula} |"
        )
    out += [
        "",
        f"block measured {tm * 1e3:.1f} ms; sum of ideals {ti * 1e3:.1f} ms (rooflined ops); headroom {tm / ti:.2f}x",
    ]
    return "\n".join(out)


def _stack(ax, x, ops, key, width, label_min_frac):
    import matplotlib.patches as mpatches  # noqa: F401

    bottom, total = 0.0, sum((getattr(o, key) or 0.0) for o in ops)
    for o in ops:
        v = getattr(o, key)
        if not v:
            continue
        ms = v * 1e3
        ax.bar(x, ms, width, bottom=bottom, color=CLASS_COLOR[o.klass], edgecolor="white", linewidth=1.2, zorder=3)
        if v / total >= label_min_frac:
            ax.text(
                x,
                bottom + ms / 2,
                f"{o.name}  {ms:,.1f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )
        bottom += ms
    return bottom


def fig_block_stacked(ops: list[BlockOp], arch: Arch, fidelity: str, title: str, source: str):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.2), gridspec_kw={"width_ratios": [1, 1]})
    panels = (("whole block", ops), ("block without RingJointSDPA", [o for o in ops if o.name != "RingJointSDPA"]))
    for ax, (ptitle, pops) in zip(axes, panels):
        _style(ax, grid_axis="y")
        rooflined = [o for o in pops if o.ideal is not None]
        top_ideal = _stack(ax, 0.0, sorted(rooflined, key=lambda o: -o.ideal), "ideal", 0.6, 0.035)
        top_meas = _stack(ax, 1.0, sorted(pops, key=lambda o: -o.measured), "measured", 0.6, 0.035)
        ax.text(
            0.0, top_ideal, f"{top_ideal:,.1f} ms", ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold"
        )
        ax.text(
            1.0, top_meas, f"{top_meas:,.1f} ms", ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold"
        )
        ax.hlines(top_meas, -0.3, 1.3, color=INK, lw=1.2, ls=(0, (3, 2)), zorder=5)
        ax.annotate(
            f"{top_meas / top_ideal:.2f}x",
            (0.0, top_meas),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_2,
        )
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["roofline ideal\n(per op, summed)", "measured\n(Tracy, per op)"], fontsize=9, color=INK)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, max(top_ideal, top_meas) * 1.1)
        ax.set_ylabel("ms per transformer block, per device", fontsize=9, color=INK_2)
        ax.set_title(ptitle, fontsize=10, color=INK, loc="left")
    handles = [Patch(color=CLASS_COLOR[k], label=CLASS_LABEL[k]) for k in ("compute", "dram", "fabric", "other")]
    handles.append(Line2D([], [], color=INK, ls=(0, (3, 2)), label="measured total"))
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.935),
        ncol=5,
        frameon=False,
        fontsize=8.5,
        title="segment colour = bound class",
        title_fontsize=8.5,
    )
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.945, f"{_constants_line(arch, fidelity)}   ·   source {source}", fontsize=8.5, color=INK_2)
    fig.text(0.01, 0.005, BOUND_NOTE, fontsize=7.8, color=INK_MUTED, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    return fig


def fig_block_ops(ops: list[BlockOp], arch: Arch, fidelity: str, title: str, source: str):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = sorted(ops, key=lambda o: o.measured)
    fig, ax = plt.subplots(figsize=(13, 0.48 * len(rows) + 3.2))
    _style(ax, grid_axis="x")
    ax.set_xscale("log")
    ys = list(range(len(rows)))
    for y, o in zip(ys, rows):
        c = CLASS_COLOR[o.klass]
        ax.barh(y + 0.18, o.measured * 1e3, 0.34, color=c, alpha=0.55, zorder=3)
        note = f"{o.measured * 1e3:,.2f} ms measured"
        if o.ideal is not None:
            ax.barh(y - 0.18, o.ideal * 1e3, 0.34, color=c, zorder=3)
            note += f"  ·  ideal {o.ideal * 1e3:,.2f} ({CLASS_LABEL[o.limiter]}, {100 * o.ideal / o.measured:.0f}% util, {o.measured / o.ideal:.1f}x)"
        ax.text(o.measured * 1e3 * 1.08, y, note, va="center", fontsize=8, color=INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{o.name}  ×{o.calls}" if o.calls > 1 else o.name for o in rows], fontsize=8.5, color=INK)
    ax.set_xlabel("ms per block (log)", fontsize=9, color=INK_2)
    xmax = max(o.measured for o in rows) * 1e3
    ax.set_xlim(0.05, xmax * 40)
    handles = [
        Patch(color=CLASS_COLOR[k], label=CLASS_LABEL[k] + " (ideal, solid)") for k in ("compute", "dram", "fabric")
    ]
    handles.append(Patch(color=INK_MUTED, alpha=0.55, label="measured (faded)"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.93, f"{_constants_line(arch, fidelity)}   ·   source {source}", fontsize=8.5, color=INK_2)
    fig.text(0.01, 0.005, BOUND_NOTE, fontsize=7.8, color=INK_MUTED, va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.1, 1, 0.91))
    return fig


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--figs",
        default="all",
        help="comma list of roofline,bars,stacked,nstar,block_stacked,block_ops or all (default) or none",
    )
    p.add_argument("--dump", action="store_true", help="print the constants and roofline tables (markdown) to stdout")
    p.add_argument(
        "--selftest", action="store_true", help="check the port against the artifact rows and the perf-doc anchors"
    )
    p.add_argument(
        "--fidelity",
        default="HiFi2",
        choices=sorted(FIDELITY_CYCLES),
        help="math fidelity of the speed-of-light (model runs HiFi2)",
    )
    p.add_argument("--dram", default="spec", help="WH DRAM BW: spec (288) | measured (267) | perf_model (258) | <GB/s>")
    p.add_argument("--links", type=int, default=None, help="override WH links per direction (mesh config uses 4)")
    p.add_argument(
        "--M", type=int, default=M_15S_768P_16_9, help="rows per device (default 13664 = 15 s / 768P / 16:9 at SP=8)"
    )
    p.add_argument(
        "--ops",
        default="agmm",
        help=f"ops to roofline: agmm (default: {', '.join(s.name for s in AGMM_OPS)}), all, or a comma list of op names / families",
    )
    p.add_argument(
        "--include-ff2", action="store_true", help="alias for adding ff2 to --ops (plain matmul + reduce-scatter)"
    )
    p.add_argument(
        "--include-refiner", action="store_true", help="add the token-refiner instances of the selected ops at M=64"
    )
    p.add_argument("--no-bh", action="store_true", help="drop the Blackhole side of the comparison")
    p.add_argument(
        "--measured-csv",
        default=None,
        help="sweep_mm_block_sizes.py results CSV; best OK time per op replaces the shipped numbers",
    )
    p.add_argument("--no-measured", action="store_true", help="do not overlay measured times")
    p.add_argument(
        "--profile-csv",
        default=DEFAULT_PROFILE_CSV,
        help="Tracy ops_perf_results CSV of one block (fsdp1 15 s profile by default)",
    )
    p.add_argument(
        "--no-block",
        "--agmm-only",
        dest="no_block",
        action="store_true",
        help="selected-op figures/tables only, no whole-block mode (block figures need --profile-csv)",
    )
    p.add_argument("--out-dir", default="transformer_roofline_out")
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()

    if args.selftest:
        selftest()
        if not args.dump and args.figs == "none":
            return

    dram_bw = WH_DRAM_BW.get(args.dram)
    if dram_bw is None:
        dram_bw = float(args.dram) * 1e9
    wh = replace(WH, dram_bw=dram_bw, num_links=args.links or WH.num_links)
    arches = [wh] if args.no_bh else [wh, BH]

    ops = select_ops(args.ops + (",ff2" if args.include_ff2 else ""))
    measured: dict[str, float] = {}
    if not args.no_measured:
        measured = dict(MEASURED_US_WH_15S) if args.M == M_15S_768P_16_9 else {}
        if args.measured_csv:
            measured.update(load_measured_csv(args.measured_csv, args.M, ops))

    rows_by_arch: dict[str, list[Roofline]] = {}
    for arch in arches:
        rows = [
            roofline(args.M, op, arch, args.fidelity, measured_us=measured.get(op.name) if arch is wh else None)
            for op in ops
        ]
        if args.include_refiner:
            rows += [roofline(M_REFINER, op, arch, args.fidelity) for op in ops]
        rows_by_arch[arch.short] = rows

    all_rows = [r for rows in rows_by_arch.values() for r in rows]
    block_ops = None
    if not args.no_block:
        if os.path.exists(args.profile_csv):
            block_ops = load_block_profile(args.profile_csv, wh, args.fidelity)
        else:
            print(
                f"note: no block profile at {args.profile_csv}; block figures skipped (pass --profile-csv or --no-block)"
            )
    if args.dump and block_ops:
        print(
            dump_block_table(
                block_ops,
                f"Transformer block — {wh.name}, per device, {args.fidelity}, {os.path.basename(args.profile_csv)}",
            )
        )
        print()
    if args.dump:
        print(constants_table(arches, args.fidelity))
        print()
        for key, rows in rows_by_arch.items():
            print(dump_table(rows, f"{rows[0].arch.name} — per device, M = {args.M}, {args.fidelity}"))
            print()

    figs = set() if args.figs == "none" else set(args.figs.split(","))
    if "all" in figs:
        figs = {"roofline", "bars", "stacked", "nstar"} | (set() if args.no_block else {"block_stacked", "block_ops"})
    if not figs and not args.dump and not args.selftest:
        p.error("nothing to do: pass --dump, --selftest or --figs")
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "roofline.csv")
    write_csv(all_rows, csv_path)
    written = []
    if figs:
        import matplotlib

        matplotlib.use("Agg")
        tag = f"M{args.M}"
        op_names = ", ".join(op.name for op in ops)
        shape_title = (
            f"MiniMax-H3 transformer block, 15 s / 768P / 16:9 (M = {args.M} rows per device, TP = 4, SP = 8)"
            if args.M == M_15S_768P_16_9
            else f"MiniMax-H3 transformer block, M = {args.M} rows per device (TP = 4)"
        )
        if args.no_measured or not measured:
            measured_note = "no measurement"
        elif args.measured_csv:
            measured_note = f"measured, best swept blocking ({os.path.basename(args.measured_csv)})"
        else:
            measured_note = "measured, shipped blocking (MiniMaxH3_wormhole_perf.md)"
        if "roofline" in figs:
            fig = fig_roofline(
                rows_by_arch[wh.short],
                wh,
                args.fidelity,
                f"Roofline on the Wormhole Galaxy, {op_names} — {shape_title}",
                measured_note,
            )
            path = os.path.join(args.out_dir, f"roofline_wh_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if "bars" in figs:
            fig = fig_time_bars(
                rows_by_arch, args.fidelity, f"Best-case time per resource, {op_names} — {shape_title}", measured_note
            )
            path = os.path.join(args.out_dir, f"time_bars_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if "stacked" in figs:
            fig = fig_stacked(
                rows_by_arch,
                args.fidelity,
                f"{op_names} per block, stacked — {shape_title.replace('MiniMax-H3 transformer block, ', '')}",
                measured_note,
            )
            path = os.path.join(args.out_dir, f"stacked_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if block_ops and "block_stacked" in figs:
            fig = fig_block_stacked(
                block_ops,
                wh,
                args.fidelity,
                f"Roofline vs measured, every op — {shape_title}",
                os.path.basename(args.profile_csv),
            )
            path = os.path.join(args.out_dir, f"block_stacked_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if block_ops and "block_ops" in figs:
            fig = fig_block_ops(
                block_ops,
                wh,
                args.fidelity,
                f"Per-op roofline vs measured — {shape_title}",
                os.path.basename(args.profile_csv),
            )
            path = os.path.join(args.out_dir, f"block_ops_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if "nstar" in figs:
            fig = fig_nstar(
                arches, ops, args.fidelity, "Regime crossover N* vs link count — Wormhole vs Blackhole Galaxy"
            )
            path = os.path.join(args.out_dir, "nstar_links.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
    for path in written + [csv_path]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
