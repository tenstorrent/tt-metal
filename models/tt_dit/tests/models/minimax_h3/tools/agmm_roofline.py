# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Roofline for the MiniMax-H3 AGMMs on the Wormhole 4x8 Galaxy, 15 s / 768P / 16:9.

AGMM = `ttnn.experimental.all_gather_minimal_matmul_async`, the fused TP all-gather + matmul every
column-parallel linear in the H3 DiT runs on (`models/tt_dit/layers/linear.py`). One transformer block
has exactly three: to_qkv, to_out and ff1 (fused SwiGLU); ff2 is a plain matmul + reduce-scatter and is
only included with `--include-ff2`. 50 blocks (+2 token-refiner blocks at M=64) per forward, 49
forwards per video. Not a test; pytest leaves it alone. Host-only: numpy + matplotlib, no ttnn/torch.

    python models/tt_dit/tests/models/minimax_h3/tools/agmm_roofline.py --dump
    python models/tt_dit/tests/models/minimax_h3/tools/agmm_roofline.py --figs all --measured-csv sweep_results_mm.csv
    python models/tt_dit/tests/models/minimax_h3/tools/agmm_roofline.py --selftest

Model (ported unchanged from `origin/cglagovich/agmm_analysis:agmm/roofline_lib.py`, the source of the
"AGMM Regime & Block Tuning" and "AGMM shape database" artifacts), per device, bf16, ring R = TP,
L links per direction, K_gathered = K_local * R:

    FLOPs          = 2 * M * K_gathered * N
    bytes_dram     = 2 * (M * K_gathered + K_gathered * N)     gathered in0 + resident in1
    bytes_per_link = (R - 1) * (2 * M * K_local) / (2 * L)     bidirectional ring split over L links
    peak_flops     = cores * (4096 / fidelity_cycles) * clock  4096 FLOP/cycle/core at LoFi, HiFi2 = 2048
    ideal          = max(FLOPs/peak, bytes_dram/DRAM_BW, bytes_per_link/LINK_BW), limiter = argmax
    N*             = peak / (2 * LINK_BW) * (R - 1) / (R * L)  compute<->fabric crossover, M-independent

Every constant in those artifacts is Blackhole Galaxy (12x9 = 108-core AGMM grid at 1.35 GHz =
298.6 TFLOP/s HiFi2, 512 GB/s DRAM, 2 links x 25 GB/s). The Wormhole Galaxy the model ships on
(`MESH_4X8_RING_WH` = `4x8nl4`, TP=4 / SP=8, Ring, 4 links) is a different part on every axis; the
`WH` constants below are each traced to the repo file that defines them, and the `BH` constants are
kept verbatim, labelled as inherited, for the side-by-side. Note the two fabrics coincide on aggregate
ingress (4 x 12.5 = 2 x 25 = 50 GB/s per direction), so the fabric bars are identical across arches.

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
from dataclasses import dataclass, replace

FIDELITY_CYCLES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
LOFI_FLOP_PER_CYCLE_PER_CORE = 4096  # 8x16 x 16x16 per cycle, 2*8*16*16 -- matmul_device_operation.cpp:2791
BF16_BYTES = 2
TILE = 32


@dataclass(frozen=True)
class Arch:
    name: str
    short: str
    clock_hz: float
    agmm_cores: int  # matmul worker grid of the AGMM (one row/column reserved for the CCL muxes)
    agmm_grid: tuple[int, int]
    full_cores: int  # full compute grid, what a plain matmul (ff2) runs on
    dram_bw: float  # bytes/s
    link_bw: float  # bytes/s per unidirectional ethernet link
    num_links: int  # links per direction the model's mesh config uses
    ring_size: int  # TP factor
    l1_bytes: int
    source: str

    def peak_flops(self, fidelity: str = "HiFi2", cores: int | None = None) -> float:
        cores = self.agmm_cores if cores is None else cores
        return cores * (LOFI_FLOP_PER_CYCLE_PER_CORE / FIDELITY_CYCLES[fidelity]) * self.clock_hz

    def n_star(self, fidelity: str = "HiFi2", num_links: int | None = None) -> float:
        """Output width above which the AGMM is compute-bound rather than fabric-bound (any M)."""
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
    agmm_cores=64,  # 8x8: agmm_worker_grid reserves the bottom row of the 8x9 grid (utils/matmul.py:407)
    agmm_grid=(8, 8),
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
    agmm_cores=108,  # 12x9
    agmm_grid=(12, 9),
    full_cores=120,  # 12x10 -- sdpa_perf_utils.py:69 GALAXY_GRID
    dram_bw=512e9,
    link_bw=25e9,  # 400 Gbps nominal, "currently limited to half BW" -- ccl_common.cpp:2132
    num_links=2,  # BH_GALAXY channels count 2 -- tools/scaleout/generate_mgd/generate_mgd.cpp
    ring_size=4,
    l1_bytes=1536 * 1024,  # blackhole_140_arch.yaml:109 (the artifacts budget 1400 KB of it for CBs)
    source="origin/cglagovich/agmm_analysis:agmm/roofline_lib.py (298.6 TFLOP/s, 512 GB/s, 25 GB/s/link)",
)


@dataclass(frozen=True)
class Op:
    name: str
    K: int  # K after the gather (= per-device K for the plain ff2 matmul)
    N: int  # per-device output width
    fusion: str
    kind: str = "agmm"  # "agmm" | "mm+rs"
    marker: str = "o"


# The three AGMMs of one transformer block, per device at TP=4 (agmm_config.py, attention_minimax_h3.py).
# hidden 5376, inner 7168 (56 heads x 128), ffn 14336: qkv N = 3*inner/4, out N = hidden/4, ff1 N = 2*ffn/4.
OPS = [
    Op("to_qkv", 5376, 5376, "chunks=3", marker="o"),
    Op("to_out", 7168, 1344, "addcmul", marker="s"),
    Op("ff1", 5376, 7168, "SwiGLU", marker="^"),
]
FF2 = Op("ff2", 3584, 5376, "MM + reduce-scatter", kind="mm+rs", marker="D")

# 15 s / 768P / 16:9: 1344x768, 362 frames -> 107 latent frames x 24x42 patches = 107856 video rows
# + 603 audio latents x 2 channels + 39 text tokens = 109101, padded to SP*TILE*... = 109312
# (`packing.padded_sequence_length`), 13664 rows per device at SP=8 (MiniMaxH3_wormhole_perf.md:129).
M_15S_768P_16_9 = 13664
M_REFINER = 64  # the 2 token-refiner blocks run the same three (K, N) over the 39-token text stream
BLOCKS_PER_FORWARD = 50
REFINER_BLOCKS_PER_FORWARD = 2
FORWARDS_PER_VIDEO = 49

# Shipped blockings on the WH 8x8 grid at M=13664, HiFi2 (MiniMaxH3_wormhole_perf.md:365-371).
MEASURED_US_WH_15S = {"to_qkv": 10401.8, "to_out": 4332.8, "ff1": 15709.9, "ff2": 6770.7}
SWEEP_USE_CASE_TO_OP = {"qkv": "to_qkv", "plain": "to_out", "ff1_swiglu": "ff1", "ff2": "ff2"}

RESOURCES = ("compute", "dram", "fabric")
RESOURCE_LABEL = {"compute": "compute", "dram": "DRAM", "fabric": "fabric"}
RESOURCE_COLOR = {"compute": "#2a78d6", "dram": "#eb6834", "fabric": "#1baf7a"}
# ops get the next three categorical slots so an op colour is never mistaken for a resource colour
OP_COLOR = {"to_qkv": "#eda100", "to_out": "#e87ba4", "ff1": "#008300", "ff2": "#8a8983"}
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
        cores = arch.agmm_cores
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
    agmm = [r for r in rows if r.op.kind == "agmm" and r.M != M_REFINER]
    if agmm:
        ideal_block = sum(r.ideal for r in agmm)
        line = f"AGMM ideal per block {ideal_block * 1e3:.2f} ms -> {ideal_block * BLOCKS_PER_FORWARD * 1e3:.0f} ms per forward ({BLOCKS_PER_FORWARD} blocks)"
        if all(r.measured is not None for r in agmm):
            meas_block = sum(r.measured for r in agmm)
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
            "AGMM matmul grid",
            lambda a: f"{a.agmm_grid[0]}x{a.agmm_grid[1]} = {a.agmm_cores} cores (full {a.full_cores})",
        ),
        (f"peak @ {fidelity}, AGMM grid", lambda a: f"{a.peak_flops(fidelity) / 1e12:.1f} TFLOP/s"),
        ("peak @ LoFi, AGMM grid", lambda a: f"{a.peak_flops('LoFi') / 1e12:.1f} TFLOP/s"),
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
        f"{arch.short}: {arch.agmm_cores} cores x {LOFI_FLOP_PER_CYCLE_PER_CORE // FIDELITY_CYCLES[fidelity]} FLOP/cycle x "
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
        f"Hollow diamond = {measured_note}. Fabric bars coincide across arches: both rings ingest 50 GB/s per direction.",
        fontsize=8,
        color=INK_MUTED,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=3.0)
    return fig


def fig_stacked(rows_by_arch: dict[str, list[Roofline]], fidelity: str, title: str, measured_note: str):
    """One stacked bar per arch: the three ops' ideal times summed, colour-coded by op, with the summed
    measured wall time marked on top (and, where measured, a second stack of the measured times)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    _style(ax, grid_axis="y")
    width = 0.55
    columns = []  # (x, label, [(op, seconds)], hatch, sublabel)
    x = 0.0
    for key, rows in rows_by_arch.items():
        agmm = [r for r in rows if r.op.kind == "agmm" and r.M != M_REFINER]
        arch = agmm[0].arch
        columns.append(
            (x, f"{arch.short} ideal", [(r.op.name, r.ideal) for r in agmm], None, _constants_line(arch, fidelity))
        )
        x += 1.0
        if all(r.measured is not None for r in agmm):
            columns.append((x, f"{arch.short} measured", [(r.op.name, r.measured) for r in agmm], "//", None))
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
                color=OP_COLOR[op_name],
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
    ax.set_ylabel("time per transformer block, ms (three AGMM calls)", fontsize=9, color=INK_2)
    handles = [Patch(color=OP_COLOR[name], label=name) for name in ("to_qkv", "to_out", "ff1")]
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
        "Ideal = each op's roofline lower bound (its binding resource at 100% of peak), summed over the three AGMMs of one "
        f"block.\nMeasured = {measured_note}.",
        fontsize=8,
        color=INK_MUTED,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    return fig


def fig_nstar(arches: list[Arch], ops: list[Op], fidelity: str, title: str):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _style(ax)
    ax.set_yscale("log")
    ticks = [800, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(800, 9000)
    links = [1, 2, 3, 4]
    styles = {"WH": dict(color=INK, ls="-", marker="o"), "BH": dict(color=INK_MUTED, ls=(0, (4, 2)), marker="s")}
    for arch in arches:
        ns = [arch.n_star(fidelity, L) for L in links]
        st = styles.get(arch.short, dict(color=INK_2, ls=":", marker="^"))
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
    r = roofline(1216, Op("ltx", 4096, 8, "—"), BH, "HiFi2")
    close(r.t_fabric * 1e6, 74.7, 0.05)
    close(r.t_dram * 1e6, 19.6, 0.05)
    close(r.t_compute * 1e6, 0.27, 0.01)
    # ltx_m4864_k4096_n3072_c3: t_compute 409.9, t_dram 127.0, t_fabric 298.8 -> compute.
    r = roofline(4864, Op("ltx", 4096, 3072, "chunks3"), BH, "HiFi2")
    close(r.t_compute * 1e6, 409.9, 0.1)
    close(r.t_dram * 1e6, 127.0, 0.1)
    close(r.t_fabric * 1e6, 298.8, 0.1)
    assert r.limiter == "compute"
    close(BH.peak_flops("HiFi2") / 1e12, 298.6, 0.05)
    close(BH.n_star("HiFi2", 1), 4479, 1)  # artifact N* table, ring 4 / 1 link
    # Wormhole anchors (MiniMaxH3_wormhole_perf.md roofline section: ff1 1.05 T, 8.0 ms at 64 cores).
    close(WH.peak_flops("HiFi2") / 1e12, 131.07, 0.01)
    r = roofline(M_15S_768P_16_9, OPS[2], WH, "HiFi2", measured_us=MEASURED_US_WH_15S["ff1"])
    close(r.flops / 1e12, 1.053, 0.001)
    close(r.t_compute * 1e6, 8034.4, 0.5)
    close(r.t_dram * 1e6, 777.7, 0.5)
    close(r.t_fabric * 1e6, 1101.9, 0.5)
    assert r.limiter == "compute"
    close(r.util("compute"), 0.511, 0.002)
    close(WH.n_star("HiFi2"), 983, 1)
    # to_out: compute-bound on WH at 4 links, fabric-bound on BH at 2 links.
    assert roofline(M_15S_768P_16_9, OPS[1], WH).limiter == "compute"
    assert roofline(M_15S_768P_16_9, OPS[1], BH).limiter == "fabric"
    print("selftest OK")


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--figs", default="all", help="comma list of roofline,bars,stacked,nstar or all (default) or none")
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
    p.add_argument("--include-ff2", action="store_true", help="add ff2 (plain matmul + reduce-scatter, not an AGMM)")
    p.add_argument("--include-refiner", action="store_true", help="add the token-refiner AGMMs at M=64")
    p.add_argument("--no-bh", action="store_true", help="drop the Blackhole side of the comparison")
    p.add_argument(
        "--measured-csv",
        default=None,
        help="sweep_mm_block_sizes.py results CSV; best OK time per op replaces the shipped numbers",
    )
    p.add_argument("--no-measured", action="store_true", help="do not overlay measured times")
    p.add_argument("--out-dir", default="agmm_roofline_out")
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

    ops = list(OPS) + ([FF2] if args.include_ff2 else [])
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
            rows += [roofline(M_REFINER, op, arch, args.fidelity) for op in OPS]
        rows_by_arch[arch.short] = rows

    all_rows = [r for rows in rows_by_arch.values() for r in rows]
    if args.dump:
        print(constants_table(arches, args.fidelity))
        print()
        for key, rows in rows_by_arch.items():
            print(dump_table(rows, f"{rows[0].arch.name} — per device, M = {args.M}, {args.fidelity}"))
            print()

    figs = set() if args.figs == "none" else set(args.figs.split(","))
    if "all" in figs:
        figs = {"roofline", "bars", "stacked", "nstar"}
    if not figs and not args.dump and not args.selftest:
        p.error("nothing to do: pass --dump, --selftest or --figs")
    if figs:
        import matplotlib

        matplotlib.use("Agg")
        os.makedirs(args.out_dir, exist_ok=True)
        write_csv(all_rows, os.path.join(args.out_dir, "agmm_roofline.csv"))
        tag = f"M{args.M}"
        shape_title = (
            f"MiniMax-H3 AGMMs, 15 s / 768P / 16:9 (M = {args.M} rows per device, TP = 4, SP = 8)"
            if args.M == M_15S_768P_16_9
            else f"MiniMax-H3 AGMMs, M = {args.M} rows per device (TP = 4)"
        )
        if args.no_measured or not measured:
            measured_note = "no measurement"
        elif args.measured_csv:
            measured_note = f"measured, best swept blocking ({os.path.basename(args.measured_csv)})"
        else:
            measured_note = "measured, shipped blocking (MiniMaxH3_wormhole_perf.md)"
        written = []
        if "roofline" in figs:
            fig = fig_roofline(
                rows_by_arch[wh.short],
                wh,
                args.fidelity,
                f"Roofline on the Wormhole Galaxy — {shape_title}",
                measured_note,
            )
            path = os.path.join(args.out_dir, f"agmm_roofline_wh_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if "bars" in figs:
            fig = fig_time_bars(
                rows_by_arch, args.fidelity, f"Best-case time per resource — {shape_title}", measured_note
            )
            path = os.path.join(args.out_dir, f"agmm_time_bars_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if "stacked" in figs:
            fig = fig_stacked(
                rows_by_arch,
                args.fidelity,
                f"Three AGMMs per block, stacked — {shape_title.replace('MiniMax-H3 AGMMs, ', '')}",
                measured_note,
            )
            path = os.path.join(args.out_dir, f"agmm_stacked_{tag}.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        if "nstar" in figs:
            fig = fig_nstar(
                arches, ops, args.fidelity, "Regime crossover N* vs link count — Wormhole vs Blackhole Galaxy"
            )
            path = os.path.join(args.out_dir, "agmm_nstar_links.png")
            fig.savefig(path, dpi=args.dpi)
            written.append(path)
        for path in written + [os.path.join(args.out_dir, "agmm_roofline.csv")]:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
