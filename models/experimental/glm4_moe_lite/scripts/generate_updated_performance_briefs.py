#!/usr/bin/env python3
"""Generate the July 2026 GLM-4.7 Flash/REAP performance briefs and summary image."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


FLASH_PDF = Path("/home/tt-admin/sdawle/glm47_flash_wh_glx/GLM-4.7-Flash_Performance_Brief_Final_20260724.pdf")
REAP_PDF = Path("/home/tt-admin/sdawle/glm47_reap_218b/GLM-4.7-REAP-218B_Performance_Brief_Final_20260724.pdf")
OPT_IMAGE = Path("/home/tt-admin/sdawle/glm47_flash_wh_glx/GLM-4.7_Optimization_Journey.png")
FLASH_MINOR_PDF = Path(
    "/home/tt-admin/sdawle/glm47_flash_wh_glx/GLM-4.7-Flash_Performance_Brief_Minor_Update_v2_20260724.pdf"
)

BLUE = "#2468b4"
ORANGE = "#d27600"
GREEN = "#17865b"
RED = "#c73a3a"
PURPLE = "#7552a8"
DARK = "#17202a"
MID = "#536270"
LIGHT = "#eef3f7"
GRID = "#d8e0e7"


def _page(title: str, subtitle: str, page_no: int, total_pages: int):
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.945), 1, 0.055, color=BLUE, transform=ax.transAxes))
    ax.text(0.055, 0.968, title, color="white", fontsize=16, fontweight="bold", va="center")
    ax.text(0.055, 0.925, subtitle, color=MID, fontsize=9, va="top")
    ax.text(0.945, 0.025, f"{page_no}/{total_pages}", color=MID, fontsize=8, ha="right")
    ax.text(0.055, 0.025, "Wormhole Galaxy decode performance brief · updated 2026-07-24", color=MID, fontsize=8)
    return fig, ax


def _section(ax, y: float, title: str, color: str = BLUE) -> float:
    ax.text(0.055, y, title, fontsize=12, fontweight="bold", color=color, va="top")
    ax.plot([0.055, 0.945], [y - 0.012, y - 0.012], color=GRID, lw=1)
    return y - 0.035


def _paragraph(ax, y: float, text: str, width: int = 108, size: float = 9, color: str = DARK) -> float:
    wrapped = textwrap.fill(text, width=width)
    ax.text(0.055, y, wrapped, fontsize=size, color=color, va="top", linespacing=1.35)
    return y - 0.023 * (wrapped.count("\n") + 1) - 0.012


def _metrics(ax, y: float, metrics: list[tuple[str, str, str]], height: float = 0.105) -> float:
    gap = 0.012
    width = (0.89 - gap * (len(metrics) - 1)) / len(metrics)
    for i, (label, value, note) in enumerate(metrics):
        x = 0.055 + i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, y - height),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.008",
                facecolor=LIGHT,
                edgecolor=GRID,
                linewidth=0.8,
            )
        )
        ax.text(x + 0.012, y - 0.024, label, fontsize=8, color=MID, va="top")
        ax.text(x + 0.012, y - 0.052, value, fontsize=15, color=BLUE, fontweight="bold", va="top")
        ax.text(x + 0.012, y - 0.082, note, fontsize=7.2, color=MID, va="top")
    return y - height - 0.025


def _table(
    ax,
    y: float,
    headers: list[str],
    rows: list[list[str]],
    widths: list[float] | None = None,
    font_size: float = 7.7,
    row_height: float = 0.035,
) -> float:
    ncols = len(headers)
    if widths is None:
        widths = [0.89 / ncols] * ncols
    scale = 0.89 / sum(widths)
    widths = [w * scale for w in widths]
    x0 = 0.055
    x_positions = [x0]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width)
    ax.add_patch(plt.Rectangle((x0, y - row_height), 0.89, row_height, color=DARK))
    for x, header in zip(x_positions, headers):
        ax.text(
            x + 0.007, y - row_height / 2, header, color="white", fontsize=font_size, fontweight="bold", va="center"
        )
    current_y = y - row_height
    for row_idx, row in enumerate(rows):
        current_y -= row_height
        ax.add_patch(
            plt.Rectangle(
                (x0, current_y),
                0.89,
                row_height,
                color="white" if row_idx % 2 else LIGHT,
                ec=GRID,
                lw=0.4,
            )
        )
        for x, value in zip(x_positions, row):
            ax.text(x + 0.007, current_y + row_height / 2, str(value), color=DARK, fontsize=font_size, va="center")
    return current_y - 0.025


def _bullet_list(ax, y: float, items: list[str], color: str = DARK, size: float = 8.5) -> float:
    for item in items:
        wrapped = textwrap.wrap(item, width=103)
        ax.text(0.065, y, "•", color=BLUE, fontsize=10, va="top")
        ax.text(0.083, y, "\n".join(wrapped), color=color, fontsize=size, va="top", linespacing=1.3)
        y -= 0.022 * len(wrapped) + 0.012
    return y


def generate_flash_pdf() -> None:
    FLASH_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(FLASH_PDF) as pdf:
        fig, ax = _page(
            "GLM-4.7-Flash — Decode Performance",
            "Current measured winner, placement-aware roofline, and next optimizations",
            1,
            4,
        )
        y = 0.885
        y = _metrics(
            ax,
            y,
            [
                ("B1 @ ISL=128", "51.3 ms", "19.49 tok/s · traced steady decode"),
                ("Historical baseline", "74.8 ms", "13.37 tok/s · published Galaxy baseline"),
                ("Latency improvement", "31.4%", "+45.8% throughput vs baseline"),
                ("Peak aggregate", "591.5 tok/s", "B128 @ ISL=128"),
            ],
        )
        y = _section(ax, y, "Executive update")
        y = _paragraph(
            ax,
            y,
            "The validated optimized_v1 stack reduces B1 decode from 74.8 ms to 51.3 ms at ISL=128. "
            "The same sweep reaches 449.4 aggregate tok/s at B32 and 591.5 tok/s at B128. These numbers use "
            "BF16 KV cache; BF8 KV remains experimental because end-to-end greedy output diverged from BF16.",
        )
        y = _section(ax, y, "Placement-aware active-byte calculation")
        y = _table(
            ax,
            y,
            ["Quantity", "Calculation", "Result"],
            [
                ["Dense/non-routed", "2.160B / 29.943B", "7.2% parameters"],
                ["Routed experts", "27.783B / 29.943B", "92.8% parameters"],
                ["Resident experts", "64 experts / 32 ASICs", "2 experts/chip"],
                ["Dense scan (BF8 winner)", "(2.160B − 0.317B) × 1 B", "1.843 GB/ASIC/token"],
                ["One BF8 expert", "27.783 GB / 64", "0.434 GB across 47 layers"],
                ["Ideal critical payload", "1.843 + 0.434", "2.277 GB/token"],
                ["Two-expert collision", "1.843 + 2×0.434", "2.711 GB/token"],
            ],
            widths=[0.26, 0.38, 0.25],
        )
        y = _section(ax, y, "Roofline interpretation")
        _bullet_list(
            ax,
            y,
            [
                "WH per-ASIC ceiling: 288 GB/s ÷ payload = 126.5 tok/s ideal or 106.2 tok/s with a two-expert collision.",
                "53% MoE heuristic: 67.0 tok/s ideal or 56.3 tok/s collision. This is a weight-only target, not an end-to-end guarantee.",
                "At 51.3 ms, payload-equivalent bandwidth is 44.4–52.8 GB/s, or 15.4–18.3% of the 288 GB/s ASIC peak.",
            ],
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-Flash — Measurements", "Validated optimized_v1 sweep and current profile interpretation", 2, 4
        )
        y = 0.885
        y = _section(ax, y, "B1 context-length sweep")
        y = _table(
            ax,
            y,
            ["ISL", "Mean latency", "Throughput", "Change vs ISL=128"],
            [
                ["128", "51.3 ms", "19.49 tok/s", "reference"],
                ["512", "52.5 ms", "19.05 tok/s", "+1.2 ms"],
                ["1K", "53.4 ms", "18.73 tok/s", "+2.1 ms"],
                ["2K", "53.9 ms", "18.55 tok/s", "+2.6 ms"],
                ["4K", "55.0 ms", "18.18 tok/s", "+3.7 ms"],
                ["8K", "57.2 ms", "17.48 tok/s", "+5.9 ms"],
                ["16K", "61.6 ms", "16.23 tok/s", "+10.3 ms"],
                ["32K", "70.6 ms", "14.16 tok/s", "+19.3 ms"],
                ["64K", "87.9 ms", "11.38 tok/s", "+36.6 ms"],
            ],
            widths=[0.16, 0.23, 0.25, 0.25],
            row_height=0.032,
        )
        y = _section(ax, y, "Batch scaling at ISL=128")
        y = _table(
            ax,
            y,
            ["Batch", "Aggregate tok/s", "Per-user tok/s", "Mean latency"],
            [
                ["1", "19.5", "19.49", "51.3 ms"],
                ["4", "73.3", "18.32", "54.6 ms"],
                ["8", "139.9", "17.48", "57.2 ms"],
                ["16", "256.8", "16.05", "62.3 ms"],
                ["32", "449.4", "14.04", "71.2 ms"],
                ["64", "529.8", "8.28", "120.8 ms"],
                ["128", "591.5", "4.62", "216.4 ms"],
            ],
            row_height=0.032,
        )
        y = _section(ax, y, "Profile-derived remaining cost")
        _paragraph(
            ax,
            y,
            "The prior 49.76 ms firmware profile contained 3,875 operations/device/token. Regular matmul, routing, "
            "attention/RoPE, collectives, normalization, and layout movement are serialized rather than one continuous "
            "DRAM stream. The GlobalCB plan estimates matmul at ~39.5% of device time and M=1 bandwidth efficiency near "
            "20%; these are profiler estimates and must be re-profiled against the 51.3 ms winner.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-Flash — Applied Optimizations",
            "Only validated/default winner-stack changes are credited to optimized_v1",
            3,
            4,
        )
        y = 0.885
        y = _section(ax, y, "Applied winner stack")
        y = _table(
            ax,
            y,
            ["Optimization", "Status in winner", "Measured evidence"],
            [
                ["Traced decode + device sampling", "Applied", "Basis of all sweep numbers"],
                ["BF8 dense weights", "Applied", "~7% B1 decode win"],
                ["Explicit matmul program config", "Default-on", "58.1 → 54.2 ms intermediate"],
                ["in0_block_w=8 tuning", "Applied", "54.2 → 53.4 ms intermediate"],
                ["Fused collective epilogue", "Default-on", "Correctness validated; bundle win"],
                ["Buffered MoE all-reduce", "Default-on", "Correctness validated; bundle win"],
                ["Down routing scale in sparse matmul", "Applied", "~0.9 ms measured"],
                ["Fused GLM router", "Applied for B1", "~0.8 ms measured"],
                ["Sharded/L1 RMSNorm", "Applied", "Winner bundle"],
                ["Fused QKV-A + shared gate/up", "Applied", "Winner bundle"],
                ["L1 activations, EP and router", "Applied", "Winner bundle"],
                ["4-link ring CCL + fused MLP/MoE reduce", "Applied", "Winner bundle"],
            ],
            widths=[0.36, 0.22, 0.31],
            font_size=7.3,
            row_height=0.034,
        )
        y = _section(ax, y, "Evaluated but not credited")
        _bullet_list(
            ax,
            y,
            [
                "Fused Q concat/transpose and singleton reshape were removed after regressions.",
                "Fused KV branch, trace 2CQ, LM-head sharding and DRAM-sharded attention are disabled in the winning sweep.",
                "BF8 KV cache passed kernel PCC/update tests but changed end-to-end greedy output; BF16 KV remains the accuracy default.",
                "GlobalCB prefetch infrastructure is ported but not wired into the hot path.",
            ],
            size=8.1,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-Flash — Next Optimization Plan", "Prioritized by measured headroom and implementation risk", 4, 4
        )
        y = 0.885
        y = _section(ax, y, "Recommended sequence")
        y = _table(
            ax,
            y,
            ["Priority", "Workstream", "Evidence / expected outcome"],
            [
                ["P0", "Re-profile the 51.3 ms winner", "Refresh op shares before attributing remaining gap"],
                ["P1", "Wire GlobalCB weight prefetch", "Targets matmul BW; high risk, several-ms hypothesis"],
                ["P2", "Fuse B1 KV update layout path", "Remove layout conversion before cache update"],
                ["P3", "Revisit expert gate/up fusion", "Implemented but absent from winning sweep"],
                ["P4", "Attention weight/layout sharding", "Potential BW win; trace stability unresolved"],
                ["P5", "KV-cache memory partitioning", "Unlock larger batch × ISL without BF8 quality risk"],
            ],
            widths=[0.12, 0.34, 0.43],
            row_height=0.047,
        )
        y = _section(ax, y, "Target ladder")
        y = _metrics(
            ax,
            y,
            [
                ("Current", "51.3 ms", "Validated optimized_v1"),
                ("Near milestone", "47–49 ms", "Prefetch/layout fusion gate"),
                ("Stretch", "43–45 ms", "Collective and layout consolidation"),
                ("Weight-only target", "15–18 ms", "BF8-dense model; not an E2E forecast"),
            ],
            height=0.11,
        )
        y = _section(ax, y, "Methodology cautions")
        _bullet_list(
            ax,
            y,
            [
                "The 53% factor is a heuristic inherited from GPT-OSS; it is not a measured Flash utilization constant.",
                "The updated roofline uses BF8 dense weights because optimized_v1 explicitly sets DENSE_TT_DTYPE=bf8; the prior BF16-dense roofline is not comparable.",
                "Active bytes and bandwidth must use the same scope. Galaxy aggregate bandwidth divided by logical per-token bytes is invalid.",
                "Optimization gains are not additive; several changes only win as a coherent bundle.",
                "Source of truth for current performance: sweep_isl_batch_complete_20260724/sweep_results.csv.",
            ],
        )
        pdf.savefig(fig)
        plt.close(fig)


def generate_flash_minor_update_pdf() -> None:
    """Preserve the original three-page brief structure with minimal updates."""
    with PdfPages(FLASH_MINOR_PDF) as pdf:
        fig, ax = _page(
            "GLM-4.7-Flash — Throughput Methodology",
            "Current baseline: 51.3 ms/token (19.49 tok/s/user), B1 decode, 32-chip Wormhole Galaxy.",
            1,
            3,
        )
        y = 0.885
        y = _section(ax, y, "1. Start with the same placement-aware roofline")
        y = _paragraph(
            ax,
            y,
            "Hardware ceiling = peak per-ASIC DRAM bandwidth / critical-ASIC active bytes. Practical MoE target = "
            "hardware ceiling × 0.53. Measured effective bandwidth = critical-path bytes / measured latency. "
            "Bandwidth and bytes must use the same hardware scope.",
        )
        y = _section(ax, y, "2. Convert model composition into runtime bytes")
        y = _table(
            ax,
            y,
            ["Quantity", "Calculation", "Result"],
            [
                ["Dense / non-routed", "2.160B / 29.943B", "7.2% of parameters"],
                ["Routed experts", "27.783B / 29.943B", "92.8% of parameters"],
                ["Resident experts", "64 experts / 32 ASICs", "2 experts per chip"],
                ["Dense scan — optimized_v1 BF8", "(2.160B − 0.317B) × 1 B", "1.843 GB/ASIC/token"],
                ["One BF8 expert", "27.783 GB / 64", "0.434 GB across 47 layers"],
                ["Top-4 aggregate expert bytes", "27.783 GB × 4/64", "1.736 GB across selected ASICs"],
            ],
            widths=[0.29, 0.36, 0.24],
            font_size=7.2,
            row_height=0.028,
        )
        y = _section(ax, y, "3. Map active experts to the busiest ASIC")
        y = _table(
            ax,
            y,
            ["Placement case", "Critical payload", "Payload mix"],
            [
                ["Ideal: one selected expert", "1.843 + 0.434 = 2.277 GB", "80.9% dense / 19.1% expert"],
                ["Collision: two selected experts", "1.843 + 2×0.434 = 2.711 GB", "68.0% dense / 32.0% expert"],
            ],
            widths=[0.34, 0.31, 0.24],
            font_size=7.3,
            row_height=0.034,
        )
        y = _section(ax, y, "4. Compute ceiling, practical target, and current utilization")
        y = _table(
            ax,
            y,
            ["Metric", "Ideal placement", "Two-expert collision"],
            [
                ["Hardware ceiling", "126.5 tok/s (7.9 ms)", "106.2 tok/s (9.4 ms)"],
                ["53% MoE heuristic", "67.0 tok/s (14.9 ms)", "56.3 tok/s (17.8 ms)"],
                ["Current effective bandwidth", "44.4 GB/s", "52.8 GB/s"],
                ["Current peak utilization", "15.4%", "18.3%"],
                ["Current throughput", "19.49 tok/s", "19.49 tok/s"],
            ],
            widths=[0.32, 0.285, 0.285],
            font_size=7.2,
            row_height=0.030,
        )
        _paragraph(
            ax,
            y,
            "Interpretation: 15–18 ms is a BF8-dense, weight-only heuristic. It excludes routing, CCL, norms, "
            "attention, layouts, dispatch, and synchronization. The previous 27–30 ms calculation assumed BF16 "
            "dense weights and is not the correct roofline for optimized_v1.",
            size=8.3,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-Flash — Current Profile and MoE Tax",
            "Original profile structure retained; completed optimizations are marked explicitly.",
            2,
            3,
        )
        y = 0.885
        y = _section(ax, y, "Why dense-style bandwidth efficiency becomes lower for MoE")
        y = _table(
            ax,
            y,
            ["Factor", "Estimated loss", "Flash evidence"],
            [
                ["Small expert GEMMs", "−12 to −22 pts", "M=1 work padded to tile execution"],
                ["Multiple expert paths", "−2 to −4 pts", "Top-4 gate/up/down execution"],
                ["Router + coordination", "−3 to −5 pts", "Top-k, gather, remap, scatter"],
                ["Irregular expert reads", "−2 to −4 pts", "Runtime-selected expert blocks"],
                ["Scheduling bubbles", "−0.5 to −1 pt", "47 serial layers and many kernels"],
                ["53% combined heuristic", "−26 to −28 pts", "Structural estimate, not measured bound"],
            ],
            widths=[0.28, 0.22, 0.39],
            font_size=7.1,
            row_height=0.032,
        )
        y = _section(ax, y, "Most recent detailed profile available")
        y = _table(
            ax,
            y,
            ["Operation group", "Profile time", "Current interpretation"],
            [
                ["Regular matmul", "18.57 ms", "Largest prior bucket; re-profile 51.3 ms winner"],
                ["TopK + gather + remap", "6.55 ms", "Fused B1 router now applied"],
                ["SDPA + RoPE", "4.07 ms", "Q concat fusion regressed; KV-layout work remains"],
                ["All-reduce + fast reduce", "3.10 ms", "Fused epilogue + buffered AR now applied"],
                ["LayerNorm", "2.72 ms", "Sharded/L1 norm now applied"],
                ["Sparse expert matmul", "0.93 ms", "Down routing scale now folded in"],
                ["Layout / movement", "~10.4 ms", "Still fragmented; profile predates final winner"],
            ],
            widths=[0.30, 0.19, 0.40],
            font_size=7.0,
            row_height=0.034,
        )
        y = _section(ax, y, "Optimizations applied since the original brief")
        _bullet_list(
            ax,
            y,
            [
                "BF8 dense weights, explicit matmul program configuration, and in0_block_w=8 tuning.",
                "Fused collective epilogue, buffered MoE all-reduce, and fused MLP/MoE reduction.",
                "Routing scale folded into sparse down-projection and fused GLM router for B1.",
                "Sharded/L1 RMSNorm, L1 decode/EP/router tensors, fused QKV-A, and fused shared gate/up.",
                "BF16 KV cache retained; BF8 KV is not enabled because end-to-end greedy output diverged.",
            ],
            size=8.2,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-Flash — Targets and Workstreams",
            "Same milestone format as the original brief, updated for optimized_v1.",
            3,
            3,
        )
        y = 0.885
        y = _section(ax, y, "Throughput summary")
        y = _table(
            ax,
            y,
            ["Configuration", "Latency", "Throughput", "Status / assumption"],
            [
                ["Historical published baseline", "74.8 ms", "13.37 tok/s", "Galaxy B1 @ ISL=128"],
                ["Current optimized_v1", "51.3 ms", "19.49 tok/s", "Validated traced steady decode"],
                ["WH ceiling — ideal", "7.9 ms", "126.5 tok/s", "BF8 dense + BF8 experts"],
                ["WH ceiling — collision", "9.4 ms", "106.2 tok/s", "Two selected experts on ASIC"],
                ["53% target — ideal", "14.9 ms", "67.0 tok/s", "Weight-only heuristic"],
                ["53% target — collision", "17.8 ms", "56.3 tok/s", "Conservative weight-only heuristic"],
            ],
            widths=[0.31, 0.18, 0.20, 0.30],
            font_size=7.0,
            row_height=0.034,
        )
        y = _section(ax, y, "Updated workstreams")
        y = _table(
            ax,
            y,
            ["#", "Workstream", "Status / evidence", "Milestone"],
            [
                ["1", "Fuse router selection", "Completed; ~0.8 ms measured", "Included in 51.3 ms"],
                ["2", "Fold sparse-down scale", "Completed; ~0.9 ms measured", "Included in 51.3 ms"],
                ["3", "Collective epilogue + buffered AR", "Completed and default-on", "Included in 51.3 ms"],
                ["4", "Pre-SDPA/Q concat fusion", "Attempted; regressed and removed", "No credit"],
                ["5", "Fuse KV-update layout", "Current implementation task", "B1 ≤49 ms gate"],
                ["6", "GlobalCB weight prefetch", "Ported, not wired; high risk", "Stretch: 43–47 ms"],
                ["7", "KV-cache partitioning", "Needed for high batch × ISL", "Capacity, not B1 latency"],
            ],
            widths=[0.07, 0.30, 0.34, 0.28],
            font_size=6.8,
            row_height=0.036,
        )
        y = _section(ax, y, "Bottom line")
        y = _paragraph(
            ax,
            y,
            "Optimized_v1 improves latency by 31.4% and throughput by 45.8% versus the published baseline. "
            "The next defensible milestone is 47–49 ms after fresh profiling and KV-layout/prefetch work. "
            "The 56–67 tok/s BF8-dense roofline remains a weight-only analytical target, not an immediate "
            "end-to-end forecast.",
        )
        _bullet_list(
            ax,
            y,
            [
                "Assumptions: WH 288 GB/s/ASIC; dense BF8; experts BF8; KV cache BF16; top-4/64; MTP disabled.",
                "Source of current measurements: sweep_isl_batch_complete_20260724/sweep_results.csv.",
            ],
            size=8.2,
        )
        pdf.savefig(fig)
        plt.close(fig)


def generate_reap_pdf() -> None:
    REAP_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(REAP_PDF) as pdf:
        fig, ax = _page(
            "GLM-4.7-REAP-218B — Decode Performance",
            "Validated winner configuration, placement-aware roofline, and next optimizations",
            1,
            4,
        )
        y = 0.885
        y = _metrics(
            ax,
            y,
            [
                ("B1 @ ISL=128", "129.0 ms", "7.75 tok/s · traced steady decode"),
                ("Pre-combo baseline", "143.2 ms", "~6.98 tok/s · same protocol"),
                ("Latency improvement", "9.9%", "~14.2 ms saved"),
                ("B8 winner", "57.4 tok/s", "139.4 ms/user · aggregate derived"),
            ],
        )
        y = _section(ax, y, "Executive update")
        y = _paragraph(
            ax,
            y,
            "The confirmed REAP winner reduces B1 decode from approximately 143 ms to 129 ms through a coherent "
            "four-knob configuration: L1 normalization, LoFi sparse MoE, fused expert gate/up, and LoFi attention. "
            "The gain is synergistic and must not be represented as the sum of isolated knob measurements.",
        )
        y = _section(ax, y, "Placement-aware active-byte calculation")
        y = _table(
            ax,
            y,
            ["Quantity", "Calculation", "Result"],
            [
                ["Dense/non-routed", "16.805B / 218.383B", "7.7% parameters"],
                ["Routed experts", "201.578B / 218.383B", "92.3% parameters"],
                ["Resident experts", "96 experts / 32 ASICs", "3 experts/chip"],
                ["Dense scan per ASIC", "(16.805B − 0.776B)×2 / TP8", "4.007 GB/token"],
                ["One BF4 expert", "(201.578 GB / 96) / 2", "1.050 GB across 92 layers"],
                ["BF4 ideal payload", "4.007 + 1.050", "5.057 GB/token"],
                ["BF4 collision payload", "4.007 + 2×1.050", "6.107 GB/token"],
            ],
            widths=[0.27, 0.38, 0.24],
        )
        y = _section(ax, y, "Roofline interpretation")
        _bullet_list(
            ax,
            y,
            [
                "BF4 weight-only ceiling: 56.9 tok/s ideal or 47.2 tok/s with a two-expert collision.",
                "53% heuristic target: 30.2 tok/s ideal or 25.0 tok/s collision.",
                "At 129 ms, payload-equivalent bandwidth is 39.2–47.3 GB/s, or 13.6–16.4% of ASIC peak.",
            ],
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-REAP-218B — Measurements", "Confirmed A/B results and corrected profile interpretation", 2, 4
        )
        y = 0.885
        y = _section(ax, y, "Validated decode measurements")
        y = _table(
            ax,
            y,
            ["Configuration", "B1 mean", "Throughput", "Status"],
            [
                ["A/B baseline", "143.2–143.3 ms", "~6.98 tok/s", "Measured, same protocol"],
                ["Confirm baseline", "142.8–143.1 ms", "6.99 tok/s", "Measured"],
                ["Winner repeat 1", "129.0 ms", "7.75 tok/s", "Measured"],
                ["Winner repeat 2", "129.1 ms", "7.75 tok/s", "Measured"],
                ["Winner next run", "129.2 ms", "7.74 tok/s", "Measured"],
                ["B8 winner", "139.4 ms/user", "57.4 agg tok/s", "Aggregate derived"],
            ],
            widths=[0.30, 0.21, 0.22, 0.26],
            row_height=0.043,
        )
        y = _section(ax, y, "Profile interpretation — corrected")
        y = _table(
            ax,
            y,
            ["Profile scope", "Observed share", "How to use it"],
            [
                ["Full 92-layer decode signpost", "CCL ~8%; matmul ~22%", "Use shares, not absolute wall time"],
                ["4-layer Tracy slice", "RS+AG ~27%; matmul ~50%", "Contaminated/slice-specific"],
                ["Standalone M=32 matmuls", "27–54% of 288 GB/s", "Shape/config guidance only"],
            ],
            widths=[0.34, 0.26, 0.39],
            row_height=0.052,
        )
        y = _section(ax, y, "Key conclusion")
        _paragraph(
            ax,
            y,
            "The old statement that communication consumes ~27% of full decode is not defensible: that figure came "
            "from a four-layer profile slice. The clean full-layer decode profile puts CCL near 8%, limiting the gain "
            "available from another collective-only optimization. Further progress needs layout, residual-sharding, "
            "or production-path matmul changes rather than another environment-knob sweep.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-REAP-218B — Applied Optimizations",
            "Validated winner configuration and evaluated alternatives",
            3,
            4,
        )
        y = 0.885
        y = _section(ax, y, "Applied winner stack")
        y = _table(
            ax,
            y,
            ["Optimization", "Isolated result", "Winner status"],
            [
                ["L1 RMSNorm (GLM4_MOE_NORM_L1=1)", "~2.2 ms win", "Applied"],
                ["LoFi sparse MoE", "Part of ~1.6 ms pair", "Applied"],
                ["Fused expert gate/up", "Part of ~1.6 ms pair", "Applied + cache fix"],
                ["LoFi attention fidelity", "~0.4 ms alone", "Applied"],
                ["Four-knob coherent combination", "143.2 → 129.0 ms", "Validated winner"],
                ["EP L1", "Disabling regressed ~4.3 ms", "Retained"],
                ["Fused shared/EP reduce", "Disabling regressed ~7.8 ms", "Retained"],
                ["4-link CCL", "2 links regressed ~2.2 ms", "Retained"],
            ],
            widths=[0.42, 0.24, 0.23],
            row_height=0.043,
        )
        y = _section(ax, y, "Evaluated and exhausted")
        _bullet_list(
            ax,
            y,
            [
                "Ring MoE TP reduce was performance-neutral (140.7 → 140.6 ms) despite correct output.",
                "Async reduce-scatter/all-gather and full-mesh reduce were neutral.",
                "Eight CCL links, alternate topologies, fused QK-RoPE, skip-slice and additional fidelity knobs were flat, worse, or unstable.",
                "DRAM-shard disable/prefetcher and 32-core norm configurations crashed; they are not applied wins.",
                "Approximately 50 hardware-planner probes produced no committed end-to-end winner.",
            ],
            size=8.2,
        )
        y = _section(ax, y, "Configuration caveat")
        _paragraph(
            ax,
            y,
            "The four winner knobs are env-driven and code defaults remain conservative/default-off. Launch scripts "
            "must export the validated combination to reproduce 129 ms.",
            size=8.5,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = _page(
            "GLM-4.7-REAP-218B — Next Optimization Plan",
            "Architecture-oriented work after the knob and CCL sweep",
            4,
            4,
        )
        y = 0.885
        y = _section(ax, y, "Recommended sequence")
        y = _table(
            ax,
            y,
            ["Priority", "Workstream", "Evidence / expected outcome"],
            [
                ["P0", "Ship winner env in production launch", "Locks in measured ~10% B1 improvement"],
                ["P1", "Audit production DRAM-sharded matmuls", "Standalone shapes reach only 27–54% peak"],
                ["P2", "Sharded residual across attention/MoE", "Avoid post-RS all-gather; large engineering"],
                ["P3", "Fuse RMSNorm and layout boundaries", "Repeated small-op/layout population"],
                ["P4", "Repair prefetcher + sharded RoPE", "Current sub-device path is broken"],
                ["P5", "MLA-style K-sharded attention", "Re-architecture; single-digit CCL ceiling alone"],
            ],
            widths=[0.12, 0.36, 0.41],
            row_height=0.048,
        )
        y = _section(ax, y, "Target ladder")
        y = _metrics(
            ax,
            y,
            [
                ("Current", "129 ms", "Validated winner"),
                ("Near milestone", "110–120 ms", "Matmul/layout work"),
                ("Architecture stretch", "80–100 ms", "Residual/layout redesign"),
                ("Weight-only target", "33–40 ms", "Not an immediate E2E forecast"),
            ],
            height=0.11,
        )
        y = _section(ax, y, "Methodology cautions")
        _bullet_list(
            ax,
            y,
            [
                "Use ~143 ms as the pre-combo A/B baseline and 129.0 ms as the current winner.",
                "The clean full-layer CCL share is ~8%; do not reuse the 27% four-layer-slice figure as full-token cost.",
                "BF4 halves expert bytes but not dense attention, collectives, routing, norms, layouts, or synchronization.",
                "The 25–30 tok/s practical roofline is a weight-only heuristic and remains 3.2–3.9× above measured throughput.",
            ],
        )
        pdf.savefig(fig)
        plt.close(fig)


def _opt_box(ax, x: float, y: float, w: float, h: float, title: str, detail: str, color: str, status: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor="white",
            edgecolor=color,
            linewidth=1.8,
        )
    )
    ax.text(x + 0.015, y + h - 0.026, title, fontsize=10, fontweight="bold", color=DARK, va="top")
    ax.text(x + 0.015, y + h - 0.058, textwrap.fill(detail, 40), fontsize=7.7, color=MID, va="top", linespacing=1.25)
    ax.text(x + w - 0.012, y + 0.012, status, fontsize=7.2, color=color, ha="right", va="bottom", fontweight="bold")


def generate_optimization_image() -> None:
    OPT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.95, "GLM-4.7 Decode Optimization Journey", fontsize=23, fontweight="bold", color=DARK, va="top")
    ax.text(
        0.04,
        0.905,
        "Validated optimizations applied to the current Wormhole Galaxy winners · gains are measured where isolated and otherwise credited to the coherent bundle",
        fontsize=10,
        color=MID,
        va="top",
    )

    ax.add_patch(plt.Rectangle((0.035, 0.81), 0.45, 0.065, color=BLUE))
    ax.text(0.055, 0.842, "GLM-4.7-Flash", color="white", fontsize=16, fontweight="bold", va="center")
    ax.text(0.465, 0.842, "74.8 → 51.3 ms  |  31.4% lower", color="white", fontsize=12, ha="right", va="center")

    flash_boxes = [
        ("Execution", "Trace + on-device sampling; defensive clone/typecast removal", "APPLIED"),
        ("Precision", "BF8 dense and expert weights; BF16 KV retained for quality", "~7% DENSE WIN"),
        ("Matmul tuning", "Explicit program config and in0_block_w=8", "58.1 → 53.4 MS*"),
        ("MoE fusion", "Collective epilogue, buffered all-reduce, fused MLP/MoE reduce", "APPLIED"),
        ("Routing", "Down post-scale folded into sparse matmul; fused GLM B1 router", "~1.7 MS"),
        ("Memory/layout", "Sharded L1 norm, L1 activations/EP/router, fused QKV-A/shared gate-up", "APPLIED"),
    ]
    for idx, (title, detail, status) in enumerate(flash_boxes):
        col, row = idx % 2, idx // 2
        _opt_box(ax, 0.04 + col * 0.225, 0.665 - row * 0.145, 0.205, 0.115, title, detail, BLUE, status)

    ax.add_patch(plt.Rectangle((0.515, 0.81), 0.45, 0.065, color=PURPLE))
    ax.text(0.535, 0.842, "GLM-4.7-REAP-218B", color="white", fontsize=16, fontweight="bold", va="center")
    ax.text(0.945, 0.842, "143.2 → 129.0 ms  |  9.9% lower", color="white", fontsize=12, ha="right", va="center")

    reap_boxes = [
        ("L1 normalization", "Move repeated RMSNorm working sets to L1", "~2.2 MS"),
        ("Sparse fidelity", "LoFi sparse MoE math in validated winner", "APPLIED"),
        ("Expert fusion", "Fuse expert gate/up with coherent weight-cache fix", "~1.6 MS PAIR"),
        ("Attention fidelity", "LoFi attention path", "~0.4 MS ALONE"),
        ("Collectives retained", "EP L1, fused shared/EP reduce and 4-link CCL", "REGRESSION IF OFF"),
        ("Coherent combination", "Four winner knobs interact synergistically", "~14.2 MS TOTAL"),
    ]
    for idx, (title, detail, status) in enumerate(reap_boxes):
        col, row = idx % 2, idx // 2
        _opt_box(ax, 0.52 + col * 0.225, 0.665 - row * 0.145, 0.205, 0.115, title, detail, PURPLE, status)

    ax.add_patch(
        FancyBboxPatch(
            (0.04, 0.105),
            0.925,
            0.13,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=LIGHT,
            edgecolor=GRID,
            linewidth=1,
        )
    )
    ax.text(0.06, 0.205, "Next optimization frontier", fontsize=12, fontweight="bold", color=DARK, va="top")
    ax.text(
        0.06,
        0.17,
        "Flash: re-profile 51.3 ms → wire GlobalCB weight prefetch → fuse KV-update layout → partition KV memory",
        fontsize=9,
        color=BLUE,
        va="top",
    )
    ax.text(
        0.06,
        0.135,
        "REAP: ship winner env → production DRAM-sharded matmul audit → sharded residual/layout redesign → repair prefetcher + RoPE",
        fontsize=9,
        color=PURPLE,
        va="top",
    )
    ax.text(
        0.96,
        0.04,
        "* Intermediate Flash measurements are not additive with the final winner. Source: validated sweeps and A/B artifacts, 2026-07-24.",
        fontsize=7.5,
        color=MID,
        ha="right",
    )

    fig.savefig(OPT_IMAGE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    generate_flash_pdf()
    generate_flash_minor_update_pdf()
    generate_reap_pdf()
    generate_optimization_image()
    print(FLASH_PDF)
    print(FLASH_MINOR_PDF)
    print(REAP_PDF)
    print(OPT_IMAGE)


if __name__ == "__main__":
    main()
