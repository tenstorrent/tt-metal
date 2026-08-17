#!/usr/bin/env python3
"""fig-b1-tensix-mesh.pdf — Tensix core + NoC mesh schematic (brief 02-background).

Architecture drawing; facts from the Hot Chips talk (vasiljevic2024blackhole)
and the evidence pack header (13x10 grid, 130 workers, p150a). No data CSVs.
Tufte rules per fig/README.md: no gridlines, no chartjunk, direct annotation.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ------- palette (neutral schematic tones; accent = mesh highlight) -------
C_L1 = "#DBE9F6"      # light blue: memory
C_RISC = "#F0F0F0"    # light grey: RISC-V processors
C_DST = "#FDEBD2"     # light orange: Dst regfile
C_CELL = "#FFFFFF"    # mesh cell face
C_EDGE = "#555555"    # box edges
C_LINK = "#BBBBBB"    # NoC links
C_ACC = "#4C72B0"     # highlighted core
C_TXT = "#222222"
FS = 5.2              # base font size

fig, ax = plt.subplots(figsize=(3.45, 2.05))
ax.set_xlim(0, 100)
ax.set_ylim(0, 58)
ax.set_aspect("equal")
ax.axis("off")


def box(x0, y0, x1, y1, fc, label=None, fs=FS, lw=0.6, rounded=False, ls="-",
        label_dy=0.0, weight="normal", ec=C_EDGE):
    if rounded:
        p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                           boxstyle="round,pad=0.4,rounding_size=1.2",
                           fc=fc, ec=ec, lw=lw, ls=ls)
    else:
        p = Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + label_dy, label, ha="center",
                va="center", fontsize=fs, color=C_TXT, weight=weight,
                linespacing=1.15)
    return p


def arrow(x0, y0, x1, y1, style="-|>", lw=0.6, color="#444444", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=4, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, linestyle=ls))


# ============================ left: one Tensix core ========================
box(1, 2, 55, 55, "none", rounded=True, lw=0.8)
ax.text(3, 52.2, "Tensix core", fontsize=6.2, weight="bold", color=C_TXT)

# L1 band with CBs
box(4, 39, 53, 50, C_L1)
ax.text(28.5, 46.6, "L1 SRAM ≈1.5 MB", ha="center", fontsize=FS + 0.4,
        color=C_TXT, weight="bold")
ax.text(17.5, 42.2, "software-managed", ha="center", fontsize=FS - 0.4,
        color=C_TXT)
ax.text(36.8, 42.0, "CBs", ha="right", va="center", fontsize=FS - 0.4,
        color=C_TXT)
for i in range(3):
    box(38.2 + i * 4.9, 40.4, 42.0 + i * 4.9, 43.6, "white", lw=0.5)

# data-movement RISC-Vs
box(4, 25, 13.7, 36, C_RISC, "RISC-V\nNoC rd", fs=FS - 0.2)
box(15.2, 25, 24.9, 36, C_RISC, "RISC-V\nNoC wr", fs=FS - 0.2)
arrow(8.85, 36, 8.85, 39, style="<|-|>")
arrow(20.05, 36, 20.05, 39, style="<|-|>")

# compute pipeline: unpack -> math(SFPU) -> pack, Dst below
box(26.6, 25, 35.4, 36, C_RISC, "unpack", fs=FS - 0.5)
box(36.9, 25, 46.3, 36, C_RISC)
ax.text(41.6, 33.0, "math", ha="center", va="center", fontsize=FS - 0.5)
box(37.7, 26, 45.5, 30.9, "#E4EEF9", "SFPU\n32-lane", fs=FS - 1.0, lw=0.5)
box(47.8, 25, 53.8, 36, C_RISC, "pack", fs=FS - 0.5)
ax.text(40.2, 14.9, "compute: 3 RISC-V", ha="center", fontsize=FS - 0.8,
        color="#555555")
arrow(35.4, 30.5, 36.9, 30.5)
arrow(46.3, 30.5, 47.8, 30.5)
arrow(31, 39, 31, 36)              # L1 -> unpack
arrow(50.8, 36, 50.8, 39)          # pack -> L1
box(36.9, 17, 53.8, 22.3, C_DST, "Dst regfile", fs=FS - 0.4)
arrow(41.6, 25, 41.6, 22.3)        # math -> Dst
arrow(50.8, 22.3, 50.8, 25)        # Dst -> pack

# NoC routers
box(6, 5, 23, 13, "#E8E8E8", "NoC router ×2", fs=FS - 0.2)
arrow(8.85, 25, 8.85, 13, style="<|-|>")
arrow(20.05, 25, 20.05, 13, style="<|-|>")

# ============================ right: 13x10 mesh ============================
mx0, my0, pitch, cell = 60.5, 17.5, 3.0, 2.3
ncol, nrow = 13, 10
# mesh links through cell centers
for r in range(nrow):
    y = my0 + r * pitch + cell / 2
    ax.plot([mx0 + cell / 2, mx0 + (ncol - 1) * pitch + cell / 2], [y, y],
            lw=0.4, color=C_LINK, zorder=1)
for c in range(ncol):
    x = mx0 + c * pitch + cell / 2
    ax.plot([x, x], [my0 + cell / 2, my0 + (nrow - 1) * pitch + cell / 2],
            lw=0.4, color=C_LINK, zorder=1)
for r in range(nrow):
    for c in range(ncol):
        fc = C_ACC if (r == 0 and c == 0) else C_CELL
        ax.add_patch(Rectangle((mx0 + c * pitch, my0 + r * pitch), cell, cell,
                               fc=fc, ec="#888888", lw=0.35, zorder=2))

ax.text(78.5, 51.5, "2D NoC mesh · 13×10 · 130 workers",
        ha="center", fontsize=FS + 0.4, weight="bold", color=C_TXT)
ax.text(78.5, 48.6, "(this unit; harvesting varies)", ha="center",
        fontsize=FS - 0.6, color="#555555")

# connector: core detail -> highlighted mesh cell
arrow(55, 18.5, mx0, my0 + cell / 2, style="-", lw=0.5, color=C_ACC, ls=(0, (2, 1.5)))

# the "lacks" annotation on a NoC edge
lx = mx0 + 4 * pitch + cell / 2   # a link in the bottom row
ly = my0 + cell / 2
arrow(lx + 8, 10.8, lx + 1.5, ly - 0.2, style="-|>", lw=0.5, color=C_TXT)
ax.text(79.5, 8.2, "no atomics · no scatter · no global barrier",
        ha="center", fontsize=FS - 0.1, weight="bold", color=C_TXT)
ax.text(79.5, 4.9, "inter-core sync: NoC semaphores", ha="center",
        fontsize=FS - 0.4, color="#555555")

fig.savefig("fig-b1-tensix-mesh.pdf", bbox_inches="tight", pad_inches=0.02)
print("wrote fig-b1-tensix-mesh.pdf")
