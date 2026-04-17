#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Generate dm_noc_perf_results.xlsx  —  single Results sheet with inline legend.
Usage: python3 gen_perf_excel.py
"""

import csv
import re
import pandas as pd
import xlsxwriter

# ── Data loading ──────────────────────────────────────────────────────────────


def parse_test_name(name):
    m = re.match(
        r"Dm(CacheL2FlushNoc|DirectSramNoc)_(TwoPhase|PerIter)_(BarrierEnd|BarrierPerIter)"
        r"(_UpdateNocAddr)?(?:_(\d+)Cores)?$",
        name,
    )
    if not m:
        return None
    return (
        "Cache+L2Flush" if m.group(1) == "CacheL2FlushNoc" else "DirectSRAM",
        m.group(2),
        m.group(3),
        "Yes" if m.group(4) else "No",
        int(m.group(5)) if m.group(5) else 1,
    )


def load_csv(path, has_header, iterations):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        if has_header:
            next(reader)
        for line in reader:
            if len(line) < 4:
                continue
            dims = parse_test_name(line[0])
            if dims is None:
                continue
            wm, ls, bm, ua, nc = dims
            wa, na, ta = int(line[1]), int(line[2]), int(line[3])
            rows.append(
                {
                    "WriteMode": wm,
                    "LoopStyle": ls,
                    "BarrierMode": bm,
                    "UpdateNocAddr": ua,
                    "NumCores": nc,
                    "Iterations": iterations,
                    "write_avg": wa if wa != 0 else None,
                    "noc_avg": na if na != 0 else None,
                    "total_avg": ta,
                }
            )
    return pd.DataFrame(rows)


df_all = load_csv("dm_noc_perf_results.txt", has_header=True, iterations=100)

# ── Workbook ──────────────────────────────────────────────────────────────────

OUTPUT = "dm_noc_perf_results.xlsx"
wb = xlsxwriter.Workbook(OUTPUT)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "navy": "#1F3864",
    "navy2": "#2E4D7B",
    "cache_light": "#DEEAF1",
    "cache_mid": "#BDD7EE",
    "sram_light": "#E2EFDA",
    "sram_mid": "#C6EFCE",
    "loop_2p": "#EDE7F6",  # lavender — TwoPhase
    "loop_pi": "#FBE9E7",  # peach    — PerIter
    "barr_end": "#FFF9C4",  # pale yellow — BarrierEnd
    "barr_per": "#FFE0B2",  # pale orange — BarrierPerIter
    "upd_no": "#F5F5F5",  # neutral
    "upd_yes": "#E8F5E9",  # pale mint
    "iter_100": "#E3F2FD",  # pale blue
    "iter_10": "#ECEFF1",  # pale grey
    "na_bg": "#F5F5F5",
    "na_fg": "#9E9E9E",
    "leg_hdr": "#37474F",  # dark slate
    "leg_sec": "#ECEFF1",  # section band
    "white": "#FFFFFF",
}


def F(**kw):
    """Shorthand for wb.add_format."""
    return wb.add_format(kw)


# ── Common formats ────────────────────────────────────────────────────────────
f_title = F(bold=True, font_size=13, font_color=C["navy"], valign="vcenter")
f_hdr_dim = F(
    bold=True, bg_color=C["navy"], font_color="white", border=1, align="center", valign="vcenter", text_wrap=True
)
f_hdr_met = F(
    bold=True, bg_color=C["navy2"], font_color="white", border=1, align="center", valign="vcenter", text_wrap=True
)
f_na = F(bg_color=C["na_bg"], font_color=C["na_fg"], italic=True, border=1, align="center")

# Value → cell format (text)
DIM_FMTS = {
    "Cache+L2Flush": F(bg_color=C["cache_light"], border=1, align="center"),
    "DirectSRAM": F(bg_color=C["sram_light"], border=1, align="center"),
    "TwoPhase": F(bg_color=C["loop_2p"], border=1, align="center"),
    "PerIter": F(bg_color=C["loop_pi"], border=1, align="center"),
    "BarrierEnd": F(bg_color=C["barr_end"], border=1, align="center"),
    "BarrierPerIter": F(bg_color=C["barr_per"], border=1, align="center"),
    "No": F(bg_color=C["upd_no"], border=1, align="center"),
    "Yes": F(bg_color=C["upd_yes"], border=1, align="center"),
    1: F(bg_color=C["white"], border=1, align="center"),
    2: F(bg_color="#E8EAF6", border=1, align="center"),  # indigo tint
    3: F(bg_color="#F3E5F5", border=1, align="center"),  # purple tint
    4: F(bg_color="#FCE4EC", border=1, align="center"),  # pink tint
    100: F(bg_color=C["iter_100"], border=1, align="center"),
    10: F(bg_color=C["iter_10"], border=1, align="center"),
}
NUM_FMT = F(bg_color=C["white"], border=1, align="center", num_format="0")

# ── Sheet: Results ────────────────────────────────────────────────────────────
ws = wb.add_worksheet("Results")
ws.set_zoom(85)
ws.freeze_panes(2, 0)
ws.set_paper(9)  # A4

# Title
ws.merge_range("A1:H1", "DM NOC Performance  —  Quasar DM Core", f_title)
ws.set_row(0, 22)
ws.set_row(1, 34)

# Column layout  (key, header_label, width, hdr_fmt)
COLS = [
    ("WriteMode", "Write Mode", 16, f_hdr_dim),
    ("LoopStyle", "Loop Style", 12, f_hdr_dim),
    ("BarrierMode", "Barrier Mode", 14, f_hdr_dim),
    ("UpdateNocAddr", "Update NOC Addr", 15, f_hdr_dim),
    ("NumCores", "Num\nCores", 9, f_hdr_dim),
    ("Iterations", "Iterations", 11, f_hdr_dim),
    ("write_avg", "write_avg\n(cycles)", 13, f_hdr_met),
    ("noc_avg", "noc_avg\n(cycles)", 13, f_hdr_met),
    ("total_avg", "total_avg\n(cycles)", 13, f_hdr_met),
]

for ci, (_, label, width, hfmt) in enumerate(COLS):
    ws.write(1, ci, label, hfmt)
    ws.set_column(ci, ci, width)

ws.autofilter(1, 0, 1 + len(df_all), len(COLS) - 1)

# Data rows
DATA_ROW_START = 2
for ri, r in df_all.iterrows():
    row = DATA_ROW_START + ri
    ws.set_row(row, 16)
    for ci, (key, _, _, _) in enumerate(COLS):  # noqa: unpack label/width/hfmt silently
        val = r[key]
        if key in ("write_avg", "noc_avg", "total_avg"):
            if pd.isna(val):
                ws.write(row, ci, "n/a", f_na)
            else:
                ws.write_number(row, ci, int(val), NUM_FMT)
        else:
            ws.write(row, ci, val, DIM_FMTS.get(val, f_na))

# Conditional 3-colour scale on metric columns (green = fast, red = slow)
last_data_row = DATA_ROW_START + len(df_all) - 1
for ci, col_name in [(6, "write_avg"), (7, "noc_avg"), (8, "total_avg")]:
    col_letter = chr(ord("A") + ci)
    cell_range = f"{col_letter}{DATA_ROW_START + 1}:{col_letter}{last_data_row + 1}"
    ws.conditional_format(
        cell_range,
        {
            "type": "3_color_scale",
            "min_color": "#63BE7B",  # green  — fastest
            "mid_color": "#FFEB84",  # yellow — mid
            "max_color": "#F8696B",  # red    — slowest
        },
    )

# ── Inline legend  (cols J–M, right of data) ─────────────────────────────────
LGAP = 1  # spacer column between data and legend
LC = len(COLS) + LGAP  # legend start column index

f_leg_title = F(bold=True, font_size=11, font_color=C["navy"], bg_color=C["leg_sec"], border=1, valign="vcenter")
f_leg_hdr = F(bold=True, bg_color=C["leg_hdr"], font_color="white", border=1, align="center", valign="vcenter")
f_leg_sec = F(bold=True, bg_color=C["leg_sec"], font_color=C["navy"], border=1, valign="vcenter", left=2)
f_leg_val = F(bg_color=C["white"], border=1, align="center", valign="vcenter", italic=True)
f_leg_desc = F(bg_color=C["white"], border=1, valign="vcenter", text_wrap=True, font_size=9)

# Matching value formats for the legend colour column
LEG_VAL_FMTS = {
    k: F(bg_color=v, border=1, align="center", valign="vcenter", italic=True, font_size=9)
    for k, v in {
        "Cache+L2Flush": C["cache_light"],
        "DirectSRAM": C["sram_light"],
        "TwoPhase": C["loop_2p"],
        "PerIter": C["loop_pi"],
        "BarrierEnd": C["barr_end"],
        "BarrierPerIter": C["barr_per"],
        "No": C["upd_no"],
        "Yes": C["upd_yes"],
        "1": C["white"],
        "2": "#E8EAF6",
        "3": "#F3E5F5",
        "4": "#FCE4EC",
        "100": C["iter_100"],
        "10": C["iter_10"],
    }.items()
}
LEG_VAL_FMTS["n/a"] = F(
    bg_color=C["na_bg"], border=1, align="center", valign="vcenter", italic=True, font_size=9, font_color=C["na_fg"]
)

ws.set_column(LC - LGAP, LC - LGAP, 2)  # spacer
ws.set_column(LC, LC, 18)  # Value
ws.set_column(LC + 1, LC + 1, 52)  # Description

ws.merge_range(1, LC, 1, LC + 1, "Legend", f_leg_hdr)

LEGEND = [
    # (section_label, [(value_str, description), ...])
    (
        "Write Mode",
        [
            (
                "Cache+L2Flush",
                "Write 3×64B to L1 cache, then flush_l2_cache_range() evicts dirty lines to SRAM before NOC.",
            ),
            (
                "DirectSRAM",
                "Write via non-cacheable SRAM alias (addr + 0x400000). Bypasses L2 entirely; data in SRAM immediately.",
            ),
        ],
    ),
    (
        "Loop Style",
        [
            (
                "TwoPhase",
                "Two separate loops: all writes first, then all NOC sends. write_avg and noc_avg measured independently.",
            ),
            (
                "PerIter",
                "Single loop: each iteration writes + sends immediately. Only total_avg reported (phases not separated).",
            ),
        ],
    ),
    (
        "Barrier Mode",
        [
            (
                "BarrierEnd",
                "One noc_writes_sent barrier after the full loop. NOC engine pipelines all issues; measures batched throughput.",
            ),
            (
                "BarrierPerIter",
                "Barrier after every NOC send. Fully serialised; measures per-transaction round-trip latency.",
            ),
        ],
    ),
    (
        "Update NOC Addr",
        [
            ("No", "Address registers (SRC_ADDR, DEST_ADDR) set once in setup. No per-iteration ROCC overhead."),
            (
                "Yes",
                "set_src/dest_reg_cmdbuf() called before every issue. Measures 2 ROCC writes/iter. "
                "Also serialises CPU between flush and NOC issue — can reduce cache/NOC contention on Cache+L2Flush path.",
            ),
        ],
    ),
    (
        "Num Cores",
        [
            ("1", "Single core baseline. src={0,0}, dst={1,0}."),
            ("2", "2 cores active in parallel. src[i]={0,i}, dst[i]={1,i}. Result is avg across cores."),
            ("3", "3 cores active in parallel. Tests whether shared NOC/cache bandwidth degrades per-core perf."),
            ("4", "4 cores active in parallel. Maximum load scenario in these tests."),
        ],
    ),
    (
        "Iterations",
        [
            ("100", "100-iteration run. Amortises warmup effects; provides steady-state per-iteration averages."),
        ],
    ),
    (
        "Metric columns",
        [
            ("write_avg", "Avg cycles/iter for the memory-write phase (TwoPhase only; n/a for PerIter)."),
            ("noc_avg", "Avg cycles/iter for the NOC-issue phase (TwoPhase only; n/a for PerIter)."),
            (
                "total_avg",
                "Primary metric. TwoPhase: write_avg + noc_avg. PerIter: combined loop avg. "
                "Color scale: green = fastest, red = slowest.",
            ),
        ],
    ),
]

leg_row = 2
for section, entries in LEGEND:
    ws.merge_range(leg_row, LC, leg_row, LC + 1, section, f_leg_sec)
    ws.set_row(leg_row, 16)
    leg_row += 1
    for val_str, desc in entries:
        vfmt = LEG_VAL_FMTS.get(val_str, f_leg_val)
        ws.write(leg_row, LC, val_str, vfmt)
        ws.write(leg_row, LC + 1, desc, f_leg_desc)
        lines = max(1, len(desc) // 60 + desc.count("\n"))
        ws.set_row(leg_row, max(16, lines * 13))
        leg_row += 1

wb.close()
print(f"Written: {OUTPUT}")
