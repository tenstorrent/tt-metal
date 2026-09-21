# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Render recorded common-reader measurements; never execute an attention producer."""

import hashlib
import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "bfp4-lofi-v2"
# ID, label, family, bytes per 1024-value K OR V tile, source templates.
CHOICES = [
    ("A", "Main BF16", "canonical", 2048, "chain-h10-{n}-main-v1.json"),
    ("B", "FAST BF16 + Q256 reset", "canonical", 2048, "chain-h10-{n}-fast-v1.json"),
    ("C", "Balanced FP32 QK4/PV2", "canonical", 2048, "chain-h10-{n}-balanced-v1.json"),
    ("D", "ACCURATE FP32 QK4/PV4", "canonical", 2048, "chain-h10-{n}-accurate-v1.json"),
    ("E", "LoFi B8 / compensated BF16", "b8", 1088, "chain-h10-{n}-lofi_fast_b8-v1.json"),
    ("F", "LoFi B8 / FP32 / native exp", "b8", 1088, "native{short}-lofi_fp32_b8-v1.json"),
    ("G", "LoFi B4 / compensated BF16", "b4", 576, "chain-h10-{n}-lofi_fast_b4-v1.json"),
    ("H", "LoFi B4 / FP32 / native exp", "b4", 576, "native{short}-lofi_fp32_b4-v1.json"),
    ("I", "Adaptive B4 / denom-only BF16", "b4", 576, "adaptive{short}-b4_b4-minus-normal-v1.json"),
    ("J", "2 x B4 residual / FP32", "residual", 1152, "fullchip_residual/residual44-chain-h10-{n}-qscale-v1.json"),
    ("K", "B4 + B8 residual / FP32", "residual", 1664, "fullchip_residual/residual48-chain-h10-{n}-qscale-v1.json"),
]
COLORS = {"canonical": "#234c88", "b8": "#097d70", "b4": "#bc5f12", "residual": "#8253a0"}


def frontier(rows):
    return sorted(
        [a for a in rows if not any(
            b["l2_pct"] <= a["l2_pct"] and b["tflops"] >= a["tflops"]
            and (b["l2_pct"] < a["l2_pct"] or b["tflops"] > a["tflops"])
            for b in rows
        )], key=lambda a: a["l2_pct"]
    )


def main():
    rows = []
    for n, short in [(32768, "32k"), (262144, "256k")]:
        for ident, label, family, tile_bytes, template in CHOICES:
            path = DATA / template.format(n=n, short=short)
            raw = path.read_bytes()
            data = json.loads(raw)
            assert data["length"] == n and data["heads"] == 10 and data["seed"] == 1240
            assert data["distribution"] == "normal" and data["sample_rows"] == 128
            assert data["actual_cores"] == 110
            assert data.get("q_chunk", 256) == 256
            row = dict(
                id=ident, label=label, family=family, length=n,
                tile_bytes_per_k_or_v=tile_bytes,
                kv_payload_fraction_of_bf16=tile_bytes / 2048,
                l2_pct=data["accuracy"]["l2_pct"], pcc=data["accuracy"]["pcc"],
                tflops=data["combined_tflops"],
                source=str(path.relative_to(HERE.parent)),
                source_sha256=hashlib.sha256(raw).hexdigest(),
                preprocess_check_recorded=data.get("check_preprocess"),
            )
            assert row["l2_pct"] > 0 and row["tflops"] > 0
            rows.append(row)

    manifest = dict(
        schema="sdpa-ring-pareto-v1", rows=rows,
        contract="Recorded H10 D128 noncausal normal seed1240 Q256/K512 C110 common-reader; original BF16 FP64 reference, Q128 sample/all KV; BF16 output; combined trace time includes device preparation, excludes upload. Not resident/no-DM, production dispatch, or ring timing.",
        caveats=[
            "B is the canonical FAST numerical choice with the private Q256 correctness reset, not byte-identical frozen code.",
            "BF16 paths retain two KV slots, FP32 one. These are separate historical runs, not confidence intervals or re-timed post-format runs.",
            "I has known coherent-numerator stress failures; a normal-input Pareto point is not acceptance.",
            "J/K send both components, use Q prescale 1.0028 and twice the QK/PV matmul work; useful throughput excludes redundant work.",
            "Payload arithmetic includes native shared-exponent bytes but excludes wire framing/alignment and any optional transform metadata; assumes compressed components are sent without intermediate BF16 expansion.",
            "Some historical full-chip timings disabled expensive exact preprocessing checks; separate primitive/small-suite checks do not retroactively add gates to these records.",
        ],
        strict_frontiers={str(n): [r["id"] for r in frontier([r for r in rows if r["length"] == n])] for n in [32768, 262144]},
    )
    (HERE / "measurements.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Square canvas avoids Quick Look's fill-and-crop thumbnail behavior.
    # The PNG export crops only the blank area below the 890-pixel figure.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1440" viewBox="0 0 1440 1440" role="img">',
           '<title>SDPA accuracy versus full-chip throughput, including compressed KV choices</title>',
           '<desc>Two panels show 32K and 256K normal-input measurements. Lower L2 and higher throughput are better. Letters identify eleven discrete choices; dashed lines join the measured normal-input frontiers, not qualified operating ranges.</desc>',
           '<rect width="1440" height="1440" fill="white"/>',
           '<style>text {font-family:Arial,Helvetica,sans-serif;fill:#18283c;font-size:17px}.title{font-size:27px;font-weight:bold}.small{font-size:15px}.grid{stroke:#e2e7ee;stroke-width:1}</style>']

    def text(x, y, value, cls="", anchor="start"):
        svg.append(f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}">{html.escape(value)}</text>')

    text(44, 43, "SDPA: accuracy, chip throughput, and K/V payload", "title")
    text(44, 72, "H10 / D128 / Q256 / K512 · normal BF16 inputs · device preprocessing included")
    all_x = [r["l2_pct"] for r in rows] + [0.5]
    lo, hi = min(all_x) / 1.35, max(all_x) * 1.65
    ymin, ymax = 50, 255
    for index, n in enumerate([32768, 262144]):
        left, top, width, height = 90 + 708 * index, 130, 590, 390
        x = lambda v: left + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * width
        y = lambda v: top + height - (v - ymin) / (ymax - ymin) * height
        text(left + width / 2, 110, f"{n:,} tokens", anchor="middle")
        for tick in [75, 100, 125, 150, 175, 200, 225, 250]:
            svg.append(f'<line x1="{left}" y1="{y(tick)}" x2="{left+width}" y2="{y(tick)}" class="grid"/>')
            text(left - 12, y(tick) + 5, str(tick), "small", "end")
        for tick in [.2, .5, 1, 2, 5, 10, 20]:
            svg.append(f'<line x1="{x(tick)}" y1="{top}" x2="{x(tick)}" y2="{top+height}" class="grid"/>')
            text(x(tick), top + height + 25, str(tick), "small", "middle")
        svg.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#9caabc"/>')
        svg.append(f'<line x1="{x(.5)}" y1="{top}" x2="{x(.5)}" y2="{top+height}" stroke="#929ba8" stroke-dasharray="3 5"/>')
        text(x(.5) + 7, top + 19, "0.5%", "small")
        selected = [r for r in rows if r["length"] == n]
        points = " ".join(f'{x(r["l2_pct"]):.2f},{y(r["tflops"]):.2f}' for r in frontier(selected))
        svg.append(f'<polyline points="{points}" fill="none" stroke="#a0a9b5" stroke-width="2" stroke-dasharray="6 5"/>')
        offsets = {"A": (11, -9), "B": (-13, 23), "C": (11, -10), "D": (-9, 25), "E": (12, -12), "F": (12, 22), "G": (-13, -11), "H": (-14, 24), "I": (-13, -13), "J": (10, -9), "K": (10, 23)}
        for r in selected:
            px, py, color = x(r["l2_pct"]), y(r["tflops"]), COLORS[r["family"]]
            svg.append(f'<g><title>{html.escape(r["label"])}: L2 {r["l2_pct"]:.4f}%, {r["tflops"]:.2f} TFLOP/s, KV payload {100*r["kv_payload_fraction_of_bf16"]:.3f}% of BF16</title>')
            if r["family"] == "canonical":
                svg.append(f'<rect x="{px-5}" y="{py-5}" width="10" height="10" fill="{color}"/>')
            elif r["family"] == "b4":
                svg.append(f'<path d="M{px},{py-7} L{px+7},{py+6} L{px-7},{py+6} Z" fill="{color}"/>')
            elif r["family"] == "residual":
                svg.append(f'<path d="M{px},{py-7} L{px+7},{py} L{px},{py+7} L{px-7},{py} Z" fill="{color}"/>')
            else:
                svg.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{color}"/>')
            dx, dy = offsets[r["id"]]
            text(px + dx, py + dy, r["id"], anchor="end" if dx < 0 else "start")
            svg.append('</g>')
        text(left + width / 2, 570, "Relative L2 error (%) · log scale · lower is better", anchor="middle")
        svg.append(f'<text transform="translate({left-59},{top+height/2}) rotate(-90)" text-anchor="middle">Combined chip TFLOP/s · higher is better</text>')
    text(44, 610, "Four canonical BF16-input choices", "small")
    text(748, 610, "Low-precision choices · K/V payload relative to BF16", "small")
    for i, (ident, label, family, tile_bytes, _) in enumerate(CHOICES):
        if i < 4:
            xx, yy = 44, 639 + i * 28
        else:
            xx, yy = 748, 639 + (i - 4) * 28
        svg.append(f'<circle cx="{xx+5}" cy="{yy-5}" r="5" fill="{COLORS[family]}"/>')
        text(xx + 20, yy, f"{ident}  {label}" + (f" · {tile_bytes/2048*100:.1f}%" if i >= 4 else ""), "small")
    text(44, 775, "B: private correctness reset; frozen snapshots unchanged.", "small")
    text(44, 800, "Dashed: strict frontier of these normal-input observations only.", "small")
    text(44, 825, "I: known stress failures. J/K: multiple components, not single BFP4.", "small")
    text(44, 863, "Original-input FP64 reference: 128 Q rows/head, all K/V. No ring timing or model-quality claim. Separate recorded runs; no error bars.", "small")
    svg.append('</svg>')
    (HERE / "sdpa-ring-pareto.svg").write_text("\n".join(svg) + "\n")
    print(json.dumps({"rows": len(rows), "frontiers": manifest["strict_frontiers"]}, indent=2))
    for row in rows:
        print(f'{row["length"]} {row["id"]} {row["l2_pct"]:.4f}% {row["tflops"]:.2f} TF/s PCC={row["pcc"]:.8f}')


if __name__ == "__main__":
    main()
