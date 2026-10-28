import json
import statistics
import sys

results_path, page_path = sys.argv[1], sys.argv[2]
results = json.load(open(results_path))["results"]

times = {}
correct = {}
for r in results:
    if "step_ms" not in r:
        continue
    times.setdefault(r["variant"], []).append(statistics.median(r["step_ms"]))
    diff = r.get("max_rel_param_diff_vs_serial", 0.0)
    correct[r["variant"]] = correct.get(r["variant"], True) and diff == 0.0
ms = {v: statistics.median(t) for v, t in times.items()}
spread = {v: (min(t), max(t)) for v, t in times.items()}
serial, racy = ms["serial"], ms["racy"]


def share(v):
    return (serial - ms[v]) / (serial - racy)


ROWS = [
    ("serial", "Serial", "collectives on queue 0's compute cores"),
    ("overlap", "Ordering derived, as first built", "events recorded when the consumer asks"),
    ("eager", "+ events after each block's collectives", "waits use the earliest event that covers them"),
    ("eagerpark", "+ FSDP's frees parked to the step's end", "buffers the collectives touched"),
    ("marks", "+ events where each block's compute ends", "for collectives waiting on compute"),
    ("racy", "No ordering", "the ceiling; results wrong"),
]
ROWS = [row for row in ROWS if row[0] in ms]

W, label_w, top, row_h = 900, 330, 34, 46
x0, x1 = label_w + 20, W - 120
lo, hi = -0.25, 1.0


def x(v):
    return x0 + (v - lo) / (hi - lo) * (x1 - x0)


H = top + row_h * len(ROWS) + 30
svg = [
    f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="chartTitle">',
    '<title id="chartTitle">Share of the possible overlap each variant recovers, from serial (0 %) to no ordering (100 %).</title>',
    '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    '<rect width="6" height="6" fill="var(--bad-soft)"/><line x1="0" y1="0" x2="0" y2="6" stroke="var(--bad)" stroke-width="2"/></pattern></defs>',
]
for tick in (-0.25, 0, 0.25, 0.5, 0.75, 1.0):
    tx = x(tick)
    svg.append(
        f'<line x1="{tx:.1f}" y1="{top - 8}" x2="{tx:.1f}" y2="{H - 26}" stroke="var(--rule)" stroke-width="{1.6 if tick == 0 else 1}"/>'
    )
    svg.append(f'<text x="{tx:.1f}" y="{H - 8}" text-anchor="middle" class="small">{round(tick * 100):d} %</text>')
svg.append(f'<text x="{x0}" y="{top - 16}" class="small">share of the possible overlap recovered</text>')
for i, (v, label, note) in enumerate(ROWS):
    y = top + i * row_h
    s = share(v)
    ok = correct.get(v, True)
    bx, bw = (x(0), x(s) - x(0)) if s >= 0 else (x(s), x(0) - x(s))
    fill = "url(#hatch)" if not ok else ("var(--accent)" if v not in ("serial",) else "var(--compute)")
    svg.append(f'<text x="0" y="{y + 17}" font-size="13.5" font-weight="500">{label}</text>')
    svg.append(f'<text x="0" y="{y + 34}" class="small">{note}</text>')
    if abs(bw) > 0.5:
        svg.append(f'<rect x="{bx:.1f}" y="{y + 8}" width="{bw:.1f}" height="22" rx="2" fill="{fill}"/>')
    else:
        svg.append(f'<rect x="{x(0) - 1:.1f}" y="{y + 8}" width="2" height="22" fill="var(--compute)"/>')
    lo_ms, hi_ms = spread[v]
    value = f"{ms[v]:.1f} ms"
    pct = f"{s * 100:+.0f} %" if v not in ("serial", "racy") else ("0 %" if v == "serial" else "100 %")
    status = "" if ok else " · wrong"
    svg.append(
        f'<text x="{x1 + 10}" y="{y + 18}" font-size="13" class="mono" style="font-variant-numeric:tabular-nums">{value}</text>'
    )
    svg.append(f'<text x="{x1 + 10}" y="{y + 34}" class="small">{pct}{status}</text>')
svg.append("</svg>")
chart = "\n".join(svg)

passes = {v: ", ".join(f"{t:.1f}" for t in times[v]) for v in ms}
best = max((v for v in ms if correct.get(v, True) and v not in ("serial", "racy")), key=share)
caption = (
    f"Median of three interleaved passes of 6 steps each, all in one process. Per pass: parked frees "
    f"{passes['eagerpark']} ms; events where compute ends {passes['marks']} ms; serial {passes['serial']} ms. "
    f"Every bar except the hatched one matched the serial run exactly."
)
page = open(page_path).read()
page = page.replace("<!--CHART-->", chart)
page = page.replace("<!--CHARTCAPTION-->", caption)
page = page.replace("<!--SERIAL-->", f"{serial:.1f}")
page = page.replace("<!--RACY-->", f"{racy:.1f}")
page = page.replace("<!--DATAONLY-->", f"{ms.get('dataonly', float('nan')):.1f}")
recovered = share(best)
page = page.replace("<!--RECOVERED-->", f"{recovered * 100:.0f} %")
best_pass = (serial - min(times[best])) / (serial - racy)
page = page.replace("<!--BEST-->", f"{best_pass * 100:.0f} %")
open(page_path, "w").write(page)
for v, label, _ in ROWS:
    print(
        f"{v:10s} {ms[v]:7.2f} ms  share {share(v) * 100:+6.1f} %  correct {correct.get(v, True)}  spread {spread[v]}"
    )
print("best", best)
