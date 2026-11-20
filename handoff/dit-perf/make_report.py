#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build the timings page from one or more `run_matrix.py` result directories.

    make_report.py --run "Bin6 (7 DRAM banks)=~/dit-perf-runs/b10u20-bin6-full" \
                   --pending "Full-DRAM (8 banks)" --out report.html

Each --run is LABEL=DIR (DIR holds results.json and machine.json, scored or not). Each --pending adds an
empty column to fill in later by re-running this script with `--run` for that machine.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ORDER = [
    "wan-t2v-720p",
    "wan-i2v-720p",
    "h3-t2va-15s",
    "h3-fl2va-15s",
    "h3-ref2va-image-15s",
    "h3-ref2va-video_with_sound-15s",
    "h3-ref2va-mixed-15s",
]
TITLES = {
    "wan-t2v-720p": ("Wan2.2 T2V", "1280x720, 81 frames, 40 steps"),
    "wan-i2v-720p": ("Wan2.2 I2V", "1280x720, 81 frames, 40 steps"),
    "h3-t2va-15s": ("MiniMax-H3 T2VA", "15 s, 1344x768 with audio"),
    "h3-fl2va-15s": ("MiniMax-H3 FL2VA", "15 s, first + last keyframe"),
    "h3-ref2va-image-15s": ("MiniMax-H3 Ref2VA", "15 s, reference: image"),
    "h3-ref2va-video_with_sound-15s": ("MiniMax-H3 Ref2VA", "15 s, reference: video with sound"),
    "h3-ref2va-mixed-15s": ("MiniMax-H3 Ref2VA", "15 s, reference: mixed"),
}
GOOD = ("PASS", "PERF_LIMIT")


def load(label: str, d: Path) -> dict:
    rows = {r["case"]: r for r in json.loads((d / "results.json").read_text())}
    machine = json.loads((d / "machine.json").read_text()) if (d / "machine.json").is_file() else {}
    return {"label": label, "machine": machine, "rows": rows}


def num(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def cell(row: dict | None) -> str:
    if row is None:
        return '<td class="pending" colspan="1">not run</td>'
    p = row.get("perf") or {}
    status = row["status"]
    cls = {"PASS": "ok", "PERF_LIMIT": "ok"}.get(status, "bad")
    if status not in GOOD:
        return f'<td class="{cls}"><span class="pill bad">{html.escape(status)}</span><div class="sub">{html.escape((row.get("detail") or "")[:90])}</div></td>'
    total = (p.get("total") or {}).get("mean")
    den = (p.get("denoise") or {}).get("mean")
    vae = (p.get("vae_decode") or {}).get("mean")
    note = ""
    if status == "PERF_LIMIT":
        note = '<span class="pill warn" title="Ran and measured; tripped the full-DRAM timing limits in the Wan test">over Wan limit</span>'
    return (
        f'<td class="ok num"><div class="big">{num(total)} s</div>'
        f'<div class="sub">denoise {num(den)} · vae {num(vae, 2)}</div>{note}</td>'
    )


def quality(row: dict | None) -> str:
    if row is None or row["status"] not in GOOD:
        return '<td class="pending">–</td>'
    q = row.get("quality") or {}
    if not q:
        return '<td class="pending">not scored</td>'
    if "error" in q:
        return f'<td class="sub">{html.escape(q["error"][:80])}</td>'
    bits = []
    clip = q.get("clip", {}).get("mean")
    if clip is not None:
        bits.append(f"CLIP {clip:.1f}")
    vb = q.get("vbench") or {}
    if vb:
        bits.append(
            "imaging {:.2f} · smooth {:.2f}".format(
                vb.get("imaging_quality", float("nan")), vb.get("motion_smoothness", float("nan"))
            )
        )
    fr = q.get("frames") or {}
    flag = ""
    if fr.get("flat_or_static"):
        flag = '<span class="pill bad">flat/static</span>'
    elif q.get("vbench_ok") is False or q.get("clip_ok") is False:
        flag = '<span class="pill warn">below fox bars</span>'
    elif q.get("vbench_ok") or q.get("clip_ok"):
        flag = '<span class="pill good">meets bars</span>'
    return f'<td class="num"><div>{"<br>".join(bits)}</div>{flag}</td>'


PAGE = """<title>DiT Perf Matrix</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* One table: a row per workload, a column per machine; units in seconds, 130 W TDP on every chip. */
:root{--bg:#f6f7f8;--panel:#ffffff;--fg:#14202b;--mut:#5d6b78;--line:#dde2e7;--accent:#0b6e8f;
--good:#1b7a46;--goodbg:#e2f3e8;--warn:#8a5a00;--warnbg:#fbefd2;--bad:#a8261c;--badbg:#fbe4e1;--pend:#eef0f3;
--sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f161d;--panel:#16202a;--fg:#e6edf3;--mut:#93a2b0;--line:#2a3743;--accent:#5cc3e3;
--good:#6fd09b;--goodbg:#143323;--warn:#e8b454;--warnbg:#3a2c0d;--bad:#ff8f84;--badbg:#401a16;--pend:#1b2732;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f161d;--panel:#16202a;--fg:#e6edf3;--mut:#93a2b0;--line:#2a3743;--accent:#5cc3e3;
--good:#6fd09b;--goodbg:#143323;--warn:#e8b454;--warnbg:#3a2c0d;--bad:#ff8f84;--badbg:#401a16;--pend:#1b2732;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.5;
padding-block:28px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:1100px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 4px;text-wrap:balance}
.lede{color:var(--mut);margin:0 0 20px;max-width:70ch}
.machines{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:22px}
.m{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 14px;min-width:0;flex:1 1 260px}
.m b{display:block}.m span{color:var(--mut);font-size:.85rem;font-family:var(--mono);word-break:break-word}
.wrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:640px}
th,td{padding:10px 14px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
tr:last-child td{border-bottom:0}
th{font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);font-weight:600;background:var(--bg)}
td.task b{display:block}td.task span,.sub{color:var(--mut);font-size:.82rem}
.big{font-size:1.25rem;font-weight:600;font-variant-numeric:tabular-nums}
.num{font-variant-numeric:tabular-nums}
.pending{background:var(--pend);color:var(--mut);font-style:italic}
.pill{display:inline-block;font-size:.74rem;padding:1px 8px;border-radius:99px;margin-top:4px;font-weight:600}
.pill.good{background:var(--goodbg);color:var(--good)}.pill.warn{background:var(--warnbg);color:var(--warn)}.pill.bad{background:var(--badbg);color:var(--bad)}
h2{font-size:1.05rem;margin:28px 0 8px}
.notes{color:var(--mut);max-width:75ch;margin:0;padding-left:1.1em}.notes li{margin-bottom:6px}
code{font-family:var(--mono);font-size:.85em}
</style>
<main>
<h1>DiT Perf Matrix</h1>
<p class="lede">Wan2.2 and MiniMax-H3 wall-clock on Blackhole Galaxy 4x8, one workload per row. Every chip runs at the same TDP limit (__TDP__ W), read back from the runtime before each run. Add a machine by running <code>run_matrix.py run</code> on it and rebuilding this page.</p>
<div class="machines">__MACHINES__</div>
<div class="wrap"><table>
<thead><tr><th>Workload</th>__HEAD__</tr></thead>
<tbody>__BODY__</tbody></table></div>
<h2>Notes</h2>
<ul class="notes">__NOTES__</ul>
</main>
"""


def build(runs: list[dict], pending: list[str], extra_notes: list[str] | None = None) -> str:
    cols = [r["label"] for r in runs] + pending
    machines = ""
    for r in runs:
        m = r["machine"]
        machines += (
            f'<div class="m"><b>{html.escape(r["label"])}</b><span>{html.escape(str(m.get("host", "?")))} · '
            f'{m.get("chips", "?")} chips · {html.escape(str(m.get("git_branch", "?")))} @ {html.escape(str(m.get("git_sha", "?")))}</span></div>'
        )
    for p in pending:
        machines += f'<div class="m"><b>{html.escape(p)}</b><span>not run yet</span></div>'
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    qhead = "".join(f"<th>{html.escape(c)} quality</th>" for c in cols)
    body = ""
    cases = [c for c in ORDER if any(c in r["rows"] for r in runs)] or ORDER
    for case in cases:
        t, sub = TITLES.get(case, (case, ""))
        body += f'<tr><td class="task"><b>{html.escape(t)}</b><span>{html.escape(sub)}</span></td>'
        body += "".join(cell(r["rows"].get(case)) for r in runs)
        body += "".join('<td class="pending">not run</td>' for _ in pending) + "</tr>"
    qbody = ""
    for case in cases:
        t, sub = TITLES.get(case, (case, ""))
        qbody += f'<tr><td class="task"><b>{html.escape(t)}</b><span>{html.escape(sub)}</span></td>'
        qbody += "".join(quality(r["rows"].get(case)) for r in runs)
        qbody += "".join('<td class="pending">–</td>' for _ in pending) + "</tr>"
    tdp = next((r["machine"].get("tdp_requested") for r in runs if r["machine"].get("tdp_requested")), 130)
    notes = [
        "Time is end-to-end seconds for one warm generation (text encode, denoise, VAE decode, and for H3 audio decode); weight loading and kernel compile are excluded.",
        "“over Wan limit” means the run finished and was measured, but a stage time fell outside the limits Wan's own test uses for a full-DRAM device. The time shown is real.",
        "Quality cells show CLIP prompt alignment and two VBench dimensions. Pass/fail bars exist only for H3 T2VA (fox prompt) and Wan (its VBench gate); FL2VA and Ref2VA are conditioned on other content, so they are reported without a bar.",
    ]
    notes += extra_notes or []
    out = PAGE.replace("__TDP__", str(tdp)).replace("__MACHINES__", machines).replace("__HEAD__", head)
    out = out.replace("__BODY__", body).replace("__NOTES__", "".join(f"<li>{n}</li>" for n in notes))
    quality_block = (
        f'<h2>Output quality</h2><div class="wrap"><table><thead><tr><th>Workload</th>{qhead}</tr></thead>'
        f"<tbody>{qbody}</tbody></table></div>"
    )
    return out.replace("<h2>Notes</h2>", quality_block + "<h2>Notes</h2>", 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=DIR")
    ap.add_argument("--pending", action="append", default=[], metavar="LABEL")
    ap.add_argument("--note", action="append", default=[], help="extra line for the Notes list (HTML allowed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = []
    for spec in a.run:
        label, _, d = spec.partition("=")
        runs.append(load(label, Path(d).expanduser()))
    Path(a.out).write_text(build(runs, a.pending, a.note))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
