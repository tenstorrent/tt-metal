# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Render TOPK_LEDGER.html's data tables from competition-sweep outputs.

The ledger's prose (attribution, takeaways, methodology) is hand-written; the
NUMBERS are not. This script regenerates the three marker-delimited regions
from the deterministic sweep outputs, so the whole pipeline is scripted:

    _canonical_topk_sweep.py --competition [--with-blaze]  ->  CSV/result JSONs
    _canonical_topk_sweep.py --competition --layers-competition op \
        --ks K --ns W --op-num-slices P                    ->  P-sweep cells
    _topk_ledger_render.py                                 ->  ledger tables

Regions (HTML comments in TOPK_LEDGER.html):
    <!-- EXEC_NUMBERS --> ... <!-- /EXEC_NUMBERS -->      time / vs-stock /
        cores / P-range rows of the executive summary (anchor k=2048 W=65536)
    <!-- COMPETITION_TABLE --> ... <!-- /COMPETITION_TABLE -->
    <!-- PSWEEP_TABLE --> ... <!-- /PSWEEP_TABLE -->

Usage:
    python tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py \
        --competition-dir <dir> --psweep-dir <dir> [--ledger TOPK_LEDGER.html]

Anything this script cannot find (no blaze cell, missing P points) renders as
an em dash — it never invents a number.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

REPO = os.environ.get("TT_METAL_HOME", os.path.dirname(os.path.abspath(__file__)) + "/../../../../..")

ANCHOR_K, ANCHOR_W = 2048, 65536


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def us_fmt(x, cls=""):
    if x is None:
        return '<td class="n flat">—</td>'
    s = f"{x:,.1f}" if x < 10000 else f"{x / 1000:,.1f} ms"
    c = f" {cls}" if cls else ""
    return f'<td class="n{c}">{s}</td>'


def load_competition(comp_dir):
    path = os.path.join(comp_dir, "competition_table.csv")
    return list(csv.DictReader(open(path)))


def load_scenarios(scen_dir):
    path = os.path.join(scen_dir, "scenarios_table.csv")
    return list(csv.DictReader(open(path)))


def load_psweep(psweep_dir):
    """Return {(k, W): {P: us}} from per-cell result JSONs (op layer, _pN ids)."""
    out = {}
    for f in glob.glob(os.path.join(psweep_dir, "results", "comp_op_k*_p*.json")):
        m = re.search(r"comp_op_k(\d+)_w(\d+)_p(\d+)", os.path.basename(f))
        if not m:
            continue
        rec = json.load(open(f))
        us = fnum(rec.get("ns_median"))
        if rec.get("status") == "MEASURED" and us is not None:
            out.setdefault((int(m.group(1)), int(m.group(2))), {})[int(m.group(3))] = us / 1000.0
    return out


def render_competition_table(rows):
    body = []
    for r in rows:
        k, W = int(r["k"]), int(r["W"])
        pb, ro, op = fnum(r.get("prebranch_us")), fnum(r.get("routed_us")), fnum(r.get("op_us"))
        rf, bl, osk = fnum(r.get("roofline_us")), fnum(r.get("blaze_us")), fnum(r.get("opstock_us"))
        gap = f'<td class="n flat">{op / rf:.1f}×</td>' if op and rf else '<td class="n flat">—</td>'
        spr = f'<td class="n win">{pb / ro:,.0f}×</td>' if pb and ro else '<td class="n flat">—</td>'
        spo = f'<td class="n win">{pb / op:,.0f}×</td>' if pb and op else '<td class="n flat">—</td>'
        blc = f'<td class="n">{bl:,.1f}</td>' if bl else '<td class="n flat">—</td>'
        body.append(
            f'      <tr><td class="n">{k:,}</td><td class="n">{W:,}</td>{us_fmt(pb)}{us_fmt(osk)}'
            f'{us_fmt(ro, "win")}{us_fmt(op, "win")}<td class="n">{r.get("op_cores", "")}</td>{blc}{us_fmt(rf)}{gap}{spr}{spo}</tr>'
        )
    head = (
        '<thead><tr><th class="n">k</th><th class="n">N</th><th class="n">stock ttnn.topk</th>'
        '<th class="n">stock topk_large_indices</th><th class="n ours">ttnn.topk (our routing)</th>'
        '<th class="n ours">topk_large_indices (our multi-core)</th><th class="n">cores</th>'
        '<th class="n">blaze</th><th class="n">roofline</th><th class="n">gap ours/roof</th>'
        '<th class="n">speedup: routing</th><th class="n">speedup: multi-core op</th></tr></thead>'
    )
    return (
        f'  <div class="tablewrap"><table>\n    {head}\n    <tbody>\n'
        + "\n".join(body)
        + "\n    </tbody>\n  </table></div>"
    )


def render_psweep_table(psweep, comp_rows):
    stock = {}
    for r in comp_rows:
        v = fnum(r.get("prebranch_us"))
        if v:
            stock[(int(r["k"]), int(r["W"]))] = v
    body = []
    for (k, W), pts in sorted(psweep.items()):
        cells = "".join(
            f'<td class="n{" win" if p == min(pts, key=pts.get) else ""}">{pts[p]:,.1f}</td>' for p in sorted(pts)
        )
        header_ps = "".join(f'<th class="n">P={p}</th>' for p in sorted(pts))
        sb = stock.get((k, W))
        rng = f"{sb / max(pts.values()):,.0f}×–{sb / min(pts.values()):,.0f}× vs stock ttnn.topk" if sb else "—"
        body.append(
            f'  <div class="tablewrap"><table><thead><tr><th>k={k:,} N={W:,}</th>{header_ps}<th>speedup range</th></tr></thead>'
            f'<tbody><tr><td>runtime (µs)</td>{cells}<td class="n">{rng}</td></tr></tbody></table></div>'
        )
    return "\n".join(body)


def render_model_scenarios(rows):
    """MODEL_SCENARIOS region. One speedup semantics, stated in the header:
    speedup = today / best-ours, always. 'best ours' names which variant won
    (routed ttnn.topk or the direct op); 'to capture it' says in plain words
    what adopting the best number costs. Numbers only from MEASURED cells."""
    import html as _html

    body = []
    for r in rows:
        esc = lambda s: _html.escape(s or "")  # noqa: E731
        shape = f"{int(r['rows'])}\u00d7{int(r['n']):,}, k={int(r['k'])}"
        today_e = r.get("today_engine", "")
        today_label = {"stocknow": "ttnn.topk", "routed": "ttnn.topk", "op": "topk_large_indices"}.get(today_e, today_e)
        today = fnum(r.get("today_us"))
        ro, op = fnum(r.get("routed_us")), fnum(r.get("op_us"))
        today_cores = r.get(f"{today_e}_cores", "")
        if today is not None and today_cores:
            today_td = f'<td class="n">{today:,.1f} <span class="flat">@{today_cores}c</span></td>'
        else:
            today_td = us_fmt(today)

        # best ours: the cheaper measured variant, with its core count
        cands = [(v, n, r.get(f"{key}_cores", "")) for v, n, key in
                 ((ro, "routed ttnn.topk", "routed"), (op, "direct op", "op")) if v is not None]
        if cands:
            best, best_name, best_cores = min(cands)
            bc = f" @{best_cores}c" if best_cores else ""
            best_td = f'<td class="n ours">{best:,.1f}{bc} <span class="flat">({best_name})</span></td>'
        else:
            best, best_name, best_td = None, None, '<td class="n flat">not run</td>'

        if today is not None and best is not None:
            ratio = today / best
            rs = f"{ratio:,.0f}\u00d7" if ratio >= 100 else f"{ratio:,.1f}\u00d7"
            sp = f'<td class="n win">{rs}</td>' if ratio >= 1.05 else f'<td class="n flat">{rs}</td>'
        else:
            sp = '<td class="n flat">\u2014</td>'

        # what adopting the best number costs, in plain words
        if best is None:
            capture = "\u2014"
        elif today_e == "op" and best_name == "direct op":
            capture = "nothing \u2014 already on the best engine"
        elif best_name == "routed ttnn.topk":
            capture = "same API; drop indices_tensor/sub_core_grids/stable from the call"
        elif today_label == "ttnn.topk" and best_name == "direct op":
            if int(r["k"]) < 16:
                capture = "switch to the direct op (k rounds to 16, slice back); or extend the route \u2014 A/B vs envelope pending"
            else:
                capture = "switch the call to the direct op (RM output, u32 indices)"
                if ro is not None and today is not None and ro < today:
                    capture += (
                        f"; or same API \u2014 drop indices_tensor/sub_core_grids/stable: "
                        f"{ro:,.1f} \u00b5s ({today / ro:,.1f}\u00d7)"
                    )
        else:
            capture = "\u2014"

        note = esc(r.get("notes", ""))
        if r.get("calls_note"):
            note = f'{note} <span class="flat">[{esc(r["calls_note"])}]</span>' if note else esc(r["calls_note"])
        body.append(
            f'      <tr><td>{esc(r["scenario"])}</td>'
            f'<td>{esc(r.get("model", ""))}<br><code>{esc(r.get("callsite", ""))}</code></td>'
            f'<td class="n">{shape}</td>'
            f'<td class="n">{esc(today_label)}</td>{today_td}'
            f"{best_td}{sp}"
            f"<td>{capture}</td>"
            f"<td>{note}</td></tr>"
        )
    head = (
        '  <div class="tablewrap"><table style="min-width: 1700px">\n'
        '    <thead><tr><th>scenario</th><th>model + callsite</th><th class="n">shape (rows\u00d7N, k)</th>'
        '<th>calls today</th><th class="n">today \u00b5s</th>'
        '<th class="n ours">best ours \u00b5s (variant)</th>'
        '<th class="n">speedup (today \u00f7 best ours)</th>'
        '<th>to capture it</th><th>note</th></tr></thead>'
    )
    return head + "\n    <tbody>\n" + "\n".join(body) + "\n    </tbody>\n  </table></div>"

def render_exec_numbers(rows, psweep):
    r = next((x for x in rows if int(x["k"]) == ANCHOR_K and int(x["W"]) == ANCHOR_W), None)
    if r is None:
        sys.exit(f"anchor ({ANCHOR_K},{ANCHOR_W}) missing from competition CSV")
    pb, ro, op = fnum(r.get("prebranch_us")), fnum(r.get("routed_us")), fnum(r.get("op_us"))
    rf, bl, osk = fnum(r.get("roofline_us")), fnum(r.get("blaze_us")), fnum(r.get("opstock_us"))
    opc = r.get("op_cores", "")
    pts = psweep.get((ANCHOR_K, ANCHOR_W), {})
    if pts:
        pmin, pmax = min(pts), max(pts)
        best, worst = min(pts.values()), max(pts.values())
        prange_op = f"{worst:,.1f}→{best:,.1f} µs over P={pmin}–{pmax}"
        sprange_op = f"{pb / worst:,.0f}×–{pb / best:,.0f}× vs stock" if pb else "—"
    else:
        prange_op = sprange_op = "—"

    def n(v, suffix="", stars=""):
        return f"{v:,.1f}{suffix}{stars}" if v is not None else "—"

    def ratio(a, b):
        return f"{a / b:,.0f}×" if a is not None and b not in (None, 0) else "—"

    rows_html = [
        f'      <tr><td>time</td><td class="n flat">{n(rf, " µs")}</td><td class="n flat">{n(bl, " µs", "†")}</td>'
        f'<td class="n ours gl"><strong>{n(op, " µs")}</strong></td><td class="n gr">{n(osk, " µs", "‡")}</td>'
        f'<td class="n ours gl"><strong>{n(ro, " µs")}</strong></td><td class="n gr">{n(pb, " µs")}</td></tr>',
        f'      <tr><td>vs stock ttnn.topk</td><td class="n flat">—</td><td class="n flat">{ratio(pb, bl)} (fused†)</td>'
        f'<td class="n ours gl"><strong>{ratio(pb, op)}</strong></td><td class="n gr">{ratio(pb, osk)}</td>'
        f'<td class="n ours gl"><strong>{ratio(pb, ro)}</strong></td><td class="n gr">1×</td></tr>',
        f'      <tr><td>cores</td><td class="n flat">128 (assumed)</td><td class="n flat">32</td>'
        f'<td class="n ours gl">{opc}</td><td class="n gr">1/row</td><td class="n ours gl">{opc}</td><td class="n gr">1</td></tr>',
        f'      <tr><td>runtime over supported P</td><td class="n flat">—</td><td class="n flat">—</td>'
        f'<td class="n ours gl">{prange_op} ({sprange_op})</td><td class="n gr">—</td>'
        f'<td class="n ours gl">tracks the op + fixed layout envelope</td><td class="n gr">—</td></tr>',
    ]
    return "\n".join(rows_html)


def splice(text, marker, payload):
    a, b = f"<!-- {marker} -->", f"<!-- /{marker} -->"
    if a not in text or b not in text:
        sys.exit(f"marker {marker} missing from ledger")
    pre, rest = text.split(a, 1)
    _, post = rest.split(b, 1)
    return pre + a + "\n" + payload + "\n" + b + post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition-dir", required=True)
    ap.add_argument("--psweep-dir", required=True)
    ap.add_argument(
        "--scenarios-dir", default=None, help="dir with scenarios_table.csv (optional; MODEL_SCENARIOS region)"
    )
    ap.add_argument("--ledger", default=os.path.join(REPO, "TOPK_LEDGER.html"))
    args = ap.parse_args()

    rows = load_competition(args.competition_dir)
    psweep = load_psweep(args.psweep_dir)

    t = open(args.ledger).read()
    t = splice(t, "EXEC_NUMBERS", render_exec_numbers(rows, psweep))
    t = splice(t, "COMPETITION_TABLE", render_competition_table(rows))
    t = splice(t, "PSWEEP_TABLE", render_psweep_table(psweep, rows))
    n_scen = ""
    if args.scenarios_dir:
        scen_rows = load_scenarios(args.scenarios_dir)
        t = splice(t, "MODEL_SCENARIOS", render_model_scenarios(scen_rows))
        n_scen = f", {len(scen_rows)} scenario rows"
    open(args.ledger, "w").write(t)
    print(f"rendered {args.ledger}: {len(rows)} competition rows, {len(psweep)} P-sweep shapes{n_scen}")


if __name__ == "__main__":
    main()
