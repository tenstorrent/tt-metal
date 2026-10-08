#!/usr/bin/env python3
"""Generate $DREAM_HOME/<c>/history.md (read by workers) and tree.html (for humans) from git + the ledger.

history.py --campaign rmsnorm-prefill [--out-dir DIR]
"""

import argparse
import datetime
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream.campaign import load_campaign  # noqa: E402
from dream.tree import load_round, read_node_file, recorded_rounds  # noqa: E402


def reflection_next(text: str | None) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lower().startswith("## what a child"):
            return next((l.strip("- ").strip() for l in lines[i + 1 :] if l.strip() and not l.startswith("#")), "")
    return ""


def collect(c):
    base_p = c.ledger / "baseline.json"
    baseline = json.loads(base_p.read_text()) if base_p.exists() else None
    rounds = []
    for r in recorded_rounds(c):
        rr = load_round(c, r)
        step_of = {}
        for d in rr.decisions:
            if d.get("type") == "decision":
                for it in d.get("batch", []):
                    step_of[it["node"]] = d["step"]
        nodes = []
        for n in rr.nodes:
            score = json.loads(read_node_file(c, n.node_id, "eval/score.json") or "{}")
            nodes.append(
                {
                    "obs": n,
                    "shapes": score.get("shapes", {}),
                    "next": reflection_next(read_node_file(c, n.node_id, "reflection.md")),
                    "step": step_of.get(n.node_id, n.attempt),
                }
            )
        rounds.append({"rr": rr, "nodes": nodes, "closed": rr.closed(), "lost": rr.lost()})
    return baseline, rounds


def fmt_score(n) -> str:
    return f"{n.score:.4f}" if n.ok else f"✗ {n.fail_class}"


def write_md(c, baseline, rounds, path: Path):
    allnodes = [x for r in rounds for x in r["nodes"]]
    valid = [x for x in allnodes if x["obs"].ok]
    best = max(valid, key=lambda x: x["obs"].score) if valid else None
    shape_ids = [s["id"] for s in c.cfg["shapes"]]
    L = [f"# {c.name}: discovery history", ""]
    L.append(
        f"Generated {datetime.datetime.now().isoformat(timespec='minutes')} from git tags `dream/{c.name}/n/*` and the ledger."
    )
    L.append("")
    if baseline:
        L.append(
            f"**Baseline** (µs, per-chip mean device kernel time; noise ±{baseline['noise_pct']}%): "
            + ", ".join(f"{sid} {baseline['shapes'][sid]['us_chip_mean']:.2f}" for sid in shape_ids)
        )
    if best:
        b = best["obs"]
        L.append(f"**Best valid:** `{b.node_id}` score {b.score:.4f} ({b.mechanism})")
    L.append(
        f"**Attempts:** {len(allnodes)} committed ({len(valid)} valid) over {len(rounds)} round(s); "
        f"{sum(r['lost'] for r in rounds)} lost."
    )
    L.append("")
    if valid:
        L += ["## Leaderboard (top 10 valid)", ""]
        L.append("| node | score | mechanism | " + " | ".join(f"{s} µs" for s in shape_ids) + " |")
        L.append("|---|---|---|" + "---|" * len(shape_ids))
        for x in sorted(valid, key=lambda x: -x["obs"].score)[:10]:
            us = [f"{x['shapes'].get(s, {}).get('us_chip_mean', float('nan')):.2f}" for s in shape_ids]
            L.append(f"| `{x['obs'].node_id}` | {x['obs'].score:.4f} | {x['obs'].mechanism} | " + " | ".join(us) + " |")
        L.append("")
    for r in rounds:
        rr, m = r["rr"], r["rr"].manifest
        L.append(f"## Round r{rr.round:02d}")
        L.append("")
        if m:
            L.append(
                f"Policy `{m.get('policy')}` (beta {m.get('beta')}), W={m.get('W')}, R={m.get('R')}, "
                f"root `{m.get('round_root')}`. Plan: {m.get('reason')}"
            )
            L.append("")
        by_branch = {}
        for x in r["nodes"]:
            by_branch.setdefault(x["obs"].branch, []).append(x)
        for b in sorted(by_branch):
            closed = r["closed"].get(b)
            L.append(f"### Branch b{b:02d}" + (f" (closed: {closed})" if closed else ""))
            L.append("")
            L.append("| node | parent | mechanism | tags | score | Δ parent | fail_class | next |")
            L.append("|---|---|---|---|---|---|---|---|")
            for x in by_branch[b]:
                n = x["obs"]
                d = f"{n.delta_vs_parent:+.4f}" if n.delta_vs_parent is not None else "-"
                L.append(
                    f"| `{n.node_id}` | {n.parent} | {n.mechanism} | {', '.join(n.tags)} | "
                    f"{n.score:.4f} | {d} | {n.fail_class} | {x['next']} |"
                )
            L.append("")
    P = c.attempts_rel("<id>")
    L += [
        "## Reading a node in full",
        "",
        "```bash",
        f"git show dream/{c.name}/n/<id>:{P}/proposal.md",
        f"git show dream/{c.name}/n/<id>:{P}/reflection.md",
        f"git show dream/{c.name}/n/<id>:{P}/eval/summary.md",
        f"git diff dream/{c.name}/n/<id>~1 dream/{c.name}/n/<id> -- . ':!agent_orch'",
        "```",
    ]
    path.write_text("\n".join(L) + "\n")


CSS = """
:root{--bg:#f5f6f8;--surface:#fff;--fg:#1a2130;--muted:#5d6779;--line:#d8dde6;--accent:#2a5bd7;
--win:#1f8a5b;--win-soft:#d9f1e5;--win-strong:#a9e0c3;--flat:#eceef2;--fail:#c2413b;--fail-soft:#f8dedc}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11151d;--surface:#181e29;--fg:#e4e8f0;
--muted:#98a2b5;--line:#2b3342;--accent:#7ea2ff;--win:#5cd69c;--win-soft:#173a2b;--win-strong:#1f5a3f;--flat:#252c39;
--fail:#ff8a80;--fail-soft:#43201f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11151d;--surface:#181e29;--fg:#e4e8f0;--muted:#98a2b5;--line:#2b3342;--accent:#7ea2ff;
--win:#5cd69c;--win-soft:#173a2b;--win-strong:#1f5a3f;--flat:#252c39;--fail:#ff8a80;--fail-soft:#43201f;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif;padding-inline:18px;padding-block:28px 60px}
main{max-width:1000px;margin:0 auto;display:flex;flex-direction:column;gap:36px}
h1{font-size:1.8rem;margin:0}h2{font-size:1.25rem;margin:0 0 10px}
.mono,td.num{font-family:ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.stats{display:flex;flex-wrap:wrap;gap:12px}.stat{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:150px}
.stat b{display:block;font-size:1.3rem}.stat span{color:var(--muted);font-size:.8rem}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--muted);font-weight:600}
.grid td.cell{text-align:center;border-radius:4px;font-family:ui-monospace,Menlo,monospace;font-size:.8rem;border:2px solid var(--bg)}
.win{background:var(--win-soft);color:var(--win)}.big{background:var(--win-strong);color:var(--win);font-weight:700}
.flat{background:var(--flat);color:var(--muted)}.fail{background:var(--fail-soft);color:var(--fail)}
.best{outline:2px solid var(--win);outline-offset:-2px}tr.closed td{opacity:.45}
.note{color:var(--muted);font-size:.85rem}
.bar{height:10px;border-radius:3px;background:var(--accent)}.bar.base{background:var(--line)}
svg text{fill:var(--muted);font-size:11px}svg .axis{stroke:var(--line)}svg .best{stroke:var(--win);fill:none;stroke-width:2.5}
svg .pt{fill:var(--accent);opacity:.55}svg .ptf{fill:var(--fail);opacity:.55}svg .rb{stroke:var(--muted);stroke-dasharray:4 4}
"""


def cell_class(n, noise):
    if not n.ok:
        return "fail"
    if n.score >= 1.10:
        return "big"
    if n.score > 1 + noise / 100:
        return "win"
    return "flat"


def progress_svg(seq, rounds_bounds):
    if not seq:
        return "<p class='note'>No attempts yet.</p>"
    W, H, pl, pr, pt, pb = 760, 240, 46, 12, 12, 30
    scores = [s for s, ok in seq if ok] + [1.0]
    lo, hi = min(scores + [1.0]) - 0.02, max(scores) + 0.02
    n = len(seq)
    X = lambda i: pl + (W - pl - pr) * (i / max(1, n))  # noqa: E731
    Y = lambda v: pt + (H - pt - pb) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Best score vs attempts">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        out.append(f'<line class="axis" x1="{pl}" x2="{W - pr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
        out.append(f'<text x="{pl - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    for i, label in rounds_bounds:
        out.append(f'<line class="rb" x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{pt}" y2="{H - pb}"/>')
        out.append(f'<text x="{X(i) + 4:.1f}" y="{pt + 10}">{label}</text>')
    best, pts = 1.0, [(X(0), Y(1.0))]
    for i, (s, ok) in enumerate(seq, start=1):
        if ok:
            out.append(f'<circle class="pt" cx="{X(i):.1f}" cy="{Y(s):.1f}" r="3"/>')
            best = max(best, s)
        else:
            out.append(f'<circle class="ptf" cx="{X(i):.1f}" cy="{H - pb - 3}" r="2.5"/>')
        pts += [(X(i), pts[-1][1]), (X(i), Y(best))]
    out.append('<polyline class="best" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
    out.append(f'<text x="{(W + pl) / 2:.0f}" y="{H - 6}" text-anchor="middle">attempts (cumulative)</text></svg>')
    return "".join(out)


def write_html(c, baseline, rounds, path: Path):
    noise = baseline["noise_pct"] if baseline else 2.0
    allnodes = [x for r in rounds for x in r["nodes"]]
    valid = [x for x in allnodes if x["obs"].ok]
    best = max(valid, key=lambda x: x["obs"].score) if valid else None
    e = html.escape
    H = [f"<title>{e(c.name)} discovery tree</title><style>{CSS}</style><main>"]
    H.append(
        f"<header><h1>{e(c.name)}</h1><p class='note'>{e(c.cfg.get('op', ''))} · generated "
        f"{datetime.datetime.now().isoformat(timespec='minutes')}</p></header>"
    )
    H.append("<section class='stats'>")
    H.append(
        f"<div class='stat'><b>{best['obs'].score:.4f}</b><span>best score ({e(best['obs'].node_id)})</span></div>"
        if best
        else "<div class='stat'><b>-</b><span>best score</span></div>"
    )
    H.append(f"<div class='stat'><b>{len(allnodes)}</b><span>attempts ({len(valid)} valid)</span></div>")
    H.append(f"<div class='stat'><b>{len(rounds)}</b><span>rounds</span></div>")
    H.append(f"<div class='stat'><b>±{noise}%</b><span>noise band</span></div></section>")

    seq, bounds, count = [], [], 0
    for r in rounds:
        bounds.append((count, f"r{r['rr'].round:02d}"))
        for x in sorted(r["nodes"], key=lambda x: (x["step"], x["obs"].branch)):
            seq.append((x["obs"].score, x["obs"].ok))
            count += 1
    H.append(
        "<section><h2>Best score vs. attempts</h2>"
        + progress_svg(seq, bounds)
        + "<p class='note'>Line: best valid score so far. Dots: each valid attempt; red dots on the floor: failed attempts.</p></section>"
    )

    for r in rounds:
        rr, m = r["rr"], r["rr"].manifest
        maxa = max([len(v) for v in rr.branches.values()] + [m.get("R") or 0, 1])
        H.append(
            f"<section><h2>Round r{rr.round:02d}</h2><p class='note'>policy {e(str(m.get('policy')))}, "
            f"W={m.get('W')}, R={m.get('R')}, {len(rr.nodes)} attempts, {rr.steps_done()} steps. {e(str(m.get('reason', '')))}</p>"
        )
        H.append(
            "<div class='wrap'><table class='grid'><tr><th>branch</th>"
            + "".join(f"<th>a{a:02d}</th>" for a in range(1, maxa + 1))
            + "<th>first idea</th></tr>"
        )
        for b in sorted(rr.branches):
            nodes = rr.branches[b]
            closed = r["closed"].get(b)
            H.append(f"<tr class='{'closed' if closed else ''}'><td class='mono'>b{b:02d}</td>")
            for a in range(maxa):
                if a < len(nodes):
                    n = nodes[a]
                    cls = cell_class(n, noise) + (" best" if best and n.node_id == best["obs"].node_id else "")
                    label = f"{n.score:.3f}" if n.ok else "✗ " + n.fail_class.replace("_error", "").replace("_fail", "")
                    H.append(f"<td class='cell {cls}' title='{e(n.node_id)}: {e(n.mechanism)}'>{e(label)}</td>")
                else:
                    H.append("<td></td>")
            H.append(f"<td>{e(nodes[0].mechanism)}{' · closed: ' + e(closed) if closed else ''}</td></tr>")
        H.append("</table></div></section>")

    if best and baseline:
        H.append(
            f"<section><h2>Best node vs. baseline</h2><p class='note'>{e(best['obs'].node_id)}: {e(best['obs'].mechanism)}</p>"
            "<div class='wrap'><table><tr><th>shape</th><th>baseline µs</th><th>best µs</th><th>speedup</th><th></th></tr>"
        )
        mx = max(baseline["shapes"][s]["us_chip_mean"] for s in baseline["shapes"])
        for s in c.cfg["shapes"]:
            sid = s["id"]
            base_us = baseline["shapes"][sid]["us_chip_mean"]
            nu = best["shapes"].get(sid, {}).get("us_chip_mean")
            if nu is None:
                continue
            H.append(
                f"<tr><td class='mono'>{e(sid)}</td><td class='num'>{base_us:.2f}</td><td class='num'>{nu:.2f}</td>"
                f"<td class='num'>{base_us / nu:.3f}×</td><td style='min-width:160px'><div class='bar base' style='width:{base_us / mx * 100:.0f}%'></div>"
                f"<div class='bar' style='width:{nu / mx * 100:.0f}%;margin-top:3px'></div></td></tr>"
            )
        H.append("</table></div></section>")

    H.append(
        "<section><h2>Rounds</h2><div class='wrap'><table><tr><th>round</th><th>policy</th><th>attempts</th><th>valid</th>"
        "<th>lost</th><th>best</th></tr>"
    )
    for r in rounds:
        rr = r["rr"]
        ok = [n.score for n in rr.nodes if n.ok]
        H.append(
            f"<tr><td class='mono'>r{rr.round:02d}</td><td>{e(str(rr.manifest.get('policy', '-')))}</td>"
            f"<td class='num'>{len(rr.nodes)}</td><td class='num'>{len(ok)}</td><td class='num'>{r['lost']}</td>"
            f"<td class='num'>{max(ok) if ok else 1.0:.4f}</td></tr>"
        )
    H.append("</table></div></section></main>")
    path.write_text("\n".join(H) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    out = args.out_dir or c.home
    out.mkdir(parents=True, exist_ok=True)
    baseline, rounds = collect(c)
    write_md(c, baseline, rounds, out / "history.md")
    write_html(c, baseline, rounds, out / "tree.html")
    print(f"wrote {out / 'history.md'} and {out / 'tree.html'}")


if __name__ == "__main__":
    main()
