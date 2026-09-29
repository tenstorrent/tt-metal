# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Render one PR comment from the per-architecture ``summary.json`` files.

The layout follows how SFPU bounty PRs already report results (old/new cycles,
speedup, old/new max ULP per arch), so a reviewer reads it the same way. Every
number comes from the summaries; the AI summary, if any, is inserted later at
``AI_SUMMARY_MARKER`` by the workflow and is labelled as such.
"""

import json
import math
from pathlib import Path

from perf import ROWS_PER_TILE

COMMENT_MARKER = "<!-- llk-sfpu-report -->"
AI_SUMMARY_MARKER = "<!-- llk-sfpu-report:ai-summary -->"
STALE_DAYS = 14

_ARCH = {"wormhole": "Wormhole", "blackhole": "Blackhole"}
_FMT = {"Float16_b": "bf16", "Float16": "fp16", "Float32": "fp32", "Bfp8_b": "bfp8"}


def _fmt_pair(s):
    a, _, b = s.partition("->")
    a, b = _FMT.get(a, a), _FMT.get(b, b)
    return a if a == b else f"{a}→{b}"


def _num(x, digits=0):
    if x is None:
        return "–"
    if isinstance(x, float) and x != x:
        return "–"
    return f"{x:,.{digits}f}"


def _speedup(base, head):
    if not base or not head:
        return "–"
    s = base / head
    txt = f"{s:.2f}x"
    return f"**{txt}**" if s >= 1.005 else txt


def _flag(cell):
    if cell.get("regression"):
        return "⚠️ "
    return ""


def _bold_if_better(new, old, lower_is_better=True):
    if new is None or old is None:
        return _num(new)
    new_r, old_r = round(new), round(old)
    better = new_r < old_r if lower_is_better else new_r > old_r
    return f"**{_num(new)}**" if better else _num(new)


def perf_section(summaries):
    lines = [
        "### Performance",
        "",
        "Cycles per tile, `TILE_LOOP`, speed of light; median of "
        f"{summaries[0]['iterations']} interleaved runs per side. `math` is `MATH_ISOLATE` "
        "(the SFPU kernel alone), `L1→L1` the whole unpack→math→pack path. "
        f"`≈/row` is `math` ÷ {ROWS_PER_TILE}, the per-iteration unit bounty issues use "
        "(derived, includes loop overhead). ⚠️ = slower beyond the gate thresholds "
        f"(BH {summaries[0]['thresholds']['blackhole']:.0%}, WH {summaries[0]['thresholds']['wormhole']:.0%}, "
        f"and > {summaries[0]['thresholds']['min_cycles']:.0f} cycles per loop).",
        "",
        "| arch | op | format | dest_acc | approx | math old | math new | speedup | ≈/row old → new | L1→L1 old → new | text size old → new |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for s in summaries:
        for family, by_schedule in s["perf"].items():
            for row in by_schedule.get("loadmacro", []):
                i, _, o = row["formats"].partition("->")
                rows.append(
                    ((s["arch"], row["op"], i, o, row["approx"], row["dest_acc"]), row)
                )
    header = lines[-2:]
    main, rest = [], []
    for key, row, cells in _merge_approx(rows, _perf_cells):
        # The main table: every row that moved, plus each op's bf16 rows so the
        # op's cost is always visible. The rest (the other format pairs of an
        # unchanged op) folds away.
        moved = any(
            row.get(rt, {}).get("regression") or row.get(rt, {}).get("improvement")
            for rt in ("MATH_ISOLATE", "L1_TO_L1")
        )
        primary = key[2] == key[3] == "Float16_b"
        (main if moved or primary else rest).append(
            _row_prefix(key) + "| " + " | ".join(cells) + " |"
        )
    lines += main
    if not rows:
        lines.append("| – | no op was measured | | | | | | | | | |")
    if rest:
        lines += [
            "",
            f"<details><summary>{len(rest)} more format combination(s), none beyond the thresholds</summary>",
            "",
            *header,
            *rest,
            "",
            "</details>",
        ]
    lines += ["", _loadmacro_note(summaries), ""]
    return lines


def _perf_cells(row):
    m, l1 = row.get("MATH_ISOLATE", {}), row.get("L1_TO_L1", {})
    text = (
        f"{_num(row['text_base'])} → {_bold_if_better(row['text_head'], row['text_base'])} B"
        if row.get("text_base")
        else "–"
    )
    return (
        _num(m.get("base")),
        f"{_flag(m)}{_bold_if_better(m.get('head'), m.get('base'))}",
        _speedup(m.get("base"), m.get("head")),
        f"{_num((m.get('base') or 0) / ROWS_PER_TILE, 1)} → {_num((m.get('head') or 0) / ROWS_PER_TILE, 1)}",
        f"{_num(l1.get('base'))} → {_flag(l1)}{_bold_if_better(l1.get('head'), l1.get('base'))}",
        text,
    )


def _loadmacro_note(summaries):
    """SFPLOADMACRO-disabled rows, only for kernels whose code differs between schedules."""
    rows = []
    for s in summaries:
        for family, by_schedule in s["perf"].items():
            on = {
                (r["op"], r["formats"], r["dest_acc"], r["approx"]): r
                for r in by_schedule.get("loadmacro", [])
            }
            for r in by_schedule.get("no-loadmacro", []):
                k = (r["op"], r["formats"], r["dest_acc"], r["approx"])
                if (
                    k in on
                    and on[k].get("text_head") == r.get("text_head")
                    and on[k].get("text_base") == r.get("text_base")
                ):
                    continue  # no SFPLOADMACRO path: the same code either way
                m = r.get("MATH_ISOLATE", {})
                rows.append(
                    f"| {_ARCH[s['arch']]} | {r['op']} | {_fmt_pair(r['formats'])} | {r['dest_acc']} | {r['approx']} "
                    f"| {_num(m.get('base'))} | {_flag(m)}{_bold_if_better(m.get('head'), m.get('base'))} | {_speedup(m.get('base'), m.get('head'))} |"
                )
    if not rows:
        return "No measured kernel has an SFPLOADMACRO path: its code is the same with `-DDISABLE_SFPLOADMACRO`."
    return "\n".join(
        [
            "<details><summary>SFPLOADMACRO disabled (<code>-DDISABLE_SFPLOADMACRO</code>): the fallback schedule</summary>",
            "",
            "| arch | op | format | dest_acc | approx | math old | math new | speedup |",
            "|---|---|---|---|---|---:|---:|---:|",
            *rows,
            "",
            "</details>",
        ]
    )


_FMT_ORDER = {"Float16_b": 0, "Float16": 1, "Float32": 2, "Bfp8_b": 3}


def _merge_approx(rows, cells):
    """Rows identical but for ``approx`` become one row with approx ``Yes/No``.

    ``rows`` are ``(key, payload)`` with key ``(arch, op, in, out, approx, dest)``;
    ``cells(payload)`` is what the table shows. Many ops ignore the approx flag.
    """
    grouped = {}
    for key, payload in rows:
        arch, op, i, o, approx, dest = key
        grouped.setdefault((arch, op, i, o, dest, cells(payload)), []).append(
            (approx, payload)
        )
    out = []
    for (arch, op, i, o, dest, shown), items in grouped.items():
        approx = "/".join(sorted({a for a, _ in items}, reverse=True))
        out.append(((arch, op, i, o, approx, dest), items[0][1], shown))
    return sorted(
        out,
        key=lambda r: (r[0][0], r[0][1], _FMT_ORDER.get(r[0][2], 9), r[0][5], r[0][4]),
    )


def _acc_regressed(rec):
    """Worse overall: a higher max, more lanes worse than better, or new non-finite."""
    b, h = rec["base"], rec["head"]
    return (
        h["max"] > b["max"]
        or rec["worse"] > rec["better"]
        or h["nonfinite"] > b["nonfinite"]
    )


def _acc_cells(rec):
    b, h = rec["base"], rec["head"]
    flag = "⚠️ " if _acc_regressed(rec) else ""
    return (
        _num(b["max"]),
        f"{flag}{_bold_if_better(h['max'], b['max'])}",
        f"{_num(b['mean'], 3)} → {_num(h['mean'], 3)}",
        f"{b['le1']:.2%} → {h['le1']:.2%}",
        f"{_num(rec['worse'])} / {_num(rec['better'])}",
        f"{b['nonfinite']} → {h['nonfinite']}",
    )


def _row_prefix(key):
    arch, op, i, o, approx, dest = key
    return f"| {_ARCH[arch]} | {op} | {_fmt_pair(f'{i}->{o}')} | {dest} | {approx} "


def accuracy_section(summaries):
    head = [
        "| arch | op | format | dest_acc | approx | max old | max new | mean old → new | ≤1 ULP old → new | worse / better | non-finite old → new |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    changed, same = [], []
    for s in summaries:
        for rec in s["accuracy"]:
            if "base" not in rec:
                continue
            key = (
                s["arch"],
                *rec["key"][:2],
                rec["key"][2],
                rec["key"][3],
                rec["key"][4],
            )
            (same if rec.get("bit_identical") else changed).append((key, rec))
    lines = [
        "### Accuracy",
        "",
        "ULP steps against the correctly rounded reference (the golden of the nightly ULP sweep), "
        "over every finite input of the format (fp32: every 65,536th value). Lanes a step count "
        "cannot describe (NaN, subnormal input, padding) are left out here and covered by the edge "
        "cases below. `worse / better` counts lanes whose error changed.",
        "",
    ]
    if changed:
        lines += head + [
            _row_prefix(k) + "| " + " | ".join(cells) + " |"
            for k, _, cells in _merge_approx(changed, _acc_cells)
        ]
    else:
        lines.append(
            "The PR does not change any result: every swept variant returns the same bits on both sides."
        )
    if same:
        lines += [
            "",
            f"<details><summary>{len(same)} variant(s) return bit-identical results on both sides</summary>",
            "",
            *head,
            *[
                _row_prefix(k) + "| " + " | ".join(cells) + " |"
                for k, _, cells in _merge_approx(same, _acc_cells)
            ],
            "",
            "</details>",
        ]
    return lines + [""]


def _val(x):
    if x != x:
        return "nan"
    if x == 0:
        return "-0" if math.copysign(1.0, x) < 0 else "0"
    return f"{x:.9g}"


def _kind(x):
    """The part of a result that a changed value rarely should change."""
    if x != x:
        return "nan"
    if math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    if x == 0:
        return "-0" if math.copysign(1.0, x) < 0 else "+0"
    return "+" if x > 0 else "-"


def edge_section(summaries):
    """What the hardware returns for the special inputs, where the PR changed it."""
    severe, minor, nan_lost, nan_gained, total = [], [], [], [], 0
    for s in summaries:
        for rec in s["accuracy"]:
            sp = rec.get("specials")
            if not sp:
                continue
            total += 1
            op, i, o, approx, dest = rec["key"]
            where = f"{_ARCH[s['arch']]} | {op} | {_fmt_pair(f'{i}->{o}')} | {dest} | {approx}"
            for c in sp["changed"]:
                bad = _kind(c["old"]) != _kind(c["new"])
                row = (
                    f"| {where} | `{_val(c['input'])}` | `{_val(c['old'])}` | "
                    f"{'⚠️ ' if bad else ''}`{_val(c['new'])}` |"
                )
                (severe if bad else minor).append(row)
            nan = sp["nan_propagates"]
            if nan["base"] and not nan["head"]:
                nan_lost.append(where)
            elif nan["head"] and not nan["base"]:
                nan_gained.append(where)
    header = [
        "| arch | op | format | dest_acc | approx | input | old | new |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines = [
        "### Edge cases",
        "",
        "What the hardware returns for inputs the sweep does not feed: NaN, ±inf, ±0, subnormals, and "
        "the format's largest and smallest normal values, where the PR changed it. ⚠️ = the result "
        "changed kind (finite ↔ NaN/inf, sign, zero).",
        "",
    ]
    if severe:
        lines += header + severe[:40]
        if len(severe) > 40:
            lines.append(f"| … | {len(severe) - 40} more in the artifact | | | | | | |")
    elif not minor:
        lines.append(f"No special-input result changed across the {total} variant(s).")
    else:
        lines.append("No special-input result changed kind.")
    if minor:
        lines += [
            "",
            f"<details><summary>{len(minor)} special-input result(s) changed value only</summary>",
            "",
            *header,
            *minor[:60],
            "",
            "</details>",
        ]
    if nan_lost:
        lines += [
            "",
            "⚠️ NaN input no longer returns NaN: "
            + "; ".join(f"`{w}`" for w in nan_lost[:10]),
        ]
    if nan_gained:
        lines += [
            "",
            "NaN input now returns NaN (it did not before): "
            + "; ".join(f"`{w}`" for w in nan_gained[:10]),
        ]
    return lines + [""]


def cross_arch_section(summaries):
    """Whether Wormhole and Blackhole return the same bits on the head side."""
    if len(summaries) < 2:
        return []
    by_arch = {
        s["arch"]: {tuple(r["key"]): r.get("head_digest") for r in s["accuracy"]}
        for s in summaries
    }
    wh, bh = by_arch.get("wormhole", {}), by_arch.get("blackhole", {})
    common = [k for k in wh if k in bh and wh[k] and bh[k]]
    differ = [k for k in common if wh[k] != bh[k]]
    if not common:
        return []
    text = (
        f"Wormhole and Blackhole return bit-identical results on all {len(common)} swept variants."
        if not differ
        else f"Wormhole and Blackhole differ on {len(differ)} of {len(common)} swept variants: "
        + ", ".join(
            f"`{k[0]} {_fmt_pair(k[1] + '->' + k[2])} dest_acc={k[4]} approx={k[3]}`"
            for k in differ[:8]
        )
    )
    return ["### Wormhole vs Blackhole", "", text, ""]


def header(summaries):
    s = summaries[0]
    archs = ", ".join(f"{_ARCH[x['arch']]} ({x['host_board']})" for x in summaries)
    lines = [
        COMMENT_MARKER,
        "## LLK SFPU report",
        "",
        f"Tested `{s['head_sha'][:10]}` (PR head) against `{s['base_sha'][:10]}` (merge-base) on {archs}. "
        "Silicon, speed of light.",
    ]
    if s.get("head_moved_to"):
        lines.append(
            f"> [!NOTE]\n> The PR has moved to `{s['head_moved_to'][:10]}` since; comment `/llk-sfpu-test` to re-run."
        )
    age = s.get("merge_base_age_days")
    if age is not None and age > STALE_DAYS:
        lines.append(
            f"> [!WARNING]\n> The merge-base is {age} days old. Main may have changed these kernels since: "
            "rebase, or re-run with `/llk-sfpu-test --base main`."
        )
    ops = s["ops"]
    lines += [
        "",
        f"**Ops measured:** {', '.join(f'`{o}`' for o in ops['measured']) or 'none'} ({ops['why']}).",
    ]
    if ops.get("not_covered"):
        lines.append(
            f"**Changed but not measured:** {', '.join(f'`{o}`' for o in ops['not_covered'])}."
        )
    lines += ["", AI_SUMMARY_MARKER, ""]
    return lines


def notes_section(summaries):
    s = summaries[0]
    notes = []
    if s.get("not_applied"):
        notes.append(
            "Files this PR changes that the measurement does not use (host-side test code runs from main): "
            + ", ".join(f"`{p}`" for p in s["not_applied"][:10])
        )
    for n in s.get("notes", []):
        notes.append(n)
    if not notes:
        return []
    return ["### Notes for reviewers", "", *[f"- {n}" for n in notes], ""]


def footer(summaries):
    s = summaries[0]
    lines = [
        "<details><summary>Reproduce</summary>",
        "",
        "```",
        *s.get("commands", []),
        "```",
        "",
        f"Tool `{s['tool_sha'][:10]}`, mode `{s['mode']}`. Runs: {', '.join(x.get('run_url') or x['host'] for x in summaries)}.",
        "",
        "</details>",
    ]
    return lines


def render(summaries):
    summaries = sorted(summaries, key=lambda s: s["arch"] != "wormhole")
    parts = header(summaries)
    parts += perf_section(summaries)
    parts += accuracy_section(summaries)
    parts += edge_section(summaries)
    parts += cross_arch_section(summaries)
    parts += notes_section(summaries)
    parts += footer(summaries)
    return _fit("\n".join(parts) + "\n")


#: A GitHub comment holds 65,536 characters; leave room for the AI summary and footer.
MAX_CHARS = 60000


def _fit(text):
    """Keep the comment postable: fold <details> bodies first, then cut tables."""
    if len(text) <= MAX_CHARS:
        return text
    import re

    text = re.sub(
        r"(<details><summary>.*?</summary>)\n.*?\n</details>",
        r"\1\n\n(Too long for a comment: see the run's `llk-sfpu-report` artifact.)\n\n</details>",
        text,
        flags=re.S,
    )
    if len(text) <= MAX_CHARS:
        return text
    lines, out, size = text.split("\n"), [], 0
    for line in lines:
        if line.startswith("| ") and size + len(line) > MAX_CHARS - 500:
            continue
        out.append(line)
        size += len(line) + 1
    out.append(
        "\n_Some table rows were cut to fit a comment; the artifact has all of them._"
    )
    return "\n".join(out)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Render the LLK SFPU report comment.")
    ap.add_argument("summaries", nargs="+", help="summary.json per architecture")
    ap.add_argument("-o", "--out", default="-")
    args = ap.parse_args(argv)
    text = render([json.loads(Path(p).read_text()) for p in args.summaries])
    if args.out == "-":
        print(text, end="")
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
