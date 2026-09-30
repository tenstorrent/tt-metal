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
    lines = lines[:-2]
    moved, rest = [], []
    for key, row, cells in _merge_approx(rows, _perf_cells):
        line = _row_prefix(key) + "| " + " | ".join(cells) + " |"
        (moved if _perf_moved(row) else rest).append((_perf_regressed(row), line))
    if moved:
        # Regressions first: they are what a reviewer is looking for.
        lines += header + [l for _, l in sorted(moved, key=lambda m: not m[0])]
    elif rows:
        lines.append(f"No variant moved beyond the thresholds ({len(rest)} measured).")
    else:
        lines.append("No op was measured.")
    if moved and rest:
        lines += [""]
    if rest:
        lines += [
            f"<details><summary>{len(rest)} variant(s) within the thresholds</summary>",
            "",
            *header,
            *[l for _, l in rest],
            "",
            "</details>",
        ]
    lines += ["", _loadmacro_note(summaries), ""]
    return lines


def _perf_regressed(row):
    return any(row.get(rt, {}).get("regression") for rt in ("MATH_ISOLATE", "L1_TO_L1"))


def _perf_moved(row):
    return any(
        row.get(rt, {}).get("regression") or row.get(rt, {}).get("improvement")
        for rt in ("MATH_ISOLATE", "L1_TO_L1")
    )


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


def _max_grew(old, new):
    """A higher max error that means something: any growth of a small max, or more
    than 1% of a large one (a max of hundreds of millions of steps moving by 50 is the
    same broken range, not a new regression)."""
    return new > old and (old < 16 or new > old * 1.01)


def _acc_reasons(rec):
    """Which parts of an accuracy row regressed: ``max``, ``lanes``, ``nonfinite``.

    The report puts the ⚠️ on exactly these cells, so a flagged row says why.
    """
    b, h = rec["base"], rec["head"]
    if b.get("metric") == "exact":
        return {"wrong"} if _exact_regressed(rec) else set()
    reasons = set()
    if _max_grew(b["max"], h["max"]):
        reasons.add("max")
    if (rec["worse"] - rec["better"]) / max(h["lanes"], 1) >= NOTABLE_LANE_SHARE:
        reasons.add("lanes")
    if h["nonfinite"] > b["nonfinite"]:
        reasons.add("nonfinite")
    return reasons


def _acc_regressed(rec):
    """Worse overall: a real rise in the max, clearly more lanes worse than better, or
    new non-finite results."""
    return bool(_acc_reasons(rec))


#: Changed lanes below this share of the measured lanes, with the same max error, are
#: noise to a reviewer: the row folds away.
NOTABLE_LANE_SHARE = 0.001


def _acc_notable(rec):
    b, h = rec["base"], rec["head"]
    if rec.get("bit_identical"):
        return False
    if _acc_regressed(rec):
        return True
    if b.get("metric") == "exact":
        return h["wrong"] != b["wrong"]
    moved = (rec["worse"] + rec["better"]) / max(h["lanes"], 1)
    return (
        h["max"] != b["max"]
        or h["nonfinite"] != b["nonfinite"]
        or moved >= NOTABLE_LANE_SHARE
    )


def _acc_cells(rec):
    b, h = rec["base"], rec["head"]
    why = _acc_reasons(rec)
    warn = lambda part: "⚠️ " if part in why else ""  # noqa: E731
    return (
        _num(b["max"]),
        f"{warn('max')}{_bold_if_better(h['max'], b['max'])}",
        f"{_num(b['mean'], 3)} → {_num(h['mean'], 3)}",
        f"{b['le1']:.2%} → {h['le1']:.2%}",
        f"{warn('lanes')}{_num(rec['worse'])} / {_num(rec['better'])}",
        f"{b['nonfinite']} → {warn('nonfinite')}{h['nonfinite']}",
    )


def _row_prefix(key):
    arch, op, i, o, approx, dest = key
    return f"| {_ARCH[arch]} | {op} | {_fmt_pair(f'{i}->{o}')} | {dest} | {approx} "


def _exact_regressed(rec):
    b, h = rec["base"], rec["head"]
    return h["wrong"] > b["wrong"]


def _exact_cells(rec):
    b, h = rec["base"], rec["head"]
    flag = "⚠️ " if _exact_regressed(rec) else ""
    return (
        _num(h["lanes"]),
        _num(b["wrong"]),
        f"{flag}{_bold_if_better(h['wrong'], b['wrong'])}",
        f"{_num(rec['worse'])} / {_num(rec['better'])}",
    )


def _table(rows, header, cells):
    return header + [
        _row_prefix(k) + "| " + " | ".join(c) + " |"
        for k, _, c in _merge_approx(rows, cells)
    ]


def accuracy_section(summaries):
    ulp_head = [
        "| arch | op | format | dest_acc | approx | max old | max new | mean old → new | ≤1 ULP old → new | worse / better | non-finite old → new |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    exact_head = [
        "| arch | op | format | dest_acc | approx | lanes | wrong old | wrong new | newly wrong / newly right |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    groups = {
        ("ulp", False): [],
        ("ulp", True): [],
        ("exact", False): [],
        ("exact", True): [],
    }
    binary = False
    for s in summaries:
        for rec in s["accuracy"]:
            if "base" not in rec:
                continue
            binary |= bool(rec.get("binary"))
            key = (s["arch"], *rec["key"])
            metric = rec["base"].get("metric", "ulp")
            groups[(metric, bool(rec.get("bit_identical")))].append((key, rec))
    lines = [
        "### Accuracy",
        "",
        "ULP steps against the correctly rounded reference (the golden of the functional tests). "
        "Unary ops: every finite input of the format (fp32: every 65,536th value). Binary ops: "
        "random operand pairs from the op's functional-test domain, the same draw on both sides. "
        "Lanes a step count cannot describe (NaN, subnormal input, padding) are left out here and "
        "covered by the edge cases below. `worse / better` counts lanes whose error changed. "
        "⚠️ marks the cell that got worse: `max new` for a real rise in the max, "
        "`worse / better` for at least 0.1% of the lanes net worse, `non-finite` for new "
        "non-finite results.",
        "",
    ]
    shown_ulp = [r for r in groups[("ulp", False)] if _acc_notable(r[1])]
    shown_exact = [r for r in groups[("exact", False)] if _acc_notable(r[1])]
    folded_ulp = groups[("ulp", True)] + [
        r for r in groups[("ulp", False)] if not _acc_notable(r[1])
    ]
    folded_exact = groups[("exact", True)] + [
        r for r in groups[("exact", False)] if not _acc_notable(r[1])
    ]
    regressed_first = lambda rows: sorted(
        rows, key=lambda r: not _acc_regressed(r[1])
    )  # noqa: E731
    if shown_ulp:
        lines += _table(regressed_first(shown_ulp), ulp_head, _acc_cells)
    if shown_exact:
        lines += [
            "",
            "Comparisons and integer ops have no ULP: a lane is right or wrong.",
            "",
            *_table(regressed_first(shown_exact), exact_head, _exact_cells),
        ]
    if not shown_ulp and not shown_exact:
        lines.append(
            "No accuracy change worth a row: max error and lane counts are unchanged."
        )
    folded = len(folded_ulp) + len(folded_exact)
    if folded:
        lines += [
            "",
            f"<details><summary>{folded} variant(s) unchanged, or changed in a handful of lanes only</summary>",
            "",
        ]
        if folded_ulp:
            lines += _table(folded_ulp, ulp_head, _acc_cells)
        if folded_exact:
            lines += ["", *_table(folded_exact, exact_head, _exact_cells)]
        lines += ["", "</details>"]
    return lines + [""]


def _val(x):
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join(_val(v) for v in x) + ")"
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


def findings(summaries):
    """Every ⚠️ of the report, one line each: the top of the comment, and ``--check``.

    Returns dicts with ``kind`` (perf / accuracy / edge), ``arch``, ``op``, ``fmt`` (the
    input format, for a narrowed re-run) and ``text``.
    """
    grouped = {}

    def add(kind, arch, op, fmt, where, what, approx):
        # ``where`` starts with the format pair; rows that differ only in it (or in
        # approx) and read the same are one finding.
        pair, _, rest = where.partition(" ")
        key = (kind, arch, op, rest, what)
        entry = grouped.setdefault(key, {"approx": set(), "pairs": [], "fmts": []})
        entry["approx"].add(approx)
        if pair not in entry["pairs"]:
            entry["pairs"].append(pair)
        if fmt not in entry["fmts"]:
            entry["fmts"].append(fmt)

    for s in summaries:
        for family, by_schedule in s["perf"].items():
            default = {
                (r["op"], r["formats"], r["dest_acc"], r["approx"]): r
                for r in by_schedule.get("loadmacro", [])
            }
            for schedule, rows in by_schedule.items():
                for row in rows:
                    if not _perf_regressed(row):
                        continue
                    twin = default.get(
                        (row["op"], row["formats"], row["dest_acc"], row["approx"])
                    )
                    if (
                        schedule != "loadmacro"
                        and twin is not None
                        and (
                            _perf_regressed(twin)
                            or twin.get("text_head") == row.get("text_head")
                        )
                    ):
                        continue  # the same code, or already listed for the default build
                    m, l1 = row.get("MATH_ISOLATE", {}), row.get("L1_TO_L1", {})
                    label, cell = ("math", m) if m.get("regression") else ("L1→L1", l1)
                    sched = "" if schedule == "loadmacro" else ", SFPLOADMACRO disabled"
                    where = (
                        f"{_fmt_pair(row['formats'])} dest_acc={row['dest_acc']}{sched}"
                    )
                    what = (
                        f"{label} {_num(cell['base'])} → {_num(cell['head'])} cycles/tile "
                        f"({_speedup(cell['base'], cell['head']).strip('*')})"
                    )
                    add(
                        "perf",
                        s["arch"],
                        row["op"],
                        row["formats"].split("->")[0],
                        where,
                        what,
                        row["approx"],
                    )
        for rec in s["accuracy"]:
            op, i, o, approx, dest = rec["key"]
            where = f"{_fmt_pair(f'{i}->{o}')} dest_acc={dest}"
            if "base" in rec and _acc_regressed(rec):
                b, h = rec["base"], rec["head"]
                if b.get("metric") == "exact":
                    what = f"wrong lanes {_num(b['wrong'])} → {_num(h['wrong'])}"
                else:
                    what = f"max ULP {_num(b['max'])} → {_num(h['max'])}"
                    if rec["worse"] > rec["better"]:
                        what += f", {_num(rec['worse'])} lanes worse / {_num(rec['better'])} better"
                    if h["nonfinite"] > b["nonfinite"]:
                        what += f", non-finite {b['nonfinite']} → {h['nonfinite']}"
                add("accuracy", s["arch"], op, i, where, what, approx)
            sp = rec.get("specials")
            if sp:
                severe = [
                    c for c in sp["changed"] if _kind(c["old"]) != _kind(c["new"])
                ]
                if severe:
                    c = severe[0]
                    what = f"`f{_val(c['input']) if isinstance(c['input'], (list, tuple)) else '(' + _val(c['input']) + ')'}`: `{_val(c['old'])}` → `{_val(c['new'])}`"
                    if len(severe) > 1:
                        what += f" (+{len(severe) - 1} more special input(s))"
                    add("edge", s["arch"], op, i, where, what, approx)
                if sp["nan_propagates"]["base"] and not sp["nan_propagates"]["head"]:
                    add(
                        "edge",
                        s["arch"],
                        op,
                        i,
                        where,
                        "a NaN input no longer returns NaN",
                        approx,
                    )
    out = []
    order = {"perf": 0, "accuracy": 1, "edge": 2}
    for (kind, arch, op, rest, what), e in sorted(
        grouped.items(), key=lambda kv: (order[kv[0][0]], kv[0][1:])
    ):
        approx = "/".join(sorted(e["approx"], reverse=True))
        out.append(
            {
                "kind": kind,
                "arch": arch,
                "op": op,
                "fmt": ",".join(e["fmts"]),
                "text": f"{kind}: {_ARCH[arch]} `{op}` {'/'.join(e['pairs'])} {rest} approx={approx}: {what}",
            }
        )
    return out


def glance_section(summaries):
    found = findings(summaries)
    if not found:
        return [
            "**No regressions.** Nothing is slower beyond the thresholds, and no accuracy or "
            "edge-case result got worse. Details below.",
            "",
        ]
    return [
        f"**⚠️ {len(found)} regression(s)**",
        "",
        *[f"- {f['text']}" for f in found[:20]],
        "",
    ]


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
    lines += [""]
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
    head, base, tool = s["head_sha"], s["base_sha"], s["tool_sha"]
    mode = " --mode rebase" if s["mode"] == "rebase" else ""
    ops = ",".join(s["ops"]["measured"])
    pr = s.get("pr_number")
    fetch = f"+refs/pull/{pr}/head:refs/remotes/origin/pr/{pr}" if pr else head
    lines = [
        "<details><summary>Reproduce</summary>",
        "",
        "On a machine with the card, in a tt-metal checkout, in the tt-llk test venv "
        "(`tt_metal/tt-llk/tests/setup_external_testing_env.sh`). The tool runs from the "
        "revision below; the PR only supplies device code. `--check` exits 1 when the result "
        "has a ⚠️, so the same command is the test that fails today and passes once fixed.",
        "",
        "```bash",
        f"git fetch origin {tool} {fetch} {base}",
        f"git checkout --detach {tool}",
        "cd tt_metal/tt-llk",
    ]
    for x in summaries:
        cli = f"python3 sfpu_report/cli.py --arch {x['arch']} --head {head} --base {base}{mode} run"
        lines += [
            "",
            f"# the whole report, {_ARCH[x['arch']]}",
            f"{cli} --ops {ops or '<op>'} --check",
        ]
        per_op = {}
        for f in findings([x]):
            fmts = per_op.setdefault(f["op"], [])
            fmts += [m for m in f["fmt"].split(",") if m not in fmts]
        if per_op:
            lines.append(f"# only what regressed, {_ARCH[x['arch']]}")
            lines += [
                f"{cli} --ops {op} --formats {','.join(fmts)} --check"
                for op, fmts in list(per_op.items())[:10]
            ]
    lines += [
        "```",
        "",
        f"Tool `{tool[:10]}`, mode `{s['mode']}`. Runs: "
        + ", ".join(x.get("run_url") or x["host"] for x in summaries)
        + ".",
        "",
        "</details>",
    ]
    return lines


def render(summaries):
    summaries = sorted(summaries, key=lambda s: s["arch"] != "wormhole")
    parts = header(summaries)
    parts += glance_section(summaries)
    parts += [AI_SUMMARY_MARKER, ""]
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
