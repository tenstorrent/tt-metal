# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Render one PR comment from the per-architecture ``summary.json`` files.

The layout follows how SFPU bounty PRs already report results (old/new cycles,
speedup, old/new max ULP per arch), so a reviewer reads it the same way. Every
number comes from the summaries.
"""

import json
import math
import re
from pathlib import Path

from overlay import HARNESS_PYTHON
from perf import DEVICE_COMPUTED_OPERAND, ROWS_PER_TILE

COMMENT_MARKER = "<!-- llk-sfpu-report -->"
STALE_DAYS = 14

_ARCH = {"wormhole": "Wormhole", "blackhole": "Blackhole"}
_FMT = {"Float16_b": "bf16", "Float16": "fp16", "Float32": "fp32", "Bfp8_b": "bfp8"}


def code(text):
    """``text`` as inline code. File names come from the PR, and git allows any of them:
    a backtick or a newline would end the code span and put the rest of the name into a
    bot comment as Markdown. Characters outside the plain set become ``?``."""
    return "`" + re.sub(r"[^A-Za-z0-9_./+-]", "?", str(text)) + "`"


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
        for by_schedule in s["perf"].values():
            for row in by_schedule.get("loadmacro", []):
                i, _, o = row["formats"].partition("->")
                rows.append(((s["arch"], row["op"], i, o, row["approx"], row["dest_acc"]), row))
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
    for op in sorted({row["op"].split(" (", 1)[0] for _, row in rows} & DEVICE_COMPUTED_OPERAND.keys()):
        operand, work = DEVICE_COMPUTED_OPERAND[op]
        lines += [
            "",
            f"`{op}`: the test feeds {operand} as an input. ttnn computes it on the device before "
            f"this kernel runs ({work}), so the old cycles leave that work out, and a kernel that "
            "does it itself measures slower here than it runs in ttnn. Compare such a change in ttnn.",
        ]
    lines += ["", _loadmacro_note(summaries), ""]
    return lines


def _perf_regressed(row):
    """Slower beyond the thresholds, or ran on the base side and not on the head."""
    return row.get("missing") == "head" or any(row.get(rt, {}).get("regression") for rt in ("MATH_ISOLATE", "L1_TO_L1"))


def _perf_moved(row):
    return bool(row.get("missing")) or any(
        row.get(rt, {}).get("regression") or row.get(rt, {}).get("improvement") for rt in ("MATH_ISOLATE", "L1_TO_L1")
    )


#: A perf row's cells for a variant that ran on one side only.
_ONE_SIDED = {
    "head": ("–", "⚠️ no result", "–", "–", "ran on base only", "–"),
    "base": ("–", "new", "–", "–", "ran on head only", "–"),
}


def _perf_cells(row):
    if row.get("missing"):
        return _ONE_SIDED[row["missing"]]
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
        for by_schedule in s["perf"].values():
            on = {(r["op"], r["formats"], r["dest_acc"], r["approx"]): r for r in by_schedule.get("loadmacro", [])}
            for r in by_schedule.get("no-loadmacro", []):
                k = (r["op"], r["formats"], r["dest_acc"], r["approx"])
                if r.get("missing"):
                    continue  # its loadmacro twin says so
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
        grouped.setdefault((arch, op, i, o, dest, cells(payload)), []).append((approx, payload))
    out = []
    for (arch, op, i, o, dest, shown), items in grouped.items():
        approx = "/".join(sorted({a for a, _ in items}, reverse=True))
        out.append(((arch, op, i, o, approx, dest), items[0][1], shown))
    return sorted(
        out,
        key=lambda r: (r[0][0], r[0][1], _FMT_ORDER.get(r[0][2], 9), r[0][5], r[0][4]),
    )


#: A max error below this many ULP is small: any rise of it is a regression.
SMALL_MAX_ULP = 16

#: A larger max is a regression when it grows by more than this factor: a max of hundreds
#: of millions of steps moving by 50 is the same broken range, not a new regression.
MAX_GROWTH = 1.01


def _max_grew(old, new):
    """A higher max error that means something."""
    return new > old and (old < SMALL_MAX_ULP or new > old * MAX_GROWTH)


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
    return h["max"] != b["max"] or h["nonfinite"] != b["nonfinite"] or moved >= NOTABLE_LANE_SHARE


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
    return header + [_row_prefix(k) + "| " + " | ".join(c) + " |" for k, _, c in _merge_approx(rows, cells)]


def _no_accuracy(summaries):
    """The measured ops an arch has no accuracy result for, on either side: `op`, or
    `op` (Arch) when only some archs lack it. The notes say why (no driver covers it,
    its run failed, Typecast). A variant with a result on one side only is not in it:
    the findings and the notes report those."""
    lacking = {}
    for s in summaries:
        have = {rec["key"][0] for rec in s["accuracy"]}
        for op in s["ops"]["measured"]:
            if op not in have:
                lacking.setdefault(op, []).append(_ARCH[s["arch"]])
    return [
        f"`{op}`" if len(archs) == len(summaries) else f"`{op}` ({', '.join(archs)})" for op, archs in lacking.items()
    ]


def _no_accuracy_line(summaries):
    lacking = _no_accuracy(summaries)
    return f"Not measured for {', '.join(lacking)}: see the notes." if lacking else None


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
    not_measured = _no_accuracy_line(summaries)
    perf_only = _perf_only_variants(summaries)
    if perf_only:
        perf_only = [
            "",
            "Measured for perf, not for accuracy: " + ", ".join(perf_only) + ". Accuracy runs unary "
            "ops with fast mode off, and binary ops with approx off.",
        ]
    if not any(groups.values()):
        # Nothing to compare: "unchanged" would read as measured.
        what = not_measured or "No variant has accuracy results on both sides; see run.log."
        return ["### Accuracy", "", what, *perf_only, ""]
    lines = [
        "### Accuracy",
        "",
        "ULP steps against the correctly rounded reference (the golden of the functional tests). "
        "Unary ops: every normal input of the format, and zero (fp32: every 65,536th value). Binary ops: "
        "the operand pairs of the op's functional test (random over its domain; crafted pairs, "
        "ties included, for comparisons), the same on both sides. "
        "Lanes a step count cannot describe (NaN, subnormal input, padding) are left out here and "
        "covered by the edge cases below. `worse / better` counts lanes whose error changed. "
        "⚠️ marks the cell that got worse: `max new` for a real rise in the max, "
        "`worse / better` for at least 0.1% of the lanes net worse, `non-finite` for new "
        "non-finite results.",
        "",
    ]
    shown_ulp = [r for r in groups[("ulp", False)] if _acc_notable(r[1])]
    shown_exact = [r for r in groups[("exact", False)] if _acc_notable(r[1])]
    folded_ulp = groups[("ulp", True)] + [r for r in groups[("ulp", False)] if not _acc_notable(r[1])]
    folded_exact = groups[("exact", True)] + [r for r in groups[("exact", False)] if not _acc_notable(r[1])]
    regressed_first = lambda rows: sorted(rows, key=lambda r: not _acc_regressed(r[1]))  # noqa: E731
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
        lines.append("No accuracy change worth a row: max error and lane counts are unchanged.")
    if not_measured:
        lines += ["", not_measured]
    lines += perf_only
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


def _perf_only(family, row):
    """Whether accuracy does not run this perf variant: fast mode on, or a binary op's approx Yes."""
    return row.get("fast_mode") == "Yes" or (family == "binary" and row["approx"] != "No")


def _perf_only_variants(summaries):
    """The perf variants accuracy does not run."""
    out = []
    for s in summaries:
        for family, by_schedule in s["perf"].items():
            for row in by_schedule.get("loadmacro", []):
                if _perf_only(family, row):
                    op = row["op"].split(" (", 1)[0]
                    mode = "fast mode" if row.get("fast_mode") == "Yes" else f"approx={row['approx']}"
                    out.append(f"`{op}` {mode}")
    return list(dict.fromkeys(out))


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


def _edge_change(c):
    """How a special-input result changed: None (its value only), "fixed" (its kind, to
    the reference's) or "worse" (its kind otherwise, or with no reference to go by)."""
    new = _kind(c["new"])
    if _kind(c["old"]) == new:
        return None
    ref = c.get("ref")
    return "fixed" if ref is not None and _kind(ref) == new else "worse"


def edge_section(summaries):
    """What the hardware returns for the special inputs, where the PR changed it."""
    severe, fixed, minor, nan_lost, nan_gained, total = [], [], [], [], [], 0
    for s in summaries:
        for rec in s["accuracy"]:
            sp = rec.get("specials")
            if not sp:
                continue
            total += 1
            op, i, o, approx, dest = rec["key"]
            where = f"{_ARCH[s['arch']]} | {op} | {_fmt_pair(f'{i}->{o}')} | {dest} | {approx}"
            for c in sp["changed"]:
                change = _edge_change(c)
                new = f"`{_val(c['new'])}`"
                new = {"worse": f"⚠️ {new}", "fixed": f"**{new}**"}.get(change, new)
                row = f"| {where} | `{_val(c['input'])}` | `{_val(c['old'])}` | {new} |"
                {"worse": severe, "fixed": fixed}.get(change, minor).append(row)
            nan = sp["nan_propagates"]
            if nan["base"] and not nan["head"]:
                nan_lost.append(where)
            elif nan["head"] and not nan["base"]:
                nan_gained.append(where)
    not_measured = _no_accuracy_line(summaries)
    if not total:
        what = not_measured or "No variant has special-input results on both sides; see run.log."
        return ["### Edge cases", "", what, ""]
    header = [
        "| arch | op | format | dest_acc | approx | input | old | new |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines = [
        "### Edge cases",
        "",
        "What the hardware returns for inputs the sweep does not feed: NaN, ±inf, ±0, subnormals, and "
        "the format's largest and smallest normal values, where the PR changed it. ⚠️ = the result "
        "changed kind (finite ↔ NaN/inf, sign, zero) away from the reference; **bold** = it changed "
        "kind to the reference's. For NaN, -0 and subnormal inputs the reference models the unpacker, "
        "not IEEE, so any change of kind there is ⚠️.",
        "",
    ]
    shown = severe + fixed
    if shown:
        lines += header + shown[:40]
        if len(shown) > 40:
            lines.append(f"| … | {len(shown) - 40} more in the artifact | | | | | | |")
    elif not minor:
        lines.append(f"No special-input result changed across the {total} variant(s).")
    else:
        lines.append("No special-input result changed kind.")
    if not_measured:
        lines += ["", not_measured]
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
            "⚠️ NaN input no longer returns NaN: " + "; ".join(f"`{w}`" for w in nan_lost[:10]),
        ]
    if nan_gained:
        lines += [
            "",
            "NaN input now returns NaN (it did not before): " + "; ".join(f"`{w}`" for w in nan_gained[:10]),
        ]
    return lines + [""]


def cross_arch_section(summaries):
    """Whether Wormhole and Blackhole return the same bits on the head side."""
    if len(summaries) < 2:
        return []
    by_arch = {s["arch"]: {tuple(r["key"]): r.get("head_digest") for r in s["accuracy"]} for s in summaries}
    wh, bh = by_arch.get("wormhole", {}), by_arch.get("blackhole", {})
    common = [k for k in wh if k in bh and wh[k] and bh[k]]
    differ = [k for k in common if wh[k] != bh[k]]
    if not common:
        return []
    text = (
        f"Wormhole and Blackhole return bit-identical results on all {len(common)} swept variants."
        if not differ
        else f"Wormhole and Blackhole differ on {len(differ)} of {len(common)} swept variants: "
        + ", ".join(f"`{k[0]} {_fmt_pair(k[1] + '->' + k[2])} dest_acc={k[4]} approx={k[3]}`" for k in differ[:8])
    )
    return ["### Wormhole vs Blackhole", "", text, ""]


#: A kernel that runs over the tile takes at least a cycle per row. An op under that,
#: and this many times faster than before, did not run. (A kernel that touches a few
#: rows only, like SfpuAddTopRow's, takes ~15 cycles on both sides.)
NOT_RUN_SPEEDUP = 10


def _not_run(s):
    """{op: evidence} for the ops that did not run on ``s``'s PR side, so its numbers
    there are not the PR's kernel: main's harness asked for something the PR's device
    code no longer runs (#54080), or the PR's kernel stopped computing.

    Accuracy decides: every variant hands back an input unchanged on the PR side, and
    none did before. The perf variants accuracy does not run, and an op without
    accuracy results, go by their cycles: under a cycle per row, and NOT_RUN_SPEEDUP
    times faster than before.
    """
    returns, cycles, perf_only = {}, {}, {}
    for rec in s["accuracy"]:
        if "base" in rec and "returns_input" in rec:
            returns.setdefault(rec["key"][0], []).append(rec["returns_input"])
    for family, by_schedule in s["perf"].items():
        for row in by_schedule.get("loadmacro", []):
            m = row.get("MATH_ISOLATE", {})
            if m.get("base") is not None and m.get("head") is not None:
                op = row["op"].split(" (", 1)[0]
                cycles.setdefault(op, []).append((m["base"], m["head"]))
                if _perf_only(family, row):
                    perf_only.setdefault(op, []).append((m["base"], m["head"]))

    def collapsed(timed):
        return all(head < ROWS_PER_TILE and base >= NOT_RUN_SPEEDUP * head for base, head in timed)

    out = {}
    for op in s["ops"]["measured"]:
        sides, timed = returns.get(op), cycles.get(op, [])
        if sides:
            if all(r["head"] and not r["base"] for r in sides) and collapsed(perf_only.get(op, [])):
                out[op] = {"returns": sides[0]["head"], "cycles": timed}
        elif timed and collapsed(timed):
            out[op] = {"returns": None, "cycles": timed}
    return out


def _without_not_run(s):
    """``s`` without the results of the ops that did not run on its PR side."""
    gone = _not_run(s)
    if not gone:
        return s
    keep = lambda label: label.split(" (", 1)[0] not in gone  # noqa: E731
    return dict(
        s,
        ops=dict(s["ops"], measured=[o for o in s["ops"]["measured"] if keep(o)]),
        perf={
            family: {schedule: [r for r in rows if keep(r["op"])] for schedule, rows in by_schedule.items()}
            for family, by_schedule in s["perf"].items()
        },
        accuracy=[r for r in s["accuracy"] if keep(r["key"][0])],
    )


def _span(values):
    lo, hi = min(values), max(values)
    return _num(lo) if round(lo) == round(hi) else f"{_num(lo)}–{_num(hi)}"


def _evidence(ev, short=False):
    """What shows that an op did not run: clauses whose subject is the op."""
    out = []
    if ev["returns"]:
        out.append(f"returns its {ev['returns']} input unchanged" + ("" if short else " on every lane"))
    if ev["cycles"]:
        new, old = _span([h for _, h in ev["cycles"]]), _span([b for b, _ in ev["cycles"]])
        out.append(f"takes {new} math cycles" + ("/tile" if short else " per tile") + f" ({old} before)")
    return out


def _harness_changed(summaries):
    """{op: [files]}: the harness Python the PR changes for each measured op."""
    out = {}
    for s in summaries:
        for op, files in s.get("harness_changed", {}).items():
            out.setdefault(op, list(files))
    return out


def _harness_files(files):
    """The op's registration first, then the tests, then what they import."""
    rank = lambda p: (p != "helpers/llk_params.py", p.count("/"), p)  # noqa: E731
    names = sorted((p.removeprefix(HARNESS_PYTHON) for p in files), key=rank)
    return ", ".join(code(p) for p in names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")


def not_run_section(summaries):
    """The ops that did not run on the PR side: what shows it, the likely cause, and
    that their numbers are left out of the report."""
    by_op = {}
    for s in summaries:
        for op, ev in _not_run(s).items():
            entry = by_op.setdefault(op, {"archs": [], "returns": None, "cycles": []})
            entry["archs"].append(_ARCH[s["arch"]])
            entry["returns"] = entry["returns"] or ev["returns"]
            entry["cycles"] += ev["cycles"]
    harness = _harness_changed(summaries)
    python = any(p.startswith(HARNESS_PYTHON) for s in summaries for p in s.get("not_applied", []))
    kernel = (
        "the PR's kernel, or its dispatch in the device test code (`sfpu_operations.h`), no longer computes "
        "it: a regression, unless the PR meant that."
    )
    lines = []
    for op, ev in by_op.items():
        if op in harness:
            cause = (
                f"This PR also changes `{op}` in the test harness's Python ({_harness_files(harness[op])}), "
                "which the report runs from main: main's harness most likely asks for the op in a way "
                "the PR's device code no longer runs. Measure it with the PR's own tests."
            )
        elif python:
            cause = (
                "This PR also changes the test harness's Python (see the notes), which the report runs "
                f"from main. If that is not why, {kernel}"
            )
        else:
            cause = kernel[0].upper() + kernel[1:]
        lines += [
            f"**⚠️ `{op}` did not run on the PR side** ({', '.join(ev['archs'])}). There it "
            f"{' and '.join(_evidence(ev))}. {cause} Its numbers are left out of this report.",
            "",
        ]
    return lines


def findings(summaries):
    """Every ⚠️ of the report, one line each: the top of the comment, and ``--check``.

    Returns dicts with ``kind`` (not run / perf / accuracy / edge), ``arch``, ``op``,
    ``fmt`` (the input format, for a narrowed re-run) and ``text``. An op that did not
    run on the PR side is one ``not run`` finding, and nothing else: its numbers are not
    the PR's.
    """
    not_run = []
    for s in summaries:
        for op, ev in _not_run(s).items():
            fmts = [r["key"][1] for r in s["accuracy"] if r["key"][0] == op] + [
                row["formats"].split("->")[0]
                for by_schedule in s["perf"].values()
                for row in by_schedule.get("loadmacro", [])
                if row["op"].split(" (", 1)[0] == op
            ]
            not_run.append(
                {
                    "kind": "not run",
                    "arch": s["arch"],
                    "op": op,
                    "fmt": ",".join(dict.fromkeys(fmts)),
                    "text": f"{_ARCH[s['arch']]} `{op}` on the PR side: {'; '.join(_evidence(ev, short=True))}",
                }
            )
    summaries = [_without_not_run(s) for s in summaries]
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
        for by_schedule in s["perf"].values():
            default = {(r["op"], r["formats"], r["dest_acc"], r["approx"]): r for r in by_schedule.get("loadmacro", [])}
            for schedule, rows in by_schedule.items():
                for row in rows:
                    if not _perf_regressed(row):
                        continue
                    twin = default.get((row["op"], row["formats"], row["dest_acc"], row["approx"]))
                    if (
                        schedule != "loadmacro"
                        and twin is not None
                        and (_perf_regressed(twin) or twin.get("text_head") == row.get("text_head"))
                    ):
                        continue  # the same code, or already listed for the default build
                    sched = "" if schedule == "loadmacro" else ", SFPLOADMACRO disabled"
                    where = f"{_fmt_pair(row['formats'])} dest_acc={row['dest_acc']}{sched}"
                    m, l1 = row.get("MATH_ISOLATE", {}), row.get("L1_TO_L1", {})
                    label, cell = ("math", m) if m.get("regression") else ("L1→L1", l1)
                    if row.get("missing"):
                        what = "ran on base, no result on head (did not build or run; see run.log)"
                    else:
                        what = (
                            f"{label} {_num(cell['base'])} → {_num(cell['head'])} cycles/tile "
                            f"({_speedup(cell['base'], cell['head']).strip('*')})"
                        )
                        computed = DEVICE_COMPUTED_OPERAND.get(row["op"].split(" (", 1)[0])
                        if computed:
                            what += f"; the old cycles leave out {computed[0]}, see Performance"
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
            for missing in rec.get("missing", []):
                side, _, kind = missing.partition(" ")
                if side == "head":
                    what = f"{kind}: ran on base, no result on head (did not build or run; see run.log)"
                    add("accuracy", s["arch"], op, i, where, what, approx)
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
                severe = [c for c in sp["changed"] if _edge_change(c) == "worse"]
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
    out = not_run
    order = {"perf": 0, "accuracy": 1, "edge": 2}
    for (kind, arch, op, rest, what), e in sorted(grouped.items(), key=lambda kv: (order[kv[0][0]], kv[0][1:])):
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


def glance_section(summaries, not_measured=(), other_ops=False):
    """The verdict. ``other_ops``: some ops did not run on the PR side and are named
    above, so a clean verdict is about the other ops."""
    found = findings(summaries)
    if not found:
        clean = "No regressions in the other ops" if other_ops else "No regressions"
        if not_measured:
            missing = ", ".join(_ARCH[x["arch"]] for x in not_measured)
            return [
                f"**{clean} on {', '.join(_ARCH[x['arch']] for x in summaries)}.** "
                f"{missing} was not measured: see the notes.",
                "",
            ]
        lacking = _no_accuracy(summaries)
        if lacking:
            compared = any("base" in r or "specials" in r for s in summaries for r in s["accuracy"])
            return [
                f"**{clean}.** Nothing is slower beyond the thresholds"
                + (", and no accuracy or edge-case result got worse" if compared else "")
                + f". Accuracy was not measured for {', '.join(lacking)}: see the notes.",
                "",
            ]
        return [
            f"**{clean}.** Nothing is slower beyond the thresholds, and no accuracy or "
            "edge-case result got worse. Details below.",
            "",
        ]
    return [
        f"**⚠️ {len(found)} regression(s)**",
        "",
        *[f"- {f['text']}" for f in found[:20]],
        "",
    ]


def _compared(x):
    """What one arch's sides were, by its mode."""
    if x["mode"] == "rebase":
        return f"this PR's kernel changes on top of main (`{x['tool_sha'][:10]}`) against main itself"
    return f"`{x['head_sha'][:10]}` (PR head) against `{x['base_sha'][:10]}` (merge-base)"


def header(summaries):
    s = summaries[0]
    on = lambda xs: ", ".join(f"{_ARCH[x['arch']]} ({x['host_board']})" for x in xs)  # noqa: E731
    if len({x["mode"] for x in summaries}) == 1:
        tested = f"Tested {_compared(s)} on {on(summaries)}."
    else:  # one arch fell back to rebase mode
        tested = "Tested " + ", and ".join(f"{_compared(x)} on {on([x])}" for x in summaries) + "."
    lines = [
        COMMENT_MARKER,
        "## LLK SFPU report",
        "",
        f"{tested} Silicon, speed of light.",
    ]
    if s.get("head_moved_to"):
        lines.append(
            f"> [!NOTE]\n> The PR has moved to `{s['head_moved_to'][:10]}` since; comment `/llk-sfpu-test` to re-run."
        )
    age = s.get("merge_base_age_days")
    if any(x["mode"] != "rebase" for x in summaries) and age is not None and age > STALE_DAYS:
        lines.append(
            f"> [!WARNING]\n> The merge-base is {age} days old. Main may have changed these kernels since: "
            "rebase, or re-run with `/llk-sfpu-test --base main`."
        )
    # Each arch measures what changed on it: a Blackhole-only PR measures nothing on Wormhole.
    measured = list(dict.fromkeys(o for x in summaries for o in x["ops"]["measured"]))
    not_covered = list(dict.fromkeys(o for x in summaries for o in x["ops"].get("not_covered", [])))
    if measured:
        lines += ["", f"**Ops measured:** {', '.join(f'`{o}`' for o in measured)} ({s['ops']['why']})."]
    elif any(x.get("detection") is not None for x in summaries):
        lines += ["", "**Ops measured:** none. No op this report covers compiles to different code on the two sides."]
    else:
        lines += ["", "**Ops measured:** none."]
    if not_covered:
        lines.append(f"**Changed but not measured:** {', '.join(f'`{o}`' for o in not_covered)}.")
    # The ops that did not run name their harness changes where they are reported.
    gone = {op for x in summaries for op in _not_run(x)}
    harness = {op: files for op, files in _harness_changed(summaries).items() if op not in gone}
    if harness:
        lines.append(
            "**Harness changes left out:** this PR also changes the test harness's Python for "
            + ", ".join(f"`{op}` ({_harness_files(files)})" for op, files in harness.items())
            + ", which the report runs from main: the numbers for "
            + ("that op" if len(harness) == 1 else "those ops")
            + " use main's tests and reference, not the PR's."
        )
    lines += [""]
    return lines


def _skipped_note(arch):
    return (
        f"{_ARCH[arch]} was not measured: this PR changes no device code {_ARCH[arch]} builds "
        "(`/llk-sfpu-test --arch all` measures it anyway)."
    )


def notes_section(summaries, not_measured=(), skipped=()):
    s = summaries[0]
    notes = [_skipped_note(a) for a in skipped]
    if s.get("not_applied"):
        notes.append(
            "Files this PR changes that the measurement does not use (host-side test code runs from main): "
            + ", ".join(code(p) for p in s["not_applied"][:10])
        )
    # A note every arch wrote is shown once; one only some archs wrote names them.
    per_arch = [x.get("notes", []) for x in summaries]
    for n in dict.fromkeys(n for arch_notes in per_arch for n in arch_notes):
        archs = [_ARCH[x["arch"]] for x, arch_notes in zip(summaries, per_arch) if n in arch_notes]
        notes.append(n if len(archs) == len(summaries) else f"{', '.join(archs)}: {n}")
    for x in not_measured:
        notes.append(f"{_ARCH[x['arch']]} was not measured. {x['not_measured']}")
    if not notes:
        return []
    return ["### Notes for reviewers", "", *[f"- {n}" for n in notes], ""]


def _runs(summaries):
    """Where each summary was measured: the CI run (one for all archs), else the host."""
    return ", ".join(dict.fromkeys(x.get("run_url") or x["host"] for x in summaries))


def not_measured_report(summaries):
    """The whole comment when no architecture could be measured: why, and what to do."""
    reasons = list(dict.fromkeys(x["not_measured"] for x in summaries))
    lines = [COMMENT_MARKER, "## LLK SFPU report", ""]
    if len(reasons) == 1:
        lines += [f"**Not measured** on {', '.join(_ARCH[x['arch']] for x in summaries)}. {reasons[0]}", ""]
    else:
        for x in summaries:
            lines += [f"**Not measured** on {_ARCH[x['arch']]}. {x['not_measured']}", ""]
    ran = [x for x in summaries if x.get("head_sha")]  # a leg that did not finish left no summary
    if not ran:
        return lines
    s = ran[0]
    if s.get("head_moved_to"):
        lines += [
            f"> [!NOTE]\n> The PR has moved to `{s['head_moved_to'][:10]}` since; comment `/llk-sfpu-test` to re-run.",
            "",
        ]
    conflicts = sorted({f for x in summaries for f in x.get("conflicts", [])})
    if conflicts:
        lines += ["Conflicting files:", "", *[f"- {code(f)}" for f in conflicts], ""]
    lines.append(
        f"PR head `{s['head_sha'][:10]}`, merge-base `{s['base_sha'][:10]}`, main `{s['tool_sha'][:10]}`. Runs: "
        + _runs(ran)
        + "."
    )
    return lines


def _modes(summaries):
    modes = {x["mode"] for x in summaries}
    if len(modes) == 1:
        return f"`{modes.pop()}`"
    return ", ".join(f"`{x['mode']}` ({_ARCH[x['arch']]})" for x in summaries)


def footer(summaries):
    s = summaries[0]
    head, base, tool = s["head_sha"], s["base_sha"], s["tool_sha"]
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
    ]
    for x in summaries:
        # Per arch: each detects its own ops, and one may have fallen back to rebase mode.
        mode = " --mode rebase" if x["mode"] == "rebase" else ""
        ops = ",".join(x["ops"]["measured"])
        cli = f"python3 .github/llk_sfpu_report/cli.py --arch {x['arch']} --head {head} --base {base}{mode} run"
        lines += [
            "",
            f"# the whole report, {_ARCH[x['arch']]}",
            f"{cli} --ops {ops or '<op>'} --check",
        ]
        per_op = {}
        for f in findings([x]):
            # A broadcast row's label ("SfpuElwadd (bcast Row)") is not an --ops name.
            fmts = per_op.setdefault(f["op"].split(" (", 1)[0], [])
            fmts += [m for m in f["fmt"].split(",") if m not in fmts]
        if per_op:
            lines.append(f"# only what regressed, {_ARCH[x['arch']]}")
            lines += [f"{cli} --ops {op} --formats {','.join(fmts)} --check" for op, fmts in list(per_op.items())[:10]]
    lines += [
        "```",
        "",
        f"Tool `{tool[:10]}`, mode {_modes(summaries)}. Runs: " + _runs(summaries) + ".",
        "",
        "</details>",
    ]
    return lines


def render(summaries, expected=(), run_url=None, skipped=()):
    """The comment. An arch in ``expected`` without a summary is reported as not
    measured: its leg failed or timed out before writing one. An arch in ``skipped``
    was never run: the PR changes no device code it builds."""
    present = {s["arch"] for s in summaries}
    run = f"the [run]({run_url})" if run_url else "the run"
    summaries = list(summaries) + [
        {"arch": a, "not_measured": f"Its measurement did not finish; see {run}."} for a in expected if a not in present
    ]
    summaries = sorted(summaries, key=lambda s: s["arch"] != "wormhole")
    not_measured = [s for s in summaries if s.get("not_measured")]
    summaries = [s for s in summaries if not s.get("not_measured")]
    if not summaries and not not_measured:
        return _fit("\n".join(nothing_to_measure_report(skipped, run_url)) + "\n")
    if not summaries:
        lines = not_measured_report(not_measured) + [f"{_skipped_note(a)}\n" for a in skipped]
        return _fit("\n".join(lines) + "\n")
    notes = notes_section(summaries, not_measured, skipped)
    if not any(s["ops"]["measured"] for s in summaries):
        # Empty tables and "no regressions" say nothing here; the notes say why.
        return _fit("\n".join(header(summaries) + notes + footer(summaries)) + "\n")
    not_run = not_run_section(summaries)
    parts = header(summaries) + not_run
    # The rest is about the ops that ran: what ran in place of the others is not the PR's.
    shown = [_without_not_run(s) for s in summaries]
    if any(s["ops"]["measured"] for s in shown):
        parts += glance_section(shown, not_measured, other_ops=bool(not_run))
        parts += perf_section(shown)
        parts += accuracy_section(shown)
        parts += edge_section(shown)
        parts += cross_arch_section(shown)
    parts += notes
    parts += footer(summaries)
    return _fit("\n".join(parts) + "\n")


def nothing_to_measure_report(skipped, run_url=None):
    """The whole comment when the PR changes no device code a measured chip builds."""
    archs = " or ".join(_ARCH[a] for a in skipped)
    lines = [
        COMMENT_MARKER,
        "## LLK SFPU report",
        "",
        f"**Not measured.** This PR changes no device code {archs} builds. The report compares "
        "device code only: host-side test code runs from main, and Quasar is not covered. "
        "`/llk-sfpu-test --arch all` measures both anyway.",
        "",
    ]
    if run_url:
        lines += [f"Run: {run_url}.", ""]
    return lines


#: A GitHub comment holds 65,536 characters; leave some headroom.
MAX_CHARS = 60000


def _fit(text):
    """Keep the comment postable: fold <details> bodies first, then cut tables."""
    if len(text) <= MAX_CHARS:
        return text
    text = re.sub(
        r"(<details><summary>(?!Reproduce<).*?</summary>)\n.*?\n</details>",
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
    out.append("\n_Some table rows were cut to fit a comment; the artifact has all of them._")
    return "\n".join(out)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Render the LLK SFPU report comment.")
    ap.add_argument("summaries", nargs="*", help="summary.json per architecture")
    ap.add_argument("-o", "--out", default="-")
    ap.add_argument("--expect", default="", help="comma-separated archs that should have a summary")
    ap.add_argument("--skipped", default="", help="comma-separated archs the PR's diff did not touch, so not run")
    ap.add_argument("--run-url", help="the CI run, linked for an arch that did not finish")
    args = ap.parse_args(argv)
    expected, skipped = ([a for a in arg.split(",") if a] for arg in (args.expect, args.skipped))
    if not (args.summaries or expected or skipped):
        ap.error("nothing to report: pass summaries, --expect or --skipped")
    summaries = [json.loads(Path(p).read_text()) for p in args.summaries]
    text = render(summaries, expected, args.run_url, skipped)
    if args.out == "-":
        print(text, end="")
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
