# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Renders results documents (one per op and arch) into the PR comment. Pure; no device, no ttnn.

python render.py results_bh.json results_wh.json > comment.md
"""

import json
import re
import sys

MARKER = "<!-- dm-perf-compare -->"
# GitHub rejects a comment over 65536 characters; the workflow adds its own header and run link.
COMMENT_LIMIT = 60000
# Detail levels, most to least: everything, no collapsed detail, only calls that need attention, no chart.
LEVELS = 4

# Reasons a call has no verdict that say nothing about the PR.
BENIGN_REASONS = ("no device time", "no round measured")

FOOTER = """\
<sub>**How this was measured.** "Default" is the op as users call it, with no implementation forced.
Each side runs from its own build in its own process; rounds alternate sides. Verdicts use device
time (sum of kernel durations, best of several reps, median over rounds) and count only when the change
clears the band measured in the same run and at least 4 of 5 rounds agree. Host time on shared CI machines
moves tens of percent between identical runs, so it is listed as detail against its own measured resolution
and never decides a verdict. Host-side cost of a routing change needs a local paired run.</sub>"""


def _label(call):
    shape = "x".join(str(d) for d in call["shape"])
    args = ", ".join("%s=%s" % (k, json.dumps(v, separators=(",", ":"))) for k, v in sorted(call["args"].items()))
    return " ".join(p for p in (shape, call["dtype"], call["layout"], str(call["memory_config"]), args) if p)


def _speed(pct):
    """One convention: percent change in time, with the word."""
    if abs(pct) < 0.05:
        return "0%"
    return "%.1f%% %s" % (abs(pct), "slower" if pct > 0 else "faster")


def _us(ns):
    return "-" if not ns else "%.2f" % (ns / 1000.0)


def _sha(tree):
    return (tree or {}).get("sha", "?")[:9]


def _arch(doc):
    names = doc.get("device_arch") or ["?"]
    return ", ".join(str(n).split(".")[-1].title() for n in names)


def _alarming(comparison):
    return not comparison["comparable"] and not comparison["reason"].startswith(BENIGN_REASONS)


def _device_ns(row):
    return (row.get("device") or {}).get("default", {}).get("ns")


def _only_option(row):
    after = row["after"]
    return after["codegen"]["status"] in ("refused", "error") or after["native"]["status"] in ("refused", "error")


def needs_a_look(doc):
    """One line per finding: slower calls, calls another implementation beats, results to distrust."""
    found = []
    for row in doc["calls"]:
        label = _label(row["call"])
        q1 = row["q1"]
        if q1["comparable"] and q1["verdict"] == "slower":
            found.append("**slower** %s: %s" % (_speed(q1["median_pct"]), label))
        elif _alarming(q1):
            found.append("**cannot compare** (%s): %s" % (q1["reason"], label))
        for impl in row["beaten_by"]:
            found.append(
                "**%s beats the default** by %s: %s"
                % (impl, _speed(row["q2"][impl]["median_pct"]).replace(" faster", ""), label)
            )
        for impl, q2 in row["q2"].items():
            if _alarming(q2) and q2["reason"].startswith("output differs"):
                found.append("**%s output differs from the default**: %s" % (impl, label))
    return found


def _answers(doc):
    s = doc["summary"]
    q1, q2 = s["q1"], s["q2"]
    geomean = s["q1_geomean_pct"]
    speed = "%d slower, %d faster, %d no change" % (q1["slower"], q1["faster"], q1["no change"])
    if q1["not comparable"]:
        speed += ", %d not comparable" % q1["not comparable"]
    if geomean is not None:
        speed += " (geomean %s)" % _speed(geomean)
    fastest = "%d yes, %d no" % (q2["default fastest"], q2["beaten"])
    if q2["not comparable"]:
        fastest += ", %d not comparable" % q2["not comparable"]
    return [
        "| Question | Answer |",
        "|---|---|",
        "| Did this PR change speed? | %s |" % speed,
        "| Does the default pick the fastest option? | %s |" % fastest,
    ]


def _sorted_rows(doc):
    """Slower first, calls without a Q1 verdict last."""
    comparable = [r for r in doc["calls"] if r["q1"]["comparable"]]
    rest = [r for r in doc["calls"] if not r["q1"]["comparable"]]
    return sorted(comparable, key=lambda r: -r["q1"]["median_pct"]) + rest


def _chart(doc, rows):
    changes = [round(r["q1"]["median_pct"], 2) for r in rows if r["q1"]["comparable"]]
    if not changes:
        return []
    band = round(doc["noise"]["device_band_pct"], 2)
    low, high = min(changes + [-band]), max(changes + [band])
    pad = max(1.0, (high - low) * 0.1)
    return [
        "```mermaid",
        "xychart-beta",
        '    title "Change in device time per call, slower first (band: ±%.1f%%)"' % band,
        "    x-axis [%s]" % ", ".join(str(i + 1) for i in range(len(changes))),
        '    y-axis "%% change in time" %d --> %d' % (low - pad, high + pad + 1),
        "    bar [%s]" % ", ".join(str(c) for c in changes),
        "    line [%s]" % ", ".join([str(band)] * len(changes)),
        "    line [%s]" % ", ".join([str(-band)] * len(changes)),
        "```",
    ]


def _fastest(row):
    used = row["fastest_option_used"]
    if used is None:
        reason = next((q["reason"] for q in row["q2"].values() if not q["comparable"]), "")
        return "n/a (%s)" % reason if reason else "n/a"
    if not used:
        return "**no**: " + ", ".join(row["beaten_by"])
    return "yes (only option)" if _only_option(row) else "yes"


def _table(rows, shown_ids):
    out = ["| # | Call | Before µs | After µs | Change | Fastest option used? |", "|--:|---|--:|--:|---|---|"]
    for n, row in enumerate(rows, 1):
        if id(row) not in shown_ids:
            continue
        q1 = row["q1"]
        before = (row.get("device_before") or {}).get("ns")
        change = _speed(q1["median_pct"]) if q1["comparable"] else "n/a (%s)" % q1["reason"]
        if q1["comparable"] and q1["verdict"] == "no change":
            change = "no change (%s)" % change
        out.append(
            "| %d | %s | %s | %s | %s | %s |"
            % (n, _label(row["call"]), _us(before), _us(_device_ns(row)), change, _fastest(row))
        )
    return out


def _detail(rows):
    out = [
        "| Call | Native µs | Codegen µs | Codegen vs native | Default uses | Host µs before → after |",
        "|---|--:|--:|---|---|---|",
    ]
    for row in rows:
        device = row.get("device") or {}
        native, codegen = (device.get(i, {}).get("ns") for i in ("native", "codegen"))
        versus = _speed((codegen / native - 1.0) * 100.0) if native and codegen else "-"
        host = row["q1_host"]
        before = row["before"]["default"]["us"]
        after = row["after_default_only"]["us"]
        host_us = "-"
        if host["comparable"]:
            host_us = "%.1f → %.1f%s" % (
                min(v for v in before if v is not None),
                min(v for v in after if v is not None),
                " (moved)" if host["moved"] else "",
            )
        out.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (_label(row["call"]), _us(native), _us(codegen), versus, row["default_uses"] or "-", host_us)
        )
    return out


def _skipped(doc):
    """Refused calls grouped by side, implementation and reason, so a column of dashes explains itself."""
    groups = {}
    for entry in doc["skipped"]:
        reason = re.sub(r"@ \S+", "", entry["error"] or "").strip()[:160]
        groups.setdefault((entry["side"], entry["impl"], entry["status"], reason), 0)
        groups[(entry["side"], entry["impl"], entry["status"], reason)] += 1
    return [
        "- %d × %s %s on the %s build: `%s`" % (n, impl, status, side, reason)
        for (side, impl, status, reason), n in sorted(groups.items())
    ]


def _section(doc, level):
    noise = doc["noise"]
    trees = doc.get("trees", {})
    rows = _sorted_rows(doc)
    out = [
        "### %s: %s" % (_arch(doc), doc["op"]),
        "",
        "before `%s` → after `%s` · %d calls · device band ±%.1f%%, host resolution ±%.0f%%"
        % (
            _sha(trees.get("before")),
            _sha(trees.get("after")),
            len(rows),
            noise["device_band_pct"],
            noise["host_resolution_pct"],
        ),
        "",
    ]
    if not noise["reliable"]:
        out += [
            '> ⚠️ The device rounds disagreed with each other, so "no change" and "default is fastest" prove nothing. Run again.',
            "",
        ]
    findings = needs_a_look(doc)
    out += ["**Needs a look**" if findings else "**Nothing needs a look.**", ""]
    out += ["- " + f for f in findings] + ([""] if findings else [])
    out += _answers(doc) + [""]
    if level < 3:
        out += _chart(doc, rows) + [""]
    flagged = {id(r) for r in rows if r["q1"].get("verdict") == "slower" or r["beaten_by"] or _alarming(r["q1"])}
    shown = rows if level < 2 else [r for r in rows if id(r) in flagged]
    out += _table(rows, {id(r) for r in shown}) + [""]
    if len(shown) < len(rows):
        out += [
            "%d calls without a finding left out to fit the comment; the results file has them all."
            % (len(rows) - len(shown)),
            "",
        ]
    if level < 1:
        out += ["<details><summary>Detail: native, codegen, default uses, host time</summary>", ""]
        out += _detail(rows) + [""]
        skipped = _skipped(doc)
        if skipped:
            out += ["Not measured:"] + skipped + [""]
        out += ["</details>", ""]
    return out


def render(docs, limit=COMMENT_LIMIT):
    """The comment body; sheds detail one level at a time until it fits `limit`."""
    docs = sorted(docs, key=lambda d: (d["op"], _arch(d)))
    text = ""
    for level in range(LEVELS):
        body = ["## DM perf compare", ""]
        for doc in docs:
            body += _section(doc, level)
        body += [FOOTER]
        text = MARKER + "\n" + "\n".join(body) + "\n"
        if len(text) <= limit:
            return text
    return text[: limit - 200] + "\n\n_Comment truncated; see the results file._\n"


def main(argv):
    docs = []
    for path in argv[1:]:
        with open(path) as f:
            docs.append(json.load(f))
    sys.stdout.write(render(docs))


if __name__ == "__main__":
    main(sys.argv)
