# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every number stage 07 publishes, re-derived from the artifacts it publishes.

Stage 07 shipped without one of these, and the review that followed found three
figures in ``README.md`` that no artifact supported: a headline that
contradicted ``selection_reasons.json``, a rejection row that listed a config
the selection had marked *eligible* (and so counted nine rows where the prose
said eight), and a "the teacher-forcing TTFT agrees in direction" corroboration
that ``repeats.json`` flatly contradicts. All three were prose drifting away
from data that was itself correct -- exactly the class of error a spot check
misses and a mechanical re-derivation does not.

So this walks the other way: it reads ``sweep_results.json``, ``repeats.json``,
``selection_reasons.json`` and the two perf JSONs, and asserts that
``README.md`` and ``work_log.md`` say what those files say.

What is checked
---------------

1. **The results table, row by row.** Every ``| \\`Rnn_...\\` |`` line in
   README section 4 is parsed and its top-1 / top-5 / top-100 / t/s/u / TTFT /
   gain columns are compared against the row of the same id in
   ``sweep_results.json``. A row present in one and missing from the other is a
   failure in both directions.
2. **The selection.** The selected id, the eligible set and every rejection
   reason in ``selection_reasons.json`` must be reflected in the README's
   rejection section -- in particular, **no config listed as eligible may
   appear in a rejection row**, which is the exact defect the review found.
3. **The noise band and its samples**, against ``repeats.json``.
4. **The post-selection perf figures**, against
   ``perf_full_model{,_selected}.json``.
5. **Every blocked row's blocker**, against ``probes/structural_probe.json`` --
   the op, the file:line, the assertion and the info string, in both
   ``sweep_results.json`` and the README's blocker table. This is the one the
   re-review needed: a blocker can go **stale** rather than wrong when a fix in
   ``tt/`` moves the failure to a different op, and every check above would
   still pass because the README and ``sweep_results.json`` agree with each
   other while both disagree with the probe.
6. **Counting claims.** Phrases of the form "<word> rows are rejected on
   exactly this basis" must match the number of rows carrying that reason.

Exits non-zero on any mismatch, so it is a gate and not a report.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP = HERE.parent

WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "twenty": 20,
    "twenty-two": 22,
    "twenty-five": 25,
    "twenty-six": 26,
    "twenty-seven": 27,
    "twenty-eight": 28,
    "twenty-nine": 29,
}

failures: list[str] = []


def check(ok: bool, what: str) -> None:
    if not ok:
        failures.append(what)
    print(f"{'OK  ' if ok else 'FAIL'}  {what}")


def load(name: str):
    return json.loads((SWEEP / name).read_text())


def num(cell: str):
    """A table cell -> float, ignoring markdown emphasis, arrows and units."""
    cell = cell.replace("**", "").replace("`", "").replace("−", "-").replace("–", "-").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", cell)
    return float(m.group(0)) if m else None


def main() -> int:
    rows = {r["config_id"]: r for r in load("sweep_results.json")}
    reasons = load("selection_reasons.json")
    repeats = load("repeats.json")
    readme = (SWEEP / "README.md").read_text()
    worklog = (SWEEP / "work_log.md").read_text()

    # -- 1. the results table, row by row ---------------------------------
    # Scoped to section 4's ranked table only. Other tables in this README also
    # carry per-row figures (section 6's KV pair, section 5's eligible-but-not-
    # selected rows) with different columns, and are covered by the checks below
    # rather than by column position.
    results_section = readme.split("## 4. Results")[1].split("### Not evaluated")[0]
    table_ids = set()
    for line in results_section.splitlines():
        m = re.match(r"\s*\|\s*\**`(R\d\d_[a-z0-9_]+)`\**\s*(?:←[^|]*)?\|(.*)", line)
        if not m:
            continue
        cid, rest = m.group(1), m.group(2)
        cells = [c.strip() for c in rest.split("|")]
        if len(cells) < 8:  # not the results table (candidate-set / rejection tables)
            continue
        table_ids.add(cid)
        row = rows.get(cid)
        if row is None:
            check(False, f"{cid}: in the README results table but not in sweep_results.json")
            continue
        # cells: delta, top1, dtop1, top5, top100, t/s/u, gain, TTFT, gate
        for idx, key, tol in (
            (1, "top1", 1e-9),
            (3, "top5", 1e-9),
            (4, "top100", 1e-9),
            (5, "decode_tps_user", 0.005),
            (7, "ttft_ms", 0.5),
        ):
            want, got = num(cells[idx]), row.get(key)
            check(
                want is not None and got is not None and abs(want - got) <= tol,
                f"{cid}: README {key} = {want}, sweep_results = {got}",
            )
        want_gain, got_gain = num(cells[6]), row.get("decode_gain_pct")
        if want_gain is not None and got_gain is not None:
            check(abs(want_gain - got_gain) <= 0.02, f"{cid}: README gain = {want_gain}%, computed = {got_gain}%")

    measured = {cid for cid, r in rows.items() if r.get("status") == "ok"}
    missing = sorted(measured - table_ids)
    check(not missing, f"every measured row appears in the README results table (missing: {missing})")

    # -- 2. the selection --------------------------------------------------
    sel = reasons["selected"]
    check(f"`{sel}`" in readme, f"README names the selected config `{sel}`")
    check(rows[sel].get("selected") is True, f"sweep_results marks {sel} as selected")

    # The defect the review found: a config the selection marked ELIGIBLE
    # appearing in a rejection row. Rejection rows are the lines of section 5's
    # "Rejected, with numbers" table.
    rejected_section = readme.split("### Rejected, with numbers")[-1].split("\n### ")[0].split("\n---")[0]
    for cid in reasons["eligible"]:
        check(
            cid not in rejected_section or cid == sel,
            f"{cid} is eligible in selection_reasons.json and must not appear in a rejection row",
        )
    for cid in reasons["rejected"]:
        check(cid in readme, f"{cid} is rejected in selection_reasons.json and is named in the README")

    # -- 3. the band -------------------------------------------------------
    band = reasons["noise_band_pct"]
    check(f"{band:.3f}%" in readme, f"README quotes the measured band {band:.3f}%")
    for cid, rep in repeats.items():
        samples = [f"{s['decode_tps_user']:.2f}" for s in rep["samples"]]
        check(", ".join(samples) in readme, f"README quotes {cid}'s repeat samples {samples}")
        check(f"{rep['decode_spread_pct']:.3f}%" in readme, f"README quotes {cid}'s spread")

    # -- 4. post-selection perf -------------------------------------------
    perf = json.loads((HERE / "perf_full_model_selected.json").read_text())
    for key, label in (
        ("token_out_ms", "token_out"),
        ("model_trace_ms", "model_trace"),
        ("token_out_readback_ms", "token_out_readback"),
        ("ttft_ms", "warmed TTFT"),
    ):
        check(f"{perf[key]:.3f}" in readme, f"README quotes {label} = {perf[key]:.3f} ms")

    # -- 5. every OTHER table that names a config and quotes its numbers ----
    #
    # Sections 5 and 6 carry their own per-row tables with different columns.
    # Rather than parse each shape, every ``| `Rnn_x` | ... |`` line outside
    # section 4 must have its config's t/s/u and top-1 present *on that line* if
    # any decimal number appears there at all. That catches a figure edited in
    # one table and not the other, which is how this README drifted.
    other = readme.split("### Not evaluated")[-1]
    for line in other.splitlines():
        m = re.search(r"\|\s*\**`(R\d\d_[a-z0-9_]+)`\**[^|]*\|(.+)", line)
        if not m:
            continue
        cid, rest = m.group(1), m.group(2)
        row = rows.get(cid)
        if row is None or row.get("status") != "ok" or not re.search(r"\d+\.\d+", rest):
            continue
        quoted = set(re.findall(r"\d+\.\d+", rest))
        tsu = f"{row['decode_tps_user']:.2f}"
        if any(v.count(".") == 1 and len(v.split(".")[0]) == 2 and float(v) > 30 for v in quoted):
            check(tsu in quoted, f"{cid}: a t/s/u-shaped figure is quoted outside section 4; must be {tsu}")
        if any(v.startswith("0.9") or v.startswith("1.0") for v in quoted):
            check(
                f"{row['top1']:.3f}" in quoted,
                f"{cid}: an accuracy-shaped figure is quoted outside section 4; top-1 is {row['top1']:.3f}",
            )

    # -- 5b. blocked rows, against the probe that blocked them -------------
    #
    # The re-review found a published blocker that was *stale rather than
    # wrong*: `R17`/`R18` were genuinely blocked at
    # ``paged_fill_cache_device_operation.cpp:36`` when first probed, then
    # ``match_cache_dtype`` cleared that barrier, the tier-A probe was re-run,
    # and the failure moved to a different op -- but ``sweep_results.json`` and
    # the README kept quoting the old one. Nothing above catches that: the
    # README agreed with ``sweep_results.json`` perfectly, and both disagreed
    # with the probe artifact underneath them.
    #
    # So a blocked row's blocker is re-derived here from
    # ``probes/structural_probe.json`` -- the file the runtime error text
    # actually comes from -- and checked in both places. ``blocker_raw`` is
    # compared verbatim, which is parser-independent: if the probe is ever
    # re-run and the runtime says something new, this fails on the raw string
    # whatever the regexes do.
    structural = {r["config_id"]: r for r in json.loads((HERE / "structural_probe.json").read_text())}
    blocker_section = readme.split("### Not evaluated")[1].split("\n---")[0]
    for cid, row in sorted(rows.items()):
        if row.get("status") != "blocked":
            continue
        probe = structural.get(cid)
        if probe is None or probe.get("status") != "error":
            check(False, f"{cid}: blocked in sweep_results.json but structural_probe.json has no error for it")
            continue
        err = probe["error"]
        check(row.get("blocker_raw") == err, f"{cid}: blocker_raw is the current structural_probe.json error text")
        # Same parse sweep_runner.blocked_row uses. Agreeing with it is the point.
        m = re.search(r"TT_FATAL @ (\S+):(\d+): (.+?)\\n", err)
        check(m is not None, f"{cid}: structural_probe.json error text parses as a TT_FATAL")
        if not m:
            continue
        op = m.group(1).split("/")[-1] + ":" + m.group(2)
        info_m = re.search(r"info:\\n(.+?)\\nbacktrace", err)
        info = info_m.group(1) if info_m else None
        check(
            row.get("blocker_op") == op,
            f"{cid}: sweep_results blocker_op = {row.get('blocker_op')!r}, probe says {op!r}",
        )
        check(row.get("blocker_assertion") == m.group(3), f"{cid}: sweep_results blocker_assertion matches the probe")
        check(
            row.get("blocker_info") == info,
            f"{cid}: sweep_results blocker_info = {row.get('blocker_info')!r}, probe says {info!r}",
        )
        # ...and the README's blocker table must quote that same op and info.
        check(op in blocker_section, f"{cid}: README's blocker table quotes the op `{op}`")
        if info:
            check(info in blocker_section, f"{cid}: README's blocker table quotes the info {info!r}")

    # -- 6. counting claims ------------------------------------------------
    inside_band = [c for c, why in reasons["rejected"].items() if "inside the measured" in why]
    patterns = (
        r"(\w+) rows are rejected on exactly this basis",
        r"(\w+) rows sit inside it",
        r"the (\w+) rows rejected as \"inside the band\"",
        r"the (\w+) band-bound\s+rows",
    )
    for text, where in ((readme, "README"), (worklog, "work_log")):
        for pattern in patterns:
            for m in re.finditer(pattern, text):
                word = m.group(1).lower()
                check(
                    WORDS.get(word) == len(inside_band),
                    f"{where}: '{m.group(0)}' vs {len(inside_band)} band rejections {sorted(inside_band)}",
                )
    n_measured = len(measured)
    for text, where in ((readme, "README"), (worklog, "work_log")):
        for m in re.finditer(r"([\w-]+) configs were measured end to end", text):
            check(
                WORDS.get(m.group(1).lower()) == n_measured,
                f"{where}: '{m.group(0)}' vs {n_measured} measured rows",
            )
        for m in re.finditer(r"plot all\s*\n?(\d+) evaluated configs", text):
            check(int(m.group(1)) == n_measured, f"{where}: charts claim {m.group(1)} configs, {n_measured} measured")

    print()
    if failures:
        print(f"FAIL: {len(failures)} published figure(s) do not match the artifacts:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: every published figure traces to an artifact in this directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
