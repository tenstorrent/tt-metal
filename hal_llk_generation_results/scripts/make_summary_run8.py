#!/usr/bin/env python3
"""Short summary of the run-8 series (8a-8e): SUMMARY_RUN8.md and slide/slide_draft.md.
Usage (from the folder root):  python3 scripts/make_summary_run8.py
Reads only files in this folder, with the same sources and parsers as RESULTS.md (scripts/make_results.py).
Runs of the series that are not in runs/ yet are listed as pending; medians use the runs that are present."""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_results as mr  # noqa: E402

ROOT = mr.ROOT
SERIES = ["8a", "8b", "8c", "8d", "8e"]
CONFIG_KEYS = ["base_commit", "hal_commit", "docs_commit", "docs_invalid", "model", "claude_version", "timeout",
               "max_turns", "max_budget_usd", "tests", "expect_passed", "hal_filter", "sfpi_compiler"]


def compute(root=ROOT):
    """All numbers of the summary, read from runs/run8*/ and perf/perf_summary_hal.json."""
    perf_all = json.loads((root / "perf" / "perf_summary_hal.json").read_text())
    rows = []
    for n in SERIES:
        d = root / "runs" / f"run{n}"
        row = {"run": n, "present": d.is_dir()}
        rows.append(row)
        if not d.is_dir():
            continue
        m = mr.meta(d / "harness_logs" / "run_meta.txt")
        row["prompt"] = Path(m.get("prompt_file", "?")).name
        row["config"] = {k: m.get(k) for k in CONFIG_KEYS}
        expect = int(m["expect_passed"].rsplit("=", 1)[1])
        c = mr.harness_checks(d / "harness_logs" / "SUMMARY.txt")
        pcs = sorted((d / "postcheck").glob("checks_*.txt"))
        post = mr.postcheck(pcs[0]) if pcs else []
        checks = [c["original"], c["hal"], c["dirty"]] + [pf for _, _, pf in post]
        row["expect"] = expect
        row["n_checks"] = len(checks)
        row["postcheck_done"] = len(post) == 4
        row["harness_pass"] = all(p == expect and f == 0 for p, f in (c["original"], c["hal"], c["dirty"]))
        row["all_pass"] = (len(post) == 4 and all(e == 0 for _, e, _ in post)
                           and all(p == expect and f == 0 for p, f in checks))
        off = d / "static_counts" / "assert_off" / "tt_tti_counts.md"
        if off.exists():
            t = mr.avg_table(off)
            row["orig"], row["hal"] = t[("original", "sum")], t[("hal", "sum")]
        p = perf_all.get(n)
        if p:
            row["cyc"] = {"orig": p["l1"]["orig"], "hal": p["l1"]["new"], "ratio": p["l1"]["ratio_median"],
                          "n": p["l1"]["n"], "faster": p["l1"]["faster_variants"],
                          "unp": p["unp"]["ratio_median"], "math": p["math"]["ratio_median"],
                          "pack": p["pack"]["ratio_median"]}

    present = [r for r in rows if r["present"]]
    static = [r for r in present if "hal" in r]
    cyc = [r for r in present if "cyc" in r]
    S = {"rows": rows, "present": [r["run"] for r in present], "n_static": len(static), "n_cyc": len(cyc)}

    # identical configuration inside the series (else the series is not 5 repeats)
    diff = []
    if present:
        ref = present[0]
        for k in CONFIG_KEYS:
            if any(r["config"].get(k) != ref["config"].get(k) for r in present):
                diff.append(k)
        if any(r["prompt"] != ref["prompt"] for r in present):
            diff.append("prompt_file")
        S["prompt"] = ref["prompt"]
        S["expect"] = ref["expect"]
    S["config_diff"] = diff

    # original side: same frozen code in every run, so its counts must be identical
    if static:
        o = static[0]["orig"]
        S["orig"] = o
        S["orig_same"] = all(r["orig"] == o for r in static)
        for c in ("text", "tti", "tt_all"):
            vals = [r["hal"][c] for r in static]
            S[f"hal_{c}"] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    if cyc:
        ratios = [r["cyc"]["ratio"] for r in cyc]
        S["cyc_ratio"] = {"median": statistics.median(ratios), "min": min(ratios), "max": max(ratios)}
        S["cyc_orig"] = sorted({round(r["cyc"]["orig"], 2) for r in cyc})
        S["cyc_faster"] = sum(r["cyc"]["faster"] for r in cyc)
        S["cyc_variants"] = sum(r["cyc"]["n"] for r in cyc)
    S["n_pass"] = sum(1 for r in present if r["all_pass"])
    S["n_harness_pass"] = sum(1 for r in present if r["harness_pass"])
    S["no_postcheck"] = [r["run"] for r in present if not r["postcheck_done"]]
    S["no_cyc"] = [r["run"] for r in present if "cyc" not in r]
    S["n_postcheck"] = sum(1 for r in present if r["postcheck_done"])
    S["has_7"] = any(p.name.startswith("run7") for p in (root / "runs").iterdir())
    return S


# ---------- rendering (English) ----------
def num(x, dec=0):
    return f"{x:.{dec}f}"


def pct(new, orig):
    v = round((new / orig - 1) * 100)
    return f"+{v}%" if v > 0 else (f"−{-v}%" if v < 0 else "0%")


def rng(d, dec=0, unit=""):
    """'4191 B' for one run, '4191 B (4100–4300)' for several."""
    s = f"{num(d['median'], dec)}{unit}"
    return s if d["min"] == d["max"] else f"{s} ({num(d['min'], dec)}–{num(d['max'], dec)})"


def series_label(S):
    p = S["present"]
    if len(p) == len(SERIES):
        return f"runs {SERIES[0]}–{SERIES[-1]} ({len(SERIES)} repeats with the same setup)"
    missing = [n for n in SERIES if n not in p]
    have = ", ".join(p) if p else "none"
    return f"run{'s' if len(p) != 1 else ''} {have} so far ({', '.join(missing)}: same setup, still running)"


def summary_md(S):
    o = S.get("orig")
    L = ["# Run 8 series (8a–8e): HAL vs original, key numbers", "",
         "Generated by `python3 scripts/make_summary_run8.py` from the files in `runs/` and `perf/` "
         "(same sources as `RESULTS.md`, which has the full tables).", ""]
    if len(S["present"]) < len(SERIES):
        L += [f"**Status: {series_label(S)}.** Medians below use only the runs that are present.", ""]
    L += ["**Task.** A Claude Code agent (claude-opus-5-5) re-implements the LLK functions used by tt-llk "
          "`test_eltwise_binary` (ELWADD, `-k add`, Blackhole p150b) from a signature-only spec, using only HAL, which it "
          "knows from the documentation alone. HAL source and the original LLKs are read-denied.", "",
          f"**Setup.** Identical in every run of the series (`runs/run8*/harness_logs/run_meta.txt`): same commits, model, "
          f"limits, frozen test set and prompt `{S.get('prompt', '?')}`."]
    if S["config_diff"]:
        L += ["", "**WARNING: these settings differ between the runs:** " + ", ".join(f"`{k}`" for k in S["config_diff"])]
    L += ["", "| | Tests passed | Code size `.text` (B) | TTI | TT | Cycles per tile, HAL ÷ original |",
          "|---|---|---|---|---|---|"]
    if o:
        oc = (f"1 ({' / '.join(num(x, 1) for x in S['cyc_orig'])} cycles)" if "cyc_orig" in S else "")
        L.append(f"| Original LLK | {S['expect']} of {S['expect']} | {num(o['text'])} | {num(o['tti'])} | "
                 f"{num(o['tt_all'])} | {oc} |")
    for r in S["rows"]:
        if not r["present"]:
            L.append(f"| HAL {r['run']} | still running | | | | |")
            continue
        if not r["postcheck_done"]:
            tp = (f"{r['expect']} of {r['expect']} in the 3 harness checks; postcheck not run yet" if r["harness_pass"]
                  else "**FAIL in a harness check**; postcheck not run yet")
        elif r["all_pass"]:
            tp = f"{r['expect']} of {r['expect']}, all {r['n_checks']} checks"
        else:
            tp = "**FAIL in at least one check**"
        if "hal" in r:
            h, oo = r["hal"], r["orig"]
            st = (f"{num(h['text'])} ({pct(h['text'], oo['text'])}) | {num(h['tti'])} ({pct(h['tti'], oo['tti'])}) | "
                  f"{num(h['tt_all'])} ({pct(h['tt_all'], oo['tt_all'])})")
        else:
            st = "not built yet | | "
        cy = f"{r['cyc']['ratio']:.2f}×" if "cyc" in r else "not measured yet"
        L.append(f"| HAL {r['run']} | {tp} | {st} | {cy} |")
    if o and S["n_static"]:
        cy = (f"**{S['cyc_ratio']['median']:.2f}×**" + (f" ({S['n_cyc']} runs)" if S["n_cyc"] < S["n_static"] else "")
              if "cyc_ratio" in S else "")
        tp = f"all 7 checks pass: {S['n_pass']} of {len(S['present'])} runs" + (
            f"; harness checks only: {', '.join(S['no_postcheck'])}" if S["no_postcheck"] else "")
        L.append(f"| **HAL median ({S['n_static']} of {len(SERIES)} runs)** | {tp} | "
                 f"**{num(S['hal_text']['median'])} ({pct(S['hal_text']['median'], o['text'])})** | "
                 f"**{num(S['hal_tti']['median'])} ({pct(S['hal_tti']['median'], o['tti'])})** | "
                 f"**{num(S['hal_tt_all']['median'])} ({pct(S['hal_tt_all']['median'], o['tt_all'])})** | {cy} |")
    L += ["", "## What the columns mean", "",
          f"- **Tests passed:** `test_eltwise_binary -k add` must give {S.get('expect', '?')} passed, 0 failed in every check: "
          "original (harness baseline), HAL after a card reset, original right after HAL without reset (harness), and "
          "the 4 postcheck steps (reset→HAL, no reset→original, no reset→HAL, reset→HAL). Sources: "
          "`harness_logs/SUMMARY.txt`, `postcheck/checks_*.txt`.",
          "- **Code size, TTI, TT:** average per build over the 10 builds of `-k add`, unpack + math + pack, both sides "
          "compiled without LLK asserts (`TT_LLK_DISABLE_ASSERTS=1`). Source: `static_counts/assert_off/tt_tti_counts.md`. "
          "TTI = Tensix instructions embedded in the code; TT = Tensix instructions built at run time and written to the "
          "instruction buffer (static estimate). Counts in the compiled code, not executed instructions.",
          "- **Cycles:** LLK perf test `perf_eltwise_binary.py -k Elwadd`, L1→L1 cycles per tile, asserts off, original and "
          "HAL alternating on the same board; median of 3 runs per variant, then median of the per-variant HAL ÷ original "
          "ratios over 16 variants. Source: `perf/perf_summary_hal.json`." +
          (f" HAL was faster than the original in {S['cyc_faster']} of {S['cyc_variants']} measured variants."
           if "cyc_faster" in S else "") +
          (" Per run, HAL ÷ original for the whole kernel (L1→L1) and for each thread measured alone "
           "(unpack / math / pack isolate): " + "; ".join(
               f"{r['run']} {r['cyc']['ratio']:.2f} ({r['cyc']['unp']:.2f} / {r['cyc']['math']:.2f} / {r['cyc']['pack']:.2f})"
               for r in S["rows"] if "cyc" in r) + "." if "cyc_faster" in S else ""),
          "", "## Limits", "",
          "- One kernel (eltwise add) on one board type.",
          "- The perf test does not check results. Only 4 of its 16 Elwadd variants use formats that the functional test "
          "(and therefore the agent) verified.",
          "- TT is an estimate: some stores to the instruction buffer could not be resolved statically (column "
          "'unresolved' in `tt_tti_counts.md`)."]
    if any(n in S["present"] for n in SERIES[1:]):
        L.append("- 8a ran with the harness version that writes Serbian messages, 8b–8e with the English one "
                 "(`scripts/harness/VERSIONS.md`). The agent-visible difference is the language of five spec marker "
                 "comments; the deny list was adapted to the reorganised folders on bh-30.")
    if S.get("orig") and not S["orig_same"]:
        L.append("- **WARNING:** the original's static counts differ between runs; the table shows those of the first run.")
    if S["has_7"]:
        L.append("- Runs 7a–7e did the same task with an earlier prompt (without the section on instruction cost of "
                 "per-tile code); they are in `RESULTS.md`.")
    return "\n".join(L) + "\n"


def slide_md(S):
    o = S.get("orig")
    n = len(S["present"])
    L = ["# Slide draft (one slide)", "",
         "Generated by `python3 scripts/make_summary_run8.py`; numbers from `SUMMARY_RUN8.md`.", ""]
    if n < len(SERIES):
        L += [f"*DRAFT: numbers are from {series_label(S)}.*", ""]
    if not o:
        return "\n".join(L + ["No run of the series has static counts yet."]) + "\n"
    if S["n_pass"] == n:
        tests = (f"All {n} pass" if n > 1 else "It passes") + f" all {S['expect']} test variants, also after a card reset " \
            "and right after the original kernel."
    elif S["n_harness_pass"] == n:
        tests = (f"All {n} pass all {S['expect']} test variants after a card reset; the full recheck (HAL right after the "
                 f"original kernel, repeated resets) passed for {S['n_pass']} of {n} and is not run yet for "
                 f"{', '.join(S['no_postcheck'])}.")
    else:
        tests = f"Only {S['n_harness_pass']} of {n} pass all {S['expect']} test variants (see SUMMARY_RUN8.md)."
    agg = "median of the 5 runs, range in brackets" if S["n_static"] == len(SERIES) else \
        ("run " + S["present"][0] if S["n_static"] == 1 else f"median of {S['n_static']} runs, range in brackets")
    L += ["**AI agent writes Blackhole LLK kernels on top of HAL**", "",
          "- **Experiment.** A Claude Code agent (claude-opus-5-5) re-implements the LLK functions of tt-llk "
          f"`test_eltwise_binary` (ELWADD) from a signature-only spec, using only HAL, which it knows from the "
          f"documentation alone. {series_label(S)[0].upper() + series_label(S)[1:]}. {tests}",
          "  Details and reproduction: **[GitHub draft PR: link to be added]**",
          f"- **Results.** HAL code vs the hand-written original LLKs ({agg}):",
          f"  - **Code size (`.text`):** {num(o['text'])} → {rng(S['hal_text'], unit=' B')}, "
          f"**{pct(S['hal_text']['median'], o['text'])}**",
          f"  - **TTI** (Tensix instructions embedded in the code): {num(o['tti'])} → {rng(S['hal_tti'])}, "
          f"**{pct(S['hal_tti']['median'], o['tti'])}**",
          f"  - **TT** (Tensix instructions built at run time, static estimate): {num(o['tt_all'])} → "
          f"{rng(S['hal_tt_all'])}, **{pct(S['hal_tt_all']['median'], o['tt_all'])}**"]
    if "cyc_ratio" in S:
        c = S["cyc_ratio"]
        parts = ([] if c["min"] == c["max"] else [f"{c['min']:.2f}–{c['max']:.2f}"]) + \
            ([f"{S['n_cyc']} runs, {', '.join(S['no_cyc'])} not measured yet"] if S["no_cyc"] else [])
        rr = f" ({'; '.join(parts)})" if parts else ""
        L.append(f"  - **Speed** (LLK perf test, cycles per tile): HAL needs **{c['median']:.2f}×** the cycles of "
                 f"the original{rr}")
    L += ["", "*Footnote: static counts in the compiled code (both sides without LLK asserts), not executed "
          "instructions; one kernel (eltwise add).*"]
    return "\n".join(L) + "\n"


def main():
    S = compute()
    (ROOT / "SUMMARY_RUN8.md").write_text(summary_md(S))
    (ROOT / "slide" / "slide_draft.md").write_text(slide_md(S))
    print(f"wrote SUMMARY_RUN8.md and slide/slide_draft.md (runs present: {' '.join(S['present']) or 'none'}; "
          f"static counts: {S['n_static']}, cycles: {S['n_cyc']}, config differences: {S['config_diff'] or 'none'})")


if __name__ == "__main__":
    main()
