#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summary of one or more hal-llk-gen runs, built only from the files the stages wrote.

  run_summary.py run <state dir> [--json out.json] [--md out.md]     one run: text to stdout
  run_summary.py compare <state dir> [<state dir> ...] [--md out.md]  table over runs: text to stdout

<state dir> = <WORK>/runs/<name> (run.env, stages.tsv, run_dir.txt, postcheck/, perf/, static/, dyn_status.txt).
Nothing here reads the agent's transcript or final report.
"""
import argparse
import json
import os
import re
import statistics as st
import sys

TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE = os.path.join(TOOL_DIR, "reference", "run_series.json")
COUNT_RE = re.compile(
    r"(\d+) (passed|failed|skipped|errors?|deselected|xfailed|xpassed)"
)
STEP_NAMES = {
    "1": "postcheck 1: tt-smi -r, HAL",
    "2": "postcheck 2: no reset, original",
    "3": "postcheck 3: no reset, HAL",
    "4": "postcheck 4: tt-smi -r, HAL",
}


# ----------------------------------------------------------------------------------- readers ---
def read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return None


def read_kv(path):
    out = {}
    for line in (read_text(path) or "").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def last_line(path):
    lines = [x for x in (read_text(path) or "").splitlines() if x.strip()]
    return lines[-1] if lines else ""


def pytest_counts(line):
    """'=== 1660 passed, 20 skipped, 2800 deselected in 201s ===' -> {'passed': 1660, ...}; None without counts."""
    if " in " not in line:
        return None
    found = {}
    for n, kind in COUNT_RE.findall(line):
        found["errors" if kind.startswith("error") else kind] = int(n)
    return found or None


def expected_counts(spec):
    """'test_eltwise_binary:add=1660 ...' -> {'test_eltwise_binary': 1660} (keyed by test name)."""
    out = {}
    for item in (spec or "").split():
        if "=" in item:
            key, n = item.rsplit("=", 1)
            out[key.split(":")[0]] = int(n)
    return out


def fmt(x, digits=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{digits}f}"
    return f"{x:,}"


def pct(new, old):
    if new is None or old in (None, 0):
        return "-"
    return f"{100.0 * (new - old) / old:+.1f}%"


# ---------------------------------------------------------------------------------- collect ---
def check(stage, step, test, path, expected, exit_code=None):
    line = last_line(path)
    c = pytest_counts(line) or {}
    ok = (
        exit_code in (0, None)
        and c.get("passed", 0) > 0
        and c.get("failed", 0) == 0
        and c.get("errors", 0) == 0
        and (expected is None or c.get("passed") == expected)
    )
    if exit_code is None and not c:
        ok = False
    return dict(
        stage=stage,
        step=step,
        test=test,
        file=path,
        exit=exit_code,
        passed=c.get("passed"),
        failed=c.get("failed", 0) if c else None,
        skipped=c.get("skipped", 0) if c else None,
        errors=c.get("errors", 0) if c else None,
        expected=expected,
        ok=ok,
        present=os.path.exists(path),
    )


def static_side(mode_dir):
    """Per build averages, unpack + math + pack: tt_tti_counts.json if present, else codesize.json."""
    t = read_json(os.path.join(mode_dir, "tt_tti_counts.json"))
    if t and t.get("runs"):
        s = t["runs"][0]["summary"]
        out = {"source": "tt_tti_counts"}
        for side in ("original", "hal"):
            a = s[side]["sum"]["avg"]
            out[side] = dict(
                text_bytes=a.get("text_bytes"),
                tti=a.get("tti"),
                tt=a.get("tt_all_paths"),
                tt_unresolved=a.get("unresolved"),
                builds=s[side]["sum"].get("builds"),
            )
        return out
    c = read_json(os.path.join(mode_dir, "codesize.json"))
    if c and c.get("per_component"):
        pc = c["per_component"]
        out = {"source": "codesize"}
        for side, key in (("original", "orig"), ("hal", "hal")):
            try:
                out[side] = dict(
                    text_bytes=round(
                        sum(
                            pc[k][key]["text"]["mean"]
                            for k in ("unpack", "math", "pack")
                        ),
                        1,
                    ),
                    tti=round(
                        sum(
                            pc[k][key]["instr_tensix"]["mean"]
                            for k in ("unpack", "math", "pack")
                        ),
                        1,
                    ),
                    tt=None,
                    tt_unresolved=None,
                    builds=pc["unpack"][key]["text"].get("n"),
                )
            except (KeyError, TypeError):
                return None
        return out
    return None


def collect(state):
    env = read_kv(os.path.join(state, "run.env"))
    run_dir = (read_text(os.path.join(state, "run_dir.txt")) or "").strip() or None
    logs = os.path.join(run_dir, "logs") if run_dir else None
    meta = read_kv(os.path.join(logs, "run_meta.txt")) if logs else {}
    metrics = read_json(os.path.join(logs, "metrics.json")) if logs else None
    stages = []
    for line in (read_text(os.path.join(state, "stages.tsv")) or "").splitlines():
        p = line.split("\t")
        if len(p) >= 3:
            stages.append(
                dict(
                    stage=p[0],
                    status=p[1],
                    exit=p[2],
                    start=p[3] if len(p) > 3 else "",
                    end=p[4] if len(p) > 4 else "",
                )
            )
    tests = [
        t.split(":")[0] for t in (meta.get("tests") or env.get("TESTS", "")).split()
    ]
    expected = expected_counts(meta.get("expect_passed") or env.get("EXPECT_PASSED"))
    s = dict(
        name=os.path.basename(os.path.normpath(state)),
        state_dir=os.path.abspath(state),
        run_dir=run_dir,
        profile=env.get("PROFILE"),
        reference_series=env.get("REFERENCE_SERIES"),
        stages=stages,
    )
    s["config"] = dict(
        base_commit=meta.get("base_commit") or env.get("BASE_COMMIT"),
        hal_commit=meta.get("hal_commit") or env.get("HAL_COMMIT"),
        docs_commit=meta.get("docs_commit") or env.get("DOCS_COMMIT"),
        prompt=os.path.basename(meta.get("prompt_file") or env.get("PROMPT_FILE", "")),
        prompt_md5=meta.get("prompt_md5"),
        model=meta.get("model") or env.get("MODEL"),
        claude_version=(meta.get("claude_version") or "").split(" ")[0] or None,
        tests=meta.get("tests") or env.get("TESTS"),
        expect_passed=meta.get("expect_passed") or env.get("EXPECT_PASSED"),
        host=meta.get("host"),
        dry_run=meta.get("dry_run") == "1",
    )

    agent = dict(exit_code=None, minutes=None)
    if "claude_exit_code" in meta:
        agent["exit_code"] = int(meta["claude_exit_code"])
        agent["minutes"] = round(int(meta.get("claude_seconds", 0)) / 60.0, 1)
    if metrics:
        proc = metrics.get("process") or {}
        ffp = proc.get("first_full_pass") or {}
        fin = proc.get("final_code_first_full_pass") or {}
        first = sorted(ffp.items())[0][1] if ffp else None
        final = sorted(fin.items())[0][1] if fin else None
        agent.update(
            cost_usd=metrics.get("cost_usd"),
            turns=metrics.get("num_turns"),
            pytest_commands=metrics.get("pytest_runs_total"),
            pytest_by_kind=metrics.get("pytest_runs_by_kind"),
            iterations=proc.get("iterations"),
            first_full_pass_iteration=first.get("iteration") if first else None,
            first_full_pass_min=first.get("elapsed_min") if first else None,
            final_code_full_pass_iteration=final.get("iteration") if final else None,
            denied_tool_calls=(metrics.get("counts") or {}).get("denied"),
        )
    s["agent"] = agent

    checks = []
    if logs:
        for t in tests:
            exp = expected.get(t)
            hal_exit = meta.get(f"hal_test_exit_code_{t}")
            dirty_exit = meta.get(f"dirty_exit_code_{t}")
            checks.append(
                check(
                    "harness",
                    "original (baseline, after tt-smi -r)",
                    t,
                    os.path.join(logs, f"baseline_{t}.txt"),
                    exp,
                )
            )
            if hal_exit is not None or os.path.exists(
                os.path.join(logs, f"hal_test_{t}.txt")
            ):
                checks.append(
                    check(
                        "harness",
                        "HAL after tt-smi -r",
                        t,
                        os.path.join(logs, f"hal_test_{t}.txt"),
                        exp,
                        int(hal_exit) if hal_exit is not None else None,
                    )
                )
            if dirty_exit is not None:
                checks.append(
                    check(
                        "harness",
                        "original right after HAL, no reset",
                        t,
                        os.path.join(logs, f"dirty_{t}.txt"),
                        exp,
                        int(dirty_exit),
                    )
                )
    pc_dir = os.path.join(state, "postcheck")
    pc_lines = read_text(os.path.join(pc_dir, "checks.txt"))
    if pc_lines:
        for line in pc_lines.splitlines():
            m = re.match(r"step(\d)_(hal|original)_(\S+)\.txt: exit (\d+),", line)
            if m:
                step, _side, t, rc = m.groups()
                checks.append(
                    check(
                        "postcheck",
                        STEP_NAMES[step],
                        t,
                        os.path.join(pc_dir, f"step{step}_{_side}_{t}.txt"),
                        expected.get(t),
                        int(rc),
                    )
                )
    s["checks"] = checks
    n_ok = sum(c["ok"] for c in checks)
    harness_done = bool(logs) and os.path.exists(os.path.join(logs, "SUMMARY.txt"))
    postcheck_done = any(c["stage"] == "postcheck" for c in checks)
    if s["config"]["dry_run"]:
        verdict = "DRY RUN (no agent)"
    elif not harness_done:
        verdict = "INCOMPLETE (harness did not finish)"
    elif agent.get("exit_code") not in (0, None):
        verdict = f"AGENT DID NOT FINISH (exit {agent['exit_code']})"
    elif n_ok != len(checks):
        verdict = "FAIL"
    elif postcheck_done:
        verdict = "PASS"
    else:
        verdict = "PASS (harness checks only)"
    s["correctness"] = dict(verdict=verdict, checks_ok=n_ok, checks_total=len(checks))

    if logs:
        cs = (read_json(os.path.join(logs, "codestats.json")) or {}).get("total") or {}
        raw = read_text(os.path.join(logs, "raw_check.txt")) or ""
        raw_part, _, nih_part = raw.partition("# NOT_IN_HAL marks")
        s["code"] = dict(
            loc=cs.get("loc"),
            hex_literals=cs.get("hex_literals"),
            numeric_shifts=cs.get("numeric_shifts"),
            hal_uses=cs.get("hal_uses"),
            raw_instruction_lines=(
                len(re.findall(r"TTI_|TT_OP_|cfg_reg_rmw", raw_part)) if raw else None
            ),
            not_in_hal_marks=(
                sum(1 for x in nih_part.splitlines() if "NOT_IN_HAL" in x)
                if raw
                else None
            ),
        )

        def total(path, pattern):
            m = re.search(pattern, last_line(path))
            return int(m.group(1)) if m else None

        probe = last_line(os.path.join(logs, "probe_check.txt"))
        m = re.search(r"(\d+) traces", probe)
        s["isolation"] = dict(
            forbidden_attempts=total(
                os.path.join(logs, "forbidden_access.txt"), r"total attempts: (\d+)"
            ),
            mcp_calls=total(
                os.path.join(logs, "mcp_calls.txt"), r"total MCP calls: (\d+)"
            ),
            probe_traces=int(m.group(1)) if m else None,
        )

    s["static"] = {}
    for mode in ("assert_off", "assert_on"):
        v = static_side(os.path.join(state, "static", mode))
        if v:
            s["static"][mode] = v
    s["perf"] = read_json(os.path.join(state, "perf", "perf_summary.json"))
    if s["perf"]:
        s["perf"].pop("per_variant_l1", None)
    dyn = read_text(os.path.join(state, "dyn_status.txt"))
    s["dyn_status"] = dyn.strip().splitlines() if dyn else None
    return s


def reference(series):
    ref = read_json(REFERENCE)
    if not ref or not series:
        return None
    runs = [r for r in ref["runs"].values() if r["series"] == series]
    if not runs:
        return None

    def med(k):
        vals = [r[k] for r in runs if r.get(k) is not None]
        return st.median(vals) if vals else None

    return dict(
        series=series,
        n=len(runs),
        names=sorted(k for k, r in ref["runs"].items() if r["series"] == series),
        text_bytes=med("text_bytes"),
        tti=med("tti"),
        tt=med("tt_all_paths"),
        cycles_l1_ratio=med("cycles_l1_ratio"),
        original=ref["original"],
    )


# ----------------------------------------------------------------------------------- render ---
def check_line(c):
    if not c["present"]:
        res = "no log"
    elif c["passed"] is None and c["failed"] is None:
        res = f"no pytest result line (exit {c['exit']})"
    else:
        res = f"{fmt(c['passed'])} passed, {c['failed']} failed, {c['skipped']} skipped"
        if c["errors"]:
            res += f", {c['errors']} errors"
    mark = "ok  " if c["ok"] else "FAIL"
    return f"  [{mark}] {c['step']:<38} {c['test']:<24} {res}"


def render_run(s):
    w = []
    cfg, ag = s["config"], s["agent"]
    sh = lambda x: (x or "-")[:8]  # noqa: E731
    w.append(f"==================== hal-llk-gen run '{s['name']}' ====================")
    w.append(
        f"Result:      {s['correctness']['verdict']}   ({s['correctness']['checks_ok']} of {s['correctness']['checks_total']} test checks ok)"
    )
    w.append(
        f"Profile:     {s['profile'] or '-'}   base {sh(cfg['base_commit'])}  HAL {sh(cfg['hal_commit'])}  docs {sh(cfg['docs_commit'])}  "
        f"prompt {cfg['prompt'] or '-'} ({sh(cfg['prompt_md5'])})"
    )
    w.append(
        f"Agent:       {cfg['model'] or '-'}, Claude Code {cfg['claude_version'] or '-'}, tests: {cfg['tests'] or '-'}"
    )
    if ag.get("exit_code") is not None:
        line = f"             exit {ag['exit_code']}, {fmt(ag['minutes'])} min"
        if ag.get("cost_usd") is not None:
            line += f", ${ag['cost_usd']:.2f}, {fmt(ag.get('turns'))} turns, {fmt(ag.get('pytest_commands'))} pytest commands"
        w.append(line)
        if ag.get("first_full_pass_iteration") is not None:
            w.append(
                f"             first full pass of the test selection: iteration {ag['first_full_pass_iteration']} "
                f"of {fmt(ag.get('iterations'))} ({fmt(ag['first_full_pass_min'])} min)"
            )
        elif ag.get("cost_usd") is not None:
            w.append(
                "             first full pass of the test selection: none recorded"
            )
        if ag.get("pytest_by_kind"):
            w.append(
                "             pytest commands by outcome: "
                + ", ".join(f"{k} {v}" for k, v in ag["pytest_by_kind"].items())
            )
    w.append("Test checks (pytest result lines):")
    w += [check_line(c) for c in s["checks"]] or ["  (none)"]
    if s.get("code") and any(v is not None for v in s["code"].values()):
        c = s["code"]
        w.append(
            f"Code:        {fmt(c['loc'])} LOC, {fmt(c['hex_literals'])} hex literals, {fmt(c['raw_instruction_lines'])} raw instruction lines, "
            f"{fmt(c['not_in_hal_marks'])} NOT_IN_HAL marks, {fmt(c['hal_uses'])} hal:: uses"
        )
    if s.get("isolation") and any(v is not None for v in s["isolation"].values()):
        i = s["isolation"]
        w.append(
            f"Isolation:   {fmt(i['forbidden_attempts'])} forbidden-path tool calls, {fmt(i['probe_traces'])} HAL probing traces, "
            f"{fmt(i['mcp_calls'])} MCP calls"
            + (
                f", {fmt(s['agent'].get('denied_tool_calls'))} denied tool calls"
                if s["agent"].get("denied_tool_calls") is not None
                else ""
            )
        )
    ref = reference(s.get("reference_series"))
    for mode, title in (
        ("assert_off", "LLK asserts off"),
        ("assert_on", "LLK asserts on (framework default)"),
    ):
        v = s["static"].get(mode)
        if not v:
            continue
        o, h = v["original"], v["hal"]
        src = (
            "tt_tti_counts"
            if v["source"] == "tt_tti_counts"
            else "codesize (no ELFRAY: no TT)"
        )
        w.append(
            f"Static, {title}, average per build over {h.get('builds') or '-'} builds, unpack+math+pack [{src}]:"
        )
        hdr = f"  {'':<28}{'original':>12}{'generated':>12}{'change':>10}"
        show_ref = ref and mode == "assert_off" and v["source"] == "tt_tti_counts"
        if show_ref:
            hdr += f"{'run ' + ref['series'] + ' median':>18}"
        w.append(hdr)
        rows = [
            (
                (
                    ".text bytes"
                    if v["source"] == "tt_tti_counts"
                    else "text bytes (size tool)"
                ),
                "text_bytes",
            )
        ]
        rows += [
            (
                "TTI" if v["source"] == "tt_tti_counts" else "Tensix instr. (objdump)",
                "tti",
            )
        ]
        if v["source"] == "tt_tti_counts":
            rows += [("TT (all paths)", "tt")]
        for label, k in rows:
            line = f"  {label:<28}{fmt(o[k]):>12}{fmt(h[k]):>12}{pct(h[k], o[k]):>10}"
            if show_ref:
                r = ref[k]
                line += f"{fmt(r) + ' (' + pct(r, ref['original'][{'tt': 'tt_all_paths'}.get(k, k)]) + ')':>18}"
            w.append(line)
        if v["source"] == "tt_tti_counts" and h.get("tt_unresolved"):
            w.append(
                f"  (TT: {fmt(h['tt_unresolved'])} stores per build not resolved statically, see static/{mode}/tt_tti_counts.md)"
            )
    p = s.get("perf")
    if p and p.get("l1"):
        r = p["l1"]
        line = (
            f"Cycles:      L1->L1 per tile, asserts off, {p.get('reps')} reps: original {r['orig']:.2f}, generated {r['new']:.2f}, "
            f"ratio {r['ratio_median']:.3f} ({r['ratio_min']:.3f}-{r['ratio_max']:.3f}), faster in {r['faster_variants']} of {r['n']} variants"
        )
        w.append(line)
        iso = [
            f"{lab} {p[k]['ratio_median']:.2f}"
            for k, lab in (("unp", "unpack"), ("math", "math"), ("pack", "pack"))
            if p.get(k)
        ]
        if iso:
            w.append(
                "             one thread alone (ratio): "
                + ", ".join(iso)
                + f"; repetition spread max {p.get('rep_spread_max_pct', 0):.2f}%"
            )
        if ref and ref.get("cycles_l1_ratio"):
            w.append(
                f"             run {ref['series']} series median ratio: {ref['cycles_l1_ratio']:.3f}"
            )
    if s.get("dyn_status"):
        w.append("Dynamic:     " + "; ".join(s["dyn_status"][-6:]))
    if s["stages"]:
        w.append(
            "Stages:      "
            + ", ".join(f"{x['stage']} {x['status']}" for x in s["stages"])
        )
    w.append(
        f"Files:       {s['state_dir']}  (summary.md, progress.log, stage folders)"
    )
    if s["run_dir"]:
        w.append(f"             harness run: {s['run_dir']}")
    if ref:
        w.append(
            f"Reference:   run {ref['series']} series = runs {', '.join(ref['names'])} (reference/run_series.json)"
        )
    w.append("=" * 70)
    return "\n".join(w)


def render_md(s):
    text = render_run(s)
    return f"# hal-llk-gen run {s['name']}\n\n```\n{text}\n```\n"


def compare_rows(summaries):
    rows = []
    for s in summaries:
        v = s["static"].get("assert_off") or {}
        h = v.get("hal") or {}
        p = (s.get("perf") or {}).get("l1") or {}
        ag = s["agent"]
        rows.append(
            dict(
                name=s["name"],
                profile=s["profile"],
                verdict=s["correctness"]["verdict"],
                checks=f"{s['correctness']['checks_ok']}/{s['correctness']['checks_total']}",
                minutes=ag.get("minutes"),
                cost=ag.get("cost_usd"),
                first_pass=ag.get("first_full_pass_iteration"),
                text=h.get("text_bytes"),
                tti=h.get("tti"),
                tt=h.get("tt"),
                cycles=p.get("ratio_median"),
                original=v.get("original"),
            )
        )
    return rows


def render_compare(summaries, markdown=False):
    rows = compare_rows(summaries)
    cols = [
        ("run", "name", "{}"),
        ("profile", "profile", "{}"),
        ("result", "verdict", "{}"),
        ("checks", "checks", "{}"),
        ("agent min", "minutes", "{:.1f}"),
        ("cost $", "cost", "{:.2f}"),
        ("1st full pass", "first_pass", "{}"),
        (".text B", "text", "{:.1f}"),
        ("TTI", "tti", "{:.1f}"),
        ("TT", "tt", "{:.1f}"),
        ("cycles x", "cycles", "{:.3f}"),
    ]

    def cell(r, key, f):
        v = r.get(key)
        return "-" if v is None else f.format(v)

    table = [[c[0] for c in cols]] + [[cell(r, k, f) for _, k, f in cols] for r in rows]

    def med(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return st.median(vals) if vals else None

    medrow = {
        k: med(k)
        for _, k, _ in cols
        if k not in ("name", "profile", "verdict", "checks")
    }
    medrow.update(name=f"median ({len(rows)})", profile="", verdict="", checks="")
    table.append([cell(medrow, k, f) for _, k, f in cols])
    orig = next((r["original"] for r in rows if r.get("original")), None)
    if orig:
        o = dict(
            name="original LLK",
            profile="",
            verdict="",
            checks="",
            text=orig.get("text_bytes"),
            tti=orig.get("tti"),
            tt=orig.get("tt"),
            cycles=1.0,
        )
        table.append([cell(o, k, f) for _, k, f in cols])
    ref = reference(
        next(
            (s.get("reference_series") for s in summaries if s.get("reference_series")),
            None,
        )
    )
    if ref:
        rr = dict(
            name=f"run {ref['series']} median",
            profile=f"{ref['n']} runs",
            verdict="",
            checks="",
            text=ref["text_bytes"],
            tti=ref["tti"],
            tt=ref["tt"],
            cycles=ref["cycles_l1_ratio"],
        )
        table.append([cell(rr, k, f) for _, k, f in cols])
    if markdown:
        out = ["| " + " | ".join(table[0]) + " |", "|" + "---|" * len(cols)]
        out += ["| " + " | ".join(r) + " |" for r in table[1:]]
    else:
        widths = [max(len(r[i]) for r in table) for i in range(len(cols))]
        out = [
            "  ".join(
                x.ljust(widths[i]) if i < 4 else x.rjust(widths[i])
                for i, x in enumerate(r)
            )
            for r in table
        ]
        out.insert(1, "-" * len(out[0]))
        out.insert(len(rows) + 2, "-" * len(out[0]))
    notes = [
        "",
        "Static counts: average per build, unpack+math+pack, LLK asserts off (static/assert_off). "
        "cycles x = perf L1->L1 cycles per tile, generated / original, median over variants.",
    ]
    return "\n".join(out + notes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("state")
    r.add_argument("--json")
    r.add_argument("--md")
    c = sub.add_parser("compare")
    c.add_argument("states", nargs="+")
    c.add_argument("--md")
    a = ap.parse_args()
    if a.cmd == "run":
        s = collect(a.state)
        print(render_run(s))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(s, f, indent=1)
        if a.md:
            with open(a.md, "w") as f:
                f.write(render_md(s))
        return 0 if s["correctness"]["verdict"].startswith("PASS") else 1
    summaries = [collect(x) for x in a.states]
    print(render_compare(summaries))
    if a.md:
        with open(a.md, "w") as f:
            f.write(
                "# hal-llk-gen runs\n\n"
                + render_compare(summaries, markdown=True)
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
