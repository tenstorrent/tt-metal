#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Grade a codegen op port: run the bands, apply the thresholds, return a verdict.

This is the tool the porting agent calls. It owns every pass/fail decision so that the agent's only
influence on the outcome is the C++ it writes -- it cannot choose the cases, the thresholds, or the
measurement method, and it cannot edit the tests, because `check_write_paths` re-derives the diff
from the base commit on every single invocation and refuses to measure a tree that has wandered
outside the port's own files. That check is the difference between a performance claim and a
performance assertion.

Thresholds are carried over from the pipeline this replaces
(`agentic_port/skills/verify/lib/constants.py`). The tie bands are not slop: op dispatch at this
scale is a few microseconds, so a strict ratio comparison would fail on host scheduling noise. Each
band is paired with an absolute escape so that a "loss" too small to matter cannot block a port,
while a relative guard keeps the escape from waving through a genuine regression on a tiny op.

The wall-clock band is judged in aggregate rather than case by case, which the per-case tie band
alone could not do. Run 31406186048 returned `back-to-translate` on six consecutive `verify` calls
with a *different* small subset of cases failing each time, ratios clustered at a median of ~0.99
and every device measurement passing: requiring ~24 independently noisy draws to each clear the tie
band is a coin flip, not a gate. So a case below the band is now recorded as marginal rather than
failing, and the port is refused when the noise stops being noise: one case slower than the noise
floor can explain, or marginal cases on more than a bounded fraction of the measurements. A port that
is genuinely slower is still refused, because a systematic loss lands every large case in the
marginal bucket at once. Device time, which is measured on the device and does not drift with host
scheduling, stays a strict per-case gate.

Aggregates are computed per stratum as well as globally, against the classes `strata.py` chose the
sample from. A global count is exactly what lets one slow class of inputs hide: a fifth of the
measurements marginal sits inside the global allowance and is also what a whole dtype being slower
looks like from far enough away. The coverage report says which classes were measured and how deeply,
because a ratio quoted without it is a claim about the sample rather than about the op.

Demotion is read off the measurements rather than declared. `is_demoted()` decides which in-scope
configurations `auto` hands back to native, and there is no way to know which those should be before
measuring -- the sweep says what codegen *can* serve and nothing says what it serves *well*. So the
wall band observes where `auto` actually goes, cases routed to native are reported and not graded,
and the ones that are graded and still fail come back as `demotion_candidates`: fix them or route
them away. `DEMOTION_CAP` is what stops the second option from being a way to launder a failure.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import scaffold  # noqa: E402 - must follow the path insertion above
import strata  # noqa: E402

# --- thresholds (agentic_port/skills/verify/lib/constants.py) --------------------------------
WALL_TIE_BAND = 0.98
WALL_TIE_ABS_US = 3.0
# Below this a wall-clock loss is larger than host scheduling noise explains, and one case is enough
# to refuse the port. Between this and WALL_TIE_BAND a case is marginal: tolerated individually,
# counted in aggregate.
WALL_REGRESSION_FLOOR = 0.90
WALL_MARGINAL_FRACTION = 0.25
# The marginal rule is applied per stratum as well as globally, but only where a stratum holds
# enough measurements for "most of this class is marginal" to mean anything. Two draws both landing
# under the tie band is the coin flip the aggregate rule was introduced to stop reading as signal,
# so below this a stratum contributes to the global count and is not judged on its own.
MIN_STRATUM_FOR_WALL_AGGREGATE = 3
# A demoted case is one `auto` hands back to native, so it is not a case this port serves and it is
# not graded. That exemption is what makes `is_demoted()` implementable at all -- a config the
# generated kernel is genuinely worse at has to be routable away rather than blocking the port -- and
# it is also the obvious way to launder a failure into a win. Past this fraction of the measured
# in-scope cases the port is serving a minority of its own declared scope, and buying a second
# implementation to do that is not worth the complexity.
DEMOTION_CAP = 0.5
DEVICE_VS_NATIVE_TIE_BAND = 1.0
DEVICE_VS_NATIVE_TIE_ABS_NS = 300.0
DEVICE_VS_NATIVE_TIE_ABS_BAND = 0.95
DEVICE_VS_GENERIC_TIE_BAND = 0.95
MIN_PAIRED_SAMPLES_FOR_CI = 5

WALL_PASS = "pass"
WALL_MARGINAL = "marginal"
WALL_REGRESSION = "regression"

VERDICT_WIN = "win"
VERDICT_BACK_TO_TRANSLATE = "back-to-translate"
VERDICT_NOT_A_CANDIDATE = "not-a-candidate"
VERDICT_BLOCKED = "blocked"


# --------------------------------------------------------------------------------------------
# Write-path guard
# --------------------------------------------------------------------------------------------


def allowed_prefixes(op: str, category: str) -> list[str]:
    base = f"ttnn/cpp/ttnn/operations/{category}"
    return [
        f"{base}/{op}/codegen/",
        f"{base}/{op}/{op}.cpp",
        f"{base}/{op}/{op}.hpp",
        f"{base}/{op}/{op}_nanobind.cpp",
        f"{base}/{op}/{op}_nanobind.hpp",
        f"{base}/sources.cmake",
        f"{base}/CMakeLists.txt",
        # The emitted routing test. Allowed because it has to ship with the port, and pinned by
        # `check_routing_test` so that allowing it does not hand the agent a test it can weaken. The
        # path comes from the emitter rather than being spelled again here, so the write the emitter
        # makes and the write this permits cannot disagree.
        scaffold.test_path(op, category),
    ]


def check_routing_test(manifest: str, op: str, category: str, repo: Path) -> str | None:
    """Re-render the routing test and report how the tree's copy differs, if it does.

    The write-path guard alone cannot protect this file, because the file has to be writable for the
    emitter to create it in the first place. So it is pinned by regeneration instead: the emitter is
    a pure function of the ledger, and the emitter itself lives outside the allowed prefixes, so an
    agent that edits either the test or the generator is caught -- the test by this check, the
    generator by the guard that runs before it.

    The copyright year is normalised out. It is the one part of the file that depends on the clock
    rather than the ledger, and a run spanning New Year is not a reason to refuse a port.
    """
    relative = scaffold.test_path(op, category)
    path = repo / relative
    if not path.is_file():
        return f"the routing test is missing at {relative}; the emitter never ran"

    # Local: building the ledger imports the generator's sweep module, which is only importable in
    # the built container, and gate.py is imported by its own tests outside one.
    import yaml

    import ledger

    cases = ledger.build_ledger(yaml.safe_load(Path(manifest).read_text()) or {})
    expected = scaffold.render_routing_test(op, category, cases)

    def normalise(text: str) -> str:
        return re.sub(r"(SPDX-FileCopyrightText: © )\d{4}", r"\g<1>YYYY", text)

    if normalise(path.read_text()) == normalise(expected):
        return None
    return (
        f"{relative} does not match what the emitter produces. It is generated "
        "from the coverage ledger and is not yours to edit; restore it by re-running "
        f"`python3 .github/scripts/port/scaffold.py --op {op} --emit-test-only "
        "--ttmetal-home /work --codegen-root /codegen`."
    )


def check_write_paths(base_sha: str, op: str, category: str, repo: Path) -> list[str]:
    """Return the changed paths that fall outside the port's own files.

    Recomputed from the base commit every call rather than trusted from a prior one, because the
    whole point is to catch a tree that changed since the last check.
    """
    out = subprocess.run(
        ["git", "diff", "--name-only", base_sha],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    changed = {p for p in (out + untracked).splitlines() if p.strip()}
    prefixes = allowed_prefixes(op, category)
    return sorted(p for p in changed if not any(p == a or p.startswith(a) for a in prefixes))


# --------------------------------------------------------------------------------------------
# Band execution
# --------------------------------------------------------------------------------------------


def run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)


def build_ledger(manifest: str, out: Path, repo: Path) -> dict:
    proc = run([sys.executable, str(SCRIPTS / "ledger.py"), "--manifest", manifest, "--out", str(out)], repo)
    if proc.returncode != 0:
        raise RuntimeError(f"ledger failed: {proc.stderr[-2000:]}")
    return json.loads(out.read_text())


def run_measure(
    op: str, manifest: str, band: str, out: Path, repo: Path, extra: list[str], env: dict | None = None
) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPTS / "measure.py"),
        "--op",
        op,
        # The manifest, not the ledger JSON written beside it: a case's kwargs can hold live ttnn
        # objects that JSON flattens to strings, so measure.py re-expands the sweep in process. See
        # the note at the top of its `main`.
        "--manifest",
        manifest,
        "--band",
        band,
        "--out",
        str(out),
        *extra,
    ]
    proc = run(cmd, repo, env)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(f"measure {band} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    return json.loads(out.read_text())


def run_device_band(
    op: str, manifest: str, out: Path, repo: Path, limit: int, reps: int, reports: Path, select: str
) -> dict:
    """Device timings need a tracy-enabled build, so this leg runs under `python3 -m tracy`, which
    sets TT_METAL_DEVICE_PROFILER and post-processes the device logs into an ops report.

    Usage is `python3 -m tracy [options] scriptfile [args]` with the script positional, and `-o`
    pins the artifacts folder so the CSV is found by path rather than by globbing a shared
    directory that may hold reports from an earlier iteration.
    """
    proc = run(
        [
            sys.executable,
            "-m",
            "tracy",
            "-p",
            "-r",
            "-o",
            str(reports),
            str(SCRIPTS / "measure.py"),
            "--op",
            op,
            "--manifest",
            manifest,
            "--band",
            "device",
            "--out",
            str(out),
            "--limit",
            str(limit),
            "--reps",
            str(reps),
            # Both perf bands must select the same cases: the wall band supplies the case list the
            # verdict is keyed on, and this band's profiler rows are attributed positionally against
            # its own dispatch order. The selection is deterministic, so passing the same strategy
            # and the same budget is all it takes -- and passing a different one would silently
            # grade two different samples against each other.
            "--select",
            select,
        ],
        repo,
        {"TRACY_NO_INVARIANT_CHECK": "1"},
    )
    if not out.is_file():
        raise RuntimeError(f"device band produced no output:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    return json.loads(out.read_text())


def latest_ops_csv(repo: Path, reports: Path) -> Path | None:
    for root in (reports, repo / "generated/profiler/reports"):
        matches = sorted(glob.glob(str(root / "**/ops_perf_results_*.csv"), recursive=True))
        if matches:
            return Path(max(matches, key=lambda p: Path(p).stat().st_mtime))
    return None


def attribute_device_rows(csv_path: Path, order: list[dict]) -> tuple[dict, list[str]]:
    """Join profiler rows to dispatches by order, and refuse to guess if they do not line up.

    Attribution is positional because signpost columns do not reliably survive post-processing. The
    safety net is the op-code consistency check: if one leg maps to more than one op code, the
    alignment is wrong and the numbers are discarded rather than reported.
    """
    notes: list[str] = []
    with csv_path.open() as fh:
        rows = [r for r in csv.DictReader(fh)]

    def col(row, *names):
        for n in names:
            for key in row:
                if key.strip().upper() == n:
                    return row[key]
        return None

    durations = []
    for row in rows:
        value = col(row, "DEVICE KERNEL DURATION [NS]")
        code = col(row, "OP CODE")
        if value not in (None, ""):
            try:
                durations.append((code, float(value)))
            except ValueError:
                continue

    dispatched = [o for o in order if o.get("error") is None and o.get("leg") != "setup"]
    if len(durations) != len(dispatched):
        notes.append(
            f"device attribution inconclusive: {len(durations)} profiler rows vs "
            f"{len(dispatched)} recorded dispatches"
        )
        return {}, notes

    samples: dict[tuple[str, str], list[float]] = {}
    codes: dict[str, set] = {}
    for (code, ns), entry in zip(durations, dispatched):
        leg = entry["leg"]
        if leg.endswith(":warmup"):
            continue  # consumed positionally, but a first call measures compilation, not the op
        samples.setdefault((entry["case_id"], leg), []).append(ns)
        codes.setdefault(leg, set()).add(code)

    for leg, seen in codes.items():
        if len(seen) > 1:
            notes.append(f"device attribution inconclusive: leg {leg!r} mapped to op codes {sorted(seen)}")
            return {}, notes
    if codes:
        notes.append("op codes: " + ", ".join(f"{leg}={sorted(s)[0]}" for leg, s in sorted(codes.items())))
    return samples, notes


# --------------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------------


def classify_wall(native_us: float, ported_us: float) -> str:
    """Bucket one case's wall-clock result as a pass, as noise, or as a real regression.

    A single case under the tie band says almost nothing on its own; `grade` decides what a
    collection of marginal cases means.
    """
    if ported_us <= 0:
        return WALL_REGRESSION
    ratio = native_us / ported_us
    if ratio >= WALL_TIE_BAND:
        return WALL_PASS
    if native_us > 0.0 and (ported_us - native_us) <= WALL_TIE_ABS_US:
        return WALL_PASS
    return WALL_MARGINAL if ratio >= WALL_REGRESSION_FLOOR else WALL_REGRESSION


def wall_aggregate(entries: list[dict]) -> dict:
    """Apply the marginal-allowance rule to a set of measurements.

    Extracted so the global check and the per-stratum checks are literally the same rule; two copies
    of a threshold policy is how they end up disagreeing.
    """
    measured = [c for c in entries if c.get("wall_class")]
    marginal = [c["case_id"] for c in measured if c["wall_class"] == WALL_MARGINAL]
    regressions = [c["case_id"] for c in measured if c["wall_class"] == WALL_REGRESSION]
    # At least one marginal case is always allowed: with two dozen draws, holding a port for a
    # single one is the coin flip this rule exists to remove.
    allowance = max(1, int(len(measured) * WALL_MARGINAL_FRACTION))
    return {
        "measured": len(measured),
        "marginal": marginal,
        "marginal_allowance": allowance,
        "regressions": regressions,
        "ok": not regressions and len(marginal) <= allowance,
    }


def device_vs_native_passes(native_ns: float, ported_ns: float) -> bool:
    if ported_ns <= 0:
        return False
    ratio = native_ns / ported_ns
    if ratio >= DEVICE_VS_NATIVE_TIE_BAND:
        return True
    # Absolute escape, fenced by a relative guard: a sub-300ns deficit is below the noise floor,
    # but only counts when the port is not also losing badly in proportional terms.
    return (ported_ns - native_ns) <= DEVICE_VS_NATIVE_TIE_ABS_NS and ratio >= DEVICE_VS_NATIVE_TIE_ABS_BAND


def coverage_report(selection: dict, configs: list[dict]) -> dict:
    """What the sample actually covered, per class of inputs.

    Derived from what was graded rather than from what was planned, so a case that errored out
    leaves its stratum reporting zero measurements instead of inheriting the plan's optimism. This
    is the table the pull request has to quote: a per-class ratio is the difference between "the
    port is faster" and "the port is faster on the cases we happened to look at".
    """
    planned = (selection or {}).get("strata") or {}
    graded: dict[str, list[dict]] = {}
    for entry in configs:
        if entry.get("stratum"):
            graded.setdefault(entry["stratum"], []).append(entry)

    rows = {}
    for key in sorted(set(planned) | set(graded)):
        entries = graded.get(key, [])
        walls = [c["wall_ratio"] for c in entries if c.get("wall_ratio")]
        devices = [c["device_vs_native"] for c in entries if c.get("device_vs_native")]
        rows[key] = {
            "cases_in_ledger": planned.get(key, {}).get("total"),
            "measured": len(entries),
            "wall_ratio_min": min(walls) if walls else None,
            "device_vs_native_min": min(devices) if devices else None,
        }

    unmeasured = [key for key, row in rows.items() if not row["measured"]]
    return {
        "select": (selection or {}).get("select"),
        "axes": (selection or {}).get("axes") or [],
        # Resolution the sample does not have. Not a gap in the classes that were tracked, but the
        # pull request has to say it out loud rather than implying the grid was covered.
        "axes_dropped": (selection or {}).get("axes_dropped") or [],
        "strata": rows,
        "strata_unmeasured": unmeasured,
        "complete": bool(rows) and not unmeasured,
    }


def grade(wall: dict, device_samples: dict, cases: list[dict], selection: dict | None = None) -> dict:
    by_id = {c["case_id"]: c for c in cases}
    selection = selection or {}
    axes = selection.get("axes") or []
    configs = []
    for record in wall.get("results", []):
        if record.get("error"):
            configs.append({"case_id": record["case_id"], "error": record["error"], "passes": False})
            continue
        case_id = record["case_id"]
        native_us, ported_us = record["native_us"], record["ported_us"]
        case = by_id.get(case_id, {})
        entry = {
            "case_id": case_id,
            "dtype": case.get("dtype"),
            "layout": case.get("layout"),
            # Named by the same function that chose the sample, so a stratum in this report and a
            # stratum in the coverage table are the same thing by construction.
            "stratum": strata.stratum_key(case, axes) if case else None,
            "wall_native_us": native_us,
            "wall_ported_us": ported_us,
            "wall_ratio": (native_us / ported_us) if ported_us else None,
        }

        native_ns = device_samples.get((case_id, "native"))
        ported_ns = device_samples.get((case_id, "ported"))
        generic_ns = device_samples.get((case_id, "generic"))
        if native_ns and ported_ns:
            entry["device_native_ns"] = min(native_ns)
            entry["device_ported_ns"] = min(ported_ns)
            entry["device_vs_native"] = min(native_ns) / min(ported_ns)
        if generic_ns and ported_ns:
            entry["device_generic_ns"] = min(generic_ns)
            entry["device_vs_generic_op"] = min(generic_ns) / min(ported_ns)

        entry["wall_class"] = classify_wall(native_us, ported_us)
        checks = {"wall": entry["wall_class"] != WALL_REGRESSION}
        if "device_vs_native" in entry:
            checks["device_vs_native"] = device_vs_native_passes(entry["device_native_ns"], entry["device_ported_ns"])
        if "device_vs_generic_op" in entry:
            checks["device_vs_generic_op"] = entry["device_vs_generic_op"] >= DEVICE_VS_GENERIC_TIE_BAND
        entry["checks"] = checks
        entry["passes"] = all(checks.values())
        # Measured with `implementation="codegen"` either way, so the numbers are real even for a
        # case `auto` will never send to codegen. That is deliberate: it is what makes an unnecessary
        # demotion visible instead of just absent.
        entry["routes_to"] = record.get("routes_to")
        entry["demoted"] = record.get("routes_to") == "native"
        configs.append(entry)

    # Only the cases `auto` actually sends to codegen are graded. The rest are reported in full and
    # judged by the cap below, because "this port is fast on everything it kept" is a claim that
    # means nothing without "and here is how much it kept".
    served = [c for c in configs if not c.get("demoted")]
    demoted = [c for c in configs if c.get("demoted")]
    demoted_fraction = len(demoted) / len(configs) if configs else 0.0

    ratios = [c["wall_ratio"] for c in served if c.get("wall_ratio")]
    overall = wall_aggregate(served)
    wall_median = statistics.median(ratios) if ratios else None
    # The median is reported but deliberately not a gate: with marginals capped, a port that is
    # systematically slower puts every large case in the marginal bucket and fails there, while a
    # median gate would instead break the small-op absolute escape, where a 15% ratio loss is two
    # microseconds and the project has already decided that does not matter.

    # The same rule again, within each class of inputs. A global count lets one slow class hide
    # behind the classes that pass: a fifth of the measurements marginal is inside the global
    # allowance, and is also what "every case of one dtype is slower" looks like from far enough
    # away. Small strata are exempt from the marginal half of the rule (see the constant), but a
    # regression below the noise floor still fails wherever it appears, because the global check
    # counts it too.
    per_stratum: dict[str, dict] = {}
    for key in sorted({c.get("stratum") for c in served if c.get("stratum")}):
        entry = wall_aggregate([c for c in served if c.get("stratum") == key])
        entry["judged"] = entry["measured"] >= MIN_STRATUM_FOR_WALL_AGGREGATE
        per_stratum[key] = entry
    failing_strata = [k for k, v in per_stratum.items() if v["judged"] and not v["ok"]]
    wall_aggregate_ok = overall["ok"] and not failing_strata

    # A win has to come from a case the port actually serves. A demoted case running faster under
    # forced codegen is evidence the demotion is unnecessary, not evidence the port is a win.
    strict_win = any((c.get("wall_ratio") or 0) > 1.0 or (c.get("device_vs_native") or 0) > 1.0 for c in served)
    all_pass = bool(served) and all(c.get("passes") for c in served) and wall_aggregate_ok
    if not configs:
        verdict = VERDICT_BLOCKED
    elif demoted_fraction > DEMOTION_CAP:
        # Routing away more than half of the measured scope. Whatever is left may well be faster,
        # but a second implementation serving a minority of its own declared surface is not worth
        # maintaining, and this is also what demoting-until-green looks like.
        verdict = VERDICT_NOT_A_CANDIDATE
    elif all_pass and strict_win:
        verdict = VERDICT_WIN
    elif all_pass:
        # Every gate held but nothing actually got faster; shipping this buys complexity for free.
        verdict = VERDICT_NOT_A_CANDIDATE
    else:
        verdict = VERDICT_BACK_TO_TRANSLATE

    device_ratios = [c["device_vs_native"] for c in served if c.get("device_vs_native")]
    return {
        "verdict": verdict,
        "has_strict_win": strict_win,
        "configs": configs,
        "failing": [c["case_id"] for c in served if not c.get("passes")],
        "coverage": coverage_report(selection, configs),
        "routing": {
            # What the port must either fix or route away. Derived from measurement rather than
            # asserted in advance, which is the only way to know it: the sweep's `invalidate_vector`
            # says what codegen *can* serve, and nothing in the manifest says what it serves *well*.
            # Per-case failures, plus the cases responsible for an aggregate failure. Without the
            # second half a port can be refused for a whole class being marginal while both `failing`
            # and this list come back empty, which tells the agent to fix something and does not say
            # what: a marginal case passes its own per-case checks by design.
            "demotion_candidates": [
                {
                    "case_id": c["case_id"],
                    "stratum": c.get("stratum"),
                    "wall_class": c.get("wall_class"),
                    "wall_ratio": c.get("wall_ratio"),
                    "device_vs_native": c.get("device_vs_native"),
                }
                for c in served
                if not c.get("passes")
                or (c.get("wall_class") != WALL_PASS and (c.get("stratum") in set(failing_strata) or not overall["ok"]))
            ],
            "demoted": [c["case_id"] for c in demoted],
            "demoted_fraction": round(demoted_fraction, 3),
            "demotion_cap": DEMOTION_CAP,
            # Demoted and yet unambiguously fine under forced codegen: the demotion is costing
            # performance for no reason. Not a failure, but the port should not route these away.
            #
            # "Unambiguously" is carrying weight. Requiring only that one metric beat native flags a
            # case that is slower on wall clock and faster on device -- which is a normal shape for a
            # case worth demoting, so the loose version calls every sensible demotion a mistake and
            # the field becomes noise. A case qualifies only if it cleared every gate with no
            # marginal caveat and still came out ahead.
            "demoted_but_faster": [
                c["case_id"]
                for c in demoted
                if c.get("passes")
                and c.get("wall_class") == WALL_PASS
                and ((c.get("wall_ratio") or 0) > 1.0 or (c.get("device_vs_native") or 0) > 1.0)
            ],
            # Absent when the build cannot answer the program-cache query, in which case nothing was
            # exempted and the gate is simply stricter than it needs to be.
            "probe": "program_cache" if any(c.get("routes_to") for c in configs) else "unavailable",
        },
        "summary": {
            # Device time is worst-case, because it is stable enough for the weakest configuration to
            # mean something. Wall clock is reported as a distribution, because it is not.
            "wall_ratio_min": min(ratios) if ratios else None,
            "wall_ratio_median": wall_median,
            "wall_marginal": overall["marginal"],
            "wall_marginal_allowance": overall["marginal_allowance"],
            "wall_regressions": overall["regressions"],
            "wall_aggregate_ok": wall_aggregate_ok,
            "wall_failing_strata": failing_strata,
            "wall_by_stratum": per_stratum,
            "device_vs_native_min": min(device_ratios) if device_ratios else None,
            "cases_measured": len(configs),
            "cases_graded": len(served),
        },
    }


def grade_correctness(results: list[dict]) -> dict:
    failures = [r for r in results if not r.get("equal") or r.get("error")]
    out_of_scope = [r for r in results if r.get("scope") == "out"]
    routing = [r for r in out_of_scope if r.get("routing_ok") is False]
    # `routing_ok` is None when the build could not answer the program-cache query. Counted, because
    # the routing check reading as a pass while it verified nothing is the bug just fixed in
    # measure.py and the one worth being loud about.
    unverified = [r["case_id"] for r in out_of_scope if r.get("routing_ok") is None]
    return {
        "total": len(results),
        "failures": [
            {
                "case_id": r["case_id"],
                "scope": r.get("scope"),
                "error": r.get("error"),
                "max_abs_diff": r.get("max_abs_diff"),
            }
            for r in failures[:25]
        ],
        "failure_count": len(failures),
        "routing_violations": [r["case_id"] for r in routing],
        "routing_unverified": unverified[:25],
        "routing_unverified_count": len(unverified),
        "passes": not failures and not routing and not unverified,
    }


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--band", default="both", choices=["correctness", "performance", "both"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--category", default="data_movement")
    ap.add_argument("--base-sha", default=os.environ.get("PORT_BASE_SHA", ""))
    ap.add_argument("--work", default="/tmp/port-gate")
    ap.add_argument("--limit", type=int, default=24, help="cases per perf band")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument(
        "--select",
        default="stratified",
        choices=["stratified", "prefix"],
        help="how the perf bands spend their case budget; `prefix` is the old flat slice",
    )
    ap.add_argument("--skip-write-check", action="store_true", help="baseline runs, before any edits")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    report: dict = {"op": args.op, "band": args.band, "notes": []}

    if not args.skip_write_check:
        if not args.base_sha:
            print(
                json.dumps(
                    {"verdict": VERDICT_BLOCKED, "error": "no --base-sha; refusing to measure an unverifiable tree"},
                    indent=2,
                )
            )
            return 2
        stray = check_write_paths(args.base_sha, args.op, args.category, repo)
        if stray:
            print(
                json.dumps(
                    {
                        "verdict": VERDICT_BLOCKED,
                        "error": "changes outside the port's own files; the harness and build scripts are off limits",
                        "unexpected_changes": stray,
                        "allowed": allowed_prefixes(args.op, args.category),
                    },
                    indent=2,
                )
            )
            return 2

        # After the guard, deliberately: the guard is what establishes that the emitter itself is
        # unmodified, which is what makes re-rendering a meaningful comparison.
        try:
            drift = check_routing_test(args.manifest, args.op, args.category, repo)
        except Exception as exc:  # noqa: BLE001
            drift = f"could not verify the routing test: {type(exc).__name__}: {exc}"
        if drift:
            print(json.dumps({"verdict": VERDICT_BLOCKED, "error": drift}, indent=2))
            return 2

    try:
        ledger_path = work / f"{args.op}_ledger.json"
        ledger = build_ledger(args.manifest, ledger_path, repo)
        report["ledger_counts"] = ledger["counts"]

        if args.band in ("correctness", "both"):
            out = work / f"{args.op}_correctness.json"
            data = run_measure(args.op, args.manifest, "correctness", out, repo, [])
            report["correctness"] = grade_correctness(data["results"])
            report["correctness"]["golden"] = data.get("golden")

        if args.band in ("performance", "both"):
            wall = run_measure(
                args.op,
                args.manifest,
                "wall",
                work / f"{args.op}_wall.json",
                repo,
                ["--limit", str(args.limit), "--iters", str(args.iters), "--select", args.select],
            )
            device_samples: dict = {}
            try:
                reports = work / f"{args.op}_profiler"
                dev = run_device_band(
                    args.op,
                    args.manifest,
                    work / f"{args.op}_device.json",
                    repo,
                    args.limit,
                    args.reps,
                    reports,
                    args.select,
                )
                csv_path = latest_ops_csv(repo, reports)
                if csv_path is None:
                    report["notes"].append("no ops_perf_results CSV found; device band skipped")
                else:
                    device_samples, notes = attribute_device_rows(csv_path, dev["order"])
                    report["notes"].extend(notes)
            except Exception as exc:  # noqa: BLE001 - wall alone still yields a usable verdict
                report["notes"].append(f"device band unavailable: {exc}")
            report["performance"] = grade(wall, device_samples, ledger["cases"], wall.get("selection"))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"verdict": VERDICT_BLOCKED, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2

    correctness_ok = report.get("correctness", {}).get("passes", True)
    perf = report.get("performance", {})
    if not correctness_ok:
        report["verdict"] = VERDICT_BACK_TO_TRANSLATE
    else:
        report["verdict"] = perf.get("verdict", VERDICT_WIN if args.band == "correctness" else VERDICT_BLOCKED)

    Path(work / f"{args.op}_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["verdict"] == VERDICT_WIN else 1


if __name__ == "__main__":
    sys.exit(main())
