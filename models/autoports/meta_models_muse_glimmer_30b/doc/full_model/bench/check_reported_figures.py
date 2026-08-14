# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check that the figures quoted in README.md come from a committed run.

The optimized-multichip stage's review found the same evidence defect six rounds
running: a number quoted from a superseded run, or from no run at all.  This is the
same guard for this stage, and it is deliberately not vacuous -- every pattern it
knows about has to *resolve* to a value in an evidence artifact, and a pattern that
matches nothing in the README is itself an error (the README changed and this file
did not).

Checked figures and their sources:

============================================  ==========================================
README figure                                 source
============================================  ==========================================
TTFT / token-out decode / logits-only decode  doc/full_model/evidence_perf.json
sampling-trace ms/token                       doc/full_model/evidence_perf.json
prefill top-1/top-5/top-100                   doc/full_model/evidence_accuracy.json
teacher-forcing top-1/top-5/top-100           doc/full_model/evidence_accuracy.json
per-device DRAM bytes                         doc/full_model/evidence_accuracy.json
LM-head sweep ms                              doc/full_model/logs/lm_head_sweep.log
============================================  ==========================================

Usage::

    python doc/full_model/bench/check_reported_figures.py
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time

#: The share above which the README's Tracy table must list an op. Stated in the
#: README next to the table; kept here so the gate and the prose cannot drift apart.
TRACY_TABLE_THRESHOLD = 3.0

ROOT = pathlib.Path(__file__).resolve().parents[3]
DOC = ROOT / "doc/full_model"
README = DOC / "README.md"


def load(name: str) -> dict:
    path = DOC / name
    if not path.is_file():
        raise SystemExit(f"missing {path}; the README cannot be checked against a run that does not exist")
    return json.loads(path.read_text())


def close(quoted: float, actual: float, *, tolerance: float = 0.005) -> bool:
    """Quoted figures are rounded; allow the rounding, not a different run."""
    if actual == 0:
        return abs(quoted) < 1e-9
    return abs(quoted - actual) / abs(actual) <= tolerance


def main() -> int:
    text = README.read_text()
    accuracy = load("evidence_accuracy.json")
    perf_path = DOC / "evidence_perf.json"
    perf = json.loads(perf_path.read_text())["performance"] if perf_path.is_file() else None

    failures: list[str] = []
    checked = 0
    #: Stage-owned implementation files, filled in by the provenance check below and
    #: reused by the artifact-ordering check. Empty means the provenance check did not
    #: run, in which case ordering is checked for exact mtimes only.
    owned: list[pathlib.Path] = []

    def check(label: str, pattern: str, actual: float, *, tolerance: float = 0.005) -> None:
        """Check **every** occurrence, not just the first.

        The README quotes its headline figures twice -- once in the Result table and
        once in the Performance section -- and an earlier version of this function
        used ``re.search``, so updating the first and leaving the second stale passed
        the check. That happened: after the embedding-gather fix the Performance table
        still carried the pre-fix TTFT and token-out numbers while this script
        reported "figures OK".
        """
        nonlocal checked
        matches = re.findall(pattern, text)
        if not matches:
            failures.append(f"{label}: README has no figure matching {pattern!r}")
            return
        for index, quoted_text in enumerate(matches):
            quoted = float(quoted_text if isinstance(quoted_text, str) else quoted_text[0])
            checked += 1
            if not close(quoted, actual, tolerance=tolerance):
                where = "" if len(matches) == 1 else f" (occurrence {index + 1} of {len(matches)})"
                failures.append(f"{label}{where}: README says {quoted}, the run says {actual}")

    capacity = accuracy["capacity"]
    check(
        "per-device long-lived DRAM (GB)",
        r"\*\*([0-9.]+) GB/device\*\* of long-lived",
        capacity["per_device_total_bytes"] / 1e9,
    )
    check(
        "per-device KV cache at full context (GB)",
        r"KV cache \(52 layers, 131072 tokens\) \| ([0-9.]+) GB",
        capacity["per_device_kv_cache_bytes"] / 1e9,
    )

    # The accuracy table names its own source for every row -- gate, reference, the
    # three metrics and the evidence file -- so each row is resolved against exactly
    # the run it claims, rather than against whichever run a positional regex hits
    # first.  A row naming a file or a reference that does not exist is a failure.
    row_pattern = re.compile(
        r"\| (prefill|decode) \(`(run_prefill_check|run_teacher_forcing)`\) \| (bf16|fp32 control) \| "
        r"\**([0-9.]+)\** \| \**([0-9.]+)\** \| \**([0-9.]+)\** \| `([^`]+)`"
    )
    # Resolved by name from the doc directory rather than from a fixed list, so that
    # adding an evidence file to the README does not silently skip its rows -- only a
    # name with no file behind it is a failure.
    sources: dict[str, dict] = {"evidence_accuracy.json": accuracy}
    perf_json = json.loads(perf_path.read_text()) if perf_path.is_file() else None
    if perf_json is not None:
        sources["evidence_perf.json"] = perf_json

    def source_for(name: str) -> dict | None:
        if name not in sources:
            path = DOC / name
            if not path.is_file():
                return None
            sources[name] = json.loads(path.read_text())
        return sources[name]

    reference_files = {
        "bf16": "readiness_aime24_chat.refpt",
        "fp32 control": "readiness_aime24_chat_fp32.refpt",
    }
    accuracy_rows = row_pattern.findall(text)
    if not accuracy_rows:
        failures.append("the README accuracy table has no rows this checker recognises")
    for gate, runner, reference, top1, top5, top100, source_name in accuracy_rows:
        source = source_for(source_name)
        if source is None:
            failures.append(f"{gate}/{reference}: README cites {source_name}, which does not exist")
            continue
        key = "prefill_check" if runner == "run_prefill_check" else "teacher_forcing"
        rows = source.get(f"{key}_by_reference")
        if rows:
            entry = rows.get(reference_files[reference])
            if entry is None:
                failures.append(f"{gate}/{reference}: {source_name} has no rows for {reference_files[reference]}")
                continue
            stats = entry["per_entry"][0]
        elif reference == "bf16" and source.get(key, {}).get("per_entry"):
            stats = source[key]["per_entry"][0]
        else:
            failures.append(f"{gate}/{reference}: {source_name} has no {key} rows")
            continue
        for label, quoted, actual in (
            ("top-1", top1, stats["top1"]),
            ("top-5", top5, stats["top5"]),
            (f"top-{stats['k']}", top100, stats["top100"]),
        ):
            checked += 1
            if not close(float(quoted), actual, tolerance=0.001):
                failures.append(f"{gate}/{reference} {label} in {source_name}: README says {quoted}, run says {actual}")

    if perf:
        check("TTFT (ms)", r"\*\*([0-9.]+) ms\*\* TTFT", perf["ttft_ms"]["min"], tolerance=0.02)
        check(
            "token-out decode (t/s/u)",
            r"\*\*([0-9.]+) t/s/u\*\* token-out",
            perf["token_out_decode_tok_s_u"],
            tolerance=0.02,
        )
        # The **ms/token** side of the headline pair, not just the t/s/u reciprocals. A
        # round-18 mutation sweep showed 104 of 151 decimals in the body could be scaled
        # by 37 % with the gate still passing, and these two were the most load-bearing
        # of them -- every "the trace lands under the floor" argument is stated in ms.
        check(
            "token-out decode (ms/token)",
            r"\| \*\*token-out decode\*\* \| ([0-9.]+) ms/token \|",
            perf["token_out_decode_ms_per_token"]["min"],
        )
        check(
            "traced logits-only decode (ms/token)",
            r"\| \*\*traced logits-only decode\*\* \| ([0-9.]+) ms/token \|",
            perf["traced_decode_logits_only_ms_per_token"]["min"],
        )
        check(
            "logits-only decode (t/s/u)",
            r"\*\*([0-9.]+) t/s/u\*\* traced logits-only",
            perf["traced_decode_logits_only_tok_s_u"],
            tolerance=0.02,
        )

    # The LM-head winner has to be the row the sweep log actually won with.
    sweep = DOC / "logs/lm_head_sweep.log"
    if sweep.is_file():
        rows = re.findall(r"PROBE lm_head (\S+) (\S+) .*?ms=([0-9.]+) pcc=([0-9.]+)", sweep.read_text())
        if not rows:
            failures.append("lm_head_sweep.log has no PROBE rows")
        else:
            best = min(rows, key=lambda row: float(row[2]))
            checked += 1
            if f"{float(best[2]):.4f}" not in text:
                failures.append(f"README does not quote the sweep's winning time {best[2]} " f"({best[0]} {best[1]})")
            if best[0] not in text or best[1].replace("_b", "") not in text.lower().replace("bfp", "bfloat"):
                pass  # the contract/dtype names are prose; the time is the anchor

        # The core-count range is quoted under a *dtype-scoped* claim ("worth nothing at
        # BFP4"), and for several rounds its upper endpoint was a BFP8 row -- a 3.2 %
        # apparent spread where the real BFP4 spread is 0.4 %. The winner check above
        # could not catch it, because it only resolves the sweep's minimum. Both bands
        # are derived here, per dtype, so a precision-scoped range cannot borrow an
        # endpoint from the other precision again.
        bands: dict[str, list[float]] = {}
        for dtype, ms in re.findall(
            r"PROBE lm_head dram_sharded (\S+) cores=\d+ in0=1 ms=([0-9.]+)", sweep.read_text()
        ):
            bands.setdefault(dtype, []).append(float(ms))
        # The corpus is the **markdown and the source**, not the markdown alone. Round 11
        # corrected this band "in both records" and a third copy survived in
        # ``tt/model.py``'s module docstring for four more rounds, because every value
        # check stopped at README.md while only the retracted-*string* sweep walked
        # ``tt/``. A wrong measurement in the file a later stage reads first is worse
        # than a wrong one in the prose, not better.
        corpus = [("README.md", text)]
        for extra in (DOC / "work_log.md", ROOT / "tt/model.py", ROOT / "tt/generator.py"):
            if extra.is_file():
                corpus.append((extra.name, extra.read_text()))
        told = ("superseded", "withdraw", "retract", "history", "wrong", "quoted as", "borrow", "corrected")
        for document, body in corpus:
            flat = body.replace("–", "-").replace("—", "-")
            for dtype, values in sorted(bands.items()):
                if len(values) < 2:
                    continue
                span = f"{min(values):.4f}-{max(values):.4f}"
                other = [f"{v:.4f}" for name, vs in bands.items() if name != dtype for v in (min(vs), max(vs))]
                # A file only has to quote the band if it quotes a band at all; every file
                # that quotes one has to get it right, and none may cross the dtypes.
                quotes_a_band = any(
                    f"{value:.4f}-" in flat or f"-{value:.4f}" in flat
                    for group in bands.values()
                    for value in (min(group), max(group))
                )
                if not quotes_a_band:
                    continue
                checked += 1
                if span not in flat:
                    failures.append(
                        f"{document} quotes an LM-head core-count band but not the {dtype} "
                        f"in0_block_w=1 range {span} (derived from {len(values)} rows of lm_head_sweep.log)"
                    )
                for endpoint in other:
                    for pair in (f"{min(values):.4f}-{endpoint}", f"{endpoint}-{max(values):.4f}"):
                        hits = [line for line in flat.splitlines() if pair in line]
                        if hits and not all(any(word in line.lower() for word in told) for line in hits):
                            failures.append(
                                f"{document} pairs a {dtype} endpoint with {endpoint}, which is not a {dtype} row"
                            )

    # Two pytest figures the README quotes as *claims about the test suite* rather than
    # about the model: the watcher subset and the reverse-order independence run. Both
    # were hand-maintained, and a review round found the reverse-order figure quoting a
    # number that a forward-order run had produced. Each is now derived from the console
    # the README names, so the count, the time and the log all have to agree.
    for label, log_name, pattern in (
        # The console name is *read out of the README* rather than hard-coded: it was
        # hard-coded once, the rebuild renamed the console, and the check went on
        # resolving the superseded one -- passing only because its tolerance was wider
        # than the difference between the two runs.
        (
            "watcher subset",
            "logs/"
            + (sorted(set(re.findall(r"`logs/(watcher_run_final\d+\.log)`", text)))[-1:] or ["watcher_run.log"])[0],
            r"\*\*(\d+) passed in ([0-9.]+) s, and the watcher log is clean\*\*",
        ),
        (
            "reverse-order suite",
            "logs/reverse_order_run.log",
            r"\*\*(\d+) passed in ([0-9.]+) s\*\*, in reverse order",
        ),
    ):
        console = DOC / log_name
        quoted = re.search(pattern, text)
        if quoted is None:
            failures.append(f"{label}: README has no figure matching {pattern!r}")
            continue
        if not console.is_file():
            failures.append(f"{label}: README quotes a figure whose console {log_name} does not exist")
            continue
        run = re.search(r"(\d+) passed[^\n]*? in ([0-9.]+)s", console.read_text(errors="ignore"))
        if run is None:
            failures.append(f"{label}: {log_name} has no pytest summary line to resolve the figure against")
            continue
        checked += 2
        if int(quoted.group(1)) != int(run.group(1)):
            failures.append(f"{label}: README says {quoted.group(1)} passed, {log_name} says {run.group(1)}")
        # Matched at the precision it is quoted to, not at 1 %: a 1 % window on a 230 s
        # run is +/-2.3 s, which is wider than the difference between two *different*
        # runs -- so the check could not detect the very defect it was added for, and a
        # mutation test proved it (230.19 -> 230.58 passed silently).
        said = quoted.group(2)
        if float(said) != round(float(run.group(2)), len(said.partition(".")[2])):
            failures.append(f"{label}: README says {said} s, {log_name} says {run.group(2)} s")

    # The sampling trace is the figure the whole sampler investigation turns on, and it
    # was listed in this script's docstring without ever being checked. It is checked
    # here in both forms the README quotes, so a regression of
    # ``topk_split_to_power_of_2`` back to the single-core path -- which the 46-case test
    # suite passes either way -- cannot pass this gate silently.
    if perf and "sampling_trace_ms_per_token" in perf:
        sampling_ms = perf["sampling_trace_ms_per_token"]["min"]
        check("sampling trace (ms/token)", r"sampling trace alone \| ([0-9.]+) ms/token", sampling_ms)
        check(
            "sampling trace share of token-out",
            r"([0-9.]+) % of the token-out step",
            100.0 * sampling_ms / perf["token_out_decode_ms_per_token"]["min"],
            tolerance=0.05,
        )

    # The two terminal-path PCCs are quoted from a console log rather than from JSON,
    # which is exactly how they went stale once: the README claimed PCC 1.000000000 and
    # 0.999987 while logs/terminal_probe.log held only its first line, because the probe
    # had been re-run into a truncated file. So they are resolved against the log's
    # actual contents rather than trusted.
    probe_log = DOC / "logs/terminal_probe.log"
    for label, quoted in (
        ("fractured-embedding PCC", "1.000000000"),
        ("terminal-norm PCC", "0.999987"),
    ):
        checked += 1
        if quoted not in text:
            failures.append(f"{label}: README no longer quotes {quoted}")
        elif not probe_log.is_file():
            failures.append(f"{label}: README quotes {quoted} but {probe_log.name} is missing")
        elif quoted not in probe_log.read_text():
            failures.append(f"{label}: README quotes {quoted}, which {probe_log.name} does not contain")

    # ``work_log.md`` is a required deliverable too, and it went stale independently:
    # it published a shipped token-out of 23.94 ms/token against the run's 23.786. It is
    # not fully pattern-checked here -- it is a narrative and quotes historical values on
    # purpose -- but the figures it presents as **shipped** are.
    #
    # These were checked at 2 % once, which is wider than the whole process-to-process
    # decode spread (0.08 %), so a *previous pass's* 23.820 sat here passing the gate
    # while the README said 23.800. They are matched at the precision they are quoted to
    # instead: round the run's value to the quoted number of decimals and require
    # equality, which is what the derived tables already do.
    work_log = DOC / "work_log.md"
    if work_log.is_file() and perf:
        log_text = work_log.read_text()
        for label, pattern, actual in (
            # ``\s+`` rather than a literal space: the work log is hard-wrapped, so these
            # figures routinely straddle a line break.
            (
                "work log token-out (ms/token)",
                r"token-out \*\*[0-9.]+\s*->\s*([0-9.]+) ms/token\*\*",
                perf["token_out_decode_ms_per_token"]["min"],
            ),
            (
                "work log token-out (t/s/u)",
                r"30\.41\s*->\s*\*\*([0-9.]+) t/s/u\*\*",
                perf["token_out_decode_tok_s_u"],
            ),
            (
                "work log sampling trace (ms)",
                r"Sampling trace \*\*[0-9.]+\s*->\s*([0-9.]+) ms\*\*",
                perf["sampling_trace_ms_per_token"]["min"],
            ),
            # The work log's TTFT figures were the ungated ones: §14 quotes the split's
            # before/after and §18 the re-measurement delta, and both name the *shipped*
            # value as the right-hand side. TTFT is the figure that moves most between
            # processes, so an ungated one here is a stale figure waiting to happen.
            (
                "work log TTFT (ms)",
                r"TTFT (?:[0-9.]+\s*->\s*)?\*\*([0-9.]+) ms\*\*",
                perf["ttft_ms"]["min"],
            ),
            (
                "work log TTFT delta (ms)",
                r"TTFT by [0-9.]+ %\s*\([0-9.]+\s*->\s*([0-9.]+)\s*ms\)",
                perf["ttft_ms"]["min"],
            ),
        ):
            found = re.findall(pattern, log_text)
            if not found:
                failures.append(f"{label}: work_log.md has no figure matching {pattern!r}")
                continue
            for quoted in found:
                checked += 1
                decimals = len(quoted.partition(".")[2])
                if float(quoted) != round(actual, decimals):
                    failures.append(
                        f"{label}: work_log says {quoted}, the run says {actual} "
                        f"({round(actual, decimals)} at the quoted precision)"
                    )

    # The topk-geometry and Tracy composition tables were both hand-maintained against
    # artifacts whose 3rd and 4th decimals move run to run, so every re-measurement
    # required a manual requote and three review rounds caught one that was missed.
    # They are **derived** here instead: the README must quote what the artifact says,
    # to the precision the README uses, and a re-run that moves a number fails this gate
    # rather than silently disagreeing with the table.
    geometry_path = DOC / "topk_geometry_probe.json"
    if geometry_path.is_file():
        geometry = {int(r["width"]): r for r in json.loads(geometry_path.read_text()) if r.get("calls") == 1}
        for width, pattern in (
            (50688, r"\| 50688 \(shipped shard\) \| \*\*([0-9.]+)\*\* \|"),
            (65536, r"\| 65536 \(padded shard\) \| ([0-9.]+) \|"),
            (32768, r"\| 32768 \| ([0-9.]+) \|"),
            (8192, r"\| 8192 \| ([0-9.]+) \|"),
            (4096, r"\| 4096 \| ([0-9.]+) \|"),
        ):
            if width in geometry:
                check(f"topk geometry {width} (ms)", pattern, geometry[width]["ms"], tolerance=0.02)

    tracy_path = DOC / "tracy/sampling_perf_report_stacked.csv"
    if tracy_path.is_file():
        rows = {}
        for line in tracy_path.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) > 3:
                rows[parts[1].split(" (")[0]] = (float(parts[0]), float(parts[2]), parts[3])
        # The **row set** is derived, not just the values. The README states the table
        # shows every op above TRACY_TABLE_THRESHOLD % of the trace, and an earlier
        # version quietly omitted one that qualified (Typecast at 4.50 %, larger than a
        # row it did list) -- which a fixed op list could never catch, because it only
        # ever checked the rows someone had already chosen to write down.
        for op, (share, _total, _calls) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            if share < TRACY_TABLE_THRESHOLD:
                continue
            pattern = rf"\| `{op}` \| ([0-9.]+) % \|"
            microseconds = rf"\| `{op}` \| [0-9.]+ % \| ([0-9.]+) us \|"
            found_us = re.search(microseconds, text)
            if found_us is not None:
                checked += 1
                said = found_us.group(1)
                if float(said) != round(_total, len(said.partition(".")[2])):
                    failures.append(f"tracy {op} (us): README says {said}, the run says {_total}")
            if not re.search(pattern, text):
                failures.append(
                    f"tracy {op} is {share} % of the sampling trace, above the {TRACY_TABLE_THRESHOLD} % "
                    "the README says the table lists, but it has no row"
                )
                continue
            check(f"tracy {op} (%)", pattern, share, tolerance=0.02)

    # Provenance: the artifacts table names one console as the source of the headline
    # figures, and that row was wrong in three consecutive review rounds -- each time
    # naming a log that predated the last implementation edit, i.e. a build that was not
    # shipped. Values are gated everywhere else in this file; provenance was not, which
    # is exactly why it kept drifting. It is gated here.
    # ``.gz`` is accepted because the consoles are ~850 KB each and the repo's
    # ``check-large-files`` pre-commit hook rejects anything over 500 KB, so every
    # oversized log is committed gzipped (``gzip`` preserves the mtime, which is what
    # the ordering claim below rests on).
    named = re.search(r"\| `logs/(final\d+\.log(?:\.gz)?)` \| the console of the final pass", text)
    if named is None:
        failures.append("the artifacts table names no console as the source of the headline figures")
    else:
        console = DOC / "logs" / named.group(1)
        checked += 1
        if not console.is_file():
            failures.append(f"the artifacts table names {named.group(1)}, which does not exist")
        else:
            # Only files this stage actually *changed* count as its implementation.
            # An mtime-only sweep blamed the wrong file once already: a pre-commit
            # hook touched `tests/test_multichip_decoder.py`, whose content is
            # identical to HEAD, and the gate reported it as the edit that
            # invalidated the console. A tracked file that `git diff` says is
            # unmodified is, by definition, not part of this stage's build.
            repo = ROOT.parents[2]
            candidates = [
                path
                for root in (
                    ROOT / "tt",
                    ROOT / "tests",
                    ROOT.parents[1] / "common/sampling",
                    ROOT.parents[1] / "common/readiness_check",
                )
                for path in root.rglob("*.py")
                if "__pycache__" not in str(path)
            ]
            # "Stage-owned" means *changed by this stage*, which is a diff against the
            # commit the stage started from -- not against HEAD. Comparing to HEAD works
            # only while the stage is uncommitted; the moment it is committed, every file
            # matches HEAD and the set goes empty. The base SHA is read from the work
            # log's own "Starting point" line so it is stated once, not twice.
            base = None
            work_log_path = DOC / "work_log.md"
            if work_log_path.is_file():
                found_base = re.search(r"Starting point: [^`]*`([0-9a-f]{7,40})`", work_log_path.read_text())
                if found_base:
                    base = found_base.group(1)
            unmodified = set()
            if candidates:
                rels = [str(path.relative_to(repo)) for path in candidates]
                tracked = subprocess.run(
                    ["git", "ls-files", "--", *rels], cwd=repo, capture_output=True, text=True
                ).stdout.split()
                if base is None:
                    failures.append("work_log.md states no starting-point commit, so stage ownership cannot be derived")
                    changed = tracked  # fail open rather than silently empty
                else:
                    changed = subprocess.run(
                        ["git", "diff", "--name-only", base, "--", *rels],
                        cwd=repo,
                        capture_output=True,
                        text=True,
                    ).stdout.split()
                unmodified = {repo / name for name in set(tracked) - set(changed)}
            owned[:] = [path for path in candidates if path not in unmodified]

            # Files whose bytes still match the hashes the record publishes ARE the
            # measured build, whatever their mtime says. The pre-commit hooks stash and
            # restore unstaged files at commit time -- content preserved, mtime rewritten
            # -- so after the first commit the mtime proxy reported implementation
            # "edits" that never happened. The hash is the direct statement; the mtime
            # comparison below only applies to files the record does not pin.
            pinned = dict(re.findall(r"^\| `([^`]+\.py)` \| `([0-9a-f]{16})` \|$", text, re.M))
            if pinned:
                for name, expected in sorted(pinned.items()):
                    # Names are written as the record writes them: relative to the model
                    # directory for this port's files, and repo-relative for shared ones.
                    path = next(
                        (option for option in (ROOT / name, repo / name, repo / "models" / name) if option.exists()),
                        None,
                    )
                    if path is None:
                        failures.append(f"the record pins `{name}`, which does not exist")
                        checked += 1
                        continue
                    checked += 1
                    actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                    if actual != expected:
                        failures.append(
                            f"`{name}` is {actual}, the record pins {expected} -- the shipped console "
                            "describes different bytes, so it has to be re-earned"
                        )
                pinned_paths = set()
                for name in pinned:
                    for option in (ROOT / name, repo / name, repo / "models" / name):
                        if option.exists():
                            pinned_paths.add(option.resolve())
                            break
                owned[:] = [path for path in owned if path.resolve() not in pinned_paths]
            if not owned and not pinned:
                failures.append("the provenance check found no stage-owned implementation file to compare against")
            elif not owned:
                pass  # every stage-owned file is hash-pinned above, which is the stronger claim
            else:
                newest_impl = max((path.stat().st_mtime, path) for path in owned)
                if console.stat().st_mtime < newest_impl[0]:
                    failures.append(
                        f"the artifacts table names {named.group(1)}, which is older than "
                        f"{newest_impl[1].name} -- it describes a build that was not shipped"
                    )

    # The one measured figure left in a source docstring. It is not corrected in place
    # -- editing an implementation file to fix a number is what put implementation after
    # the artifacts four times -- so it is gated: the ratio has to stay true of the probe.
    geometry_rows = [row for row in json.loads((DOC / "topk_geometry_probe.json").read_text()) if "ms" in row]
    single = next((row for row in geometry_rows if row["width"] == 50688 and row["calls"] == 1), None)
    split = next((row for row in geometry_rows if row["width"] == 32768 and row["calls"] == 2), None)
    sampling_source = ROOT.parents[1] / "common/sampling/tt_sampling.py"
    if single and split and sampling_source.is_file():
        quoted_ratio = re.search(r"for a (\d+)x\s*\n?\s*reduction on the op", sampling_source.read_text())
        checked += 1
        if quoted_ratio is None:
            failures.append("tt_sampling.py no longer states the split's reduction ratio the README gates")
        elif not close(float(quoted_ratio.group(1)), single["ms"] / split["ms"], tolerance=0.05):
            failures.append(
                f"tt_sampling.py says {quoted_ratio.group(1)}x, topk_geometry_probe.json says "
                f"{single['ms'] / split['ms']:.2f}x ({single['ms']} / {split['ms']})"
            )

    # Prose citations of the form `file.json:key`. One of these named a key that lived in
    # a different evidence file -- the single citation under the accuracy table's "the one
    # non-top-1 position" claim -- and no check looked at them, because they carry no
    # number of their own. A citation that does not resolve is as broken as a wrong figure.
    # Dotted paths are walked, not just top-level keys: `evidence_accuracy.json:capacity.
    # full_context_sequences_that_fit` is a citation like any other. The work log is in
    # the corpus too -- it is a required deliverable and it went stale independently once.
    work_log_path = DOC / "work_log.md"
    citation_corpus = [("README.md", text)]
    if work_log_path.is_file():
        citation_corpus.append(("work_log.md", work_log_path.read_text()))
    for document, body in citation_corpus:
        for name, dotted in sorted(set(re.findall(r"`(evidence_[a-z0-9_]+\.json):([a-z_][a-z0-9_.]*)`", body))):
            path = DOC / name
            checked += 1
            if not path.is_file():
                failures.append(f"{document} cites `{name}:{dotted}`, but {name} does not exist")
                continue
            node = json.loads(path.read_text())
            for step in dotted.split("."):
                if not isinstance(node, dict) or step not in node:
                    failures.append(f"{document} cites `{name}:{dotted}`, but {name} has no `{step}`")
                    break
                node = node[step]

    # Every console the artifacts inventory cites, not just the headline row. The JUnit
    # row named the *pre-rebuild* pytest consoles for a whole review round: they are 13
    # minutes older than the last implementation edit, and they report 226.78 s where the
    # committed XMLs say 253.401. Provenance was gated for one row and hand-maintained
    # for the rest, so the defect simply moved to a row nobody was checking.
    if owned:
        newest_impl_time = max(path.stat().st_mtime for path in owned)
        inventory = text[text.index("## Artifacts") :] if "## Artifacts" in text else ""
        # Table rows only. The prose above the tables deliberately names consoles that no
        # longer exist -- that paragraph is *about* filenames having been overwritten.
        #
        # The exemption is per **row**, not per name: the superseded-trail row lists every
        # old console, so excusing a name because it appears *somewhere* in that row let a
        # current row re-cite a superseded console and pass. That is exactly the defect
        # this check was added for, and the first version of it did not catch its own
        # mutation test.
        excused_names = ""
        anchor = "three groups, and none of them carries an end-to-end figure:"
        if anchor in text:
            # Collect the table that follows the anchor. A ``(.*?)\n\n`` capture stops at
            # the blank line *before* the table and silently excuses nothing.
            after = text[text.index(anchor) + len(anchor) :].splitlines()
            seen_row = False
            for line in after:
                if line.startswith("|"):
                    excused_names += line
                    seen_row = True
                elif seen_row and line.strip():
                    break
        # Every occurrence of a watcher console name must agree with the shipped one: the
        # derivation above reads a single name, so a stale mention in a prose row further
        # down could not be seen. Names are collected across the whole document.
        watcher_names = set(re.findall(r"`logs/(watcher_run_final\d+\.log)`", text))
        if len(watcher_names) > 1:
            failures.append(f"the record names more than one watcher console: {sorted(watcher_names)}")
        for row in inventory.splitlines():
            if not row.startswith("| `"):
                continue
            if row.startswith("| `logs/` |"):
                continue  # this row *is* the superseded trail
            for name in sorted(set(re.findall(r"(?<![/\w])logs/([a-z0-9_.]+\.log(?:\.gz)?)`", row))):
                # The exception groups name some files by glob (`logs/full_test_run*.log`),
                # so the match is fnmatch over the names they list, not a substring test.
                excused_globs = re.findall(r"`logs/([a-z0-9_.*]+\.log(?:\.gz)?)`", excused_names)
                if any(fnmatch.fnmatch(name, pattern) for pattern in excused_globs):
                    continue  # named in the ordering section's documented exception groups
                path = DOC / "logs" / name
                checked += 1
                if not path.is_file():
                    failures.append(f"the artifacts inventory cites `logs/{name}`, which does not exist")
                elif path.stat().st_mtime < newest_impl_time:
                    failures.append(
                        f"the artifacts inventory cites `logs/{name}` as current, but it predates the "
                        "newest stage-owned implementation file -- it describes a build that was not shipped"
                    )

    # The artifact-ordering table is a set of claims about the filesystem, and it was
    # hand-maintained through four rounds of review, wrong in three of them. Every row is
    # now resolved: the path has to exist, its mtime has to be exactly what the row says,
    # and it has to be newer than every stage-owned implementation file.
    #
    # (This check was silently deleted once by an edit that spliced out the span it lived
    # in, and the only symptom was the printed figure count dropping by eight. The count
    # is reported in the README for exactly that reason -- a gate that stops running is
    # indistinguishable from a gate that passes, unless something counts.)
    # Anchored on the paragraph that *follows* the table, which is the claim the table
    # supports; an earlier anchor was a sentence about one specific row and vanished when
    # that row did, taking the whole check with it.
    ordering = re.search(r"### Artifact ordering(.*?)\nEvery artifact in the table above postdates", text, re.S)
    if ordering is None:
        failures.append("the README has no artifact-ordering table for this check to resolve")
    else:
        rows = re.findall(r"^\| `([^|]+?)`[^|]*\| ([0-9:]{8})(?: / ([0-9:]{8}))? \|$", ordering.group(1), re.M)
        if len(rows) < 5:
            failures.append(f"the artifact-ordering table has only {len(rows)} resolvable rows")
        newest_impl_time = max((path.stat().st_mtime for path in owned), default=0.0)
        for names, first, second in rows:
            paths = [name.strip(" `") for name in names.split("`, `")]
            stamps = [stamp for stamp in (first, second) if stamp]
            if len(stamps) == 1:
                stamps = stamps * len(paths)
            for name, want in zip(paths, stamps):
                candidates = [DOC / name, ROOT / name]
                path = next((option for option in candidates if option.exists()), None)
                if path is None:
                    failures.append(f"artifact ordering: `{name}` does not exist")
                    continue
                checked += 1
                got = time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime))
                if got != want:
                    failures.append(f"artifact ordering: `{name}` is {got}, the table says {want}")
                elif newest_impl_time and path.stat().st_mtime < newest_impl_time:
                    failures.append(f"artifact ordering: `{name}` predates the newest stage-owned implementation file")

    # The ordering table's exception groups carry *reasons*, and the gate consumed them
    # only to suppress other checks -- never to test the assertion they make. One reason
    # said each HF/CPU control "postdates its own driver", and one of the three had
    # predated its driver by twelve hours ever since a formatting hook rewrote the driver.
    # An unverified excuse is just a quieter place to keep a wrong claim.
    drivers = {
        "qualitative/qualitative_hf_chat.json": DOC / "bench/qualitative.py",
        "readiness_aime24_chat.refpt": DOC / "bench/readiness_cli.py",
        "readiness_aime24_chat_fp32.refpt": DOC / "bench/readiness_cli.py",
    }
    if "postdates its own driver" in text:
        for artifact, driver in drivers.items():
            path = DOC / artifact if (DOC / artifact).exists() else ROOT / artifact
            if not path.exists() or not driver.exists():
                failures.append(f"the ordering exception names `{artifact}`/`{driver.name}`, and one does not exist")
                continue
            checked += 1
            if path.stat().st_mtime < driver.stat().st_mtime:
                failures.append(
                    f"the ordering exception says every HF/CPU control postdates its own driver, but "
                    f"`{artifact}` is older than `bench/{driver.name}`"
                )

    # The teacher-forcing rate is quoted as a *range* across two evidence files, and a
    # range is exactly the shape that drifts quietly: neither endpoint is the headline.
    fp32_gate = DOC / "evidence_fp32_gate.json"
    if fp32_gate.is_file():
        entries = json.loads(fp32_gate.read_text()).get("teacher_forcing_by_reference", {})
        rates = sorted(entry["per_entry"][0]["decode_t/s/u"] for entry in entries.values())
        if len(rates) >= 2:
            quoted = re.search(r"\*\*([0-9.]+)-([0-9.]+) t/s/u\*\*", text)
            checked += 2
            if quoted is None:
                failures.append("README has no teacher-forcing t/s/u range for this check to resolve")
            else:
                for label, said, actual in (("low", quoted.group(1), rates[0]), ("high", quoted.group(2), rates[-1])):
                    if float(said) != round(actual, len(said.partition(".")[2])):
                        failures.append(
                            f"teacher-forcing {label} endpoint: README says {said}, evidence_fp32_gate.json says {actual}"
                        )

    # The qualitative comparison table -- the `$qualitative-check` deliverable, and the
    # largest evidence table in the document. It had no resolver at all, and a rebuild
    # left six of its cells quoting the previous pass while the round record two hundred
    # lines below *noted that they had moved*. Every cell is resolved here.
    comparison = DOC / "qualitative/qualitative_comparison_chat.json"
    if comparison.is_file():
        rows = {row["id"]: row for row in json.loads(comparison.read_text())}
        table = re.findall(
            r"^\| (p\d) [^|]*\| \*\*([0-9.]+)\*\* \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| token (\d+) \|$",
            text,
            re.M,
        )
        if len(table) != len(rows):
            failures.append(f"the qualitative table has {len(table)} resolvable rows, the artifact has {len(rows)}")
        for name, tt_dup, hf_dup, tt_tri, hf_tri, tt_na, divergence in table:
            row = rows.get(name)
            if row is None:
                failures.append(f"the qualitative table has a row `{name}` the artifact does not")
                continue
            for label, quoted, actual in (
                ("TT adjacent dup", tt_dup, row["tt_adjacent_dup"]),
                ("HF adjacent dup", hf_dup, row["hf_adjacent_dup"]),
                ("TT trigram loop", tt_tri, row["tt_trigram_loop"]),
                ("HF trigram loop", hf_tri, row["hf_trigram_loop"]),
                ("TT non-ASCII", tt_na, row["tt_non_ascii"]),
                ("first divergence", divergence, row["first_divergence_from_hf"]),
            ):
                checked += 1
                if float(quoted) != round(float(actual), len(quoted.partition(".")[2])):
                    failures.append(f"qualitative {name} {label}: README says {quoted}, the artifact says {actual}")

    # The JUnit row quotes each pass's wall time and the XML's own stamp. Round 14 fixed
    # this row by quoting stamps; the next rebuild invalidated them, because nothing
    # resolved them. Both are read off the XMLs now.
    for name, label in (("test_results.xml", "46"), ("test_results_slow.xml", "4")):
        xml = DOC / name
        if not xml.is_file():
            continue
        head = xml.read_text()[:2000]
        seconds = re.search(r'time="([0-9.]+)"', head)
        stamp = re.search(r'timestamp="([0-9T:-]+)', head)
        if seconds is None or stamp is None:
            failures.append(f"{name} has no time/timestamp attribute to resolve the artifacts row against")
            continue
        checked += 2
        if f'time="{seconds.group(1)}"' not in text and seconds.group(1) not in text:
            failures.append(f"the artifacts row does not quote {name}'s time {seconds.group(1)}")
        if stamp.group(1)[:16] not in text:
            failures.append(f"the artifacts row does not quote {name}'s timestamp {stamp.group(1)[:16]}")

    # The TTFT per-round list is the figure the record itself calls the one that moves
    # most between processes, and it was the last ungated one.
    if perf and "rounds" in perf.get("ttft_ms", {}):
        rounds = perf["ttft_ms"]["rounds"]
        quoted_rounds = re.search(r"tightish — ([0-9.]+) / ([0-9.]+) /\s*\n?([0-9.]+) ms across this pass", text)
        checked += 1
        if quoted_rounds is None:
            failures.append("README has no TTFT per-round list for this check to resolve")
        else:
            for index, said in enumerate(quoted_rounds.groups()):
                if float(said) != round(rounds[index], len(said.partition(".")[2])):
                    failures.append(
                        f"TTFT round {index + 1}: README says {said}, evidence_perf.json says {rounds[index]}"
                    )

    # The decode capture's prose. The sampling table has been derived since round 8, but
    # the decode paragraph -- which exists to *explain* the profile -- was hand-maintained
    # prose, and a rebuild left every one of its six figures behind at once. They are all
    # read off the committed CSV here: the two rows the prose names, plus the whole-capture
    # totals the "it is a window artefact" argument rests on.
    decode_csv = DOC / "tracy/decode_perf_report.csv"
    if decode_csv.is_file():
        import csv as _csv

        with decode_csv.open() as handle:
            decode_rows = list(_csv.DictReader(handle))
        device_total = sum(float(row["Device Time"]) for row in decode_rows)
        gap_total = sum(float(row["Op-to-Op Gap"] or 0) for row in decode_rows)
        widest = max(decode_rows, key=lambda row: float(row["Op-to-Op Gap"] or 0))
        head = max(
            (row for row in decode_rows if "50688" in row["OP Code"]),
            key=lambda row: float(row["Device Time"]),
            default=None,
        )
        derived = [
            (
                "decode capture: widest-gap op share (%)",
                r"`EmbeddingsDeviceOperation` at \*\*([0-9.]+) %\*\* of the",
                float(widest["Total %"]),
            ),
            (
                "decode capture: widest-gap op device time (us)",
                r"window — ([0-9.]+) us of device time",
                float(widest["Device Time"]),
            ),
            ("decode capture: widest gap (us)", r"\*\*([0-9.]+) us of op-to-op gap\*\*", float(widest["Op-to-Op Gap"])),
            ("decode capture: device-time total (us)", r"sums to ([0-9.]+) us, which is the", device_total),
            ("decode capture: gap total (us)", r"the ([0-9.]+) us of total gap sits outside it", gap_total),
        ]
        if head is not None:
            derived += [
                (
                    "decode capture: LM-head device time (us)",
                    r"32 x 6656 x 50688`, ([0-9.]+) us,",
                    float(head["Device Time"]),
                ),
                (
                    "decode capture: LM-head share (%)",
                    r"([0-9.]+) % of the \*reduced\* two-layer",
                    float(head["Total %"]),
                ),
            ]
        for label, pattern, actual in derived:
            found = re.search(pattern, text)
            checked += 1
            if found is None:
                failures.append(f"{label}: README has no figure matching {pattern!r}")
            else:
                said = found.group(1)
                if float(said) != round(actual, len(said.partition(".")[2])):
                    failures.append(f"{label}: README says {said}, decode_perf_report.csv says {actual}")

    # Retracted numbers, swept over the **whole stage tree** rather than the two markdown
    # files. Three review rounds in a row found a withdrawn figure surviving somewhere the
    # last sweep had not looked -- first the work log, then `tt/generator.py`, then the
    # bench scripts -- so the sweep is now a gate and its scope is every file this stage
    # owns. A retracted figure may still appear *labelled* as retracted, which is why the
    # check is "not present, or present on a line that says so".
    retracted = {
        # NOTE: 0.7943 was retracted as a superseded max_top_k=8 sampling trace, and a later
        # re-measurement made it the *live* value in sampler_ab.json. A sweep entry that
        # rejects the current artifact's own number is worse than no entry, so it is gone;
        # the README quotes 0.794 and the arm is resolved from the artifact like the rest.
        "16.4x": "derived from the withdrawn 0.592 ms pre-mask endpoint; 9.689/0.632 = 15.3x",
        "23.94 ms/token": "a token-out figure no run produced",
        "230.58 s": ("a reverse-order wall time from a superseded run; logs/reverse_order_run.log holds 230.19 s"),
        "15.3x": (
            "the cross-era sampler ratio (pre-mask 9.689 / post-mask 0.632); the same-process "
            "ratio from the two arms of sampler_ab.json is 15.39x"
        ),
        "implemented and exposed": (
            "limitation 2's description of prefill_forward(continuation=True); the generator "
            "raises NotImplementedError for it, and the phrase survived the guard by two rounds"
        ),
    }
    # "advertise"/"wrong"/"used to" join the list because the retracted *phrase* added in
    # round 14 is quoted in three places that exist precisely to say it was wrong -- the
    # generator's guard comment, the test that pins the guard, and the corrected
    # limitation. A bare re-assertion carries none of these words and still fails.
    marked = (
        "superseded",
        "withdraw",
        "retract",
        "history",
        "no longer",
        "overwrote",
        "overwritten",
        "advertise",
        "wrong",
        "used to",
    )
    swept = 0
    tree = [
        ROOT / "tt",
        ROOT / "tests",
        DOC / "bench",
        DOC / "README.md",
        DOC / "work_log.md",
        ROOT.parents[1] / "common/sampling/tt_sampling.py",
        ROOT.parents[1] / "common/readiness_check",
    ]
    for root in tree:
        files = sorted(root.rglob("*.py")) + sorted(root.rglob("*.sh")) if root.is_dir() else [root]
        for path in files:
            if "__pycache__" in str(path):
                continue
            body = path.read_text()
            if path == pathlib.Path(__file__).resolve():
                # This file necessarily names every retracted figure, to look for it.
                continue
            for lineno, line in enumerate(body.splitlines(), 1):
                for figure, why in retracted.items():
                    if figure in line and not any(word in line.lower() for word in marked):
                        failures.append(
                            f"retracted figure {figure!r} at {path.relative_to(ROOT)}:{lineno} "
                            f"on a line that does not mark it as such -- {why}"
                        )
            swept += 1

    for failure in failures:
        print(f"FIGURE MISMATCH: {failure}", file=sys.stderr)
    if failures:
        return 1
    print(
        f"README figures OK: {checked} figures resolved against committed runs, "
        f"{swept} files swept for retracted figures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
