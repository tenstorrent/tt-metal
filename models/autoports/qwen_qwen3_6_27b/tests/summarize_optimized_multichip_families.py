# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only stage5 search inventory. Never chooses a final/default candidate.

Reads job manifests, command/result JSON and the append-only command ledger.
Only missing-result failures consult logs: at most the final32KiB/64 lines.
Repeated output names retain distinct execution provenance; surviving results
are never attributed to older overwritten attempts or a different job policy.
"""

import argparse
import collections
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def numeric_leaves(value):
    if isinstance(value, dict):
        return [n for v in value.values() for n in numeric_leaves(v)]
    if isinstance(value, list):
        return [n for v in value for n in numeric_leaves(v)]
    return [value] if type(value) in (int, float) else []


def family(name):
    prefix = name.split("_", 1)[0]
    for group in ("geometry", "readers", "precision", "mmrs", "integrated", "placement", "mlp", "prefill", "flat"):
        if prefix.startswith(group):
            return group, prefix
    groups = {
        "family1": "residual_norm",
        "family2": "attention_packing",
        "family3": "mmrs_persistent",
        "persistent": "mmrs_persistent",
        "persistentfix1": "mmrs_persistent",
        "b1family": "batch1",
        "before": "baseline",
        "rope": "rope",
        "packed": "packed_correction",
        "interim": "profile",
    }
    return groups.get(prefix, "other"), prefix


class Catalog:
    def __init__(self, directory, repo):
        self.doc, self.repo = directory.resolve(), repo.resolve()
        self.warnings, self.cache, self.rows = [], {}, []
        self.declarations = collections.defaultdict(list)
        self.covered = set()

    def path(self, value):
        p = Path(value)
        return p.resolve() if p.is_absolute() else (self.repo / p).resolve()

    def display(self, path):
        try:
            return str(Path(path).resolve().relative_to(self.repo))
        except ValueError:
            return str(path)

    def read(self, path):
        if path not in self.cache:
            try:
                if path.stat().st_size > 16 * 1024**2:
                    raise ValueError("JSON larger than16MiB; excluded from compact inventory")
                self.cache[path] = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                self.warnings.append(f"{self.display(path)}: {exc}")
                self.cache[path] = None
        return self.cache[path]

    def option(self, argv, flag):
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None

    def command_key(self, argv):
        args = list(argv)
        for flag in ("--output", "--reference", "--activation-file", "--policy-file"):
            value = self.option(args, flag)
            if value:
                args[args.index(flag) + 1] = str(self.path(value))
        value = self.option(args, "--policy-json")
        if value:
            try:
                args[args.index("--policy-json") + 1] = compact(json.loads(value))
            except ValueError:
                pass
        return tuple(args)

    def job(self, command, result=None):
        argv = command.get("command", [])
        result = result or {}
        policy = result.get("candidate_policy")
        if policy is None:
            try:
                policy_file = self.option(argv, "--policy-file")
                policy = (
                    self.read(self.path(policy_file))
                    if policy_file
                    else json.loads(self.option(argv, "--policy-json") or "{}")
                )
            except ValueError:
                policy = {}
        policy = policy if isinstance(policy, dict) else {}
        job = {
            "policy": policy,
            "reference": self.option(argv, "--reference"),
            "activation_file": self.option(argv, "--activation-file"),
            "baseline": "--baseline" in argv,
            "legacy_decode_layout": "--legacy-decode-layout" in argv,
        }
        for key in ("layer", "batch", "length"):
            value = result.get(key, self.option(argv, "--" + key))
            try:
                job[key] = int(value) if value is not None else None
            except (ValueError, TypeError):
                job[key] = None
        return job

    def job_key(self, job):
        normalized = {k: job.get(k) for k in ("layer", "batch", "length")}
        normalized.update({k: bool(job.get(k, False)) for k in ("baseline", "legacy_decode_layout")})
        normalized["policy"] = job.get("policy", {})
        for k in ("reference", "activation_file"):
            normalized[k] = str(self.path(job[k])) if job.get(k) else None
        return compact(normalized)

    def attach_declarations(self, output, job):
        key = (str(output), self.job_key(job))
        self.covered.add(key)
        return self.declarations.get(key, [])

    @staticmethod
    def sha(record):
        value = record.get("source_sha256", "")
        if isinstance(value, dict):
            return next((v for k, v in value.items() if k.endswith("tt/multichip_decoder.py")), "")
        return value

    @staticmethod
    def timestamp(record):
        try:
            return datetime.datetime.fromisoformat(record["utc"].replace("Z", "+00:00")).timestamp()
        except (KeyError, ValueError, TypeError):
            return None

    def failure_tail(self, path):
        if not path.is_file():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 32768))
            lines = stream.read(32768).decode(errors="replace").splitlines()[-64:]
        bad = [
            re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            for line in lines
            if re.search(
                r"TT_FATAL|AssertionError|RuntimeError|ValueError|TypeError|Traceback|timed out|timeout|SIGKILL|Killed|ERROR",
                line,
                re.I,
            )
        ]
        return " ; ".join(line[:240] for line in bad[-3:])

    def make_row(
        self,
        name,
        output,
        job,
        *,
        result=None,
        command=None,
        command_path=None,
        log_path=None,
        ledger=None,
        ledger_ref="",
        historical=False,
        declared=None,
    ):
        result = result or {}
        command = command or {}
        declarations = declared if declared is not None else self.attach_declarations(output, job)
        group, fam = family(name)
        actual = result.get("selected_policy", {})
        requested = job.get("policy", {})
        role = "+".join(sorted(requested.get("geometry", {})))
        if not role:
            role = next(
                (
                    r
                    for r in ("gateup", "input", "output", "down", "mlp", "attn", "ccl", "norm", "rope")
                    if re.search(r"(?:^|_)" + r, name)
                ),
                "whole_decoder",
            )
        values = []
        for key in ("prefill", "decode", "contribution", "per_user_pcc", "state"):
            values.extend(numeric_leaves(result.get(key, {})))
        for case in result.get("trace", {}).get("cases", []):
            for key in ("baseline_output_pcc", "baseline_state_pcc"):
                values.extend(numeric_leaves(case.get(key, [])))
        partial = bool(result.get("timing_ms")) and not result.get("timing")
        if partial:
            values.extend(numeric_leaves(result.get("pcc", [])))
            values.extend(numeric_leaves(result.get("replay_pcc", [])))
        if result.get("stress"):
            values.extend(numeric_leaves(result["stress"].get("output_pcc", [])))
            values.extend(numeric_leaves(result["stress"].get("state_pcc", [])))
        valid_scores = values and all(math.isfinite(v) for v in values)
        pcc_min = min(values) if valid_scores else None
        timing, provenance = result.get("timing", {}), result.get("provenance", {})
        code = ledger.get("exit_code") if ledger else None
        reason = ""
        if historical:
            status = "failed_historical" if code not in (None, 0) else "historical_result_unavailable"
            reason = f"Ledger exit={code}; output/sidecar name was reused or the original result is unavailable. Current result is not attributed to this execution."
        elif result:
            if result.get("pass") is False or code not in (None, 0) or result.get("finite") is False:
                status, reason = (
                    "failed",
                    f"reported_pass={result.get('pass')}; exit={code}; finite={result.get('finite')}",
                )
            elif values and (not valid_scores or pcc_min < 0.995):
                status, reason = "failed", "Non-finite or below0.995 reported PCC"
            elif result.get("trace", {}).get("bit_equal_eager") is False:
                status, reason = "failed", "Trace/eager mismatch"
            elif result.get("pass") is True:
                status = "pass"
            elif partial:
                status, reason = (
                    "partial_probe",
                    "Role-only microbenchmark; not a full-decoder correctness or latency measurement",
                )
            else:
                status, reason = "incomplete_result", "No explicit passing result"
        else:
            reason = self.failure_tail(log_path) if log_path else ""
            status = "failed" if code not in (None, 0) or reason else "pending"
            reason = reason or (
                f"No result JSON; ledger exit={code}" if ledger else "No measured result for this declared job variant"
            )
        matrices = result.get("matrix_allocations", {})
        geometry = {
            key: {
                k: actual.get("geometry", {}).get(key, {}).get(k, actual.get(k))
                for k in ("cores", "readers", "block", "pipeline")
            }
            for key in matrices
        }
        dtypes = {key: {k: value.get(k) for k in ("dtype", "padded_shape")} for key, value in matrices.items()}
        if not matrices and actual:
            geometry = {
                "global": {k: actual.get(k) for k in ("cores", "readers", "block", "pipeline")},
                "overrides": actual.get("geometry", {}),
            }
        fidelity = {k: actual[k] for k in actual if "dtype" in k or "fidelity" in k}
        instrumented = (
            str(provenance.get("watcher", "0")) not in ("0", "None", "")
            or str(provenance.get("profiler", "0")) not in ("0", "None", "")
            or timing.get("tracking_enabled") is True
        )
        eligible = (
            status == "pass"
            and command_path is not None
            and provenance.get("regime") == "serial"
            and not instrumented
            and (number(timing.get("decode_trace_ms")) or 0) > 0
        )
        row = {
            "family_group": group,
            "family": fam,
            "candidate": name,
            "role": role,
            "layer": job.get("layer"),
            "batch": job.get("batch"),
            "length": job.get("length"),
            "status": status,
            "reported_pass": result.get("pass"),
            "minimum_pcc": pcc_min,
            "prefill_ms": number(timing.get("prefill_ms")),
            "decode_ms": number(timing.get("decode_trace_ms")),
            "measurement_scope": result.get("timing_scope", "role-only probe") if partial else "whole_decoder",
            "partial_probe_timing_ms": result.get("timing_ms", {}) if partial else {},
            "comparison_eligible": eligible,
            "regime": provenance.get("regime", (ledger or {}).get("regime", "unknown")),
            "instrumented": instrumented,
            "control_name": "",
            "control_result": "",
            "comparison_scope": "",
            "prefill_change_pct": None,
            "decode_change_pct": None,
            "provisional_decision": (
                "measured_unpaired_provisional"
                if eligible
                else ("correctness_only_or_provenance_incomplete" if status == "pass" else status)
            ),
            "reason": reason,
            "actual_policy": actual,
            "requested_policy": requested,
            "actual_geometry": geometry,
            "actual_matrix_dtype_shape": dtypes,
            "policy_dtype_fidelity": fidelity,
            "policy_evidence": (
                "result.selected_policy + matrix_allocations"
                if actual
                else "requested only; effective geometry/dtype unknown"
            ),
            "source_sha256": result.get("source_sha256", self.sha(command) or self.sha(ledger or {})),
            "utc": command.get("utc", (ledger or {}).get("utc", "")),
            "result_path": self.display(output),
            "command_path": self.display(command_path) if command_path else "",
            "log_path": self.display(log_path) if log_path and log_path.exists() else "",
            "ledger_record": ledger_ref,
            "manifest_variants": declarations,
            "command_argv": command.get("command", (ledger or {}).get("command", [])),
            "reference_path": job.get("reference"),
            "activation_file": job.get("activation_file"),
            "fixture_metadata": result.get("reference_metadata", {}),
        }
        self.rows.append(row)

    def collect(self):
        files = sorted(self.doc.glob("*.json"))
        for path in files:
            if path.name in ("candidate_search.json",):
                continue
            d = self.read(path)
            if (
                isinstance(d, list)
                and d
                and all(isinstance(j, dict) and {"name", "layer", "batch", "length"} <= j.keys() for j in d)
            ):
                for index, job in enumerate(d):
                    output = self.doc / (job["name"] + ".json")
                    self.declarations[(str(output), self.job_key(job))].append(
                        {"manifest": self.display(path), "index": index, "job": job}
                    )
        ledger, used = [], set()
        ledger_path = self.doc / "candidate_commands.jsonl"
        if ledger_path.exists():
            with ledger_path.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict) and isinstance(record.get("command"), list):
                            ledger.append((record, f"{self.display(ledger_path)}:{line_number}"))
                    except ValueError:
                        self.warnings.append(f"{self.display(ledger_path)}:{line_number}: incomplete/malformed record")
        handled_results = set()
        for command_path in sorted(self.doc.glob("*.command.json")):
            command = self.read(command_path)
            if not isinstance(command, dict) or not isinstance(command.get("command"), list):
                continue
            argv = command["command"]
            name = command_path.name.removesuffix(".command.json")
            sibling = self.doc / (name + ".json")
            result = self.read(sibling) if sibling.exists() else None
            result = result if isinstance(result, dict) and "selected_policy" in result and "layer" in result else None
            if not result and not all(self.option(argv, "--" + k) is not None for k in ("layer", "batch", "length")):
                continue
            output = self.path(self.option(argv, "--output")) if self.option(argv, "--output") else sibling
            # Archived sidecars/logs with a suffix retain their own failure;
            # never borrow the newly overwritten JSON at the original --output.
            if result:
                output = sibling
                handled_results.add(sibling)
            matches = [
                i
                for i, (r, _) in enumerate(ledger)
                if i not in used
                and self.command_key(r["command"]) == self.command_key(argv)
                and self.sha(r) == self.sha(command)
            ]
            selected = None
            stamp = self.timestamp(command)
            if matches and stamp is not None:
                candidates = [
                    (abs(stamp - self.timestamp(ledger[i][0])), i)
                    for i in matches
                    if self.timestamp(ledger[i][0]) is not None and abs(stamp - self.timestamp(ledger[i][0])) <= 30
                ]
                if candidates:
                    selected = min(candidates)[1]
            rec, ref = ledger[selected] if selected is not None else (None, "")
            if selected is not None:
                used.add(selected)
            self.make_row(
                name,
                output,
                self.job(command, result),
                result=result,
                command=command,
                command_path=command_path,
                log_path=self.doc / (name + ".log"),
                ledger=rec,
                ledger_ref=ref,
            )
        for path in files:
            result = self.cache.get(path)
            if (
                path not in handled_results
                and isinstance(result, dict)
                and "selected_policy" in result
                and "layer" in result
            ):
                self.make_row(path.stem, path, self.job({}, result), result=result)
        for i, (record, ref) in enumerate(ledger):
            if i in used:
                continue
            output_arg = self.option(record["command"], "--output")
            if not output_arg:
                continue
            output = self.path(output_arg)
            job = record.get("job", self.job(record))
            self.make_row(job.get("name", output.stem), output, job, ledger=record, ledger_ref=ref, historical=True)
        for key, declarations in self.declarations.items():
            if key not in self.covered:
                job = declarations[0]["job"]
                self.make_row(job["name"], Path(key[0]), job, declared=declarations)
        # Merge only identical artifact/execution/command variants, never names
        # alone. Policies from superseded manifests remain distinct.
        unique = {}
        for row in self.rows:
            # Newer probes retain a source map, older decoder results one SHA.
            # Canonicalize only the identity key; preserve the full provenance.
            key = (
                row["result_path"],
                compact(row["source_sha256"]),
                row["utc"],
                row["status"],
                self.command_key(row["command_argv"]),
                compact(row["requested_policy"]),
            )
            if key not in unique:
                unique[key] = row
            else:
                old = unique[key]
                old["manifest_variants"] = list(
                    {compact(d): d for d in old["manifest_variants"] + row["manifest_variants"]}.values()
                )
        self.rows = list(unique.values())
        self.rows.sort(key=lambda r: (r["family_group"], r["family"], r["candidate"], r["utc"], r["ledger_record"]))
        self.compare()
        return self.rows

    @staticmethod
    def is_control(row):
        name = row["candidate"]
        if re.search(r"_(control|base|legacy)_", name) or name.startswith("before_"):
            return True
        return (
            (row["family_group"] == "placement" and "_L1_" in name)
            or (row["family_group"] == "mlp" and "_packed_" in name)
            or (row["family_group"] == "mmrs" and "_c128_" in name)
        )

    def compare(self):
        good = [r for r in self.rows if r["comparison_eligible"]]
        for row in good:
            shape = tuple(row[k] for k in ("layer", "batch", "length"))
            compatible = [
                r
                for r in good
                if tuple(r[k] for k in ("layer", "batch", "length")) == shape
                and r["fixture_metadata"]
                and r["fixture_metadata"] == row["fixture_metadata"]
            ]
            same_family = [
                r
                for r in compatible
                if r["family"] == row["family"] and r["source_sha256"] == row["source_sha256"] and self.is_control(r)
            ]
            controls = same_family or [r for r in compatible if r["candidate"].startswith("before_")]
            if len(controls) != 1:
                continue
            control = controls[0]
            row["control_name"], row["control_result"] = control["candidate"], control["result_path"]
            row["comparison_scope"] = (
                "same family/source/fixture"
                if same_family
                else "stage before baseline; possibly cross-source, not an isolated knob attribution"
            )
            if control is row:
                row["provisional_decision"] = "control_only_not_default_selection"
                continue
            for phase in ("prefill", "decode"):
                a, b = row[phase + "_ms"], control[phase + "_ms"]
                if number(a) is not None and number(b) is not None and b > 0:
                    row[phase + "_change_pct"] = (a / b - 1) * 100
            delta = row["decode_change_pct"]
            row["provisional_decision"] = (
                ("faster_decode_provisional" if delta < 0 else "slower_or_equal_decode_provisional")
                if delta is not None
                else "measured_unpaired_provisional"
            )


def markdown(rows, catalog, destination):
    counts = collections.Counter(r["status"] for r in rows)
    lines = [
        "# Stage5 candidate search inventory",
        "",
        "Generated from job manifests, result/command JSON, and the append-only command ledger. Every decision is provisional; this table selects no final default.",
        "",
        "CSV is the complete machine-readable record: full effective/requested policy, per-matrix geometry and uploaded dtype/shape, command argv, source/UTC, exact artifact paths and all manifest variants.",
        "",
        "PCC is the minimum reported output/contribution/per-user/state and changed-input trace score. Role-only microbenchmarks have their own scope and never populate full-decoder latency columns. Dtypes/shapes come from result matrix allocations; fidelity and geometry are recorded effective policy, not proof of executed kernel configuration. Pending rows expose requested policy only.",
        "",
        "Comparisons require serial, uninstrumented successful results with identical shape and fixture metadata. A unique same-family/source control is preferred; otherwise only an unambiguous stage-before fixture is used. Negative delta means faster. Cross-source comparisons do not isolate one knob.",
        "",
        "Reused names are split by actual command/source/UTC provenance. Historical executions whose JSON was overwritten retain ledger exit status without borrowing current measurements. Duplicate manifest declarations share one row; unmatched policy variants remain pending.",
        "",
        "Missing-result error extraction reads only the final32KiB/64 lines of that attempt's log. Historical attempts never borrow the latest log. No role-share percentages are inferred.",
        "",
        "Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
    ]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["family_group"]].append(row)
    for required in ("geometry", "readers", "precision", "mmrs", "integrated", "placement", "mlp"):
        groups.setdefault(required, [])

    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    def link_target(value):
        return os.path.relpath(catalog.path(value), destination.resolve().parent)

    for group, entries in sorted(groups.items()):
        lines += [f"## {group}", ""]
        if not entries:
            lines += ["Pending: no job manifest or result discovered for this required family.", ""]
            continue
        lines += [
            "| Candidate / artifact | Role | L / B / S | Status | PCC min | Prefill ms | Decode ms | Control | Δ decode % | Provisional decision |",
            "|---|---|---|---|---:|---:|---:|---|---:|---|",
        ]
        for row in entries:
            link = (
                f"[{row['candidate']}]({link_target(row['result_path'])})" if row["actual_policy"] else row["candidate"]
            )
            shape = "/".join(str(row[k]) if row[k] is not None else "?" for k in ("layer", "batch", "length"))
            lines.append(
                "| "
                + " | ".join(
                    cell(v)
                    for v in (
                        link,
                        row["role"],
                        shape,
                        row["status"],
                        row["minimum_pcc"],
                        row["prefill_ms"],
                        row["decode_ms"],
                        row["control_name"] or "unpaired",
                        row["decode_change_pct"],
                        row["provisional_decision"],
                    )
                )
                + " |"
            )
        lines += ["", "Effective geometry/dtype details and exact requested variants:", ""]
        for row in entries:
            policy = row["actual_policy"] or row["requested_policy"]
            keys = (
                "cores",
                "readers",
                "block",
                "pipeline",
                "geometry",
                "attention_dtype",
                "mlp_dtype",
                "attention_fidelity",
                "mlp_fidelity",
                "attention_activation_dtype",
                "mlp_activation_dtype",
                "ccl_dtype",
                "ccl_decode_memory",
                "fused_mmrs",
                "fused_mmrs_decode",
                "mmrs_chunk_rows",
                "mmrs_max_rows",
                "packed_mlp",
                "minimal_swiglu",
                "packed_qkvg",
                "gather_norm",
                "residual_layout",
                "prefill_chunk",
                "matmul_row_cap",
                "flat_prefill",
                "recurrent_variant",
            )
            knobs = {k: policy[k] for k in keys if k in policy}
            source = "effective result" if row["actual_policy"] else "requested only"
            lines.append(
                f"- **{row['candidate']}** ({row['utc'] or 'no execution timestamp'}; {source}): `{compact(knobs)}`"
            )
            if row["actual_matrix_dtype_shape"]:
                lines.append("  Uploaded matrix dtype/shape: `" + compact(row["actual_matrix_dtype_shape"]) + "`")
            if row["partial_probe_timing_ms"]:
                lines.append(
                    "  Role-only timing (ms): `"
                    + compact(row["partial_probe_timing_ms"])
                    + "`; scope: "
                    + cell(row["measurement_scope"])
                )
            if row["reason"]:
                lines.append("  Reason: " + cell(row["reason"]))
            refs = [row[k] for k in ("command_path", "ledger_record", "log_path") if row[k]]
            if refs:
                lines.append("  Provenance: " + "; ".join(f"`{r}`" for r in refs))
        lines.append("")
    profiles = sorted(catalog.doc.glob("perf/*interim*/decode_perf_report.txt"))
    if profiles:
        lines += [
            "## Interim measured role tables",
            "",
            "Use these profiler tables for measured role costs; no shares are inferred by this summarizer.",
            "",
        ]
        lines += [f"- [{p.parent.name}]({link_target(p)})" for p in profiles]
        lines.append("")
    if catalog.warnings:
        lines += ["## Incomplete inputs", ""] + ["- " + cell(x) for x in catalog.warnings] + [""]
    return "\n".join(lines)


def main():
    model = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=model / "doc/optimized_multichip_decoder")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    catalog = Catalog(args.directory, model.parents[2])
    rows = catalog.collect()
    csv_path, md_path = (
        args.csv or args.directory / "candidate_search.csv",
        args.markdown or args.directory / "candidate_search.md",
    )
    if rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: compact(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    else:
        csv_path.write_text("family_group,candidate,status\n")
    md_path.write_text(markdown(rows, catalog, md_path) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "status": dict(collections.Counter(r["status"] for r in rows)),
                "warnings": len(catalog.warnings),
                "csv": str(csv_path),
                "markdown": str(md_path),
            }
        )
    )


if __name__ == "__main__":
    main()
