# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Resumable target-source → native golden/cache validation of a reviewed C++ port.

Source retrieval remains deterministic. Translation is authored/reviewed before
initialization; this driver never invents a port, repairs failures, or queries a DB.
"""

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path

from tools.generic_op_to_factory import (
    classify_failures,
    compare_baseline,
    dependency_substitutions,
    export_run,
    factory_contract,
    migration_workflow,
    prepare_baseline,
    prepare_target,
    test_evidence,
)
from tools.generic_op_to_factory.classify_failures import parse_junit_xml
from tools.generic_op_to_factory.export_run import ExportError, _hash_file, _safe_path, verify_export
from tools.generic_op_to_factory.migration_workflow import (
    Workflow,
    digest,
    locked,
    now,
    write_json,
    _interrupted,
    ENV_KEYS,
    PROBE,
)
from tools.generic_op_to_factory.prepare_baseline import GitTree, verify_preparation

STAGES = (
    "build",
    "factory_contract",
    "source_smoke",
    "source",
    "source_compare",
    "native_smoke",
    "native",
    "native_compare",
    "cache",
    "review",
    "complete",
)

REVIEW_TOPICS = (
    "descriptor_factory",
    "cb_kernel_semaphore_ids",
    "argument_wiring",
    "cache_hit_overhead",
    "test_gaps",
    "alias_transitions",
)


def verify_review(receipt, plan_sha256):
    """Validate a declared independent review, not authenticate an agent's identity.

    The receipt is supplied after tests/re-review, and bound to the exact plan.
    Actual independence and the truth of dispositions require human/agent review.
    """
    if not isinstance(receipt, dict) or receipt.get("plan_sha256") != plan_sha256:
        raise ExportError("Independent review must match this validation plan")
    for key in ("author", "reviewer"):
        if not isinstance(receipt.get(key), str) or not receipt[key].strip():
            raise ExportError("Review requires author and independent reviewer identities")
    if receipt["author"].strip() == receipt["reviewer"].strip():
        raise ExportError("Reviewer must be independent of the author")
    topics = receipt.get("topics")
    if not isinstance(topics, dict) or any(
        not isinstance(topics.get(key), str) or not topics[key].strip() for key in REVIEW_TOPICS
    ):
        raise ExportError("Review must address all required topics with evidence")
    findings = receipt.get("findings")
    if not isinstance(findings, list):
        raise ExportError("Review requires an explicit findings list")
    for finding in findings:
        if (
            not isinstance(finding, dict)
            or finding.get("status") not in ("fixed", "not_a_defect", "accepted_limitation")
            or any(
                not isinstance(finding.get(key), str) or not finding[key].strip() for key in ("finding", "disposition")
            )
        ):
            raise ExportError("Every review finding needs a resolved disposition")
    performance = receipt.get("performance")
    if (
        not isinstance(performance, dict)
        or performance.get("classification") not in ("measured", "not_measured")
        or not isinstance(performance.get("assessment"), str)
        or not performance["assessment"].strip()
        or not isinstance(performance.get("measurements"), list)
        or (performance["classification"] == "measured") != bool(performance["measurements"])
    ):
        raise ExportError("Review must distinguish performance measurements from estimates")
    evidence = {}
    for measurement in performance["measurements"]:
        if not isinstance(measurement, dict) or not isinstance(measurement.get("path"), str):
            raise ExportError("Measurement evidence requires a path and sha256")
        path = Path(measurement["path"])
        if (
            not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or file_hash(path) != measurement.get("sha256")
        ):
            raise ExportError("Performance measurement evidence is missing or changed")
        evidence[str(path)] = measurement["sha256"]
    return evidence


def file_hash(path):
    return _hash_file(Path(path))[0]


def plan(config):
    required = {
        "runtime",
        "target_revision",
        "preparation",
        "export",
        "phase",
        "workspace",
        "source_entry",
        "native_entry",
        "cache_test",
        "build_argv",
        "precompile",
        "allow_recorded_failures",
        "factory_contract",
    }
    if required - config.keys() or config.keys() - required - {
        "environment",
        "precompile_workers",
        "command_timeout_seconds",
        "dependency_substitutions",
        "source_aliases",
        "smoke_nodeid",
    }:
        raise ExportError("Missing or unknown port validation config keys")
    config = dict(config)
    for key in ("runtime", "preparation", "export", "workspace"):
        path = Path(config[key])
        if not path.is_absolute() or path.is_symlink():
            raise ExportError(f"{key} must be an absolute, unredirected path")
        config[key] = str(path.resolve())
    config.setdefault("environment", {})
    config.setdefault("precompile_workers", 6)
    config.setdefault("command_timeout_seconds", None)
    config.setdefault("dependency_substitutions", [])
    config.setdefault("source_aliases", [])
    config.setdefault("smoke_nodeid", None)
    aliases = config["source_aliases"]
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise ExportError("source_aliases must be a list of entry-point strings")
    if len(set(aliases)) != len(aliases):
        raise ExportError("source_aliases must be unique")
    if config["environment"].keys() - ENV_KEYS:
        raise ExportError("Unsupported environment override")
    if not all(isinstance(v, str) for v in config["environment"].values()):
        raise ExportError("Environment values must be strings")
    for key in ("precompile", "allow_recorded_failures"):
        if type(config[key]) is not bool:
            raise ExportError(f"{key} must be an explicit boolean")
    for key in ("precompile_workers", "command_timeout_seconds"):
        value = config[key]
        if key == "command_timeout_seconds" and value is None:
            continue
        if type(value) is not int or value <= 0:
            raise ExportError(f"{key} must be a positive integer")
    build = config["build_argv"]
    if (
        not isinstance(build, list)
        or not build
        or build[0] not in ("./build_metal.sh", ".github/scripts/copilot-build.sh")
    ):
        raise ExportError("Build must use build_metal.sh or its CI wrapper")
    if any(
        not isinstance(a, str)
        or not a
        or a.split("=")[0] in ("--clean", "--configure-only", "--build-dir", "--install-prefix")
        for a in build
    ):
        raise ExportError("Unsupported build arguments")
    manifest = verify_preparation(config["preparation"])
    frozen = verify_export(config["export"])
    # The prepared source and exported comparison data must belong to the same snapshot.
    run = json.loads((Path(config["export"]) / "records/run.json").read_bytes())
    if (
        run["starting_commit"] != manifest["metal_commit"]
        or run["eval_commit"] != manifest["eval_commit"]
        or run["prompt_name"] != manifest["operation"]
    ):
        raise ExportError("Preparation and export identities differ")
    for entry in manifest["files"]:
        if entry["origin"] == "db" and file_hash(Path(config["export"]) / entry["export_path"]) != entry["sha256"]:
            raise ExportError("Prepared source differs from export")
    target = prepare_target.inspect(
        config["preparation"],
        config["runtime"],
        config["target_revision"],
        allow_installed=True,
        substitutions=config["dependency_substitutions"],
    )
    suite = f"eval/golden_tests/{run['golden_name']}"
    if config["smoke_nodeid"] is not None:
        smoke = config["smoke_nodeid"]
        if not isinstance(smoke, str) or not smoke.startswith(suite + "/") or "::" not in smoke:
            raise ExportError("Choose an explicit smoke case in the recorded suite, or omit smoke_nodeid")
        _safe_path(smoke.split("::")[0])
    _safe_path(config["cache_test"])
    if not config["cache_test"].startswith("tests/") or not config["cache_test"].endswith(".py"):
        raise ExportError("cache_test must identify a checked-in-style Python test path")
    if not config["source_entry"].startswith("ttnn.operations." + manifest["operation"] + ":"):
        raise ExportError("Source entry must select the frozen operation package")
    for entry in (config["source_entry"], config["native_entry"], *aliases):
        module, sep, symbol = entry.partition(":")
        if not sep or not symbol.isidentifier() or not all(p.isidentifier() for p in module.split(".")):
            raise ExportError("Entries must use module.path:symbol")
    if config["source_entry"] == config["native_entry"] or config["native_entry"] in aliases:
        raise ExportError("Source and native entries must differ")
    rows = [
        json.loads(line) for line in (Path(config["export"]) / "records/test_results.jsonl").read_text().splitlines()
    ]
    rows = [row for row in rows if row.get("phase") == config["phase"]]
    historical = compare_baseline.compare_outcomes(rows, rows)
    if historical["observed_failures"] and not config["allow_recorded_failures"]:
        raise ExportError("Recorded baseline has failures; explicit acceptance is required")
    runtime = Path(config["runtime"])
    config["factory_contract"] = factory_contract.validate(config["factory_contract"], runtime)
    for path in (
        runtime / config["cache_test"],
        runtime / "tools/generic_op_to_factory/native_adapter.py",
        runtime / "scripts/run_safe_pytest.sh",
    ):
        if not path.is_file() or not path.resolve().is_relative_to(runtime):
            raise ExportError(f"Required validation source is missing/redirected: {path}")
    adapter = runtime / "tools/generic_op_to_factory/native_adapter.py"
    if file_hash(adapter) != test_evidence.ADAPTER_SHA256:
        raise ExportError("Target native_adapter.py must match this flow version before initialization")
    test_evidence.check_runner(runtime / "scripts/run_safe_pytest.sh", precompile=config["precompile"])
    tree = GitTree(runtime, config["target_revision"])
    untracked = tree.git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    untracked_hashes = {}
    for relative in filter(None, untracked):
        path = runtime / _safe_path(relative)
        if path.is_symlink() or not path.resolve().is_relative_to(runtime):
            raise ExportError(f"Untracked source is redirected: {relative}")
        untracked_hashes[relative] = file_hash(path)
    # Include tracked changes and all untracked non-ignored source; no hidden source edits on resume.
    return {
        "config": config,
        "target_inputs": target,
        "snapshot_sha256": frozen["snapshot_sha256"],
        "preparation_sha256": manifest["preparation_sha256"],
        "suite": suite,
        "tracked_diff_sha256": digest(tree.git("diff", "--binary", "HEAD", "--").decode()),
        "untracked_files": untracked_hashes,
        "implementation": {
            p.name: file_hash(p)
            for p in (
                Path(__file__),
                Path(prepare_target.__file__),
                Path(compare_baseline.__file__),
                Path(migration_workflow.__file__),
                Path(prepare_baseline.__file__),
                Path(export_run.__file__),
                Path(classify_failures.__file__),
                Path(dependency_substitutions.__file__),
                Path(factory_contract.__file__),
                Path(test_evidence.__file__),
            )
        },
        "recorded_case_count": len(rows),
        "historical_failures": historical["observed_failures"],
    }


class PortValidation:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.planned = json.loads((self.workspace / "plan.json").read_bytes())
        self.config = self.planned["config"]
        self.runtime = Path(self.config["runtime"])
        self.state = json.loads((self.workspace / "state.json").read_bytes())
        # Reuse the audited command/process-group/environment runner, not its recorded-revision stage logic.
        self.runner = object.__new__(Workflow)
        self.runner.runtime, self.runner.workspace = self.runtime, self.workspace
        self.runner.config = {**self.config, "capture_metrics": False}

    def validate(self):
        if str(self.workspace) != self.config["workspace"] or digest(self.planned) != self.state["plan_sha256"]:
            raise ExportError("Validation workspace identity changed")
        if plan(self.config) != self.planned:
            raise ExportError("Port source, input or configuration drift; use a new validation workspace")
        for record in self.state["stages"].values():
            if record["status"] == "complete":
                for path, sha in record["evidence"].items():
                    if file_hash(path) != sha:
                        raise ExportError(f"Validation evidence/runtime changed: {path}")

    def execute(self, stage, attempt):
        c = self.config
        if stage == "build":
            self.runner.command(c["build_argv"], attempt, "build", activate=True)
            self.runner.command(["python3", "-c", PROBE], attempt, "runtime-probe", activate=True)
            probe = Workflow.parse_probe((attempt / "runtime-probe.log").read_text())
            write_json(attempt / "runtime.json", probe)
            return {str(p): file_hash(p) for p in (self.runtime / "build_Release/lib").glob("*.so*") if p.is_file()}
        if stage == "factory_contract":
            probe = attempt / "factory_contract.cpp"
            probe.write_text(factory_contract.render(c["factory_contract"]))
            build = self.runtime / "build_Release"
            database = None
            metadata = {}
            if (build / "build.ninja").is_file():
                # Read the completed build's actual commands, without reconfiguring
                # CMake or disabling unity just to export a compilation database.
                for path in build.rglob("*.ninja"):
                    if path.resolve() != path or not path.is_file():
                        raise ExportError("Redirected Ninja build metadata")
                    metadata[str(path)] = file_hash(path)
                self.runner.command(
                    ["ninja", "-C", str(build), "-t", "compdb"],
                    attempt,
                    "compile-database",
                    stdout_file="compile-database.json",
                )
                database = attempt / "compile-database.json"
            argv, cwd, evidence = factory_contract.compile_invocation(
                c["factory_contract"], self.runtime, probe, database=database
            )
            evidence.update(metadata)
            self.runner.command(argv, attempt, "factory-contract", cwd=cwd)
            write_json(attempt / "contract.json", {"kind": "ProgramDescriptor", **c["factory_contract"]})
            return evidence
        if stage in ("source_smoke", "source", "native_smoke", "native", "cache"):
            if stage.endswith("smoke") and c["smoke_nodeid"] is None:
                write_json(
                    attempt / "skipped.json", {"reason": "Optional smoke omitted; full golden suite remains required"}
                )
                return {}
            mode = "source" if stage.startswith("source") else "native"
            route = {
                "source": c["source_entry"],
                "native": c["native_entry"],
                "mode": mode,
                "aliases": c["source_aliases"],
            }
            write_json(attempt / "route.json", route)
            test = (
                c["cache_test"]
                if stage == "cache"
                else (c["smoke_nodeid"] if stage.endswith("smoke") else self.planned["suite"])
            )
            argv = ["./scripts/run_safe_pytest.sh", "--run-all"]
            if stage in ("source", "native") and c["precompile"]:
                argv += [
                    "--precompile",
                    "--precompile-workers",
                    str(c["precompile_workers"]),
                ]
            else:
                argv += ["--no-precompile"]
            argv += [
                test,
                "-q",
                "--tb=short",
                "-o",
                "addopts=--import-mode=importlib",
                "-p",
                "eval.hang_plugin",
                "-p",
                "eval.metrics_plugin",
                "-p",
                "eval.axes_plugin",
            ]
            if stage != "cache":
                argv += [
                    "-p",
                    "tools.generic_op_to_factory.native_adapter",
                    "--migration-route",
                    str(attempt / "route.json"),
                ]
            argv += [f"--junitxml={attempt / 'junit.xml'}"]
            self.runner.command(
                argv,
                attempt,
                stage,
                activate=True,
                device=True,
                allowed=(0, 1) if stage in ("source", "native") else (0,),
            )
            observed = test_evidence.verify(
                attempt / f"{stage}.log", attempt / "junit.xml", route=route if stage != "cache" else None
            )
            write_json(attempt / "execution.json", observed)
            rows = parse_junit_xml(attempt / "junit.xml")
            if not rows:
                raise ExportError("No test outcomes")
            if stage == "cache" and (
                not any(r["status"] == "passed" for r in rows) or any(r["status"] != "passed" for r in rows)
            ):
                raise ExportError("Every explicitly selected cache test must pass")
            return {}
        if stage == "source_compare":
            junit = self.workspace / self.state["stages"]["source"]["attempts"][-1] / "junit.xml"
            comparison = compare_baseline.compare(c["export"], junit, c["phase"])
        elif stage == "native_compare":
            source = self.workspace / self.state["stages"]["source"]["attempts"][-1] / "junit.xml"
            native = self.workspace / self.state["stages"]["native"]["attempts"][-1] / "junit.xml"
            comparison = compare_baseline.compare_outcomes(parse_junit_xml(source), parse_junit_xml(native))
        elif stage == "review":
            receipt_path = self.workspace / "review.json"
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ExportError(
                    "Independent review required: supply workspace/review.json for this plan; "
                    "inspect findings, fix/revalidate in a new workspace if needed, then --retry"
                )
            receipt = json.loads(receipt_path.read_bytes())
            evidence = verify_review(receipt, self.state["plan_sha256"])
            write_json(attempt / "review.json", receipt)
            evidence[str(receipt_path)] = file_hash(receipt_path)
            return evidence
        elif stage == "complete":
            write_json(
                attempt / "result.json",
                {
                    "native_migration_validated": True,
                    "historical_failures": self.planned["historical_failures"],
                    "production_ready": False,
                    "independent_review_recorded": True,
                    "factory_kind": "ProgramDescriptor",
                    "factory_contract": c["factory_contract"],
                    "baseline_scope": self.planned["target_inputs"]["baseline_scope"],
                    "dependency_substitutions": self.planned["target_inputs"]["dependency_substitutions"],
                    "scope": "Recorded golden outcomes/tolerances and explicitly supplied cache regression tests; no performance or trace claim",
                },
            )
            return {}
        else:
            raise ExportError(f"Unknown validation stage: {stage}")
        write_json(attempt / "comparison.json", comparison)
        if not comparison["outcomes_match"]:
            raise ExportError("Case outcomes differ; see comparison.json")
        return {}

    def run(self, through="complete", retry=False):
        with locked(self.workspace):
            self.state = json.loads((self.workspace / "state.json").read_bytes())
            self.validate()
            for stage in STAGES[: STAGES.index(through) + 1]:
                record = self.state["stages"][stage]
                if record["status"] == "complete":
                    continue
                if record["status"] != "pending":
                    if not retry:
                        raise ExportError(f"{stage} is {record['status']}; inspect then explicitly --retry")
                    for old in record["attempts"]:
                        for p in (self.workspace / old).glob("*.command.json"):
                            command = json.loads(p.read_bytes())
                            if "finished_at" not in command:
                                if not command.get("pid"):
                                    raise ExportError("Unfinished command identity unknown")
                                try:
                                    os.kill(command["pid"], 0)
                                except ProcessLookupError:
                                    pass
                                else:
                                    raise ExportError("Unfinished command may still be alive")
                attempt = self.workspace / "attempts" / stage / f"{len(record['attempts']) + 1:03d}"
                if not attempt.resolve().is_relative_to(self.workspace):
                    raise ExportError("Attempt path is redirected")
                attempt.mkdir(parents=True)
                record["attempts"].append(str(attempt.relative_to(self.workspace)))
                record.update(status="running", started_at=now())
                write_json(self.workspace / "state.json", self.state)
                print(f"PORT: {stage} started ({attempt})", flush=True)
                try:
                    evidence = self.execute(stage, attempt)
                    self.validate()
                    evidence.update({str(p): file_hash(p) for p in attempt.iterdir() if p.is_file()})
                    record.update(status="complete", evidence=evidence, finished_at=now())
                except BaseException as error:
                    record.update(
                        status=(
                            "interrupted"
                            if isinstance(error, (KeyboardInterrupt, subprocess.TimeoutExpired))
                            else "blocked"
                        ),
                        error=str(error),
                        finished_at=now(),
                    )
                    write_json(self.workspace / "state.json", self.state)
                    raise
                write_json(self.workspace / "state.json", self.state)
                print(f"PORT: {stage} complete", flush=True)
            return self.state


def initialize(config):
    planned = plan(config)
    workspace = Path(planned["config"]["workspace"])
    # Evidence must not contaminate the source snapshot or frozen input packages.
    for key in ("preparation", "export"):
        if workspace.is_relative_to(Path(planned["config"][key])):
            raise ExportError("Evidence cannot be written inside frozen inputs")
    if workspace.is_relative_to(Path(planned["config"]["runtime"])):
        raise ExportError("Use an evidence workspace outside the target repository")
    workspace.mkdir(parents=True)
    state = {
        "plan_sha256": digest(planned),
        "created_at": now(),
        "stages": {stage: {"status": "pending", "attempts": []} for stage in STAGES},
    }
    write_json(workspace / "plan.json", planned)
    write_json(workspace / "state.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "init"):
        sub.add_parser(name).add_argument("--config", type=Path, required=True)
    command = sub.add_parser("run")
    command.add_argument("--workspace", type=Path, required=True)
    command.add_argument("--through", choices=STAGES, default="complete")
    command.add_argument("--retry", action="store_true")
    sub.add_parser("status").add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _interrupted)
    try:
        if args.action in ("plan", "init"):
            result = (plan if args.action == "plan" else initialize)(json.loads(args.config.read_bytes()))
        elif args.action == "status":
            result = json.loads((args.workspace / "state.json").read_bytes())
        else:
            result = PortValidation(args.workspace).run(args.through, args.retry)
        print(json.dumps(result, indent=2, sort_keys=True))
    except KeyboardInterrupt:
        parser.exit(130, "Port validation interrupted; inspect evidence before retry\n")
    except (
        ExportError,
        OSError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(2, f"Port validation blocked: {error}\n")


if __name__ == "__main__":
    main()
