# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Resumable recorded-baseline milestone of the generic migration workflow.

No run IDs, operation-specific recipes, LLM retrieval, or automatic repair.
The target baseline, C++ translation and cache tests are explicit future gates.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path

from tools.generic_op_to_factory import (
    classify_failures,
    compare_baseline,
    dependency_substitutions,
    export_run,
    prepare_baseline,
    test_evidence,
)
from tools.generic_op_to_factory.export_run import ExportError, _hash_file, _safe_path, json_bytes

STAGES = (
    "prepare",
    "checkout",
    "evaluator",
    "install",
    "environment",
    "build",
    "collect",
    "smoke",
    "baseline",
    "compare",
)
FUTURE_GATES = ("target_baseline", "cpp_translation", "equivalence", "cache_validation")
ENV_KEYS = {
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CPM_SOURCE_CACHE",
    "CMAKE_PREFIX_PATH",
    "CCACHE_DIR",
    "CCACHE_BASEDIR",
    "CCACHE_CONFIGPATH",
    "LD_LIBRARY_PATH",
}
ACTIVATE = [
    "bash",
    "-c",
    'source ./python_env/bin/activate && exec "$@"',
    "migration-workflow",
]
PROBE = """import importlib.metadata, json, pathlib, sys, ttnn
root = pathlib.Path.cwd().resolve()
paths = {"ttnn": ttnn.__file__, "extension": ttnn._ttnn.__file__}
assert all(pathlib.Path(p).resolve().is_relative_to(root) for p in paths.values()), paths
assert pathlib.Path(sys.executable).absolute().is_relative_to(root / "python_env"), sys.executable
print("WORKFLOW_RUNTIME_JSON=" + json.dumps({"paths": paths, "python": sys.executable,
    "packages": sorted((d.metadata.get("Name", ""), d.version) for d in importlib.metadata.distributions())}))
"""


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path, value):
    """Atomically publish only workflow-owned metadata; never truncate a log."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".state-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def implementation():
    modules = (
        export_run,
        prepare_baseline,
        compare_baseline,
        classify_failures,
        dependency_substitutions,
        test_evidence,
    )
    paths = [Path(__file__), *(Path(module.__file__) for module in modules)]
    return {path.name: _hash_file(path)[0] for path in paths}


def _absolute(value, label):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ExportError(f"{label} must be an absolute local path")
    if Path(value).is_symlink():
        raise ExportError(f"{label} must not be a symlink")
    return str(Path(value).resolve())


def plan(raw):
    required = {
        "export",
        "metal_repository",
        "eval_repository",
        "workspace",
        "target_revision",
        "phase",
        "precompile",
        "capture_metrics",
        "build_argv",
    }
    optional = {
        "environment",
        "configure_argv",
        "sfpi_directory",
        "smoke_nodeid",
        "precompile_workers",
        "command_timeout_seconds",
        "evaluator_path",
        "dependency_substitutions",
    }
    if not isinstance(raw, dict) or required - raw.keys() or raw.keys() - required - optional:
        raise ExportError("Config has missing or unknown keys; see MIGRATION_WORKFLOW.md")
    config = dict(raw)
    for key in ("export", "metal_repository", "eval_repository", "workspace"):
        config[key] = _absolute(config[key], key)
    workspace = Path(config["workspace"])
    for key in ("export", "metal_repository", "eval_repository"):
        other = Path(config[key])
        if workspace.is_relative_to(other) or other.is_relative_to(workspace):
            raise ExportError("Workspace must be disjoint from repositories and frozen export")
    if config["phase"] is not None and (not isinstance(config["phase"], str) or not config["phase"]):
        raise ExportError("phase must be an explicit name or null (unphased)")
    for key in ("precompile", "capture_metrics"):
        if not isinstance(config[key], bool):
            raise ExportError(f"{key} must be an explicit boolean")
    build = config["build_argv"]
    if (
        not isinstance(build, list)
        or not build
        or build[0] not in ("./build_metal.sh", ".github/scripts/copilot-build.sh")
    ):
        raise ExportError("build_argv must invoke build_metal.sh or its CI wrapper")
    if any(not isinstance(arg, str) or not arg for arg in build) or any(
        arg.split("=")[0] in ("--clean", "--configure-only", "--build-dir", "--install-prefix") for arg in build
    ):
        raise ExportError("Destructive, configure-only or relocated builds are not supported")
    configure = config.setdefault("configure_argv", [])
    if not isinstance(configure, list) or any(not isinstance(arg, str) or not arg for arg in configure):
        raise ExportError("configure_argv must be an argument array")
    if configure and (
        configure[0] != "cmake" or any(arg in ("--build", "--install", "--workflow", "-P", "-E") for arg in configure)
    ):
        raise ExportError("configure_argv must only configure CMake")
    env = config.setdefault("environment", {})
    if not isinstance(env, dict) or env.keys() - ENV_KEYS or any(not isinstance(value, str) for value in env.values()):
        raise ExportError("environment contains unsupported keys or non-string values")
    config.setdefault("sfpi_directory", None)
    if config["sfpi_directory"]:
        config["sfpi_directory"] = _absolute(config["sfpi_directory"], "sfpi_directory")
        if not configure:
            raise ExportError("An existing SFPI directory requires explicit configure_argv")
    config.setdefault("smoke_nodeid", None)
    config.setdefault("precompile_workers", 8)
    config.setdefault("command_timeout_seconds", None)
    for key in ("precompile_workers", "command_timeout_seconds"):
        value = config[key]
        if value is None and key == "command_timeout_seconds":
            continue
        if type(value) is not int or value <= 0:
            raise ExportError(f"{key} must be a positive integer")
    _safe_path(config.setdefault("evaluator_path", "tt_metal/third_party/tt_ops_code_gen"))
    # Restrict nested-repository ownership to the known integration namespace.
    if not config["evaluator_path"].startswith("tt_metal/third_party/"):
        raise ExportError("evaluator_path must be below tt_metal/third_party")
    frozen = export_run.verify_export(config["export"])
    run = json.loads((Path(config["export"]) / "records/run.json").read_bytes())
    source_tree = prepare_baseline.GitTree(config["metal_repository"], run.get("starting_commit"))
    substitutions = dependency_substitutions.resolve(config.setdefault("dependency_substitutions", []), source_tree)
    prepare_baseline.GitTree(config["eval_repository"], run.get("eval_commit"))
    prepare_baseline.GitTree(config["metal_repository"], config["target_revision"])
    suite = run.get("golden_name")
    operation = run.get("prompt_name")
    if not all(isinstance(name, str) and name.isascii() and name.isidentifier() for name in (suite, operation)):
        raise ExportError("Recorded operation and suite must be Python identifiers")
    smoke = config["smoke_nodeid"]
    if smoke is not None and (
        not isinstance(smoke, str) or not smoke.startswith(f"eval/golden_tests/{suite}/") or "::" not in smoke
    ):
        raise ExportError("smoke_nodeid must explicitly select a case in the recorded golden suite")
    if smoke:
        _safe_path(smoke.split("::", 1)[0])
    rows = []
    with (Path(config["export"]) / "records/test_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("phase") == config["phase"]:
                rows.append(row)
    compare_baseline.compare_outcomes(rows, rows)  # reject empty/ambiguous phase before mutations
    external = {}
    if config["sfpi_directory"]:
        compiler = Path(config["sfpi_directory"]) / "compiler/bin/riscv-tt-elf-g++"
        external[str(compiler)] = _hash_file(compiler)[0]
    return {
        "format_version": 1,
        "config": config,
        "implementation": implementation(),
        "input_snapshot_sha256": frozen["snapshot_sha256"],
        "external_inputs": external,
        "dependency_substitutions": substitutions,
        "baseline_scope": dependency_substitutions.scope(substitutions),
        "metal_revision": run["starting_commit"],
        "eval_revision": run["eval_commit"],
        "operation": operation,
        "golden_suite": suite,
        "recorded_case_count": len(rows),
        "stages": list(STAGES),
        "future_gates": list(FUTURE_GATES),
        "migration_ready": False,
    }


def initialize(raw):
    planned = plan(raw)
    workspace = Path(planned["config"]["workspace"])
    workspace.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir()  # Exclusive: never adopt or erase an existing workspace.
    write_json(workspace / "plan.json", planned)
    state = {
        "format_version": 1,
        "plan_sha256": digest(planned),
        "created_at": now(),
        "baseline_scope": planned["baseline_scope"],
        "stages": {name: {"status": "pending", "attempts": []} for name in STAGES},
        "future_gates": {name: "not_implemented" for name in FUTURE_GATES},
        "migration_ready": False,
    }
    write_json(workspace / "state.json", state)
    return state


@contextmanager
def locked(workspace):
    path = workspace / ".workflow.lock"
    if path.is_symlink():
        raise ExportError("Workspace lock is redirected")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExportError("Another driver owns this workspace") from error
        yield


def _interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


class Workflow:
    def __init__(self, workspace):
        self.workspace = Path(_absolute(str(workspace), "workspace"))
        self.planned = json.loads((self.workspace / "plan.json").read_bytes())
        self.config = self.planned["config"]
        self.runtime = self.workspace / "runtime"
        self.prepared = self.workspace / "prepared"
        self.state = json.loads((self.workspace / "state.json").read_bytes())

    def save(self):
        write_json(self.workspace / "state.json", self.state)

    def receipt(self, stage):
        record = self.state["stages"][stage]
        return json.loads((self.workspace / record["receipt"]).read_bytes())

    def fingerprint(self, path):
        path = Path(path)
        if not path.resolve().is_relative_to(self.workspace):
            raise ExportError("Evidence or runtime file points outside the workspace")
        sha, size = _hash_file(path)
        return {
            "path": str(path.relative_to(self.workspace)),
            "sha256": sha,
            "size_bytes": size,
        }

    def git(self, cwd, *args):
        return (
            subprocess.check_output(
                ["git", "--no-replace-objects", "-C", str(cwd), *args],
                stderr=subprocess.PIPE,
            )
            .decode()
            .strip()
        )

    def head(self, path, expected):
        if self.git(path, "rev-parse", "HEAD") != expected:
            raise ExportError(f"Changed runtime revision: {path}")
        if self.git(path, "diff", "--name-only", "--ignore-submodules=all", "HEAD", "--"):
            raise ExportError(f"Modified tracked runtime files: {path}")

    def validate(self):
        if self.runtime.is_symlink() or self.runtime.resolve() != self.runtime:
            raise ExportError("Runtime directory is redirected")
        if str(self.workspace) != self.config["workspace"] or digest(self.planned) != self.state["plan_sha256"]:
            raise ExportError("Workspace identity/plan changed")
        if plan(self.config) != self.planned:
            raise ExportError("Inputs, configuration or driver implementation changed; use a new workspace")
        for stage in STAGES:
            record = self.state["stages"][stage]
            if record["status"] != "complete":
                continue
            receipt_path = self.workspace / _safe_path(record["receipt"])
            if _hash_file(receipt_path)[0] != record["receipt_sha256"]:
                raise ExportError(f"Changed stage receipt: {stage}")
            for entry in self.receipt(stage)["evidence"]:
                path = self.workspace / _safe_path(entry["path"])
                if self.fingerprint(path) != entry:
                    raise ExportError(f"Changed evidence/runtime output: {entry['path']}")
        if self.state["stages"]["prepare"]["status"] == "complete":
            manifest = prepare_baseline.verify_preparation(self.prepared)
            if manifest["preparation_sha256"] != self.receipt("prepare")["result"]["preparation_sha256"]:
                raise ExportError("Preparation lineage changed")
        if self.state["stages"]["checkout"]["status"] == "complete":
            self.head(self.runtime, self.planned["metal_revision"])
        if self.state["stages"]["evaluator"]["status"] == "complete":
            evaluator = self.runtime / self.config["evaluator_path"]
            if (
                not evaluator.resolve().is_relative_to(self.runtime)
                or (self.runtime / "eval").resolve() != (evaluator / "eval").resolve()
            ):
                raise ExportError("Evaluator directory is redirected")
            self.head(
                self.runtime / self.config["evaluator_path"],
                self.planned["eval_revision"],
            )
            if (
                self.git(self.runtime, "submodule", "status", "--recursive")
                != self.receipt("evaluator")["result"]["submodules"]
            ):
                raise ExportError("Runtime submodule state changed")
            self.git(
                self.runtime,
                "submodule",
                "foreach",
                "--quiet",
                "--recursive",
                "git diff --quiet HEAD --",
            )
        if self.state["stages"]["install"]["status"] == "complete":
            self.source_audit()
        if self.state["stages"]["build"]["status"] == "complete":
            build_attempt = self.workspace / self.state["stages"]["build"]["attempts"][-1]
            saved = json.loads((build_attempt / "runtime.json").read_bytes())
            env, _ = self.environment(build_attempt)
            observed = subprocess.run(
                ACTIVATE + ["python3", "-c", PROBE],
                cwd=self.runtime,
                env=env,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.config["command_timeout_seconds"] or 60,
            )
            if self.parse_probe(observed.stdout) != saved:
                raise ExportError("Loaded runtime or Python package environment changed")

    @staticmethod
    def parse_probe(output):
        probes = [
            line.removeprefix("WORKFLOW_RUNTIME_JSON=")
            for line in output.splitlines()
            if line.startswith("WORKFLOW_RUNTIME_JSON=")
        ]
        if len(probes) != 1:
            raise ExportError("Runtime identity probe did not produce one valid report")
        return json.loads(probes[0])

    def source_audit(self):
        self.head(self.runtime, self.planned["metal_revision"])
        self.head(self.runtime / self.config["evaluator_path"], self.planned["eval_revision"])
        dependency_substitutions.check_runtime(self.planned["dependency_substitutions"], self.runtime, installed=True)
        dependency_substitutions.check_declared_headers(
            self.planned["dependency_substitutions"],
            self.runtime,
            prepare_baseline.GitTree(self.runtime, self.planned["metal_revision"]),
        )
        manifest = prepare_baseline.verify_preparation(self.prepared)
        for entry in manifest["files"]:
            path = entry["path"]
            if path.startswith("overlay/"):
                actual = self.runtime / path.removeprefix("overlay/")
            elif path.startswith("reference/metal/"):
                actual = self.runtime / path.removeprefix("reference/metal/")
            else:
                continue
            if _hash_file(actual)[0] != entry["sha256"]:
                raise ExportError(f"Installed source/dependency changed: {actual}")
        source = self.runtime / "ttnn/ttnn/operations" / self.planned["operation"]
        expected = {
            entry["export_path"].removeprefix("source/") for entry in manifest["files"] if entry["origin"] == "db"
        }
        actual = {
            p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        }
        if actual != expected:
            raise ExportError("Installed operation has missing or additional source files")

    def environment(self, attempt, device=False):
        # Preserve only basic host paths/local identity, not ambient pytest,
        # simulator, database or profiler configuration. Credentials stay out
        # of configuration and saved metadata.
        env = {
            key: os.environ[key]
            for key in (
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "TMPDIR",
                "SSH_AUTH_SOCK",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
            )
            if key in os.environ
        }
        overrides = dict(self.config["environment"])
        overrides.update(
            TT_METAL_HOME=str(self.runtime),
            PYTHONPATH=str(self.runtime),
            TT_METAL_CACHE=str(self.workspace / "device-cache"),
            TT_METAL_ENV="dev",
            TT_METAL_CCACHE_KERNEL_SUPPORT="1",
            PYTEST_ADDOPTS="",
            CI="false",
            LOGURU_LEVEL="ERROR",
        )
        if device:
            overrides.update(
                PYTEST_AXES_JSON=str(attempt / "axes.json"),
                PYTEST_EXTRAS_JSON=str(attempt / "extras.json"),
                TT_DEVICE_TIMING_LOG=str(attempt / "timings.jsonl"),
            )
            if self.config["capture_metrics"]:
                overrides.update(
                    TT_METAL_DEVICE_PROFILER="1",
                    TT_METAL_PROFILER_MID_RUN_DUMP="1",
                    TT_METAL_PROFILER_CPP_POST_PROCESS="1",
                    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES="1",
                )
        env.update(overrides)
        return env, overrides

    def command(
        self,
        argv,
        attempt,
        label,
        *,
        cwd=None,
        activate=False,
        allowed=(0,),
        device=False,
        stdout_file=None,
    ):
        argv = [arg.replace("{runtime}", str(self.runtime)).replace("{workspace}", str(self.workspace)) for arg in argv]
        if activate:
            if not (self.runtime / "python_env/bin/activate").is_file():
                raise ExportError("Runtime environment has not been created")
            argv = ACTIVATE + argv
        env, overrides = self.environment(attempt, device)
        metadata = {
            "argv": argv,
            "cwd": str(cwd or self.runtime),
            "environment_overrides": overrides,
            "started_at": now(),
            "inherited_keys": sorted(set(env) - set(overrides)),
        }
        meta_path = attempt / f"{label}.command.json"
        write_json(meta_path, metadata)
        with ExitStack() as streams:
            log = streams.enter_context((attempt / f"{label}.log").open("xb"))
            output = log
            if stdout_file is not None:
                if Path(stdout_file).name != stdout_file:
                    raise ExportError("Command stdout destination must be a filename in its attempt")
                output = streams.enter_context((attempt / stdout_file).open("xb"))
            child = subprocess.Popen(
                argv,
                cwd=cwd or self.runtime,
                env=env,
                stdout=output,
                stderr=log if stdout_file is not None else subprocess.STDOUT,
                start_new_session=True,
            )
            metadata["pid"] = child.pid
            write_json(meta_path, metadata)
            try:
                code = child.wait(timeout=self.config["command_timeout_seconds"])
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                # Signal only our new process group. The safe wrapper gets its
                # normal cleanup/dirty-device handling before forced termination.
                try:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                finally:
                    metadata.update(returncode=child.poll(), interrupted=True, finished_at=now())
                    write_json(meta_path, metadata)
                raise
        metadata.update(returncode=code, finished_at=now())
        write_json(meta_path, metadata)
        if code not in allowed:
            raise ExportError(f"{label} exited {code}; see {attempt / (label + '.log')}")
        return code

    def execute(self, stage, attempt):
        config, planned = self.config, self.planned
        if stage == "prepare":
            result = prepare_baseline.prepare(
                config["export"],
                config["eval_repository"],
                config["metal_repository"],
                self.prepared,
            )
            return {"preparation_sha256": result["preparation_sha256"]}, [self.prepared / "baseline.json"]
        if stage == "checkout":
            if not self.runtime.exists():
                self.command(
                    [
                        "git",
                        "worktree",
                        "add",
                        "--detach",
                        str(self.runtime),
                        planned["metal_revision"],
                    ],
                    attempt,
                    "worktree",
                    cwd=Path(config["metal_repository"]),
                )
            self.head(self.runtime, planned["metal_revision"])
            if self.git(self.runtime, "rev-parse", "--path-format=absolute", "--git-common-dir") != self.git(
                config["metal_repository"],
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ):
                raise ExportError("Runtime is not a worktree of the configured repository")
            self.command(
                ["git", "submodule", "update", "--init", "--recursive"],
                attempt,
                "submodules",
            )
            return {"revision": planned["metal_revision"]}, []
        if stage == "evaluator":
            evaluator = self.runtime / config["evaluator_path"]
            if not evaluator.resolve().is_relative_to(self.runtime):
                raise ExportError("Evaluator path leaves runtime")
            if (evaluator / ".git").exists():
                if self.git(evaluator, "status", "--porcelain"):
                    raise ExportError("Evaluator checkout is dirty; refusing to change revision")
                self.command(
                    [
                        "git",
                        "fetch",
                        "--no-tags",
                        config["eval_repository"],
                        planned["eval_revision"],
                    ],
                    attempt,
                    "eval-objects",
                    cwd=evaluator,
                )
                self.command(
                    ["git", "checkout", "--detach", planned["eval_revision"]],
                    attempt,
                    "eval-checkout",
                    cwd=evaluator,
                )
            else:
                if evaluator.exists() and any(evaluator.iterdir()):
                    raise ExportError("Evaluator directory is occupied")
                evaluator.parent.mkdir(parents=True, exist_ok=True)
                self.command(
                    [
                        "git",
                        "worktree",
                        "add",
                        "--detach",
                        str(evaluator),
                        planned["eval_revision"],
                    ],
                    attempt,
                    "eval-worktree",
                    cwd=Path(config["eval_repository"]),
                )
            self.command(
                ["git", "submodule", "update", "--init", "--recursive"],
                attempt,
                "eval-submodules",
                cwd=evaluator,
            )
            link = self.runtime / "eval"
            if not link.exists() and not link.is_symlink():
                link.symlink_to(Path(config["evaluator_path"]) / "eval", target_is_directory=True)
            if link.resolve() != (evaluator / "eval").resolve():
                raise ExportError("Runtime eval path does not select the pinned evaluator")
            self.head(evaluator, planned["eval_revision"])
            return {"submodules": self.git(self.runtime, "submodule", "status", "--recursive")}, []
        if stage == "install":
            substitutions = planned["dependency_substitutions"]
            dependency_substitutions.check_declared_headers(
                substitutions, self.runtime, prepare_baseline.GitTree(self.runtime, planned["metal_revision"])
            )
            dependency_substitutions.check_runtime(substitutions, self.runtime, installed=False)
            result = prepare_baseline.install(self.prepared, self.runtime)
            dependency_substitutions.install(substitutions, self.runtime)
            result["dependency_substitutions"] = substitutions
            result["baseline_scope"] = planned["baseline_scope"]
            self.source_audit()
            return result, []
        if stage == "environment":
            if (self.runtime / "python_env").exists():
                raise ExportError("Environment already exists without a completed checkpoint; use a new workspace")
            self.command(["./create_venv.sh"], attempt, "create-venv")
            return {}, [self.runtime / "python_env/bin/activate"]
        if stage == "build":
            if config["sfpi_directory"]:
                link = self.runtime / "runtime/sfpi"
                if not link.parent.resolve().is_relative_to(self.runtime):
                    raise ExportError("Runtime toolchain directory is redirected")
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(config["sfpi_directory"], target_is_directory=True)
                if not link.is_symlink() or link.resolve() != Path(config["sfpi_directory"]):
                    raise ExportError("SFPI location is occupied by a different toolchain")
            if config["configure_argv"]:
                self.command(config["configure_argv"], attempt, "configure", activate=True)
            self.command(config["build_argv"], attempt, "build", activate=True)
            self.command(["python3", "-c", PROBE], attempt, "runtime-probe", activate=True)
            probe = self.parse_probe((attempt / "runtime-probe.log").read_text())
            write_json(attempt / "runtime.json", probe)
            outputs = [Path(path) for path in probe["paths"].values()]
            outputs.extend(path for path in (self.runtime / "build_Release/lib").glob("*.so*") if path.is_file())
            cache = self.runtime / "build_Release/CMakeCache.txt"
            if cache.exists():
                outputs.append(cache)
            self.source_audit()
            return {"runtime": "local import verified"}, outputs
        if stage in ("collect", "smoke", "baseline"):
            if stage == "smoke" and not config["smoke_nodeid"]:
                return {
                    "skipped": True,
                    "reason": "No smoke node configured; full baseline still required",
                }, []
            self.source_audit()
            test_evidence.check_runner(
                self.runtime / "scripts/run_safe_pytest.sh",
                precompile=stage == "baseline" and config["precompile"],
                require_raw=False,
            )
            argv = ["./scripts/run_safe_pytest.sh", "--run-all"]
            if stage == "baseline" and config["precompile"]:
                argv += [
                    "--precompile",
                    "--precompile-workers",
                    str(config["precompile_workers"]),
                ]
            else:
                argv += ["--no-precompile"]
            argv += [
                (config["smoke_nodeid"] if stage == "smoke" else f"eval/golden_tests/{planned['golden_suite']}"),
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
            if stage == "collect":
                argv += ["--collect-only"]
            else:
                argv += [f"--junitxml={attempt / 'junit.xml'}"]
            code = self.command(
                argv,
                attempt,
                stage,
                activate=True,
                allowed=(0, 1) if stage == "baseline" else (0,),
                device=stage != "collect",
            )
            if "SAFE_PYTEST_RESULT: HANG" in (attempt / f"{stage}.log").read_text():
                raise ExportError("Safe runner reported a device hang")
            self.source_audit()
            if stage != "collect":
                observed = test_evidence.verify(attempt / f"{stage}.log", attempt / "junit.xml", require_raw=False)
                write_json(attempt / "execution.json", observed)
                rows = classify_failures.parse_junit_xml(attempt / "junit.xml")
                if not rows:
                    raise ExportError("No test outcomes were produced")
                return {
                    "returncode": code,
                    "junit": str((attempt / "junit.xml").relative_to(self.workspace)),
                    "case_count": len(rows),
                }, []
            return {"returncode": code}, []
        if stage == "compare":
            junit = self.workspace / self.receipt("baseline")["result"]["junit"]
            result = compare_baseline.compare(config["export"], junit, config["phase"])
            result["baseline_scope"] = planned["baseline_scope"]
            result["dependency_substitutions"] = planned["dependency_substitutions"]
            write_json(attempt / "comparison.json", result)
            if not result["outcomes_match"]:
                raise ExportError(
                    "Baseline differs from selected historical phase; comparison.json records the differences"
                )
            return {
                "outcomes_match": True,
                "observed_failures": result["observed_failures"],
                "baseline_scope": planned["baseline_scope"],
                "migration_ready": False,
            }, []
        raise ExportError(f"Stage is not implemented: {stage}")

    def run(self, through="compare", retry=False):
        if through not in STAGES:
            raise ExportError("Only the recorded-baseline milestone is implemented")
        with locked(self.workspace):
            # Reload after acquiring the lock, not from a potentially stale instance.
            self.state = json.loads((self.workspace / "state.json").read_bytes())
            self.validate()
            for stage in STAGES[: STAGES.index(through) + 1]:
                record = self.state["stages"][stage]
                if record["status"] == "complete":
                    continue
                if record["status"] != "pending":
                    if not retry:
                        raise ExportError(
                            f"Stage {stage} is {record['status']}; inspect evidence, then explicitly --retry"
                        )
                    for old in record["attempts"]:
                        for path in (self.workspace / old).glob("*.command.json"):
                            command = json.loads(path.read_bytes())
                            if "finished_at" not in command:
                                if not command.get("pid"):
                                    raise ExportError(
                                        "An unfinished command has unknown process identity; refusing to retry"
                                    )
                                try:
                                    os.kill(command["pid"], 0)
                                except ProcessLookupError:
                                    pass
                                else:
                                    raise ExportError("An unfinished command may still be alive; refusing to retry")
                attempt = self.workspace / "attempts" / stage / f"{len(record['attempts']) + 1:03d}"
                if not attempt.resolve().is_relative_to(self.workspace):
                    raise ExportError("Attempt directory is redirected")
                attempt.mkdir(parents=True)
                record["attempts"].append(str(attempt.relative_to(self.workspace)))
                record.update(status="running", started_at=now())
                self.save()
                print(f"WORKFLOW: {stage} started ({attempt})", flush=True)
                try:
                    result, outputs = self.execute(stage, attempt)
                    outputs.extend(path for path in attempt.iterdir() if path.is_file())
                    evidence = [self.fingerprint(path) for path in sorted(set(outputs))]
                    receipt = attempt / "receipt.json"
                    write_json(
                        receipt,
                        {"stage": stage, "result": result, "evidence": evidence},
                    )
                    record.update(
                        status="complete",
                        receipt=str(receipt.relative_to(self.workspace)),
                        receipt_sha256=_hash_file(receipt)[0],
                        finished_at=now(),
                    )
                    record.pop("error", None)
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
                    self.save()
                    raise
                self.save()
                print(f"WORKFLOW: {stage} complete", flush=True)
            return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "init"):
        sub.add_parser(action).add_argument("--config", type=Path, required=True)
    for action in ("status", "run"):
        command = sub.add_parser(action)
        command.add_argument("--workspace", type=Path, required=True)
        if action == "run":
            command.add_argument("--through", choices=STAGES, default="compare")
            command.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    try:
        if args.action in ("plan", "init"):
            raw = json.loads(args.config.read_bytes())
            result = plan(raw) if args.action == "plan" else initialize(raw)
        elif args.action == "status":
            result = json.loads((args.workspace / "state.json").read_bytes())
        else:
            signal.signal(signal.SIGTERM, _interrupted)
            result = Workflow(args.workspace).run(args.through, args.retry)
        print(json.dumps(result, indent=2, sort_keys=True))
    except KeyboardInterrupt:
        parser.exit(130, "Workflow interrupted; inspect saved attempt before retrying\n")
    except (
        ExportError,
        OSError,
        ValueError,
        KeyError,
        subprocess.SubprocessError,
    ) as error:
        parser.exit(2, f"Workflow blocked: {error}\n")


if __name__ == "__main__":
    main()
