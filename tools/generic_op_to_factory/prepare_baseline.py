# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prepare pinned baseline inputs without executing source or changing a checkout.

This is source preparation, not proof of historical runtime equivalence. The
recorded starting_commit pins the runtime; eval_commit pins the selected suite.
"""

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.generic_op_to_factory.export_run import (
    ExportError,
    _file_entry,
    _safe_path,
    _validate_paths,
    _write_bytes,
    json_bytes,
    verify_export,
)


class GitTree:
    def __init__(self, repository, revision):
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ExportError("A full recorded Git commit is required")
        self.repository = repository
        self.revision = revision
        if self.git("cat-file", "-t", revision).strip() != b"commit":
            raise ExportError("Recorded revision must identify a commit")
        self.entries = {}
        for entry in self.git("ls-tree", "-r", "-z", revision).split(b"\0"):
            if entry:
                metadata, name = entry.split(b"\t", 1)
                mode, kind, oid = metadata.decode().split()
                self.entries[name.decode()] = (mode, kind, oid)

    def git(self, *arguments):
        try:
            return subprocess.run(
                ["git", "--no-replace-objects", "-C", str(self.repository), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError as error:
            raise ExportError("Required Git object is unavailable in the supplied repository") from error

    def read(self, path):
        _safe_path(path)
        mode, kind, oid = self.entries.get(path, (None, None, None))
        if mode not in ("100644", "100755") or kind != "blob":
            raise ExportError(f"Required regular file is absent at the recorded revision: {path}")
        return self.git("cat-file", "blob", oid)


def eval_closure(tree, suite):
    """Follow static eval imports, retaining package initializers and conftests.

    Other Python packages and dynamic imports are inventoried, not claimed closed.
    All selected suite files are retained, including non-Python test data.
    """
    prefix = f"eval/golden_tests/{suite}/"
    pending = {path for path in tree.entries if path.startswith(prefix)}
    if not pending:
        raise ExportError("Recorded golden suite is absent")
    pending.update(
        (
            "eval/__init__.py",
            "eval/golden_tests/__init__.py",
            "eval/golden_tests/conftest.py",
        )
    )
    pending.update(f"eval/{name}.py" for name in ("hang_plugin", "metrics_plugin", "axes_plugin"))
    files, imports, dynamic = {}, set(), []

    def add_module(module, required):
        parts = module.split(".")
        path = "/".join(parts)
        candidates = (path + ".py", path + "/__init__.py")
        found = next((candidate for candidate in candidates if candidate in tree.entries), None)
        if not found:
            if required:
                raise ExportError(f"Unresolved local eval import: {module}")
            return
        pending.add(found)
        for count in range(1, len(parts)):
            initializer = "/".join(parts[:count]) + "/__init__.py"
            if initializer in tree.entries:
                pending.add(initializer)

    while pending:
        path = min(pending)
        pending.remove(path)
        if path in files:
            continue
        raw = files[path] = tree.read(path)
        if not path.endswith(".py"):
            continue
        try:
            syntax = ast.parse(raw, filename=path)
        except SyntaxError as error:
            raise ExportError(f"Cannot inventory Python imports: {path}") from error
        for node in ast.walk(syntax):
            modules = []
            if isinstance(node, ast.Import):
                modules = [(alias.name, []) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = path.split("/")[:-1]
                    if node.level > len(package):
                        raise ExportError(f"Relative import escapes package: {path}")
                    module = ".".join(package[: len(package) - node.level + 1] + ([module] if module else []))
                modules = [(module, [alias.name for alias in node.names])]
            elif isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                if name in ("__import__", "import_module", "exec", "eval"):
                    dynamic.append({"path": path, "line": node.lineno, "call": name})
            for module, members in modules:
                if module == "eval" or module.startswith("eval."):
                    add_module(module, True)
                    for member in members:
                        if member != "*":
                            add_module(module + "." + member, False)
                else:
                    imports.add(module)
    return (
        files,
        sorted(imports),
        sorted(dynamic, key=lambda item: (item["path"], item["line"])),
    )


def prepare(export, eval_repository, metal_repository, output):
    export, output = Path(export), Path(output)
    original = verify_export(export)
    run = json.loads((export / "records/run.json").read_bytes())
    operation, suite = run.get("prompt_name"), run.get("golden_name")
    for name in (operation, suite):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name, flags=re.ASCII):
            raise ExportError("Operation and golden-suite names must be Python identifiers")
    if output.exists() or output.is_symlink():
        raise ExportError("Output already exists")
    evaluator = GitTree(eval_repository, run.get("eval_commit"))
    metal = GitTree(metal_repository, run.get("starting_commit"))
    golden, imports, dynamic = eval_closure(evaluator, suite)
    payload, metadata = {}, {}

    def add(destination, raw, **origin):
        _safe_path(destination)
        if destination in payload:
            raise ExportError(f"Duplicate preparation path: {destination}")
        payload[destination], metadata[destination] = raw, origin

    for entry in original["files"]:
        if entry.get("materialized") and entry["table"] in ("host_code", "kernels"):
            relative = entry["path"].removeprefix("source/")
            destination = f"overlay/ttnn/ttnn/operations/{operation}/{relative}"
            add(
                destination,
                (export / entry["path"]).read_bytes(),
                origin="db",
                export_path=entry["path"],
            )
    for path, raw in golden.items():
        add(
            "overlay/" + path,
            raw,
            origin="eval",
            commit=evaluator.revision,
            git_path=path,
        )
    # Preserve the canonical helper family as-is, not a rewritten or purported
    # minimal include closure. Platform/API headers remain in the pinned runtime.
    helper_paths = [path for path in metal.entries if path.startswith("ttnn/cpp/ttnn/kernel_lib/")]
    if not helper_paths:
        raise ExportError("Canonical kernel helper library is absent")
    reference_paths = helper_paths + [
        "ttnn/ttnn/operations/_op_contract.py",
        "conftest.py",
        "pytest.ini",
        "scripts/run_safe_pytest.sh",
        ".gitmodules",
    ]
    for path in reference_paths:
        add(
            "reference/metal/" + path,
            metal.read(path),
            origin="metal",
            commit=metal.revision,
            git_path=path,
        )
    for path in ("eval/run_eval.py", "eval/eval_test_runner.sh"):
        add(
            "reference/evaluator/" + path,
            evaluator.read(path),
            origin="eval",
            commit=evaluator.revision,
            git_path=path,
        )
    gitlinks = {path: oid for path, (mode, _, oid) in metal.entries.items() if mode == "160000"}
    manifest = {
        "format_version": 1,
        "input_snapshot_sha256": original["snapshot_sha256"],
        "input_source_sha256": original["source_sha256"],
        "run_id": run["id"],
        "operation": operation,
        "golden_suite": suite,
        "metal_commit": metal.revision,
        "eval_commit": evaluator.revision,
        "parent_gitlinks": gitlinks,
        "eval_revision_selection": "recorded eval_commit; parent gitlink is retained separately, not silently substituted",
        "historical_status": {
            key: run.get(key) for key in ("status", "failure_reason", "golden_passed", "golden_total")
        },
        "recorded_runtime": {
            key: run.get(key) for key in ("arch", "runtime_backend", "target_spec", "op_metadata_json")
        },
        "external_python_imports": imports,
        "dynamic_python_calls": dynamic,
        "dependency_scope": "static eval imports plus entire canonical kernel_lib; full pinned tt-metal build and submodules still required",
        "baseline_reproduced": False,
        "migration_ready": False,
        "unresolved": [
            "Confirm loaded runtime, build configuration, submodules, Python packages and device architecture",
            "Audit dynamic imports, runtime file accesses and dependencies outside eval; static import inventory is not a closure proof",
            "Original full environment and candidate-phase association are not recorded; compare a new baseline without claiming exact historical replay",
            "Run the complete pinned golden suite; preserve failures, refusals, skips and incomplete execution separately",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".baseline-", dir=output.parent) as staging:
        package = Path(staging) / "package"
        package.mkdir()
        for path, raw in sorted(payload.items()):
            _write_bytes(package, path, raw)
        manifest["files"] = [_file_entry(package, path, **metadata[path]) for path in sorted(payload)]
        manifest["preparation_sha256"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
        _write_bytes(package, "baseline.json", json_bytes(manifest))
        try:
            output.mkdir()
        except FileExistsError as error:
            raise ExportError("Output already exists") from error
        try:
            package.rename(output)
        except OSError:
            output.rmdir()
            raise
    return manifest


def verify_preparation(package):
    package = Path(package)
    if package.is_symlink() or not package.is_dir():
        raise ExportError("Preparation must be a real directory")
    try:
        manifest = json.loads((package / "baseline.json").read_bytes())
        unsigned = {key: value for key, value in manifest.items() if key != "preparation_sha256"}
        if (
            manifest["format_version"] != 1
            or hashlib.sha256(json_bytes(unsigned)).hexdigest() != manifest["preparation_sha256"]
        ):
            raise ExportError("Preparation manifest checksum/version mismatch")
        _validate_paths([entry["path"] for entry in manifest["files"]])
        expected = {entry["path"] for entry in manifest["files"]} | {"baseline.json"}
        actual = set()
        for path in package.rglob("*"):
            if path.is_symlink():
                raise ExportError("Preparation contains a symlink")
            if path.is_file():
                actual.add(path.relative_to(package).as_posix())
        if actual != expected:
            raise ExportError("Preparation has missing or unexpected files")
        for entry in manifest["files"]:
            observed = _file_entry(package, entry["path"])
            if any(observed[key] != entry[key] for key in ("sha256", "size_bytes")):
                raise ExportError(f"Preparation file checksum mismatch: {entry['path']}")
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise ExportError("Invalid preparation package") from error
    return manifest


def install(package, runtime):
    """Install only absent operation source into an explicit pinned runtime.

    Golden/eval and canonical runtime files must already match their pinned
    inputs. Never overwrite source, change revisions, build or execute tests.
    """
    package, runtime = Path(package), Path(runtime).resolve()
    manifest = verify_preparation(package)
    operation = manifest["operation"]
    if not re.fullmatch(r"[A-Za-z_]\w*", operation, flags=re.ASCII):
        raise ExportError("Invalid operation name")
    metal = GitTree(runtime, manifest["metal_commit"])
    if metal.git("rev-parse", "HEAD").decode().strip() != manifest["metal_commit"]:
        raise ExportError("Runtime HEAD differs from recorded starting_commit")
    if metal.git("diff", "--name-only", "--ignore-submodules=all", "HEAD", "--").strip():
        raise ExportError("Runtime has modified tracked files")
    eval_root = runtime / "eval"
    evaluator = GitTree(eval_root, manifest["eval_commit"])
    if evaluator.git("rev-parse", "HEAD").decode().strip() != manifest["eval_commit"]:
        raise ExportError("Runtime evaluator HEAD differs from recorded eval_commit")
    for entry in manifest["files"]:
        path = entry["path"]
        if path.startswith("reference/metal/"):
            installed = runtime / path.removeprefix("reference/metal/")
        elif path.startswith("overlay/eval/"):
            installed = runtime / path.removeprefix("overlay/")
        else:
            continue
        if not installed.is_file() or installed.read_bytes() != (package / path).read_bytes():
            raise ExportError(f"Runtime dependency differs from pinned input: {path}")
    parent = runtime / "ttnn/ttnn/operations"
    if parent.resolve() != parent or not parent.is_dir():
        raise ExportError("Runtime operation directory is missing or redirected")
    destination = parent / operation
    if destination.exists() or destination.is_symlink():
        raise ExportError("Operation destination already exists; refusing to overwrite")
    source = package / "overlay/ttnn/ttnn/operations" / operation
    with tempfile.TemporaryDirectory(prefix=".baseline-install-", dir=runtime.parent) as staging:
        staged = Path(staging) / "operation"
        shutil.copytree(source, staged)
        try:
            destination.mkdir()
        except FileExistsError as error:
            raise ExportError("Operation destination already exists") from error
        try:
            staged.rename(destination)
        except OSError:
            destination.rmdir()
            raise
    return {
        "operation": operation,
        "runtime": str(runtime),
        "preparation_sha256": manifest["preparation_sha256"],
        "baseline_reproduced": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export", type=Path)
    action.add_argument("--verify", type=Path)
    action.add_argument("--install", type=Path)
    parser.add_argument("--eval-repository", type=Path)
    parser.add_argument("--metal-repository", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.export and not all((args.eval_repository, args.metal_repository, args.output)):
        parser.error("--export requires --eval-repository, --metal-repository and --output")
    if args.install and not args.runtime:
        parser.error("--install requires --runtime")
    try:
        if args.install:
            print(json.dumps(install(args.install, args.runtime)))
            return
        if args.verify:
            manifest = verify_preparation(args.verify)
        else:
            manifest = prepare(args.export, args.eval_repository, args.metal_repository, args.output)
    except (ExportError, OSError) as error:
        parser.exit(1, f"Baseline preparation failed: {error}\n")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "run_id",
                    "preparation_sha256",
                    "baseline_reproduced",
                    "migration_ready",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
