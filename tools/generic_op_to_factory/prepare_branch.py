# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Prepare migration from a complete, checkpointed evaluated branch, never DB source.

The selected branch is resolved once. A fresh detached worktree retains its full
tree and recursive gitlinks; preparation never commits, fetches, resets or
rewrites the evaluated checkout. Dirty evaluated worktrees require a checkpoint.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tools.generic_op_to_factory import test_evidence
from tools.generic_op_to_factory.export_run import ExportError, _hash_file, _safe_path, json_bytes
from tools.generic_op_to_factory.prepare_baseline import GitTree

FORMAT_VERSION = 1
ARTIFACT_DIR = "generated/generic_op_to_factory"
ARTIFACT_IGNORE = "# Local migration evidence and caches; never operation source.\n*\n"
TOOL_FILES = {
    "scripts/run_safe_pytest.sh": "RUNNER_SHA256",
    "tools/generic_op_to_factory/native_adapter.py": "ADAPTER_SHA256",
}


def git(repository, *args):
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repository), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise ExportError(f"Git {args[0]} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def absolute(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ExportError("Use absolute, unredirected paths")
    return path


def artifact_path(runtime, path):
    """Reserve a worktree-local namespace; never admit source paths as outputs."""
    root = absolute(runtime) / ARTIFACT_DIR
    path = absolute(path)
    if path == root or not path.is_relative_to(root):
        raise ExportError(f"Migration artifacts must be inside {root}/<directory>")
    return path


def check_artifacts(runtime):
    runtime = absolute(runtime)
    root = absolute(runtime / ARTIFACT_DIR)
    if git(runtime, "ls-files", "--cached", "--", ARTIFACT_DIR).strip():
        raise ExportError("Migration artifact directory cannot contain tracked source")
    marker = root / ".gitignore"
    if marker.is_symlink() or not marker.is_file() or marker.read_text() != ARTIFACT_IGNORE:
        raise ExportError("Migration artifact ignore marker is missing or changed")


def initialize_artifacts(runtime):
    """Create only a local ignore marker, without editing repository Git config."""
    runtime = absolute(runtime)
    root = absolute(runtime / ARTIFACT_DIR)
    if git(runtime, "ls-files", "--cached", "--", ARTIFACT_DIR).strip():
        raise ExportError("Migration artifact directory cannot contain tracked source")
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".gitignore"
    if not marker.exists() and not marker.is_symlink():
        with marker.open("x") as stream:
            stream.write(ARTIFACT_IGNORE)
    check_artifacts(runtime)


def resolve_branch(repository, branch):
    repository = absolute(repository)
    if not isinstance(branch, str) or not branch or branch.startswith("-"):
        raise ExportError("An explicit local or remote-tracking branch is required")
    refs = [branch] if branch.startswith("refs/") else [f"refs/heads/{branch}", f"refs/remotes/{branch}"]
    available = set(
        git(repository, "for-each-ref", "--format=%(refname)", "refs/heads/", "refs/remotes/").decode().splitlines()
    )
    matches = [ref for ref in refs if ref in available]
    if len(matches) != 1:
        raise ExportError("Branch is absent or ambiguous; supply an exact ref (fetch explicitly if needed)")
    ref = matches[0]
    revision = git(repository, "rev-parse", "--verify", ref + "^{commit}").decode().strip()
    # A branch pointer does not capture an agent's uncommitted supporting changes.
    # Check every local worktree that currently represents this branch/commit.
    for record in git(repository, "worktree", "list", "--porcelain").decode().strip().split("\n\n"):
        fields = dict(line.split(" ", 1) for line in record.splitlines() if " " in line)
        if fields.get("branch") == ref or fields.get("HEAD") == revision:
            checkout = Path(fields["worktree"])
            if not checkout.is_dir():
                raise ExportError("Evaluated worktree is unavailable; resolve it before preparing migration")
            if git(checkout, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none").strip():
                raise ExportError(
                    "Evaluated worktree is dirty; checkpoint all run changes and submodule revisions first"
                )
    return ref, revision


def submodules(runtime, revision):
    """Verify initialized, clean recursive submodules at the branch's own gitlinks."""
    result = {}
    pending = [(Path(runtime), revision)]
    while pending:
        repository, commit = pending.pop()
        tree = GitTree(repository, commit)
        for relative, (mode, _, oid) in tree.entries.items():
            if mode != "160000":
                continue
            child = repository / _safe_path(relative)
            if child.resolve() != child or not (child / ".git").exists():
                raise ExportError(f"Submodule is uninitialized or redirected: {child}")
            if git(child, "rev-parse", "HEAD").decode().strip() != oid:
                raise ExportError(f"Submodule differs from evaluated branch gitlink: {child}")
            if git(child, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none").strip():
                raise ExportError(f"Evaluated submodule has working-tree changes: {child}")
            result[str(child.relative_to(runtime))] = oid
            pending.append((child, oid))
    return dict(sorted(result.items()))


def payload_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_snapshot(directory):
    directory = absolute(directory)
    path = directory / "branch.json"
    if path.is_symlink():
        raise ExportError("Branch snapshot is redirected")
    record = json.loads(path.read_bytes())
    payload = {key: value for key, value in record.items() if key != "snapshot_sha256"}
    if record.get("format_version") != FORMAT_VERSION or record.get("snapshot_sha256") != payload_hash(payload):
        raise ExportError("Evaluated branch snapshot checksum/version mismatch")
    return record


def inspect(directory, runtime, migration_paths=()):
    """Admit only explicitly named migration edits; keep source and tests frozen."""
    runtime = absolute(runtime)
    artifact_path(runtime, directory)
    check_artifacts(runtime)
    record = read_snapshot(directory)
    if str(runtime) != record["runtime"]:
        raise ExportError("Runtime differs from the evaluated branch snapshot")
    base = record["source_revision"]
    tree = GitTree(runtime, base)
    if any(path == ARTIFACT_DIR or path.startswith(ARTIFACT_DIR + "/") for path in tree.entries):
        raise ExportError("Evaluated checkpoint uses the reserved migration artifact directory")
    git(runtime, "merge-base", "--is-ancestor", base, "HEAD")
    paths = list(migration_paths)
    if any(not isinstance(path, str) for path in paths) or len(set(paths)) != len(paths):
        raise ExportError("migration_paths must contain unique repository-relative file paths")
    protected = (record["operation_path"], "eval", *record["submodules"])
    for path in paths:
        _safe_path(path)
        if (
            path in (".gitmodules", ".gitignore", "conftest.py", "pytest.ini", "build_metal.sh", "create_venv.sh")
            or (path.startswith("tests/") and path in tree.entries)
            or path.startswith(".git/")
            or path == ".git"
            or path == ARTIFACT_DIR
            or path.startswith(ARTIFACT_DIR + "/")
            or any(path == prefix or path.startswith(prefix + "/") for prefix in protected)
        ):
            raise ExportError(f"Migration edits cannot replace evaluated source, tests or runtime setup: {path}")
        target = runtime / path
        if target.is_symlink() or target.resolve() != target or target.is_dir():
            raise ExportError(f"migration_paths must name unredirected files, not directories: {path}")
    if submodules(runtime, base) != record["submodules"]:
        raise ExportError("Evaluated submodule inventory changed")
    changed = set(filter(None, tree.git("diff", "--no-renames", "--name-only", "-z", base, "--").decode().split("\0")))
    changed.update(filter(None, tree.git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")))
    # Tooling updates are explicit, separately hash-checked exceptions, not
    # permission to substitute an SDK helper or install source from an export.
    tooling = {}
    for path in sorted(changed):
        actual = runtime / _safe_path(path)
        if path in TOOL_FILES:
            if (
                actual.resolve() != actual
                or not actual.is_file()
                or _hash_file(actual)[0] != getattr(test_evidence, TOOL_FILES[path])
            ):
                raise ExportError(f"Validation tool differs from this flow's reviewed version: {path}")
            tooling[path] = _hash_file(actual)[0]
        elif path not in paths:
            raise ExportError(
                f"Undeclared change since evaluated checkpoint: {path}; list native edits in migration_paths"
            )
    suite = runtime / record["suite"]
    if not suite.is_dir() or not suite.resolve().is_relative_to(runtime):
        raise ExportError("Evaluated golden suite is absent or points outside the runtime")
    return {
        **record,
        "target_revision": git(runtime, "rev-parse", "HEAD").decode().strip(),
        "tooling_updates": tooling,
        "dependency_substitutions": [],
        "baseline_scope": "Complete evaluated branch checkpoint and its gitlinks; fresh source/native comparison on one build",
        "migration_ready": False,
    }


def install_tools(runtime):
    """Install only reviewed test tooling into the new migration worktree."""
    runtime = absolute(runtime)
    driver = Path(__file__).resolve().parents[2]
    for relative, pin in TOOL_FILES.items():
        source, target = driver / relative, runtime / relative
        if _hash_file(source)[0] != getattr(test_evidence, pin):
            raise ExportError(f"Driver tool does not match its reviewed pin: {relative}")
        if target.resolve() != target or (target.exists() and not target.is_file()):
            raise ExportError(f"Validation tool destination is redirected: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        target.chmod(source.stat().st_mode & 0o777)


def prepare(repository, branch, runtime, operation, golden_suite, output, *, with_validation_tools=False):
    repository, runtime, output = absolute(repository), absolute(runtime), absolute(output)
    for name in (operation, golden_suite):
        if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
            raise ExportError("Operation and golden suite must be Python identifiers")
    if runtime.exists() or runtime.is_symlink() or output.exists() or output.is_symlink():
        raise ExportError("Runtime and evidence destinations must be new")
    artifact_path(runtime, output)
    if runtime.is_relative_to(repository) or repository.is_relative_to(runtime):
        raise ExportError("Use a target worktree separate from the source repository")
    ref, revision = resolve_branch(repository, branch)
    tree = GitTree(repository, revision)
    if any(path == ARTIFACT_DIR or path.startswith(ARTIFACT_DIR + "/") for path in tree.entries):
        raise ExportError("Evaluated checkpoint uses the reserved migration artifact directory")
    operation_path = f"ttnn/ttnn/operations/{operation}"
    if not any(path.startswith(operation_path + "/") for path in tree.entries):
        raise ExportError("Operation is absent from evaluated branch; checkpoint run changes, do not install DB source")
    # Git requires a new/empty target. Capture its first command before creating
    # worktree-local evidence; subsequent command logs stream directly to disk.
    commands = []

    def run(cwd, args):
        log = output / f"command-{len(commands) + 1}.log"
        commands.append({"cwd": str(cwd), "argv": args, "log": str(log)})
        if not output.exists():
            result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                # No usable worktree may exist. Report Git's diagnostic without
                # creating a misleading target or scattering logs in its parent.
                raise ExportError(f"Worktree creation failed: {result.stdout.decode(errors='replace')}")
            initialize_artifacts(runtime)
            output.mkdir(parents=True)
            log.write_bytes(result.stdout)
        else:
            (output / "commands.json").write_bytes(json_bytes(commands))
            with log.open("wb") as stream:
                result = subprocess.run(args, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        commands[-1]["exit_code"] = result.returncode
        (output / "commands.json").write_bytes(json_bytes(commands))
        if result.returncode:
            raise ExportError(f"Branch preparation failed; inspect {log}; partial worktree retained")

    run(repository, ["git", "--no-replace-objects", "worktree", "add", "--detach", str(runtime), revision])
    # --checkout overrides an evaluated .gitmodules 'update = none': initialize
    # every recorded gitlink, never substitute the DB's evaluator/start revision.
    run(runtime, ["git", "--no-replace-objects", "submodule", "update", "--init", "--recursive", "--checkout"])
    modules = submodules(runtime, revision)
    for relative in tree.entries:
        if relative.startswith(operation_path + "/") and not (runtime / relative).resolve().is_relative_to(runtime):
            raise ExportError(f"Evaluated operation source points outside its checkpoint: {relative}")
    suite = f"eval/golden_tests/{golden_suite}"
    suite_path = runtime / suite
    if not suite_path.is_dir() or not suite_path.resolve().is_relative_to(runtime):
        raise ExportError(
            "Golden suite is missing from evaluated tree/gitlinks; no historical evaluator will be substituted"
        )
    if git(runtime, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none").strip():
        raise ExportError("Prepared evaluated runtime is not clean")
    record = {
        "format_version": FORMAT_VERSION,
        "input_mode": "evaluated_branch",
        "source_repository": str(repository),
        "source_branch": ref,
        "source_revision": revision,
        "runtime": str(runtime),
        "operation": operation,
        "operation_path": operation_path,
        "suite": suite,
        "submodules": modules,
    }
    record["snapshot_sha256"] = payload_hash(record)
    (output / "branch.json").write_bytes(json_bytes(record))
    if with_validation_tools:
        install_tools(runtime)
    return inspect(output, runtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--golden-suite", required=True)
    parser.add_argument("--output", type=Path, help=f"Default: RUNTIME/{ARTIFACT_DIR}/inputs")
    parser.add_argument(
        "--with-validation-tools",
        action="store_true",
        help="Install this flow's reviewed safe runner and test-only route adapter into the new worktree",
    )
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                prepare(
                    args.repository,
                    args.branch,
                    args.runtime,
                    args.operation,
                    args.golden_suite,
                    args.output or args.runtime / ARTIFACT_DIR / "inputs",
                    with_validation_tools=args.with_validation_tools,
                ),
                indent=2,
            )
        )
    except (ExportError, OSError, ValueError) as error:
        parser.exit(2, f"Evaluated branch preparation blocked: {error}\n")


if __name__ == "__main__":
    main()
