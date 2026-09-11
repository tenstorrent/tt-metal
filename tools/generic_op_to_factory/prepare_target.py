# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Install a verified source candidate on an explicit target, reporting helper drift.

Does not translate, overwrite helpers, or assert target baseline equivalence.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from tools.generic_op_to_factory.export_run import ExportError, _hash_file, _safe_path, json_bytes
from tools.generic_op_to_factory.prepare_baseline import GitTree, verify_preparation


def inspect(package, runtime, revision, *, allow_installed=False):
    package, runtime = Path(package), Path(runtime).resolve()
    manifest = verify_preparation(package)
    metal = GitTree(runtime, revision)
    if metal.git("rev-parse", "HEAD").decode().strip() != revision:
        raise ExportError("Target HEAD differs from explicit target revision")
    evaluator = GitTree(runtime / "eval", manifest["eval_commit"])
    if evaluator.git("rev-parse", "HEAD").decode().strip() != manifest["eval_commit"]:
        raise ExportError("Target evaluator differs from recorded eval revision")
    dependencies = []
    for entry in manifest["files"]:
        path = entry["path"]
        if path.startswith("overlay/eval/"):
            actual = runtime / path.removeprefix("overlay/")
            if not actual.is_file() or _hash_file(actual)[0] != entry["sha256"]:
                raise ExportError(f"Pinned golden/harness file changed: {actual}")
        elif path.startswith("reference/metal/"):
            relative = path.removeprefix("reference/metal/")
            actual = runtime / relative
            # This conservative milestone refuses absent reference dependencies.
            # It reports changed canonical helpers instead of copying old ones.
            if not actual.is_file() or not actual.resolve().is_relative_to(runtime):
                raise ExportError(f"Missing or redirected target reference: {relative}")
            target_bytes = metal.read(relative)
            if actual.read_bytes() != target_bytes:
                raise ExportError(f"Target reference has uncommitted changes: {relative}")
            observed = hashlib.sha256(target_bytes).hexdigest()
            dependencies.append(
                {
                    "path": relative,
                    "recorded_sha256": entry["sha256"],
                    "target_sha256": observed,
                    "changed": observed != entry["sha256"],
                }
            )
    destination = runtime / "ttnn/ttnn/operations" / _safe_path(manifest["operation"])
    if destination.parent.resolve() != destination.parent:
        raise ExportError("Operation parent is redirected")
    if destination.exists() or destination.is_symlink():
        if not allow_installed or destination.is_symlink():
            raise ExportError("Operation destination exists; refusing overwrite")
        source = package / "overlay/ttnn/ttnn/operations" / manifest["operation"]
        expected = {p.relative_to(source) for p in source.rglob("*") if p.is_file()}
        actual = {
            p.relative_to(destination) for p in destination.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        }
        if expected != actual or any(
            not (destination / p).resolve().is_relative_to(destination)
            or (destination / p).read_bytes() != (source / p).read_bytes()
            for p in expected
        ):
            raise ExportError("Installed source differs from frozen preparation")
    elif allow_installed:
        raise ExportError("Recorded source must be installed before validating a port")
    return {
        "preparation_sha256": manifest["preparation_sha256"],
        "operation": manifest["operation"],
        "runtime": str(runtime),
        "recorded_revision": manifest["metal_commit"],
        "target_revision": revision,
        "eval_revision": manifest["eval_commit"],
        "dependencies": dependencies,
        "target_baseline_reproduced": False,
        "migration_ready": False,
    }


def install(package, runtime, revision, output):
    report = inspect(package, runtime, revision)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ExportError("Report destination exists")
    runtime = Path(report["runtime"])
    source = Path(package) / "overlay/ttnn/ttnn/operations" / report["operation"]
    destination = runtime / "ttnn/ttnn/operations" / report["operation"]
    # Reserve evidence first; retain the directory on failure rather than hide an attempt.
    output.mkdir(parents=True)
    (output / "target-inputs.json").write_bytes(json_bytes(report))
    with tempfile.TemporaryDirectory(prefix=".target-install-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "operation"
        shutil.copytree(source, staged)
        destination.mkdir()  # Exclusive reservation, never overwrite an existing operation.
        staged.rename(destination)
    report["installed"] = True
    (output / "installation.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Install and write new evidence directory; omit for read-only inspection",
    )
    args = parser.parse_args()
    try:
        if args.output:
            result = install(args.preparation, args.runtime, args.target_revision, args.output)
        else:
            result = inspect(args.preparation, args.runtime, args.target_revision)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (ExportError, OSError, ValueError) as error:
        parser.exit(2, f"Target preparation blocked: {error}\n")


if __name__ == "__main__":
    main()
