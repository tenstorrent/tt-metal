# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native/ELF and Python/input identities; helper APIs never import TTNN."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

NATIVE_ROOTS = (
    "ttnn/cpp",
    "ttnn/api",
    "ttnn/CMakeLists.txt",
    "tt_metal",
    "tt_stl",
    "CMakeLists.txt",
    "CMakePresets.json",
    "cmake",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE, timeout=60)


def _file_sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _file_record(path):
    """Hash current bytes, rejecting a file that changes while being read."""
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Identity input must be a regular file: {path}")
    before = path.stat()
    sha256 = _file_sha(path)
    after = path.stat()
    if any(getattr(before, key) != getattr(after, key) for key in ("st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise RuntimeError(f"Identity file changed while hashing: {path}")
    return dict(path=str(path), size=after.st_size, sha256=sha256)


def source_input_identity(repo_root=None, *, model_root=None, explicit_paths=(), readiness_root=None):
    """Hash Python sources and explicit config/build/reference inputs on CPU.

    ``source_digest`` excludes explicit inputs so different candidates can
    establish identical execution sources. ``digest`` binds the whole payload
    for before/after checks. Neither digest includes Git HEAD, documentation,
    bytecode, generated reports, score caches, or unrelated test harnesses.
    File paths are resolved absolute paths and actual file bytes are hashed.

    ``readiness_root`` names the installed readiness_check package directory;
    omitted, its spec is located without importing the package. TTNN sources
    include the checkout plus any already-loaded TTNN Python package location.
    Unknown future harness schemas are not interpreted here. Explicit inputs
    must exist; policy and input semantic validation remains the caller's job.
    """
    if repo_root is None:
        repo_root = _git(Path(__file__).parent, "rev-parse", "--show-toplevel").decode().strip()
    repo = Path(repo_root).resolve(strict=True)
    model = (
        Path(model_root).resolve(strict=True)
        if model_root is not None
        else repo / "models/autoports/qwen_qwen3_6_35b_a3b"
    )
    source_paths = set()
    missing_optional = []

    def python_tree(directory, *, recursive):
        directory = Path(directory).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError(f"Python source root must be a directory: {directory}")
        paths = sorted(directory.rglob("*.py") if recursive else directory.glob("*.py"))
        if not paths:
            raise ValueError(f"No Python source files found: {directory}")
        source_paths.update(path.resolve(strict=True) for path in paths if path.is_file())

    python_tree(model / "tt", recursive=True)
    for name in (
        "datatype_candidate.py",
        "run_full_model_performance.py",
        "runtime_audit.py",
        "datatype_runtime_identity.py",
    ):
        source_paths.add((model / "tests" / name).resolve(strict=True))
    optional = model / "tests/datatype_p1_runner.py"
    if optional.exists():
        source_paths.add(optional.resolve(strict=True))
    else:
        missing_optional.append(str(optional.resolve()))
    if readiness_root is None:
        loaded = sys.modules.get("readiness_check")
        locations = list(getattr(loaded, "__path__", ())) if loaded is not None else []
        if not locations:
            spec = importlib.util.find_spec("readiness_check")
            locations = list(spec.submodule_search_locations or ()) if spec is not None else []
        if len(locations) != 1:
            raise RuntimeError("Locate one installed readiness_check package on PYTHONPATH or pass readiness_root")
        readiness_root = locations[0]
    python_tree(readiness_root, recursive=True)
    python_tree(repo / "ttnn/ttnn", recursive=True)
    loaded_ttnn = sys.modules.get("ttnn")
    if loaded_ttnn is not None:
        for location in getattr(loaded_ttnn, "__path__", ()):
            python_tree(location, recursive=True)
    source_files = [_file_record(path) for path in sorted(source_paths)]
    inputs = {Path(path).resolve(strict=True) for path in explicit_paths}
    input_files = [_file_record(path) for path in sorted(inputs)]
    sources = dict(source_files=source_files, missing_optional=sorted(missing_optional))
    payload = dict(**sources, input_files=input_files)
    return dict(schema_version=1, payload=payload, source_digest=digest(sources), digest=digest(payload))


def _worktree(root, scopes):
    patch = _git(root, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--", *scopes)
    names = sorted(
        filter(None, _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *scopes).split(b"\0"))
    )
    h = hashlib.sha256()
    for raw in names:
        path = root / raw.decode()
        # Hash link targets rather than silently reading files outside the source tree.
        contents = ("link:" + str(path.readlink())).encode() if path.is_symlink() else bytes.fromhex(_file_sha(path))
        h.update(raw + b"\0" + contents + b"\0")
    return dict(
        dirty_diff_sha256=hashlib.sha256(patch).hexdigest(), untracked_count=len(names), untracked_sha256=h.hexdigest()
    )


def native_source_identity(repo_root):
    root = Path(repo_root).resolve()
    trees = {}
    for scope in NATIVE_ROOTS:
        value = _git(root, "ls-tree", "HEAD", "--", scope).decode().strip()
        # ls-tree records absent paths as absent, and supports both trees and build files.
        trees[scope] = value.split("\t", 1)[0] if value else None
    submodules = []
    status = _git(root, "submodule", "status", "--recursive").decode()
    for line in status.splitlines():
        fields = line[1:].split()
        commit, relative = fields[:2]
        if not any(relative == scope or relative.startswith(scope + "/") for scope in NATIVE_ROOTS):
            continue
        entry = dict(path=relative, commit=commit, initialized=line[0] != "-")
        if entry["initialized"]:
            entry.update(_worktree(root / relative, (".",)))
        submodules.append(entry)
    return dict(tracked_objects=trees, **_worktree(root, NATIVE_ROOTS), submodules=submodules)


def loaded_elf_identity(maps_path="/proc/self/maps", *, require_loaded=True):
    """Inspect mapped TT libraries, never import ttnn or initialize a device."""
    libraries = {}
    for line in Path(maps_path).read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        name = fields[5]
        basename = Path(name.removesuffix(" (deleted)")).name
        if not (basename.startswith("_ttnn") or basename.startswith(("libtt_", "libtt-"))):
            continue
        if ".so" not in basename:
            continue
        if name.endswith(" (deleted)"):
            raise RuntimeError(f"Loaded TT library was deleted: {name}")
        path = Path(name).resolve(strict=True)
        if str(path) in libraries:
            continue
        with path.open("rb") as handle:
            if handle.read(4) != b"\x7fELF":
                raise RuntimeError(f"Mapped TT library is not ELF: {path}")
        libraries[str(path)] = dict(path=str(path), size=path.stat().st_size, sha256=_file_sha(path))
    result = sorted(libraries.values(), key=lambda row: row["path"])
    if require_loaded and not (
        any(Path(row["path"]).name.startswith("_ttnn") for row in result)
        and any(Path(row["path"]).name.startswith("libtt_metal.so") for row in result)
    ):
        raise RuntimeError(
            "Import ttnn before capturing runtime identity; mapped TTNN and tt-metal libraries are required"
        )
    return result


def runtime_identity(repo_root=None, *, maps_path="/proc/self/maps", require_loaded=True):
    """Return payload plus stable digest; caller must already have imported ttnn."""
    if repo_root is None:
        repo_root = _git(Path(__file__).parent, "rev-parse", "--show-toplevel").decode().strip()
    payload = dict(
        native_source=native_source_identity(repo_root),
        loaded_elf=loaded_elf_identity(maps_path, require_loaded=require_loaded),
    )
    return dict(schema_version=1, payload=payload, digest=digest(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite runtime identity evidence")
    # CLI is for the serialized launcher only. Import loads libraries, not devices.
    import ttnn  # noqa: F401

    result = runtime_identity(args.repo_root)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
