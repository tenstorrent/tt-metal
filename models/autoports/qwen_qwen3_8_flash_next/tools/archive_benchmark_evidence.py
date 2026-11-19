#!/usr/bin/env python3
"""Bundle finished benchmark evidence for repository file-size limits.

One tar.xz stream per run/setup scope, split into <=450 KiB pieces. Originals
remain untouched. The manifest records paths/sizes and bundle/chunk SHA-256;
individual artifacts are not separately compressed or hashed. Restore validates
the complete compressed stream before extracting and never overwrites conflicts.
Run benchmark/context evidence checks after restoration for their artifact hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

MODEL_DIR = Path(__file__).resolve().parents[1]
BENCHMARK = MODEL_DIR / "doc/benchmark"
CHUNK_BYTES = 450 * 1024
ARCHIVES = "evidence_archives"


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(p in ("", ".", "..") for p in path.parts) or str(path) != name:
        raise ValueError(f"Unsafe evidence path: {name!r}")
    return path


def target_path(root, name):
    relative = safe_name(name)
    target = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in restoration path: {current}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Restoration escapes destination: {target}")
    return target


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def same_bytes(first, second):
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while block := left.read(1024 * 1024):
            if block != right.read(len(block)):
                return False
    return True


def create(scope):
    """Create one immutable scope bundle; caller must establish it is finished."""
    scope = Path(scope).resolve()
    destination = scope / ARCHIVES
    if destination.exists():
        raise ValueError(f"Archive already exists; refusing to replace it: {destination}")
    originals = []
    for path in sorted(scope.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Evidence bundles accept regular files/directories only: {path}")
        if path.is_dir():
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Non-regular evidence file: {path}")
        originals.append((path, info))
    if not originals:
        raise ValueError("Cannot archive an empty scope")
    with tempfile.TemporaryDirectory(prefix=".benchmark-bundle-", dir=scope) as scratch:
        scratch = Path(scratch)
        packed = scratch / "bundle.tar.xz"
        inventory = []
        with tarfile.open(packed, "w:xz", preset=6, format=tarfile.PAX_FORMAT) as archive:
            for path, observed in originals:
                name = path.relative_to(scope).as_posix()
                safe_name(name)
                with path.open("rb") as source:
                    before = os.fstat(source.fileno())
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                        observed.st_ino,
                        observed.st_size,
                        observed.st_mtime_ns,
                    ):
                        raise ValueError(f"Evidence changed before bundling: {path}")
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode, entry.mtime = (
                        before.st_size,
                        stat.S_IMODE(before.st_mode),
                        int(before.st_mtime),
                    )
                    archive.addfile(entry, source)
                    after = os.fstat(source.fileno())
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ino,
                ):
                    raise ValueError(f"Evidence changed during bundling: {path}")
                inventory.append({"path": name, "bytes": before.st_size})
        # Check the whole inventory again: no earlier member may change while
        # later members are compressed. This is not a substitute for stopping
        # owned writers before invoking the helper.
        for path, before in originals:
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError(f"Evidence changed while the scope was archived: {path}")
        pending = scratch / ARCHIVES
        pending.mkdir()
        chunks = []
        with packed.open("rb") as source:
            while block := source.read(CHUNK_BYTES):
                name = f"bundle.tar.xz.part{len(chunks):04d}"
                (pending / name).write_bytes(block)
                chunks.append({"path": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        manifest = {
            "schema_version": 1,
            "format": "scope tar.xz split stream",
            "scope": scope.name,
            "originals_preserved": True,
            "inventory": inventory,
            "bundle": {"bytes": packed.stat().st_size, "sha256": digest(packed), "chunks": chunks},
            "chunk_limit_bytes": CHUNK_BYTES,
            "integrity_scope": "Bundle/chunk SHA-256 and original path/size inventory; benchmark checks retain their own required artifact hashes.",
            "restore_working_directory": "The scope directory containing evidence_archives/",
            "restore_command": f"python ../../../tools/{Path(__file__).name} restore --manifest {ARCHIVES}/manifest.json",
        }
        text = json.dumps(manifest, indent=2) + "\n"
        if len(text.encode()) > CHUNK_BYTES:
            raise ValueError(
                "Manifest exceeds repository size margin; preserve originals and revise inventory packaging"
            )
        (pending / "manifest.json").write_text(text)
        pending.rename(destination)
    return destination / "manifest.json"


def restore(manifest_path, destination=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("format") != "scope tar.xz split stream":
        raise ValueError("Unsupported archive manifest")
    root = Path(destination).resolve() if destination is not None else manifest_path.parent.parent
    root.mkdir(parents=True, exist_ok=True)
    inventory = {entry["path"]: entry["bytes"] for entry in manifest["inventory"]}
    if not inventory or not manifest["bundle"]["chunks"]:
        raise ValueError("Empty archive inventory or chunk list")
    if len(inventory) != len(manifest["inventory"]):
        raise ValueError("Duplicate inventory path")
    for name, size in inventory.items():
        target_path(root, name)
        if type(size) is not int or size < 0:
            raise ValueError("Invalid inventory byte count")
    with tempfile.TemporaryDirectory(prefix=".benchmark-restore-", dir=root) as scratch:
        scratch = Path(scratch)
        packed = scratch / "bundle.tar.xz"
        bundle_hash, bundle_bytes, names = hashlib.sha256(), 0, set()
        with packed.open("wb") as output:
            for chunk in manifest["bundle"]["chunks"]:
                name = chunk["path"]
                if len(safe_name(name).parts) != 1 or name in names:
                    raise ValueError("Unsafe or duplicate chunk name")
                names.add(name)
                path = target_path(manifest_path.parent, name)
                if (
                    type(chunk["bytes"]) is not int
                    or not 0 < chunk["bytes"] <= CHUNK_BYTES
                    or path.stat().st_size != chunk["bytes"]
                    or digest(path) != chunk["sha256"]
                ):
                    raise ValueError(f"Chunk size/hash mismatch: {name}")
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        bundle_hash.update(block)
                        bundle_bytes += len(block)
        if bundle_bytes != manifest["bundle"]["bytes"] or bundle_hash.hexdigest() != manifest["bundle"]["sha256"]:
            raise ValueError("Combined archive size/hash mismatch")
        staging = scratch / "files"
        staging.mkdir()
        seen = set()
        with lzma.open(packed, "rb") as compressed, tarfile.open(fileobj=compressed, mode="r|") as archive:
            for entry in archive:
                name = entry.name
                if not entry.isfile() or name in seen or name not in inventory or entry.size != inventory[name]:
                    raise ValueError(f"Unexpected archive member or size: {name}")
                seen.add(name)
                staged = target_path(staging, name)
                staged.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(entry) as source, staged.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if staged.stat().st_size != inventory[name]:
                    raise ValueError(f"Truncated archive member: {name}")
                os.chmod(staged, entry.mode & 0o777)
                os.utime(staged, (entry.mtime, entry.mtime))
        if seen != set(inventory):
            raise ValueError("Archive inventory incomplete")
        existing, missing = [], []
        for name in sorted(inventory):
            target, staged = target_path(root, name), target_path(staging, name)
            for parent in target.parents:
                if parent == root:
                    break
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f"Non-directory restoration parent: {parent}")
            if target.exists():
                if not target.is_file() or not same_bytes(target, staged):
                    raise ValueError(f"Existing evidence differs; refusing overwrite: {target}")
                existing.append(name)
            else:
                missing.append(name)
        for name in missing:
            target = target_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(target_path(staging, name), target)  # atomic no-overwrite
    return {"restored_files": len(missing), "existing_identical_files": len(existing), "destination": str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("create")
    make.add_argument("--scope", type=Path, required=True)
    make.add_argument(
        "--finished", action="store_true", help="Assert all checks/writes are finished and owned server stopped"
    )
    unpack = commands.add_parser("restore")
    unpack.add_argument("--manifest", type=Path, required=True)
    unpack.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        scope = args.scope.resolve()
        if not args.finished:
            parser.error("Create only after evidence is finished and owned writers stop; pass --finished")
        if scope.parent != BENCHMARK or not (scope.name == "setup" or scope.name.startswith("run")):
            parser.error("Scope must be doc/benchmark/setup or a doc/benchmark/run* directory")
        print(create(scope))
    else:
        print(json.dumps(restore(args.manifest, args.destination), indent=2))


if __name__ == "__main__":
    main()
