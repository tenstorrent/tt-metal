# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Loads what decode reads for one snapshot: the snapshot's .json and .bin, and the manifest it was taken with, checked to
belong together. It also compares the build that wrote the manifest with the checkout decode runs from.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import lookup

SNAPSHOT_VERSION = 1
# How much of a SHA a warning shows, as much as the build's kGitShaShort.
SHORT_SHA = 10
# The keys decode reads in each router's entry, with their JSON types. started_ns and error can also be null.
ROUTER_KEYS = {"liveness": list, "core_registers": dict, "areas": dict, "regions": dict, "streams": dict}
NULLABLE_ROUTER_KEYS = {"started_ns": int, "error": dict}


class InputError(ValueError):
    """A snapshot, its .bin or its manifest can't be read, or they don't belong together."""


@dataclass(frozen=True)
class Checkout:
    """A git commit, and whether the checkout's tracked files have changes not committed to it."""

    sha: str
    dirty: bool


@dataclass(frozen=True)
class Input:
    """All the inputs to the decoder, checked to belong together."""

    snapshot_path: pathlib.Path
    snapshot_sha256: str  # of the .json's bytes
    snapshot: dict
    raw: bytes  # the .bin
    manifest_path: pathlib.Path
    manifest: dict
    warnings: tuple[str, ...]


def get_checkout(directory: pathlib.Path) -> Checkout | None:
    """
    The commit the git checkout holding `directory` is at, worked out as the build works out the run's
    (cmake/version.cmake). None when `directory` isn't in a checkout or git can't be run.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        # diff-index exits 1 only when tracked files differ from HEAD.
        diff = subprocess.run(["git", "-C", str(directory), "diff-index", "--quiet", "HEAD", "--"], capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return Checkout(head.stdout.strip(), diff.returncode == 1)


def _describe(checkout: Checkout) -> str:
    return checkout.sha[:SHORT_SHA] + (" with local changes" if checkout.dirty else "")


def get_build_warnings(manifest: dict, checkout: Checkout | None) -> list[str]:
    """
    Warnings for when the build that wrote the manifest and decode's checkout may not be the same code. The manifest's
    values decode the same either way, but decode's rules are code, so a fabric change no manifest value shows needs
    the same commit on both sides to be caught.
    """
    if "git_sha" not in lookup(manifest, "run", "object"):
        return [
            "the manifest doesn't say which build wrote it (no run.git_sha), so decode can't tell whether its rules "
            "match the run's"
        ]
    built = Checkout(lookup(manifest, "run.git_sha", "string"), lookup(manifest, "run.git_dirty", "boolean"))
    if not built.sha:
        return [
            "the run was built outside a git checkout (run.git_sha is empty), so decode can't tell whether its "
            "rules match the run's"
        ]
    if checkout is None:
        return [
            f"the run was built from {_describe(built)}; decode isn't in a git checkout, so it can't tell "
            "whether its rules match the run's"
        ]
    if built.sha != checkout.sha or built.dirty or checkout.dirty:
        return [f"the run was built from {_describe(built)}; decode is at {_describe(checkout)}"]
    return []


def _read(path: pathlib.Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise InputError(f"can't read {what} {path}: {error.strerror}") from None


def _parse_json(data: bytes, path: pathlib.Path, what: str):
    try:
        return json.loads(data)
    except ValueError as error:
        raise InputError(f"{what} {path} isn't JSON: {error}") from None


def _get(snapshot, path: str, kind: type, snapshot_path: pathlib.Path):
    """The value at a dotted path in the snapshot, which must be of `kind`."""
    value = snapshot
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise InputError(f"snapshot {snapshot_path} has no {path}")
        value = value[key]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        article = "an" if kind.__name__[0] in "aeiou" else "a"
        raise InputError(f"snapshot {snapshot_path}: {path} is {value!r}, not {article} {kind.__name__}")
    return value


def _find_manifest(
    snapshot_path: pathlib.Path, snapshot: dict, manifest_paths: Sequence[pathlib.Path]
) -> tuple[pathlib.Path, bytes]:
    """
    The manifest the snapshot was taken with: the one of `manifest_paths` whose sha256 is the snapshot's, or with none
    given, the file the snapshot names or a file of that name beside the snapshot.
    """
    sha256 = _get(snapshot, "manifest.sha256", str, snapshot_path)
    if manifest_paths:
        for path in manifest_paths:
            data = _read(path, "manifest")
            if hashlib.sha256(data).hexdigest() == sha256:
                return path, data
        raise InputError(
            f"none of the {len(manifest_paths)} manifests given is the one snapshot {snapshot_path} was taken with, "
            f"whose sha256 is {sha256}"
        )

    named = pathlib.Path(_get(snapshot, "manifest.file", str, snapshot_path))
    for path in (named, snapshot_path.parent / named.name):
        if path.is_file():
            data = _read(path, "manifest")
            actual = hashlib.sha256(data).hexdigest()
            if actual != sha256:
                raise InputError(
                    f"manifest {path} has sha256 {actual}, but snapshot {snapshot_path} was taken with {sha256}. A "
                    "fabric init rewrites the manifest, so this is another run's; pass the right one with --manifest"
                )
            return path, data
    raise InputError(
        f"snapshot {snapshot_path} was taken with manifest {named}, which isn't there or beside the snapshot; pass it "
        "with --manifest"
    )


def load_input(
    snapshot_path: pathlib.Path, checkout: Checkout | None, manifest_paths: Sequence[pathlib.Path] = ()
) -> Input:
    """
    Reads a snapshot's .json, its .bin and its manifest, and checks that they belong together. `checkout` is decode's
    own (get_checkout), for the build identity warnings. `manifest_paths` are the manifests to choose from, for a
    snapshot whose manifest has moved. Raises InputError when the snapshot can't be decoded.
    """
    snapshot_bytes = _read(snapshot_path, "snapshot")
    snapshot = _parse_json(snapshot_bytes, snapshot_path, "snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("kind") != "fabric_snapshot":
        raise InputError(f"{snapshot_path} isn't a fabric snapshot")
    version = _get(snapshot, "snapshot_version", int, snapshot_path)
    if version != SNAPSHOT_VERSION:
        raise InputError(f"snapshot {snapshot_path} is version {version}; decode reads version {SNAPSHOT_VERSION}")
    _get(snapshot, "capture", dict, snapshot_path)
    for path, entry in _get(snapshot, "routers", dict, snapshot_path).items():
        for key, kind in ROUTER_KEYS.items():
            _get(snapshot, f"routers.{path}.{key}", kind, snapshot_path)
        for key, kind in NULLABLE_ROUTER_KEYS.items():
            if key not in entry:
                raise InputError(f"snapshot {snapshot_path} has no routers.{path}.{key}")
            if entry[key] is not None:
                _get(snapshot, f"routers.{path}.{key}", kind, snapshot_path)

    raw_path = snapshot_path.parent / _get(snapshot, "raw.file", str, snapshot_path)
    raw = _read(raw_path, "the snapshot's .bin")
    size = _get(snapshot, "raw.size", int, snapshot_path)
    if len(raw) != size:
        raise InputError(f"{raw_path} is {len(raw)} bytes; snapshot {snapshot_path} says {size}")
    if hashlib.sha256(raw).hexdigest() != _get(snapshot, "raw.sha256", str, snapshot_path):
        raise InputError(f"{raw_path}'s sha256 isn't the one snapshot {snapshot_path} records")

    manifest_path, manifest_bytes = _find_manifest(snapshot_path, snapshot, manifest_paths)
    manifest = _parse_json(manifest_bytes, manifest_path, "manifest")
    return Input(
        snapshot_path=snapshot_path,
        snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
        snapshot=snapshot,
        raw=raw,
        manifest_path=manifest_path,
        manifest=manifest,
        warnings=tuple(get_build_warnings(manifest, checkout)),
    )
