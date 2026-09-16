# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Discover and verify offline fabric-debug artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tt_metal.fabric.debug.visualizer.capture.manifest import FabricManifest, ManifestError, load_manifest


class DecodeError(ValueError):
    """The supplied artifacts cannot form one decoded fabric state."""


@dataclass(frozen=True)
class ManifestArtifact:
    path: Path
    manifest: FabricManifest


@dataclass(frozen=True)
class RawArtifact:
    path: Path
    size: int
    sha256: str
    verified: bool


@dataclass(frozen=True)
class DecodeInput:
    manifest: ManifestArtifact
    snapshot_path: Path | None
    snapshot_sha256: str | None
    snapshot: dict[str, Any] | None
    raw: RawArtifact | None
    manifest_sha_verified: bool | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except OSError as error:
        raise DecodeError(f"could not read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DecodeError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise DecodeError(f"{path} must contain a JSON object")
    return value


def _json_paths(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    result: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            result.update(candidate.resolve() for candidate in path.glob("*.json"))
        elif path.suffix == ".json":
            result.add(path.resolve())
        else:
            raise DecodeError(f"input must be a directory or JSON file: {path}")
    return tuple(sorted(result))


def _run_key(run: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(run.get(key) for key in ("arch", "fabric_config", "host_rank", "mpi_rank", "world_size"))


def _validate_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    if snapshot.get("kind") != "fabric_debug_snapshot":
        raise DecodeError(f"{path} is not a fabric debug snapshot")
    if snapshot.get("snapshot_version") != 1:
        raise DecodeError(f"{path} has unsupported snapshot_version {snapshot.get('snapshot_version')!r}")
    for key in ("manifest", "raw"):
        if not isinstance(snapshot.get(key), dict):
            raise DecodeError(f"{path}: snapshot.{key} must be an object")
    samples = snapshot.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise DecodeError(f"{path}: v1 decode requires exactly one sample")
    if not isinstance(samples[0], dict) or not isinstance(samples[0].get("routers"), list):
        raise DecodeError(f"{path}: snapshot.samples[0].routers must be an array")


def _raw_artifact(snapshot_path: Path, snapshot: dict[str, Any], skip_hash: bool) -> RawArtifact | None:
    reference = snapshot["raw"]
    raw_path = snapshot_path.parent / str(reference.get("file", ""))
    if not raw_path.is_file():
        return None
    actual_size = raw_path.stat().st_size
    expected_size = reference.get("size")
    if actual_size != expected_size:
        raise DecodeError(f"{raw_path}: size {actual_size} does not match snapshot value {expected_size}")
    expected_sha = reference.get("sha256")
    if skip_hash:
        return RawArtifact(raw_path.resolve(), actual_size, str(expected_sha), False)
    actual_sha = _sha256(raw_path)
    if actual_sha != expected_sha:
        raise DecodeError(f"{raw_path}: sha256 {actual_sha} does not match snapshot value {expected_sha}")
    return RawArtifact(raw_path.resolve(), actual_size, actual_sha, True)


def discover_inputs(
    paths: Iterable[str | Path],
    *,
    allow_manifest_mismatch: bool = False,
    skip_bin_hash: bool = False,
) -> tuple[DecodeInput, ...]:
    """Pair snapshots to manifests and verify their raw sidecars."""

    manifests: list[ManifestArtifact] = []
    snapshots: list[tuple[Path, dict[str, Any]]] = []
    for path in _json_paths(paths):
        value = _json_object(path)
        kind = value.get("kind")
        if kind == "fabric_debug_manifest":
            try:
                manifests.append(ManifestArtifact(path, load_manifest(path)))
            except ManifestError as error:
                raise DecodeError(str(error)) from error
        elif kind == "fabric_debug_snapshot":
            _validate_snapshot(path, value)
            snapshots.append((path, value))

    if not manifests:
        raise DecodeError("no fabric debug manifest found")
    if not snapshots:
        raise DecodeError("no fabric debug snapshot found")

    by_sha = {artifact.manifest.sha256: artifact for artifact in manifests}
    result: list[DecodeInput] = []
    paired_manifest_paths: set[Path] = set()
    for snapshot_path, snapshot in snapshots:
        reference = snapshot["manifest"]
        expected_sha = reference.get("sha256")
        manifest = by_sha.get(expected_sha)
        verified = manifest is not None
        if manifest is None and allow_manifest_mismatch:
            candidates = [
                artifact
                for artifact in manifests
                if _run_key(artifact.manifest.run) == _run_key(reference.get("run", {}))
            ]
            if len(candidates) == 1:
                manifest = candidates[0]
        if manifest is None:
            raise DecodeError(
                f"{snapshot_path}: no supplied manifest has snapshot sha256 {expected_sha!r}"
            )
        result.append(
            DecodeInput(
                manifest=manifest,
                snapshot_path=snapshot_path,
                snapshot_sha256=_sha256(snapshot_path),
                snapshot=snapshot,
                raw=_raw_artifact(snapshot_path, snapshot, skip_bin_hash),
                manifest_sha_verified=verified,
            )
        )
        paired_manifest_paths.add(manifest.path)

    run_pairs = {
        (item.manifest.manifest.run["arch"], item.manifest.manifest.run["fabric_config"])
        for item in result
    }
    if len(run_pairs) != 1:
        raise DecodeError(f"inputs mix run architectures or fabric configurations: {sorted(run_pairs)!r}")
    for manifest in manifests:
        if manifest.path not in paired_manifest_paths:
            result.append(
                DecodeInput(
                    manifest=manifest,
                    snapshot_path=None,
                    snapshot_sha256=None,
                    snapshot=None,
                    raw=None,
                    manifest_sha_verified=None,
                )
            )
    all_run_pairs = {
        (item.manifest.manifest.run["arch"], item.manifest.manifest.run["fabric_config"])
        for item in result
    }
    if len(all_run_pairs) != 1:
        raise DecodeError(f"inputs mix run architectures or fabric configurations: {sorted(all_run_pairs)!r}")
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.manifest.manifest.run["mpi_rank"],
                "" if item.snapshot_path is None else item.snapshot_path.as_posix(),
            ),
        )
    )
