# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Probe real Qwen3.8 BF4 cache binding without opening a device.

This diagnostic is intentionally read-only.  It constructs the exact
``Qwen38BF4Cache`` identity which owns the already-staged 49-slot corpus and
calls its public ``verify_layer`` interface for every backbone and MTP slot.
It records an exact interface blocker when the diagnostic tensorbins cannot
be enumerated by the manifest-backed cache; it never publishes a manifest or
loads tensor payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    PINNED_CHECKPOINT_REVISION,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4CacheIdentity,
    Qwen38BF4Cache,
    bf4_converter_source_identity,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import Qwen38MeshContract

MODE = "diagnostic_non_promoting_cpu_bf4_binding_probe"
PHYSICAL_IDS = (1, 0, 2, 3)
DRAM_BANK_RING_ORDER = (4, 0, 2, 6, 7, 3, 5, 1)  # the eight-bank Blackhole ring, banks by worker (y, x) descending
EXPECTED_SLOTS = tuple(("backbone", index) for index in range(48)) + (("mtp", 0),)
LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _loaded_device_descriptors() -> list[str]:
    descriptors = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("/dev/tenstorrent"):
            descriptors.append(target)
    return sorted(descriptors)


def probe(*, artifact_root: Path, expected_identity_key: str) -> dict[str, Any]:
    if os.environ.get("TT_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU binding probe requires TT_VISIBLE_DEVICES to be the exact empty string")
    if _loaded_device_descriptors():
        raise RuntimeError(f"CPU binding probe inherited device descriptors: {_loaded_device_descriptors()}")
    if LOWER_HEX_64.fullmatch(expected_identity_key) is None:
        raise ValueError("BF4 identity key must be exact lowercase SHA-256")
    if not artifact_root.is_absolute():
        raise ValueError("artifact root must be absolute")
    artifact_root = artifact_root.resolve(strict=True)
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {artifact_root}")

    identity = BF4CacheIdentity(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=CHECKPOINT_TENSOR_MANIFEST_SHA256,
        converter_sources=bf4_converter_source_identity(),
        mesh_shape=(1, 4),
        physical_ids=PHYSICAL_IDS,
        ring_size=len(DRAM_BANK_RING_ORDER),
        dram_bank_ring_order=DRAM_BANK_RING_ORDER,
    )
    if identity.key != expected_identity_key or artifact_root.name != identity.key:
        raise RuntimeError(
            f"BF4 identity does not bind the requested root: {identity.key} != "
            f"{expected_identity_key} != {artifact_root.name}"
        )
    cache = Qwen38BF4Cache(artifact_root.parent, identity, Qwen38MeshContract(PHYSICAL_IDS))
    if cache.root != artifact_root:
        raise RuntimeError(f"real BF4 cache resolved {cache.root}, expected {artifact_root}")

    bound = []
    missing = []
    for namespace, layer_index in EXPECTED_SLOTS:
        record = cache.verify_layer(namespace, layer_index)
        if record is None:
            missing.append([namespace, layer_index])
        else:
            bound.append([record.namespace, record.layer_index])

    descriptors = _loaded_device_descriptors()
    if descriptors:
        raise RuntimeError(f"CPU binding probe opened device descriptors: {descriptors}")
    manifest_exists = cache.manifest_path.exists()
    status = "pass" if len(bound) == len(EXPECTED_SLOTS) else "blocked"
    blocker = None
    if status == "blocked":
        blocker = {
            "interface": "Qwen38BF4Cache.verify_layer",
            "detail": (
                "the diagnostic corpus has no manifest.json, so the real cache returns no BF4LayerRecord; "
                "Qwen38TTNNBuilder.require_bf4_layers therefore reports every slot missing and "
                "Qwen38BF4Streamer.load_layer has no record to load"
            ),
            "nearest_required_interface": (
                "a diagnostic-only, read-only record binder/loader for the already hash-pinned local-shape tensorbins"
            ),
        }
    return {
        "mode": MODE,
        "status": status,
        "production_qualification": False,
        "device_opened": False,
        "device_locks_acquired": False,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "completed_utc": _utc_now(),
        "artifact_root": str(artifact_root),
        "cache_class": type(cache).__name__,
        "identity_key": identity.key,
        "physical_ids": list(identity.physical_ids),
        "dram_bank_ring_order": list(identity.dram_bank_ring_order),
        "manifest_path": str(cache.manifest_path),
        "manifest_exists": manifest_exists,
        "enumerated_slots": len(EXPECTED_SLOTS),
        "bound_records": bound,
        "bound_record_count": len(bound),
        "missing_records": missing,
        "missing_record_count": len(missing),
        "blocker": blocker,
    }


def _write_json_exclusive(path: Path, document: dict[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError("output parent must be an existing absolute directory")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-identity-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(
        artifact_root=args.artifact_root,
        expected_identity_key=args.expected_identity_key,
    )
    _write_json_exclusive(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
