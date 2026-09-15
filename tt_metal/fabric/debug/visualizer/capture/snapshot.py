# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Assemble a fabric debug snapshot from peeked router samples."""

import importlib.metadata
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_VERSION = 1


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_identity(run):
    return {
        "arch": run["arch"],
        "fabric_config": run["fabric_config"],
        "host_rank": run["host_rank"],
        "mpi_rank": run["mpi_rank"],
        "world_size": run["world_size"],
    }


def owner_alive(proc_root: Path = Path("/proc")) -> bool | None:
    """Best-effort check for another process holding a Tenstorrent device fd."""

    permission_denied = False
    try:
        process_dirs = tuple(proc_root.iterdir())
    except OSError:
        return None

    for process_dir in process_dirs:
        if not process_dir.name.isdigit() or process_dir.name == str(os.getpid()):
            continue
        try:
            file_descriptors = process_dir.joinpath("fd").iterdir()
            for descriptor in file_descriptors:
                try:
                    target = descriptor.readlink()
                except FileNotFoundError:
                    continue
                except PermissionError:
                    permission_denied = True
                    continue
                except OSError:
                    continue
                if str(target).startswith("/dev/tenstorrent/"):
                    return True
        except PermissionError:
            permission_denied = True
        except (FileNotFoundError, NotADirectoryError):
            continue

    return None if permission_denied else False


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def capture_provenance(argv=None) -> dict:
    """Return host/tool provenance that does not affect device reads."""

    return {
        "ttexalens_version": _distribution_version("tt-exalens"),
        "tt_umd_version": _distribution_version("tt-umd"),
        "hostname": socket.gethostname(),
        "owner_alive": owner_alive(),
        "argv": list(sys.argv if argv is None else argv),
    }


def build_snapshot(manifest, samples, provenance, captured_at=None, raw=None):
    """Build one v1 snapshot object. Does not write a file.

    *samples* is a list of peek_manifest() results. Each already carries the
    time its own reads began, so this function never invents a capture time;
    *captured_at* is only when the file itself was assembled.
    """

    if not samples:
        raise ValueError("a snapshot needs at least one sample")

    snapshot = {
        "snapshot_version": SNAPSHOT_VERSION,
        "kind": "fabric_debug_snapshot",
        "captured_at": captured_at if captured_at is not None else utc_timestamp(),
        "manifest": {
            "path": str(manifest.path),
            "manifest_version": manifest.data["manifest_version"],
            "sha256": manifest.sha256,
            "run": run_identity(manifest.run),
        },
        "provenance": dict(provenance),
        "samples": list(samples),
    }
    if raw is not None:
        snapshot["raw"] = dict(raw)
    return snapshot
