# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Assemble a fabric debug snapshot from peeked router samples."""

from datetime import datetime, timezone

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


def build_snapshot(manifest, samples, captured_at=None):
    """Build one v1 snapshot object. Does not write a file.

    *samples* is a list of peek_manifest() results. Each already carries the
    time its own reads began, so this function never invents a capture time;
    *captured_at* is only when the file itself was assembled.
    """

    if not samples:
        raise ValueError("a snapshot needs at least one sample")

    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "kind": "fabric_debug_snapshot",
        "captured_at": captured_at if captured_at is not None else utc_timestamp(),
        "manifest": {
            "path": str(manifest.path),
            "manifest_version": manifest.data["manifest_version"],
            "run": run_identity(manifest.run),
        },
        "samples": list(samples),
    }
