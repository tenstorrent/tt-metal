# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Takes the decoded information, from the decode inputs, and puts it into the decoded file. The file lists
its sources, so the ranks of a whole fabric can share its shape. README.md in this directory describes the format.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import pathlib

from tt_metal.fabric.debug.visualizer.decode.health import get_health
from tt_metal.fabric.debug.visualizer.decode.inputs import Checkout, Input
from tt_metal.fabric.debug.visualizer.decode.routers import decode_router
from tt_metal.fabric.debug.visualizer.decode.type_parser import get_types_from_manifest

DECODED_VERSION = 1
SNAPSHOT_PREFIX = "fabric_snapshot"
DECODED_PREFIX = "fabric_decoded"


def _utc_text(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_decoded_name(snapshot_path: pathlib.Path) -> str:
    """The decoded file's name for a snapshot's: fabric_snapshot_rank_1_of_1_<UTC>.json gives fabric_decoded_rank_..."""
    name = snapshot_path.name
    return (
        DECODED_PREFIX + name[len(SNAPSHOT_PREFIX) :]
        if name.startswith(SNAPSHOT_PREFIX)
        else f"{DECODED_PREFIX}_{name}"
    )


def _get_source(loaded: Input) -> dict:
    return {
        "snapshot": {"file": str(loaded.snapshot_path.resolve()), "sha256": loaded.snapshot_sha256},
        "manifest": {
            "file": str(loaded.manifest_path.resolve()),
            "sha256": loaded.snapshot["manifest"]["sha256"],
            "run": loaded.manifest["run"],
        },
        "capture": loaded.snapshot["capture"],
        "warnings": list(loaded.warnings),
    }


def decode_snapshot(loaded: Input, checkout: Checkout | None, decoded_at: datetime.datetime) -> dict:
    """
    The decoded file for a loaded snapshot. `checkout` is decode's own, recorded as the decoder, and `decoded_at` is
    when the decode ran. Raises ManifestError or InputError, as decode_router and get_health do.
    """
    types = get_types_from_manifest(loaded.manifest)
    routers = {}
    for path in loaded.snapshot["routers"]:
        decoded = decode_router(loaded, types, path)
        values = dataclasses.asdict(decoded)
        del values["path"]  # the router's key
        routers[path] = {"source": 0, **values, "health": dataclasses.asdict(get_health(loaded, decoded))}
    return {
        "kind": "fabric_decoded",
        "decoded_version": DECODED_VERSION,
        "decoded_at": _utc_text(decoded_at),
        "decoder": None if checkout is None else {"git_sha": checkout.sha, "git_dirty": checkout.dirty},
        "sources": [_get_source(loaded)],
        "routers": routers,
    }


def write_decoded(decoded: dict, path: pathlib.Path):
    """Writes the decoded file to a temporary file beside `path`, then renames it into place."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(decoded, indent=2) + "\n")
    os.replace(temporary, path)
