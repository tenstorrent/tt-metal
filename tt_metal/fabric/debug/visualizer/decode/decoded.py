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

from tt_metal.fabric.debug.visualizer.decode.flow_control import ReceiverFlow, SenderFlow, get_channel_flows
from tt_metal.fabric.debug.visualizer.decode.health import get_health
from tt_metal.fabric.debug.visualizer.decode.inputs import Checkout, Input
from tt_metal.fabric.debug.visualizer.decode.ring_buffer_slots import (
    RingBufferSlot,
    get_ring_buffer_slots,
    get_slot_context,
)
from tt_metal.fabric.debug.visualizer.decode.route_context import get_route_context
from tt_metal.fabric.debug.visualizer.decode.routers import decode_router
from tt_metal.fabric.debug.visualizer.decode.routing_tables import ROUTING_TABLE, get_routing_table_store
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


def _get_source(snapshot: Input) -> dict:
    return {
        "snapshot": {"file": str(snapshot.json_path.resolve()), "sha256": snapshot.json_sha256},
        "manifest": {
            "file": str(snapshot.manifest_path.resolve()),
            "sha256": snapshot.json["manifest"]["sha256"],
            "run": snapshot.manifest["run"],
        },
        "capture": snapshot.json["capture"],
        "warnings": list(snapshot.warnings),
    }


def _get_channels(
    flows: dict[str, SenderFlow | ReceiverFlow], slots: dict[str, list[RingBufferSlot]]
) -> dict[str, dict]:
    """Each active channel's counts and credits, and its ring buffer slots: None when capture didn't read the ring."""
    channels = {}
    for path, flow in flows.items():
        channel_slots = slots.get(path)
        channels[path] = {
            "kind": "receiver" if isinstance(flow, ReceiverFlow) else "sender",
            **dataclasses.asdict(flow),
            "slots": None if channel_slots is None else [dataclasses.asdict(slot) for slot in channel_slots],
        }
    return channels


def decode_snapshot(snapshot: Input, checkout: Checkout | None, decoded_at: datetime.datetime) -> dict:
    """
    The decoded file for a loaded snapshot. `checkout` is decode's own, recorded as the decoder, and `decoded_at` is
    when the decode ran. Raises ManifestError or InputError, as the modules it calls do.
    """
    types = get_types_from_manifest(snapshot.manifest)
    decoded_routers = {path: decode_router(snapshot, types, path) for path in snapshot.json["routers"]}
    route_context = get_route_context(snapshot.manifest)
    store = get_routing_table_store(snapshot, types, decoded_routers, route_context)
    slot_context = get_slot_context(snapshot.manifest, types, route_context)

    routers = {}
    for path, router in decoded_routers.items():
        values = dataclasses.asdict(router)
        del values["path"]  # the router's key
        # Each router's routing table is stored once in routing_tables, with only its own part in routing.
        values["areas"].pop(ROUTING_TABLE, None)
        table_ref = store.routers.get(path)
        table = None if table_ref is None else store.copies[table_ref.table_key]
        slots = get_ring_buffer_slots(snapshot, types, slot_context, router, table)
        routers[path] = {
            "source": 0,
            **values,
            "health": dataclasses.asdict(get_health(snapshot, router)),
            "routing": None if table_ref is None else dataclasses.asdict(table_ref),
            "channels": _get_channels(get_channel_flows(snapshot, decoded_routers, path), slots),
        }
    return {
        "kind": "fabric_decoded",
        "decoded_version": DECODED_VERSION,
        "decoded_at": _utc_text(decoded_at),
        "decoder": None if checkout is None else {"git_sha": checkout.sha, "git_dirty": checkout.dirty},
        "sources": [_get_source(snapshot)],
        "routing_tables": {key: dataclasses.asdict(table) for key, table in store.copies.items()},
        "routers": routers,
    }


def write_decoded(decoded: dict, path: pathlib.Path):
    """Writes the decoded file to a temporary file beside `path`, then renames it into place."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(decoded, indent=2) + "\n")
    os.replace(temporary, path)
