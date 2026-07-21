#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_receiver_sockets

Description:
    Dumps receiver_socket_md for every MeshSocket receiver endpoint (discovered via the
    Inspector getSockets RPC). status: starved (bytes_sent==bytes_acked, upstream idle),
    backed-up (occupancy==fifo_total_size), or flowing. Upstream is the incoming graph edge,
    resolved to a real device + logical core.

Owner:
    onenezicTT
"""

from dataclasses import dataclass

from triage import ScriptConfig, hex_serializer, triage_field, run_script, log_check
from socket_reader import discover_sockets, read_receiver_md
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context

script_config = ScriptConfig(
    depends=["inspector_data", "run_checks", "metal_device_id_mapping"],
)


@dataclass
class ReceiverSocketRow:
    device_id: int = triage_field("Dev")
    location: OnChipCoordinate = triage_field("Loc")
    config_addr: int = triage_field("Config Addr", hex_serializer)
    status: str = triage_field("Status")
    bytes_sent: int = triage_field("bytes_sent")
    bytes_acked: int = triage_field("bytes_acked")
    occupancy: int = triage_field("occupancy")
    fifo_total_size: int = triage_field("fifo_size")
    read_ptr: int = triage_field("read_ptr", hex_serializer)
    fifo_addr: int = triage_field("fifo_addr", hex_serializer)
    upstream: str = triage_field("Upstream")


def run(args, context: Context):
    from inspector_data import run as get_inspector_data
    from run_checks import run as get_run_checks
    from metal_device_id_mapping import run as get_metal_device_id_mapping

    endpoints = [
        ep
        for ep in discover_sockets(
            get_inspector_data(args, context),
            get_metal_device_id_mapping(args, context),
            get_run_checks(args, context),
        )
        if ep.role == "receiver"
    ]
    if not endpoints:
        return None

    rows = []
    for ep in endpoints:
        try:
            md = read_receiver_md(ep)
        except Exception as e:
            log_check(False, f"Failed to read receiver socket md at {ep.location} addr {hex(ep.config_addr)}: {e}")
            continue
        upstream = "host(h2d)" if md.is_h2d else (ep.peers[0].label() if ep.peers else "")
        rows.append(
            ReceiverSocketRow(
                device_id=ep.location.device.id,
                location=ep.location,
                config_addr=ep.config_addr,
                status=md.status,
                bytes_sent=md.bytes_sent,
                bytes_acked=md.bytes_acked,
                occupancy=md.occupancy,
                fifo_total_size=md.fifo_total_size,
                read_ptr=md.read_ptr,
                fifo_addr=md.fifo_addr,
                upstream=upstream,
            )
        )
    return rows or None


if __name__ == "__main__":
    run_script()
