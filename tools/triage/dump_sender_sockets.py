#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_sender_sockets

Description:
    Dumps sender_socket_md for every MeshSocket sender endpoint (discovered via the Inspector
    getSockets RPC), one row per downstream. A sender has one bytes_sent but a bytes_acked per
    downstream; occupancy = bytes_sent - bytes_acked[i]. status: idle (nothing sent), drained
    (all acked), backed-up (downstream not draining), or flowing. Downstream is the outgoing
    graph edge, resolved to a real device + logical core.

Owner:
    onenezicTT
"""

from dataclasses import dataclass

from triage import ScriptConfig, hex_serializer, triage_field, run_script, log_check
from socket_reader import discover_sockets, read_sender_md
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context

script_config = ScriptConfig(
    depends=["inspector_data", "run_checks", "metal_device_id_mapping"],
)


@dataclass
class SenderSocketRow:
    device_id: int = triage_field("Dev")
    location: OnChipCoordinate = triage_field("Loc")
    config_addr: int = triage_field("Config Addr", hex_serializer)
    downstream_idx: int = triage_field("ds")
    status: str = triage_field("Status")
    bytes_sent: int = triage_field("bytes_sent")
    bytes_acked: int = triage_field("bytes_acked")
    occupancy: int = triage_field("occupancy")
    downstream_fifo_total_size: int = triage_field("down_fifo_size")
    write_ptr: int = triage_field("write_ptr", hex_serializer)
    downstream: str = triage_field("Downstream")


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
        if ep.role == "sender"
    ]
    if not endpoints:
        return None

    rows = []
    for ep in endpoints:
        try:
            md = read_sender_md(ep)
        except Exception as e:
            log_check(False, f"Failed to read sender socket md at {ep.location} addr {hex(ep.config_addr)}: {e}")
            continue
        for i in range(len(md.bytes_acked)):
            peer = ep.peers[i] if i < len(ep.peers) else None
            downstream = "host(d2h)" if md.is_d2h else (peer.label() if peer else "")
            rows.append(
                SenderSocketRow(
                    device_id=ep.location.device.id,
                    location=ep.location,
                    config_addr=ep.config_addr,
                    downstream_idx=i,
                    status=md.status(i),
                    bytes_sent=md.bytes_sent,
                    bytes_acked=md.bytes_acked[i],
                    occupancy=md.occupancy(i),
                    downstream_fifo_total_size=md.downstream_fifo_total_size,
                    write_ptr=md.write_ptr,
                    downstream=downstream,
                )
            )
    return rows or None


if __name__ == "__main__":
    run_script()
