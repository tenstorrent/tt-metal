#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_sockets

Description:
    Dumps raw MeshSocket flow-control metadata for every socket endpoint on this host.

    Endpoints come from the Inspector getSockets RPC, which supplies each config buffer address,
    the socket graph edges, and the arch-dependent L1 layout constants needed to locate a sender's
    bytes_acked array.

    Kind column:
      edge      sender/receiver pair matched on this host, so both config buffers were read
      sender    unpaired sender - receiver is on another rank, only the @snd columns were read
      receiver  unpaired receiver - only the @rcv columns were read

    bytes_sent and bytes_acked each exist in BOTH config buffers, so the counter columns name
    where the value was read rather than which copy to believe:

      sent@snd    sender_socket_md.bytes_sent
      sent@rcv    receiver_socket_md.bytes_sent          (written remotely by the sender)
      acked@snd   sender_socket_md.bytes_acked_array[i]  (written remotely by the receiver)
      acked@rcv   receiver_socket_md.bytes_acked

    For d2d the locally-written copies are only refreshed by update_socket_config,
    so on a live hang they still read the host-init 0. That needs to be fixed.

Owner:
    onenezicTT
"""

import struct
from dataclasses import dataclass, field
from typing import TypeAlias

from triage import ScriptConfig, hex_serializer, log_check, log_check_location, run_script, triage_field
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.tt_exalens_lib import read_from_device
from ttexalens.umd_device import TimeoutDeviceRegisterError

script_config = ScriptConfig(
    depends=["inspector_data", "run_checks", "metal_device_id_mapping"],
)

RECV_FMT = "<6I"  # bytes_sent, read_ptr, fifo_addr, fifo_total_size, bytes_acked, is_h2d
SEND_FMT = "<7I"  # bytes_sent, num_downstreams, write_ptr, dstr_bytes_sent_addr, dstr_fifo_addr,
#                   dstr_fifo_total_size, is_d2h


@dataclass
class Peer:
    mesh_id: int
    fabric_chip_id: int
    core_x: int
    core_y: int
    device_id: int | None = None

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.mesh_id, self.fabric_chip_id, self.core_x, self.core_y)

    def label(self) -> str:
        dev = f"dev{self.device_id} " if self.device_id is not None else ""
        return f"{dev}chip{self.fabric_chip_id}/mesh{self.mesh_id} ({self.core_x},{self.core_y})"


@dataclass
class ReceiverMd:
    bytes_sent: int
    read_ptr: int
    fifo_addr: int
    fifo_total_size: int
    bytes_acked: int
    is_h2d: int


@dataclass
class SenderMd:
    bytes_sent: int
    num_downstreams: int
    write_ptr: int
    fifo_total_size: int
    is_d2h: int
    bytes_acked: list[int]


@dataclass
class Endpoint:
    role: str
    location: OnChipCoordinate
    config_addr: int
    fifo_size: int
    md_size: int
    acked_stride: int
    key: tuple[int, int, int, int]
    peers: list[Peer] = field(default_factory=list)
    md: ReceiverMd | SenderMd | None = None
    claimed: bool = False


@dataclass
class SocketRow:
    kind: str = triage_field("Kind")
    device: str = triage_field("Dev")
    location: OnChipCoordinate = triage_field("Loc")
    config_addr: int = triage_field("Config Addr", hex_serializer)
    downstream: int | None = triage_field("Down")
    sent_at_sender: int | None = triage_field("sent@snd")
    sent_at_receiver: int | None = triage_field("sent@rcv")
    acked_at_sender: int | None = triage_field("acked@snd")
    acked_at_receiver: int | None = triage_field("acked@rcv")
    fifo_size: int | None = triage_field("fifo")
    peer: str = triage_field("Peer")
    write_ptr: int | None = triage_field("wr_ptr", verbose=1)
    read_ptr: int | None = triage_field("rd_ptr", verbose=1)
    num_downstreams: int | None = triage_field("n_down", verbose=1)
    is_d2h: int | None = triage_field("is_d2h", verbose=1)
    is_h2d: int | None = triage_field("is_h2d", verbose=1)


def read_md(ep: Endpoint) -> ReceiverMd | SenderMd:
    if ep.role == "receiver":
        raw = read_from_device(ep.location, ep.config_addr, num_bytes=struct.calcsize(RECV_FMT))
        return ReceiverMd(*struct.unpack(RECV_FMT, raw))

    sent, n_down, wr, _dstr_sent_addr, _dstr_fifo, fifo, is_d2h = struct.unpack(
        SEND_FMT, read_from_device(ep.location, ep.config_addr, num_bytes=struct.calcsize(SEND_FMT))
    )
    acked_base = ep.config_addr + ep.md_size
    acked = [
        struct.unpack("<I", read_from_device(ep.location, acked_base + i * ep.acked_stride, num_bytes=4))[0]
        for i in range(max(n_down, 1))
    ]
    return SenderMd(sent, n_down, wr, fifo, is_d2h, acked)


def discover(inspector_data, id_mapping, run_checks) -> list[Endpoint]:
    sockets = list(inspector_data.getSockets().sockets)

    def to_device(metal_id: int):
        if not id_mapping.has_metal_device_id(metal_id):
            return None
        return run_checks.get_device_by_unique_id(id_mapping.get_unique_id(metal_id))

    fabric_to_device_id: dict[tuple[int, int], int] = {}
    for s in sockets:
        for c in s.connections:
            device = to_device(int(c.localChipId))
            if device is not None:
                fabric_to_device_id[(int(c.localMeshId), int(c.localFabricChipId))] = device.id

    endpoints: list[Endpoint] = []
    for s in sockets:
        role = "sender" if s.isSender else "receiver"
        if not s.connections:
            continue

        # The config buffer is one L1 buffer height-sharded a page per socket core, every shard at
        # the same address, so a multi-core socket needs one read per core.
        by_core: dict[tuple[int, int, int], list] = {}
        for c in s.connections:
            by_core.setdefault((int(c.localChipId), int(c.localCoreX), int(c.localCoreY)), []).append(c)

        for (chip_id, core_x, core_y), conns in by_core.items():
            device = to_device(chip_id)
            if device is None:
                continue
            endpoints.append(
                Endpoint(
                    role=role,
                    location=OnChipCoordinate(core_x, core_y, "logical", device, "tensix"),
                    config_addr=int(s.configBufferAddress),
                    fifo_size=int(s.fifoSize),
                    md_size=int(s.senderMdSizeBytes),
                    acked_stride=int(s.bytesAckedStrideBytes),
                    key=(int(conns[0].localMeshId), int(conns[0].localFabricChipId), core_x, core_y),
                    peers=[
                        Peer(
                            mesh_id=int(c.peerMeshId),
                            fabric_chip_id=int(c.peerFabricChipId),
                            core_x=int(c.peerCoreX),
                            core_y=int(c.peerCoreY),
                            device_id=fabric_to_device_id.get((int(c.peerMeshId), int(c.peerFabricChipId))),
                        )
                        for c in conns
                    ],
                )
            )

    return endpoints


def sender_row(snd: Endpoint, index: int, rcv: Endpoint | None, device: str) -> SocketRow:
    s = snd.md if isinstance(snd.md, SenderMd) else None
    r = rcv.md if rcv is not None and isinstance(rcv.md, ReceiverMd) else None
    return SocketRow(
        kind="edge" if rcv else "sender",
        device=device,
        location=snd.location,
        config_addr=snd.config_addr,
        downstream=index,
        sent_at_sender=s.bytes_sent if s else None,
        sent_at_receiver=r.bytes_sent if r else None,
        acked_at_sender=s.bytes_acked[index] if s and index < len(s.bytes_acked) else None,
        acked_at_receiver=r.bytes_acked if r else None,
        fifo_size=(r.fifo_total_size if r else s.fifo_total_size if s else 0) or snd.fifo_size,
        peer=snd.peers[index].label() if index < len(snd.peers) else "?",
        write_ptr=s.write_ptr if s else None,
        read_ptr=r.read_ptr if r else None,
        num_downstreams=s.num_downstreams if s else None,
        is_d2h=s.is_d2h if s else None,
        is_h2d=r.is_h2d if r else None,
    )


def receiver_row(rcv: Endpoint, device: str) -> SocketRow:
    r = rcv.md if isinstance(rcv.md, ReceiverMd) else None
    return SocketRow(
        kind="receiver",
        device=device,
        location=rcv.location,
        config_addr=rcv.config_addr,
        downstream=None,
        sent_at_sender=None,
        sent_at_receiver=r.bytes_sent if r else None,
        acked_at_sender=None,
        acked_at_receiver=r.bytes_acked if r else None,
        fifo_size=(r.fifo_total_size if r else 0) or rcv.fifo_size,
        peer=", ".join(p.label() for p in rcv.peers),
        write_ptr=None,
        read_ptr=r.read_ptr if r else None,
        num_downstreams=None,
        is_d2h=None,
        is_h2d=r.is_h2d if r else None,
    )


Edge: TypeAlias = tuple[Endpoint, int, Endpoint | None]


def pair(endpoints: list[Endpoint]) -> tuple[list[Edge], list[Endpoint]]:
    """Match senders to local receivers using Inspector keys only, so no device reads are needed
    and the row set cannot depend on the order devices are visited in.

    Returns (sender-anchored edges, receivers no edge covers).
    """
    single_device = len({ep.location.device.id for ep in endpoints}) == 1
    if not single_device and all(ep.key[:2] == (0, 0) for ep in endpoints):
        log_check(False, "dump_sockets: fabric node ids unset on every endpoint, cannot pair sender/receiver")
        receivers: dict[tuple[int, int, int, int], Endpoint] = {}
    else:
        receivers = {ep.key: ep for ep in endpoints if ep.role == "receiver"}

    edges: list[Edge] = []
    for ep in endpoints:
        if ep.role != "sender":
            continue
        for i, peer in enumerate(ep.peers):
            rcv = receivers.get(peer.key)
            if rcv:
                rcv.claimed = True
            edges.append((ep, i, rcv))
    return edges, [ep for ep in endpoints if ep.role == "receiver" and not ep.claimed]


def run(args, context: Context):
    from inspector_data import run as get_inspector_data
    from run_checks import run as get_run_checks, device_description_serializer
    from metal_device_id_mapping import run as get_metal_device_id_mapping

    run_checks = get_run_checks(args, context)
    endpoints = discover(
        get_inspector_data(args, context),
        get_metal_device_id_mapping(args, context),
        run_checks,
    )
    if not endpoints:
        return None

    by_device: dict[int, list[Endpoint]] = {}
    for ep in endpoints:
        by_device.setdefault(ep.location.device.unique_id, []).append(ep)

    def read_device_mds(device: Device) -> list[Endpoint] | None:
        eps = by_device.get(device.unique_id)
        if not eps:
            return None
        for ep in eps:
            try:
                ep.md = read_md(ep)
            except TimeoutDeviceRegisterError:
                raise  # let run_per_device_check mark the device (and its remotes) broken
            except Exception as e:
                log_check_location(
                    ep.location, False, f"{ep.role} socket config buffer 0x{ep.config_addr:x}: {type(e).__name__}: {e}"
                )
        return eps

    device_label: dict[int, str] = {}
    for result in run_checks.run_per_device_check(read_device_mds) or []:
        device_label[result.device_description.device.unique_id] = device_description_serializer(
            result.device_description
        )

    edges, lone_receivers = pair(endpoints)
    rows = [
        sender_row(snd, index, rcv, device_label[snd.location.device.unique_id])
        for snd, index, rcv in edges
        if snd.location.device.unique_id in device_label
    ]
    rows += [
        receiver_row(rcv, device_label[rcv.location.device.unique_id])
        for rcv in lone_receivers
        if rcv.location.device.unique_id in device_label
    ]
    return rows or None


if __name__ == "__main__":
    run_script()
