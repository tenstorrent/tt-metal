#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode + discovery for MeshSocket flow-control metadata (tt_metal/hw/inc/hostdev/socket.h).

Occupancy = bytes_sent - bytes_acked. Receiver starved (==) => upstream never sent;
sender/receiver backed-up (== fifo_total_size) => downstream never drained. A sender has one
bytes_sent but a bytes_acked per downstream, so back-pressure is reported per downstream edge.
"""

import struct
from dataclasses import dataclass, field

from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device

L1_ALIGNMENT = 16  # WH & BH
SENDER_MD_SIZE = 32  # align(sizeof(sender_socket_md)=28, 16)
BYTES_ACKED_STRIDE = 16  # align(sizeof(uint32_t), 16)


@dataclass
class SocketPeer:
    """The endpoint on the other side of a socket connection, from the getSockets record."""

    mesh_id: int
    fabric_chip_id: int
    core_x: int
    core_y: int
    device_id: int | None = None  # resolved tt-exalens device, if the peer chip is locally visible

    def label(self) -> str:
        dev = f"dev{self.device_id} " if self.device_id is not None else ""
        return f"{dev}chip{self.fabric_chip_id}/mesh{self.mesh_id} ({self.core_x},{self.core_y})"


@dataclass
class SocketEndpoint:
    role: str  # "sender" | "receiver"
    location: OnChipCoordinate  # active core owning the config buffer
    config_addr: int
    peers: list[SocketPeer] = field(default_factory=list)  # 1 upstream (receiver) or N downstreams (sender)


def _read(location: OnChipCoordinate, addr: int, nbytes: int) -> bytes:
    return read_from_device(location, addr, num_bytes=nbytes)


def _status(bytes_sent: int, bytes_acked: int, fifo_total_size: int, sender: bool) -> str:
    if sender and bytes_sent == 0:
        return "idle"
    occ = bytes_sent - bytes_acked
    if occ == 0:
        return "drained" if sender else "starved"
    return "backed-up" if occ >= fifo_total_size else "flowing"


@dataclass
class ReceiverSocketMd:
    bytes_sent: int
    read_ptr: int
    fifo_addr: int
    fifo_total_size: int
    bytes_acked: int
    is_h2d: int

    @property
    def occupancy(self) -> int:
        return self.bytes_sent - self.bytes_acked

    @property
    def status(self) -> str:
        return _status(self.bytes_sent, self.bytes_acked, self.fifo_total_size, sender=False)


@dataclass
class SenderSocketMd:
    bytes_sent: int
    num_downstreams: int
    write_ptr: int
    downstream_fifo_total_size: int
    is_d2h: int
    bytes_acked: list[int]  # one per downstream

    def occupancy(self, i: int) -> int:
        return self.bytes_sent - self.bytes_acked[i]

    def status(self, i: int) -> str:
        return _status(self.bytes_sent, self.bytes_acked[i], self.downstream_fifo_total_size, sender=True)


def read_receiver_md(ep: SocketEndpoint) -> ReceiverSocketMd:
    bs, rp, fa, fts, ba, ish = struct.unpack_from("<6I", _read(ep.location, ep.config_addr, 24), 0)
    return ReceiverSocketMd(bs, rp, fa, fts, ba, ish)


def read_sender_md(ep: SocketEndpoint) -> SenderSocketMd:
    bs, nd, wp, _dbsa, _dfa, dfts, isd = struct.unpack_from("<7I", _read(ep.location, ep.config_addr, 28), 0)
    acked_base = ep.config_addr + SENDER_MD_SIZE
    acked = [
        struct.unpack_from("<I", _read(ep.location, acked_base + i * BYTES_ACKED_STRIDE, 4), 0)[0]
        for i in range(max(nd, 1))
    ]
    return SenderSocketMd(bs, nd, wp, dfts, isd, acked)


def discover_sockets(inspector_data, metal_device_id_mapping, run_checks) -> list[SocketEndpoint]:
    """Discover socket endpoints from the Inspector getSockets RPC (mesh_socket_created hook)."""
    try:
        sockets = inspector_data.getSockets().sockets
    except Exception:
        return []

    def metal_to_device(metal_id: int):
        if not metal_device_id_mapping.has_metal_device_id(metal_id):
            return None
        return run_checks.get_device_by_unique_id(metal_device_id_mapping.get_unique_id(metal_id))

    # fabric node id -> tt-exalens device id, from every endpoint's local fields (resolves peers).
    fabric_to_device_id: dict[tuple[int, int], int] = {}
    for s in sockets:
        for c in s.connections:
            device = metal_to_device(int(c.localChipId))
            if device is not None:
                fabric_to_device_id[(int(c.localMeshId), int(c.localFabricChipId))] = device.id

    endpoints: list[SocketEndpoint] = []
    for s in sockets:
        role = "sender" if s.isSender else "receiver"
        conns = list(s.connections)
        if not conns:
            continue
        device = metal_to_device(int(conns[0].localChipId))
        if device is None:
            continue
        location = OnChipCoordinate(int(conns[0].localCoreX), int(conns[0].localCoreY), "logical", device, "tensix")
        peers = [
            SocketPeer(
                mesh_id=int(c.peerMeshId),
                fabric_chip_id=int(c.peerFabricChipId),
                core_x=int(c.peerCoreX),
                core_y=int(c.peerCoreY),
                device_id=fabric_to_device_id.get((int(c.peerMeshId), int(c.peerFabricChipId))),
            )
            for c in conns
        ]
        endpoints.append(
            SocketEndpoint(role=role, location=location, config_addr=int(s.configBufferAddress), peers=peers)
        )
    return endpoints
