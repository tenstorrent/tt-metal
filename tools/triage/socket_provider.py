#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    socket_provider

Description:
    Socket cores owned by this rank, from the Inspector getSockets RPC: MeshSocket cores, and the
    device side of D2H and H2D sockets. get_cached_senders(location) and
    get_cached_receivers(location) read a core's config buffers once and return the same rows after.

    A D2D kernel flushes its own counters to L1 only at exit, so live values are read from its
    Socket{Sender,Receiver}Interface local, matched on config_addr. This halts the core. A kernel
    frame column reads N/A when no kernel was running on the core or the local was optimized out.

Owner:
    onenezicTT
"""

import struct
from dataclasses import dataclass, field
from functools import cache

from callstack_provider import CallstackProvider, run as get_callstack_provider
from inspector_data import run as get_inspector_data
from metal_device_id_mapping import run as get_metal_device_id_mapping
from run_checks import run as get_run_checks
from triage import (
    ScriptConfig,
    hex_serializer,
    log_warning_location,
    log_warning_risc,
    run_script,
    triage_field,
    triage_singleton,
)
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.exceptions import DebugSymbolError
from ttexalens.tt_exalens_lib import read_from_device
from ttexalens.umd_device import TimeoutDeviceRegisterError

script_config = ScriptConfig(
    data_provider=True,
    depends=["inspector_data", "run_checks", "metal_device_id_mapping", "callstack_provider"],
)


RECV_FMT = "<6I"  # bytes_sent, read_ptr, fifo_addr, fifo_total_size, bytes_acked, is_h2d
SEND_FMT = "<7I"  # bytes_sent, num_downstreams, write_ptr, dstr_bytes_sent_addr, dstr_fifo_addr,
#                   dstr_fifo_total_size, is_d2h


def node_label(mesh_id: int, fabric_chip_id: int) -> str:
    """Fabric node id, the one name for a chip that is the same on every rank."""
    return f"chip{fabric_chip_id}/mesh{mesh_id}"


@dataclass
class Peer:
    mesh_id: int
    fabric_chip_id: int
    core_x: int
    core_y: int
    device_id: int | None = None
    is_host: bool = False

    def label(self) -> str:
        if self.is_host:
            return "host"
        where = f"dev{self.device_id}" if self.device_id is not None else node_label(self.mesh_id, self.fabric_chip_id)
        return f"{where} ({self.core_x},{self.core_y})"


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
    downstream_config_addr: int  # the peer receiver's config buffer address
    fifo_total_size: int
    is_d2h: int
    bytes_acked: list[int]


@dataclass
class SocketCore:
    role: str
    location: OnChipCoordinate
    node: str  # this core's own fabric node id
    config_addr: int
    acked_offset: int
    acked_stride: int
    peers: list[Peer] = field(default_factory=list)
    md: ReceiverMd | SenderMd | None = None
    interface: dict[str, int] = field(default_factory=dict)  # fields read from the kernel's Socket*Interface local


@dataclass
class SenderSocketData:
    node: str = triage_field("Node")
    config_addr: int = triage_field("Config Addr", hex_serializer)
    downstream_config_addr: int | None = triage_field("Downstream Addr", hex_serializer)
    num_downstreams: int = triage_field("Downstreams")
    downstream: int = triage_field("Downstream #")
    sent: int | None = triage_field("sent")
    acked: int = triage_field("acked")
    write_ptr: int | None = triage_field("wr_off")
    fifo_size: int = triage_field("fifo")
    peer: str = triage_field("Peer")


@dataclass
class ReceiverSocketData:
    node: str = triage_field("Node")
    config_addr: int = triage_field("Config Addr", hex_serializer)
    sent: int = triage_field("sent")
    acked: int | None = triage_field("acked")
    read_ptr: int | None = triage_field("rd_addr", hex_serializer)
    fifo_size: int = triage_field("fifo")
    peer: str = triage_field("Peer")


def read_md(core: SocketCore) -> ReceiverMd | SenderMd:
    if core.role == "receiver":
        raw = read_from_device(core.location, core.config_addr, num_bytes=struct.calcsize(RECV_FMT))
        return ReceiverMd(*struct.unpack(RECV_FMT, raw))

    sent, n_down, wr, dstr_config_addr, _dstr_fifo, fifo, is_d2h = struct.unpack(
        SEND_FMT, read_from_device(core.location, core.config_addr, num_bytes=struct.calcsize(SEND_FMT))
    )
    acked_base = core.config_addr + core.acked_offset
    acked = [
        struct.unpack("<I", read_from_device(core.location, acked_base + i * core.acked_stride, num_bytes=4))[0]
        for i in range(max(n_down, 1))
    ]
    return SenderMd(sent, n_down, wr, dstr_config_addr, fifo, is_d2h, acked)


def read_interface(core: SocketCore, callstack_provider: CallstackProvider) -> dict[str, int]:
    """The kernel's Socket{Sender,Receiver}Interface local, found by matching its config_addr."""
    fields = ("bytes_sent", "write_ptr") if core.role == "sender" else ("bytes_acked", "read_ptr")
    dispatcher_data = callstack_provider.dispatcher_data
    for risc_debug in core.location.noc_block.all_riscs:
        risc_name = risc_debug.risc_location.risc_name
        try:
            if dispatcher_data.is_idle_in_default_view(risc_debug.risc_location):
                continue
            frames = callstack_provider.get_cached_callstacks(
                risc_debug, use_full_callstack=True
            ).kernel_callstack_with_message.callstack
        except TimeoutDeviceRegisterError:
            raise
        except Exception as e:
            log_warning_risc(risc_name, core.location, f"socket callstack: {e}")
            continue
        for frame in frames:
            for var in frame.locals + frame.arguments:
                if var.value is None:
                    continue
                try:
                    # get_member cannot index into an array of interfaces.
                    candidates = [var.value[i] for i in range(len(var.value))]
                except Exception:
                    candidates = [var.value]
                for candidate in candidates:
                    try:
                        values = {f: int(candidate.get_member(f).read_value()) for f in ("config_addr", *fields)}
                    except DebugSymbolError:
                        break  # elements share a type, so one miss rules out the rest
                    except TimeoutDeviceRegisterError:
                        raise
                    except Exception as e:
                        log_warning_risc(risc_name, core.location, f"socket interface: {e}")
                        continue
                    if values.pop("config_addr") == core.config_addr:
                        return values
    return {}


def discover(inspector_data, id_mapping, run_checks) -> list[SocketCore]:
    sockets = list(inspector_data.getSockets().sockets)

    def to_device(metal_id: int):
        if not id_mapping.has_metal_device_id(metal_id):
            return None
        return run_checks.get_device_by_unique_id(id_mapping.get_unique_id(metal_id))

    # Fabric node id -> device id, so a peer owned by this host can be labelled like the Dev column.
    fabric_to_device_id: dict[tuple[int, int], int] = {}
    for s in sockets:
        for e in s.localCores:
            device = to_device(int(e.chipId))
            if device is not None:
                fabric_to_device_id[(int(s.localMeshId), int(e.core.fabricChipId))] = device.id

    cores: list[SocketCore] = []
    for s in sockets:
        role = "sender" if s.isSender else "receiver"
        peer_mesh_id = int(s.peerMeshId)
        for e in s.localCores:
            device = to_device(int(e.chipId))
            if device is None:
                continue
            cores.append(
                SocketCore(
                    role=role,
                    location=OnChipCoordinate(int(e.core.coreX), int(e.core.coreY), "logical", device, "tensix"),
                    node=node_label(int(s.localMeshId), int(e.core.fabricChipId)),
                    config_addr=int(s.configBufferAddress),
                    acked_offset=int(s.bytesAckedOffsetBytes),
                    acked_stride=int(s.bytesAckedStrideBytes),
                    peers=(
                        [Peer(mesh_id=0, fabric_chip_id=0, core_x=0, core_y=0, is_host=True)]
                        if s.peerIsHost
                        else [
                            Peer(
                                mesh_id=peer_mesh_id,
                                fabric_chip_id=int(p.fabricChipId),
                                core_x=int(p.coreX),
                                core_y=int(p.coreY),
                                device_id=fabric_to_device_id.get((peer_mesh_id, int(p.fabricChipId))),
                            )
                            for p in e.peers
                        ]
                    ),
                )
            )

    return cores


def sender_row(core: SocketCore, md: SenderMd, index: int) -> SenderSocketData:
    return SenderSocketData(
        node=core.node,
        config_addr=core.config_addr,
        downstream_config_addr=None if core.peers[index].is_host else md.downstream_config_addr,
        downstream=index,
        sent=core.interface.get("bytes_sent", md.bytes_sent if md.is_d2h else None),
        acked=md.bytes_acked[index],
        write_ptr=core.interface.get("write_ptr"),
        fifo_size=md.fifo_total_size,
        num_downstreams=md.num_downstreams,
        peer=core.peers[index].label(),
    )


def receiver_row(core: SocketCore, md: ReceiverMd) -> ReceiverSocketData:
    return ReceiverSocketData(
        node=core.node,
        config_addr=core.config_addr,
        sent=md.bytes_sent,
        acked=core.interface.get("bytes_acked", md.bytes_acked if md.is_h2d else None),
        read_ptr=core.interface.get("read_ptr"),
        fifo_size=md.fifo_total_size,
        peer=", ".join(p.label() for p in core.peers),
    )


class SocketProvider:
    def __init__(self, cores: list[SocketCore], callstack_provider: CallstackProvider):
        self.callstack_provider = callstack_provider
        self.cores_by_location: dict[OnChipCoordinate, list[SocketCore]] = {}
        for core in cores:
            self.cores_by_location.setdefault(core.location, []).append(core)

    @cache
    def get_cached_senders(self, location: OnChipCoordinate) -> list[SenderSocketData]:
        return [
            sender_row(core, core.md, i)
            for core in self.read_cores(location, "sender")
            if isinstance(core.md, SenderMd)
            for i in range(len(core.peers))
        ]

    @cache
    def get_cached_receivers(self, location: OnChipCoordinate) -> list[ReceiverSocketData]:
        return [
            receiver_row(core, core.md)
            for core in self.read_cores(location, "receiver")
            if isinstance(core.md, ReceiverMd)
        ]

    def read_cores(self, location: OnChipCoordinate, role: str) -> list[SocketCore]:
        """This location's cores in the given role whose config buffer could be read."""
        cores: list[SocketCore] = []
        for core in self.cores_by_location.get(location, []):
            if core.role != role:
                continue
            try:
                core.md = read_md(core)
            except TimeoutDeviceRegisterError:
                raise  # let run_per_device_check mark the device broken
            except Exception as e:
                log_warning_location(core.location, f"{core.role} socket config buffer 0x{core.config_addr:x}: {e}")
                continue
            core.interface = read_interface(core, self.callstack_provider)
            cores.append(core)
        return cores


@triage_singleton
def run(args, context: Context) -> SocketProvider:
    cores = discover(
        get_inspector_data(args, context), get_metal_device_id_mapping(args, context), get_run_checks(args, context)
    )
    return SocketProvider(cores, get_callstack_provider(args, context))


if __name__ == "__main__":
    run_script()
