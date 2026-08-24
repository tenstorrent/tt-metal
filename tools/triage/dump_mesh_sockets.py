#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_mesh_sockets

Description:
    Dumps raw MeshSocket flow-control metadata from the L1 config buffers, one row per endpoint.
    Endpoints come from the Inspector getSockets RPC. MeshSocket only.

    Each socket has two endpoints, a sender and a receiver, each with its own config buffer on its
    own core, and each gets its own row. Role says which one, and columns that live in the other
    endpoint's buffer are N/A. To join the two halves of a socket, match a sender row's Downstream
    Addr against a receiver row's Config Addr; that works the same on one host or across ranks, where
    a rank only ever sees its own endpoints. A sender with several downstreams gets one row per
    downstream, numbered in Downstream #. All core coordinates are logical. An endpoint whose config
    buffer could not be read gets no row.

    Node is this endpoint's own fabric node id, the one chip name that is identical on every rank. A
    peer on this host shows as its device id, matching the Dev column; a peer on another rank shows as
    a fabric node id, which you match against the Node column of that rank's output.

    Counter columns, by where the value is read from:

      sent@snd    sender bytes_sent                                    kernel frame
      acked@snd   sender_socket_md.bytes_acked_array[i]                L1, written by the receiver
      wr_off      sender write_ptr, offset from downstream_fifo_addr   kernel frame
      sent@rcv    receiver_socket_md.bytes_sent                        L1, written by the sender
      acked@rcv   receiver bytes_acked                                 kernel frame
      rd_addr     receiver read_ptr, absolute L1 address               kernel frame

    The L1 copies of the counters a kernel owns are only refreshed by update_socket_config at kernel
    exit, so they hold where the previous invocation finished. The current values are read instead from
    the SocketSenderInterface / SocketReceiverInterface local in the kernel's frame, matched on
    config_addr. That halts the endpoint core. A column reads ? when the frame could not be walked or
    the local was optimized out.

Owner:
    onenezicTT
"""

import struct
from dataclasses import dataclass, field

from callstack_provider import CallstackProvider, run as get_callstack_provider
from inspector_data import run as get_inspector_data
from metal_device_id_mapping import run as get_metal_device_id_mapping
from run_checks import run as get_run_checks
from triage import ScriptConfig, hex_serializer, log_warning_location, log_warning_risc, run_script, triage_field
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.device import Device
from ttexalens.exceptions import DebugSymbolError
from ttexalens.tt_exalens_lib import read_from_device
from ttexalens.umd_device import TimeoutDeviceRegisterError

script_config = ScriptConfig(
    depends=["inspector_data", "run_checks", "metal_device_id_mapping", "callstack_provider"],
)


RECV_FMT = "<6I"  # bytes_sent, read_ptr, fifo_addr, fifo_total_size, bytes_acked, is_h2d
SEND_FMT = "<7I"  # bytes_sent, num_downstreams, write_ptr, dstr_bytes_sent_addr, dstr_fifo_addr,
#                   dstr_fifo_total_size, is_d2h


def hex_or_unknown(value: int | str | None) -> str:
    return value if isinstance(value, str) else hex_serializer(value)


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

    def label(self) -> str:
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
class Endpoint:
    role: str
    location: OnChipCoordinate
    node: str  # this endpoint's own fabric node id
    config_addr: int
    md_size: int
    acked_stride: int
    peers: list[Peer] = field(default_factory=list)
    md: ReceiverMd | SenderMd | None = None
    interface: dict[str, int] = field(default_factory=dict)  # fields read from the kernel's Socket*Interface local


@dataclass
class SocketRow:
    role: str = triage_field("Role")
    location: OnChipCoordinate = triage_field("Loc")
    node: str = triage_field("Node")
    config_addr: int = triage_field("Config Addr", hex_serializer)
    downstream_config_addr: int | None = triage_field("Downstream Addr", hex_serializer)
    num_downstreams: int | None = triage_field("Downstreams")
    downstream: int | None = triage_field("Downstream #")
    sent_at_sender: int | str | None = triage_field("sent@snd")
    acked_at_sender: int | None = triage_field("acked@snd")
    write_ptr: int | str | None = triage_field("wr_off")
    sent_at_receiver: int | None = triage_field("sent@rcv")
    acked_at_receiver: int | str | None = triage_field("acked@rcv")
    read_ptr: int | str | None = triage_field("rd_addr", hex_or_unknown)
    fifo_size: int | None = triage_field("fifo")
    peer: str = triage_field("Peer")


def read_md(ep: Endpoint) -> ReceiverMd | SenderMd:
    if ep.role == "receiver":
        raw = read_from_device(ep.location, ep.config_addr, num_bytes=struct.calcsize(RECV_FMT))
        return ReceiverMd(*struct.unpack(RECV_FMT, raw))

    sent, n_down, wr, dstr_config_addr, _dstr_fifo, fifo, is_d2h = struct.unpack(
        SEND_FMT, read_from_device(ep.location, ep.config_addr, num_bytes=struct.calcsize(SEND_FMT))
    )
    acked_base = ep.config_addr + ep.md_size
    acked = [
        struct.unpack("<I", read_from_device(ep.location, acked_base + i * ep.acked_stride, num_bytes=4))[0]
        for i in range(max(n_down, 1))
    ]
    return SenderMd(sent, n_down, wr, dstr_config_addr, fifo, is_d2h, acked)


def read_interface(ep: Endpoint, callstack_provider: CallstackProvider) -> dict[str, int]:
    """The kernel's Socket{Sender,Receiver}Interface local, found by matching its config_addr."""
    fields = ("bytes_sent", "write_ptr") if ep.role == "sender" else ("bytes_acked", "read_ptr")
    dispatcher_data = callstack_provider.dispatcher_data
    for risc_name in ep.location.noc_block.risc_names:
        try:
            if dispatcher_data.is_idle_in_default_view(ep.location, risc_name):
                continue
            frames = callstack_provider.get_cached_callstacks(
                ep.location, risc_name, use_full_callstack=True
            ).kernel_callstack_with_message.callstack
        except TimeoutDeviceRegisterError:
            raise
        except Exception as e:
            log_warning_risc(risc_name, ep.location, f"socket callstack: {e}")
            continue
        for frame in frames:
            for var in frame.locals + frame.arguments:
                if var.value is None:
                    continue
                try:
                    # A kernel holding an array of interfaces keeps them out of reach of get_member.
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
                        log_warning_risc(risc_name, ep.location, f"socket interface: {e}")
                        continue
                    if values.pop("config_addr") == ep.config_addr:
                        return values
    return {}


def discover(inspector_data, id_mapping, run_checks) -> list[Endpoint]:
    sockets = list(inspector_data.getSockets().sockets)

    def to_device(metal_id: int):
        if not id_mapping.has_metal_device_id(metal_id):
            return None
        return run_checks.get_device_by_unique_id(id_mapping.get_unique_id(metal_id))

    # Fabric node id -> device id, so a peer owned by this host can be labelled like the Dev column.
    fabric_to_device_id: dict[tuple[int, int], int] = {}
    for s in sockets:
        for e in s.endpoints:
            device = to_device(int(e.chipId))
            if device is not None:
                fabric_to_device_id[(int(s.localMeshId), int(e.fabricChipId))] = device.id

    endpoints: list[Endpoint] = []
    for s in sockets:
        role = "sender" if s.isSender else "receiver"
        peer_mesh_id = int(s.peerMeshId)
        for e in s.endpoints:
            device = to_device(int(e.chipId))
            if device is None:
                continue
            endpoints.append(
                Endpoint(
                    role=role,
                    location=OnChipCoordinate(int(e.coreX), int(e.coreY), "logical", device, "tensix"),
                    node=node_label(int(s.localMeshId), int(e.fabricChipId)),
                    config_addr=int(s.configBufferAddress),
                    md_size=int(s.senderMdSizeBytes),
                    acked_stride=int(s.bytesAckedStrideBytes),
                    peers=[
                        Peer(
                            mesh_id=peer_mesh_id,
                            fabric_chip_id=int(p.fabricChipId),
                            core_x=int(p.coreX),
                            core_y=int(p.coreY),
                            device_id=fabric_to_device_id.get((peer_mesh_id, int(p.fabricChipId))),
                        )
                        for p in e.peers
                    ],
                )
            )

    return endpoints


def sender_row(ep: Endpoint, md: SenderMd, index: int) -> SocketRow:
    return SocketRow(
        role="sender",
        location=ep.location,
        node=ep.node,
        config_addr=ep.config_addr,
        downstream_config_addr=md.downstream_config_addr,
        downstream=index,
        sent_at_sender=ep.interface.get("bytes_sent", "?"),
        acked_at_sender=md.bytes_acked[index],
        write_ptr=ep.interface.get("write_ptr", "?"),
        sent_at_receiver=None,  # receiver's buffer
        acked_at_receiver=None,  # receiver's buffer
        read_ptr=None,  # receiver's buffer
        fifo_size=md.fifo_total_size,
        num_downstreams=md.num_downstreams,
        peer=ep.peers[index].label(),
    )


def receiver_row(ep: Endpoint, md: ReceiverMd) -> SocketRow:
    return SocketRow(
        role="receiver",
        location=ep.location,
        node=ep.node,
        config_addr=ep.config_addr,
        downstream_config_addr=None,  # sender's buffer
        downstream=None,  # sender-side index
        sent_at_sender=None,  # sender's buffer
        acked_at_sender=None,  # sender's buffer
        write_ptr=None,  # sender's buffer
        sent_at_receiver=md.bytes_sent,
        acked_at_receiver=ep.interface.get("bytes_acked", "?"),
        read_ptr=ep.interface.get("read_ptr", "?"),
        fifo_size=md.fifo_total_size,
        num_downstreams=None,  # sender's buffer
        peer=", ".join(p.label() for p in ep.peers),
    )


def endpoint_rows(ep: Endpoint) -> list[SocketRow]:
    if isinstance(ep.md, SenderMd):
        return [sender_row(ep, ep.md, i) for i in range(len(ep.peers))]
    elif isinstance(ep.md, ReceiverMd):
        return [receiver_row(ep, ep.md)]
    return []


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)
    endpoints = discover(get_inspector_data(args, context), get_metal_device_id_mapping(args, context), run_checks)
    if not endpoints:
        return None

    by_device: dict[int, list[Endpoint]] = {}
    for ep in endpoints:
        by_device.setdefault(ep.location.device.unique_id, []).append(ep)

    def collect_socket_rows(device: Device) -> list[SocketRow]:
        rows: list[SocketRow] = []
        for ep in by_device.get(device.unique_id, []):
            try:
                ep.md = read_md(ep)
            except TimeoutDeviceRegisterError:
                raise  # let run_per_device_check mark the device (and its remotes) broken
            except Exception as e:
                log_warning_location(ep.location, f"{ep.role} socket config buffer 0x{ep.config_addr:x}: {e}")
                continue
            ep.interface = read_interface(ep, callstack_provider)
            rows += endpoint_rows(ep)
        return rows

    return run_checks.run_per_device_check(collect_socket_rows)


if __name__ == "__main__":
    run_script()
