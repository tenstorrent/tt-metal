# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Reads what a router's channel counts and credits say (ie. free slots, credits received, etc). We cannot tell
which slots are pending and which are free, but we can tell how many are pending and how many are free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tt_metal.fabric.debug.visualizer.common.manifest_lookup import ManifestError, lookup
from tt_metal.fabric.debug.visualizer.common.router_paths import get_router
from tt_metal.fabric.debug.visualizer.decode.inputs import Input
from tt_metal.fabric.debug.visualizer.decode.routers import DecodedRouter

# The manifest names the flow rules read.
ACTIVE = "active"
WORKER_PRODUCERS = ("worker", "tensix_mux")  # the producers that connect through the worker protocol
CONNECTION_UNUSED = "unused_connection_value"
CONNECTION_OPEN = "open_connection_value"
CONNECTION_CLOSE_REQUESTED = "close_connection_request_value"
CONNECTION_NAMES = {CONNECTION_UNUSED, CONNECTION_OPEN, CONNECTION_CLOSE_REQUESTED}
STREAM_REGISTER = "stream_register"
L1_COUNTER = "l1_counter"
OWN_SENDER_COMPACT = "own_sender_compact"
PEER_SENDER_COMPACT = "peer_sender_compact"
# Each credit type's array in the sender's credit counters, and the array the peer's receiver keeps its total in.
CREDIT_ARRAYS = {
    "completed": ("to_sender_completion", "receiver_completion"),
    "acked": ("to_sender_ack", "receiver_ack"),
}
# What a credit of each type frees: a slot of the peer's receiver, or the sender's own slot (first-level ack).
CREDIT_FREES = {"completed": "receiver_slot", "acked": "sender_slot"}
WORD = 1 << 32  # counters wrap at 2^32


@dataclass(frozen=True)
class Count:
    """
    A count, read before and after the router's L1 when it comes from a stream register. A count from L1 is read once,
    so its before and after are the same. When the snapshot can't give it, both are None and `not_observable` says why.
    """

    before: int | None
    after: int | None
    not_observable: str | None = None


@dataclass(frozen=True)
class CreditFlow:
    """One type of credit coming back to a sender channel from the peer's receiver."""

    frees: str  # receiver_slot (a completion) or sender_slot (an ack)
    backing: str  # how the credits travel: stream_register or l1_counter, from credit_transport
    backing_reasons: list[str]  # why the VC uses that backing, from credit_transport
    pending: Count  # landed at the sender, not yet processed
    total_landed: Count  # landed since the router started
    peer_total_sent: Count  # sent by the peer's receiver since it started
    on_link: Count  # peer_total_sent - total_landed, signed: normally 0 (see _get_signed_difference)


@dataclass(frozen=True)
class ProducerCore:
    """The core of the worker on a worker channel, from conn_info.worker_xy."""

    x: int
    y: int
    whose: str  # current: the worker connected now; last: the last one to connect, or (0, 0) if none ever has


@dataclass(frozen=True)
class SenderFlow:
    depth: int  # the channel's slots
    producer: str | None  # the manifest's: worker, tensix_mux, a sibling router's path, or None when nothing feeds it
    connection: dict | None  # on a worker channel, the connection word as the type parser decodes it; None elsewhere
    producer_core: ProducerCore | None  # on a worker channel; None elsewhere
    unsent: Count  # packets the producer wrote that the router hasn't sent: depth - free_slots
    uncredited: Count  # packets the producer wrote that it hasn't been credited for: unsent, and sent but not credited
    credits: dict[str, CreditFlow]  # by credit type: completed, and acked with first-level ack
    speedy_batch: int | None  # on a speedy VC0, the batch credits are processed in: a pending up to it is normal
    within_bounds: bool | None  # whether 0 <= unsent <= uncredited <= depth wherever both were read; None if neither
    moved: bool | None  # whether a register count changed between before and after; None when none was read twice


@dataclass(frozen=True)
class ReceiverFlow:
    depth: int  # the channel's slots
    waiting: Count  # without first-level ack: packets arrived and not yet forwarded or delivered, from pkts_sent
    unacked: Count  # with first-level ack: packets arrived and not yet acked, from pkts_sent
    moved: bool | None  # whether pkts_sent changed between before and after; None when it wasn't read twice


def _not_observable(reason: str) -> Count:
    return Count(None, None, reason)


def _read_once(value: int | None, what: str) -> Count:
    return _not_observable(f"capture didn't read {what}") if value is None else Count(value, value)


def _read_stream(router: DecodedRouter, path: str, transform=lambda value: value) -> Count:
    """A stream reference's count, before and after, through `transform`."""
    stream = router.streams.get(path)
    if stream is None:
        return _not_observable(f"the builder allocated no register for {path}")
    if stream.before is None and stream.after is None:
        return _not_observable(f"capture didn't read {path}")
    return Count(
        None if stream.before is None else transform(stream.before),
        None if stream.after is None else transform(stream.after),
    )


def _get_signed_difference(later: int, earlier: int) -> int:
    """
    later - earlier, for two counters that wrap at 2^32, as a signed 32-bit value. A wrap between them still subtracts
    right, and a counter read after the one it trails, as two routers read at different moments while traffic moves,
    comes out a small negative number rather than one near 2^32.
    """
    difference = (later - earlier) % WORD
    return difference - WORD if difference >= WORD // 2 else difference


def _get_member(struct: dict, name: str, where: str):
    if name not in struct:
        raise ManifestError(f"{where} has no {name} member; its members are {sorted(struct)}")
    return struct[name]


@dataclass(frozen=True)
class _RouterView:
    """What every channel of one router reads."""

    path: str
    router: DecodedRouter
    node: dict
    where: str
    peer_path: str | None
    first_level_ack_vc0: bool
    speedy_batch: int | None
    transport: dict[str, tuple[str, list[str]]]  # each VC's backing and its reasons, from credit_transport


def _get_view(snapshot: Input, decoded_routers: Mapping[str, DecodedRouter], path: str) -> _RouterView:
    manifest = snapshot.manifest
    node, where = get_router(manifest, path)
    names = set(lookup(manifest, "enums.ConnectionValue.values", "object", of="integer"))
    if names != CONNECTION_NAMES:
        raise ManifestError(
            f"enums.ConnectionValue names {sorted(names)}; the flow rules know {sorted(CONNECTION_NAMES)}"
        )
    mesh_key = path.split("/")[0]
    transport_where = f"meshes.{mesh_key}.credit_transport"
    transport = {}
    for vc, entry in lookup(manifest, transport_where, "object", of="object").items():
        backing = lookup(entry, "backing", "string", where=f"{transport_where}.{vc}")
        if backing not in (STREAM_REGISTER, L1_COUNTER):
            raise ManifestError(
                f"{transport_where}.{vc}.backing is {backing!r}; the flow rules know {STREAM_REGISTER} and {L1_COUNTER}"
            )
        transport[vc] = (backing, lookup(entry, "reasons", "array", of="string", where=f"{transport_where}.{vc}"))
    speedy = lookup(node, "fields.speedy_vc0.value", "boolean", where=where)
    batch = lookup(node, "fields.sender_credit_amortization.value", "integer", where=where)
    return _RouterView(
        path=path,
        router=decoded_routers[path],
        node=node,
        where=where,
        peer_path=lookup(node, "link.peer", ("string", "null"), where=where),
        first_level_ack_vc0=lookup(node, "shape.vc0_bubble_flow_control", "boolean", where=where),
        speedy_batch=batch if speedy else None,
        transport=transport,
    )


def _get_worker_state(view: _RouterView, channel_path: str) -> tuple[dict | None, ProducerCore | None, Count]:
    """A worker channel's connection word, its worker's core, and the packets its workers haven't been credited for."""
    connection = view.router.regions.get(f"{channel_path}/fields/connection")
    conn_info = view.router.regions.get(f"{channel_path}/fields/conn_info")
    cursor = view.router.regions.get(f"{channel_path}/fields/buffer_index_sem")
    conn_where = f"{view.where}.{channel_path.replace('/', '.')}.fields.conn_info"

    connected = connection is not None and connection["name"] in (CONNECTION_OPEN, CONNECTION_CLOSE_REQUESTED)
    producer_core = None
    if conn_info is not None:
        worker_xy = _get_member(conn_info, "worker_xy", conn_where)
        producer_core = ProducerCore(
            _get_member(worker_xy, "x", f"{conn_where}.worker_xy"),
            _get_member(worker_xy, "y", f"{conn_where}.worker_xy"),
            "current" if connected else "last",
        )

    if connection is None:
        uncredited = _not_observable("capture didn't read the connection word")
    elif connection["name"] is None:
        uncredited = _not_observable(f"the connection word holds {connection['raw']}, which names no state")
    elif connected:
        uncredited = _not_observable(
            "a worker is connected: it keeps its write counter in its own variables, and the router writes its read "
            "counter into the worker's L1"
        )
    elif cursor is None or conn_info is None:
        uncredited = _not_observable("capture didn't read buffer_index_sem and conn_info")
    else:
        # With no worker connected, the cursor holds the last worker's write counter, and edm_read_counter the
        # router's read counter, kept current on every credit.
        cursor_where = f"{view.where}.{channel_path.replace('/', '.')}.fields.buffer_index_sem"
        written = _get_member(cursor, "write_counter", cursor_where)
        credited = _get_member(conn_info, "edm_read_counter", conn_where)
        uncredited = Count((written - credited) % WORD, (written - credited) % WORD)
    return connection, producer_core, uncredited


def _get_uncredited_from_sibling(
    view: _RouterView,
    decoded_routers: Mapping[str, DecodedRouter],
    manifest: dict,
    sibling_path: str,
    channel_path: str,
    channel_where: str,
    depth: int,
) -> Count:
    """depth - free_slots of the sibling's edge that lands on the channel: the sibling's view of its free slots."""
    target = f"{view.path}/{channel_path}"
    sibling_node, sibling_where = get_router(manifest, sibling_path)
    edges = lookup(sibling_node, "intra_chip_downstream_edges", "object", of="object", where=sibling_where)
    for vc, vc_edges in edges.items():
        for key, edge in vc_edges.items():
            edge_where = f"{sibling_where}.intra_chip_downstream_edges.{vc}.{key}"
            if lookup(edge, "downstream_channel", "string", where=edge_where) != target:
                continue
            sibling = decoded_routers.get(sibling_path)
            if sibling is None:
                return _not_observable(f"the producer, {sibling_path}, isn't in this snapshot")
            return _read_stream(
                sibling, f"intra_chip_downstream_edges/{vc}/{key}/fields/free_slots", lambda free: depth - free
            )
    raise ManifestError(
        f"{channel_where}.producer is {sibling_path!r}, but none of its intra_chip_downstream_edges lands on {target}"
    )


def _get_peer_total(
    view: _RouterView, decoded_routers: Mapping[str, DecodedRouter], peer_array: str, index: int
) -> Count:
    if view.peer_path is None:
        return _not_observable("the router's link has no peer")
    peer = decoded_routers.get(view.peer_path)
    if peer is None:
        return _not_observable(f"the peer, {view.peer_path}, isn't in this snapshot")
    totals = peer.regions.get(f"credit_counters/{peer_array}")
    if totals is None:
        return _not_observable(f"capture didn't read the peer's credit_counters/{peer_array}")
    if not 0 <= index < len(totals):
        raise ManifestError(f"{view.peer_path}'s credit_counters/{peer_array} has no element {index}")
    return Count(totals[index], totals[index])


def _check_index_space(node: dict, where: str, array: str, expected: str):
    space = lookup(node, f"credit_counters.{array}.index_space", "string", where=where)
    if space != expected:
        raise ManifestError(f"{where}.credit_counters.{array}.index_space is {space!r}; the flow rules read {expected}")


def _get_credit_flows(
    view: _RouterView,
    decoded_routers: Mapping[str, DecodedRouter],
    manifest: dict,
    channel: dict,
    channel_where: str,
    channel_path: str,
    vc: str,
) -> dict[str, CreditFlow]:
    credits = lookup(channel, "credits", "object", where=channel_where)
    unknown = sorted(set(credits) - set(CREDIT_ARRAYS))
    if unknown:
        raise ManifestError(f"{channel_where}.credits names {unknown}; the flow rules know {sorted(CREDIT_ARRAYS)}")
    if vc not in view.transport:
        raise ManifestError(f"meshes.{view.path.split('/')[0]}.credit_transport has no {vc}")
    backing, reasons = view.transport[vc]

    flows = {}
    for credit_type, reference in credits.items():
        path = f"{channel_path}/credits/{credit_type}"
        reference_where = f"{channel_where}.credits.{credit_type}"
        frees = CREDIT_FREES[credit_type]
        if reference is None:
            missing = _not_observable("the builder allocated no register for it")
            flows[credit_type] = CreditFlow(frees, backing, reasons, missing, missing, missing, missing)
            continue
        is_stream = "stream_id" in reference
        if is_stream != (backing == STREAM_REGISTER):
            kind = "a stream register" if is_stream else "an {array, index}"
            raise ManifestError(
                f"{reference_where} is {kind} reference, but meshes.{view.path.split('/')[0]}.credit_transport.{vc} "
                f"says {backing}"
            )
        if is_stream:
            used_up = _not_observable("a stream register's credits are used up as the sender processes them")
            flows[credit_type] = CreditFlow(
                frees, backing, reasons, _read_stream(view.router, path), used_up, used_up, used_up
            )
            continue

        own_array, peer_array = CREDIT_ARRAYS[credit_type]
        element = view.router.elements[path]
        if element.array != f"credit_counters/{own_array}":
            raise ManifestError(
                f"{reference_where}.array is {element.array!r}; the flow rules read credit_counters/{own_array}"
            )
        _check_index_space(view.node, view.where, own_array, OWN_SENDER_COMPACT)
        if view.peer_path is not None and view.peer_path in decoded_routers:
            peer_node, peer_where = get_router(manifest, view.peer_path)
            _check_index_space(peer_node, peer_where, peer_array, PEER_SENDER_COMPACT)
        landed = _read_once(element.value, path)
        sent = _get_peer_total(view, decoded_routers, peer_array, element.index)
        if landed.not_observable or sent.not_observable:
            on_link = _not_observable(f"it needs both totals: {landed.not_observable or sent.not_observable}")
        else:
            on_link = Count(
                _get_signed_difference(sent.before, landed.before), _get_signed_difference(sent.after, landed.after)
            )
        pending = _not_observable("the sender keeps its count of processed credits in its own variables")
        flows[credit_type] = CreditFlow(frees, backing, reasons, pending, landed, sent, on_link)
    return flows


def _is_within_bounds(unsent: Count, uncredited: Count, depth: int) -> bool | None:
    checks = []
    for sample in ("before", "after"):
        unsent_value, uncredited_value = getattr(unsent, sample), getattr(uncredited, sample)
        if unsent_value is not None:
            checks.append(0 <= unsent_value <= depth)
            if uncredited_value is not None:
                checks.append(unsent_value <= uncredited_value <= depth)
    return all(checks) if checks else None


def _has_moved(*counts: Count) -> bool | None:
    read_twice = [count for count in counts if count.before is not None and count.after is not None]
    return any(count.before != count.after for count in read_twice) if read_twice else None


def get_channel_flows(
    snapshot: Input, decoded_routers: Mapping[str, DecodedRouter], router_path: str
) -> dict[str, SenderFlow | ReceiverFlow]:
    """
    The counts and credits of each active channel of the router at `router_path`, by the channel's path in the router,
    such as channels/senders/vc0/ch0. `decoded_routers` holds every router of the snapshot: a sender channel's
    uncredited count can come from the sibling router that feeds it, and its credits' totals from the peer router. A
    channel no ERISC runs has none: its control words were never initialized. Raises ManifestError for a manifest
    value the rules can't use.
    """
    manifest = snapshot.manifest
    view = _get_view(snapshot, decoded_routers, router_path)
    flows = {}
    for kind in ("senders", "receivers"):
        for vc, channels in lookup(view.node, f"channels.{kind}", "object", of="object", where=view.where).items():
            for key, channel in channels.items():
                channel_path = f"channels/{kind}/{vc}/{key}"
                channel_where = f"{view.where}.channels.{kind}.{vc}.{key}"
                if lookup(channel, "status", "string", where=channel_where) != ACTIVE:
                    continue
                depth = lookup(channel, "ring_buffer.num_elements", "integer", where=channel_where)

                if kind == "receivers":
                    pkts_sent = _read_stream(view.router, f"{channel_path}/fields/pkts_sent")
                    if vc == "vc0" and view.first_level_ack_vc0:
                        waiting = _not_observable("with first-level ack, pkts_sent drops at the ack: see unacked")
                        flows[channel_path] = ReceiverFlow(depth, waiting, pkts_sent, _has_moved(pkts_sent))
                    else:
                        unacked = _not_observable("first-level ack is off on this VC")
                        flows[channel_path] = ReceiverFlow(depth, pkts_sent, unacked, _has_moved(pkts_sent))
                    continue

                producer = lookup(channel, "producer", ("string", "null"), where=channel_where)
                unsent = _read_stream(view.router, f"{channel_path}/fields/free_slots", lambda free: depth - free)
                connection = producer_core = None
                if producer in WORKER_PRODUCERS:
                    connection, producer_core, uncredited = _get_worker_state(view, channel_path)
                elif producer is None:
                    uncredited = _not_observable("nothing feeds the channel")
                else:
                    uncredited = _get_uncredited_from_sibling(
                        view, decoded_routers, manifest, producer, channel_path, channel_where, depth
                    )
                credits = _get_credit_flows(view, decoded_routers, manifest, channel, channel_where, channel_path, vc)
                flows[channel_path] = SenderFlow(
                    depth=depth,
                    producer=producer,
                    connection=connection,
                    producer_core=producer_core,
                    unsent=unsent,
                    uncredited=uncredited,
                    credits=credits,
                    speedy_batch=view.speedy_batch if vc == "vc0" else None,
                    within_bounds=_is_within_bounds(unsent, uncredited, depth),
                    moved=_has_moved(unsent, uncredited, *(credit.pending for credit in credits.values())),
                )
    return flows
