#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tabulate current x280 records without recomputing measurements.

Reads current-format JSONL records; records and console captures from before
the record builder are read by the frozen tabulator (docs/measurement.md
"Collection and reproduction"). Malformed records are errors.
"""

import argparse
import json
import math
import re
import sys
import hashlib
from pathlib import Path
from typing import NamedTuple


class ParseError(Exception):
    """A required result could not be parsed."""


# Reducers. Every value the tabulator computes from a record's raw fields, and
# the one conversion from ticks to nanoseconds, is one of these; the table
# specs below only compose them.


def lookup(record, path):
    """The value at key path `path` in `record`. ParseError names the first missing key."""
    value = record
    for depth, key in enumerate(path):
        if not isinstance(value, dict) or key not in value:
            raise ParseError("record has no %s" % ".".join(path[: depth + 1]))
        value = value[key]
    return value


def counter_value(record, path):
    """The raw counter or count at `path`: an integer in [0, 2**64)."""
    value = lookup(record, path)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        raise ParseError("%s is %r, not an unsigned 64-bit integer" % (".".join(path), value))
    return value


def raw_population(value):
    """Whether `value` is a raw sample population: `count` ticks, each an integer in [0, 2**63)."""
    ticks = value.get("raw_ticks") if isinstance(value, dict) else None
    return (
        isinstance(ticks, list)
        and value.get("count") == len(ticks)
        and all(type(t) is int and 0 <= t < 1 << 63 for t in ticks)
    )


def raw_samples(record, key):
    """The raw ticks of the sample population `key`, in recorded order."""
    population = lookup(record, (key,))
    if not raw_population(population):
        raise ParseError("%s is not a raw sample population" % key)
    return population["raw_ticks"]


def floor_index_percentile(samples, fraction):
    """The sample at index floor(fraction * (n - 1)) of the sorted population, or None when empty.

    The one percentile rule for every population: records carry raw samples only.
    """
    if not samples:
        return None
    ordered = sorted(samples)
    return ordered[int(fraction * (len(ordered) - 1))]


def endpoint_difference(start, end):
    """Ticks from raw endpoint `start` to `end` of one counter, modulo 2**64."""
    return (end - start) % (1 << 64)


def ticks_to_ns(ticks, hz):
    """`ticks` of a counter running at `hz` in nanoseconds; None when either is None."""
    if ticks is None or hz is None:
        return None
    return ticks * 1e9 / hz


# The spec vocabulary: each builds a function of one table Row from the
# reducers above. A value is None only where a reducer has none: a converted
# value of a row whose clock did not validate, or a statistic of an empty
# population. A missing or malformed input raises ParseError.


def field(*path):
    """The raw counter or count at `path`."""
    return lambda row: counter_value(row.record, path)


def samples(key):
    """The row's raw sample population `key`."""
    return lambda row: raw_samples(row.record, key)


def each(key, item):
    """`item` of every object in the row's nonempty list `key`."""

    def value(row):
        entries = lookup(row.record, (key,))
        if not isinstance(entries, list) or not entries:
            raise ParseError("%s is not a nonempty list" % key)
        return [counter_value(entry, (item,)) for entry in entries]

    return value


def pooled(key):
    """The population `key` of every record joined to the row, concatenated."""
    return lambda row: [t for record in row.joined for t in raw_samples(record, key)]


def pooled_sum(value):
    """`value` summed over the records joined to the row."""
    return lambda row: sum(value(row._replace(record=record)) for record in row.joined)


def constant(number):
    return lambda row: number


def percentile(population, fraction):
    return lambda row: floor_index_percentile(population(row), fraction)


def mean(population):
    def value(row):
        values = population(row)
        return sum(values) / len(values) if values else None

    return value


def count(population):
    return lambda row: len(population(row))


def difference(start, end):
    """Endpoint difference, modulo 2**64."""
    return lambda row: endpoint_difference(start(row), end(row))


def total(*values):
    def value(row):
        parts = [v(row) for v in values]
        return None if None in parts else sum(parts)

    return value


def product(*values):
    def value(row):
        parts = [v(row) for v in values]
        return None if None in parts else math.prod(parts)

    return value


def ratio(numerator, denominator):
    """A per-item value or a rate; a zero denominator is an error, not a figure."""

    def value(row):
        top, bottom = numerator(row), denominator(row)
        if top is None or bottom is None:
            return None
        if bottom == 0:
            raise ParseError("ratio with a zero denominator")
        return top / bottom

    return value


def ns(ticks):
    """`ticks` converted at the row's rate (Row.hz), the one conversion."""
    return lambda row: ticks_to_ns(ticks(row), row.hz)


class Column(NamedTuple):
    head: str
    value: object
    # An identity column is printed once in the caption when it holds one value
    # over the whole table; a measurement column always stays a column.
    identity: bool = False


def ident(key, head=None, value=None):
    """An identity column: `value`, or the record's `key` as recorded."""
    return Column(head or key, value or (lambda row: lookup(row.record, (key,))), identity=True)


class Join(NamedTuple):
    """The member's records sharing a row's `key` and matching `select`.

    Their `index` fields must be exactly 0 to the row's `count` field minus one.
    """

    key: str
    select: dict
    count: str
    index: str


class Table(NamedTuple):
    """One result table of a member.

    It holds the member's records that match every `select` entry, which maps a
    key, or a tuple key path, to the values allowed there. `columns` None
    recognises the records and tabulates none of them.
    """

    title: str
    select: dict
    columns: tuple = None
    join: Join = None


QUANTILES = (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))


def quantile_columns(name, population, unit, converted=False):
    """The floor-index p50/p90/p99 of `population` in `unit`, and converted to ns when asked."""
    columns = [Column("%s %s %s" % (name, q, unit), percentile(population, f)) for q, f in QUANTILES]
    if converted:
        columns += [Column("%s %s ns" % (name, q), ns(percentile(population, f))) for q, f in QUANTILES]
    return columns


COUNTERS = [ident("wall_counter", "wall counter"), ident("cpu_counter", "cpu counter")]

P2P_IDENTITY = [
    ident("first_direction", "first dir"),
    ident("participant", "who"),
    ident("phase"),
    ident("direction", "dir"),
    ident("role"),
    ident("sample_kind", "kind"),
    ident("samples", "N"),
    ident("warmup", "W"),
    ident("payload_size", "payload"),
    ident("same_chip", "same chip"),
    # null for a participant that opens no window (x280_same_tile, every Metal peer).
    ident("window_mode", "window", lambda row: lookup(row.record, ("window_mode",)) or "none"),
    ident("tlb_index", "tlb"),
    ident("tx_mode", "tx", lambda row: tx_mode(row.record)),
    ident("path_sentinel", "sentinel"),
] + COUNTERS
P2P_VALUES = quantile_columns("wall", samples("wall"), "ticks", converted=True) + quantile_columns(
    "cpu", samples("cpu_cycles"), "cycles"
)
P2P_TABLES = (Table("", {}, tuple(P2P_IDENTITY + P2P_VALUES)),)
METAL_PEER_MEMBERS = {"p2p_latency tensix_peer_tile", "p2p_latency erisc_peer_tile", "p2p_latency drisc_peer_tile"}
TX_MODES = ("wait_for_completion", "wait_for_departure")

BATCH_WALL = field("raw_wall_ticks")
BATCH_VALUES = [
    Column("wall ticks", BATCH_WALL),
    Column("cpu cycles", field("cpu_cycles")),
    Column("ns", ns(BATCH_WALL)),
    Column("ns/item", ratio(ns(BATCH_WALL), field("items"))),
    Column("cpu/item", ratio(field("cpu_cycles"), field("items"))),
    Column("cpu/wall", ratio(field("cpu_cycles"), BATCH_WALL)),
]

TB_IDENTITY = [
    ident("transfer_size", "size B"),
    ident("boundary"),
    ident("warmup", "W"),
    ident("samples", "N"),
    ident("items"),
    ident("logical_bytes", "logical B"),
]
TB_VALUES = COUNTERS + BATCH_VALUES + [Column("GB/s", ratio(field("logical_bytes"), ns(BATCH_WALL)))]
TB_TABLES = (Table("", {}, tuple(TB_IDENTITY + TB_VALUES)),)


def tb_x280_harts(row):
    """The run's hart count. Records written before the hart axis carry none: one hart ran."""
    return row.record.get("harts", 1)


def tb_x280_span(row):
    """mtime ticks from the first hart's t0 to the last hart's t1 (all harts read one mtime).

    A record from before the hart axis carries no brackets: its batch is the span."""
    if "hart_brackets" not in row.record:
        return counter_value(row.record, ("raw_wall_ticks",))
    starts = each("hart_brackets", "wall_start")(row)
    ends = each("hart_brackets", "wall_end")(row)
    if len(starts) != tb_x280_harts(row):
        raise ParseError("hart_brackets holds %d harts, harts is %r" % (len(starts), tb_x280_harts(row)))
    return endpoint_difference(min(starts), max(ends))


# Every hart moves the batch's logical bytes.
TB_X280_RUN_BYTES = product(tb_x280_harts, field("logical_bytes"))


def tb_erisc_writer_ticks(row):
    """Every writer's own batch wall ticks, one per writer."""
    ticks = each("writer_batches", "wall_ticks")(row)
    if len(ticks) != lookup(row.record, ("writers",)):
        raise ParseError("writer_batches holds %d writers, writers is %r" % (len(ticks), row.record["writers"]))
    if 0 in ticks:
        raise ParseError("a writer's batch has zero wall ticks")
    return ticks


def tb_erisc_writer_rate(pick):
    """GB/s of the writer `pick` (min or max) selects by rate, at the row's rate."""

    def value(row):
        ticks = tb_erisc_writer_ticks(row)
        if row.hz is None:
            return None
        return counter_value(row.record, ("logical_bytes",)) / ticks_to_ns(pick(ticks), row.hz)

    return value


# The writers' wall clocks share no stated origin, so no span is measured: the
# start gate releases their timed loops together (to within gate ns), and the
# run ends with the slowest writer, whose interval stands for the run's.
TB_ERISC_RUN_BYTES = product(field("writers"), field("logical_bytes"))
TB_ERISC_SLOWEST_NS = ns(lambda row: max(tb_erisc_writer_ticks(row)))
# With writers > 1 the batch columns are writer 0's.
TB_ERISC_TABLES = (
    Table(
        "",
        {},
        tuple(
            TB_IDENTITY
            + [ident("writers")]
            + TB_VALUES
            + [
                Column("slowest ticks", lambda row: max(tb_erisc_writer_ticks(row))),
                Column("run GB/s", ratio(TB_ERISC_RUN_BYTES, TB_ERISC_SLOWEST_NS)),
                Column("writer GB/s min", tb_erisc_writer_rate(max)),
                Column("writer GB/s max", tb_erisc_writer_rate(min)),
                Column("gate ns", field("gate_release_host_ns")),
            ]
        ),
    ),
)
# Records written before the store-width axis carry no store_width; they stored with sw.
# With harts > 1 the batch columns are hart 0's; span and run GB/s cover every hart.
TB_X280_TABLES = (
    Table(
        "",
        {},
        tuple(
            TB_IDENTITY
            + [
                ident("store_width", "store B", lambda row: row.record.get("store_width", 4)),
                ident("harts", "harts", tb_x280_harts),
            ]
            + TB_VALUES
            + [
                Column("span ticks", tb_x280_span),
                Column("run B", TB_X280_RUN_BYTES),
                Column("run GB/s", ratio(TB_X280_RUN_BYTES, ns(tb_x280_span))),
            ]
        ),
    ),
)

# PIPELINE_STAGE_BACKEND_ID_* in kernels/shared/ubench_backend_ids.h.
PS_X280_ROUTE_COMPUTE, PS_X280_ROUTE_FORWARD = {0x10200001}, {0x10200002}

PS_HEAD = [
    ident("packet_header_bytes", "packet B"),
    ident("route_buffer_capacity", "route cap"),
    ident("instrumentation", "instr"),
]
PS_X280_POLICY = [ident("wayenable"), ident("pinned_way", "pinned way"), ident("waymask_bits", "mask bits")]
PS_TAIL = [
    ident("warmup", "W"),
    ident("timed_count", "N"),
    ident("stage_witness", "witness"),
    ident("subject"),
    ident("run_nonce", "nonce"),
] + COUNTERS
PS_COMPUTE = [
    ident("fixture"),
    ident("header_count", "headers"),
    ident("prepared_input_base", "input base"),
    ident("prepared_input_bytes", "input B"),
]
PS_FORWARD = [
    ident("source_placement", "source"),
    ident("source_base", "source base"),
    ident("source_bytes", "source B"),
    ident("hop_count", "hops"),
    ident("boundary"),
]
# The fixed-source forwarding identity every route_forward record must carry:
# FS_PACKET_HEADER_BYTES and FS_ROUTE_BUFFER_CAPACITY
# (kernels/shared/fabric_shadow.h), X280_PINNED_INPUT_HEADER_BYTES and, on X280,
# X280_PINNED_INPUT_FIRST_HEADER_BASE (rt/x280_pinned_input.h), and kHopCount
# (tests/device/suites/pipeline_stage/records.hpp).
PS_FORWARD_IDENTITY = {
    "packet_header_bytes": {96},
    "route_buffer_capacity": {35},
    "source_placement": {"pinned_zero_device"},
    "source_base": {0x0A000AC0},
    "source_bytes": {96},
    "hop_count": {4},
    "boundary": {"decision"},
    "subject": {"router forward decision"},
}
PS_SAMPLES = [Column("cpu sample mean cycles", mean(samples("cpu_samples")))] + quantile_columns(
    "cpu sample", samples("cpu_samples"), "cycles"
)


def ps_columns(identity, instrumentation):
    columns = PS_HEAD + identity + PS_X280_POLICY + PS_TAIL + BATCH_VALUES
    return tuple(columns + PS_SAMPLES if instrumentation == "per_sample" else columns)


PS_COMPUTE_TABLES = tuple(
    Table(
        "X280 %s" % instrumentation,
        {"backend_id": PS_X280_ROUTE_COMPUTE, "instrumentation": {instrumentation}},
        ps_columns(PS_COMPUTE, instrumentation),
    )
    for instrumentation in ("per_sample", "off")
)
PS_FORWARD_TABLES = (
    Table(
        "X280 decide off",
        dict(PS_FORWARD_IDENTITY, backend_id=PS_X280_ROUTE_FORWARD, instrumentation={"off"}),
        ps_columns(PS_FORWARD, "off"),
    ),
)

# pi_mode_t in packet_injection_abi.h.
PI_LATENCY, PI_THROUGHPUT, PI_BACKPRESSURE = 1, 2, 3
# Bytes one produced-counter write publishes into LIM beside the headers: the
# producer's uint32_t produced counter (kernels/packet_injection/producer.hpp).
PI_COUNTER_WRITE_BYTES = 4


def pi_inflight(record):
    """Workers one consumer pass visits together. A record without the field
    predates the setting, when every pass visited one worker."""
    return record.get("inflight", 1)


PI_RUN = [
    ident("family"),
    ident("tile_count", "T"),
    ident("worker_count", "P"),
    ident("S"),
    ident("B"),
    ident("C"),
    ident("K"),
    ident("inflight", value=lambda row: pi_inflight(row.record)),
    ident("W"),
    ident("N"),
    ident("run_value", "run"),
]
PI_WORKER = [ident("worker_index", "worker"), ident("logical_x", "x"), ident("logical_y", "y")]
PI_JOIN = Join("run_value", {"record": {"worker"}}, "worker_count", "worker_index")
PI_AGGREGATE_WALL = difference(field("aggregate_start_mtime"), field("aggregate_end_mtime"))
PI_AGGREGATE_NS = ns(PI_AGGREGATE_WALL)
PI_HEADERS = product(field("worker_count"), field("N"))
PI_AGGREGATE = [
    Column("wall ticks", PI_AGGREGATE_WALL),
    Column("cpu cycles", difference(field("aggregate_start_rdcycle"), field("aggregate_end_rdcycle"))),
    Column("ns", PI_AGGREGATE_NS),
    Column("headers", PI_HEADERS),
    Column("ns/header", ratio(PI_AGGREGATE_NS, PI_HEADERS)),
]
PI_REQUESTS = field("throughput", "requests")
PI_USEFUL_BYTES = product(PI_REQUESTS, field("throughput", "useful_bytes_per_request"))
PI_HEADER_BYTES = pooled_sum(PI_USEFUL_BYTES)
PI_THROUGHPUT_WALL = difference(field("throughput", "start_ticks"), field("throughput", "end_ticks"))
PI_THROUGHPUT_NS = ns(PI_THROUGHPUT_WALL)
PI_WORKER_THROUGHPUT = [
    Column("wall ticks", PI_THROUGHPUT_WALL),
    Column("cpu cycles", difference(field("cpu_start"), field("cpu_end"))),
    Column("ns", PI_THROUGHPUT_NS),
    Column("requests", PI_REQUESTS),
    Column("ns/request", ratio(PI_THROUGHPUT_NS, PI_REQUESTS)),
    Column("useful GB/s", ratio(PI_USEFUL_BYTES, PI_THROUGHPUT_NS)),
    Column("slot GB/s", ratio(product(PI_REQUESTS, field("throughput", "slot_bytes_per_request")), PI_THROUGHPUT_NS)),
    Column("stall episodes", field("credit_stall_episodes")),
    Column("counter writes", field("producer_counter_writes")),
]
# A measured run's records: failed runs and errored intervals match no table.
PI_MEASURED = {"successful": {True}}
PI_AGGREGATE_SELECT = dict(PI_MEASURED, record={"aggregate"}, aggregate_error={None})
PI_THROUGHPUT_SELECT = {
    **PI_MEASURED,
    "record": {"worker"},
    "mode": {PI_THROUGHPUT},
    ("throughput", "interval_error"): {None},
}


def pi_tables(worker_identity, published):
    """A packet_injection member's tables; `published` adds the counter writes' bytes to the aggregate's."""
    header_bytes = [Column("header B", PI_HEADER_BYTES), Column("header GB/s", ratio(PI_HEADER_BYTES, PI_AGGREGATE_NS))]
    if published:
        published_bytes = total(
            PI_HEADER_BYTES, product(constant(PI_COUNTER_WRITE_BYTES), pooled_sum(field("producer_counter_writes")))
        )
        header_bytes += [
            Column("published B", published_bytes),
            Column("published GB/s", ratio(published_bytes, PI_AGGREGATE_NS)),
        ]
    return (
        Table("run", dict(PI_MEASURED, record={"run"}, mode={PI_THROUGHPUT})),
        Table(
            "throughput aggregate",
            dict(PI_AGGREGATE_SELECT, mode={PI_THROUGHPUT}),
            tuple(PI_RUN + PI_AGGREGATE + header_bytes),
            PI_JOIN,
        ),
        Table("worker throughput", PI_THROUGHPUT_SELECT, tuple(PI_RUN + worker_identity + PI_WORKER_THROUGHPUT)),
    )


PI_TENSIX_WORKER = [ident("risc")] + PI_WORKER
PI_TENSIX_TABLES = pi_tables(PI_TENSIX_WORKER, published=False) + (
    Table(
        "latency, pooled over workers",
        dict(PI_MEASURED, record={"run"}, mode={PI_LATENCY}),
        tuple(
            PI_RUN
            + [Column("samples", count(pooled("latency")))]
            + quantile_columns("latency", pooled("latency"), "ticks", converted=True)
        ),
        PI_JOIN,
    ),
    Table(
        "backpressure, pooled over workers",
        dict(PI_MEASURED, record={"run"}, mode={PI_BACKPRESSURE}),
        tuple(
            PI_RUN
            + [Column("stalls", count(pooled("credit_stalls")))]
            + quantile_columns("stall", pooled("credit_stalls"), "ticks", converted=True)
            + quantile_columns("stall rtt", pooled("stall_triggered_rtt"), "ticks", converted=True)
        ),
        PI_JOIN,
    ),
    Table(
        "latency and backpressure aggregate",
        dict(PI_AGGREGATE_SELECT, mode={PI_LATENCY, PI_BACKPRESSURE}),
        tuple(PI_RUN + PI_AGGREGATE),
    ),
    Table(
        "worker latency",
        dict(PI_MEASURED, record={"worker"}, mode={PI_LATENCY}),
        tuple(PI_RUN + PI_TENSIX_WORKER + quantile_columns("latency", samples("latency"), "ticks", converted=True)),
    ),
    Table(
        "worker backpressure",
        dict(PI_MEASURED, record={"worker"}, mode={PI_BACKPRESSURE}),
        tuple(
            PI_RUN
            + PI_TENSIX_WORKER
            + quantile_columns("stall", samples("credit_stalls"), "ticks", converted=True)
            + quantile_columns("stall cpu", samples("credit_stall_cpu_cycles"), "cycles")
            + quantile_columns("stall rtt", samples("stall_triggered_rtt"), "ticks", converted=True)
            + quantile_columns("stall rtt cpu", samples("stall_triggered_rtt_cpu_cycles"), "cycles")
        ),
    ),
)

CS_STAMPS = each("participants", "raw_timestamp")
CS_TABLES = (
    Table(
        "",
        {"measurement_schema": {1}},
        (
            ident("release"),
            ident("run_nonce", "nonce"),
            ident("chip_id", "chip"),
            ident("origins_comparable", "comparable origins"),
            ident("wall_counter", "wall counter"),
            Column("workers", count(CS_STAMPS)),
            # The workers' timers share no origin, so the range is a raw diagnostic.
            Column("raw range ticks", difference(percentile(CS_STAMPS, 0.0), percentile(CS_STAMPS, 1.0))),
        ),
    ),
)

# eth_credit's receiving forwarder stamps its window once per batch, after the
# batch, so the window covers the packets after its first batch.
EC_WALL = difference(field("receive_window", "start_ticks"), field("receive_window", "end_ticks"))
EC_NS = ns(EC_WALL)
EC_PACKETS = difference(field("first_batch_packets"), field("packets"))
EC_TABLES = (
    Table(
        "",
        {("receive_window", "interval_error"): {None}},
        (
            ident("case"),
            ident("P"),
            ident("N_LINK"),
            ident("N_DELIVER"),
            ident("SLOT_SIZE", "slot B"),
            ident("payload_bytes_per_request", "payload B"),
            ident("BATCH"),
            ident("bidirectional", "both"),
            ident("p"),
            ident("tx_chip", "tx"),
            ident("rx_chip", "rx"),
            ident("x280_hw_index", "l2cpu"),
            ident("x280_pll_mhz", "pll MHz"),
            Column("packets", field("packets")),
            Column("k1", field("first_batch_packets")),
            Column("wall ticks", EC_WALL),
            Column("ns", EC_NS),
            Column("window packets", EC_PACKETS),
            Column("ns/packet", ratio(EC_NS, EC_PACKETS)),
            Column("Mpkt/s", ratio(product(EC_PACKETS, constant(1000)), EC_NS)),
            Column(
                "useful GB/s", ratio(product(EC_PACKETS, field("receive_window", "useful_bytes_per_request")), EC_NS)
            ),
            Column("slot GB/s", ratio(product(EC_PACKETS, field("receive_window", "slot_bytes_per_request")), EC_NS)),
        ),
    ),
)
# The Ethernet hop alone: E1 stamps the same window on its ERISC wall clock, and
# a request's useful bytes are its payload. The 400G per-link rate in packets
# is 50 GB/s over the payload.
EC_HOP_TABLES = (
    Table(
        "",
        {("receive_window", "interval_error"): {None}},
        (
            ident("case"),
            ident("P"),
            ident("N_LINK"),
            ident("SLOT_SIZE", "slot B"),
            ident("payload_bytes_per_request", "payload B"),
            ident("BATCH"),
            ident("tx_chip", "tx"),
            ident("rx_chip", "rx"),
            Column("packets", field("packets")),
            Column("k1", field("first_batch_packets")),
            Column("wall ticks", EC_WALL),
            Column("ns", EC_NS),
            Column("window packets", EC_PACKETS),
            Column("ns/packet", ratio(EC_NS, EC_PACKETS)),
            Column("Mpkt/s", ratio(product(EC_PACKETS, constant(1000)), EC_NS)),
            Column(
                "payload GB/s", ratio(product(EC_PACKETS, field("receive_window", "useful_bytes_per_request")), EC_NS)
            ),
            Column("slot GB/s", ratio(product(EC_PACKETS, field("receive_window", "slot_bytes_per_request")), EC_NS)),
            Column("400G Mpkt/s", ratio(constant(50000), field("payload_bytes_per_request"))),
        ),
    ),
)
# One stretch of the path alone (`legs` names its writes), on the window of the
# kernel that paces it: the sending forwarder for the NoC hop, counting what it
# sent.
EC_LEG_TABLES = (Table("", EC_TABLES[0].select, (ident("legs"),) + EC_TABLES[0].columns),)
# Its diagnostic records: recognised, tabulated by no table.
EC_DIAGNOSTIC = (Table("", {}),)


class DerivedRow(NamedTuple):
    """A derived table's row: only its Markdown label."""

    label: str


def tx_mode(record):
    """A p2p record's transmit mode; records written before the setting waited for completion."""
    return record.get("tx_mode", "wait_for_completion")


def turn_split(member, rows):
    """The Metal peer's TURN at each transmit mode and their difference, per first direction, as a
    Tabulated titled "<member> TURN split"; None unless some first direction has both modes.

    Each mode pools the peer's TURN samples over every record of that mode. Both terms are the
    peer's own wall ticks, so the difference crosses no clock. ns columns convert at the records'
    one validated rate and are blank when they hold more than one rate or an unvalidated one.
    Raises ParseError when the records name more than one wall counter.
    """
    turns = [row for row in rows if row.record.get("participant") != "X280" and row.record.get("sample_kind") == "TURN"]
    if len({row.record.get("wall_counter") for row in turns}) > 1:
        raise ParseError("[%s] peer TURN records name more than one wall counter" % member)
    cells = []
    for first in sorted({row.record.get("first_direction") for row in turns}):
        by_mode = [
            [row for row in turns if row.record.get("first_direction") == first and tx_mode(row.record) == mode]
            for mode in TX_MODES
        ]
        if not all(by_mode):
            continue
        rates = {(row.hz, row.warning) for mode_rows in by_mode for row in mode_rows}
        hz = None
        if len(rates) == 1:
            ((hz, warning),) = rates
            hz = None if warning else hz
        pooled = [[t for row in mode_rows for t in raw_samples(row.record, "wall")] for mode_rows in by_mode]
        stats = [(sum(p) / len(p), floor_index_percentile(p, 0.5)) for p in pooled]
        stats.append(tuple(c - d for c, d in zip(*stats)))
        line = [len(p) for p in pooled]
        for mean_ticks, p50_ticks in stats:
            line += [mean_ticks, p50_ticks, ticks_to_ns(mean_ticks, hz), ticks_to_ns(p50_ticks, hz)]
        cells.append([first] + [cell(v) for v in line])
    if not cells:
        return None
    heads = ["first dir", "completion N", "departure N"] + [
        "%s %s" % (term, stat)
        for term in ("completion", "departure", "ack")
        for stat in ("mean ticks", "p50 ticks", "mean ns", "p50 ns")
    ]
    return Tabulated(
        member + " TURN split",
        ["wall counter=%s" % cell(turns[0].record.get("wall_counter"))],
        heads,
        [False] + [True] * (len(heads) - 1),
        [DerivedRow(line[0]) for line in cells],
        cells,
    )


# Blackhole's NoC grid; rows and columns are rings (harness/inventory/inventory.hpp).
NOC_GRID = (17, 12)


def hops(src, dst, noc):
    """Router links from NoC0 coordinate `src` to `dst` on NoC `noc`, by the rule of
    harness/inventory/inventory.hpp hops(): NoC 0 runs right then down, NoC 1 the reverse.

    Raises ParseError for an off-grid endpoint or a NoC other than 0 or 1.
    """
    for end in (src, dst):
        if not all(0 <= c < n for c, n in zip(end, NOC_GRID)):
            raise ParseError("NoC0 coordinate %r is outside the %dx%d grid" % (end, *NOC_GRID))
    if noc not in (0, 1):
        raise ParseError("NoC %r is neither 0 nor 1" % (noc,))
    if noc == 1:
        src, dst = dst, src
    return sum((d - s) % n for s, d, n in zip(src, dst, NOC_GRID))


class Leg(NamedTuple):
    """A NoC write from the core at field prefix `src` to `dst` (each named by
    <prefix>_x and <prefix>_y), issued on the NoC in field `noc`. `recorded`, when
    set, names the fields holding the leg's hops and its acknowledgement's hops as
    the host recorded them, which the table shows instead of deriving them."""

    name: str
    src: str
    dst: str
    noc: str
    recorded: tuple = None


class Placement(NamedTuple):
    """A member's placement table: the records matching `select`, keyed by the
    `keys` fields, showing the `cores` coordinates and each of `legs`."""

    select: dict
    keys: tuple
    cores: tuple
    legs: tuple


def placement_table(member, rows, spec):
    """`member`'s placement as a Tabulated titled "<member> placement"; None when no record
    matching `spec.select` carries any of its cores' coordinates.

    Each leg shows its NoC, its hops and its acknowledgement's hops, taken to return on
    the same NoC: recorded where the leg names recorded fields, derived otherwise. Records with the same cells collapse into one row counting them. A
    coordinate or NoC a record does not carry leaves its cells blank.
    """

    def xy(record, prefix):
        x, y = record.get(prefix + "_x"), record.get(prefix + "_y")
        return None if x is None or y is None else (x, y)

    lines = {}
    for row in rows:
        record = row.record
        if unmet(record, spec.select) or all(xy(record, core) is None for core in spec.cores):
            continue
        line = [record.get(key) for key in spec.keys]
        line += ["(%d, %d)" % xy(record, core) if xy(record, core) else None for core in spec.cores]
        for leg in spec.legs:
            noc = record.get(leg.noc)
            if leg.recorded:
                line += [noc] + [record.get(field) for field in leg.recorded]
                continue
            src, dst = xy(record, leg.src), xy(record, leg.dst)
            known = src is not None and dst is not None and noc is not None
            line += [noc, hops(src, dst, noc) if known else None, hops(dst, src, noc) if known else None]
        key = tuple(cell(v) for v in line)
        if key not in lines:
            lines[key] = [row.label, 0]
        lines[key][1] += 1
    if not lines:
        return None
    heads = list(spec.keys) + list(spec.cores)
    heads += ["%s %s" % (leg.name, h) for leg in spec.legs for h in ("noc", "hops", "ack hops")] + ["records"]
    cells = [list(key) + [str(count)] for key, (_, count) in lines.items()]
    right = [all(c == "" or c.isdigit() for c in column) for column in zip(*cells)]
    return Tabulated(member + " placement", [], heads, right, [DerivedRow(label) for label, _ in lines.values()], cells)


P2P_METAL_PLACEMENT = {
    peer: Placement(
        {},
        ("participant", "chip_id"),
        ("x280_noc", f"{peer}_noc0"),
        (
            Leg("x280->peer", "x280_noc", f"{peer}_noc0", "x280_noc"),
            Leg("peer->x280", f"{peer}_noc0", "x280_noc", f"{peer}_noc"),
        ),
    )
    for peer in ("tensix", "erisc", "drisc")
}
PI_PLACEMENT = Placement(
    {"record": {"worker"}},
    ("worker_index", "chip_id"),
    ("x280_noc", "noc0"),
    (Leg("worker->x280", "noc0", "x280_noc", "noc"), Leg("x280->worker", "x280_noc", "noc0", "x280_noc")),
)
PLACEMENT = {
    "p2p_latency x280_same_tile": Placement({}, ("participant", "chip_id"), ("x280_noc",), ()),
    "p2p_latency x280_peer_tile": Placement(
        {},
        ("participant", "chip_id"),
        ("x280_noc", "peer_x280_noc"),
        (
            Leg("a->b", None, None, "x280_noc", ("hops_a_to_b", "hops_b_to_a")),
            Leg("b->a", None, None, "x280_noc", ("hops_b_to_a", "hops_a_to_b")),
        ),
    ),
    "p2p_latency tensix_peer_tile": P2P_METAL_PLACEMENT["tensix"],
    "p2p_latency erisc_peer_tile": P2P_METAL_PLACEMENT["erisc"],
    "p2p_latency drisc_peer_tile": P2P_METAL_PLACEMENT["drisc"],
    "packet_injection tensix_to_x280": PI_PLACEMENT,
    "packet_injection erisc_to_x280_header": PI_PLACEMENT,
    "eth_credit eth_credit": Placement(
        {},
        ("tx_chip", "rx_chip", "p"),
        ("tx_x280_noc0", "tx_erisc_noc0", "rx_erisc_noc0", "rx_x280_noc0"),
        (
            Leg("x280->e0", "tx_x280_noc0", "tx_erisc_noc0", "tx_x280_noc"),
            Leg("e0->x280", "tx_erisc_noc0", "tx_x280_noc0", "tx_erisc_noc"),
            Leg("e1->x280", "rx_erisc_noc0", "rx_x280_noc0", "rx_erisc_noc"),
            Leg("x280->e1", "rx_x280_noc0", "rx_erisc_noc0", "rx_x280_noc"),
        ),
    ),
}


# eth_credit's credit round trip as built, set against an on-paper estimate: the NoC
# traversals on its loop and a processing allowance per stage that handles the packet or its
# credit. The hop loop runs W2 into E0, W5 out of E1 and its acknowledgement, which E1 waits
# for before W6 and R2, and R3 out of E0; the consume loop adds W6 into the receiving
# forwarder and its R1 back. A traversal of h router links costs ~5 + 9h + ~5 NoC cycles
# (BlackholeA0/NoC/README.md, Performance) at its chip's AICLK, the NoC's clock domain
# (BlackholeA0/PCIExpressTile/README.md). The hop loop's stages are E0 and E1 twice each at
# their chips' AICLK and the sending forwarder at the X280 PLL; the consume loop adds E1 once
# more and the receiving forwarder. The two Ethernet crossings are in neither term.
EC_RT_HOP_TRAVERSALS = (
    ("tx_x280_noc0", "tx_erisc_noc0", "tx_x280_noc", "tx_erisc_clock"),
    ("rx_erisc_noc0", "rx_x280_noc0", "rx_erisc_noc", "rx_erisc_clock"),
    ("rx_x280_noc0", "rx_erisc_noc0", "rx_erisc_noc", "rx_erisc_clock"),
    ("tx_erisc_noc0", "tx_x280_noc0", "tx_erisc_noc", "tx_erisc_clock"),
)
EC_RT_TRAVERSALS = {
    "hop": EC_RT_HOP_TRAVERSALS,
    "consume": EC_RT_HOP_TRAVERSALS
    + (
        ("rx_erisc_noc0", "rx_x280_noc0", "rx_erisc_noc", "rx_erisc_clock"),
        ("rx_x280_noc0", "rx_erisc_noc0", "rx_x280_noc", "rx_erisc_clock"),
    ),
}
# Per loop: (ERISC stages per chip's clock evidence, X280 stages).
EC_RT_STAGES = {
    "hop": ((("tx_erisc_clock", 2), ("rx_erisc_clock", 2)), 1),
    "consume": ((("tx_erisc_clock", 2), ("rx_erisc_clock", 3)), 2),
}


def ec_rt_loop(row):
    """The record's loop, as a key of EC_RT_TRAVERSALS."""
    loop = lookup(row.record, ("loop",))
    if loop not in EC_RT_TRAVERSALS:
        raise ParseError("unknown round-trip loop %r" % (loop,))
    return loop


def ec_core(row, prefix):
    """The NoC0 coordinate a record's `prefix`_x and `prefix`_y fields hold."""
    return (lookup(row.record, (prefix + "_x",)), lookup(row.record, (prefix + "_y",)))


def ec_aiclk(row, key):
    """The rate the ERISC clock evidence at `key` validates, or None when it fails a check."""
    try:
        hz, warnings = validate_clock(lookup(row.record, (key,)))
    except ValueError as error:
        raise ParseError("%s: %s" % (key, error))
    return None if warnings else hz


def ec_rt_hops(row):
    """Router links over the round trip's NoC traversals."""
    return sum(
        hops(ec_core(row, src), ec_core(row, dst), lookup(row.record, (noc,)))
        for src, dst, noc, _ in EC_RT_TRAVERSALS[ec_rt_loop(row)]
    )


def ec_rt_noc_ns(row):
    """The round trip's NoC traversals in ns; None when a chip's AICLK did not validate."""
    total = 0.0
    for src, dst, noc, clock in EC_RT_TRAVERSALS[ec_rt_loop(row)]:
        hz = ec_aiclk(row, clock)
        if hz is None:
            return None
        total += (10 + 9 * hops(ec_core(row, src), ec_core(row, dst), lookup(row.record, (noc,)))) * 1e9 / hz
    return total


def ec_rt_estimate(cycles):
    """The on-paper round trip in ns with `cycles` of processing per stage."""

    def value(row):
        noc = ec_rt_noc_ns(row)
        if noc is None:
            return None
        erisc, x280 = EC_RT_STAGES[ec_rt_loop(row)]
        stages = sum(count * 1e9 / ec_aiclk(row, clock) for clock, count in erisc)
        stages += x280 * 1e9 / (lookup(row.record, ("x280_pll_mhz",)) * 1e6)
        return noc + cycles * stages

    return value


EC_RT_TICKS = field("round_trip_ticks")
EC_RT_MEAN_NS = ratio(ns(EC_RT_TICKS), field("round_trips"))
EC_ROUND_TRIP_TABLES = (
    Table(
        "",
        {},
        (
            ident("case"),
            ident("loop"),
            ident("P"),
            ident("N_LINK"),
            ident("N_DELIVER"),
            ident("SLOT_SIZE", "slot B"),
            ident("BATCH"),
            ident("p"),
            ident("tx_chip", "tx"),
            ident("rx_chip", "rx"),
            ident("x280_pll_mhz", "pll MHz"),
            Column("round trips", field("round_trips")),
            Column("ticks", EC_RT_TICKS),
            Column("mean ns", EC_RT_MEAN_NS),
            Column("NoC hops", ec_rt_hops),
            Column("NoC ns", ec_rt_noc_ns),
            Column("estimate 30 ns", ec_rt_estimate(30)),
            Column("estimate 80 ns", ec_rt_estimate(80)),
        ),
    ),
)

# Each member's table spec, keyed by its records' suite and member.
MEMBERS = {
    "p2p_latency x280_same_tile": P2P_TABLES,
    "p2p_latency x280_peer_tile": (
        Table(
            "",
            {},
            tuple(
                P2P_IDENTITY
                + [
                    ident("hw_index", "hw"),
                    ident("peer_hw_index", "peer"),
                    ident("hops_a_to_b", "hops a->b"),
                    ident("hops_b_to_a", "hops b->a"),
                    ident("lap", "lap hops", total(field("hops_a_to_b"), field("hops_b_to_a"))),
                ]
                + P2P_VALUES
            ),
        ),
    ),
    "p2p_latency tensix_peer_tile": P2P_TABLES,
    "p2p_latency erisc_peer_tile": P2P_TABLES,
    "p2p_latency drisc_peer_tile": P2P_TABLES,
    "transport_bandwidth x280_to_tensix": TB_X280_TABLES,
    "transport_bandwidth tensix_to_x280": TB_TABLES,
    "transport_bandwidth erisc_to_x280": TB_ERISC_TABLES,
    "pipeline_stage route_compute": PS_COMPUTE_TABLES,
    "pipeline_stage route_forward": PS_FORWARD_TABLES,
    "packet_injection tensix_to_x280": PI_TENSIX_TABLES,
    "packet_injection erisc_to_x280_header": pi_tables(PI_WORKER, published=True),
    "core_sync x280_director_characterization": CS_TABLES,
    "eth_credit eth_credit": EC_TABLES,
    "eth_credit eth_credit_hop": EC_HOP_TABLES,
    "eth_credit eth_credit_leg": EC_LEG_TABLES,
    "eth_credit eth_credit_round_trip": EC_ROUND_TRIP_TABLES,
    "eth_credit eth_credit_kernel": EC_DIAGNOSTIC,
    "eth_credit eth_credit_attribution": EC_DIAGNOSTIC,
    "eth_credit eth_credit_link": EC_DIAGNOSTIC,
}
# Tables derived across a member's records, which the per-record specs above cannot express;
# each returns a Tabulated or None.
DERIVED = {member: [lambda m, rows, spec=spec: placement_table(m, rows, spec)] for member, spec in PLACEMENT.items()}
for member in METAL_PEER_MEMBERS:
    DERIVED[member].append(turn_split)

# eth_credit's whole path beside each of its legs measured alone, by the writes inside the leg's
# window: the stretches matched at the path's shape, the pointer stores at its P, since no shape
# but P changes a pointer store. The NoC hop, Ethernet hop and receive side are disjoint and cover
# the path.
EC_SHAPE = ("P", "N_LINK", "N_DELIVER", "SLOT_SIZE", "BATCH")
EC_CEILING_STRETCHES = (
    ("NoC hop", "W1 W2 R3"),
    ("W1", "W1"),
    ("Ethernet hop", "W3 W4 R2"),
    ("receive side", "W5 W6 R1"),
)
EC_CEILING_STORES = (("W2", "W2"), ("R1", "R1"))
EC_MPKT = ratio(product(EC_PACKETS, constant(1000)), EC_NS)


def ec_ceilings(rows):
    """eth_credit's one-direction whole-path packet rate beside each leg's alone, the slowest leg,
    the credit loop's bound, which of the two binds and the path's share of it, as a Tabulated
    titled "eth_credit ceilings"; None without a one-direction whole-path row.

    The credit loop allows N_LINK packets per hop-loop round trip, the mean of the hop-loop
    eth_credit_round_trip rows at one packet in flight (N_LINK = 1) and the path's slot size.

    A stretch's rate is its Mpkt/s at the path's shape; a pointer store's is BATCH times its
    million stores per second at the path's P, as the forwarder stores once per batch. Each is the
    mean over every row of that leg and key (every p and run), at the row's own clock; a row left
    raw is no measurement. Ethernet hop records written before records named their legs are
    recognised by their member.
    """
    names = {legs: name for name, legs in EC_CEILING_STRETCHES + EC_CEILING_STORES}
    stores = [name for name, _ in EC_CEILING_STORES]
    leg_names = [name for name, _ in EC_CEILING_STRETCHES] + stores
    groups = {}
    for row in rows:
        if row.member == "eth_credit eth_credit" and not row.record.get("bidirectional"):
            name = "path"
        elif row.member == "eth_credit eth_credit_hop":
            name = "Ethernet hop"
        elif row.member == "eth_credit eth_credit_leg":
            name = names.get(row.record.get("legs"))
        else:
            continue
        rate = EC_MPKT(row) if name else None
        if rate is not None:
            key = (lookup(row.record, ("P",)),) if name in stores else tuple(lookup(row.record, (k,)) for k in EC_SHAPE)
            groups.setdefault((name, key), []).append((row, rate))
    loops = {}
    for row in rows:
        if row.member == "eth_credit eth_credit_round_trip" and row.record.get("loop") == "hop":
            trip = EC_RT_MEAN_NS(row) if lookup(row.record, ("N_LINK",)) == 1 else None
            if trip is not None:
                loops.setdefault(lookup(row.record, ("SLOT_SIZE",)), []).append(trip)
    mean = {group: sum(r for _, r in measured) / len(measured) for group, measured in groups.items()}
    shown, cells = [], []
    for shape in sorted(key for name, key in groups if name == "path"):
        # shape[0] is P and shape[4] BATCH (EC_SHAPE).
        legs = [mean.get((name, shape)) for name, _ in EC_CEILING_STRETCHES]
        legs += [shape[4] * mean[(name, shape[:1])] if (name, shape[:1]) in mean else None for name in stores]
        measured = [(rate, name) for rate, name in zip(legs, leg_names) if rate is not None]
        slowest = min(measured) if measured else (None, None)
        trips = loops.get(shape[3])
        # shape[1] is N_LINK and shape[3] SLOT_SIZE (EC_SHAPE).
        credit = shape[1] * 1000 * len(trips) / sum(trips) if trips else None
        bounds = [b for b in (slowest, (credit, "credit loop")) if b[0] is not None]
        bound = min(bounds) if bounds else (None, None)
        path = mean[("path", shape)]
        line = list(shape) + [path] + legs + [slowest[1], credit, bound[1], path / bound[0] if bound[0] else None]
        shown.append(groups[("path", shape)][0][0])
        cells.append([cell(value) for value in line])
    if not cells:
        return None
    heads = ["P", "N_LINK", "N_DELIVER", "slot B", "BATCH", "path Mpkt/s"]
    heads += ["%s Mpkt/s" % name for name in leg_names]
    heads += ["slowest leg", "credit loop Mpkt/s", "bound", "path/bound"]
    right = [True] * (len(heads) - 4) + [False, True, False, True]
    return Tabulated("eth_credit ceilings", [], heads, right, shown, cells)


# Tables computed across members' rows, which no one member's tables can hold; each returns a
# Tabulated or None.
CROSS_MEMBER = (ec_ceilings,)


def cross_member_tables(rows):
    """The CROSS_MEMBER tables of `rows`, every member's, that hold any row."""
    return [table for table in (derive(rows) for derive in CROSS_MEMBER) if table is not None]


# A table's title names it in both outputs.
assert all(
    len({table.title for table in tables}) == len(tables) for tables in MEMBERS.values()
), "two tables of one member share a title"


class Row(NamedTuple):
    """One record as a table row.

    `path` and `line` locate it and `label` leads its Markdown row. `hz` is the
    rate its ticks convert at (conversion()): validated, or measured with
    `warning` naming the checks it failed, or None with `raw` saying why its
    ticks stay raw. `joined` holds the records its table's Join selects.
    """

    record: dict
    member: str
    table: Table
    path: str
    line: int
    label: str = ""
    hz: int = None
    raw: str = None
    warning: str = None
    joined: tuple = ()

    @property
    def source(self):
        return "%s:%d" % (self.path, self.line)


def indices_cover(records, index, count):
    """Whether the `index` fields of `records` are exactly 0 to `count` - 1, each once."""
    values = [record.get(index) for record in records]
    return type(count) is int and len(values) == count and set(values) == set(range(count))


def unmet(record, select):
    """The `select` entries `record` does not satisfy, as text; empty when it matches."""
    misses = []
    for key, allowed in select.items():
        path = key if isinstance(key, tuple) else (key,)
        try:
            value = lookup(record, path)
        except ParseError:
            misses.append("%s absent" % ".".join(path))
            continue
        if value not in allowed:
            misses.append("%s %r not in %s" % (".".join(path), value, sorted(allowed, key=repr)))
    return misses


def table_row(record, path, line, label=""):
    """`record`, from line `line` of `path`, as a Row of the one table of its member it matches.

    Raises ParseError for an unregistered member, or a record
    matching no table or more than one; the error for no match names the
    nearest table's unmet entries.
    """
    source = "%s:%d" % (path, line)
    member = "%s %s" % (record.get("suite"), record.get("member"))
    tables = MEMBERS.get(member)
    if tables is None:
        raise ParseError("%s: [%s] is not a registered member" % (source, member))
    misses = [(table, unmet(record, table.select)) for table in tables]
    matches = [table for table, missed in misses if not missed]
    if len(matches) != 1:
        nearest, missed = min(misses, key=lambda entry: len(entry[1]))
        raise ParseError(
            "%s: [%s] record matches %d tables%s"
            % (
                source,
                member,
                len(matches),
                "" if matches else "; nearest (%s) needs %s" % (nearest.title, "; ".join(missed)),
            )
        )
    hz, raw, warning = conversion(record)
    return Row(record, member, matches[0], path, line, label, hz, raw, warning)


def reject_json_constant(value):
    raise ValueError("non-standard numeric constant %s" % value)


def unique_json_object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("duplicate key %r" % key)
        record[key] = value
    return record


def decode_record(source, line):
    """One JSONL line as a record object: no duplicate key and no NaN or infinity."""
    try:
        record = json.loads(line, object_pairs_hook=unique_json_object, parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ParseError("%s: not JSON: %s" % (source, error))
    if not isinstance(record, dict):
        raise ParseError("%s: JSONL record must be an object, got %s" % (source, type(record).__name__))
    return record


def parse_jsonl(path, text, label=""):
    """Rows from current-format harness JSONL, each with Markdown label `label`.

    Both outputs and the campaign runner read records through it. Raises
    ParseError as decode_record() and table_row() do.
    """
    rows = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            rows.append(table_row(decode_record("%s:%d" % (path, line_no), line), path, line_no, label))
    return rows


def cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "%.3f" % value
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


class Tabulated(NamedTuple):
    """One computed table: its title, the "head=value" identities constant over it, its column
    heads and which are right-aligned, its Rows, and their formatted cells in head order."""

    title: str
    constant: list
    heads: list
    right: list
    rows: list
    cells: list


def tabulate(member, rows):
    """`member`'s rows, one Tabulated per spec table holding any.

    Identity columns holding one value over a table move to `constant`. A row
    converted at an unvalidated measured rate, or left raw with its converted
    values blank, says so in a trailing "clock" column. Raises ParseError,
    naming the row and column, for an input a column cannot read.
    """
    out = []
    for table in MEMBERS[member]:
        table_rows = [row for row in rows if row.table is table]
        if table.columns is None or not table_rows:
            continue
        join = table.join
        if join is not None:
            table_rows = [
                row._replace(
                    joined=tuple(
                        other.record
                        for other in rows
                        if other.record.get(join.key) == row.record.get(join.key)
                        and not unmet(other.record, join.select)
                    )
                )
                for row in table_rows
            ]
        values = []
        for row in table_rows:
            if join is not None:
                count = row.record.get(join.count)
                if not indices_cover(row.joined, join.index, count):
                    raise ParseError(
                        "%s: [%s] %s %r joins %s %s, not 0..%s - 1"
                        % (
                            row.source,
                            member,
                            join.key,
                            row.record.get(join.key),
                            join.index,
                            sorted((record.get(join.index) for record in row.joined), key=repr),
                            count,
                        )
                    )
            line = []
            for column in table.columns:
                try:
                    line.append(column.value(row))
                except ParseError as error:
                    raise ParseError("%s: [%s] %s: %s" % (row.source, member, column.head, error))
            values.append(line)
        columns, constant = [], []
        for index, column in enumerate(table.columns):
            distinct = {cell(line[index]) for line in values}
            if column.identity and len(distinct) == 1:
                constant.append("%s=%s" % (column.head, distinct.pop()))
            else:
                columns.append(index)
        heads = [table.columns[i].head for i in columns]
        cells = [[cell(line[i]) for i in columns] for line in values]
        right = [
            all(
                isinstance(line[i], (int, float)) and not isinstance(line[i], bool)
                for line in values
                if line[i] is not None
            )
            for i in columns
        ]
        if any(row.raw or row.warning for row in table_rows):
            heads.append("clock")
            right.append(False)
            for line, row in zip(cells, table_rows):
                line.append(
                    "raw: " + row.raw
                    if row.raw
                    else "warning: unvalidated, converted at the measured %.0f Hz: %s" % (row.hz, row.warning)
                    if row.warning
                    else ""
                )
        out.append(
            Tabulated(member + (" " + table.title if table.title else ""), constant, heads, right, table_rows, cells)
        )
    for derive in DERIVED.get(member, ()):
        derived = derive(member, rows)
        if derived is not None:
            out.append(derived)
    return out


def render(member, rows, markdown=False):
    """`member`'s tables (tabulate) as aligned text, or as Markdown whose rows lead with their label."""
    return render_tables(tabulate(member, rows), markdown)


def render_tables(tables, markdown=False):
    """Tabulated `tables` as render() prints them."""
    out = []
    for table in tables:
        size = "%d row%s" % (len(table.rows), "" if len(table.rows) == 1 else "s")
        if markdown:
            summary = size + "." + (" Constant: %s." % ", ".join(table.constant) if table.constant else "")
            lines = [
                "### " + table.title,
                "",
                summary,
                "",
                "| configuration | %s |" % " | ".join(table.heads),
                "|---|" + "".join("---:|" if r else "---|" for r in table.right),
            ]
            lines += ["| %s | %s |" % (row.label, " | ".join(line)) for row, line in zip(table.rows, table.cells)]
        else:
            widths = [max([len(head)] + [len(line[i]) for line in table.cells]) for i, head in enumerate(table.heads)]

            def aligned(line):
                return "  ".join(c.rjust(w) if r else c.ljust(w) for c, w, r in zip(line, widths, table.right)).rstrip()

            caption = "%s -- %s" % (table.title, size)
            if table.constant:
                caption += "  [%s]" % ", ".join(table.constant)
            lines = [caption, aligned(table.heads), "  ".join("-" * w for w in widths)]
            lines += [aligned(line) for line in table.cells]
        out.append("\n".join(lines))
    return "\n\n".join(out)


def validate_clock(clock):
    """Check a measurement-schema-2 clock object; return its measured rate and its failed checks.

    Raises ValueError when the evidence is malformed: a field missing or
    ill-typed, an unknown domain or reference, or recorded bounds that disagree
    with the raw samples. Well-formed evidence returns (measured_hz, warnings):
    x280_mtime's fixed 50 MHz or the midpoint of the AICLK elapsed-rate bounds
    recomputed from the raw samples, and the measurement checks it fails --
    the producer's own errors, telemetry or arbiter outside tolerance, a
    thermal trip, a rate outside the band, excessive uncertainty -- empty when
    it validates.
    """
    if not isinstance(clock, dict):
        raise ValueError("measurement schema 2 requires clock evidence")
    errors = clock.get("errors")
    expected = clock.get("expected_hz")
    hz = clock.get("conversion_hz")
    if not isinstance(errors, list):
        raise ValueError("clock evidence has no errors list")
    warnings = ["clock evidence contains errors"] if errors else []
    if type(expected) is not int or expected <= 0 or type(hz) is not int or hz != expected:
        raise ValueError("clock conversion_hz must equal a positive expected_hz")
    domain = clock.get("domain")
    if domain == "x280_mtime":
        if expected != 50_000_000:
            raise ValueError("x280_mtime clock rate is not 50 MHz")
        return expected, warnings
    if domain != "aiclk" or clock.get("expected_source") != "cmfw_aiclk_limit_max":
        raise ValueError("unknown clock domain or AICLK expectation source")
    # The counter the evidence was sampled on: a Tensix live timer, or an idle
    # ERISC's own wall clock (execution.hpp clock_reference_at).
    if clock.get("reference_counter") not in ("tensix_live_timer", "erisc_wall_clock"):
        raise ValueError("unknown AICLK reference counter")
    # The translated (x, y) of the core sampled: a DRISC record's is a Tensix.
    core = clock.get("reference_core")
    if not isinstance(core, list) or len(core) != 2 or any(type(v) is not int or v < 0 for v in core):
        raise ValueError("AICLK reference core is malformed")
    if type(clock.get("sample_gap_ns")) is not int or clock["sample_gap_ns"] <= 0:
        raise ValueError("AICLK sample gap is invalid")
    tolerance_ppm = clock.get("rate_tolerance_ppm")
    if type(tolerance_ppm) is not int or tolerance_ppm <= 0:
        raise ValueError("AICLK rate tolerance is invalid")
    expected_mhz = expected // 1_000_000
    telemetry = []
    endpoints = []
    for name in ("start", "end"):
        endpoint = clock.get(name)
        if not isinstance(endpoint, dict) or len(endpoint.get("samples", [])) != 2:
            raise ValueError(f"AICLK {name} endpoint is malformed")
        samples = endpoint["samples"]
        if any(len(sample) != 3 or any(type(value) is not int for value in sample) for sample in samples):
            raise ValueError(f"AICLK {name} samples are malformed")
        first, second = samples
        if first[0] > first[2] or second[0] > second[2] or second[0] <= first[2] or second[1] <= first[1]:
            raise ValueError(f"AICLK {name} samples are nonmonotonic")
        ticks = second[1] - first[1]
        lower = ticks * 1e9 / (second[2] - first[0])
        upper = ticks * 1e9 / (second[0] - first[2])
        if not math.isclose(endpoint.get("rate_lower_hz", 0), lower, rel_tol=1e-12) or not math.isclose(
            endpoint.get("rate_upper_hz", 0), upper, rel_tol=1e-12
        ):
            raise ValueError(f"AICLK {name} rate bounds disagree with raw samples")
        endpoints.append((samples, lower, upper))
        sample = endpoint.get("telemetry")
        if not isinstance(sample, dict) or any(
            type(sample.get(key)) is not int
            for key in ("aiclk_limit_max_mhz", "aiclk_mhz", "aiclk_arb_max_raw", "thermal_trip_count")
        ):
            raise ValueError(f"AICLK {name} telemetry is missing or malformed")
        if sample["aiclk_limit_max_mhz"] != expected_mhz:
            warnings.append(f"AICLK {name} limit disagrees with expected_hz")
        if abs(sample["aiclk_mhz"] - expected_mhz) * 1_000_000 > expected_mhz * tolerance_ppm:
            warnings.append(f"AICLK {name} telemetry disagrees with expected_hz")
        if abs((sample["aiclk_arb_max_raw"] & 0xFFFF) - expected_mhz) * 1_000_000 > expected_mhz * tolerance_ppm:
            warnings.append(f"AICLK {name} arbiter is outside the accepted rate tolerance")
        telemetry.append(sample)
    if telemetry[0]["thermal_trip_count"] != telemetry[1]["thermal_trip_count"]:
        warnings.append("thermal trip count changed")
    elapsed_first, elapsed_second = endpoints[0][0][1], endpoints[1][0][0]
    elapsed_ticks = elapsed_second[1] - elapsed_first[1]
    elapsed_lower = elapsed_ticks * 1e9 / (elapsed_second[2] - elapsed_first[0])
    elapsed_upper = elapsed_ticks * 1e9 / (elapsed_second[0] - elapsed_first[2])
    if not math.isclose(clock.get("elapsed_rate_lower_hz", 0), elapsed_lower, rel_tol=1e-12) or not math.isclose(
        clock.get("elapsed_rate_upper_hz", 0), elapsed_upper, rel_tol=1e-12
    ):
        raise ValueError("AICLK elapsed rate bounds disagree with raw samples")
    accepted_delta = expected * tolerance_ppm / 1_000_000
    accepted_lower, accepted_upper = expected - accepted_delta, expected + accepted_delta
    bounds = [(endpoints[0][1], endpoints[0][2]), (endpoints[1][1], endpoints[1][2]), (elapsed_lower, elapsed_upper)]
    endpoint_width_ppm = clock.get("maximum_endpoint_bound_width_ppm")
    elapsed_width_ppm = clock.get("maximum_elapsed_bound_width_ppm")
    if type(endpoint_width_ppm) is not int or type(elapsed_width_ppm) is not int:
        raise ValueError("AICLK uncertainty limits are missing")
    if any(lower < accepted_lower or upper > accepted_upper for lower, upper in bounds):
        warnings.append("AICLK measured rate is outside the accepted band")
    if (
        any(upper - lower > expected * endpoint_width_ppm / 1_000_000 for lower, upper in bounds[:2])
        or elapsed_upper - elapsed_lower > expected * elapsed_width_ppm / 1_000_000
    ):
        warnings.append("AICLK measurement uncertainty is excessive")
    endpoint_gap = max(endpoints[0][1], endpoints[1][1]) - min(endpoints[0][2], endpoints[1][2])
    if endpoint_gap > accepted_delta:
        warnings.append("AICLK endpoint rates differ beyond the accepted tolerance")
    return (elapsed_lower + elapsed_upper) / 2, warnings


def validate_packet_aggregate(row):
    """Require a packet aggregate record to be a usable X280 mtime/rdcycle bracket of the measured phase."""
    if row.get("measurement_schema") != 2 or row.get("aggregate_error") is not None:
        raise ValueError(
            f"packet aggregate interval is unusable: measurement_schema "
            f"{row.get('measurement_schema')!r}, aggregate_error {row.get('aggregate_error')!r}"
        )
    if not isinstance(row.get("clock"), dict) or row["clock"].get("domain") != "x280_mtime":
        raise ValueError("packet aggregate clock is not x280_mtime")
    if row.get("wall_counter") != "x280_mtime" or row.get("cpu_counter") != "x280_rdcycle":
        raise ValueError("packet aggregate counters are not X280 mtime and rdcycle")
    if not all(
        0 < endpoint_difference(row[f"aggregate_start_{counter}"], row[f"aggregate_end_{counter}"]) < 2**63
        for counter in ("mtime", "rdcycle")
    ):
        raise ValueError(
            f"packet aggregate endpoints are not a forward interval: mtime "
            f"{row['aggregate_start_mtime']}..{row['aggregate_end_mtime']}, rdcycle "
            f"{row['aggregate_start_rdcycle']}..{row['aggregate_end_rdcycle']}"
        )


# How each configured member chooses its warm-up count: the variable whose
# absence from a configuration makes the member calibrate, None for a count
# always used as given, or "none" for a member with no warm-up. Every record
# with a clock states the source (record() in harness/output).
WARMUP_SOURCE = {
    "p2p_latency/x280_same_tile": None,
    "p2p_latency/x280_peer_tile": None,
    "p2p_latency/tensix_peer_tile": "P2P_LATENCY_TENSIX_PEER_TILE_W",
    "p2p_latency/erisc_peer_tile": "P2P_LATENCY_ERISC_PEER_TILE_W",
    "p2p_latency/drisc_peer_tile": "P2P_LATENCY_DRISC_PEER_TILE_W",
    "transport_bandwidth/x280_to_tensix": None,
    "transport_bandwidth/tensix_to_x280": "TRANSPORT_BW_W",
    "transport_bandwidth/erisc_to_x280": "TRANSPORT_BW_W",
    "pipeline_stage/route_compute": None,
    "pipeline_stage/route_forward": None,
    "packet_injection/tensix_to_x280": None,
    "packet_injection/erisc_to_x280_header": "PACKET_INJECTION_W",
    "core_sync/x280_director_characterization": "none",
    "eth_credit/eth_credit": "none",
}


# The live timer an AICLK record's evidence samples, by the record's wall
# counter: the counter itself, except a DRISC's, which the host cannot sample,
# so its evidence samples Tensix (0,0) (aiclk_reference() in
# harness/execution). record() in harness/output refuses to write a record
# that breaks it; the validator checks what it reads.
AICLK_REFERENCE = {"risc_debug_wall_clock": "tensix_live_timer"}


def expected_warmup_source(config):
    """The warmup_source every clocked record of `config` must carry."""
    variable = WARMUP_SOURCE[config["member"]]
    if variable == "none":
        return "none"
    if variable is None or variable in config["environment"]:
        return "explicit"
    return "calibrated"


def validate_campaign_records(config, rows):
    """Validate exact configuration populations without deriving measurements.

    `rows` are the records parse_jsonl() read; the join that renders a packet
    run's tables checks its worker indices.
    """
    family, member = config["member"].split("/", 1)
    environment = config["environment"]
    # eth_credit's count follows its gtest's run points, which the manifest does not name.
    records = {
        "p2p_latency": 4,
        "transport_bandwidth": 1,
        "pipeline_stage": 1,
        "core_sync": 3,
        "packet_injection": 0,
        "eth_credit": None,
    }[family]
    if family == "packet_injection" and member == "erisc_to_x280_header":
        # The writer count and B are the gtest axis, named in the filter; "all"
        # is whatever the device offered, which the run record states.
        point = re.fullmatch(
            r"--gtest_filter=Axes/PacketInjectionEriscToX280Header\.Run/" r"writers(\d+|_all)_b(\d+)",
            config["argv"][-1],
        )
        if point is None:
            raise ValueError(f"{config['id']}: unrecognized ERISC header-ingress filter")
        runs = [r for r in rows if r.get("record") == "run"]
        if len(runs) != 1:
            raise ValueError(f"{config['id']}: expected one run record, got {len(runs)}")
        worker_count = runs[0]["worker_count"] if point[1] == "_all" else int(point[1])
        if not 1 <= worker_count <= 14 or runs[0]["worker_count"] != worker_count or runs[0]["B"] != int(point[2]):
            raise ValueError(f"{config['id']}: run record does not match the requested point")
        records = 2 + worker_count
    elif family == "packet_injection":
        worker_count = (
            int(environment["PACKET_INJECTION_T"]) * {"brisc": 1, "paired": 2}[environment["PACKET_INJECTION_FAMILY"]]
        )
        records = 2 + worker_count
    if records is not None and len(rows) != records:
        raise ValueError(f"{config['id']}: expected {records} rows, got {len(rows)}")
    if family == "p2p_latency":
        participants = {r["participant"] for r in rows}
        keys = {(r["participant"], r["phase"]) for r in rows}
        if len(participants) != 2 or keys != {(p, n) for p in participants for n in (0, 1)}:
            raise ValueError("P2P participant/phase set mismatch")
        peer_tx_mode = environment.get("P2P_LATENCY_METAL_PEER_TX_MODE") or "wait_for_completion"
        for row in rows:
            if row["samples"] != int(environment[f"P2P_LATENCY_{member.upper()}_N"]):
                raise ValueError("P2P sample count mismatch")
            metal_peer = f"{family} {member}" in METAL_PEER_MEMBERS and row["participant"] != "X280"
            if tx_mode(row) != (peer_tx_mode if metal_peer else "wait_for_completion"):
                raise ValueError("P2P transmit mode mismatch")
            for name in ("wall", "cpu_cycles"):
                if len(row[name]["raw_ticks"]) != row["samples"] or row[name]["count"] != row["samples"]:
                    raise ValueError("P2P raw population mismatch")
    elif family == "packet_injection":
        runs = [r for r in rows if r.get("record") == "run"]
        aggregates = [r for r in rows if r.get("record") == "aggregate"]
        workers = [r for r in rows if r.get("record") == "worker"]
        if (
            len(runs) != 1
            or len(aggregates) != records - 1 - worker_count
            or not indices_cover(workers, "worker_index", worker_count)
            or runs[0]["worker_count"] != worker_count
        ):
            raise ValueError("packet run/aggregate/worker set mismatch")
        for row in rows:
            if not row["successful"] or row["run_value"] != runs[0]["run_value"]:
                raise ValueError("failed or unrelated packet record")
            if row["N"] != int(config["environment"]["PACKET_INJECTION_N"]):
                raise ValueError("packet configuration mismatch")
        # Configurations sharing a filter differ only in these knobs.
        recorded = dict(runs[0], INFLIGHT=pi_inflight(runs[0]))
        for knob in ("S", "C", "K", "W", "INFLIGHT"):
            if "PACKET_INJECTION_" + knob in environment and recorded[knob] != int(
                environment["PACKET_INJECTION_" + knob]
            ):
                raise ValueError(
                    f"{config['id']}: run record {knob}={recorded[knob]} differs from "
                    f"the configured {environment['PACKET_INJECTION_' + knob]}"
                )
        if aggregates:
            validate_packet_aggregate(aggregates[0])
    elif family == "eth_credit":
        windows = [r for r in rows if r["member"] in ("eth_credit", "eth_credit_hop", "eth_credit_leg")]
        trips = [r for r in rows if r["member"] == "eth_credit_round_trip"]
        if not windows and not trips:
            raise ValueError(f"{config['id']}: no eth_credit throughput or round-trip record")
        for row in trips:
            if row["round_trips"] < 1:
                raise ValueError(f"{config['id']}: case {row['case']} p {row['p']}: no round trip timed")
        # The window's first stamp follows a batch of 1..BATCH packets; zero
        # means the field was never written, which would overstate the rate.
        for row in windows:
            if row["receive_window"]["interval_error"] is not None or not 1 <= row["first_batch_packets"] <= min(
                row["BATCH"], row["packets"] - 1
            ):
                raise ValueError(
                    f"{config['id']}: case {row['case']} p {row['p']}: receive window "
                    f"{row['receive_window']['interval_error']}, first batch "
                    f"{row['first_batch_packets']} of {row['packets']} packets"
                )
        for row in rows:
            if row["member"] == "eth_credit_attribution" and (row["origins"] or row["stalled"] or row["missing"]):
                raise ValueError(f"{config['id']}: case {row['case']} attributes a failure")
            if row["member"] == "eth_credit_link" and (row["metal_failure"], row["undrained"]) != ("none", "none"):
                raise ValueError(
                    f"{config['id']}: case {row['case']} chip {row['chip']} ended "
                    f"{row['metal_failure']}, {row['undrained']}"
                )
    elif family == "core_sync":
        if {r["release"] for r in rows} != {"null", "unicast", "multicast"}:
            raise ValueError("core-sync release set mismatch")
        for row in rows:
            stamps = row["participants"]
            if len(stamps) != 16 or {(p["logical_x"], p["logical_y"]) for p in stamps} != {
                (x, y) for x in range(4) for y in range(4)
            }:
                raise ValueError("core-sync participant set mismatch")
            if any(p["outcome"] != 0 for p in stamps):
                raise ValueError("core-sync worker did not succeed")
            if row["origins_comparable"] or row.get("cpu_cycles") is not None:
                raise ValueError("cross-core duration lacks a common counter origin")
    else:
        for row in rows:
            if not 0 < row["cpu_cycles"] < 2**63 or not 0 < row["raw_wall_ticks"] < 2**63:
                raise ValueError("invalid batch counters")
            if family == "pipeline_stage" and row["items"] != int(environment["PIPELINE_STAGE_N"]):
                raise ValueError("pipeline item count mismatch")
            if family == "pipeline_stage" and any(
                arg.startswith("--gtest_filter=") and arg.endswith("_instrumented") for arg in config["argv"]
            ):
                population = row.get("cpu_samples")
                if population is None or len(population["raw_ticks"]) != int(environment["PIPELINE_STAGE_N"]):
                    raise ValueError("pipeline sample population missing")

    def check(value):
        if isinstance(value, dict):
            if "cpu_time_ns" in value:
                raise ValueError("CPU-derived nanoseconds are forbidden")
            if "raw_ticks" in value and not raw_population(value):
                raise ValueError("invalid raw sample population")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite measurement")

    warmup = expected_warmup_source(config)
    for row in rows:
        if row.get("measurement_schema") == 2:
            clock = row.get("clock")
            validate_clock(clock)
            wall = row.get("wall_counter")
            reference = AICLK_REFERENCE.get(wall, wall)
            if clock["domain"] == "aiclk" and wall is not None and clock["reference_counter"] != reference:
                raise ValueError(
                    f"{config['id']}: {wall} ticks carry AICLK evidence sampled on "
                    f"{clock['reference_counter']}, not {reference}"
                )
        if "measurement_schema" in row and row.get("warmup_source") != warmup:
            raise ValueError(
                f"{config['id']}: a record's warm-up source is {row.get('warmup_source')!r}, " f"not {warmup!r}"
            )
        check(row)


def conversion(record):
    """The rate a record's ticks convert at, why they stay raw, and why the rate is unvalidated.

    (conversion_hz, None, None) when the clock validates; (measured Hz, None,
    the failed checks) when well-formed evidence fails them; (None, the reason,
    None) for no schema-2 clock or malformed evidence.
    """
    if record.get("measurement_schema") != 2:
        return None, "measurement schema %r offers no conversion" % record.get("measurement_schema"), None
    try:
        measured, warnings = validate_clock(record.get("clock"))
    except ValueError as error:
        return None, str(error), None
    if warnings:
        return measured, None, "; ".join(warnings)
    return record["clock"]["conversion_hz"], None, None


def read_attempt(config, paths):
    """The Rows of one attempt's raw JSONL files at `paths`, if the report can tabulate them.

    Every file goes through parse_jsonl(), labelled with the configuration
    and file; the records must pass validate_campaign_records() and every
    member's tables must render. The runner calls it to decide an attempt
    and the report to read one, so both refuse the same records. Raises
    ParseError, ValueError or KeyError naming the first fault.
    """
    rows = []
    for path in paths:
        rows += parse_jsonl(str(path), path.read_text(), f"[{config['id']}]({path})")
    validate_campaign_records(config, [row.record for row in rows])
    for member in {row.member for row in rows}:
        tabulate(member, rows)
    return rows


def campaign_report(campaign, destination):
    """Regenerate one immutable report from retained attempts, checking raw hashes.

    A passed attempt whose records read_attempt() refuses counts as failed in
    the coverage, with the reason; a raw file whose hash differs is an error.
    """
    manifest = json.loads((campaign / "manifest.json").read_text())
    configs = {c["id"]: c for c in manifest["configurations"]}
    coverage = {key: {"status": "not_run", "attempts": []} for key in configs}
    master = []
    for directory in sorted((campaign / "attempts").glob("*/")):
        path = directory / "attempt.json"
        attempt = json.loads(path.read_text())
        key = attempt["configuration"]
        entry = coverage[key]
        status = attempt["status"]
        entry["attempts"].append({"path": str(path), "status": status, "error": attempt.get("error")})
        if status != "passed":
            if entry["status"] == "not_run":
                entry["status"] = status
            continue
        for artifact, digest in attempt["artifacts_sha256"].items():
            if hashlib.sha256((path.parent / artifact).read_bytes()).hexdigest() != digest:
                raise ValueError(f"artifact hash mismatch: {path.parent / artifact}")
        raws = sorted((path.parent / "raw").glob("*.jsonl"))
        digests = {str(raw): attempt["artifacts_sha256"].get("raw/" + raw.name) for raw in raws}
        unrecorded = [raw for raw, digest in digests.items() if digest is None]
        if unrecorded:
            raise ValueError(f"unrecorded raw artifact: {unrecorded[0]}")
        try:
            rows = read_attempt(configs[key], raws)
        except (ParseError, ValueError, KeyError) as error:
            entry["attempts"][-1].update(status="failed", error=f"not reportable: {error}")
            if entry["status"] == "not_run":
                entry["status"] = "failed"
            continue
        entry["status"] = "passed"
        master += [(key, path.parent.name, row, digests[row.path]) for row in rows]
    complete = all(entry["status"] == "passed" for entry in coverage.values())
    table_rows = [row for _, _, row, _ in master]
    results = [render(member, [row for row in table_rows if row.member == member], markdown=True) for member in MEMBERS]
    results.append(render_tables(cross_member_tables(table_rows), markdown=True))
    wall_rows = [row for row in table_rows if "wall_counter" in row.record]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (destination / "master_results.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "configuration": key,
                    "attempt": attempt_name,
                    "source": {"path": row.path, "line": row.line, "sha256": digest},
                    "row": row.record,
                },
                allow_nan=False,
            )
            + "\n"
            for key, attempt_name, row, digest in master
        )
    )
    (destination / "campaign.json").write_text(
        json.dumps(
            {
                "required_configurations": len(configs),
                "passed_configurations": sum(e["status"] == "passed" for e in coverage.values()),
                "records": len(master),
                "evidence": str(campaign),
                "wall_records_with_conversion": sum(row.hz is not None and not row.warning for row in wall_rows),
                "wall_records_with_clock_warning": sum(bool(row.warning) for row in wall_rows),
                "raw_only_wall_records": sum(row.hz is None for row in wall_rows),
                "manifest_sha256": hashlib.sha256((campaign / "manifest.json").read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    runs = sorted(
        {sum(a["status"] == "passed" for a in e["attempts"]) for e in coverage.values() if e["status"] == "passed"}
    )
    if not runs or runs[-1] <= 1:
        repeats = (
            "One planned run per configuration; diagnostic attempts are retained. "
            "Between-run variation is unavailable."
        )
    else:
        repeats = (
            f"Passed runs per covered configuration: {', '.join(map(str, runs))}; "
            "every passed run is tabulated below."
        )
    lines = [
        "# X280 measurement campaign",
        "",
        f"Coverage: {'complete' if complete else 'incomplete'}. {repeats}",
        "",
        "| Configuration | Status | Attempts |",
        "|---|---|---|",
    ]
    for key, entry in coverage.items():
        links = ", ".join(f"[{a['status']}]({a['path']})" for a in entry["attempts"])
        lines.append(f"| {key} | {entry['status']} | {links or '0'} |")
    lines += [
        "",
        "Raw records, with their clock evidence: [master_results.jsonl](master_results.jsonl).",
        "",
        "## Results",
        "",
        "Derived from each record's raw fields: percentiles by the floor-index rule, nanoseconds "
        "and rates at the record's validated `conversion_hz`. A record whose well-formed clock "
        "evidence fails its checks is converted at the measured rate instead, and its clock "
        "column gives that rate and the failed checks as a warning. Core-sync records offer no "
        "conversion, so their rows stay raw with the reason in the clock column; their ranges "
        "are cross-core diagnostics.",
    ]
    lines += ["\n" + table for table in results if table]
    (destination / "MASTER_RESULTS.md").write_text("\n".join(lines) + "\n")
    return complete


def main():
    ap = argparse.ArgumentParser(
        description="Tabulate current-format x280 characterization records "
        "as result tables, or write a campaign's report."
    )
    ap.add_argument("sources", nargs="*", metavar="FILE", help="JSONL record files to read; stdin when none are named")
    ap.add_argument("--campaign", type=Path)
    ap.add_argument("--report-dir", type=Path)
    args = ap.parse_args()
    if args.campaign:
        if not args.report_dir:
            ap.error("--campaign requires a new --report-dir")
        try:
            return 0 if campaign_report(args.campaign, args.report_dir) else 1
        except (ParseError, ValueError, KeyError, OSError) as error:
            print("FAIL: %s" % error, file=sys.stderr)
            return 1

    rows, failures = [], []

    sources = []
    if args.sources:
        for path in args.sources:
            try:
                with open(path, encoding="utf-8") as handle:
                    sources.append((path, handle.read()))
            except (OSError, UnicodeError) as error:
                failures.append("%s: %s" % (path, error))
    else:
        try:
            sources.append(("<stdin>", sys.stdin.buffer.read().decode("utf-8")))
        except UnicodeError as error:
            failures.append("<stdin>: %s" % error)
    for path, text in sources:
        try:
            rows += parse_jsonl(path, text)
        except ParseError as error:
            failures.append(str(error))

    for failure in failures:
        print("FAIL: %s" % failure, file=sys.stderr)
    if failures:
        return 1

    if not rows:
        where = ", ".join(p for p, _ in sources)
        print(
            "FAIL: no suite rows in %s. Either the run measured nothing -- "
            "a member exits 0 having skipped every point when its firmware "
            "image is unset -- or these are not "
            "characterization records." % where,
            file=sys.stderr,
        )
        return 1

    tables = []
    try:
        for member in MEMBERS:
            tables.append(render(member, [row for row in rows if row.member == member]))
        tables.append(render_tables(cross_member_tables(rows)))
    except ParseError as error:
        print("FAIL: %s" % error, file=sys.stderr)
        return 1
    print("\n\n".join(table for table in tables if table))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # Suppress the expected traceback when a pipe reader exits early.
        sys.stderr.close()
        sys.exit(0)
