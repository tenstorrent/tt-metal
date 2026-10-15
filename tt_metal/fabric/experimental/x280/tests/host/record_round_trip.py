#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the records test_record_round_trip.cpp wrote with the live tabulator.

Usage: record_round_trip.py RECORD_DIR MANIFEST FILES

Each RECORD_DIR/<id>.jsonl holds the records of manifest configuration <id>;
any other stem fails. Every record must name its dispatch mode. The configured
files are laid out as a campaign of passed attempts, and campaign_report must
accept it whole -- each configuration's validator,
the raw-artifact hashes -- and render each record in its member's results
table. Table mode's output and the report must show the configured files'
tables, with a value in every cell and the EXPECTED cells. A copy of a record
whose well-formed clock fails a check must be covered and converted at the
measured rate with a warning in both outputs, and one with malformed clock
evidence must fail its attempt. Exits nonzero naming every
failure, or when the directory does not hold FILES record files.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from tabulate_suites import (
    ParseError,
    all_tables,
    campaign_report,
    parse_jsonl,
    read_attempt,  # noqa: E402
    render_all,
)


# Cells both outputs must show, by table title and row, for the inputs
# test_record_round_trip.cpp writes. P2P samples are 20 + i on one counter and
# 300 + i on the other (i < 64), so the floor-index p50, p90 and p99 are
# indices 31, 56 and 62; the X280 counter is 50 MHz mtime, the Metal one a
# 1 GHz AICLK. Pipeline samples are 90 + i % 7 over batches of 40000 wall
# ticks and 70000 cycles. Packet workers take 90000 ticks and 120000 cycles
# over N requests of 96 useful and 128 slot bytes, with N counter writes.
# Every table's heads, constant or not: a column dropped or renamed fails here.
def quantiles(name, unit, converted=False):
    return {f"{name} {q} {u}" for q in ("p50", "p90", "p99") for u in ((unit, "ns") if converted else (unit,))}


P2P_HEADS = (
    {
        "who",
        "phase",
        "dir",
        "role",
        "kind",
        "first dir",
        "N",
        "W",
        "payload",
        "same chip",
        "window",
        "tlb",
        "tx",
        "sentinel",
        "wall counter",
        "cpu counter",
    }
    | quantiles("wall", "ticks", True)
    | quantiles("cpu", "cycles")
)
BATCH_HEADS = {"wall counter", "cpu counter", "wall ticks", "cpu cycles", "ns", "ns/item", "cpu/item", "cpu/wall"}
PS_HEADS = BATCH_HEADS | {"packet B", "route cap", "instr", "W", "N", "witness", "subject", "nonce"}
PS_COMPUTE_HEADS = PS_HEADS | {"fixture", "headers", "input base", "input B"}
PS_FORWARD_HEADS = PS_HEADS | {"source", "source base", "source B", "hops", "boundary"}
PS_X280_POLICY_HEADS = {"wayenable", "pinned way", "mask bits"}
PS_X280_HEADS = PS_X280_POLICY_HEADS | {"cpu sample mean cycles"} | quantiles("cpu sample", "cycles")
PI_RUN_HEADS = {"family", "T", "P", "S", "B", "C", "K", "inflight", "W", "N", "run"}
PI_AGGREGATE_HEADS = PI_RUN_HEADS | {"wall ticks", "cpu cycles", "ns", "headers", "ns/header"}
PI_THROUGHPUT_HEADS = PI_RUN_HEADS | {
    "worker",
    "x",
    "y",
    "wall ticks",
    "cpu cycles",
    "ns",
    "requests",
    "ns/request",
    "useful GB/s",
    "slot GB/s",
    "stall episodes",
    "counter writes",
}
TURN_SPLIT_HEADS = (
    {"first dir", "completion N", "departure N", "wall counter"}
    | {f"{term} {stat}" for term in ("completion", "departure", "ack") for stat in ("mean ticks", "p50 ticks")}
    | {f"{term} {stat}" for term in ("completion", "departure", "ack") for stat in ("mean ns", "p50 ns")}
)


def placement_heads(keys, cores, legs):
    return set(keys) | set(cores) | {f"{leg} {h}" for leg in legs for h in ("noc", "hops", "ack hops")} | {"records"}


METAL_PLACEMENT = ("x280->peer", "peer->x280")
PI_PLACEMENT_HEADS = placement_heads(
    ("worker_index", "chip_id"), ("x280_noc", "noc0"), ("worker->x280", "x280->worker")
)
EC_PLACEMENT_HEADS = placement_heads(
    ("tx_chip", "rx_chip", "p"),
    ("tx_x280_noc0", "tx_erisc_noc0", "rx_erisc_noc0", "rx_x280_noc0"),
    ("x280->e0", "e0->x280", "e1->x280", "x280->e1"),
)
HEADS = {
    "p2p_latency x280_same_tile placement": placement_heads(("participant", "chip_id"), ("x280_noc",), ()),
    "p2p_latency x280_peer_tile placement": placement_heads(
        ("participant", "chip_id"), ("x280_noc", "peer_x280_noc"), ("a->b", "b->a")
    ),
    "p2p_latency tensix_peer_tile placement": placement_heads(
        ("participant", "chip_id"), ("x280_noc", "tensix_noc0"), METAL_PLACEMENT
    ),
    "p2p_latency erisc_peer_tile placement": placement_heads(
        ("participant", "chip_id"), ("x280_noc", "erisc_noc0"), METAL_PLACEMENT
    ),
    "p2p_latency drisc_peer_tile placement": placement_heads(
        ("participant", "chip_id"), ("x280_noc", "drisc_noc0"), METAL_PLACEMENT
    ),
    "packet_injection tensix_to_x280 placement": PI_PLACEMENT_HEADS,
    "packet_injection erisc_to_x280_header placement": PI_PLACEMENT_HEADS,
    "eth_credit eth_credit placement": EC_PLACEMENT_HEADS,
    "eth_credit eth_credit_leg placement": EC_PLACEMENT_HEADS,
    "eth_credit eth_credit_round_trip placement": EC_PLACEMENT_HEADS,
    "eth_credit eth_credit_hop placement": placement_heads(
        ("tx_chip", "rx_chip", "p"), ("tx_erisc_noc0", "rx_erisc_noc0"), ()
    ),
    "p2p_latency x280_same_tile": P2P_HEADS,
    "p2p_latency x280_peer_tile": P2P_HEADS | {"hw", "peer", "hops a->b", "hops b->a", "lap hops"},
    "p2p_latency tensix_peer_tile": P2P_HEADS,
    "p2p_latency erisc_peer_tile": P2P_HEADS,
    "p2p_latency drisc_peer_tile": P2P_HEADS,
    "p2p_latency tensix_peer_tile TURN split": TURN_SPLIT_HEADS,
    "p2p_latency erisc_peer_tile TURN split": TURN_SPLIT_HEADS,
    "p2p_latency drisc_peer_tile TURN split": TURN_SPLIT_HEADS,
    "transport_bandwidth x280_to_tensix": BATCH_HEADS
    | {
        "size B",
        "boundary",
        "W",
        "N",
        "items",
        "logical B",
        "store B",
        "harts",
        "GB/s",
        "span ticks",
        "run B",
        "run GB/s",
    },
    "transport_bandwidth tensix_to_x280": BATCH_HEADS | {"size B", "boundary", "W", "N", "items", "logical B", "GB/s"},
    "transport_bandwidth erisc_to_x280": BATCH_HEADS
    | {"size B", "boundary", "W", "N", "items", "logical B", "writers", "GB/s"}
    | {"slowest ticks", "run GB/s", "writer GB/s min", "writer GB/s max", "gate ns"},
    "pipeline_stage route_compute X280 per_sample": PS_COMPUTE_HEADS | PS_X280_HEADS,
    "pipeline_stage route_forward X280 decide off": PS_FORWARD_HEADS | PS_X280_POLICY_HEADS,
    "packet_injection tensix_to_x280 throughput aggregate": PI_AGGREGATE_HEADS | {"header B", "header GB/s"},
    "packet_injection tensix_to_x280 worker throughput": PI_THROUGHPUT_HEADS | {"risc"},
    "packet_injection tensix_to_x280 latency, pooled over workers": PI_RUN_HEADS
    | {"samples"}
    | quantiles("latency", "ticks", True),
    "packet_injection tensix_to_x280 backpressure, pooled over workers": PI_RUN_HEADS
    | {"stalls"}
    | quantiles("stall", "ticks", True)
    | quantiles("stall rtt", "ticks", True),
    "packet_injection tensix_to_x280 latency and backpressure aggregate": PI_AGGREGATE_HEADS,
    "packet_injection tensix_to_x280 worker latency": PI_RUN_HEADS
    | {"risc", "worker", "x", "y"}
    | quantiles("latency", "ticks", True),
    "packet_injection tensix_to_x280 worker backpressure": PI_RUN_HEADS
    | {"risc", "worker", "x", "y"}
    | quantiles("stall", "ticks", True)
    | quantiles("stall cpu", "cycles")
    | quantiles("stall rtt", "ticks", True)
    | quantiles("stall rtt cpu", "cycles"),
    "packet_injection erisc_to_x280_header throughput aggregate": PI_AGGREGATE_HEADS
    | {"header B", "header GB/s", "published B", "published GB/s"},
    "packet_injection erisc_to_x280_header worker throughput": PI_THROUGHPUT_HEADS,
    "core_sync x280_director_characterization": {
        "release",
        "nonce",
        "chip",
        "comparable origins",
        "wall counter",
        "workers",
        "raw range ticks",
        "clock",
    },
    "eth_credit eth_credit_hop": {
        "case",
        "P",
        "N_LINK",
        "slot B",
        "payload B",
        "BATCH",
        "tx",
        "rx",
        "packets",
        "k1",
        "wall ticks",
        "ns",
        "window packets",
        "ns/packet",
        "Mpkt/s",
        "payload GB/s",
        "slot GB/s",
        "400G Mpkt/s",
    },
    "eth_credit eth_credit": {
        "case",
        "P",
        "N_LINK",
        "N_DELIVER",
        "slot B",
        "payload B",
        "BATCH",
        "both",
        "p",
        "tx",
        "rx",
        "l2cpu",
        "pll MHz",
        "packets",
        "k1",
        "wall ticks",
        "ns",
        "window packets",
        "ns/packet",
        "Mpkt/s",
        "useful GB/s",
        "slot GB/s",
    },
    "eth_credit eth_credit_round_trip": {
        "case",
        "loop",
        "P",
        "N_LINK",
        "N_DELIVER",
        "slot B",
        "BATCH",
        "p",
        "tx",
        "rx",
        "pll MHz",
        "round trips",
        "ticks",
        "mean ns",
        "NoC hops",
        "NoC ns",
        "estimate 30 ns",
        "estimate 80 ns",
    },
    "eth_credit ceilings": {
        "P",
        "N_LINK",
        "N_DELIVER",
        "slot B",
        "BATCH",
        "path Mpkt/s",
        "NoC hop Mpkt/s",
        "Ethernet hop Mpkt/s",
        "receive side Mpkt/s",
        "W1 Mpkt/s",
        "W2 Mpkt/s",
        "R1 Mpkt/s",
        "slowest leg",
        "credit loop Mpkt/s",
        "bound",
        "path/bound",
    },
    "eth_credit eth_credit_leg": {
        "legs",
        "case",
        "P",
        "N_LINK",
        "N_DELIVER",
        "slot B",
        "payload B",
        "BATCH",
        "both",
        "p",
        "tx",
        "rx",
        "l2cpu",
        "pll MHz",
        "packets",
        "k1",
        "wall ticks",
        "ns",
        "window packets",
        "ns/packet",
        "Mpkt/s",
        "useful GB/s",
        "slot GB/s",
    },
}

X280_PHASE = {
    "wall p50 ticks": "51",
    "wall p90 ticks": "76",
    "wall p99 ticks": "82",
    "wall p50 ns": "1020.000",
    "wall p99 ns": "1640.000",
    "cpu p50 cycles": "331",
    "cpu p90 cycles": "356",
    "cpu p99 cycles": "362",
}
METAL_PHASE = {"wall p50 ticks": "331", "wall p99 ticks": "362", "wall p50 ns": "331.000", "cpu p50 cycles": "51"}
TURN_SPLIT = {
    "first dir": "A->B",
    "completion N": "64",
    "departure N": "64",
    "completion p50 ticks": "331",
    "departure p50 ticks": "231",
    "ack mean ticks": "100.000",
    "ack p50 ticks": "100",
    "ack mean ns": "100.000",
}
PS_X280 = {
    "wall ticks": "40000",
    "cpu cycles": "70000",
    "ns": "800000.000",
    "ns/item": "195.312",
    "cpu/item": "17.090",
    "cpu/wall": "1.750",
    "cpu sample mean cycles": "92.999",
    "cpu sample p50 cycles": "93",
    "cpu sample p90 cycles": "96",
    "cpu sample p99 cycles": "96",
}
PS_X280_DECIDE = {
    "wall ticks": "40000",
    "cpu cycles": "70000",
    "ns": "800000.000",
    "ns/item": "48.828",
    "cpu/item": "4.272",
    "cpu/wall": "1.750",
    "boundary": "decision",
}
PI_WORKER = {"wall ticks": "90000", "cpu cycles": "120000", "ns": "90000.000"}


def metal_placement(to_peer, to_x280):
    return {
        "x280->peer hops": str(to_peer),
        "x280->peer ack hops": str(to_x280),
        "peer->x280 hops": str(to_x280),
        "peer->x280 ack hops": str(to_peer),
    }


EXPECTED = {
    # D1's card 0 placement (placement-inspection REPORT.md): every Metal loop is 29 hops.
    ("p2p_latency tensix_peer_tile placement", 0): dict(
        metal_placement(21, 8), **{"x280_noc": "(8, 3)", "tensix_noc0": "(1, 2)", "records": "4"}
    ),
    ("p2p_latency erisc_peer_tile placement", 1): dict(metal_placement(12, 17), participant="ERISC"),
    ("p2p_latency drisc_peer_tile placement", 0): metal_placement(18, 11),
    # The recorded hops, 6 and 4 in this test.
    ("p2p_latency x280_peer_tile placement", 0): {"a->b hops": "6", "a->b ack hops": "4", "b->a hops": "4"},
    ("packet_injection tensix_to_x280 placement", 0): {
        "noc0": "(1, 2)",
        "worker->x280 noc": "0",
        "worker->x280 hops": "8",
        "x280->worker hops": "21",
    },
    ("packet_injection erisc_to_x280_header placement", 0): {"worker->x280 hops": "17", "x280->worker hops": "12"},
    ("eth_credit eth_credit placement", 0): {
        "x280->e0 hops": "25",
        "e0->x280 hops": "4",
        "e1->x280 noc": "1",
        "e1->x280 hops": "16",
        "e1->x280 ack hops": "13",
        "x280->e1 hops": "16",
    },
    ("eth_credit eth_credit_hop placement", 0): {"tx_erisc_noc0": "(6, 1)", "rx_erisc_noc0": "(14, 1)"},
    ("p2p_latency x280_same_tile", 0): X280_PHASE,
    ("p2p_latency x280_peer_tile", 0): dict(X280_PHASE, **{"lap hops": "10"}),
    ("p2p_latency tensix_peer_tile", 0): X280_PHASE,
    # Each Metal peer's departure configuration repeats its completion one with the
    # peer's TURN 100 ticks lower; its four rows sort first.
    ("p2p_latency tensix_peer_tile", 5): METAL_PHASE,
    ("p2p_latency erisc_peer_tile", 5): METAL_PHASE,
    ("p2p_latency drisc_peer_tile", 5): METAL_PHASE,
    ("p2p_latency tensix_peer_tile", 1): {"tx": "wait_for_departure", "wall p50 ticks": "231"},
    ("p2p_latency tensix_peer_tile TURN split", 0): TURN_SPLIT,
    ("p2p_latency erisc_peer_tile TURN split", 0): TURN_SPLIT,
    ("p2p_latency drisc_peer_tile TURN split", 0): TURN_SPLIT,
    ("transport_bandwidth x280_to_tensix", 0): {
        "wall ticks": "200000",
        "cpu cycles": "350000",
        "ns": "4000000.000",
        "ns/item": "244.141",
        "cpu/item": "21.362",
        "GB/s": "0.262",
        "boundary": "RETIRED",
        "store B": "4",
        "harts": "1",
        "span ticks": "200000",
        "run GB/s": "0.262",
    },
    # Two harts over mtime 1000..201300: 2 x 16384 x 64 B in 4.006 ms.
    ("transport_bandwidth x280_to_tensix", 1): {
        "wall ticks": "200000",
        "harts": "2",
        "span ticks": "200300",
        "run B": "2097152",
        "run GB/s": "0.524",
    },
    # Two writers over 200000 and 400000 ticks of a 1 GHz AICLK, 1 MiB each.
    ("transport_bandwidth erisc_to_x280", 0): {
        "writers": "2",
        "GB/s": "5.243",
        "slowest ticks": "400000",
        "run GB/s": "5.243",
        "writer GB/s min": "2.621",
        "writer GB/s max": "5.243",
        "gate ns": "1500",
    },
    ("transport_bandwidth tensix_to_x280", 0): {
        "ns": "200000.000",
        "ns/item": "12.207",
        "GB/s": "5.243",
        "boundary": "DRAINED",
    },
    ("pipeline_stage route_compute X280 per_sample", 0): PS_X280,
    ("pipeline_stage route_forward X280 decide off", 0): PS_X280_DECIDE,
    ("packet_injection tensix_to_x280 throughput aggregate", 0): {
        "wall ticks": "50000",
        "cpu cycles": "1750000",
        "ns": "1000000.000",
        "headers": "1000",
        "header B": "96000",
        "header GB/s": "0.096",
    },
    ("packet_injection tensix_to_x280 worker throughput", 0): dict(
        PI_WORKER, **{"ns/request": "90.000", "useful GB/s": "1.067", "slot GB/s": "1.422"}
    ),
    ("packet_injection erisc_to_x280_header throughput aggregate", 0): {
        "headers": "1024",
        "header B": "98304",
        "published B": "102400",
        "published GB/s": "0.102",
    },
    ("packet_injection erisc_to_x280_header worker throughput", 0): dict(PI_WORKER, **{"ns/request": "87.891"}),
    # The paired family's two workers, 1000 requests each.
    ("packet_injection tensix_to_x280 throughput aggregate", 1): {
        "P": "2",
        "headers": "2000",
        "header B": "192000",
        "header GB/s": "0.192",
    },
    ("packet_injection tensix_to_x280 worker throughput", 2): {"risc": "1", "worker": "1", "counter writes": "1000"},
    # One worker's latency samples 300, 330, 360; its backpressure stalls 40-70
    # ticks (4-7 cycles) and stall-triggered RTTs 400-700 ticks (41-71 cycles).
    ("packet_injection tensix_to_x280 latency, pooled over workers", 0): {
        "samples": "3",
        "latency p50 ticks": "330",
        "latency p99 ticks": "330",
        "latency p50 ns": "330.000",
    },
    ("packet_injection tensix_to_x280 backpressure, pooled over workers", 0): {
        "stalls": "4",
        "stall p50 ticks": "50",
        "stall p99 ticks": "60",
        "stall rtt p50 ns": "500.000",
    },
    # Backpressure (P*N = 1000 headers) sorts before latency (3).
    ("packet_injection tensix_to_x280 latency and backpressure aggregate", 1): {
        "headers": "3",
        "ns/header": "333333.333",
    },
    ("packet_injection tensix_to_x280 worker backpressure", 0): {
        "stall cpu p50 cycles": "5",
        "stall rtt cpu p50 cycles": "51",
        "stall rtt p90 ticks": "600",
    },
    ("core_sync x280_director_characterization", 0): {
        "workers": "16",
        "raw range ticks": "15",
        "clock": "raw: measurement schema 1 offers no conversion",
    },
    # 20000 packets, 4 in the first batch, over mtime 1000..3000000 at 50 MHz:
    # 19996 packets in 59.98 ms, 96 useful and 4192 slot bytes each.
    # The same window at E1 on a 1 GHz ERISC wall clock: 4096 payload bytes each.
    ("eth_credit eth_credit_hop", 0): {
        "wall ticks": "2999000",
        "ns/packet": "149.980",
        "Mpkt/s": "6.668",
        "payload GB/s": "27.310",
        "slot GB/s": "27.950",
        "400G Mpkt/s": "12.207",
    },
    # The windows above at one shape, beside each other; the receive side's is
    # the whole path's window again, and W2 and R1 count BATCH (4) packets per
    # store of the same window.
    # 1000 round trips over 250000 mtime ticks. Hop loop: W2 (8,3)->(6,1) on
    # NoC 0 is 15 + 10 links, W5 (14,1)->(8,3) on NoC 1 6 + 10, its ack 11 + 2,
    # R3 (6,1)->(8,3) on NoC 0 2 + 2: 58 links, 4 x 10 + 9 x 58 = 562 cycles at
    # the 1 GHz AICLK; stages four ERISC at 1 GHz and one X280 at 1750 MHz,
    # 4.571 ns per cycle each. The consume loop adds W6 (16 links) and R1
    # (8,3)->(14,1) on NoC 0 (6 + 10): 90 links, 870 cycles; stages five ERISC
    # and two X280, 6.143 ns per cycle.
    ("eth_credit eth_credit_round_trip", 0): {
        "loop": "hop",
        "mean ns": "5000.000",
        "NoC hops": "58",
        "NoC ns": "562.000",
        "estimate 30 ns": "699.143",
        "estimate 80 ns": "927.714",
    },
    ("eth_credit eth_credit_round_trip", 1): {
        "loop": "consume",
        "NoC hops": "90",
        "NoC ns": "870.000",
        "estimate 30 ns": "1054.286",
        "estimate 80 ns": "1361.429",
    },
    # The hop loop above allows N_LINK (8) packets per 5000 ns round trip.
    ("eth_credit ceilings", 0): {
        "path Mpkt/s": "0.333",
        "NoC hop Mpkt/s": "0.333",
        "Ethernet hop Mpkt/s": "6.668",
        "receive side Mpkt/s": "0.333",
        "W1 Mpkt/s": "0.333",
        "W2 Mpkt/s": "1.334",
        "R1 Mpkt/s": "1.334",
        "slowest leg": "NoC hop",
        "credit loop Mpkt/s": "1.600",
        "bound": "NoC hop",
        "path/bound": "1.000",
    },
    # The same window at the sending forwarder, the NoC hop alone: k1 and the
    # rates as above, the count from packets_out.
    ("eth_credit eth_credit_leg", 1): {
        "legs": "W1 W2 R3",
        "k1": "4",
        "window packets": "19996",
        "ns/packet": "2999.600",
    },
    ("eth_credit eth_credit", 0): {
        "k1": "4",
        "wall ticks": "2999000",
        "ns": "59980000.000",
        "window packets": "19996",
        "ns/packet": "2999.600",
        "Mpkt/s": "0.333",
        "useful GB/s": "0.032",
        "slot GB/s": "1.398",
        "payload B": "4096",
    },
}


def check_records(path, rows):
    for number, row in enumerate(rows, 1):
        if "tt_metal_slow_dispatch_mode" not in row:
            raise ValueError(f"record {number} names no dispatch mode")


def check_report(directory, configs, paths):
    """Lay the configured files out as a campaign whose report must be complete."""
    campaign = Path(directory) / "campaign"
    campaign.mkdir()
    (campaign / "manifest.json").write_text(json.dumps({"configurations": [configs[p.stem] for p in paths]}))
    for path in paths:
        attempt = campaign / "attempts" / (path.stem + ".x")
        (attempt / "raw").mkdir(parents=True)
        data = path.read_bytes()
        (attempt / "raw" / "r.jsonl").write_bytes(data)
        (attempt / "attempt.json").write_text(
            json.dumps(
                {
                    "configuration": path.stem,
                    "status": "passed",
                    "artifacts_sha256": {"raw/r.jsonl": hashlib.sha256(data).hexdigest()},
                }
            )
        )
    report = Path(directory) / "report"
    if not campaign_report(campaign, report):
        coverage = json.loads((report / "coverage.json").read_text())
        raise ValueError(
            "campaign report is incomplete: "
            + "; ".join(
                f"{key}: {attempt['error']}"
                for key, entry in coverage.items()
                for attempt in entry["attempts"]
                if attempt["status"] != "passed"
            )
        )


def tables_of(rows):
    """all_tables() of `rows` as {title: (constants, rows)}, each row a {head: cell} dict."""
    return {
        table.title: (
            dict(c.split("=", 1) for c in table.constant),
            [dict(zip(table.heads, cells)) for cells in table.cells],
        )
        for table in all_tables(rows)
    }


def markdown_tables(report):
    """MASTER_RESULTS.md's results tables in tables_of()'s form, without the configuration column."""
    tables = {}
    for section in report.split("\n### ")[1:]:
        title, _, body = section.partition("\n\n")
        summary, _, table = body.partition("\n\n")
        listed = summary.partition(" Constant: ")[2][:-1]
        header, _, *lines = table.strip().split("\n")

        def cells(line):
            return [cell.strip() for cell in line.split("|")[2:-1]]

        tables[title] = (
            dict(c.split("=", 1) for c in listed.split(", ")) if listed else {},
            [dict(zip(cells(header), cells(line))) for line in lines],
        )
    return tables


def check_outputs(directory, paths):
    """Require table mode's output and the report's tables to be the configured files' tables,
    holding a value in every cell and the EXPECTED cells."""
    rows = [row for path in paths for row in parse_jsonl(str(path), path.read_text())]
    tables = tables_of(rows)
    failures = [
        f"{title} row {index}: {head} is blank"
        for title, (_, table_rows) in tables.items()
        for index, row in enumerate(table_rows)
        for head, cell in row.items()
        if not cell
    ]
    failures += [
        f"{title} has heads {sorted(set(constants) | set(table_rows[0]))}, not {sorted(HEADS.get(title, ()))}"
        for title, (constants, table_rows) in tables.items()
        if set(constants) | set(table_rows[0]) != HEADS.get(title)
    ]
    failures += [f"no table {title}" for title in HEADS if title not in tables]
    for (title, index), expected in EXPECTED.items():
        constants, table_rows = tables.get(title, ({}, []))
        shown = dict(constants, **table_rows[index]) if index < len(table_rows) else constants
        failures += [
            f"{title} row {index}: {head} is {shown.get(head)!r}, expected {cell!r}"
            for head, cell in expected.items()
            if shown.get(head) != cell
        ]

    report = markdown_tables((Path(directory) / "report" / "MASTER_RESULTS.md").read_text())
    failures += [
        f"{title}: the report shows {report.get(title)}, not {tables.get(title)}"
        for title in sorted(set(tables) | set(report))
        if tables.get(title) != report.get(title)
    ]
    run = subprocess.run(
        [sys.executable, str(TOOLS / "tabulate_suites.py")] + [str(p) for p in paths], capture_output=True, text=True
    )
    printed = render_all(rows)
    if run.returncode != 0 or run.stdout != printed + "\n":
        failures.append(f"table mode exited {run.returncode} and printed other tables: {run.stderr.strip()}")

    return failures


def check_clock_warning(directory, configs, paths):
    """Require a well-formed clock that fails its checks to be covered and converted at the measured
    rate with a warning in both outputs, and a malformed one to fail its attempt."""
    source = next(p for p in paths if p.stem == "bw.tensix_to_x280.64.drained")
    record = json.loads(source.read_text())
    record["clock"]["errors"] = ["current_frequency_mismatch"]
    root = Path(directory) / "clock_warning"
    root.mkdir()
    flagged = root / source.name
    flagged.write_text(json.dumps(record) + "\n")
    check_report(root, configs, [flagged])
    # The clean evidence's elapsed-rate bounds are 1 GHz -/+ 1000 Hz, so the measured rate is 1 GHz.
    expected = {
        "wall ticks": "200000",
        "ns": "200000.000",
        "GB/s": "5.243",
        "clock": "warning: unvalidated, converted at the measured 1000000000 Hz: " "clock evidence contains errors",
    }
    failures = []
    for output, tables in (
        ("table mode", tables_of(parse_jsonl(str(flagged), flagged.read_text()))),
        ("the report", markdown_tables((root / "report" / "MASTER_RESULTS.md").read_text())),
    ):
        ((_, (row,)),) = tables.values()
        failures += [
            f"{output}: a failed clock's {head} is {row.get(head)!r}, expected {cell!r}"
            for head, cell in expected.items()
            if row.get(head) != cell
        ]
    if json.loads((root / "report" / "campaign.json").read_text())["wall_records_with_clock_warning"] != 1:
        failures.append("campaign.json does not count the clock warning")
    del record["clock"]["rate_tolerance_ppm"]
    malformed = root / "malformed" / source.name
    malformed.parent.mkdir()
    malformed.write_text(json.dumps(record) + "\n")
    try:
        read_attempt(configs[source.stem], [malformed])
        failures.append("a malformed clock did not fail its attempt")
    except ValueError as error:
        if "AICLK rate tolerance is invalid" not in str(error):
            failures.append(f"a malformed clock failed for another reason: {error}")
    return failures


def main(directory, manifest, files):
    configs = {c["id"]: c for c in json.loads(Path(manifest).read_text())["configurations"]}
    paths = sorted(Path(directory).glob("*.jsonl"))
    failures = []
    if len(paths) != int(files):
        failures.append(f"expected {files} record files, found {len(paths)}")
    configured = []
    for path in paths:
        try:
            text = path.read_text()
            rows = [json.loads(line) for line in text.splitlines()]
            check_records(path, rows)
            if path.stem not in configs:
                raise ValueError("is not a manifest configuration")
            configured.append(path)
        except (ParseError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{path.stem}: {type(error).__name__}: {error}")
    try:
        check_report(directory, configs, configured)
        failures += check_clock_warning(directory, configs, configured)
        failures += check_outputs(directory, configured)
    except (ParseError, ValueError, KeyError, TypeError) as error:
        failures.append(f"campaign report: {type(error).__name__}: {error}")
    for failure in failures:
        print("FAIL:", failure, file=sys.stderr)
    print(f"record round trip: {len(paths)} files, {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
