# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Directed-link non-posted DRAM read/write transport model. No end-to-end bandwidth profile is charged to banks.

Packets serialize on individual links. Downstream queues are unbounded: finite
router buffers/VC credits are NOT asserted to be modeled by this first version.
"""

import argparse
from collections import Counter
import heapq
import json
import math
from pathlib import Path
from routing import links
from read_service import ReadService


def predict(topology, case, issue_cycles=(0, 0), contention=True, read_service=None):
    if read_service is not None:
        read_service.validate(topology)
    cores = case["cores"]
    payload = case["payload_bytes"]
    mode = case["mode"]
    if mode not in ("noc0", "noc1", "split"):
        raise ValueError(mode)
    traffic = case.get("traffic", "read")
    if traffic not in ("read", "write", "mixed"):
        raise ValueError("traffic must be read, write, or mixed")
    if traffic == "mixed" and mode != "split":
        raise ValueError("mixed traffic requires split mode: NoC0 reads, NoC1 writes")
    if case.get("completion", "ack") != "ack":
        raise ValueError("only remote completion is modeled; source flush and posted writes are unsupported")
    if not 1 <= payload <= 16384 or cores < 1:
        raise ValueError("positive core count and a single-packet payload of 1..16384 bytes required")
    if case["banks"] not in ("balanced", "hotspot"):
        raise ValueError("unknown bank mapping")
    if any(not math.isfinite(x) or x < 0 for x in issue_cycles):
        raise ValueError("issue costs must be finite and nonnegative")
    bank_count = len(topology["banks"])
    grid = topology["physical_grid"]
    if "logical_coords" in case:
        cols = topology["worker_grid"][0]
        workers = [topology["workers"][y * cols + x] for x, y in case["logical_coords"]]
    else:
        workers = topology["workers"][:cores]
    if len(workers) != cores:
        raise ValueError("core mapping mismatch")
    banks = [topology["banks"], topology["banks_noc1"]]
    parts = 2 if mode == "split" else 1
    batch = case.get("batch", 16)
    pages = case.get("pages", 64)
    iterations = case.get("iterations", 4)
    if pages < 1 or batch < 1 or iterations < 1 or pages % batch or batch % parts:
        raise ValueError("partial batch not supported")
    events = []
    serial = 0
    free = {}
    loads = Counter()
    waits = Counter()
    counts = Counter()
    write_counts = Counter()
    cache = {}
    ends = {}
    bank_load = Counter()
    response_load = Counter()
    request_load = Counter()

    def push(t, fn):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (t, serial, fn))

    def claim(key, t, cost):
        at = max(t, free.get(key, 0)) if contention else t
        free[key] = at + cost
        loads[key] += cost
        waits[key] += at - t
        return at

    def flight(t, c, noc, bank, response, done, direction):
        src, dst = (banks[noc][bank], workers[c]) if response else (workers[c], banks[noc][bank])
        k = (c, noc, bank, response)
        if k not in cache:
            cache[k] = (
                [(("inject", noc, *src), 5)]
                + [(link, 9) for link in links(src, dst, noc, grid)]
                + [(("eject", noc, *dst), 5)]
            )
        route = cache[k]
        carries_payload = response if direction == "read" else not response
        flits = 1 + math.ceil(payload / 64) if carries_payload else 1
        traffic = response_load if response else request_load

        def hop(now, i):
            if i == len(route):
                push(now + flits - 1, done)
                return
            resource, delay = route[i]
            traffic[resource] += flits
            at = claim(resource, now, flits)
            push(at + delay, lambda now: hop(now, i + 1))

        hop(t, 0)

    def request(t, c, noc, page, done, direction):
        bank = 0 if case["banks"] == "hotspot" else page % bank_count
        (counts if direction == "read" else write_counts)[noc] += 1

        def bank_service(now):
            occupancy = math.ceil(payload / 47.0805)
            bank_load[bank] += occupancy
            at = claim(("bank", bank), now, occupancy)
            # Writes arrive with their payload tail already accounted for. The
            # 22-cycle incremental bank delay is a simulator-derived hypothesis,
            # not another payload serialization or an ISA guarantee.
            delay = 126 + max(0, occupancy - (1 + math.ceil(payload / 64))) if direction == "read" else 22
            push(at + delay, lambda now: flight(now, c, noc, bank, True, done, direction))

        def service(now):
            if direction == "read" and read_service is not None:
                at = claim(("read_admission", noc, bank), now, read_service.interval(noc))
                push(at, bank_service)
            else:
                bank_service(now)

        flight(t, c, noc, bank, False, service, direction)

    def launch(t, c, part, round_index):
        noc = part if parts == 2 else int(mode[-1])
        direction = ("read" if part == 0 else "write") if traffic == "mixed" else traffic
        if round_index == iterations * pages // batch:
            ends[c, part] = t
            return
        remaining = batch // parts

        def complete(now):
            nonlocal remaining
            remaining -= 1
            if remaining == 0:
                push(now, lambda at: launch(at, c, part, round_index + 1))

        offset = (round_index % (pages // batch)) * batch
        for j in range(batch // parts):
            page = c * pages + offset + part * (batch // parts) + j
            # A write cannot enter the NoC before its command has been issued.
            # Preserve the legacy read timing origin until separately revalidated.
            issue_index = j + (1 if direction == "write" else 0)
            push(
                t + issue_index * issue_cycles[noc],
                lambda now, page=page: request(now, c, noc, page, complete, direction),
            )

    for c in range(cores):
        for part in range(parts):
            launch(0, c, part, 0)
    while events:
        t, _, fn = heapq.heappop(events)
        fn(t)
    assert len(ends) == cores * parts

    def hottest(counter, kind, n=8):
        return [dict(resource=list(k), flit_cycles=v / iterations) for k, v in counter.most_common() if k[0] == kind][
            :n
        ]

    link_peak = max((v for k, v in loads.items() if k[0] == "link"), default=0) / iterations
    return dict(
        estimated_cycles=max(ends.values()) / iterations,
        estimated_us=max(ends.values()) / iterations / topology["clock_mhz"],
        issue_cycles_per_request=list(issue_cycles),
        contention=contention,
        read_requests_by_noc={str(n): count // iterations for n, count in counts.items()},
        read_bytes_per_iteration=sum(counts.values()) // iterations * payload,
        write_requests_by_noc={str(n): count // iterations for n, count in write_counts.items()},
        write_bytes_per_iteration=sum(write_counts.values()) // iterations * payload,
        traffic=traffic,
        completion="remote_ack" if traffic != "read" else "read_response",
        write_admission_characterized=False,
        write_bank_delay_cycles=22 if write_counts else None,
        link_occupancy_lower_bound_us=link_peak / topology["clock_mhz"],
        bank_occupancy_lower_bound_us=max(bank_load.values()) / iterations / topology["clock_mhz"],
        hottest_response_links=hottest(response_load, "link"),
        hottest_request_links=hottest(request_load, "link"),
        hottest_all_links=hottest(loads, "link"),
        model=(
            "characterized_read_admission_plus_directed_links"
            if read_service and traffic != "write"
            else "physical_bank_service_plus_directed_links"
        ),
        read_service_profile=read_service.provenance if read_service and traffic != "write" else None,
        finite_router_buffers=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--read-service-profile", type=Path)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    topology = json.loads(a.topology.read_text())
    service = ReadService(a.read_service_profile) if a.read_service_profile else None
    rows = [json.loads(s) for s in a.measurements.read_text().splitlines()]
    # Issue-loop costs come ONLY from one-core, single-NoC cases; never use
    # multicore issue times, which include backpressure we model separately.
    issue = {}
    for r in rows:
        if r["cores"] == 1 and r["mode"] != "split":
            issue[int(r["mode"][-1]), r["payload_bytes"]] = next(iter(r["issue_cycles"].values()))
    results = []
    for r in rows:
        if r["clock_mhz"] != topology["clock_mhz"] or r["bank_count"] != len(topology["banks"]):
            raise ValueError("profile topology/clock mismatch")
        costs = tuple(issue[n, r["payload_bytes"]] for n in (0, 1))
        estimate = predict(topology, r, costs, read_service=service)
        isolated = predict(topology, r, costs, contention=False)
        measured = r["median_cycles"] / r["clock_mhz"]
        results.append(
            dict(
                case=r,
                prediction=estimate,
                uncontended_us=isolated["estimated_us"],
                measured_us=measured,
                error_pct=100 * (estimate["estimated_us"] / measured - 1),
            )
        )
        print(
            r["cores"],
            r["payload_bytes"],
            r["mode"],
            r["placement"],
            r["banks"],
            round(estimate["estimated_us"], 3),
            round(measured, 3),
            flush=True,
        )
    a.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
