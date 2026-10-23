# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate transport predictions using independent single-core issue-loop costs."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from noc_routing_model import predict
from read_service import ReadService


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--issue-measurements", type=Path, required=True)
    parser.add_argument("--read-service-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    topology = json.loads(a.topology.read_text())
    service = ReadService(a.read_service_profile) if a.read_service_profile else None
    costs = {}
    training = []
    # Only first-batch issue scopes of single-core, single-NoC batch-16 cases
    # train the issue term. Do not use full transaction spans or multicore scopes.
    for r in read(a.issue_measurements):
        if r["cores"] == 1 and r["mode"] in ["noc0", "noc1"] and r["batch"] == 16 and r["banks"] == "balanced":
            if r["clock_mhz"] != topology["clock_mhz"] or r["bank_count"] != len(topology["banks"]):
                raise ValueError("training hardware mismatch")
            k = (r.get("traffic", "read"), r["mode"], r["payload_bytes"])
            if k in costs:
                raise ValueError("duplicate issue calibration case")
            costs[k] = next(iter(r["issue_cycles"].values()))
            training.append(dict(traffic=k[0], mode=k[1], payload_bytes=k[2], issue_cycles=costs[k]))
    output = []
    for r in read(a.measurements):
        if r["clock_mhz"] != topology["clock_mhz"] or r["bank_count"] != len(topology["banks"]):
            raise ValueError("validation hardware mismatch")
        traffic = r.get("traffic", "read")
        directions = ["read", "write"] if traffic == "mixed" else [traffic, traffic]
        issue = tuple(costs[(directions[n], f"noc{n}", r["payload_bytes"])] for n in [0, 1])
        estimate = predict(topology, r, issue_cycles=issue, read_service=service)
        measured = r["median_cycles"] / r["clock_mhz"]
        output.append(
            dict(
                case={k: r[k] for k in ["cores", "mode", "payload_bytes", "batch", "banks"]},
                traffic=traffic,
                prediction=estimate,
                measured_us=measured,
                error_percent=100 * (estimate["estimated_us"] / measured - 1),
            )
        )
    result = dict(
        issue_training=training,
        cases=output,
        median_absolute_error_percent=statistics.median(abs(r["error_percent"]) for r in output),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                a.topology,
                a.measurements,
                a.issue_measurements,
                Path(__file__).with_name("noc_routing_model.py"),
                Path(__file__).with_name("routing.py"),
                Path(__file__).with_name("read_service.py"),
            ]
            + ([a.read_service_profile] if a.read_service_profile else [])
        },
        note="Only issue-loop scopes supply calibration. No write-admission or total-runtime penalty is fitted; mixed-read verification interference must be assessed separately.",
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for r in output:
        print(
            r["traffic"],
            r["case"],
            round(r["prediction"]["estimated_us"], 3),
            round(r["measured_us"], 3),
            round(r["error_percent"], 1),
        )
    print("Median absolute error %:", result["median_absolute_error_percent"])


if __name__ == "__main__":
    main()
