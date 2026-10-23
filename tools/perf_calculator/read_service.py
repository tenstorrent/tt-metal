# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent effective read-admission characterization; not an ISA timing constant.

A per-(NoC, allocator bank) resource represents the observed request-rate limit.
Its physical location is unresolved. DRAM byte service remains a separate resource
shared by both NoCs. Only balanced single-NoC small-request probes train this term;
no tilize result or split-reader result participates in its derivation.
"""

import hashlib
import json
import math
from pathlib import Path


def topology_signature(t):
    fields = {k: t[k] for k in ("banks", "banks_noc1", "physical_grid", "clock_mhz")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


class ReadService:
    def __init__(self, profile):
        self.profile = json.loads(Path(profile).read_text()) if isinstance(profile, (str, Path)) else profile
        self.intervals = self.profile["cycles_per_request_by_noc_bank"]
        if len(self.intervals) != 2 or any(not math.isfinite(x) or x <= 0 for x in self.intervals):
            raise ValueError("invalid read service intervals")

    def validate(self, topology):
        if self.profile["topology_signature"] != topology_signature(topology):
            raise ValueError("read service profile topology/clock mismatch")

    def interval(self, noc):
        return self.intervals[noc]

    @property
    def provenance(self):
        return self.profile


def characterize(measurements, topology, cores=110, payload=128):
    raw = Path(measurements).read_bytes()
    rows = [json.loads(s) for s in raw.decode().splitlines()]
    selected = []
    intervals = []
    for noc in (0, 1):
        matches = [
            r
            for r in rows
            if r["cores"] == cores
            and r["payload_bytes"] == payload
            and r["mode"] == f"noc{noc}"
            and r["banks"] == "balanced"
            and r["placement"] == "row"
        ]
        if len(matches) != 1:
            raise ValueError("need exactly one training case for each NoC")
        r = matches[0]
        nb = len(topology["banks"])
        if r["clock_mhz"] != topology["clock_mhz"] or r["bank_count"] != nb:
            raise ValueError("training topology/clock mismatch")
        # Timings are already per iteration. Do not multiply by iterations again.
        requests_per_bank = r["cores"] * r["pages"] / nb
        intervals.append(r["median_cycles"] / requests_per_bank)
        selected.append(
            {
                k: r[k]
                for k in ("cores", "payload_bytes", "mode", "pages", "batch", "median_cycles", "clock_mhz", "host")
            }
        )
    return dict(
        version=1,
        model="effective_read_admission_per_noc_bank",
        physical_resource="unresolved: aggregate read-path characterization, not DRAM array occupancy",
        topology_signature=topology_signature(topology),
        cycles_per_request_by_noc_bank=intervals,
        source_file=Path(measurements).name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        training_cases=selected,
        derivation="median_cycles_per_iteration / (cores * pages_per_core / bank_count)",
        limitations=[
            "Finite credit counts and VC buffers are not identified.",
            "Balanced saturation timing includes residual routing/barrier effects.",
            "Per-bank localization and payload-independent interval are hypotheses requiring held-out validation.",
        ],
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurements", type=Path, required=True)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(characterize(a.measurements, json.loads(a.topology.read_text())), indent=2) + "\n")
