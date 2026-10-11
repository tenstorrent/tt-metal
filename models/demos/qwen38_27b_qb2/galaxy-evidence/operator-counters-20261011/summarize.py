# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce this capture's aggregates using only the sealed local receipts."""

import collections
import csv
import gzip
import hashlib
import io
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def data(name):
    path = ROOT / "raw" / name
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def read(name):
    return json.loads(data(name))


def aggregate(rows, label_key):
    grouped = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for row in rows:
        pair = grouped[row[label_key]][row["counter"]]
        pair[0] += row["value_sum"]
        pair[1] += row["reference_sum"]
    result = {}
    for label, counters in grouped.items():
        grants = {}
        for name, (value, _) in counters.items():
            if not name.endswith("_GRANT"):
                continue
            base = name.removesuffix("_GRANT")
            requests = next((counters[k][0] for k in (base, base + "_RISC", base + "_TRISC") if k in counters), 0)
            if requests:
                grants[name] = value / requests
        result[label] = {
            "weighted_activity": {k: v / r for k, (v, r) in counters.items()},
            "grant_per_request": grants,
        }
    return result


def timings(name):
    grouped = collections.defaultdict(list)
    for row in read(name)["kernels"]:
        grouped[row.get("label", row.get("kind"))].append(row["median_us"])
    return {k: statistics.median(v) for k, v in grouped.items()}


def eager_boundaries(phase):
    directory = ROOT / "raw" / "large-operator-v4" / phase / "tracy/reports"
    files = list(directory.glob("*/ops_perf_results*.csv*"))
    assert len(files) == 1
    path = files[0]
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    groups = collections.defaultdict(list)
    active = None
    for row in csv.DictReader(io.StringIO(raw.decode())):
        if row["OP TYPE"] == "signpost":
            tag = row["OP CODE"]
            active = tag.removesuffix("_BEGIN").removeprefix("LARGE_OP_") if tag.endswith("_BEGIN") else None
        elif active and row["DEVICE ID"]:
            groups[active.rsplit("_", 1)[0], row["DEVICE ID"], active.rsplit("_", 1)[1]].append(row)
    results = collections.defaultdict(list)
    for (label, device, repeat), rows in groups.items():
        first = min(float(r["DEVICE FW START CYCLE"]) for r in rows)
        last = max(float(r["DEVICE FW END CYCLE"]) for r in rows)
        # Infer cycles/ns independently from each op; the CSV rounds duration ns.
        clock = statistics.median(
            (float(r["DEVICE FW END CYCLE"]) - float(r["DEVICE FW START CYCLE"])) / float(r["DEVICE FW DURATION [ns]"])
            for r in rows
        )
        results[label].append(
            dict(
                device=int(device),
                repeat=int(repeat),
                firmware_span_us=(last - first) / clock / 1000,
                kernel_sum_us=sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows) / 1000,
                ops=[r["OP CODE"] for r in rows],
            )
        )
    return results


def main():
    verified = 0
    for name in ("manifest.json", "extra-manifest.json"):
        for path, expected in read("large-operator-publication-v1/" + name).items():
            content = data(path)
            assert len(content) == expected["bytes"]
            assert hashlib.sha256(content).hexdigest() == expected["sha256"], path
            verified += 1
    queue = read("large-operator-v4/queue.json")
    assert queue["state"] == "completed" and queue["cleanup_completed"]
    assert len(queue["runs"]) == 17
    for row in queue["runs"]:
        assert row["state"] == "completed"
        assert all(row[k] for k in ("hardware_passed", "cleanup_completed", "exact_output_match", "coverage_passed"))

    old_control = read("large-operator-v2/before/operators.json")
    new_control = read("large-operator-v4/before/operators.json")
    for key in ("source_sha256", "precision", "device_ids", "attention_geometry"):
        assert old_control[key] == new_control[key], key
    assert {c["label"]: c["samples"][0]["hashes"] for c in old_control["cases"]} == {
        c["label"]: c["samples"][0]["hashes"] for c in new_control["cases"]
    }

    operators = []
    sources = []
    for root, passes in (("large-operator-v2", ("pass-00", "pass-01", "pass-02")), ("large-operator-v4", ())):
        paths = [f"{root}/{p}/counters.json" for p in passes]
        if root.endswith("v4"):
            paths = [f"{root}/{r['label']}/counters.json" for r in queue["runs"] if r["groups"]]
        for path in paths:
            value = read(path)
            assert value["counter_coverage_passed"]
            operators.extend(value["summaries"])
            sources.append(path)

    gdn = []
    gdn_paths = [f"gdn-counter-v4/pass-{n:02d}/counters.json" for n in range(3)]
    gdn_paths += ["gdn-counter-space-recovery-v1/instruction-counters.json"]
    gdn_paths += [f"gdn-counter-v5/pass-{n:02d}/counters.json" for n in range(6)]
    for path in gdn_paths:
        value = read(path)
        assert value["all_requested_counters_present"] and value["active_core_calls"] == 23040
        gdn.extend(r for r in value["summaries"] if r["batch"] == 16 and r["padding"] == "skip" and r["profiled"] == 0)
    controls = {}
    for phase, path in (
        ("before", "gdn-counter-v5/reused-control-analysis.json"),
        ("after", "gdn-counter-v5/after/counters.json"),
    ):
        grouped = collections.defaultdict(list)
        for row in read(path)["kernels"]:
            if row["batch"] == 16 and row["padding"] == "skip" and row["profiled"] == 0:
                grouped[row["kind"]].append(row["median_us"])
        controls[phase] = {k: statistics.median(v) for k, v in grouped.items()}

    result = dict(
        verified_files=verified,
        scope="B16/32K TP4 isolated eager operators; GDN B16 skip-padding, no extra phase zones",
        limitations=[
            "Activity fractions and grant/request ratios are not physical DRAM utilization or roofline efficiency.",
            "Counters span separate instrumented captures; they do not establish simultaneous overlap or causality.",
            "L1_0/L1_1 are counter selector groups, not proof covering every physical L1 bank or DRAM-reader core.",
            "Eager boundary firmware spans include host issue gaps; do not project them onto traced full-model latency.",
            "Kernel duration sums are not additive whole-model wall time or removable latency.",
        ],
        operator_sources=sources,
        operator_cross_version_model_and_outputs_match=True,
        operators=aggregate(operators, "label"),
        operator_kernel_controls_us={
            phase: timings(f"large-operator-v4/{phase}/counters.json") for phase in ("before", "after")
        },
        eager_boundary_controls={phase: eager_boundaries(phase) for phase in ("before", "after")},
        gdn_sources=gdn_paths,
        gdn=aggregate(gdn, "kind"),
        gdn_kernel_controls_us=controls,
        model_tsu_unchanged=20.884562,
        full_model_qualified=False,
        promoted_to_serving=False,
    )
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {verified} sealed files; summarized 17 completed operator captures and 10 GDN counter groups.")


if __name__ == "__main__":
    main()
