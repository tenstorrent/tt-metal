# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Extract aligned per-operation data for the fixed-batch layer comparison."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from models.demos.gemma4_d_p.scripts.layer_perf_report import find_ops_csv, run_tt_perf_report, write_cell_ops_csv

COLLECTIVES = ("AllGather", "ReduceScatter", "AllReduce")
MATMULS = (
    "Q/K/V projection",
    "Attention output projection",
    "MLP gate projection",
    "MLP up projection",
    "MLP down projection",
)
LABELS = {
    "AllGatherDeviceOperation": "TP all-gather",
    "ReduceScatterDeviceOperation": "TP reduce-scatter",
    "UpdatePaddedKvCacheDeviceOperation": "KV-cache writes",
    "RingJointSDPADeviceOperation": "Attention (SDPA)",
    "LayerNormDeviceOperation": "RMS normalization",
    "InterleavedToShardedDeviceOperation": "Norm input redistribution",
    "ShardedToInterleavedDeviceOperation": "Norm output redistribution",
    "NlpCreateHeadsDeviceOperation": "Split Q/K/V heads",
    "NLPConcatHeadsDeviceOperation": "Combine attention heads",
    "SliceDeviceOperation": "Local tensor slices",
    "ConcatDeviceOperation": "Local tensor concatenation",
    "TypecastDeviceOperation": "KV dtype conversion",
    "RotaryEmbeddingLlamaDeviceOperation": "RoPE rotation",
    "RotaryEmbeddingDeviceOperation": "Rotary transform",
    "GatherCodegenDeviceOperation": "Rotary-channel gather",
    "TilizeWithValPaddingDeviceOperation": "Rotary weight tiling",
    "BinaryNgDeviceOperation": "Elementwise operations",
}


def group_for(code):
    if "RingJointSDPA" in code:
        return "Attention"
    if "UpdatePaddedKvCache" in code:
        return "Cache writes"
    if any(c in code for c in COLLECTIVES):
        return "TP communication"
    if "Matmul" in code:
        return "Projections / MLP"
    if "SliceDevice" in code or "ConcatDevice" in code:
        return "Local slices / concat"
    return "Norms / RoPE / other"


def merge_device_ops(rows, expected_devices=32):
    """Match operations in execution order; never sum durations across parallel devices.

    Match tt-perf-report 1.4.1: maximum duration for ordinary ops, mean for CCL.
    Fail on missing devices, mismatched sequences or missing kernel durations.
    """
    devices = defaultdict(list)
    for row in rows:
        if row["OP TYPE"] == "tt_dnn_device":
            devices[int(row["DEVICE ID"])].append(row)
    assert len(devices) == expected_devices, f"Expected {expected_devices} devices, got {len(devices)}"
    for values in devices.values():
        values.sort(key=lambda row: int(row["GLOBAL CALL COUNT"]))
    counts = {len(values) for values in devices.values()}
    assert len(counts) == 1 and next(iter(counts)), f"Incomplete device operation sequences: {counts}"
    merged = []
    matmul_index = 0
    for index, copies in enumerate(zip(*devices.values())):
        codes = {row["OP CODE"] for row in copies}
        assert len(codes) == 1, f"Mismatched operation {index}: {codes}"
        code = copies[0]["OP CODE"]
        durations = [float(row["DEVICE KERNEL DURATION [ns]"]) / 1000 for row in copies]
        assert min(durations) > 0
        selected = copies[durations.index(max(durations))]
        if "MatmulDeviceOperation" == code:
            label = MATMULS[matmul_index]
            matmul_index += 1
        else:
            label = LABELS.get(code, code.removesuffix("DeviceOperation"))
        merged.append(
            dict(
                index=index,
                code=code,
                label=label,
                group=group_for(code),
                us=statistics.mean(durations) if any(c in code for c in COLLECTIVES) else max(durations),
                min_device_us=min(durations),
                max_device_us=max(durations),
                mean_device_us=statistics.mean(durations),
                input0=[selected.get(f"INPUT_0_{dim}_PAD[LOGICAL]", "") for dim in "WZYX"],
                fidelity=selected.get("MATH FIDELITY", ""),
                trace_id=selected.get("METAL TRACE ID", ""),
                replay_id=selected.get("METAL TRACE REPLAY SESSION ID", ""),
            )
        )
    assert matmul_index == 5, f"Expected five layer projections, got {matmul_index}"
    assert len({(op["trace_id"], op["replay_id"]) for op in merged}) == 1
    return merged


def read_cells(root, mode, output):
    manifest_paths = list((root / mode / "summaries" / "layer_perf").glob("manifest_*.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text())
    source = find_ops_csv(root / mode / "profiler")
    assert source is not None
    # Retain only signposted replay rows. Warmup and full model loading can be large.
    windows = {cell["start_signpost"]: (cell, []) for cell in manifest["cells"]}
    active = None
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["OP TYPE"] == "signpost":
                if row["OP CODE"] in windows:
                    assert active is None
                    active = windows[row["OP CODE"]]
                elif active is not None and row["OP CODE"] == active[0]["stop_signpost"]:
                    active = None
            elif active is not None:
                active[1].append(row)
    assert active is None
    cells = []
    for cell, raw in windows.values():
        ops = merge_device_ops(raw)
        name = f"{mode}_{cell['layer_type']}_chunk{cell['chunk_idx']}"
        scratch = root / "cell_inputs"
        scratch.mkdir(exist_ok=True)
        sliced = scratch / f"{name}.csv"
        reports = output / "layer_reports"
        reports.mkdir(exist_ok=True)
        report = reports / f"{name}.csv"
        bounded = [
            dict(**{"OP TYPE": "signpost", "OP CODE": cell["start_signpost"]}),
            *raw,
            dict(**{"OP TYPE": "signpost", "OP CODE": cell["stop_signpost"]}),
        ]
        assert write_cell_ops_csv(fieldnames, bounded, cell["start_signpost"], cell["stop_signpost"], sliced)
        assert run_tt_perf_report(sliced, cell["start_signpost"], cell["stop_signpost"], report)
        text_report = report.with_suffix(".txt")
        text_report.write_text(
            "\n".join(line.rstrip() for line in text_report.read_text().splitlines()).rstrip() + "\n"
        )
        with report.open(newline="") as handle:
            report_rows = [row for row in csv.DictReader(handle) if row.get("Device Time")]
        assert len(report_rows) == len(ops)
        assert abs(sum(float(row["Device Time"]) for row in report_rows) - sum(op["us"] for op in ops)) < 1e-5
        grouped = defaultdict(lambda: {"calls": 0, "us": 0.0})
        for op in ops:
            grouped[op["label"]]["calls"] += 1
            grouped[op["label"]]["us"] += op["us"]
        cells.append(
            dict(
                mode=mode,
                layer=cell["layer_type"],
                layer_index=cell["layer_idx"],
                chunk_index=cell["chunk_idx"],
                start=cell["chunk_start"],
                end=cell["request_end"],
                batch_size=manifest["batch_size"],
                useful_tokens=manifest["chunk_size"],
                wall_samples_ms=cell["samples_ms"],
                wall_median_ms=cell["measured_ms"],
                kernel_ms=sum(op["us"] for op in ops) / 1000,
                operations=ops,
                grouped=dict(grouped),
                source_csv=str(source),
                perf_report_csv=str(report.relative_to(output)),
            )
        )
    return cells


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report-repo", required=True, type=Path, help="Checkout of tt-perf-report main")
    args = parser.parse_args()
    sys.path.insert(0, str(args.report_repo / "src"))
    report_commit = subprocess.check_output(
        ["git", "-C", str(args.report_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    cells = [cell for mode in ("canonical", "chunked4") for cell in read_cells(args.input, mode, args.output)]
    method = (
        "test_prefill_layer_perf_chunk_n; global layer 5 and sliding-window layer 0; CP8/TP4; "
        "full model weights loaded, default math and activation placement. Isolated layers receive token embeddings "
        "and random initialized KV caches, as in the canonical layer test. Five replays per position; the final "
        "replay supplies operation timings. Embedding, RoPE lookup/packing, host staging and synchronization are "
        "outside the operation table. Per-cell tables are generated by tt-perf-report main. Device merge takes maximum kernel duration across "
        "32 devices for ordinary operations, mean for collectives. Each row sums its operation calls, never devices. "
        "Kernel sums are distinct from host wall time and from the unprofiled full-model measurements."
    )
    (args.output / "layer_measurements.json").write_text(
        json.dumps(dict(method=method, tt_perf_report_commit=report_commit, cells=cells), indent=2) + "\n"
    )
    flat = []
    for cell in cells:
        for op in cell["operations"]:
            flat.append({k: cell[k] for k in ("mode", "layer", "chunk_index", "start", "end")} | op)
        print(
            cell["mode"],
            cell["layer"],
            cell["chunk_index"],
            f"kernel={cell['kernel_ms']:.3f} ms",
            f"ops={len(cell['operations'])}",
        )
    write_csv(args.output / "layer_operations.csv", flat)
    aligned = []
    for layer in ("global", "local"):
        for position in ("first", "last"):
            selected = {}
            for mode in ("canonical", "chunked4"):
                subset = [c for c in cells if c["mode"] == mode and c["layer"] == layer]
                selected[mode] = (min if position == "first" else max)(subset, key=lambda c: c["start"])
            labels = list(dict.fromkeys(op["label"] for cell in selected.values() for op in cell["operations"]))
            for label in labels:
                left = selected["canonical"]["grouped"].get(label, {"calls": 0, "us": 0})
                right = selected["chunked4"]["grouped"].get(label, {"calls": 0, "us": 0})
                aligned.append(
                    dict(
                        layer=layer,
                        position=position,
                        operation=label,
                        canonical_calls=left["calls"],
                        batch_calls=right["calls"],
                        canonical_us=left["us"],
                        batch_us=right["us"],
                        extra_us=right["us"] - left["us"],
                    )
                )
    write_csv(args.output / "layer_comparison.csv", aligned)


if __name__ == "__main__":
    main()
