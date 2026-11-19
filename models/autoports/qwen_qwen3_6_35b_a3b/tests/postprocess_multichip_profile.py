# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded-memory device report processing from an immutable Tracy capture.

Export TT_ host roots directly to avoid malformed multiline Program_ source
fields in the unfiltered native export. Optional CompileProgram/write-buffer
child host timings are excluded; all device timing and op metadata are intact.
Raw captures are immutable. Native report conversion remains unchanged.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path


def compact(source, target):
    counts = dict(total=0, kept=0, ignored=0, root=0, child=0)
    with source.open(newline="") as incoming, target.open("w", newline="") as outgoing:
        reader = csv.reader(incoming)
        fields = next(reader)
        name, parent = fields.index("name"), fields.index("special_parent_text")
        writer = csv.writer(outgoing)
        writer.writerow(fields)
        for row in reader:
            assert len(row) == len(fields), "malformed host timing row"
            root = "TT_DNN" in row[name] or "TT_METAL" in row[name]
            child = "id:" in row[parent]
            counts["total"] += 1
            if root or child:
                writer.writerow(row)
                counts["kept"] += 1
                counts["root"] += int(root)
                counts["child"] += int(child)
            else:
                counts["ignored"] += 1
            if counts["total"] % 10000000 == 0:
                print(f"Host timing rows scanned: {counts['total']}; kept: {counts['kept']}", flush=True)
    assert counts["root"] > 0 and counts["total"] == counts["kept"] + counts["ignored"]
    return counts


def compact_device(source, target):
    from tracy.process_ops_logs import (
        build_sub_device_id_lookup_from_device_csv,
        normalize_device_csv_trace_field,
        parse_device_csv_meta_data,
    )

    counts = dict(total=0, metadata=0, unique=0)
    selected = {}
    with source.open(newline="") as incoming:
        preamble = incoming.readline()
        reader = csv.reader(incoming)
        fields = next(reader)
        assert len(fields) == 15
        for row in reader:
            assert len(row) == len(fields), "malformed device metadata row"
            counts["total"] += 1
            metadata = parse_device_csv_meta_data(row[14])
            if metadata is not None and "sub_device_id" in metadata:
                counts["metadata"] += 1
                key = (int(row[0]), int(row[7]), *(normalize_device_csv_trace_field(v) for v in row[8:10]))
                sid = int(metadata["sub_device_id"])
                assert key not in selected or selected[key][0] == sid, "conflicting sub-device metadata"
                selected[key] = (sid, row)
            if counts["total"] % 10000000 == 0:
                print(f"Device rows scanned: {counts['total']}; unique metadata: {len(selected)}", flush=True)
    with target.open("w", newline="") as outgoing:
        outgoing.write(preamble)
        writer = csv.writer(outgoing)
        writer.writerow(fields)
        writer.writerows(row for _, row in selected.values())
    counts["unique"] = len(selected)
    assert build_sub_device_id_lookup_from_device_csv(target) == {key: sid for key, (sid, _) in selected.items()}
    return counts


def run(artifact):
    from tracy.common import PROFILER_BIN_DIR, TRACY_CSVEXPROT_TOOL, resolve_tracy_tool_path
    from tracy.process_ops_logs import process_ops

    artifact = artifact.resolve()
    raw = artifact / "raw/.logs"
    assert json.loads((artifact / "correctness.json").read_text())["passed"]
    trace = raw / "tracy_profile_log_host.tracy"
    assert trace.is_file() and (raw / "cpp_device_perf_report.csv").is_file()
    exporter = resolve_tracy_tool_path(PROFILER_BIN_DIR, TRACY_CSVEXPROT_TOOL)
    assert exporter
    for name, options in (
        ("tracy_ops_root_times.csv", ["-u", "-c", "-f", "TT_"]),
        ("tracy_ops_data.csv", ["-m", "-s", ";"]),
    ):
        target = raw / name
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".partial")
            assert not temporary.exists(), "inspect interrupted export before retrying"
            with temporary.open("w") as stream:
                subprocess.run([str(exporter), *options, str(trace)], stdout=stream, check=True)
            temporary.replace(target)
    processing = artifact / "postprocess"
    logs = processing / ".logs"
    logs.mkdir(parents=True, exist_ok=True)
    for source in raw.iterdir():
        if source.name not in ("tracy_ops_times.csv", "tracy_ops_root_times.csv", "profile_log_device.csv"):
            target = logs / source.name
            if not target.exists():
                target.symlink_to(source)
    target = logs / "tracy_ops_times.csv"
    assert not target.exists(), "use preserved processing results rather than repeat compaction"
    host_source = raw / "tracy_ops_root_times.csv"
    counts = compact(host_source, target)
    (artifact / "host_times_compaction.json").write_text(
        json.dumps(
            dict(
                source=str(host_source),
                compact=str(target),
                predicate="name contains TT_DNN or TT_METAL, or special_parent_text contains id:",
                semantics="native root ordering and fields unchanged; no aggregation; device timing unchanged",
                excluded="optional host child timings (CompileProgram and HWCommandQueue_write_buffer)",
                export_filter="case-sensitive TT_ zone names; excludes malformed Program_ source-file fields",
                counts=counts,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Host timing compaction complete: {counts}", flush=True)
    device_target = logs / "profile_log_device.csv"
    assert not device_target.exists()
    device_counts = compact_device(raw / device_target.name, device_target)
    (artifact / "device_metadata_compaction.json").write_text(
        json.dumps(
            dict(
                source=str(raw / device_target.name),
                compact=str(device_target),
                semantics="last metadata row per device/run/trace/session; conflicts rejected; native lookup verified",
                timing_source=str(raw / "cpp_device_perf_report.csv"),
                counts=device_counts,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Device metadata compaction complete: {device_counts}", flush=True)
    process_ops(processing, name_append=None, date=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    run(parser.parse_args().artifact)
