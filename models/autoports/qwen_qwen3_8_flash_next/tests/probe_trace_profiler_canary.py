# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-core trace producer/drain diagnostic; no Tracy host client is required.

Only the canary output is written. Profiler L1 access is read-only and happens
before destructive ReadDeviceProfiler drains. Layout comes from the actual
loaded library's HAL offset arrays, cross-checked against source contracts.
Numerical success and device timing acceptance are separate fields.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
COMMON = ROOT / "tt_metal/hw/inc/hostdev/profiler_common.h"
GENERATED = ROOT / "tt_metal/llrt/hal/generated/dev_msgs_impl.hpp"
MEMORY = ROOT / "tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h"
HAL = ROOT / "tt_metal/llrt/hal/tt-1xx/blackhole/bh_hal_tensix.cpp"

KERNEL = r"""
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr auto sa = TensorAccessorArgs<0>();
    constexpr auto da = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(sa, get_arg_val<uint32_t>(0), 64);
    const auto dst = TensorAccessor(da, get_arg_val<uint32_t>(1), 64);
    cb_reserve_back(tt::CBIndex::c_0, 1);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    noc_async_read(src.get_noc_addr(0), scratch, 64);
    noc_async_read_barrier();
    noc_async_write(scratch, dst.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def elf_array(path, symbol):
    """Read an ELF64 little-endian HAL constant, never load/execute the library."""
    listing = subprocess.check_output(["nm", "-C", "-S", str(path)], text=True)
    matches = [line.split(maxsplit=3) for line in listing.splitlines() if line.endswith(" " + symbol)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one unstripped HAL symbol: {symbol}")
    address, size = (int(item, 16) for item in matches[0][:2])
    data = Path(path).read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected ELF64 little-endian host library")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phsize, phcount = struct.unpack_from("<HH", data, 54)
    for i in range(phcount):
        kind, _, offset, vaddr, _, filesz, _, _ = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phsize)
        if kind == 1 and vaddr <= address and address + size <= vaddr + filesz:
            return list(struct.unpack_from("<" + "Q" * (size // 8), data, offset + address - vaddr))
    raise ValueError("HAL symbol is not file-backed")


def enum_indices(text, name):
    text = re.sub(r"//[^\n]*", "", text)
    body = re.search(r"enum\s+" + name + r"\s*\{([^}]+)\}", text).group(1)
    maximum = int(re.search(r"PROFILER_MAX_RISC_COUNT\s*=\s*(\d+)", text).group(1))
    result, value = {}, -1
    for item in body.split(","):
        if not item.strip():
            continue
        terms = item.strip().split("=")
        if len(terms) == 2:
            expr = terms[1].strip().replace("PROFILER_MAX_RISC_COUNT", str(maximum))
            if not re.fullmatch(r"\d+(?:\s*\*\s*\d+)*", expr):
                raise ValueError(f"Unrecognized enum expression: {expr}")
            value = 1
            for factor in expr.split("*"):
                value *= int(factor)
        else:
            value += 1
        result[terms[0].strip()] = value
    return result


def layout_from_library(library):
    generated = GENERATED.read_text()
    mailbox_fields = re.search(
        r"static const uintptr_t mailboxes_t\[\] = \{(.*?)// 3 array sizes", generated, re.S
    ).group(1)
    fields = re.findall(r"sizeof\(::mailboxes_t\)|offsetof\(::mailboxes_t, (\w+)\)", mailbox_fields)
    profiler_index = fields.index("profiler")
    prefix = "tt::tt_metal::blackhole::tensix_dev_msgs::offsets::"
    arrays = {
        name: elf_array(library, prefix + name) for name in ("mailboxes_t", "profiler_msg_t", "profiler_msg_buffer_t")
    }
    mailbox_base = int(re.search(r"#define MEM_MAILBOX_BASE\s+(\d+)", MEMORY.read_text()).group(1))
    if "GET_MAILBOX_ADDRESS_HOST(profiler)" not in HAL.read_text():
        raise ValueError("HAL profiler address contract changed")
    profiler = arrays["profiler_msg_t"]
    buffer = arrays["profiler_msg_buffer_t"]
    if len(profiler) != 5 or len(buffer) != 2 or profiler[1] != profiler[2] * 4 or buffer[0] != buffer[1] * 4:
        raise ValueError("HAL profiler array layout changed")
    if profiler[0] != profiler[1] + profiler[3] * buffer[0] or profiler[3] != 5:
        raise ValueError("Unexpected Blackhole profiler processor/buffer layout")
    control = enum_indices(COMMON.read_text(), "ControlBuffer")
    marker = enum_indices(COMMON.read_text(), "BufferIndex")
    marker_bits = {
        name: int(re.search(name + r"\s*=\s*(\d+)\s*;", COMMON.read_text()).group(1))
        for name in ("PROFILER_MARKER_TS_HIGH_BITS", "PROFILER_MARKER_TIMER_ID_BITS")
    }
    if control["TRACE_REPLAY_STATUS"] >= profiler[2] or marker["CUSTOM_MARKERS"] > buffer[1]:
        raise ValueError("Source indices exceed the loaded HAL buffer sizes")
    base = mailbox_base + arrays["mailboxes_t"][profiler_index]
    return {
        "library": str(Path(library).resolve()),
        "library_sha256": sha(library),
        "mailbox_base": mailbox_base,
        "control_address": base,
        "control_words": profiler[2],
        "aggregate_address": base + profiler[1],
        "aggregate_words": buffer[1],
        "hal_arrays": arrays,
        "control_indices": control,
        "marker_indices": marker,
        "marker_bits": marker_bits,
        "derivation": "Loaded ELF HAL generated arrays + source MEM_MAILBOX_BASE and enum indices; no hardcoded L1 addresses",
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (COMMON, GENERATED, MEMORY, HAL)},
    }


def csv_evidence(path, snapshots, chip):
    if not path.exists():
        return {"passed": False, "error": "No device CSV after close"}
    with path.open() as stream:
        header = stream.readline().strip()
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    endpoints = [row for row in rows if row["zone name"] in ("TRACE-FW", "TRACE-KERNEL")]
    result = {
        "path": str(path),
        "sha256": sha(path),
        "header": header,
        "rows": len(rows),
        "aggregate_endpoint_rows": len(endpoints),
        "passed": False,
    }
    if len(endpoints) != 8:
        result["error"] = "Expected four endpoints for each of exactly two one-core replays"
        return result
    cores = {(int(row["core_x"]), int(row["core_y"])) for row in endpoints}
    if len(cores) != 1:
        result["error"] = "Canary must have exactly one participating core"
        return result
    expected = {}
    for ordinal, snapshot in enumerate(snapshots, 1):
        for name, marker in snapshot["markers"].items():
            zone, phase = name.split("/")
            expected[(ordinal, zone, phase)] = marker["cycles"]
    observed = {}
    for row in endpoints:
        encoded = int(row["run host ID"])
        key = (encoded >> 10, row["zone name"], row["type"])
        if (
            encoded & 1023 != chip
            or int(row["PCIe slot"]) != chip
            or row["RISC processor type"] != "TENSIX_RISC_AGG"
            or key in observed
        ):
            result["error"] = "Unexpected or duplicate device/core/ordinal endpoint"
            return result
        observed[key] = int(row["time[cycles since reset]"])
    result.update(physical_csv_core=list(next(iter(cores))), exact_vs_known_core_predrain=observed == expected)
    if observed != expected:
        result["error"] = "CSV endpoints do not match the independently read known logical-core payload"
        return result
    intervals = []
    for ordinal in (1, 2):
        fs, fe, ks, ke = [
            observed[(ordinal, zone, phase)]
            for zone, phase in (
                ("TRACE-FW", "ZONE_START"),
                ("TRACE-FW", "ZONE_END"),
                ("TRACE-KERNEL", "ZONE_START"),
                ("TRACE-KERNEL", "ZONE_END"),
            )
        ]
        if not fs <= ks < ke <= fe:
            result["error"] = "Invalid endpoint ordering"
            return result
        intervals.append({"ordinal": ordinal, "firmware_cycles": fe - fs, "kernel_cycles": ke - ks})
    result.update(
        passed=True,
        intervals=intervals,
        core_identity_proof="All eight serialized timestamps exactly match pre-drain bytes from independently selected logical core(0,0); no assumption that translated coordinates equal physical CSV coordinates",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiler-dir", type=Path)
    parser.add_argument("--library", type=Path, default=ROOT / "build/lib/libtt_metal.so")
    parser.add_argument("--layout-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    layout = layout_from_library(args.library)
    if args.layout_only:
        args.output.write_text(json.dumps(layout, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: layout[key]
                    for key in ("control_address", "aggregate_address", "control_words", "aggregate_words")
                }
            )
        )
        return
    if args.profiler_dir is None:
        parser.error("--profiler-dir required for hardware canary")
    expected_env = {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_TRACE_PROFILER": "1",
        "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT": "128",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "0",
        "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY": "1",
        "TT_METAL_PROFILER_TRACE_TRACKING": "1",
        "TT_METAL_TRACE_ALLOC_TRACKING": "1",
    }
    if any(os.environ.get(key) != value for key, value in expected_env.items()):
        raise ValueError(f"Set exact classic canary environment before imports: {expected_env}")
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_STREAMING_PROFILER",
        "TT_METAL_PROFILER_MID_RUN_DUMP",
        "TT_METAL_DEVICE_PROFILER_DISPATCH",
        "TT_METAL_PROFILER_ACCUMULATE",
        "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES",
    ):
        if os.environ.get(key) not in (None, "", "0"):
            raise ValueError(f"Incompatible canary setting: {key}")
    args.profiler_dir.mkdir(parents=True, exist_ok=True)
    if (args.profiler_dir / ".logs/profile_log_device.csv").exists():
        raise ValueError("Use a fresh profiler directory; existing evidence is never overwritten")
    os.environ["TT_METAL_PROFILER_DIR"] = str(args.profiler_dir.resolve())
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit

    maps = [
        line for line in Path("/proc/self/maps").read_text().splitlines() if "libtt_metal" in line or "_ttnn" in line
    ]
    libraries = {Path(line.split()[-1]).resolve() for line in maps if "libtt_metal.so" in line}
    if libraries != {args.library.resolve()}:
        raise ValueError(f"Loaded library differs from inspected HAL: {libraries}")
    report = {
        "scope": __doc__,
        "layout": layout,
        "loaded_mappings": maps,
        "module_paths": {"ttnn": ttnn.__file__, "_ttnn": ttnn._ttnn.__file__},
        "module_sha256": {"ttnn": sha(ttnn.__file__), "_ttnn": sha(ttnn._ttnn.__file__)},
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("TT_METAL") or key in ("TT_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
        },
        "probe_sha256": sha(__file__),
        "kernel_sha256": hashlib.sha256(KERNEL.encode()).hexdigest(),
        "numerical_passed": False,
        "timing_passed": False,
        "passed": False,
        "snapshots": [],
    }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    replay_snapshots = []
    try:
        if "blackhole" not in str(mesh.arch()).lower():
            raise ValueError("This HAL layout experiment requires Blackhole")
        if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
            raise ValueError("Allocation tracker was not enabled at import time")
        chip = list(mesh.get_device_ids())[0]
        core = ttnn.CoreCoord(0, 0)
        translated = mesh.worker_core_from_logical_core(core)
        report["known_participating_core"] = {
            "logical": [0, 0],
            "translated": [translated.x, translated.y],
            "device": chip,
        }
        expected = torch.arange(16, dtype=torch.int64).reshape(1, 1, 16).to(torch.uint32)
        source = ttnn.from_torch(expected, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

        def forward():
            output = ttnn.empty(
                (1, 1, 16),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            runtime = ttnn.RuntimeArgs()
            runtime[0][0] = [source.buffer_address(), output.buffer_address()]
            compile_args = (
                ttnn.TensorAccessorArgs(source).get_compile_time_args()
                + ttnn.TensorAccessorArgs(output).get_compile_time_args()
            )
            descriptor = ttnn.ProgramDescriptor(
                kernels=[
                    ttnn.KernelDescriptor(
                        kernel_source=KERNEL,
                        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                        core_ranges=ranges,
                        compile_time_args=compile_args,
                        runtime_args=runtime,
                        config=ttnn.WriterConfigDescriptor(),
                    )
                ],
                cbs=[
                    ttnn.CBDescriptor(
                        total_size=64,
                        core_ranges=ranges,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=64)
                        ],
                    )
                ],
                semaphores=[],
            )
            ttnn.generic_op([source, output], descriptor)
            return output

        def snapshot(label):
            ttnn.synchronize_device(mesh)
            control_bytes = ttnn._ttnn.cluster.read_from_core(
                chip, translated.x, translated.y, layout["control_address"], 4 * layout["control_words"]
            )
            data_bytes = ttnn._ttnn.cluster.read_from_core(
                chip, translated.x, translated.y, layout["aggregate_address"], 4 * layout["aggregate_words"]
            )
            control = list(struct.unpack("<" + "I" * layout["control_words"], control_bytes))
            data = list(struct.unpack("<" + "I" * layout["aggregate_words"], data_bytes))
            markers = {}
            for field, name in (
                ("GUARANTEED_MARKER_1_H", "TRACE-FW/ZONE_START"),
                ("GUARANTEED_MARKER_2_H", "TRACE-FW/ZONE_END"),
                ("GUARANTEED_MARKER_3_H", "TRACE-KERNEL/ZONE_START"),
                ("GUARANTEED_MARKER_4_H", "TRACE-KERNEL/ZONE_END"),
            ):
                slot = layout["marker_indices"][field]
                high, low = data[slot : slot + 2]
                high_bits = layout["marker_bits"]["PROFILER_MARKER_TS_HIGH_BITS"]
                timer_bits = layout["marker_bits"]["PROFILER_MARKER_TIMER_ID_BITS"]
                markers[name] = {
                    "high": high,
                    "low": low,
                    "cycles": ((high & ((1 << high_bits) - 1)) << 32) | low,
                    "timer_id": (high >> high_bits) & ((1 << timer_bits) - 1),
                    "valid": bool(high & (1 << (high_bits + timer_bits))),
                }
            saved = {
                "label": label,
                "control_words": control,
                "aggregate_words": data,
                "named_control": {
                    name: control[index] for name, index in layout["control_indices"].items() if index < len(control)
                },
                "sentinel": data[:4],
                "markers": markers,
            }
            report["snapshots"].append(saved)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            return saved

        with runtime_audit():
            eager = forward()
        report["eager_exact"] = torch.equal(ttnn.to_torch(eager), expected)
        ttnn.deallocate(eager)
        snapshot("before_initial_drain")
        ttnn.ReadDeviceProfiler(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            output = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(output)
        report["trace_id"] = int(trace)
        report["replays"] = []
        for ordinal in (1, 2):
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            replay_snapshots.append(snapshot(f"replay_{ordinal}_before_drain"))
            ttnn.ReadDeviceProfiler(mesh)
            exact = torch.equal(ttnn.to_torch(output), expected)
            report["replays"].append({"ordinal": ordinal, "exact": exact})
        report["runtime_audit"] = True
        report["allocation_tracking"] = bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING)
        report["numerical_passed"] = report["eager_exact"] and all(item["exact"] for item in report["replays"])
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        report["csv"] = csv_evidence(
            args.profiler_dir / ".logs/profile_log_device.csv",
            replay_snapshots,
            report.get("known_participating_core", {}).get("device", -1),
        )
        report["timing_passed"] = report["csv"]["passed"]
        report["passed"] = report["numerical_passed"] and report["timing_passed"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("numerical_passed", "timing_passed", "passed", "csv")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
