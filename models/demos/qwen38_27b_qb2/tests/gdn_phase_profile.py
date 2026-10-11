# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only kernel timing zones; no production source mutation."""

import re

RANGES = {
    "reader.cpp": (
        (
            "GDN_READER_CB_RESERVE",
            "        cb_reserve_back(0, 4);",
            "        cb_reserve_back(5, 4 * value_columns);",
        ),
        (
            "GDN_READER_DRAM_ISSUE_AND_WAIT",
            "        noc_async_read_page(head / qk_head_repeat, q, scratch);",
            "        noc_async_read_barrier();",
        ),
        (
            "GDN_READER_L1_PREPARE",
            "        const auto* values =",
            "        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4))[0] = values[385];",
        ),
    ),
    "writer.cpp": (
        (
            "GDN_WRITER_STATE_CB_WAIT",
            "        cb_wait_front(7, 4 * value_columns);",
            "        cb_wait_front(7, 4 * value_columns);",
        ),
        (
            "GDN_WRITER_DRAM_ISSUE_AND_OUTPUT_WAIT",
            "        const uint32_t state_l1 = get_read_ptr(7);",
            "        cb_wait_front(8, value_columns);",
        ),
        ("GDN_WRITER_OUTPUT_AND_BARRIER", "        const auto* tiles =", "        noc_async_write_barrier();"),
    ),
    "compute.cpp": (
        ("GDN_COMPUTE_INPUT_WAIT", "        cb_wait_front(0, 4);", "        cb_wait_front(5, 4 * value_columns);"),
        (
            "GDN_COMPUTE_DELTA",
            "        cb_reserve_back(6, value_columns);",
            "        cb_push_back(6, value_columns);",
        ),
        (
            "GDN_COMPUTE_STATE_UPDATE",
            "        cb_wait_front(6, value_columns);",
            "        cb_push_back(7, 4 * value_columns);",
        ),
        (
            "GDN_COMPUTE_OUTPUT",
            "        cb_wait_front(7, 4 * value_columns);",
            "        cb_push_back(8, value_columns);",
        ),
    ),
    "compute_resident.cpp": (
        ("GDN_RESIDENT_INPUT_WAIT", "        cb_wait_front(0, 4);", "        cb_wait_front(5, 4);"),
        ("GDN_RESIDENT_OUTPUT_AND_DEST_WAIT", "        cb_reserve_back(7, 4);", "        tile_regs_acquire();"),
        (
            "GDN_RESIDENT_DELTA",
            "        // delta = beta *",
            "        multiply(0, 1, 0);",
        ),
        (
            "GDN_RESIDENT_STATE_UPDATE",
            "        // The reduction's row zero holds delta.",
            "            add(4 + kr, 1, 4 + kr);\n        }",
        ),
        (
            "GDN_RESIDENT_OUTPUT",
            "        // Output = q^T S_new.",
            "        tile_regs_commit();",
        ),
        ("GDN_RESIDENT_PACK", "        tile_regs_wait();", "        cb_push_back(8, 1);"),
    ),
    "epilogue/reader.cpp": (
        (
            "GDN_EP_READER_WEIGHT",
            "    cb_reserve_back(dfb::weight, 4);",
            "    cb_push_back(dfb::weight, 4);",
        ),
        (
            "GDN_EP_READER_PADDING",
            "    if constexpr (input_padding != 1) {",
            "            g_init[i] = input_padding == 2 ? 0x7fc07fc0 : 0;\n        }\n    }",
        ),
        (
            "GDN_EP_READER_CB_RESERVE",
            "        cb_reserve_back(dfb::x, 4);",
            "        cb_reserve_back(dfb::gate, 4);",
        ),
        (
            "GDN_EP_READER_DMA",
            "        noc_async_read(raw.get_noc_addr(head), scratch, 512);",
            "        noc_async_read_barrier();",
        ),
        (
            "GDN_EP_READER_FORMAT",
            "        if constexpr (compact) {",
            "        cb_push_back(dfb::gate, 4);",
        ),
    ),
    "epilogue/compute.cpp": (
        ("GDN_EP_INPUT_WAIT", "        x.wait_front(Vt);", "        gate.wait_front(Vt);"),
        ("GDN_EP_MEAN_SQUARE", "        square(Vt, tmp);", "        stats.wait_front(1);"),
        ("GDN_EP_NORMALIZE", "        inverse_rms(inv);", "        stats.pop_front(1);"),
        (
            "GDN_EP_WEIGHT_AND_GATE",
            "        apply_weight(Vt, tmp);",
            "        rounded.wait_front(Vt);",
        ),
        (
            "GDN_EP_OUTPUT",
            "        if constexpr (multiply_z) {",
            "        tmp.pop_front(Vt);\n        norm.pop_front(Vt);",
        ),
    ),
    "epilogue/writer.cpp": (
        ("GDN_EP_WRITER_WAIT", "        cb_wait_front(7, 4);", "        cb_wait_front(7, 4);"),
        (
            "GDN_EP_WRITER_DMA",
            "        for (uint32_t tile = 0; tile < 4; ++tile) {",
            "        noc_async_write_barrier();",
        ),
    ),
}


def instrument(filename, source):
    if filename not in RANGES:
        raise ValueError("Unsupported kernel for phase profiling")
    for label, first, last in RANGES[filename]:
        starts = list(re.finditer("^" + re.escape(first), source, re.MULTILINE))
        if len(starts) != 1:
            raise ValueError("Kernel profiling anchor changed: " + label)
        begin = starts[0].start()
        ends = list(re.finditer("^" + re.escape(last), source[begin:], re.MULTILINE))
        if len(ends) != 1:
            raise ValueError("Kernel profiling anchor changed: " + label)
        end = begin + ends[0].end()
        source = (
            source[:begin] + '{\nDeviceZoneScopedN("' + label + '");\n' + source[begin:end] + "\n}\n" + source[end:]
        )
    return '#include "tools/profiler/kernel_profiler.hpp"\n' + source


def required_markers(*, pipeline=False):
    names = (
        (
            "reader.cpp",
            "writer.cpp",
            "compute_resident.cpp",
            "epilogue/reader.cpp",
            "epilogue/compute.cpp",
            "epilogue/writer.cpp",
        )
        if pipeline
        else ("reader.cpp", "writer.cpp", "compute.cpp")
    )
    return {label for name in names for label, _, _ in RANGES[name]}


def validate_coverage(report, *, pipeline=False):
    if (
        report.get("state") != "completed"
        or report.get("passed") is not True
        or report.get("cleanup_completed") is not True
    ):
        raise ValueError("Phase profile test lacks passing result and clean shutdown")
    expected = (
        [(b, padding, profiled) for b in (16, 32) for padding in ("zero", "skip") for profiled in (False, True)]
        if pipeline
        else [(b, n, profiled) for b in (32, 16) for n in (1, 2) for profiled in (False, True)]
    )
    key = "padding" if pipeline else "buffers"
    cases = report.get("cases", [])
    if [(r.get("batch"), r.get(key), r.get("profiled")) for r in cases] != expected or report.get("kernel_calls") != (
        48 if pipeline else 24
    ):
        raise ValueError("Incomplete phase profile coverage")
    if not pipeline:
        return
    if len(set(report.get("device_ids", []))) != 4:
        raise ValueError("Pipeline profile requires four physical ranks")
    controls = {}
    for case in cases:
        checks = case.get("checks", [])
        if [row.get("rank") for row in checks] != list(range(4)) or any(
            row.get("state", {}).get("passed") is not True
            or row.get("raw_output", {}).get("passed") is not True
            or row.get("finite_output") is not True
            for row in checks
        ):
            raise ValueError("Pipeline profile lacks passing per-rank correctness")
        hashes = case.get("hashes", {})
        if set(hashes) != {"state", "raw_output", "output"} or any(
            len(values) != 4
            or any(not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value) for value in values)
            for values in hashes.values()
        ):
            raise ValueError("Pipeline profile lacks complete output hashes")
        if controls.setdefault(case["batch"], hashes) != hashes:
            raise ValueError("Padding or timing zones changed state or output")
