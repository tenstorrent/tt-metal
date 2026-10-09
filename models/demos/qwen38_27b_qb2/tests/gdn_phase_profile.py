# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only kernel timing zones; no production source mutation."""


def instrument(filename, source):
    ranges = {
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
    }
    if filename not in ranges:
        raise ValueError("Unsupported kernel for phase profiling")
    for label, first, last in ranges[filename]:
        if source.count(first) != 1 or source.count(last) != 1:
            raise ValueError("Kernel profiling anchor changed: " + label)
        begin = source.index(first)
        end = source.index(last, begin) + len(last)
        source = (
            source[:begin] + '{\nDeviceZoneScopedN("' + label + '");\n' + source[begin:end] + "\n}\n" + source[end:]
        )
    return '#include "tools/profiler/kernel_profiler.hpp"\n' + source
