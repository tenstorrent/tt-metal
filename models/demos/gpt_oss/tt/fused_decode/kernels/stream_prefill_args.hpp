// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// StreamPrefillLinear's per-core arguments (tt/stream_prefill_linear.py): instead of per-core runtime args, a reader /
// writer reads its core's row of a DRAM argument table (row = logical y * GX + logical x) into a one-page CB once at
// kernel start. Per-core runtime args cost the host ~0.5 us per core and kernel on every call (generic_op rewrites them
// all on a program-cache hit; 0.2-0.5 ms per call on 120 cores), so the program carries common args only.
//
// compile-time args at CT: [cb_args, GX, row_bytes, accessor (table)]

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

template <uint32_t CT>
inline volatile tt_l1_ptr uint32_t* read_core_args(uint32_t table_addr) {
    constexpr uint32_t cb_args = get_compile_time_arg_val(CT);
    constexpr uint32_t GX = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(CT + 2);
    constexpr auto table_args = TensorAccessorArgs<CT + 3>();
    const auto table = TensorAccessor(table_args, table_addr, row_bytes);
    cb_reserve_back(cb_args, 1);  // (this kernel's own page: never pushed)
    const uint32_t l1 = get_write_ptr(cb_args);
    noc_async_read(table.get_noc_addr(get_absolute_logical_y() * GX + get_absolute_logical_x()), l1, row_bytes);
    noc_async_read_barrier();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}
