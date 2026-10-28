// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the decode-shaped routed expert FFN of the DeepSeek-V4.1 c1 MoE (tt/v41/moe_c1.py C1FfnMatvec;
// kernels/c1_ffn_reader.cpp has the dataflow): per selected local expert l, this core's output tiles go to tile
// row l (the expert's region) of ``y`` [EPC * 32, hidden] at the owned columns. NoC 1, unicast writes only.
//
// compile_time_args = [cb_scr, cb_out, EPC, E, NT_D, OUT_TILE, TensorAccessorArgs(y)...]
// runtime args      = [y_addr, nd0, nd1]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

constexpr uint32_t NONE = 0xFFFFFFFFu;

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t nd[2] = {get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2)};

    constexpr uint32_t cb_scr = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t EPC = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t NT_D = get_compile_time_arg_val(4);
    constexpr uint32_t OUT_TILE = get_compile_time_arg_val(5);
    constexpr auto y_args = TensorAccessorArgs<6>();

    const auto y = TensorAccessor(y_args, y_addr, OUT_TILE);
    DataflowBuffer scr(cb_scr), out(cb_out);
    const uint32_t nd_count = (nd[0] == NONE) ? 0 : ((nd[1] == NONE) ? 1 : 2);

    scr.wait_front(1);
    const uint32_t scr_l1 = (scr.get_read_ptr() + 63) & ~63u;
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_l1);
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_l1 + E * 4);
    if (nd_count == 0) {
        return;
    }
    for (uint32_t l = 0; l < EPC; ++l) {
        if (cnt[table[l]] == 0) {
            continue;
        }
        out.wait_front(nd_count);
        const uint32_t src = out.get_read_ptr();
        for (uint32_t j = 0; j < nd_count; ++j) {
            noc_async_write(src + j * OUT_TILE, y.get_noc_addr(l * NT_D + nd[j]), OUT_TILE);
        }
        noc_async_write_barrier();
        out.pop_front(nd_count);
    }
}
