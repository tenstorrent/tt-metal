// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool writer: each unit's DT output tiles go to (head h, tile row g) of q_bar [1, H, NBP, D] for Q, and of the
// stacked kv_bar [1, 2H, NBP, D] for K (heads [0, H)) and V (heads [H, 2H)).
//
// Compile-time args: 0 H, 1 RT, 2 DT, 3 NBPT, 4 NX, 5 NY, 6 UNITS, then TensorAccessorArgs of q_bar and kv_bar.
// Common runtime args: 0 q_bar address, 1 kv_bar address.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "block_pool_units.hpp"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t TILE_BYTES = 2048;
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(2);
    constexpr uint32_t NBPT = get_compile_time_arg_val(3);
    constexpr uint32_t NX = get_compile_time_arg_val(4);
    constexpr uint32_t NY = get_compile_time_arg_val(5);
    constexpr uint32_t UNITS = get_compile_time_arg_val(6);
    constexpr auto q_args = TensorAccessorArgs<7>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    using Layout = block_pool::Layout<H, RT, NBPT>;
    static_assert(Layout::UNITS == UNITS, "host unit count must match the kernel layout");
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    uint32_t unit_start, unit_count;
    Layout::range(ci, NX * NY, unit_start, unit_count);
    if (unit_count == 0) {
        return;
    }
    const auto q_bar = TensorAccessor(q_args, get_common_arg_val<uint32_t>(0));
    const auto kv_bar = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(1));

    Noc noc;
    CircularBuffer cbo(cb_out);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const block_pool::Unit un = Layout::decode(u);
        cbo.wait_front(DT);
        const uint32_t head = un.t == 2 ? H + un.h : un.h;
        const uint32_t first = (head * NBPT + un.g) * DT;
        for (uint32_t n = 0; n < DT; ++n) {
            if (un.t == 0) {
                noc.async_write(cbo, q_bar, TILE_BYTES, {.offset_bytes = n * TILE_BYTES}, {.page_id = first + n});
            } else {
                noc.async_write(cbo, kv_bar, TILE_BYTES, {.offset_bytes = n * TILE_BYTES}, {.page_id = first + n});
            }
        }
        noc.async_write_barrier();
        cbo.pop_front(DT);
    }
}
