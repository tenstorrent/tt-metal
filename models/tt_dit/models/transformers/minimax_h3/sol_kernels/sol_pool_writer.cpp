// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "sol_pool_units.hpp"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t TILE_BYTES = 2048;
#ifdef SOL_POOL_V_BFP8
constexpr uint32_t cb_out_v = 17;
constexpr uint32_t V_TILE_BYTES = 1088;
#endif
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(2);
    constexpr uint32_t NBPT = get_compile_time_arg_val(3);
    constexpr auto o0_args = TensorAccessorArgs<8>();
    constexpr auto o1_args = TensorAccessorArgs<o0_args.next_compile_time_args_offset()>();
    constexpr auto o2_args = TensorAccessorArgs<o1_args.next_compile_time_args_offset()>();

    constexpr uint32_t NX = get_compile_time_arg_val(5);
    constexpr uint32_t NY = get_compile_time_arg_val(6);
    constexpr uint32_t UNITS = get_compile_time_arg_val(7);
    constexpr uint32_t NCORES = NX * NY;
    using Layout = sol_pool::Layout<H, RT, NBPT>;
    static_assert(Layout::UNITS == UNITS, "host unit count must match the kernel layout");
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    uint32_t unit_start, unit_count;
    Layout::range(ci, NCORES, unit_start, unit_count);
    if (unit_count == 0) {
        return;
    }
    const auto o0 = TensorAccessor(o0_args, get_common_arg_val<uint32_t>(0));
    const auto o1 = TensorAccessor(o1_args, get_common_arg_val<uint32_t>(1));
    const auto o2 = TensorAccessor(o2_args, get_common_arg_val<uint32_t>(2));

    Noc noc;
    CircularBuffer cbo(cb_out);
#ifdef SOL_POOL_V_BFP8
    CircularBuffer cbov(cb_out_v);
#endif
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const sol_pool::Unit un = Layout::decode(u);
        const uint32_t t = un.t, h = un.h, g = un.g;
#ifdef SOL_POOL_V_BFP8
        CircularBuffer& cbu = t == 2 ? cbov : cbo;
        const uint32_t tile_bytes = t == 2 ? V_TILE_BYTES : TILE_BYTES;
#else
        CircularBuffer& cbu = cbo;
        constexpr uint32_t tile_bytes = TILE_BYTES;
#endif
        cbu.wait_front(DT);
        auto store = [&](auto& acc) {
            for (uint32_t n = 0; n < DT; ++n) {
                noc.async_write(
                    cbu, acc, tile_bytes, {.offset_bytes = n * tile_bytes}, {.page_id = (h * NBPT + g) * DT + n});
            }
        };
        if (t == 0) {
            store(o0);
        } else if (t == 1) {
            store(o1);
        } else {
            store(o2);
        }
        noc.async_write_barrier();
        cbu.pop_front(DT);
    }
}
