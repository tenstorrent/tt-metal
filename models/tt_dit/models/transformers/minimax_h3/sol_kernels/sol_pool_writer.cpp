// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn block-pooling pre-pass, writer (BRISC): stores each unit's DT output tiles at (head h, tile row g)
// of the unit's output tensor [1, H, NBP, D].
//
// Compile-time args: 0 H, 1 RT, 2 DT, 3 NBPT, 4 R, then TensorAccessorArgs of out0, out1, out2.
// Common runtime args: 0..2 buffer addresses of out0..out2. Per-core runtime args: none (unit range from the core's
// logical coordinates).

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t TILE_BYTES = 2048;
}  // namespace

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t DT = get_compile_time_arg_val(2);
    constexpr uint32_t NBPT = get_compile_time_arg_val(3);
    constexpr auto o0_args = TensorAccessorArgs<8>();
    constexpr auto o1_args = TensorAccessorArgs<o0_args.next_compile_time_args_offset()>();
    constexpr auto o2_args = TensorAccessorArgs<o1_args.next_compile_time_args_offset()>();

    constexpr uint32_t NX = get_compile_time_arg_val(5);
    constexpr uint32_t NY = get_compile_time_arg_val(6);
    constexpr uint32_t UNITS = get_compile_time_arg_val(7);
    // Unit range of this core from its logical coordinates (no per-core runtime args): units [ci*U/N, (ci+1)*U/N).
    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto o0 = TensorAccessor(o0_args, get_common_arg_val<uint32_t>(0));
    const auto o1 = TensorAccessor(o1_args, get_common_arg_val<uint32_t>(1));
    const auto o2 = TensorAccessor(o2_args, get_common_arg_val<uint32_t>(2));

    Noc noc;
    CircularBuffer cbo(cb_out);
    constexpr uint32_t units_per_tensor = H * NBPT;
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t t = u / units_per_tensor;
        const uint32_t h = (u % units_per_tensor) / NBPT;
        const uint32_t g = u % NBPT;
        cbo.wait_front(DT);
        auto store = [&](auto& acc) {
            for (uint32_t n = 0; n < DT; ++n) {
                noc.async_write(
                    cbo, acc, TILE_BYTES, {.offset_bytes = n * TILE_BYTES}, {.page_id = (h * NBPT + g) * DT + n});
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
        cbo.pop_front(DT);
    }
}
