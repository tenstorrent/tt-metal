// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Block residual writer, per unit (head h, local Q tile row r): the two pooled_v tile rows (each Q block's residual
// twice), zeros at the same pooled_k tiles, and the unit's 32 residual_bias rows (words 0, 1: column 0 of the two
// fp32 bias tiles, the rest zero).

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_pv = 16, cb_bias = 17, cb_zero = 18, cb_rows = 19;
constexpr uint32_t ROW_BYTES = 32 * 4;
constexpr uint32_t F32_TILE_BYTES = 4096;
}  // namespace

void kernel_main() {
    constexpr uint32_t NBPT = get_compile_time_arg_val(1);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NX = get_compile_time_arg_val(4);
    constexpr uint32_t NY = get_compile_time_arg_val(5);
    constexpr uint32_t UNITS = get_compile_time_arg_val(6);
    constexpr uint32_t PV_BFP8 = get_compile_time_arg_val(11);
    constexpr uint32_t PV_TILE_BYTES = PV_BFP8 ? 1088 : 2048;
    constexpr auto pv_args = TensorAccessorArgs<12>();
    constexpr auto pk_args = TensorAccessorArgs<pv_args.next_compile_time_args_offset()>();
    constexpr auto rb_args = TensorAccessorArgs<pk_args.next_compile_time_args_offset()>();

    constexpr uint32_t NCORES = NX * NY;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    const auto pv = TensorAccessor(pv_args, get_common_arg_val<uint32_t>(0));
    const auto pk = TensorAccessor(pk_args, get_common_arg_val<uint32_t>(1));
    const auto rb = TensorAccessor(rb_args, get_common_arg_val<uint32_t>(2));

    Noc noc;
    CircularBuffer cbpv(cb_pv), cbb(cb_bias), cbz(cb_zero), cbrows(cb_rows);
    cbz.reserve_back(1);
    cbrows.reserve_back(1);
    const uint32_t zero_base = cbz.get_write_ptr();
    const uint32_t rows_base = cbrows.get_write_ptr();
    {
        CoreLocalMem<volatile uint32_t> z(zero_base), w(rows_base);
        for (uint32_t i = 0; i < PV_TILE_BYTES / 4; ++i) {
            z[i] = 0;
        }
        for (uint32_t i = 0; i < 32 * ROW_BYTES / 4; ++i) {
            w[i] = 0;
        }
    }

    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        const uint32_t h = u / NBPT, r = u % NBPT;
        cbpv.wait_front(2 * DT);
        for (uint32_t t = 0; t < 2; ++t) {
            for (uint32_t n = 0; n < DT; ++n) {
                const uint32_t page = (h * 2 * NBPT + 2 * r + t) * DT + n;
                noc.async_write(
                    cbpv, pv, PV_TILE_BYTES, {.offset_bytes = (t * DT + n) * PV_TILE_BYTES}, {.page_id = page});
                noc.async_write(CoreLocalMem<uint32_t>(zero_base), pk, PV_TILE_BYTES, {}, {.page_id = page});
            }
        }

        cbb.wait_front(2);
        invalidate_l1_cache();
        const uint32_t bbase = cbb.get_read_ptr();
        CoreLocalMem<volatile uint32_t> b0(bbase), b1(bbase + F32_TILE_BYTES), w(rows_base);
        for (uint32_t i = 0; i < 32; ++i) {
            const uint32_t e = ((i >> 4) << 1) * 256 + (i & 15) * 16;  // element (i, 0) of a 32x32 fp32 tile
            w[i * 32] = b0[e];
            w[i * 32 + 1] = b1[e];
            noc.async_write(
                CoreLocalMem<uint32_t>(rows_base + i * ROW_BYTES),
                rb,
                ROW_BYTES,
                {},
                {.page_id = h * NBPT * 32 + r * 32 + i});
        }
        noc.async_write_barrier();
        cbpv.pop_front(2 * DT);
        cbb.pop_front(2);
    }
}
