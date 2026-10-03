// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC projection v2, writer.  (1) builds the constant tiles of the core: ONES (ones in column NCOL) and 5 selection
// tiles SEL_i[tl, 4*tl + i] = 1 (i < 4), SEL_all[tl, 4*tl + i] = 1 (all i);  (2) writes the TG valid rows (row tl =
// token tok0+tl) of the partial tile into rows [row0, row0 + TG) of its packed partial page `pg` (two face-half
// pieces).

#include <stdint.h>
#include "tools/profiler/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_ones = get_compile_time_arg_val(0);
    constexpr uint32_t cb_sel = get_compile_time_arg_val(1);
    constexpr uint32_t cb_p = get_compile_time_arg_val(2);
    constexpr uint32_t NCOL = get_compile_time_arg_val(3);
    constexpr uint32_t TG = get_compile_time_arg_val(4);
    constexpr auto p_args = TensorAccessorArgs<5>();
    constexpr uint32_t TILE = 4096;

    const uint32_t pg = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB cbo(cb_ones), cbs(cb_sel), cbp(cb_p);

    DeviceZoneScopedN("PW_ALL");
    cbo.reserve_back(1);
    cbs.reserve_back(5);
    noc.async_write_zeros(cbo, TILE, {.offset_bytes = 0});
    noc.async_write_zeros(cbs, 5 * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t face = (r >= 16 ? 2 : 0) + (NCOL >= 16 ? 1 : 0);
            o[face * 256 + (r % 16) * 16 + (NCOL % 16)] = 0x3F800000u;
        }
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t tl = 0; tl < TG; ++tl) {
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t c = 4 * tl + i;
                const uint32_t face = (tl >= 16 ? 2 : 0) + (c >= 16 ? 1 : 0);
                const uint32_t w = face * 256 + (tl % 16) * 16 + (c % 16);
                s[i * 1024 + w] = 0x3F800000u;
                s[4 * 1024 + w] = 0x3F800000u;
            }
        }
    }
    cbo.push_back(1);
    cbs.push_back(5);
    {
        DeviceZoneScopedN("PW_built");
    }

    cbp.wait_front(1);
    {
        DeviceZoneScopedN("PW_gotP");
    }
    const uint32_t dst_off = ((row0 >> 4) << 11) + ((row0 & 15) << 6);
    noc.async_write(cbp, p_acc, TG * 64, {.offset_bytes = 0}, {.page_id = pg, .offset_bytes = dst_off});
    noc.async_write(cbp, p_acc, TG * 64, {.offset_bytes = 1024}, {.page_id = pg, .offset_bytes = dst_off + 1024});
    noc.async_write_barrier();
    cbp.pop_front(1);
}
