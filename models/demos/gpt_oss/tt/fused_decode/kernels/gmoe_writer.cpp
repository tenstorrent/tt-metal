// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Grouped MoE writer (NCRISC) of both grouped matmuls (dflash/grouped_moe.py): per unit (expert e, column tile j of
// NU), the expert's slot tiles in slot order from cb_out to tile (base[e] + slot) * NU + j of the output
// (gate|up: A_g [slots x 32, 23 x 32]; down: Y_g [slots x 32, hidden]).
//
// common runtime args: [out]; compile-time args: [cb_meta_w, cb_out, NU, accessor (out)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t NU = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    constexpr uint32_t E = 32;
    const uint32_t tile = get_tile_size(cb_out);
    const auto s_out = TensorAccessor(out_args, out_addr, tile);

    cb_wait_front(cb_meta, 1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
    const uint32_t u0 = meta[0], u1 = meta[1];
    for (uint32_t u = u0; u < u1; ++u) {
        const uint32_t e = u / NU, j = u % NU, m = meta[2 + e], base = meta[2 + E + e];
#ifdef GMOE_CAP
        if (m > GMOE_CAP) {
            // the down's big expert (gmoe_down_*.cpp): per group of GMOE_CAP slots, each of this core's units of it
            const uint32_t ue1 = u1 < (e + 1) * NU ? u1 : (e + 1) * NU;
            for (uint32_t g = 0; g < m; g += GMOE_CAP) {
                const uint32_t n = m - g < GMOE_CAP ? m - g : GMOE_CAP;
                for (uint32_t uu = u; uu < ue1; ++uu) {
                    for (uint32_t s = g; s < g + n; ++s) {
                        cb_wait_front(cb_out, 1);
                        noc_async_write(get_read_ptr(cb_out), s_out.get_noc_addr((base + s) * NU + uu % NU), tile);
                        noc_async_writes_flushed();
                        cb_pop_front(cb_out, 1);
                    }
                }
            }
            u = ue1 - 1;
            continue;
        }
#endif
        for (uint32_t s = 0; s < m; ++s) {
            cb_wait_front(cb_out, 1);
            noc_async_write(get_read_ptr(cb_out), s_out.get_noc_addr((base + s) * NU + j), tile);
            noc_async_writes_flushed();
            cb_pop_front(cb_out, 1);
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_meta, 1);
}
