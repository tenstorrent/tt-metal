// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream line RS of fp32 partials (tt/fabric_stream.FabricStreamRSL), REDUCER TRISC: y = ((t0 + t1) +
// (t2 + t3)) per tile in fp32, t_c from CB_T + c (column position c; fp32 tiles unpacked straight to DST: exact), the
// adds on the SFPU (add_binary_tile, fp32), packed fp32 to CB_Y. The same association on every chip.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

#if !defined(MK_PHASED)  // standalone; a composite defines it per phase namespace
constexpr uint32_t PH_CT0 = 0;
#endif

constexpr uint32_t TILES = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t CB_T = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_Y = get_compile_time_arg_val(PH_CT0 + 3);

void kernel_main() {
    // full start-up in a composite too (rs8probe3, 2026-10-25): on a PMK tail core this reducer is the FIRST compute
    // phase, and a reconfigure-only start ran on the math / pack DEST sync and packer state the PREVIOUS program left
    // (right after program A, wrong after the op MoE: y blocks 6-7 at 0.2 vs fp32). Every other PMK compute body
    // starts the same way; an mk owner composite maps this to mkc::hw_startup (first phase only).
    compute_kernel_hw_startup(CB_T, CB_T + 1, CB_Y);
    for (uint32_t b = 0; b < TILES / NB; b++) {
        for (uint32_t c = 0; c < 4; c++) {
            cb_wait_front(CB_T + c, NB);
        }
        cb_reserve_back(CB_Y, NB);
        for (uint32_t t = 0; t < NB; t++) {
            tile_regs_acquire();
            for (uint32_t c = 0; c < 4; c++) {
                copy_init(CB_T + c);
                copy_tile(CB_T + c, t, c);
            }
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(2, 3, 2);
            add_binary_tile(0, 2, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_Y);
            tile_regs_release();
        }
        cb_push_back(CB_Y, NB);
        for (uint32_t c = 0; c < 4; c++) {
            cb_pop_front(CB_T + c, NB);
        }
    }
}
