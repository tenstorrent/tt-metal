// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream RS with the lane reduce (tt/fabric_stream.FabricStreamRSR), lane TRISC: y = own + landed per
// tile EXACTLY in fp32: both bf16 tiles unpacked straight to DST as fp32 (CB_A / CB_B UnpackToDestFp32: exact), added
// on the SFPU in fp32 (add_binary_tile), packed fp32 into CB_Y for rs_send.cpp. (The FPU elementwise add rounds its
// result to a ~10-bit mantissa: max |diff| 2^-9 vs the fp32 sum on the d0 pair, 45d0d9dc6b2.) A sum of two values does
// not depend on the order; the fixed chip order matters from 3 terms on.
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
constexpr uint32_t CB_A = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t CB_B = get_compile_time_arg_val(PH_CT0 + 3);
constexpr uint32_t CB_Y = get_compile_time_arg_val(PH_CT0 + 4);
static_assert(NB % 2 == 0, "two tile pairs per DST round (4 fp32 DST tiles)");

void kernel_main() {
    // full start-up in a composite too (rs8probe3, 2026-10-25): on a PMK tail core this reducer is the FIRST compute
    // phase, and a reconfigure-only start ran on the math / pack DEST sync and packer state the PREVIOUS program left
    // (right after program A, wrong after the op MoE: y blocks 6-7 at 0.2 vs fp32). Every other PMK compute body
    // starts the same way; an mk owner composite maps this to mkc::hw_startup (first phase only).
    compute_kernel_hw_startup(CB_A, CB_B, CB_Y);
    for (uint32_t b = 0; b < TILES / NB; b++) {
        cb_wait_front(CB_A, NB);
        cb_wait_front(CB_B, NB);
        cb_reserve_back(CB_Y, NB);
        for (uint32_t t0 = 0; t0 < NB; t0 += 2) {
            tile_regs_acquire();
            copy_init(CB_A);
            copy_tile(CB_A, t0, 0);
            copy_tile(CB_A, t0 + 1, 2);
            copy_init(CB_B);
            copy_tile(CB_B, t0, 1);
            copy_tile(CB_B, t0 + 1, 3);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            add_binary_tile(2, 3, 2);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_Y);
            pack_tile(2, CB_Y);
            tile_regs_release();
        }
        cb_push_back(CB_Y, NB);
        cb_pop_front(CB_A, NB);
        cb_pop_front(CB_B, NB);
    }
}
