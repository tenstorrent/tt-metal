// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for MoeCombine, accumulator-through-L1 form with EPW experts per dest window (EPW = 1 is the original
// one-expert window). Per window: the running sum is copied back into dest 0 (from the fp32 CB cb_acc; the very
// first product is the running sum itself), the EPW products d_e * w_e (FPU column broadcast) land in dest 1..EPW,
// and the binary SFPU adds them into dest 0 after the last FPU write of the window (an FPU write to a dest tile an
// earlier SFPU op of the window read corrupts it); the sum is packed back to cb_acc, the last window to the bf16
// output. EPW > 1 needs the full dest sync (fp32 dest holds 8 tiles in full sync, 4 in half sync: the host sets
// dst_full_sync_en with it).
// compile-time args: [cb_d, cb_w, cb_out, E, cb_acc, EPW]; runtime args: [column count]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_D = get_compile_time_arg_val(0);
constexpr uint32_t CB_W = get_compile_time_arg_val(1);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
constexpr uint32_t E = get_compile_time_arg_val(3);
constexpr uint32_t CB_ACC = get_compile_time_arg_val(4);
constexpr uint32_t EPW = get_compile_time_arg_val(5);
static_assert(E % EPW == 0 && EPW >= 1 && EPW <= 7, "experts per window must divide E and fit dest 1..7");

void kernel_main() {
    compute_kernel_hw_startup(CB_D, CB_W, CB_OUT);
    const uint32_t count = get_arg_val<uint32_t>(0);
    CircularBuffer cbd(CB_D), cbw(CB_W), cbo(CB_OUT), cba(CB_ACC);
    constexpr uint32_t windows = E / EPW;
    for (uint32_t c = 0; c < count; ++c) {
        for (uint32_t wi = 0; wi < windows; ++wi) {
            const bool first = (wi == 0);
            const bool last = (wi + 1 == windows);
            cbd.wait_front(EPW);
            cbw.wait_front(EPW);
            tile_regs_acquire();
            if (!first) {
                cba.wait_front(1);
                reconfig_data_format_srca(CB_ACC);
                copy_init(CB_ACC);
                copy_tile(CB_ACC, 0, 0);
                cba.pop_front(1);
            }
            reconfig_data_format(CB_D, CB_W);
            mul_bcast_cols_init(CB_D, CB_W);
            for (uint32_t j = 0; j < EPW; ++j) {
                // the very first product is the running sum itself (dest 0); every other product goes to 1..EPW
                mul_tiles_bcast_cols(CB_D, CB_W, j, j, (first && j == 0) ? 0 : 1 + j);
            }
            add_binary_tile_init();
            for (uint32_t j = (first ? 1 : 0); j < EPW; ++j) {
                add_binary_tile(0, 1 + j, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            if (last) {
                pack_reconfig_data_format(CB_OUT);
                cbo.reserve_back(1);
                pack_tile(0, CB_OUT, 0);
                cbo.push_back(1);
            } else {
                pack_reconfig_data_format(CB_ACC);
                cba.reserve_back(1);
                pack_tile(0, CB_ACC, 0);
                cba.push_back(1);
            }
            tile_regs_release();
            cbd.pop_front(EPW);
            cbw.pop_front(EPW);
        }
    }
}
