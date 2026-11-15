// SPDX-License-Identifier: Apache-2.0
// tp32pf K2b TRISC. Phase 1 per unit per pair of M-tiles: for each bank block b: dest[2 x 4] = act (2x2) @ w[b] (2x4);
// pack_untilize to row-major cb_rm (2 tile rows of 32 rows x 1024 cols = 64 pages of 2 KB, column block b of 4 tiles).
// Phase 2 (owners, me < NOWN): the 8 slot buffers [k][32 tokens][1024] of cb_slot (filled by the senders' NoC writes,
// pushed by the writer) are summed per row (pages are added as plain 1024-element chunks: same layout on both
// operands), tilized, handed to the writer (cb_til, 32 tiles). CT: 0 cb_a, 1 cb_w, 2 cb_rm, 3 cb_meta, 4 cb_slot, 5
// cb_sum, 6 cb_til, 7 NOWN. RT: 0 me
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_rm = get_compile_time_arg_val(2);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(3);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sum = get_compile_time_arg_val(5);
    constexpr uint32_t cb_til = get_compile_time_arg_val(6);
    constexpr uint32_t NOWN = get_compile_time_arg_val(7);
    constexpr uint32_t DBG = get_compile_time_arg_val(8);
    constexpr uint32_t PERF = get_compile_time_arg_val(9);
    const uint32_t me = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_rm);
    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t nu = plan[0];
    matmul_block_init(cb_a, cb_w, false, 4, 2, 2);
    pack_untilize_dest_init<4, 32>(cb_rm);
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t mtc = plan[3 + 3 * u];
        cb_wait_front(cb_w, 64);
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
            cb_wait_front(cb_a, 4);
            cb_reserve_back(cb_rm, 64);
            for (uint32_t b = 0; b < 8; ++b) {
                tile_regs_acquire();
                if (!(PERF & 2)) {
                    matmul_block(cb_a, cb_w, 0, b * 8, 0, false, 4, 2, 2);
                    matmul_block(cb_a, cb_w, 1, b * 8 + 4, 0, false, 4, 2, 2);
                }
                tile_regs_commit();
                tile_regs_wait();
                if (!(PERF & 1)) {
                    pack_untilize_dest<4, 32>(cb_rm, 2, b);
                }
                tile_regs_release();
            }
            cb_push_back(cb_rm, 64);
            cb_pop_front(cb_a, 4);
        }
        cb_pop_front(cb_w, 64);
    }
    pack_untilize_uninit(cb_rm);
    cb_pop_front(cb_meta, 1);
    if (me < NOWN && (DBG == 0 || DBG == 3)) {
        cb_wait_front(cb_slot, 256);
        reconfig_data_format_srca(cb_w, cb_slot);
        for (uint32_t c = 0; c < 32; ++c) {
            tile_regs_acquire();
            copy_init(cb_slot);
            copy_tile(cb_slot, c, 0);
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_slot);
            for (uint32_t k = 1; k < 8; ++k) {
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_slot, k * 32 + c, 0);
            }
            tile_regs_commit();
            if (DBG == 3) {
                cb_reserve_back(cb_til, 1);
                tile_regs_wait();
                pack_tile(0, cb_til);
                tile_regs_release();
                cb_push_back(cb_til, 1);
            } else {
                cb_reserve_back(cb_sum, 1);
                tile_regs_wait();
                pack_tile(0, cb_sum);
                tile_regs_release();
                cb_push_back(cb_sum, 1);
            }
        }
        cb_pop_front(cb_slot, 256);
        if (DBG == 3) {
            return;
        }
        cb_wait_front(cb_sum, 32);
        cb_reserve_back(cb_til, 32);
        tilize_init(cb_sum, 32, cb_til);
        tilize_block(cb_sum, 32, cb_til);
        tilize_uninit(cb_sum, cb_til);
        cb_push_back(cb_til, 32);
        cb_pop_front(cb_sum, 32);
    }
}
