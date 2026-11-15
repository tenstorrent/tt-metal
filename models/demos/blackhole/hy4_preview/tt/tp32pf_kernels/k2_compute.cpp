// SPDX-License-Identifier: Apache-2.0
// tp32pf K2 (v1 W2) TRISC: per unit, per pair of M-tiles: 48 passes q = b*6+s: dest[2 x 4] = act (2 x 2) @ w[q] (2 x
// 4), pack 8 tiles [m][n] to cb_o. CT: 0 cb_a, 1 cb_w, 2 cb_o, 3 cb_meta. RT: 0 me
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_w, cb_o);
    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t nu = plan[0];
    matmul_block_init(cb_a, cb_w, false, 4, 2, 2);
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t mtc = plan[3 + 3 * u];
        cb_wait_front(cb_w, 384);
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
            cb_wait_front(cb_a, 4);
            for (uint32_t q = 0; q < 48; ++q) {
                tile_regs_acquire();
                matmul_block(cb_a, cb_w, 0, q * 8, 0, false, 4, 2, 2);
                matmul_block(cb_a, cb_w, 1, q * 8 + 4, 0, false, 4, 2, 2);
                tile_regs_commit();
                cb_reserve_back(cb_o, 8);
                tile_regs_wait();
                for (uint32_t i = 0; i < 8; ++i) {
                    pack_tile(i, cb_o, i);
                }
                tile_regs_release();
                cb_push_back(cb_o, 8);
            }
            cb_pop_front(cb_a, 4);
        }
        cb_pop_front(cb_w, 384);
    }
    cb_pop_front(cb_meta, 1);
}
