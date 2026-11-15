// SPDX-License-Identifier: Apache-2.0
// tp32pf K1a TRISC: per unit (expert resident in cb_w, whole bank runs [b][k][j]) per pair of M-tiles: dest[2 x 4] +=
// x_b (2x24) @ w_b (24x4) over the 8 banks, then act = silu(min(g, L)) * clamp(u, -L, L) (g = tiles m*4+0/1, u =
// m*4+2/3) -> 4 act tiles [m][c] to cb_o. CT: 0 cb_x, 1 cb_w, 2 cb_o, 3 cb_meta, 4 +L bits, 5 -L bits, 6 -BIG bits. RT:
// 0 me
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_binary_sfpu.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(3);
    constexpr uint32_t lim_bits = get_compile_time_arg_val(4);
    constexpr uint32_t nlim_bits = get_compile_time_arg_val(5);
    constexpr uint32_t nbig_bits = get_compile_time_arg_val(6);
    const uint32_t me = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_o);
    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_tile_address(cb_meta, 0));
    const uint32_t nu = plan[0];
    for (uint32_t u = 0; u < nu; ++u) {
        const uint32_t mtc = plan[3 + 3 * u];
        cb_wait_front(cb_w, 768);
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
            matmul_block_init(cb_x, cb_w, false, 4, 2, 24);
            tile_regs_acquire();
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t b = (j + me) & 7;
                cb_wait_front(cb_x, 48);
                for (uint32_t k = 0; k < 24; ++k) {
                    matmul_block(cb_x, cb_w, k, b * 96 + k * 4, 0, false, 4, 2, 24);
                }
                cb_pop_front(cb_x, 48);
            }
            clamp_tile_init();
            for (uint32_t m = 0; m < 2; ++m) {
                clamp_tile(m * 4 + 0, nbig_bits, lim_bits);
                clamp_tile(m * 4 + 1, nbig_bits, lim_bits);
                clamp_tile(m * 4 + 2, nlim_bits, lim_bits);
                clamp_tile(m * 4 + 3, nlim_bits, lim_bits);
            }
            silu_tile_init();
            for (uint32_t m = 0; m < 2; ++m) {
                silu_tile(m * 4 + 0);
                silu_tile(m * 4 + 1);
            }
            mul_binary_tile_init();
            for (uint32_t m = 0; m < 2; ++m) {
                mul_binary_tile(m * 4 + 0, m * 4 + 2, m * 4 + 0);
                mul_binary_tile(m * 4 + 1, m * 4 + 3, m * 4 + 1);
            }
            tile_regs_commit();
            cb_reserve_back(cb_o, 4);
            tile_regs_wait();
            pack_tile(0, cb_o, 0);
            pack_tile(1, cb_o, 1);
            pack_tile(4, cb_o, 2);
            pack_tile(5, cb_o, 3);
            tile_regs_release();
            cb_push_back(cb_o, 4);
        }
        cb_pop_front(cb_w, 768);
    }
    cb_pop_front(cb_meta, 1);
}
