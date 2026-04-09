// Passthrough compute kernel: just copies state_in → state_out unchanged.
// Tests whether the reader/writer data pipeline is working correctly.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(2);
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_row = tt::CBIndex::c_1;
    constexpr uint32_t cb_k_col = tt::CBIndex::c_2;
    constexpr uint32_t cb_v = tt::CBIndex::c_3;
    constexpr uint32_t cb_g = tt::CBIndex::c_4;
    constexpr uint32_t cb_beta = tt::CBIndex::c_5;
    constexpr uint32_t cb_state_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        // Wait for all inputs from reader
        cb_wait_front(cb_state_in, state_tiles);
        cb_wait_front(cb_q, Kt);
        cb_wait_front(cb_k_row, Kt);
        cb_wait_front(cb_k_col, Kt);
        cb_wait_front(cb_v, Vt);
        cb_wait_front(cb_g, 1);
        cb_wait_front(cb_beta, 1);

        // Copy state_in → state_out unchanged
        cb_reserve_back(cb_state_out, state_tiles);
        copy_tile_init(cb_state_in);
        for (uint32_t s = 0; s < state_tiles; s++) {
            tile_regs_acquire();
            copy_tile(cb_state_in, s, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_state_out, s);
            tile_regs_release();
        }
        cb_push_back(cb_state_out, state_tiles);
        cb_pop_front(cb_state_in, state_tiles);

        // Produce dummy output (copy first Vt state tiles)
        cb_reserve_back(cb_out, Vt);
        cb_wait_front(cb_state_out, state_tiles);
        copy_tile_init(cb_state_out);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            tile_regs_acquire();
            copy_tile(cb_state_out, vt, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out, vt);
            tile_regs_release();
        }
        cb_push_back(cb_out, Vt);

        // Pop remaining inputs
        cb_pop_front(cb_q, Kt);
        cb_pop_front(cb_k_row, Kt);
        cb_pop_front(cb_k_col, Kt);
        cb_pop_front(cb_v, Vt);
        cb_pop_front(cb_g, 1);
        cb_pop_front(cb_beta, 1);
    }
}
