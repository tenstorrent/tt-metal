// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"

using std::uint32_t;

template <
    uint32_t Sq_chunk_t,
    uint32_t Sk_chunk_t,
    uint32_t head_dim_t>
void sdpa_inner_loop(
    const uint32_t cb_q_in,
    const uint32_t cb_kt_in,
    const uint32_t cb_qkt_out) {

}

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(2);
    constexpr uint32_t num_iter = get_compile_time_arg_val(3);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_qkt_out = tt::CBIndex::c_2;
    
    for (uint32_t iter = 0; iter < num_iter; iter++) {
        constexpr uint32_t num_q_tiles = Sq_chunk_t * head_dim_t;
        constexpr uint32_t num_kt_tiles = head_dim_t * Sk_chunk_t;
        cb_reserve_front(cb_q_in, num_q_tiles);
        cb_reserve_front(cb_kt_in, num_kt_tiles);

        sdpa_inner_loop<Sq_chunk_t, Sk_chunk_t,head_dim_t> (
            cb_q_in,
            cb_kt_in,
            cb_qkt_out);

        cb_pop_front(cb_q_in, num_q_tiles);
        cb_pop_front(cb_kt_in, num_kt_tiles);
    }
}
