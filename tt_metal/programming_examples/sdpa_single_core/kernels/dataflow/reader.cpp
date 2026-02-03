// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(2);
    constexpr uint32_t num_iter = get_compile_time_arg_val(3);

    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto kt_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_kt_in = tt::CBIndex::c_1;

    for (uint32_t i = 0; i < num_iter; i++) {
        const uint32_t num_q_tiles = Sq_chunk_t * head_dim_t;
        cb_reserve_back(cb_q_in, num_q_tiles);
        DPRINT << "Producing " << num_q_tiles << " Q tiles." << ENDL();
        // do nothing for now
        cb_push_back(cb_q_in, num_q_tiles);

        const uint32_t num_kt_tiles = head_dim_t * Sk_chunk_t;
        cb_reserve_back(cb_kt_in, num_kt_tiles);
        DPRINT << "Producing " << num_kt_tiles << " KT tiles." << ENDL();
        // do nothing for now
        cb_push_back(cb_kt_in, num_kt_tiles);
    }
}
