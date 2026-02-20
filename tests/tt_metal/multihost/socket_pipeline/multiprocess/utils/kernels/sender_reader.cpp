
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_loop_iterations = get_compile_time_arg_val(2);
constexpr uint32_t input_args_cta_idx = 3;
constexpr uint32_t input_args_crta_idx = 0;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_base_addr, input_page_size);

    for (uint32_t i = 0; i < num_loop_iterations; i++) {
        auto noc_read_addr = input_addr_gen.get_noc_addr(0);
        cb_reserve_back(cb0_id, 1);
        auto l1_write_addr = get_write_ptr(cb0_id);
        noc_async_read<input_page_size>(noc_read_addr, l1_write_addr, input_page_size);
        noc_async_read_barrier();
        cb_push_back(cb0_id, 1);
    }
}
