// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

// CBs - always read to BF16 first
constexpr uint32_t cb_input_a_idx = tt::CBIndex::c_0;
constexpr uint32_t cb_input_b_idx = tt::CBIndex::c_1;
constexpr uint32_t cb_input_a_fp32_idx = tt::CBIndex::c_2;
constexpr uint32_t cb_input_b_fp32_idx = tt::CBIndex::c_3;

#ifdef COPY_A_TO_FP32
constexpr bool copy_a_to_fp32 = true;
#else
constexpr bool copy_a_to_fp32 = false;
#endif

#ifdef COPY_B_TO_FP32
constexpr bool copy_b_to_fp32 = true;
#else
constexpr bool copy_b_to_fp32 = false;
#endif

void kernel_main() {
    uint32_t input_a_addr = get_arg_val<uint32_t>(0);
    uint32_t input_b_addr = get_arg_val<uint32_t>(1);

    const uint32_t tile_bytes_bf16 = get_tile_size(cb_input_a_idx);
    const InterleavedAddrGen<true> input_a_addr_gen = {.bank_base_address = input_a_addr, .page_size = tile_bytes_bf16};
    const InterleavedAddrGen<true> input_b_addr_gen = {.bank_base_address = input_b_addr, .page_size = tile_bytes_bf16};

    // Read tile A to BF16 CB
    cb_reserve_back(cb_input_a_idx, 1);
    uint32_t l1_write_addr_a = get_write_ptr(cb_input_a_idx);
    noc_async_read_tile(0, input_a_addr_gen, l1_write_addr_a);
    noc_async_read_barrier();
    cb_push_back(cb_input_a_idx, 1);

    // Read tile B to BF16 CB
    cb_reserve_back(cb_input_b_idx, 1);
    uint32_t l1_write_addr_b = get_write_ptr(cb_input_b_idx);
    noc_async_read_tile(0, input_b_addr_gen, l1_write_addr_b);
    noc_async_read_barrier();
    cb_push_back(cb_input_b_idx, 1);

    // Note: Format conversion to FP32 is done in compute kernel using copy_tile
    // to ensure proper data format handling
}
