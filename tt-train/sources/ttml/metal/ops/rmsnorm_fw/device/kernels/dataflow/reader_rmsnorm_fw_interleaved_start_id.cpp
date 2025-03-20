// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>

#include "dataflow_api.h"

// TODO: improve with a more efficient implementation
// using noc_async_writes
void generate_tile_with_value(uint32_t cb, uint32_t packed_value) {
    constexpr uint32_t onetile = 1U;
    cb_reserve_back(cb, onetile);
    uint32_t* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb));
    // 512 = 32x16
    for (uint32_t i = 0; i < 512U; ++i, ++ptr) {
        *ptr = packed_value;
    }
    cb_push_back(cb, onetile);
}

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_4;

    constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
    constexpr uint32_t packed_eps = get_compile_time_arg_val(1);
    constexpr uint32_t mask_w = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);

    constexpr uint32_t onetile = 1U;

#ifdef DO_MASK_W
    constexpr bool do_mask_w = true;
#else
    constexpr bool do_mask_w = false;
#endif

    // generate mask tile
    if constexpr (do_mask_w) {
        cb_reserve_back(cb_mask_w_idx, onetile);
        uint16_t* ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_mask_w_idx));
        constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
        constexpr uint16_t zero = 0x0;
        for (uint32_t face = 0; face < 4; ++face) {
            uint32_t offset = (face & 1U) << 4U;
            for (uint32_t h = 0; h < 16; ++h) {
                for (uint32_t w = 0; w < 16; ++w, ++ptr) {
                    *ptr = (offset + w < mask_w) ? one : zero;
                }
            }
        }
        cb_push_back(cb_mask_w_idx, onetile);
    }

    // generate tiles to include scalar and epsilon
    generate_tile_with_value(cb_scaler_idx, packed_scaler);
    generate_tile_with_value(cb_eps_idx, packed_eps);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> gamma_address_generator = {
        .bank_base_address = gamma_address, .page_size = tile_bytes, .data_format = data_format};

    const uint32_t max_block_size = 4;

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t idx = (start_row + i) * Wt;

#ifdef EVERYTHING_FITS_IN_L1
        cb_reserve_back(cb_input_idx, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_idx);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(idx + j, input_address_generator, l1_write_addr);
            l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_idx, Wt);

        if (i == 0) {
            cb_reserve_back(cb_gamma_idx, Wt);
            uint32_t l1_gamma_write_addr = get_write_ptr(cb_gamma_idx);
            for (uint32_t j = 0; j < Wt; ++j) {
                noc_async_read_tile(j, gamma_address_generator, l1_gamma_write_addr);
                l1_gamma_write_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_idx, Wt);
        }

#elif defined(EVERYTHING_EXCEPT_GAMMA_FITS_IN_L1)
        cb_reserve_back(cb_input_idx, Wt);
        uint32_t l1_input_write_addr = get_write_ptr(cb_input_idx);
        for (uint32_t j = 0; j < Wt; ++j) {
            noc_async_read_tile(idx + j, input_address_generator, l1_input_write_addr);
            l1_input_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_input_idx, Wt);

        for (uint32_t j = 0; j < Wt; j += block_size) {
            cb_reserve_back(cb_gamma_idx, block_size);
            uint32_t l1_gamma_write_addr = get_write_ptr(cb_gamma_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                noc_async_read_tile(j + block_idx, gamma_address_generator, l1_gamma_write_addr);
                l1_gamma_write_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_gamma_idx, block_size);
        }
#else
        for (uint32_t j = 0; j < Wt; j += block_size) {
            cb_reserve_back(cb_input_idx, block_size);
            uint32_t l1_write_addr = get_write_ptr(cb_input_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                noc_async_read_tile(idx + j + block_idx, input_address_generator, l1_write_addr);
                l1_write_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_idx, block_size);
        }

        for (uint32_t j = 0; j < Wt; j += block_size) {
            cb_reserve_back(cb_input_idx, block_size);
            uint32_t l1_write_addr = get_write_ptr(cb_input_idx);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                noc_async_read_tile(idx + j + block_idx, input_address_generator, l1_write_addr);
                l1_write_addr += tile_bytes;
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_idx, block_size);

            // reading gamma to L1
            {
                cb_reserve_back(cb_gamma_idx, block_size);
                uint32_t l1_write_addr = get_write_ptr(cb_gamma_idx);
                for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                    noc_async_read_tile(j + block_idx, gamma_address_generator, l1_write_addr);
                    l1_write_addr += tile_bytes;
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma_idx, block_size);
            }
        }
#endif
    }
}
