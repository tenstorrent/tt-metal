// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

inline void read_tiles(
    const uint32_t cb_idx,
    const InterleavedAddrGenFast<true>& addr_gen,
    const uint32_t start_tile,
    const uint32_t num_tiles,
    const uint32_t tile_bytes) {
    // Reads `num_tiles` tiles from DRAM starting at logical tile index `start_tile` into circular buffer `cb_idx`.
    cb_reserve_back(cb_idx, num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_idx);
    for (uint32_t k = 0; k < num_tiles; ++k) {
        noc_async_read_tile(start_tile + k, addr_gen, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    cb_push_back(cb_idx, num_tiles);
}

inline void read_input_tensors(
    const uint32_t start_row,
    const uint32_t num_rows_to_process,
    const uint32_t Wt,
    const uint32_t block_size,
    const uint32_t tile_bytes,
    const InterleavedAddrGenFast<true>& input_addr_gen,
    const InterleavedAddrGenFast<true>& dL_out_addr_gen,
    const InterleavedAddrGenFast<true>& gamma_addr_gen,
    const InterleavedAddrGenFast<true>& rms_a_addr_gen,
    const uint32_t cb_input_idx,
    const uint32_t cb_dL_out_idx,
    const uint32_t cb_gamma_idx,
    const uint32_t cb_rms_a_idx) {
    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        uint32_t idx = (start_row + i) * Wt;
        read_tiles(cb_rms_a_idx, rms_a_addr_gen, start_row + i, 1, tile_bytes);
        for (uint32_t j = 0; j < Wt; j += block_size) {
            read_tiles(cb_input_idx, input_addr_gen, idx + j, block_size, tile_bytes);
            read_tiles(cb_dL_out_idx, dL_out_addr_gen, idx + j, block_size, tile_bytes);
            read_tiles(cb_gamma_idx, gamma_addr_gen, j, block_size, tile_bytes);
            noc_async_read_barrier();
        }
    }
}

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_a_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    // TODO: Consider moving these constants to compile-time arguments to avoid index-mismatch issues while developing.
    // CBs with input data
    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;  // 1/c - used for scaling
    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
    constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;
    constexpr uint32_t cb_one_idx = tt::CBIndex::c_6;  // Used to reduce scale to a single value
    // CBs with output data
    // Create more intermedaite-output CBs that will be used exclusively by the writer. Do not compute anything on them
    constexpr uint32_t cb_dL_da_idx = tt::CBIndex::c_7;
    constexpr uint32_t cb_dL_dgamma_components = tt::CBIndex::c_8;
    // CBs with intermediate computations
    constexpr uint32_t cb_recip_rms_a_bcasted_idx = tt::CBIndex::c_9;
    constexpr uint32_t cb_scale_idx = tt::CBIndex::c_10;
    constexpr uint32_t cb_scale_bcasted = tt::CBIndex::c_11;

    constexpr uint32_t packed_scaler = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr uint32_t mask_w = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);

    constexpr uint32_t onetile = 1U;

#ifdef DO_MASK_W
    constexpr bool do_mask_w = true;
#else
    constexpr bool do_mask_w = false;
#endif
    // Generate scaler and mask tile.
    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint32_t packed_one = one | (one << 16U);
    constexpr uint16_t zero = 0x0;
    constexpr uint16_t minus_inf = 0xFF80;  // (bfloat16)-inf -> uint16_t

    // Generate mask tile.
    if constexpr (do_mask_w) {
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
    }

    // Generate tile with scalar.
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);
    // Generate tile with one.
    generate_tile_with_packed_bfloat16_value(cb_one_idx, packed_one);

    const uint32_t tile_bytes = get_tile_size(cb_input_idx);
    const DataFormat data_format = get_dataformat(cb_input_idx);

    const InterleavedAddrGenFast</* is_dram */ true> input_address_generator = {
        .bank_base_address = input_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> gamma_address_generator = {
        .bank_base_address = gamma_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> rms_a_address_generator = {
        .bank_base_address = rms_a_address, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast</* is_dram */ true> dL_out_address_generator = {
        .bank_base_address = dL_out_address, .page_size = tile_bytes, .data_format = data_format};

    // To be read:
    // - Input tensor a: shape [B,1,S,C], read Wt tiles for each row.
    // - Gamma: shape [1,1,1,C], despite being shared for all rows, read Wt tiles each time because the last dimension
    // might not fit in L1.
    // - RMS(a): shape [B,1,S,1], but only one scalar per row.
    // - dL_out: same shape as input, read Wt tiles for each row.

    // Read all input tensors. If everything fits in L1, this is the only read we need to do.
    read_input_tensors(
        start_row,
        num_rows_to_process,
        Wt,
        block_size,
        tile_bytes,
        input_address_generator,
        dL_out_address_generator,
        gamma_address_generator,
        rms_a_address_generator,
        cb_input_idx,
        cb_dL_out_idx,
        cb_gamma_idx,
        cb_rms_a_idx);

#ifndef EVERYTHING_FITS_IN_L1
    // If everything does not fit in L1, we need to read the big input tensors again.
    read_input_tensors(
        start_row,
        num_rows_to_process,
        Wt,
        block_size,
        tile_bytes,
        input_address_generator,
        dL_out_address_generator,
        gamma_address_generator,
        rms_a_address_generator,
        cb_input_idx,
        cb_dL_out_idx,
        cb_gamma_idx,
        cb_rms_a_idx);
#endif
}
