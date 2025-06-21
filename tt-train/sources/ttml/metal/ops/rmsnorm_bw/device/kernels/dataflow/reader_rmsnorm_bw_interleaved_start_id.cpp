// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compile_time_args.h>

#include <cstdint>

#include "debug/dprint.h"
#include "tt-train/sources/ttml/metal/ops/common/dataflow_utils.hpp"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
        break;
    }
}

inline void read_tiles(
    uint32_t cb_idx,
    const InterleavedAddrGenFast<true>& addr_gen,
    uint32_t start_tile,
    uint32_t num_tiles,
    uint32_t tile_bytes) {
    // Reads `num_tiles` tiles from DRAM starting at logical tile index `start_tile` into circular buffer `cb_idx`.
    // - For row-major tensors (like input or dL_out), `start_tile` is (row_index * Wt), so we read a full row.
    // - For gamma, which is shared across all rows, `start_tile` is 0 and we read Wt tiles only once.
    // - For per-row scalars (like RMS(a)), `num_tiles` is 1 (onetile), and `start_tile` is (start_row + i).
    cb_reserve_back(cb_idx, num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_idx);
    for (uint32_t j = 0; j < num_tiles; ++j) {
        noc_async_read_tile(start_tile + j, addr_gen, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_idx, num_tiles);
}

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t input_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t gamma_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t rms_a_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t dL_out_address = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_rows_to_process = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_row = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_input_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_w_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_scaler_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_gamma_idx = tt::CBIndex::c_3;
    constexpr uint32_t cb_rms_a_idx = tt::CBIndex::c_4;
    constexpr uint32_t cb_dL_out_idx = tt::CBIndex::c_5;

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
    // generate scaler and mask tile
    constexpr uint16_t one = 0x00003F80;  // (bfloat16)1.0 -> uint16_t
    constexpr uint16_t zero = 0x0;
    constexpr uint16_t minus_inf = 0xFF80;  // (bfloat16)-inf -> uint16_t

    // generate mask tile
    if constexpr (do_mask_w) {
        // DPRINT << "Generating mask tile with value: " << mask_w << ENDL();
        generate_mask_tile(cb_mask_w_idx, one, zero, mask_w);
        // DPRINT << "Generated mask tile done" << ENDL();
    }

    // generate tile with scalar
    generate_tile_with_packed_bfloat16_value(cb_scaler_idx, packed_scaler);

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

    for (uint32_t i = 0; i < num_rows_to_process; ++i) {
        // calculate the address of the first tile in the row
        // start_row is the number of rows already processed in other cores
        uint32_t idx = (start_row + i) * Wt;  // (take already processed rows + current row)*Wt(number of tiles in row)

#define EVERYTHING_FITS_IN_L1 1
#ifdef EVERYTHING_FITS_IN_L1
        // Input tensor a: shape [B,1,S,C], read Wt tiles for this row.
        // start_tile = idx ensures we read the correct row.
        read_tiles(cb_input_idx, input_address_generator, idx, Wt, tile_bytes);

        // Gamma: shape [1,1,1,C], shared for all rows, so read only once at i == 0.
        // start_tile = 0 because gamma is not row-dependent.
        if (i == 0) {
            read_tiles(cb_gamma_idx, gamma_address_generator, 0, Wt, tile_bytes);
        }

        // RMS(a): shape [B,1,S,1], but only one scalar per row.
        // start_tile = start_row + i selects the correct row's RMS value.
        // num_tiles = onetile because we only need one tile per row.
        read_tiles(cb_rms_a_idx, rms_a_address_generator, start_row + i, onetile, tile_bytes);

        // dL_out: same shape as input, read Wt tiles for this row.
        // start_tile = idx ensures we read the correct row.
        read_tiles(cb_dL_out_idx, dL_out_address_generator, idx, Wt, tile_bytes);
    }
#elif defined(SOME_OTHER_OPTIONS_TBD)
// TODO
#else
// TODO
#endif
}
