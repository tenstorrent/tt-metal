// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "dataflow_api.h"

#include "debug/dprint.h"

SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
SliceRange srt = SliceRange{ .h0 = 0, .h1 = 16, .hs = 1, .w0 = 0, .w1 = 2, .ws = 1 };

// inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     DPRINT << "======" << ENDL();
//     for (int32_t r = 0; r < 32; ++ r) {
//         SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 64, .ws = 2};
//         DPRINT << (uint) r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL();
//     }
//     DPRINT << "++++++" << ENDL();
// }

inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
    for (uint32_t page = 0; page < npages; ++ page) {
        DPRINT << start + page << ": ";
        for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
            DPRINT << BF16(*ptr) << " ";
        }
        DPRINT << ENDL();
    }
}

#define ALWI inline __attribute__((always_inline))

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
ALWI bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++ i) {
        ptr[i] = val;
    }
    return true;
}

ALWI bool fill_with_val_async(uint32_t local_src_addr, uint32_t begin_addr, int32_t nrows, uint32_t row_nbytes) {
    uint32_t curr_addr = begin_addr;
    uint64_t local_noc_src_addr = get_noc_addr(local_src_addr);
    for (int32_t row_i = 0; row_i < nrows; ++ row_i) {
        noc_async_read_one_packet(local_noc_src_addr, curr_addr, row_nbytes);
        // noc_async_read(local_noc_src_addr, curr_addr, row_nbytes);
        curr_addr += row_nbytes;
    }
    return true;
}

/**
 * Max-pool 2D.
 */
void kernel_main() {
    const uint32_t reader_nindices = get_arg_val<uint32_t>(0);
    const uint32_t window_h = get_arg_val<uint32_t>(1);
    const uint32_t window_w = get_arg_val<uint32_t>(2);

    const int32_t pad_w = get_arg_val<int32_t>(3);

    // channel size in bytes, multiple of 32
    const uint32_t in_nbytes_c = get_arg_val<uint32_t>(4);
    const uint32_t in_nbytes_c_log2 = get_arg_val<uint32_t>(5);

    // input tensor height / width / channels
    const int32_t in_w = get_arg_val<int32_t>(6);
    const uint32_t in_cb_nsticks = get_arg_val<uint32_t>(7);


    // compile time args
    // value of 1 in bf16 in a uin32_t
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(2);

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in1;
    constexpr uint32_t in_shard_cb_id = tt::CB::c_in2;    // local input shard
    constexpr uint32_t in_reader_indices_cb_id = tt::CB::c_in3;

    constexpr uint32_t TILE_HW = 1024;

    // Reduce scalar = 1
    cb_reserve_back(in_scalar_cb_id, 1);

    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    // fill 1 tile w/ scalar
    fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_HW, bf16_one_u16);
    cb_push_back(in_scalar_cb_id, 1);

    uint16_t minus_inf = 0xf7ff;
    // fill one row of in_cb_id rows with -inf
    uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
    fill_with_val(in_l1_write_addr, in_nbytes_c >> 1, minus_inf);
    // now replicate the row to fill the entire in_cb_id
    fill_with_val_async(in_l1_write_addr, in_l1_write_addr + in_nbytes_c, in_cb_nsticks - 1, in_nbytes_c);
    noc_async_read_barrier();

    // NOTE: batch is folded in

    uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);

    // cb_wait_front(in_reader_indices_cb_id, 1);

    uint32_t in_w_padded = in_w + 2 * pad_w;

    uint32_t counter = 0;
    while (counter < reader_nindices) {
        cb_reserve_back(in_cb_id, 1);

        uint32_t out_l1_write_addr_base = get_write_ptr(in_cb_id);
        uint32_t out_l1_write_addr = out_l1_write_addr_base;
        uint16_t top_left_local_index = reader_indices_ptr[counter];
        uint32_t h_multiples = 0;
        for (uint32_t h = 0; h < window_h; ++ h, h_multiples += in_w_padded) {
            uint32_t stick_offset = top_left_local_index + h_multiples;
            uint32_t read_offset = in_l1_read_base_addr + (stick_offset << in_nbytes_c_log2);
            noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, in_nbytes_c * window_w);
            out_l1_write_addr += in_nbytes_c * window_w;
        }
        noc_async_read_barrier();

        cb_push_back(in_cb_id, 1);

        ++ counter;
    }

    // DPRINT << "READER DONE!!" << ENDL();
} // kernel_main()
