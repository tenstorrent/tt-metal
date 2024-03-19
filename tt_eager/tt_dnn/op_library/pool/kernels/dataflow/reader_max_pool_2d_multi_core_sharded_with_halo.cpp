// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include "dataflow_api.h"

// #include "debug/dprint.h"

// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 16, .hs = 1, .w0 = 0, .w1 = 2, .ws = 1 };

// inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
//     DPRINT << "======" << ENDL();
//     for (int32_t r = 0; r < 32; ++ r) {
//         SliceRange sr = SliceRange{.h0 = r, .h1 = r+1, .hs = 1, .w0 = 0, .w1 = 64, .ws = 2};
//         DPRINT << (uint) r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL();
//     }
//     DPRINT << "++++++" << ENDL();
// }

// inline void print_pages(uint32_t l1_addr, uint32_t pagelen, uint32_t npages, uint32_t start = 0) {
//     volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr) + start * pagelen;
//     for (uint32_t page = 0; page < npages; ++ page) {
//         DPRINT << start + page << ": ";
//         for (uint32_t j = 0; j < pagelen; ++ j, ++ ptr) {
//             DPRINT << BF16(*ptr) << " ";
//         }
//         DPRINT << ENDL();
//     }
// }

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
        curr_addr += row_nbytes;
    }
    return true;
}

/**
 * Max-pool 2D.
 */
void kernel_main() {
    // input tensor address
    // const uint32_t in_addr = get_arg_val<uint32_t>(0);

    const uint32_t window_h = get_arg_val<uint32_t>(2);
    const uint32_t window_w = get_arg_val<uint32_t>(3);
    // const int32_t window_hw = get_arg_val<int32_t>(4);
    // window_hw_padded = window_hw rounded up to the tile size (can be multiple tiles)
    // const uint32_t window_hw_padded = get_arg_val<uint32_t>(5);

    const int32_t pad_h = get_arg_val<int32_t>(8);
    const int32_t pad_w = get_arg_val<int32_t>(9);

    const int32_t out_h = get_arg_val<int32_t>(10);
    const int32_t out_w = get_arg_val<int32_t>(11);

    // channel size in bytes, multiple of 32
    const uint32_t in_nbytes_c = get_arg_val<uint32_t>(14);

    // input tensor height / width / channels
    const int32_t in_h = get_arg_val<int32_t>(16);
    const int32_t in_w = get_arg_val<int32_t>(17);
    // const int32_t in_c = get_arg_val<int32_t>(19);

    // const int32_t in_cb_pagesize = get_arg_val<int32_t>(22);
    // product of window_hw_padded and in_c padded to the tile size (can be multiple tiles)
    // const int32_t in_cb_page_nelems_padded = get_arg_val<int32_t>(24);

    // out_w divided by number of out_nelems (== number of blocks per iteration)
    // const int32_t out_w_loop_count = get_arg_val<int32_t>(25);
    const uint32_t in_log_base_2_of_page_size = get_arg_val<uint32_t>(26);

    // const uint32_t nbatch = get_arg_val<uint32_t>(27);

    // const uint32_t in_hw = get_arg_val<uint32_t>(28);

    const uint32_t minus_inf_buffer_addr = get_arg_val<uint32_t>(34);
    // const uint32_t minus_inf_buffer_nbytes = get_arg_val<uint32_t>(35);
    const uint32_t in_cb_nsticks = get_arg_val<uint32_t>(36);

    // the starting offset for assigned batch input row id (batch_offset)
    // uint32_t core_offset_in_stick_id = get_arg_val<uint32_t>(37);

    // compile time args
    // constexpr bool is_in_dram = get_compile_time_arg_val(0) == 1;
    // value of 1 in bf16 in a uin32_t
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(2);
    // number of output elements per iteration == number of blocks per iteration
    // constexpr uint32_t out_nelems = get_compile_time_arg_val(3);
    // constexpr bool use_pow2 = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t stride_h = get_compile_time_arg_val(5);
    constexpr uint32_t stride_w = get_compile_time_arg_val(6);

    // constexpr uint32_t reader_noc = get_compile_time_arg_val(7);
    // constexpr uint32_t writer_noc = get_compile_time_arg_val(8);

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in4;
    constexpr uint32_t in_shard_cb_id = tt::CB::c_in2;    // local input shard
    constexpr uint32_t reader_indices_cb_id = tt::CB::c_intermed1;

    constexpr uint32_t TILE_HW = 1024;

    // DPRINT << "HAHA 0" << ENDL();

    // Reduce scalar = 1
    cb_reserve_back(in_scalar_cb_id, 1);

    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    // fill 1 tile w/ scalar
    fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_HW, bf16_one_u16);
    cb_push_back(in_scalar_cb_id, 1);

    uint16_t minus_inf = 0xf7ff;
    // fill one row of in_cb_id rows with -inf
    uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
    fill_with_val(in_l1_write_addr, in_nbytes_c>>1, minus_inf);
    // now replicate the row to fill the entire in_cb_id
    fill_with_val_async(in_l1_write_addr, in_l1_write_addr+in_nbytes_c, in_cb_nsticks-1, in_nbytes_c);
    noc_async_read_barrier();

    // NOTE: batch is folded in

    // DPRINT << "HAHA 1" << ENDL();

    // uint32_t core_out_w_i_start = get_arg_val<int32_t>(38);
    // uint32_t core_out_h_i_start = get_arg_val<int32_t>(39);
    uint32_t nsticks_per_core = get_arg_val<uint32_t>(40);

    // uint32_t nsticks_per_core_by_nblocks = get_arg_val<uint32_t>(42);

    uint32_t local_out_stick_start = get_arg_val<uint32_t>(43);
    uint32_t nsticks_per_batch = get_arg_val<uint32_t>(44);
    // uint32_t local_in_stick_start = get_arg_val<uint32_t>(45);
    // uint32_t local_in_stick_end = get_arg_val<uint32_t>(46);
    // uint32_t in_nsticks_per_batch = get_arg_val<uint32_t>(47);
    // uint32_t in_nsticks_per_core = get_arg_val<uint32_t>(48);

    // uint32_t has_left = get_arg_val<uint32_t>(49);
    // uint32_t left_noc_x = get_arg_val<uint32_t>(50);
    // uint32_t left_noc_y = get_arg_val<uint32_t>(51);
    // uint32_t has_right = get_arg_val<uint32_t>(52);
    // uint32_t right_noc_x = get_arg_val<uint32_t>(53);
    // uint32_t right_noc_y = get_arg_val<uint32_t>(54);

    // TODO: pass these as runtime args
    uint32_t in_nbytes_c_log2 = 7;  // for in_nbytes_c == 128
    // for in_nsticks_per_core == 1024, remainder mask = 0x3ff
    // uint32_t in_nsticks_per_core_rem_mask = 0x3ff;
    // uint32_t in_nsticks_per_core_rem_mask = get_arg_val<uint32_t>(55);

    // uint32_t has_left_left = get_arg_val<uint32_t>(56);
    // uint32_t left_left_noc_x = get_arg_val<uint32_t>(57);
    // uint32_t left_left_noc_y = get_arg_val<uint32_t>(58);
    // uint32_t has_right_right = get_arg_val<uint32_t>(59);
    // uint32_t right_right_noc_x = get_arg_val<uint32_t>(60);
    // uint32_t right_right_noc_y = get_arg_val<uint32_t>(61);
    // uint32_t left_in_stick_start = get_arg_val<uint32_t>(62);
    // uint32_t right_in_stick_end = get_arg_val<uint32_t>(63);

    // int32_t my_core = get_arg_val<int32_t>(64);

    int32_t initial_skip = get_arg_val<int32_t>(65);
    int32_t partial_first_row_nsticks = get_arg_val<int32_t>(66);
    int32_t partial_first_row_skip = get_arg_val<int32_t>(67);
    int32_t partial_top_image_nrows = get_arg_val<int32_t>(68);
    int32_t partial_top_image_skip = get_arg_val<int32_t>(69);
    int32_t full_nimages = get_arg_val<int32_t>(70);
    int32_t full_images_skip = get_arg_val<int32_t>(71);
    int32_t partial_bottom_image_nrows = get_arg_val<int32_t>(72);
    int32_t partial_last_row_nsticks = get_arg_val<int32_t>(73);
    int32_t start_stick = get_arg_val<int32_t>(74);

    int32_t in_start_stick = get_arg_val<int32_t>(75);
    int32_t in_first_partial_right_aligned_row_width = get_arg_val<int32_t>(76);
    int32_t in_first_partial_image_num_rows = get_arg_val<int32_t>(77);
    int32_t in_num_full_images = get_arg_val<int32_t>(78);
    int32_t in_last_partial_image_num_rows = get_arg_val<int32_t>(79);
    int32_t in_last_partial_left_aligned_row_width = get_arg_val<int32_t>(80);
    int32_t in_initial_skip = get_arg_val<int32_t>(81);
    int32_t in_skip_after_stick = get_arg_val<int32_t>(82);
    int32_t in_skip_after_partial_right_aligned_row = get_arg_val<int32_t>(83);
    int32_t in_skip_after_first_partial_image_row = get_arg_val<int32_t>(84);
    int32_t in_skip_after_full_image = get_arg_val<int32_t>(85);
    int32_t in_skip_after_each_full_row = get_arg_val<int32_t>(86);
    int32_t in_skip_after_each_stick = get_arg_val<int32_t>(87);
    // int32_t in_skip_after_each_stick = stride_w;

    uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    volatile tt_l1_ptr uint32_t* reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(reader_indices_cb_id));

    // print_pages(in_l1_read_base_addr, 64, 13 * 114, 0);

    // DPRINT << "initial_skip: " << (uint) initial_skip << ENDL();
    // DPRINT << "start_stick: " << (uint) start_stick << ENDL();
    // DPRINT << "partial_first_row_nsticks: " << (uint) partial_first_row_nsticks << ENDL();
    // DPRINT << "partial_first_row_skip: " << (uint) partial_first_row_skip << ENDL();
    // DPRINT << "partial_top_image_nrows: " << (uint) partial_top_image_nrows << ENDL();
    // DPRINT << "partial_top_image_skip: " << (uint) partial_top_image_skip << ENDL();
    // DPRINT << "full_nimages: " << (uint) full_nimages << ENDL();
    // DPRINT << "full_nimages_skip: " << (uint) full_images_skip << ENDL();
    // DPRINT << "partial_bottom_image_nrows: " << (uint) partial_bottom_image_nrows << ENDL();
    // DPRINT << "partial_last_row_nsticks: " << (uint) partial_last_row_nsticks << ENDL();
    // DPRINT << "TOTAL nsticks = " << (uint) partial_first_row_nsticks + partial_top_image_nrows * in_w + full_nimages * in_w * in_h + partial_bottom_image_nrows * in_w + partial_last_row_nsticks << ENDL();

    // DPRINT << "initial_skip: " << (uint) in_initial_skip << ENDL();
    // DPRINT << "start_stick: " << (uint) in_start_stick << ENDL();
    // DPRINT << "partial_first_row_nsticks: " << (uint) in_first_partial_right_aligned_row_width << ENDL();
    // DPRINT << "partial_first_row_skip: " << (uint) in_skip_after_partial_right_aligned_row << ENDL();
    // DPRINT << "partial_top_image_nrows: " << (uint) in_first_partial_image_num_rows << ENDL();
    // DPRINT << "partial_top_image_skip: " << (uint) in_skip_after_first_partial_image_row << ENDL();
    // DPRINT << "full_nimages: " << (uint) in_num_full_images << ENDL();
    // DPRINT << "full_nimages_skip: " << (uint) in_skip_after_full_image << ENDL();
    // DPRINT << "partial_bottom_image_nrows: " << (uint) in_last_partial_image_num_rows << ENDL();
    // DPRINT << "partial_last_row_nsticks: " << (uint) in_last_partial_left_aligned_row_width << ENDL();

    // // section 0: initial skip
    uint32_t top_left_i = in_initial_skip;
    uint32_t reader_i = 0;

    uint32_t in_w_padded = in_w + 2 * pad_w;
    // input index offsets:
    //  between each stick = stride_w
    //  between each row = + 2 * pad_w + (stride_h - 1) * row_size_padded
    //  between each batch = + pad_h * row_size_padded

    // section 1: partial first row
    for (int32_t i = 0; i < partial_first_row_nsticks; ++ i) {
        reader_indices_ptr[reader_i ++] = top_left_i;
        top_left_i += in_skip_after_each_stick;
    }
    // if (partial_first_row_nsticks > 0) {
        top_left_i += in_skip_after_partial_right_aligned_row;  // 2 * pad_w + (stride_h - 1) * in_w_padded;
    // }

    // section 2: partial first image
    for (int32_t i = 0; i < partial_top_image_nrows; ++ i) {
        for (int32_t j = 0; j < out_w; ++ j) {
            reader_indices_ptr[reader_i ++] = top_left_i;
            top_left_i += in_skip_after_each_stick;
        }
        // skip pad per row
        top_left_i += in_skip_after_each_full_row; // 2 * pad_w + (stride_h - 1) * in_w_padded;
    }
    // if (partial_top_image_nrows > 0) {
        top_left_i += in_skip_after_first_partial_image_row; // pad_h * in_w_padded;
    // }

    // section 3: full images
    for (int32_t n = 0; n < full_nimages; ++ n) {
        for (int32_t i = 0; i < out_h; ++ i) {
            for (int32_t j = 0; j < out_w; ++ j) {
                reader_indices_ptr[reader_i ++] = top_left_i;
                top_left_i += in_skip_after_each_stick;
            }
            // skip pad per row
            top_left_i += in_skip_after_each_full_row; // 2 * pad_w + (stride_h - 1) * in_w_padded;
        }
        // skip pad rows per image
        top_left_i += in_skip_after_full_image; // pad_h * in_w_padded;
    }

    // section 4: partial last image
    for (int32_t i = 0; i < partial_bottom_image_nrows; ++ i) {
        for (int32_t j = 0; j < out_w; ++ j) {
            reader_indices_ptr[reader_i ++] = top_left_i;
            top_left_i += in_skip_after_each_stick;
        }
        // skip pad per row
        top_left_i += in_skip_after_each_full_row; // 2 * pad_w + (stride_h - 1) * in_w_padded;
    }

    // section 5: partial last row
    for (int32_t i = 0; i < partial_last_row_nsticks; ++ i) {
        reader_indices_ptr[reader_i ++] = top_left_i;
        top_left_i += in_skip_after_each_stick;
    }

    // DPRINT << "reader_i = " << reader_i << ENDL();
    // for (uint32_t i = 0; i < reader_i; ++ i) {
    //     DPRINT << i << ": " << reader_indices_ptr[i] << ENDL();
    // }

    uint32_t counter = 0;
    while (counter < reader_i) {
        cb_reserve_back(in_cb_id, 1);

        uint32_t out_l1_write_addr_base = get_write_ptr(in_cb_id);
        uint32_t out_l1_write_addr = out_l1_write_addr_base;
        int32_t top_left_local_index = reader_indices_ptr[counter ++];
        uint32_t h_multiples = 0;
        for (uint32_t h = 0; h < window_h; ++ h, h_multiples += in_w_padded) {
            for (uint32_t w = 0; w < window_w; ++ w) {
                uint32_t stick_offset = top_left_local_index + (w + h_multiples);
                uint32_t read_offset = in_l1_read_base_addr + (stick_offset << in_nbytes_c_log2);
                noc_async_read_one_packet(get_noc_addr(read_offset), out_l1_write_addr, in_nbytes_c);
                out_l1_write_addr += in_nbytes_c;
            }
        }
        // print_pages(out_l1_write_addr_base, 64, 10, 0);
        noc_async_read_barrier();

        cb_push_back(in_cb_id, 1);
    }
} // kernel_main()
