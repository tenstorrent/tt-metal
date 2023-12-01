// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// #include "debug/dprint.h"

// SliceRange srr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange srt = SliceRange{ .h0 = 0, .h1 = 8, .hs = 1, .w0 = 0, .w1 = 4, .ws = 1 };

// inline void print_sticks(uint32_t l1_addr, uint32_t stick_start, uint32_t nsticks, uint32_t stick_size = 64) {
//     for (uint32_t i = stick_start; i < stick_start + nsticks; ++ i) {
//         volatile tt_l1_ptr uint16_t* l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_addr + i * stick_size * 2);
//         DPRINT << i << ": ";
//         for (uint32_t j = 0; j < stick_size; ++ j) {
//             DPRINT << BF16(l1_ptr[j]) << " ";
//         }
//         DPRINT << ENDL();
//     }
// }

// Fill an L1 buffer with the given val
inline bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++ i) {
        ptr[i] = val;
    }
    return true;
}

void kernel_main() {
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);  // has input shard
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(1); // has output shard with padding and halo
    constexpr uint32_t pad_cb_id = get_compile_time_arg_val(2); // cb for const pad val buffer
    constexpr uint32_t pad_val_u32 = get_compile_time_arg_val(3);   // pad value to fill pad buffer with
    constexpr uint32_t pad_stick_len  = get_compile_time_arg_val(4);   // stick size (size of the pad val buffer)
    constexpr uint32_t stick_nbytes  = get_compile_time_arg_val(5);   // stick size in RM bytes (post untilize)
    constexpr uint32_t in_w  = get_compile_time_arg_val(6);   // image width (w/o padding)
    constexpr uint32_t in_h  = get_compile_time_arg_val(7);   // image height (w/o padding)
    constexpr uint32_t pad_w  = get_compile_time_arg_val(8);   // pad width
    constexpr uint32_t pad_h  = get_compile_time_arg_val(9);   // pad height

    static_assert(stick_nbytes <= NOC_MAX_BURST_SIZE); // stick_nbytes used in noc_async_read_one_packet

    const uint16_t pad_val = pad_val_u32;   // >> 16;

    cb_reserve_back(pad_cb_id, 1);
    fill_with_val(get_write_ptr(pad_cb_id), pad_stick_len, pad_val);
    cb_push_back(pad_cb_id, 1);

    // for this core's local shard
    uint32_t in_nsticks = get_arg_val<uint32_t>(0);
    uint32_t out_nsticks = get_arg_val<uint32_t>(1);

    uint32_t partial_first_row_nbytes = get_arg_val<uint32_t>(2);
    // uint32_t pad_w = get_arg_val<uint32_t>(3);
    // uint32_t in_w = get_arg_val<uint32_t>(4);
    uint32_t partial_top_image_nrows = get_arg_val<uint32_t>(5);
    // uint32_t pad_h = get_arg_val<uint32_t>(6);
    // uint32_t in_h = get_arg_val<uint32_t>(7);
    uint32_t full_nimages = get_arg_val<uint32_t>(8);
    uint32_t partial_bottom_image_nrows = get_arg_val<uint32_t>(9);
    uint32_t partial_last_row_nbytes = get_arg_val<uint32_t>(10);
    uint32_t halo_for_left_left_nsticks = get_arg_val<uint32_t>(11);
    uint32_t halo_for_left_nsticks = get_arg_val<uint32_t>(12);
    uint32_t halo_for_right_nsticks = get_arg_val<uint32_t>(13);
    uint32_t halo_for_right_right_nsticks = get_arg_val<uint32_t>(14);

    uint32_t local_in_stick_start = get_arg_val<uint32_t>(15);
    uint32_t local_in_stick_end = get_arg_val<uint32_t>(16);
    uint32_t in_nsticks_per_batch = get_arg_val<uint32_t>(17);
    uint32_t in_nsticks_per_core = get_arg_val<uint32_t>(18);

    uint32_t has_left = get_arg_val<uint32_t>(19);
    uint32_t left_noc_x = get_arg_val<uint32_t>(20);
    uint32_t left_noc_y = get_arg_val<uint32_t>(21);
    uint32_t has_right = get_arg_val<uint32_t>(22);
    uint32_t right_noc_x = get_arg_val<uint32_t>(23);
    uint32_t right_noc_y = get_arg_val<uint32_t>(24);
    uint32_t has_left_left = get_arg_val<uint32_t>(25);
    uint32_t left_left_noc_x = get_arg_val<uint32_t>(26);
    uint32_t left_left_noc_y = get_arg_val<uint32_t>(27);
    uint32_t has_right_right = get_arg_val<uint32_t>(28);
    uint32_t right_right_noc_x = get_arg_val<uint32_t>(29);
    uint32_t right_right_noc_y = get_arg_val<uint32_t>(30);

    // uint32_t stick_nbytes = get_arg_val<uint32_t>(31);   // size of 1 stick (in_c_nbytes)

    // nsticks to push to left left neighbor core
    uint32_t left_left_core_nsticks = get_arg_val<uint32_t>(32);
    // total nbytes to push to left neighbor core (data + padding)
    uint32_t left_core_nbytes = get_arg_val<uint32_t>(33);
    // total nbytes to push to right neighbor core (data + padding)
    uint32_t right_core_nbytes = get_arg_val<uint32_t>(34);
    // nsticks to push to right right neighbor core
    uint32_t right_right_core_nsticks = get_arg_val<uint32_t>(35);

    // offset on left left neighbor core for its right right halo
    uint32_t left_left_core_halo_offset = get_arg_val<uint32_t>(36);
    // offset on left neighbor core for its right halo
    uint32_t left_core_halo_offset = get_arg_val<uint32_t>(37);
    // offset on right neighbor core for its left halo
    uint32_t right_core_halo_offset = get_arg_val<uint32_t>(38);
    // offset on right right neighbor core for its left left halo
    uint32_t right_right_core_halo_offset = get_arg_val<uint32_t>(39);

    // padding (2) index offset in the halo going to left neighbors
    uint32_t left_going_halo_pad_i_offset = get_arg_val<uint32_t>(40);
    // padding (2) index offset in the halo going to right neighbors
    uint32_t right_going_halo_pad_i_offset = get_arg_val<uint32_t>(41);

    uint32_t partial_first_row_skip = get_arg_val<uint32_t>(42);
    uint32_t partial_top_image_skip = get_arg_val<uint32_t>(43);
    uint32_t full_image_skip = get_arg_val<uint32_t>(44);

    uint32_t initial_pad_nsticks = get_arg_val<uint32_t>(45);

    // 1. (partial first row width + pad_w)
    // 2. (out_w + pad_w * 2) * (num full rows partial top image)
    // 3. (out_w + pad_w * 2) * (pad_h + out_h) * num full images
    // 4. (out_w + pad_w * 2) * (pad_h + num full rows partial bottom image)
    // 5. (partial last row width + pad_w)

    // DPRINT << "partial_first_row_nsticks: " << partial_first_row_nsticks << ENDL();
    // DPRINT << "partial_first_row_skip: " << partial_first_row_skip << ENDL();
    // DPRINT << "partial_top_image_nrows: " << partial_top_image_nrows << ENDL();
    // DPRINT << "partial_top_image_skip: " << partial_top_image_skip << ENDL();
    // DPRINT << "full_nimages: " << full_nimages << ENDL();
    // DPRINT << "full_nimages_skip: " << full_image_skip << ENDL();
    // DPRINT << "partial_bottom_image_nrows: " << partial_bottom_image_nrows << ENDL();
    // DPRINT << "partial_last_row_nsticks: " << partial_last_row_nsticks << ENDL();
    // DPRINT << "initial_pad_nsticks: " << initial_pad_nsticks << ENDL();

    // DPRINT << "0" << ENDL();

    uint32_t padding_addr = get_read_ptr(pad_cb_id);
    uint64_t padding_noc_addr = get_noc_addr(padding_addr);

    cb_wait_front(in_cb_id, in_nsticks);

    uint32_t in_l1_addr = get_read_ptr(in_cb_id);
    uint32_t out_base_l1_addr = get_write_ptr(out_cb_id);

    // DPRINT << "==== INPUT:" << ENDL();
    // print_sticks(in_l1_addr, 0, 128, 64);

    constexpr uint32_t halo_nsticks = (in_w + 2 * pad_w) * pad_h + 1; // + 1 is (window_w / 2)

    // DPRINT << "1" << ENDL();

    uint32_t curr_out_l1_addr = out_base_l1_addr;
    uint32_t local_sticks_out_l1_addr = curr_out_l1_addr;   // excluding any left and right halo

    // section 0
    for (uint32_t i = 0; i < initial_pad_nsticks; ++ i) {
        // this is the beginning of a new image, insert padding instead of halo
        noc_async_read_one_packet(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
        curr_out_l1_addr += stick_nbytes;
    }
    if (initial_pad_nsticks == 0) {
        // set space aside in the beginning for my left halo, which will be pushed from my left neighbor
        curr_out_l1_addr += halo_nsticks * stick_nbytes;
        local_sticks_out_l1_addr = curr_out_l1_addr;
    }

    // section 1
    uint64_t curr_in_l1_noc_addr;

    uint32_t curr_in_l1_addr = in_l1_addr;
    curr_in_l1_noc_addr = get_noc_addr(curr_in_l1_addr);
    // best perf acheived using noc_asyc_read API
    noc_async_read(curr_in_l1_noc_addr, curr_out_l1_addr, partial_first_row_nbytes);
    curr_in_l1_addr += partial_first_row_nbytes;
    curr_out_l1_addr += partial_first_row_nbytes;

    // insert padding sticks
    for (uint32_t j = 0; j < partial_first_row_skip; ++ j) {
        noc_async_read_one_packet(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
        curr_out_l1_addr += stick_nbytes;
    }

    // DPRINT << "2" << ENDL();

    // section 2
    for (uint32_t i = 0; i < partial_top_image_nrows; ++ i) {
        // copy data sticks for full row in one shot
        uint64_t noc_addr = get_noc_addr(curr_in_l1_addr);
        noc_async_read(noc_addr, curr_out_l1_addr, stick_nbytes*in_w);
        curr_in_l1_addr += stick_nbytes*in_w;
        curr_out_l1_addr += stick_nbytes*in_w;

        // padding sticks on the right, left edge
        for (uint32_t j = 0; j < 2 * pad_w; ++ j) {
            noc_async_read_one_packet(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_out_l1_addr += stick_nbytes;
        }
    }
    for (uint32_t j = 0; j < partial_top_image_skip; ++ j) {
        noc_async_read_one_packet(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
        curr_out_l1_addr += stick_nbytes;
    }

    // DPRINT << "3" << ENDL();

    // section 3
    for (uint32_t n = 0; n < full_nimages; ++ n) {
        // full image rows
        for (uint32_t i = 0; i < in_h; ++ i) {
            // copy data sticks for full row, in one shot
            curr_in_l1_noc_addr = get_noc_addr(curr_in_l1_addr);
            noc_async_read(curr_in_l1_noc_addr, curr_out_l1_addr, stick_nbytes*in_w);
            curr_in_l1_addr += stick_nbytes*in_w;
            curr_out_l1_addr += stick_nbytes*in_w;

            // padding sticks after each row
            for (uint32_t j = 0; j < 2 * pad_w; ++ j) {
                noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
                curr_out_l1_addr += stick_nbytes;
            }
        }
        // padding after full image
        for (uint32_t i = 0; i < full_image_skip; ++ i) {
            noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_out_l1_addr += stick_nbytes;
        }
    }

    // noc_async_read_barrier();   // for debug: TODO remove
    // DPRINT << "4" << ENDL();

    // section 4
    for (uint32_t i = 0; i < partial_bottom_image_nrows; ++ i) {
        // copy data sticks for full row, in one shot
        curr_in_l1_noc_addr = get_noc_addr(curr_in_l1_addr);
        noc_async_read(curr_in_l1_noc_addr, curr_out_l1_addr, stick_nbytes*in_w);
        curr_in_l1_addr += stick_nbytes*in_w;
        curr_out_l1_addr += stick_nbytes*in_w;

        // padding sticks
        for (uint32_t j = 0; j < 2 * pad_w; ++ j) {
            noc_async_read(padding_noc_addr, curr_out_l1_addr, stick_nbytes);
            curr_out_l1_addr += stick_nbytes;
        }
    }

    // section 5
    // partial row sticks
    curr_in_l1_noc_addr = get_noc_addr(curr_in_l1_addr);
    noc_async_read(curr_in_l1_noc_addr, curr_out_l1_addr, partial_last_row_nbytes);
    curr_in_l1_addr += partial_last_row_nbytes;
    curr_out_l1_addr += partial_last_row_nbytes;

    noc_async_read_barrier();   // make sure everything is read into output locations before sending halos

    // section B (push halo to right and right right neighbors cores)
    curr_in_l1_addr = curr_in_l1_addr - (in_w + 1) * stick_nbytes;  // rewind by (out_w + 1)
    curr_out_l1_addr = curr_out_l1_addr - halo_nsticks * stick_nbytes;  // rewind by 1 halo worth sticks, which needs to be pushed to right neighbors
    uint32_t right_i = 0;

    if (has_right) {
        // DPRINT << "HALO TO R = " << right_core_nsticks << " (" << right_noc_x << "," << right_noc_y << "): ";
        uint32_t out_l1_addr_right = out_base_l1_addr + right_core_halo_offset;
        // push sticks to right neighbor
        uint64_t noc_addr = get_noc_addr(right_noc_x, right_noc_y, out_l1_addr_right);
        noc_async_write(curr_out_l1_addr, noc_addr, right_core_nbytes);
        out_l1_addr_right += right_core_nbytes;
        curr_out_l1_addr += right_core_nbytes;
        // DPRINT << ENDL();
    }

    // if (has_right_right) {
    //     // DPRINT << "HALO TO RR = " << right_right_core_nsticks << " (" << right_right_noc_x << "," << right_right_noc_y << "): ";
    //     uint32_t out_l1_addr_right_right = out_base_l1_addr + right_right_core_halo_offset;
    //     // push sticks to right right neighbor
    //     for (uint32_t i = 0; i < right_right_core_nsticks; ++ i, ++ right_i) {
    //         // if (right_i == right_going_halo_pad_i_offset) {
    //         //     // send padding sticks (2 * pad_w)
    //         //     // TODO: may be the remote core can fill padding locally for its halo ...
    //         //     for (uint32_t j = 0; j < 2 * pad_w; ++ j) {
    //         //         uint64_t noc_addr = get_noc_addr(right_right_noc_x, right_right_noc_y, out_l1_addr_right_right);
    //         //         noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
    //         //         out_l1_addr_right_right += stick_nbytes;
    //         //     }
    //         //     // increament the nsticks to offset due to padding
    //         //     right_right_core_nsticks += 2 * pad_w;
    //         //     // if (i < 5) DPRINT << "P P ";
    //         // } else {
    //             uint64_t noc_addr = get_noc_addr(right_right_noc_x, right_right_noc_y, out_l1_addr_right_right);
    //             noc_async_write(curr_out_l1_addr, noc_addr, stick_nbytes);
    //             out_l1_addr_right_right += stick_nbytes;
    //             curr_out_l1_addr += stick_nbytes;
    //             // if (i < 5) DPRINT << curr_in_l1_addr << " ";
    //         // }
    //     }
    //     // DPRINT << ENDL();
    // }

    // // DPRINT << "7" << ENDL();

    // // section A (push halo to left and left left neighbors)
    // curr_in_l1_addr = in_l1_addr;   // reset to the base
    curr_out_l1_addr = local_sticks_out_l1_addr;    // reset to beginning of local data
    uint32_t left_i = 0;
    // if (has_left_left) {
    //     // DPRINT << "HALO TO LL = " << left_left_core_nsticks << " (" << left_left_noc_x << "," << left_left_noc_y << "): " << ENDL();
    //     // these sticks belong to the right right halo of the left left neighbor
    //     uint32_t out_l1_addr_left_left = out_base_l1_addr + left_left_core_halo_offset;
    //     // push sticks to left left neighbor
    //     for (uint32_t i = 0; i < left_left_core_nsticks; ++ i, ++ left_i) {
    //         if (left_i == left_going_halo_pad_i_offset) {
    //             // send padding sticks (2 * pad_w)
    //             // TODO: may be the remote core can fill padding locally for its halo ...
    //             for (uint32_t j = 0; j < 2 * pad_w; ++ j) {
    //                 uint64_t noc_addr = get_noc_addr(left_left_noc_x, left_left_noc_y, out_l1_addr_left_left);
    //                 noc_async_write(pad_val_buffer_l1_addr, noc_addr, stick_nbytes);
    //                 out_l1_addr_left_left += stick_nbytes;
    //             }
    //             // increament the nsticks to offset due to padding
    //             left_left_core_nsticks += 2 * pad_w;
    //             // if (i < 5) DPRINT << "P P ";
    //         } else {
    //             uint64_t noc_addr = get_noc_addr(left_left_noc_x, left_left_noc_y, out_l1_addr_left_left);
    //             noc_async_write(curr_in_l1_addr, noc_addr, stick_nbytes);
    //             out_l1_addr_left_left += stick_nbytes;
    //             curr_in_l1_addr += stick_nbytes;
    //             // if (i < 5) DPRINT << curr_in_l1_addr << " ";
    //         }
    //     }
    //     // DPRINT << ENDL();
    // }

    if (has_left) {
        // these sticks belong to the right halo of the left neighbor
        uint32_t out_l1_addr_left = out_base_l1_addr + left_core_halo_offset;
        uint64_t noc_addr = get_noc_addr(left_noc_x, left_noc_y, out_l1_addr_left);
        noc_async_write(curr_out_l1_addr, noc_addr, left_core_nbytes);
        out_l1_addr_left += left_core_nbytes;
        curr_out_l1_addr += left_core_nbytes;
    }

    noc_async_write_barrier();
}
