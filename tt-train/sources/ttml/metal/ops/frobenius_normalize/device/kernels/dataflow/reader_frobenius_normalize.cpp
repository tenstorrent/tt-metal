// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    // ---- Runtime args ----
    uint32_t rt = 0;
    uint32_t input_addr = get_arg_val<uint32_t>(rt++);
    uint32_t num_tiles = get_arg_val<uint32_t>(rt++);
    uint32_t start_tile_id = get_arg_val<uint32_t>(rt++);
    uint32_t sem_id = get_arg_val<uint32_t>(rt++);
    // Role flags (computed by program factory based on active grid topology)
    uint32_t do_row_receive = get_arg_val<uint32_t>(rt++);  // receive from right neighbor
    uint32_t do_row_send = get_arg_val<uint32_t>(rt++);      // send to left neighbor
    uint32_t do_col_receive = get_arg_val<uint32_t>(rt++);   // receive from below
    uint32_t do_col_send = get_arg_val<uint32_t>(rt++);      // send to above
    uint32_t is_origin = get_arg_val<uint32_t>(rt++);
    uint32_t do_bcast_col_fwd = get_arg_val<uint32_t>(rt++); // forward norm down in column
    uint32_t do_bcast_row_fwd = get_arg_val<uint32_t>(rt++); // forward norm right in row
    // Physical coords of neighbors (0 if not applicable)
    uint32_t left_phys_x = get_arg_val<uint32_t>(rt++);
    uint32_t left_phys_y = get_arg_val<uint32_t>(rt++);
    uint32_t up_phys_x = get_arg_val<uint32_t>(rt++);
    uint32_t up_phys_y = get_arg_val<uint32_t>(rt++);
    uint32_t right_phys_x = get_arg_val<uint32_t>(rt++);
    uint32_t right_phys_y = get_arg_val<uint32_t>(rt++);
    uint32_t down_phys_x = get_arg_val<uint32_t>(rt++);
    uint32_t down_phys_y = get_arg_val<uint32_t>(rt++);

    // ---- Compile-time args ----
    constexpr uint32_t packed_eps = get_compile_time_arg_val(0);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_sq_acc = tt::CBIndex::c_1;
    constexpr uint32_t cb_scalar = tt::CBIndex::c_2;
    constexpr uint32_t cb_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_norm = tt::CBIndex::c_4;
    const uint32_t tile_bytes = get_tile_size(cb_input);
    const uint32_t fp32_tile_bytes = get_tile_size(cb_scalar);

    constexpr auto input_args = TensorAccessorArgs<1>();
    const auto input_addr_gen = TensorAccessor(input_args, input_addr, tile_bytes);

    uint32_t sem_addr = get_semaphore(sem_id);
    volatile tt_l1_ptr uint32_t* sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    bool is_row_leader = (!do_row_send);  // gx == 0

    // =========================================================================
    // Pass 1: Stream input tiles to compute for square+accumulate
    // =========================================================================
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_input, 1);
        uint32_t l1_addr = get_write_ptr(cb_input);
        noc_async_read_page(start_tile_id + i, input_addr_gen, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_input, 1);
    }

    // =========================================================================
    // Row chain reduction (right -> left)
    // =========================================================================
    if (do_row_receive) {
        noc_semaphore_wait(sem_ptr, 1);
        noc_semaphore_set(sem_ptr, 0);
        cb_push_back(cb_recv, 1);
    }

    if (do_row_send) {
        cb_wait_front(cb_scalar, 1);
        uint32_t src_l1 = get_read_ptr(cb_scalar);
        uint32_t dst_cb_recv_l1 = get_write_ptr(cb_recv);
        uint64_t dst_noc_addr = get_noc_addr(left_phys_x, left_phys_y, dst_cb_recv_l1);
        noc_async_write(src_l1, dst_noc_addr, fp32_tile_bytes);
        noc_async_write_barrier();
        uint64_t dst_sem_noc = get_noc_addr(left_phys_x, left_phys_y, sem_addr);
        noc_semaphore_inc(dst_sem_noc, 1);
        cb_pop_front(cb_scalar, 1);
    }

    // =========================================================================
    // Column chain reduction (bottom -> top, row leaders only)
    // =========================================================================
    if (do_col_receive) {
        noc_semaphore_wait(sem_ptr, 1);
        noc_semaphore_set(sem_ptr, 0);
        cb_push_back(cb_recv, 1);
    }

    if (do_col_send) {
        cb_wait_front(cb_scalar, 1);
        uint32_t src_l1 = get_read_ptr(cb_scalar);
        uint32_t dst_cb_recv_l1 = get_write_ptr(cb_recv);
        uint64_t dst_noc_addr = get_noc_addr(up_phys_x, up_phys_y, dst_cb_recv_l1);
        noc_async_write(src_l1, dst_noc_addr, fp32_tile_bytes);
        noc_async_write_barrier();
        uint64_t dst_sem_noc = get_noc_addr(up_phys_x, up_phys_y, sem_addr);
        noc_semaphore_inc(dst_sem_noc, 1);
        cb_pop_front(cb_scalar, 1);
    }

    // =========================================================================
    // Origin: generate eps tile for compute (in cb_sq_acc, repurposed)
    // =========================================================================
    if (is_origin) {
        // Generate eps tile as FP32 (cb_sq_acc is FP32 format)
        // packed_eps is two packed BF16 values; we need FP32 bit pattern instead.
        // Convert BF16 to FP32 by shifting left 16 bits.
        uint32_t eps_fp32_bits = (packed_eps >> 16) << 16;  // Extract upper BF16, shift to FP32
        generate_tile_with_uint32_value(cb_sq_acc, eps_fp32_bits);
    }

    // =========================================================================
    // Broadcast: unicast chain — column (top->bottom), then row (left->right)
    // =========================================================================

    // Column broadcast (row leaders only)
    if (is_row_leader) {
        if (is_origin) {
            cb_wait_front(cb_norm, 1);
        } else {
            noc_semaphore_wait(sem_ptr, 1);
            noc_semaphore_set(sem_ptr, 0);
            cb_push_back(cb_norm, 1);
        }

        if (do_bcast_col_fwd) {
            uint32_t src_l1 = get_read_ptr(cb_norm);
            uint32_t dst_cb_norm_l1 = get_write_ptr(cb_norm);
            uint64_t dst_noc_addr = get_noc_addr(down_phys_x, down_phys_y, dst_cb_norm_l1);
            noc_async_write(src_l1, dst_noc_addr, fp32_tile_bytes);
            noc_async_write_barrier();
            uint64_t dst_sem_noc = get_noc_addr(down_phys_x, down_phys_y, sem_addr);
            noc_semaphore_inc(dst_sem_noc, 1);
        }
    }

    // Row broadcast
    if (is_row_leader) {
        if (do_bcast_row_fwd) {
            uint32_t src_l1 = get_read_ptr(cb_norm);
            uint32_t dst_cb_norm_l1 = get_write_ptr(cb_norm);
            uint64_t dst_noc_addr = get_noc_addr(right_phys_x, right_phys_y, dst_cb_norm_l1);
            noc_async_write(src_l1, dst_noc_addr, fp32_tile_bytes);
            noc_async_write_barrier();
            uint64_t dst_sem_noc = get_noc_addr(right_phys_x, right_phys_y, sem_addr);
            noc_semaphore_inc(dst_sem_noc, 1);
        }
    } else {
        noc_semaphore_wait(sem_ptr, 1);
        noc_semaphore_set(sem_ptr, 0);
        cb_push_back(cb_norm, 1);

        if (do_bcast_row_fwd) {
            uint32_t src_l1 = get_read_ptr(cb_norm);
            uint32_t dst_cb_norm_l1 = get_write_ptr(cb_norm);
            uint64_t dst_noc_addr = get_noc_addr(right_phys_x, right_phys_y, dst_cb_norm_l1);
            noc_async_write(src_l1, dst_noc_addr, fp32_tile_bytes);
            noc_async_write_barrier();
            uint64_t dst_sem_noc = get_noc_addr(right_phys_x, right_phys_y, sem_addr);
            noc_semaphore_inc(dst_sem_noc, 1);
        }
    }

    // =========================================================================
    // Pass 2: Re-read input tiles for normalization
    // =========================================================================
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_input, 1);
        uint32_t l1_addr = get_write_ptr(cb_input);
        noc_async_read_page(start_tile_id + i, input_addr_gen, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_input, 1);
    }
}
