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
    uint32_t chain_sem_id = get_arg_val<uint32_t>(rt++);
    // Chain reduction flags
    uint32_t do_row_receive = get_arg_val<uint32_t>(rt++);
    uint32_t do_row_send = get_arg_val<uint32_t>(rt++);
    uint32_t do_col_receive = get_arg_val<uint32_t>(rt++);
    uint32_t do_col_send = get_arg_val<uint32_t>(rt++);
    uint32_t is_origin = get_arg_val<uint32_t>(rt++);
    // Chain neighbor coords (left for row send, up for col send)
    uint32_t left_phys_x = get_arg_val<uint32_t>(rt++);
    uint32_t left_phys_y = get_arg_val<uint32_t>(rt++);
    uint32_t up_phys_x = get_arg_val<uint32_t>(rt++);
    uint32_t up_phys_y = get_arg_val<uint32_t>(rt++);
    // Multicast broadcast args
    uint32_t bcast_sem_id = get_arg_val<uint32_t>(rt++);
    uint32_t norm_scalar_sem_id = get_arg_val<uint32_t>(rt++);
    uint32_t mcast_start_x = get_arg_val<uint32_t>(rt++);
    uint32_t mcast_start_y = get_arg_val<uint32_t>(rt++);
    uint32_t mcast_end_x = get_arg_val<uint32_t>(rt++);
    uint32_t mcast_end_y = get_arg_val<uint32_t>(rt++);
    uint32_t num_active_cores = get_arg_val<uint32_t>(rt++);

    // ---- Compile-time args ----
    // packed_eps removed — eps now passed as compute runtime arg

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_sq_acc = tt::CBIndex::c_1;
    constexpr uint32_t cb_scalar = tt::CBIndex::c_2;
    constexpr uint32_t cb_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_norm = tt::CBIndex::c_4;

    const uint32_t tile_bytes = get_tile_size(cb_input);
    const uint32_t fp32_tile_bytes = get_tile_size(cb_scalar);

    constexpr auto input_args = TensorAccessorArgs<0>();
    const auto input_addr_gen = TensorAccessor(input_args, input_addr, tile_bytes);

    uint32_t chain_sem_addr = get_semaphore(chain_sem_id);
    volatile tt_l1_ptr uint32_t* chain_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chain_sem_addr);

    uint32_t bcast_sem_addr = get_semaphore(bcast_sem_id);
    volatile tt_l1_ptr uint32_t* bcast_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bcast_sem_addr);

    // L1 address for the 4-byte FP32 norm scalar (shared across all cores via multicast)
    uint32_t norm_scalar_addr = get_semaphore(norm_scalar_sem_id);
    volatile tt_l1_ptr uint32_t* norm_scalar_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(norm_scalar_addr);

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
        noc_semaphore_wait(chain_sem_ptr, 1);
        noc_semaphore_set(chain_sem_ptr, 0);
        cb_push_back(cb_recv, 1);
    }

    if (do_row_send) {
        cb_wait_front(cb_scalar, 1);
        uint32_t src_l1 = get_read_ptr(cb_scalar);
        uint32_t dst_cb_recv_l1 = get_write_ptr(cb_recv);
        uint64_t dst_noc_addr = get_noc_addr(left_phys_x, left_phys_y, dst_cb_recv_l1);
        noc_async_write(src_l1, dst_noc_addr, fp32_tile_bytes);
        noc_async_write_barrier();
        uint64_t dst_sem_noc = get_noc_addr(left_phys_x, left_phys_y, chain_sem_addr);
        noc_semaphore_inc(dst_sem_noc, 1);
        cb_pop_front(cb_scalar, 1);
    }

    // =========================================================================
    // Column chain reduction (bottom -> top, row leaders only)
    // =========================================================================
    if (do_col_receive) {
        noc_semaphore_wait(chain_sem_ptr, 1);
        noc_semaphore_set(chain_sem_ptr, 0);
        cb_push_back(cb_recv, 1);
    }

    if (do_col_send) {
        cb_wait_front(cb_scalar, 1);
        uint32_t src_l1 = get_read_ptr(cb_scalar);
        uint32_t dst_cb_recv_l1 = get_write_ptr(cb_recv);
        uint64_t dst_noc_addr = get_noc_addr(up_phys_x, up_phys_y, dst_cb_recv_l1);
        noc_async_write(src_l1, dst_noc_addr, fp32_tile_bytes);
        noc_async_write_barrier();
        uint64_t dst_sem_noc = get_noc_addr(up_phys_x, up_phys_y, chain_sem_addr);
        noc_semaphore_inc(dst_sem_noc, 1);
        cb_pop_front(cb_scalar, 1);
    }

    // eps is now passed as a compute runtime arg — no tile generation needed

    // =========================================================================
    // Norm broadcast: origin extracts scalar, fills a full FP32 tile.
    // For multi-core: multicast 4-byte scalar to all cores.
    // =========================================================================
    if (is_origin) {
        // Wait for compute Phase 5 to pack 1/norm into cb_norm
        cb_wait_front(cb_norm, 1);

        // Extract the FP32 scalar at position (0,0) from the packed tile in L1
        uint32_t norm_tile_l1 = get_read_ptr(cb_norm);
        uint32_t norm_val = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(norm_tile_l1);

        // Store for multicast
        *norm_scalar_ptr = norm_val;

        // Done with the Phase 5 packed tile
        cb_pop_front(cb_norm, 1);

        if (num_active_cores > 1) {
            uint64_t mcast_dst_addr = get_noc_multicast_addr(
                mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, norm_scalar_addr);
            noc_async_write_multicast_loopback_src(norm_scalar_addr, mcast_dst_addr, 4, num_active_cores);
            noc_async_write_barrier();

            *bcast_sem_ptr = 1;
            uint64_t mcast_sem_dst = get_noc_multicast_addr(
                mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, bcast_sem_addr);
            noc_semaphore_set_multicast_loopback_src(bcast_sem_addr, mcast_sem_dst, num_active_cores);
        } else {
            *bcast_sem_ptr = 1;
        }
    }

    // All cores: wait for broadcast
    noc_semaphore_wait(bcast_sem_ptr, 1);
    noc_semaphore_set(bcast_sem_ptr, 0);

    // Fill a full FP32 tile with the scalar value (all positions = 1/norm)
    uint32_t norm_val = *norm_scalar_ptr;
    generate_tile_with_uint32_value(cb_norm, norm_val);

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
