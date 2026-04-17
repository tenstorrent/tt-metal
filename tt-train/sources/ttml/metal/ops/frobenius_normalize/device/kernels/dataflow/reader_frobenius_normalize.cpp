// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tools/profiler/kernel_profiler.hpp>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

constexpr uint32_t cb_input = tt::CBIndex::c_0;
constexpr uint32_t cb_recv = tt::CBIndex::c_3;
constexpr uint32_t cb_norm = tt::CBIndex::c_4;
constexpr uint32_t cb_sq_partial = tt::CBIndex::c_7;

constexpr uint32_t origin_phys_x = get_compile_time_arg_val(0);
constexpr uint32_t origin_phys_y = get_compile_time_arg_val(1);
constexpr uint32_t num_cores = get_compile_time_arg_val(2);
constexpr uint32_t mcast_start_x = get_compile_time_arg_val(3);
constexpr uint32_t mcast_start_y = get_compile_time_arg_val(4);
constexpr uint32_t mcast_end_x = get_compile_time_arg_val(5);
constexpr uint32_t mcast_end_y = get_compile_time_arg_val(6);
constexpr uint32_t num_active_cores = get_compile_time_arg_val(7);
constexpr auto input_args = TensorAccessorArgs<8>();

void kernel_main() {
    uint32_t runtime_args_counter = 0;
    uint32_t input_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_tiles = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t start_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t reduction_sem_id = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t core_index = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t bcast_sem_id = get_arg_val<uint32_t>(runtime_args_counter++);

    const uint32_t tile_bytes = get_tile_size(cb_input);
    const auto input_addr_gen = TensorAccessor(input_args, input_addr, tile_bytes);

    uint32_t reduction_sem_addr = get_semaphore(reduction_sem_id);
    volatile tt_l1_ptr uint32_t* reduction_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reduction_sem_addr);

    uint32_t bcast_sem_addr = get_semaphore(bcast_sem_id);
    volatile tt_l1_ptr uint32_t* bcast_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bcast_sem_addr);

    const uint32_t norm_scalar_addr = get_write_ptr(cb_norm);
    volatile tt_l1_ptr uint32_t* norm_scalar_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(norm_scalar_addr);

    // cb_recv base address — used as the reduction buffer on origin.
    const uint32_t cb_recv_base = get_write_ptr(cb_recv);

    // Origin: zero-fill cb_recv tile BEFORE Pass 1 so it completes before any
    // non-origin core can write it's partial
#ifdef IS_ORIGIN
    {
        const uint32_t fp32_tile_bytes = get_tile_size(cb_recv);
        uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
        uint32_t zero_dst = cb_recv_base;
        for (uint32_t z = 0; z < fp32_tile_bytes / MEM_ZEROS_SIZE; ++z) {
            noc_async_read(zeros_noc_addr, zero_dst, MEM_ZEROS_SIZE);
            zero_dst += MEM_ZEROS_SIZE;
        }
        noc_async_read_barrier();
    }
#endif

    // =========================================================================
    // Pass 1: Stream input tiles to compute for square+accumulate.
    // =========================================================================
    {
        DeviceZoneScopedN("READER-PASS1-DRAM");
        constexpr uint32_t block_size = 4;
        for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx += block_size) {
            uint32_t current = std::min(block_size, num_tiles - tile_idx);
            read_tiles_by_row(cb_input, input_addr_gen, start_tile_id + tile_idx, current, tile_bytes, block_size);
        }
    }

    // =========================================================================
    // All-to-origin reduction: each core writes its partial sum_sq (16 bytes,
    // with bytes 4-15 zeroed) to origin's cb_recv L1 at core_index * 16.
    // Origin zeros cb_recv at kernel start, waits for semaphore, then pushes
    // the tile to compute for sfpu_reduce
    // Stride has to be 16 bytes for some reason
    // =========================================================================
    {
        DeviceZoneScopedN("READER-REDUCE-BCAST");

        // Read partial from compute and zero padding bytes for clean sfpu_reduce
        cb_wait_front(cb_sq_partial, 1);
        uint32_t src_l1 = get_read_ptr(cb_sq_partial);
        // Zero bytes 4-15 so the 16-byte NOC write is [scalar, 0, 0, 0]
        volatile tt_l1_ptr uint32_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_l1);
        src_ptr[1] = 0;
        src_ptr[2] = 0;
        src_ptr[3] = 0;

#ifdef IS_ORIGIN
        // Write own partial at core_index offset (local L1 write, no NOC)
        volatile tt_l1_ptr uint32_t* dst =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_recv_base + core_index * 16);
        dst[0] = src_ptr[0];
        dst[1] = 0;
        dst[2] = 0;
        dst[3] = 0;
        cb_pop_front(cb_sq_partial, 1);

        // Wait for all non-origin cores
        if constexpr (num_cores > 1) {
            noc_semaphore_wait(reduction_sem_ptr, num_cores - 1);
            noc_semaphore_set(reduction_sem_ptr, 0);
        }

        // Push the completed tile to cb_recv for compute sfpu_reduce
        cb_push_back(cb_recv, 1);
#else
        // NOC write [scalar, 0, 0, 0] to origin's cb_recv
        uint64_t dst_noc = get_noc_addr(origin_phys_x, origin_phys_y, cb_recv_base + core_index * 16);
        noc_async_write(src_l1, dst_noc, 16);
        noc_async_write_barrier();
        cb_pop_front(cb_sq_partial, 1);

        // Signal origin
        uint64_t origin_sem_noc = get_noc_addr(origin_phys_x, origin_phys_y, reduction_sem_addr);
        noc_semaphore_inc(origin_sem_noc, 1);
#endif

        // =====================================================================
        // Norm broadcast: origin extracts scalar, multicasts to all cores.
        // =====================================================================
#ifdef IS_ORIGIN
        cb_wait_front(cb_norm, 1);
        uint32_t norm_tile_l1 = get_read_ptr(cb_norm);
        uint32_t bcast_val = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(norm_tile_l1);
        *norm_scalar_ptr = bcast_val;
        cb_pop_front(cb_norm, 1);

        if constexpr (num_active_cores > 1) {
            uint64_t mcast_dst_addr =
                get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, norm_scalar_addr);
            noc_async_write_multicast_loopback_src(norm_scalar_addr, mcast_dst_addr, 4, num_active_cores);
            noc_async_write_barrier();

            *bcast_sem_ptr = 1;
            uint64_t mcast_sem_dst =
                get_noc_multicast_addr(mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, bcast_sem_addr);
            noc_semaphore_set_multicast_loopback_src(bcast_sem_addr, mcast_sem_dst, num_active_cores);
        } else {
            *bcast_sem_ptr = 1;
        }
#endif

        noc_semaphore_wait(bcast_sem_ptr, 1);
        noc_semaphore_set(bcast_sem_ptr, 0);

        uint32_t norm_val = *norm_scalar_ptr;
        generate_tile_with_uint32_value(cb_norm, norm_val);
    }

    // =========================================================================
    // Pass 2: Re-read input tiles for normalization (same pipelining as Pass 1)
    // =========================================================================
    {
        DeviceZoneScopedN("READER-PASS2-DRAM");
        constexpr uint32_t block_size = 4;
        for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx += block_size) {
            uint32_t current = std::min(block_size, num_tiles - tile_idx);
            read_tiles_by_row(cb_input, input_addr_gen, start_tile_id + tile_idx, current, tile_bytes, block_size);
        }
    }
}
