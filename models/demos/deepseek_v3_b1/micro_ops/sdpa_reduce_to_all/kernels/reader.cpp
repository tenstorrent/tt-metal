// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// BUFFER LAYOUT AND PACKET ORDER:
// Buffer: [L_chunk0][L_chunk1]...[L_chunkN-1][MS]
// - L data at offset 0 (contiguous) - CB aliased for zero-copy
// - MS at offset total_l_bytes (end of buffer) - copied to cb_ms
//
// Packet arrival order: MS first, then L chunks
// Semaphore increments: sem >= 1 for MS, sem >= 2 for L_chunk0, sem >= (2+i) for L_chunk_i
//
// This allows:
// - L CB to be aliased at buffer base
// - Compute can start after MS arrives, stream L chunks during compute

#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>

using tt::data_movement::common::tt_memmove;

// =============================================================================
// Compile-time arguments
// =============================================================================
static constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
static constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
static constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(2);
static constexpr uint32_t cb_r1_neighbor_ms = get_compile_time_arg_val(3);
static constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(4);
static constexpr uint32_t cb_r2_neighbor_ms = get_compile_time_arg_val(5);

static constexpr uint32_t ms_tile_size_bytes = get_compile_time_arg_val(6);
static constexpr uint32_t l_chunk_size_bytes = get_compile_time_arg_val(7);
static constexpr uint32_t num_l_chunks = get_compile_time_arg_val(8);
static constexpr uint32_t tiles_per_l_chunk = get_compile_time_arg_val(9);

// Position CB compile-time args (added for conditional reduction)
static constexpr uint32_t cb_position = get_compile_time_arg_val(10);
static constexpr uint32_t position_enabled = get_compile_time_arg_val(11);

static constexpr uint32_t out_tiles = num_l_chunks * tiles_per_l_chunk;
static constexpr uint32_t total_l_bytes = num_l_chunks * l_chunk_size_bytes;

// Semaphore thresholds: MS = 1, L_chunk_i = 2 + i
static constexpr uint32_t MS_SEM_THRESHOLD = 1;
static constexpr uint32_t L_SEM_BASE_THRESHOLD = 2;

// =============================================================================
// Helper functions
// =============================================================================

/**
 * Prepare MS data for compute.
 * MS arrives first (sem >= 1), located at end of buffer.
 */
FORCE_INLINE void prepare_ms_for_compute(
    uint32_t cb_ms, volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t recv_buffer_addr) {
    cb_reserve_back(cb_ms, 1);

    // Wait for MS packet (sem >= 1)
    noc_semaphore_wait_min(sem_ptr, MS_SEM_THRESHOLD);

    // MS is at end of buffer (offset = total_l_bytes)
    tt_memmove<true, false, false, 0>(get_write_ptr(cb_ms), recv_buffer_addr + total_l_bytes, ms_tile_size_bytes);
    cb_push_back(cb_ms, 1);
}

/**
 * Prepare L chunk for compute.
 * L chunks arrive after MS (sem >= 2 + chunk_idx).
 * L CB is aliased to buffer base, so just push (zero-copy).
 *
 * @param l_chunk_idx L chunk index (0 to num_l_chunks-1)
 */
FORCE_INLINE void prepare_l_chunk_for_compute(
    uint32_t cb_l, volatile tt_l1_ptr uint32_t* sem_ptr, uint32_t l_chunk_idx) {
    cb_reserve_back(cb_l, tiles_per_l_chunk);

    // Wait for this L chunk (sem >= 2 + l_chunk_idx)
    noc_semaphore_wait_min(sem_ptr, L_SEM_BASE_THRESHOLD + l_chunk_idx);

    // L CB is aliased to buffer, just push (zero-copy)
    cb_push_back(cb_l, tiles_per_l_chunk);
}

/**
 * Prepare all data for one round (MS first, then L chunks).
 */
FORCE_INLINE void prepare_data_for_compute(
    uint32_t cb_l, uint32_t cb_ms, uint32_t sem_addr, uint32_t recv_buffer_addr) {
    volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    // MS first (sem >= 1)
    prepare_ms_for_compute(cb_ms, sem_ptr, recv_buffer_addr);

    // L chunks (sem >= 2, 3, 4, ...)
    for (uint32_t i = 0; i < num_l_chunks; i++) {
        prepare_l_chunk_for_compute(cb_l, sem_ptr, i);
    }
    noc_semaphore_set(sem_ptr, 0);
}

// =============================================================================
// Main kernel
// =============================================================================

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t r1_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r1_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t r2_recv_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    // =========================================================================
    // Push local input (aliased CBs, no copy needed)
    // =========================================================================
    cb_reserve_back(cb_local_l, out_tiles);
    cb_push_back(cb_local_l, out_tiles);

    cb_reserve_back(cb_local_ms, 1);
    cb_push_back(cb_local_ms, 1);

    bool r2_neighbor_r1_valid = true;
    bool r1_neighbor_valid = true;

    if constexpr (position_enabled) {
        // Get r2_neighbor_r1_neighbor_idx from runtime args
        uint32_t device_idx = get_arg_val<uint32_t>(arg_idx++);
        uint32_t r1_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
        uint32_t r2_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
        uint32_t r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(arg_idx++);
        cb_reserve_back(cb_position, 1);
        uint64_t position_cb_addr = get_write_ptr(cb_position);
        volatile tt_l1_ptr uint32_t* position_data_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(position_cb_addr);
        cb_push_back(cb_position, 1);

        // R2 neighbor's R1 result is valid if at least one device in that pair was valid
        uint32_t r1_neighbor_val = position_data_base[r1_neighbor_device_idx];
        uint32_t r2_neighbor_val = position_data_base[r2_neighbor_device_idx];
        uint32_t r2_neighbor_r1_neighbor_val = position_data_base[r2_neighbor_r1_neighbor_idx];
        r2_neighbor_r1_valid = (r2_neighbor_val != 0) || (r2_neighbor_r1_neighbor_val != 0);
        r1_neighbor_valid = (r1_neighbor_val != 0);
    }

    // =========================================================================
    // Prepare R1 neighbor data for compute
    // =========================================================================
    if (r1_neighbor_valid) {
        prepare_data_for_compute(cb_r1_neighbor_l, cb_r1_neighbor_ms, r1_neighbor_sem_addr, r1_recv_buffer_addr);
    } else {
        cb_reserve_back(cb_r1_neighbor_ms, 1);
        cb_push_back(cb_r1_neighbor_ms, 1);

        cb_reserve_back(cb_r1_neighbor_l, out_tiles);
        cb_push_back(cb_r1_neighbor_l, out_tiles);
    }
    // =========================================================================
    // Prepare R2 neighbor data for compute
    // =========================================================================

    // Only receive R2 data if R2 neighbor's R1 result is valid
    if (r2_neighbor_r1_valid) {
        prepare_data_for_compute(cb_r2_neighbor_l, cb_r2_neighbor_ms, r2_neighbor_sem_addr, r2_recv_buffer_addr);
    } else {
        cb_reserve_back(cb_r2_neighbor_ms, 1);
        cb_push_back(cb_r2_neighbor_ms, 1);

        cb_reserve_back(cb_r2_neighbor_l, out_tiles);
        cb_push_back(cb_r2_neighbor_l, out_tiles);
    }
}
