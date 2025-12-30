// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

/**
 * Unified Kernel API
 *
 * Provides MPI-like primitives for cross-core communication and synchronization.
 *
 * For local-only unified kernels (3-way split), include "unified_common.h" instead,
 * which provides processor-aware read_tile() and write_tile() macros.
 *
 * This header provides multicast/unicast primitives that work with role defines.
 */

// Role constants (for multicast/unicast operations)
#define ROLE_MCAST_SENDER 3
#define ROLE_MCAST_RECEIVER 4

// Default MY_ROLE if not set by defines
#ifndef MY_ROLE
#define MY_ROLE ROLE_MCAST_RECEIVER  // Default for multicast
#endif

// INIT_ARGUMENTS macro - placeholder for argument initialization
// Users can define this themselves or use default behavior
#ifndef INIT_ARGUMENTS
#define INIT_ARGUMENTS() \
    do {                 \
    } while (0)
#endif

// Multicast primitives
#if defined(MCAST_SENDER) && MCAST_SENDER == 1

// Helper macros for multicast sender (similar to DEFINE_PERSISTENT_MCAST_SENDER_VARS)
#define DEFINE_MCAST_VARS(group)                                                                                    \
    constexpr uint32_t group##_mcast_dest_noc_start_x = get_named_compile_time_arg_val(#group "_dest_noc_start_x"); \
    constexpr uint32_t group##_mcast_dest_noc_start_y = get_named_compile_time_arg_val(#group "_dest_noc_start_y"); \
    constexpr uint32_t group##_mcast_dest_noc_end_x = get_named_compile_time_arg_val(#group "_dest_noc_end_x");     \
    constexpr uint32_t group##_mcast_dest_noc_end_y = get_named_compile_time_arg_val(#group "_dest_noc_end_y");     \
    constexpr uint32_t group##_mcast_num_cores = get_named_compile_time_arg_val(#group "_num_cores");               \
    constexpr bool group##_loopback = get_named_compile_time_arg_val(#group "_loopback");                           \
    constexpr bool group##_is_part_of_receiver_grid =                                                               \
        get_named_compile_time_arg_val(#group "_is_part_of_receiver_grid");                                         \
    uint32_t group##_data_sender_semaphore_addr =                                                                   \
        get_semaphore(get_named_compile_time_arg_val(#group "_data_sender_semaphore"));                             \
    uint32_t group##_data_receiver_semaphore_addr =                                                                 \
        get_semaphore(get_named_compile_time_arg_val(#group "_data_receiver_semaphore"));                           \
    const uint64_t group##_noc_coord = get_noc_multicast_addr<noc_index>(                                           \
        group##_mcast_dest_noc_start_x,                                                                             \
        group##_mcast_dest_noc_start_y,                                                                             \
        group##_mcast_dest_noc_end_x,                                                                               \
        group##_mcast_dest_noc_end_y,                                                                               \
        0);                                                                                                         \
    uint64_t group##_mcast_flag_noc_addr = group##_noc_coord | (uint64_t)(group##_data_receiver_semaphore_addr);    \
    volatile tt_l1_ptr uint32_t* group##_data_sender_semaphore_addr_ptr =                                           \
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(group##_data_sender_semaphore_addr);

// Multicast tile primitive (sender side)
// Expands to: wait for receivers ready, multicast data, signal completion
#define mcast_tile(tile_addr, tile_size, group)                                                                \
    do {                                                                                                       \
        DEFINE_MCAST_VARS(group);                                                                              \
        uint64_t group##_mcast_data_noc_addr = group##_noc_coord | (uint64_t)(tile_addr);                      \
        /* Wait for all receivers to be ready */                                                               \
        noc_semaphore_wait(group##_data_sender_semaphore_addr_ptr, group##_mcast_num_cores);                   \
        noc_semaphore_set(group##_data_sender_semaphore_addr_ptr, 0);                                          \
        /* Multicast the data */                                                                               \
        noc_async_write_multicast(tile_addr, group##_mcast_data_noc_addr, tile_size, group##_mcast_num_cores); \
        /* Signal receivers that data is ready */                                                              \
        noc_semaphore_set_multicast(                                                                           \
            group##_data_receiver_semaphore_addr, group##_mcast_flag_noc_addr, group##_mcast_num_cores);       \
    } while (0)

#endif  // MCAST_SENDER

// Receive tile primitive (receiver side)
#if defined(MCAST_RECEIVER) && MCAST_RECEIVER == 1

#define receive_tile(buffer)                                                                \
    ({                                                                                      \
        uint32_t cb_id = GET_CB_ID(buffer);                                                 \
        uint32_t data_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(0)); \
        volatile tt_l1_ptr uint32_t* data_receiver_semaphore_addr_ptr =                     \
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data_receiver_semaphore_addr);   \
        /* Wait for sender to signal data is ready */                                       \
        noc_semaphore_wait(data_receiver_semaphore_addr_ptr, VALID);                        \
        noc_semaphore_set(data_receiver_semaphore_addr_ptr, INVALID);                       \
        /* Data is now in local L1, read from CB */                                         \
        cb_reserve_back(cb_id, 1);                                                          \
        uint32_t l1_addr = get_write_ptr(cb_id);                                            \
        cb_push_back(cb_id, 1);                                                             \
        l1_addr;                                                                            \
    })

#endif  // MCAST_RECEIVER

// MPI-style collective primitives
// These macros hide the sender/receiver branching behind a single symmetric call,
// resolved at compile time with zero runtime overhead.

// Broadcast tile: sender reads from src and multicasts, receivers receive and write to dst
// Usage: bcast_tile(group, src_buffer, dst_buffer, tile_idx)
//   - group: multicast group name (e.g., "receivers")
//   - src_buffer: source buffer name (e.g., "in0")
//   - dst_buffer: destination buffer name (e.g., "out")
//   - tile_idx: tile index to read/write
//
// Note: This assumes src_buffer has the tile already read into its CB (via read_tile or similar).
// For receivers, receive_tile handles receiving and putting the tile in dst_buffer CB.
// The receiver then writes from the CB to the output tensor using noc_async_write_tile.
#define bcast_tile(group, src_buffer, dst_buffer, tile_idx)                             \
    do {                                                                                \
        if constexpr (MY_ROLE == ROLE_MCAST_SENDER) {                                   \
            /* Sender: get tile from source CB and multicast to receivers */            \
            uint32_t cb_id = GET_CB_ID(src_buffer);                                     \
            cb_wait_front(cb_id, 1);                                                    \
            uint32_t tile_addr = get_read_ptr(cb_id);                                   \
            uint32_t tile_size = get_tile_size(cb_id);                                  \
            mcast_tile(tile_addr, tile_size, group);                                    \
            cb_pop_front(cb_id, 1);                                                     \
        } else if constexpr (MY_ROLE == ROLE_MCAST_RECEIVER) {                          \
            /* Receiver: receive tile (puts it in dst_buffer CB) and write to output */ \
            uint32_t tile_addr = receive_tile(dst_buffer);                              \
            /* Tile is now in dst_buffer CB at tile_addr, write it to output tensor */  \
            uint32_t dst_cb_id = GET_CB_ID(dst_buffer);                                 \
            cb_wait_front(dst_cb_id, 1);                                                \
            noc_async_write_tile(tile_idx, dst_buffer, tile_addr);                      \
            noc_async_write_barrier();                                                  \
            cb_pop_front(dst_cb_id, 1);                                                 \
        }                                                                               \
    } while (0)

// Barrier primitive (simplified - can be extended)
#define barrier(group)                                                   \
    do {                                                                 \
        /* Basic barrier implementation - users may need to customize */ \
        noc_async_write_barrier();                                       \
    } while (0)

// Signal and wait primitives for semaphores
#define send_signal(semaphore_name, value)                                                               \
    do {                                                                                                 \
        uint32_t sem_addr = get_semaphore(semaphore_name##_semaphore_id);                                \
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr); \
        *sem_ptr = value;                                                                                \
    } while (0)

#define wait_signal(semaphore_name, expected_value)                                                      \
    do {                                                                                                 \
        uint32_t sem_addr = get_semaphore(semaphore_name##_semaphore_id);                                \
        volatile tt_l1_ptr uint32_t* sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr); \
        noc_semaphore_wait(sem_ptr, expected_value);                                                     \
    } while (0)
