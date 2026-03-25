// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// SPSC (WH/BH)
constexpr int16_t DEBUG_RING_BUFFER_STARTING_INDEX = -1;
constexpr int DEBUG_RING_BUFFER_SPSC_ELEMENTS = 32;

struct debug_spsc_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_SPSC_ELEMENTS];
};

// MPSC (Quasar) - lock-free ring buffer for concurrent writes using 32-bit atomics
// Works on both tt-qsr64 (DM) and tt-qsr32 (TRISC)
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = 32;
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_MASK = DEBUG_RING_BUFFER_MPSC_ELEMENTS - 1;
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_THREAD_ID_SHIFT = 27;   // Upper 5 bits for thread_idx (0-31)
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_POS_MASK = 0x07FFFFFF;  // Lower 27 bits for position

struct debug_mpsc_ring_buf_slot_t {
    uint32_t data;
    uint32_t write_id;  // [31:27] thread_idx | [26:0] (pos+1)
};

struct debug_mpsc_ring_buf_msg_t {
    uint32_t head;
    uint8_t _pad[60];  // Pad to 64-byte cache line
    debug_mpsc_ring_buf_slot_t slots[DEBUG_RING_BUFFER_MPSC_ELEMENTS];
};

// Host-side MPSC helpers
inline uint32_t debug_ring_buffer_get_thread_idx(uint32_t write_id) {
    return write_id >> DEBUG_RING_BUFFER_MPSC_THREAD_ID_SHIFT;
}

inline uint32_t debug_ring_buffer_get_position(uint32_t write_id) { return write_id & DEBUG_RING_BUFFER_MPSC_POS_MASK; }

inline bool debug_ring_buffer_is_slot_valid(uint32_t write_id, uint32_t expected_pos) {
    return debug_ring_buffer_get_position(write_id) == ((expected_pos + 1) & DEBUG_RING_BUFFER_MPSC_POS_MASK);
}

// Device-side constants (debug_ring_buf_size is in core_config.h for codegen)
#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#if defined(ARCH_QUASAR)
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS;
constexpr uint32_t DEBUG_RING_BUFFER_MASK = DEBUG_RING_BUFFER_MPSC_MASK;
constexpr uint32_t DEBUG_RING_BUFFER_THREAD_ID_SHIFT = DEBUG_RING_BUFFER_MPSC_THREAD_ID_SHIFT;
constexpr uint32_t DEBUG_RING_BUFFER_POS_MASK = DEBUG_RING_BUFFER_MPSC_POS_MASK;
#else
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_SPSC_ELEMENTS;
#endif

#endif  // KERNEL_BUILD || FW_BUILD
