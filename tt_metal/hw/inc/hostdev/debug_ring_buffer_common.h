// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Debug ring buffer to be used on Kernels.
// We use a  SPSC on WH/BH since no atomics are supported but MPSC on quasar's threaded model.

// MPSC protocol:
// 1. Atomically claim slot via fetch_add on head
// 2. Write data to claimed slot (uncached)
// 3. Publish via write_id (encodes thread_idx + position idx for validation)
// 4. Host validates empty/full slots by matching written position idx from write_id with expected position idx
// 5. Overwriting is ok - host stores recent entries for display

// SPSC (WH/BH)
constexpr int16_t DEBUG_RING_BUFFER_STARTING_INDEX = -1;
constexpr uint32_t DEBUG_RING_BUFFER_SPSC_ELEMENTS = 32;

struct debug_spsc_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_SPSC_ELEMENTS];
};

// MPSC (Quasar): lock-free ring buffer for concurrent writes using 32-bit atomics
// Works on both DMs (tt-qsr64) and TRISCs (tt-qsr32)
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_ELEMENTS = 128;
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_MASK = DEBUG_RING_BUFFER_MPSC_ELEMENTS - 1;
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_THREAD_ID_SHIFT = 27;   // Upper 5 bits for thread_idx
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_POS_MASK = 0x07FFFFFF;  // Lower 27 bits for position idx

// Per-entry slot in MPSC
// Consists of 32-bit data entry + write_id metadata
struct debug_mpsc_ring_buf_slot_t {
    uint32_t data;
    uint32_t write_id;  // [31:27] thread_idx | [26:0] (pos+1)
};

struct debug_mpsc_ring_buf_msg_t {
    uint32_t head;
    uint8_t _pad[60];  // Pad to 64-byte cache line (prevents false sharing as head will be updated on multiple cores)
    debug_mpsc_ring_buf_slot_t slots[DEBUG_RING_BUFFER_MPSC_ELEMENTS];
};

// Host-side MPSC helpers

// Extract 5-bit thread_idx from write_id
inline uint32_t debug_ring_buffer_get_thread_idx(uint32_t write_id) {
    return write_id >> DEBUG_RING_BUFFER_MPSC_THREAD_ID_SHIFT;
}

inline uint32_t debug_ring_buffer_get_pos_idx(uint32_t write_id) { return write_id & DEBUG_RING_BUFFER_MPSC_POS_MASK; }

// Checks if the written pos idx from device matches with expected pos idx on host
inline bool debug_ring_buffer_is_slot_valid(uint32_t write_id, uint32_t expected_pos) {
    return debug_ring_buffer_get_pos_idx(write_id) == ((expected_pos + 1) & DEBUG_RING_BUFFER_MPSC_POS_MASK);
}

// Device-side constants (debug_ring_buf_size is in core_config.h for codegen)
#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#if defined(ARCH_QUASAR)
constexpr uint32_t DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS;
#else
constexpr uint32_t DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_SPSC_ELEMENTS;
#endif

#endif  // KERNEL_BUILD || FW_BUILD
