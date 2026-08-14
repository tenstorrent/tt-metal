// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// SPSC (WH)
constexpr int16_t DEBUG_RING_BUFFER_STARTING_INDEX = -1;
constexpr int DEBUG_RING_BUFFER_SPSC_ELEMENTS = 32;

struct debug_spsc_ring_buf_msg_t {
    int16_t current_ptr;
    uint16_t wrapped;
    uint32_t data[DEBUG_RING_BUFFER_SPSC_ELEMENTS];
};

// MPSC (Quasar, Blackhole) - lock-free ring buffer for concurrent writes using 32-bit atomics
// Works on tt-qsr64 (DM), tt-qsr32 (TRISC), and BH BRISC/NCRISC/TRISC0-2 (Zaamo)
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = 32;
constexpr uint32_t DEBUG_RING_BUFFER_MPSC_MASK = DEBUG_RING_BUFFER_MPSC_ELEMENTS - 1;

struct debug_mpsc_ring_buf_slot_t {
    uint32_t data;
    uint32_t write_id;  // thread_idx + 1; 0 means never written
};

struct debug_mpsc_ring_buf_msg_t {
    uint32_t head;
    uint8_t _pad[60];  // Pad to 64-byte cache line
    debug_mpsc_ring_buf_slot_t slots[DEBUG_RING_BUFFER_MPSC_ELEMENTS];
};

inline uint32_t debug_ring_buffer_get_thread_idx(uint32_t write_id) { return write_id - 1; }

inline bool debug_ring_buffer_is_slot_valid(uint32_t write_id) { return write_id != 0; }

// Device-side constants (debug_ring_buf_size is in core_config.h for codegen)
#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS;
constexpr uint32_t DEBUG_RING_BUFFER_MASK = DEBUG_RING_BUFFER_MPSC_MASK;
#else
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_SPSC_ELEMENTS;
#endif

#endif  // KERNEL_BUILD || FW_BUILD
