// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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
// Capacity differs per arch: Quasar launches up to 22 concurrent user writers (6 DMs + 16
// TRISCs) pushing a handful of entries each, so needs enough headroom that one writer's
// oldest entries don't get evicted before the host can read them (head is a single counter
// shared by every writer, so entries are evicted in push order, not per-writer). Blackhole
// only ever launches 5 writers (2 DM + 3 TRISC), so the original, smaller capacity still holds.
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR = 128;
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE = 32;

struct debug_mpsc_ring_buf_slot_t {
    uint32_t data;
    uint32_t write_id;  // thread_idx + 1; 0 means never written
};

// Device-side struct, one capacity per arch. `head` and `slots` stay top-level fields (not
// nested) so device code (ring_buffer.h) accesses them exactly as before.
template <int Capacity>
struct debug_mpsc_ring_buf_msg_tmpl_t {
    uint32_t head;
    uint8_t _pad[60];  // Pad to 64-byte cache line
    debug_mpsc_ring_buf_slot_t slots[Capacity];
};

using debug_mpsc_ring_buf_msg_quasar_t = debug_mpsc_ring_buf_msg_tmpl_t<DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR>;
using debug_mpsc_ring_buf_msg_blackhole_t = debug_mpsc_ring_buf_msg_tmpl_t<DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE>;

inline uint32_t debug_ring_buffer_get_thread_idx(uint32_t write_id) { return write_id - 1; }

inline bool debug_ring_buffer_is_slot_valid(uint32_t write_id) { return write_id != 0; }

// Host-side only: the host doesn't know at compile time which arch's (differently-capacitied)
// debug_mpsc_ring_buf_msg_t it's reading, so it can't index through a single fixed struct type.
// `head` and the offset to `slots` are identical in both arch variants regardless of capacity
// (only the trailing array length differs), so raw pointer arithmetic works for either.
inline uint32_t debug_mpsc_ring_buffer_head(const uint8_t* base) { return *reinterpret_cast<const uint32_t*>(base); }

inline const debug_mpsc_ring_buf_slot_t* debug_mpsc_ring_buffer_slot(const uint8_t* base, uint32_t idx) {
    constexpr size_t kSlotsOffset = offsetof(debug_mpsc_ring_buf_msg_tmpl_t<1>, slots);
    return reinterpret_cast<const debug_mpsc_ring_buf_slot_t*>(
        base + kSlotsOffset + static_cast<size_t>(idx) * sizeof(debug_mpsc_ring_buf_slot_t));
}

// Device-side constants (debug_ring_buf_size is in core_config.h for codegen)
#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#if defined(ARCH_QUASAR)
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS_QUASAR;
using debug_mpsc_ring_buf_msg_t = debug_mpsc_ring_buf_msg_quasar_t;
#elif defined(ARCH_BLACKHOLE)
constexpr int DEBUG_RING_BUFFER_MPSC_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS_BLACKHOLE;
using debug_mpsc_ring_buf_msg_t = debug_mpsc_ring_buf_msg_blackhole_t;
#endif

#if defined(ARCH_QUASAR) || defined(ARCH_BLACKHOLE)
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_MPSC_ELEMENTS;
constexpr uint32_t DEBUG_RING_BUFFER_MASK = DEBUG_RING_BUFFER_MPSC_ELEMENTS - 1;
#else
constexpr int DEBUG_RING_BUFFER_ELEMENTS = DEBUG_RING_BUFFER_SPSC_ELEMENTS;
#endif

#endif  // KERNEL_BUILD || FW_BUILD
