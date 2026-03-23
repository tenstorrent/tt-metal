// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// L1 ring buffer shared between the BRISC reader and NCRISC pusher on the
// real-time profiler core. BRISC writes entries at write_index; NCRISC reads
// entries at read_index and pushes them to the host via PCIe. Single-producer
// single-consumer: no locks required.

constexpr uint32_t RT_PROFILER_RING_CAPACITY = 128;  // must be power-of-2
constexpr uint32_t RT_PROFILER_ENTRY_SIZE = 64;      // matches D2H socket page size

struct RtProfilerRingBuffer {
    volatile uint32_t write_index;  // incremented by BRISC after writing an entry
    volatile uint32_t read_index;   // incremented by NCRISC after pushing an entry
    volatile uint32_t terminate;    // set by BRISC to tell NCRISC to drain and exit
    uint32_t _pad[13];              // pad header to 64 bytes for alignment
    uint8_t data[RT_PROFILER_RING_CAPACITY][RT_PROFILER_ENTRY_SIZE];
};

// 64 + 128*64 = 8256 bytes total

static_assert(sizeof(RtProfilerRingBuffer) == 64 + RT_PROFILER_RING_CAPACITY * RT_PROFILER_ENTRY_SIZE);

inline bool rt_ring_full(volatile RtProfilerRingBuffer* rb) {
    return (rb->write_index - rb->read_index) >= RT_PROFILER_RING_CAPACITY;
}

inline bool rt_ring_empty(volatile RtProfilerRingBuffer* rb) { return rb->write_index == rb->read_index; }

inline uint32_t rt_ring_data_addr(volatile RtProfilerRingBuffer* rb, uint32_t index) {
    uint32_t slot = index & (RT_PROFILER_RING_CAPACITY - 1);
    return reinterpret_cast<uint32_t>(rb->data[slot]);
}
