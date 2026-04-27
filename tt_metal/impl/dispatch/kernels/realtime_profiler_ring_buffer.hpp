// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// L1 layout shared between the BRISC reader and NCRISC pusher on the reserved
// real-time profiler tensix core, plus the host. The struct lives at a deterministic
// L1 address carved out of dispatch L1 (above CommandQueueDeviceAddrType::UNRESERVED).
// It is NOT routed through the user-space allocator — the profiler tensix is excluded
// from the L1 bank table, so no user buffer can collide here.
//
// The host header `tt_metal/impl/dispatch/realtime_profiler_l1.hpp` derives the L1
// base from this header so layout decisions live in one place; both this file and
// that one must stay in sync.

constexpr uint32_t RT_PROFILER_RING_CAPACITY = 4096;  // must be power-of-2
constexpr uint32_t RT_PROFILER_ENTRY_SIZE = 64;       // matches D2H socket page size

// Ring buffer for BRISC -> NCRISC handoff. BRISC writes entries at write_index;
// NCRISC reads entries at read_index and pushes them to the host via PCIe.
// Single-producer single-consumer: no locks required.
struct RtProfilerRingBuffer {
    volatile uint32_t write_index;  // incremented by BRISC after writing an entry
    volatile uint32_t read_index;   // incremented by NCRISC after pushing an entry
    volatile uint32_t terminate;    // set by BRISC to tell NCRISC to drain and exit
    uint32_t _pad[13];              // pad header to 64 bytes for alignment
    uint8_t data[RT_PROFILER_RING_CAPACITY][RT_PROFILER_ENTRY_SIZE];
};

// 64 + 4096*64 = 262208 bytes total
static_assert(sizeof(RtProfilerRingBuffer) == 64 + RT_PROFILER_RING_CAPACITY * RT_PROFILER_ENTRY_SIZE);

// Bytes reserved for the D2H sender socket config (sender_socket_md + bytes_acked +
// d2h_sender_socket_md, each rounded up to L1_ALIGNMENT). The actual on-wire size
// today is 32+16+16=64 B at L1_ALIGNMENT=16, but we reserve more headroom so the
// layout doesn't shift if any of those structs grow.
constexpr uint32_t RT_PROFILER_SOCKET_CONFIG_SIZE = 128;

// Packed L1 layout for the reserved RT-profiler tensix core. All sub-regions live
// inside a single carve-out, so the host only computes one base address and derives
// every sub-address with offsetof().
struct RealtimeProfilerCoreL1 {
    RtProfilerRingBuffer ring;
    uint8_t socket_config[RT_PROFILER_SOCKET_CONFIG_SIZE];
};

// Ring is uint32-aligned and a multiple of 16, so the trailing socket_config region
// is naturally L1-aligned (16 B) without padding.
static_assert(sizeof(RtProfilerRingBuffer) % 16 == 0);
static_assert(sizeof(RealtimeProfilerCoreL1) == sizeof(RtProfilerRingBuffer) + RT_PROFILER_SOCKET_CONFIG_SIZE);

inline bool rt_ring_full(volatile RtProfilerRingBuffer* rb) {
    return (rb->write_index - rb->read_index) >= RT_PROFILER_RING_CAPACITY;
}

inline bool rt_ring_empty(volatile RtProfilerRingBuffer* rb) { return rb->write_index == rb->read_index; }

// Kernel-only helper: returns the L1 address (as uint32) of the ring slot at `index`.
// Computed via offset arithmetic so the file is also includable from 64-bit host TUs
// without triggering pointer-narrowing warnings.
inline uint32_t rt_ring_data_addr(volatile RtProfilerRingBuffer* rb, uint32_t index) {
    uint32_t slot = index & (RT_PROFILER_RING_CAPACITY - 1);
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(rb)) + offsetof(RtProfilerRingBuffer, data) +
           slot * RT_PROFILER_ENTRY_SIZE;
}
