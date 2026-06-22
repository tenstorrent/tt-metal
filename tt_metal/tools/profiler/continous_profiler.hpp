// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Continuous (streaming) profiler scopes for the X280 SPSC pull model.
//
// A DeviceZoneScopedN-shaped scope whose constructor/destructor emit
// ZONE_START / ZONE_END markers -- but instead of accumulating into the fixed
// 2KB per-RISC profiler buffer that kernel_profiler drains only at kernel end,
// markers STREAM into an L1 SPSC flit-ring that the X280 (L2CPU running Linux)
// drains live over the NoC. The ring backpressures: the producer blocks when the
// ring is full, so it is lossless -- a busier core is simply more likely to stall
// waiting for the X280 to flush.
//
// Marker format is byte-identical to kernel_profiler (same host-side decode):
//   word0 = 0x80000000 | ((timer_id & 0x7FFFF) << 12) | (wall_clock_hi & 0xFFF)
//   word1 =  wall_clock_lo
//   timer_id = (zone_id & 0xFFFF) | ((type << 16) & 0x7FFFF)   type: 0=START 1=END
// 8 markers (8B each) pack into one 64B NoC flit; a flit is published (w advances)
// once full, so the X280 always reads complete flits.
//
// Prototype for the X280 SPSC pull-profiler experiment; companion to
// tt_metal/programming_examples/x280_spsc and tools/tracy/x280/X280_PROFILER_PULL_DESIGN.md.
// This header is intentionally separate from kernel_profiler.hpp.
#pragma once

#include <cstdint>

// ---- SPSC ring geometry in this core's L1 (override via -D at build time) ----
#ifndef CP_RING_BASE
#define CP_RING_BASE 0x80000  // base of the flit ring
#endif
#ifndef CP_RING_CELLS
#define CP_RING_CELLS 32  // ring depth in 64B flits (must be a power of 2)
#endif
#ifndef CP_W_ADDR
#define CP_W_ADDR 0x80800  // write index (this core writes, X280 reads)
#endif
#ifndef CP_R_ADDR
#define CP_R_ADDR 0x80840  // read index  (X280 writes, this core reads)
#endif
#ifndef CP_BLOCK_ADDR
#define CP_BLOCK_ADDR 0x80880  // backpressure-event counter (diagnostic)
#endif

namespace continuous_profiler {

enum PacketType : uint32_t { ZONE_START = 0, ZONE_END = 1 };

constexpr uint32_t get_const_id(uint32_t zone_id, uint32_t type) {
    return (zone_id & 0xFFFF) | ((type << 16) & 0x7FFFF);
}

// Blackhole wall-clock debug registers (RISCV_DEBUG_REG_WALL_CLOCK_L / _H).
// Reading the low half latches the high half (matches kernel_profiler).
static constexpr uint32_t kWallLo = 0xFFB121F0u;
static constexpr uint32_t kWallHi = 0xFFB121F8u;

namespace detail {
// One streaming producer per RISC; this header is included by a single kernel TU
// so file-scope statics are per-RISC state with no init guards.
static uint32_t w_index = 0;      // flit write index (monotonic)
static uint32_t marker_slot = 0;  // 0..7: marker position within the current flit
static uint32_t blocked = 0;      // backpressure events (ring went full)

inline volatile uint32_t* ring() { return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(CP_RING_BASE)); }
inline volatile uint32_t* wreg() { return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(CP_W_ADDR)); }
inline volatile uint32_t* rreg() { return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(CP_R_ADDR)); }
inline volatile uint32_t* breg() { return reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(CP_BLOCK_ADDR)); }
}  // namespace detail

// Reset the ring + producer state. Call once at the top of kernel_main.
inline void init() {
    *detail::rreg() = 0;
    *detail::wreg() = 0;
    *detail::breg() = 0;
    detail::w_index = 0;
    detail::marker_slot = 0;
    detail::blocked = 0;
}

// Emit one 8B marker; publish the flit (advance w) once it holds 8 markers.
// Blocks while the ring is full so a marker is never overwritten (lossless).
inline __attribute__((always_inline)) void emit(uint32_t timer_id) {
    if (detail::marker_slot == 0) {
        // Starting a fresh flit: do not overwrite an un-drained flit. r is
        // published by the X280 over the NoC; a local load sees its latest value.
        if ((detail::w_index - *detail::rreg()) >= CP_RING_CELLS) {
            detail::blocked++;
            *detail::breg() = detail::blocked;
            while ((detail::w_index - *detail::rreg()) >= CP_RING_CELLS) {
                // backpressure spin (cheap local load, no NoC traffic)
            }
        }
    }

    const uint32_t lo = *reinterpret_cast<volatile uint32_t*>(kWallLo);  // latches hi
    const uint32_t hi = *reinterpret_cast<volatile uint32_t*>(kWallHi);

    volatile uint32_t* m = detail::ring() + (detail::w_index & (CP_RING_CELLS - 1)) * 16u + detail::marker_slot * 2u;
    m[0] = 0x80000000u | ((timer_id & 0x7FFFF) << 12) | (hi & 0xFFF);
    m[1] = lo;

    if (++detail::marker_slot == 8u) {
        detail::marker_slot = 0;
        asm volatile("fence" ::: "memory");  // whole flit in L1 before publishing w
        detail::w_index += 1;
        *detail::wreg() = detail::w_index;
    }
}

// DeviceZoneScopedN-shaped scope: START marker on construct, END on destruct.
template <uint32_t zone_id>
struct ZoneScoped {
    inline __attribute__((always_inline)) ZoneScoped() { emit(get_const_id(zone_id, ZONE_START)); }
    inline __attribute__((always_inline)) ~ZoneScoped() { emit(get_const_id(zone_id, ZONE_END)); }
};

}  // namespace continuous_profiler

#define CP_CONCAT2(a, b) a##b
#define CP_CONCAT(a, b) CP_CONCAT2(a, b)
// Mirror of DeviceZoneScopedN(name): a unique compile-time zone id per call site
// from __COUNTER__. The `name` string is accepted for API parity; building the
// host-side id<->name table is out of scope for this prototype.
#define ContinuousZoneScopedN(name) \
    continuous_profiler::ZoneScoped<__COUNTER__> CP_CONCAT(cp_zone_, __LINE__) {}
