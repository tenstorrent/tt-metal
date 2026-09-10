// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Resident local clock tracker. Runs on ONE IDLE ethernet core per chip for the life of the profiling
// session, sampling this chip's AICLK wall clock against the eth tile's free-running 50 MHz counter and
// emitting RAW pairs. All fitting happens on the host: nothing here is derived, averaged or fitted, so
// there is no reference rate on the device to be wrong about.
//
// TRANSPORT: the profiler's own SPSC ring, drained by the DRISC drainers like any other producer. The
// previous version wrote a private ring in UNRESERVED L1 that the host polled over MMIO at ~8-10 MB/s,
// which is the wrong pipe by three orders of magnitude -- the drain path moves 5.27 GB/s. That poll is
// what forced the bounded-bite reader, its 55% duty cycle, the torn-batch detection and the ring
// sizing; emitting as a normal producer deletes all four.
//
// WHY PP_DATA RATHER THAN A NEW WIRE TYPE: the DRISC drain kernel keeps its own copy of the packet
// walker and must know each type's length, so a new type means touching drisc.elf -- which is already
// 8 bytes from overflowing its code region (segment[0] limit 0x2c00). PP_DATA is self-describing and
// already understood end to end, at a cost of one extra word.
//
// CADENCE: 3 us, chosen against the lane budget rather than the physics. One lane holds 512 words and a
// filler sweep services it about every 208 us, so at 5 words per sample 3 us fills ~69% of the ring
// between drains; 1 us would need 164% and simply cannot fit. This costs almost nothing, because the
// transport ceiling coincides with the measured crossover at T ~ 2.8 us, where the deficit*T interval
// term meets the ~6 ns quantisation floor. Below ~3 us there was never anything to gain.
//
// An IDLE core is the right home: every trained channel is claimed by a router under the default fabric
// configs, so launching on a claimed channel would write a launch message into a live router. The
// measurement transfers because eth cores share one clock source and increment together.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t kStrideTicks = get_compile_time_arg_val(0);

constexpr uint32_t kWallClockL = 0xFFB121F0;
constexpr uint32_t kWallClockH = 0xFFB121F8;
constexpr uint32_t kRefclkLoAddr = 0xFFB98850;
constexpr uint32_t kRefclkHiAddr = 0xFFB98854;

// Reading L latches H, so L must be read first; both halves then belong to the same instant.
inline __attribute__((always_inline)) void read_wall(uint32_t& hi, uint32_t& lo) {
    lo = *reinterpret_cast<volatile uint32_t*>(kWallClockL);
    hi = *reinterpret_cast<volatile uint32_t*>(kWallClockH);
}

// The refclk pair has no latch, so it needs an explicit hi/lo/hi guard: a naive read can splice a fresh
// low onto a stale high at a 2^32 boundary. ~12 cycles.
inline __attribute__((always_inline)) void read_refclk(uint32_t& hi, uint32_t& lo) {
    volatile uint32_t* lop = reinterpret_cast<volatile uint32_t*>(kRefclkLoAddr);
    volatile uint32_t* hip = reinterpret_cast<volatile uint32_t*>(kRefclkHiAddr);
    const uint32_t h1 = *hip;
    uint32_t l = *lop;
    const uint32_t h2 = *hip;
    if (h1 != h2) {
        l = *lop;
    }
    hi = h2;
    lo = l;
}
inline __attribute__((always_inline)) uint64_t refclk64() {
    uint32_t hi, lo;
    read_refclk(hi, lo);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

#if defined(PROFILE_KERNEL)

// PP_CLOCK: optional sticky-timer word + w0 + wall_lo. Two words steady state, three worst case.
// This replaced a 5-word PP_DATA record whose 3 words of header carried 2 words of payload; the refclk
// high half needed a whole word despite moving only once per 85.9 s, and the size field described a
// payload that never varies. At 3 us the lane budget goes from ~69% of the ring between drains to
// ~28%, which is also what would make 1 us affordable at all.
constexpr uint32_t kEmitWords = 1 + 2;

// RESERVE-OR-SKIP, never the blocking reserve. kernel_profiler::ring_ensure_room() SPINS when the ring
// is full, and stalling is the one thing a producer on an eth core must not do -- it is why the fabric
// router's sync hook reserves with its own non-blocking check. A skipped sample is also cheap HERE in a
// way a skipped zone would not be: the pairs are absolute, so a gap is still measurable as a straight
// segment between the samples either side of it. Dropping beats stalling, and beats tearing.
inline __attribute__((always_inline)) bool ring_has_room(uint32_t nwords) {
    invalidate_l1_cache();
    const uint32_t head = kernel_profiler::profiler_control_buffer[kernel_profiler::HEAD_INDEX];
    return (kernel_profiler::wIndex - head) <= (kernel_profiler::RING_USABLE - nwords);
}

// The wall clock is the packet's OWN timestamp (low half here, high half from the sticky timer), and
// the refclk rides in low27 truncated to 24 bits. rhi is still read -- read_refclk() needs it for the
// hi/lo/hi splice guard -- but never transmitted: it advances once per 85.9 s and the host reconstructs
// it by unwrapping.
inline __attribute__((always_inline)) void emit_pair() {
    uint32_t whi, wlo, rhi, rlo;
    read_wall(whi, wlo);
    read_refclk(rhi, rlo);
    (void)rhi;
    kernel_profiler::ring_write_sticky_timer(whi);
    kernel_profiler::ring_write_word(kernel_profiler::ppfmt::clock_w0(kernel_profiler::ppfmt::CLOCK_LOCAL_REFCLK, rlo));
    kernel_profiler::ring_write_word(wlo);
    kernel_profiler::publish_tail();
}

#endif

void kernel_main() {
#if defined(PROFILE_KERNEL)
    bool armed = false;  // set once PROFILER_TERMINATE has been observed clear at least once
    uint64_t target = refclk64() + kStrideTicks;
    while (true) {
        // Spin to the next stride boundary in REFCLK, not in wall clock: the whole point is a cadence
        // that DVFS cannot stretch.
        uint64_t rc = refclk64();
        while (rc < target) {
            rc = refclk64();
        }
        target = rc + kStrideTicks;

        if (ring_has_room(kEmitWords)) {
            emit_pair();
        }

        // Teardown is the profiler's own terminate flag, which is what terminate_eth_producers() sets.
        // The old mailbox command pair and its arm timeout are gone with the private ring they served.
        //
        // ARM ON A CLEAR READ FIRST. The flag is a session-teardown signal, not a "you may run" signal,
        // and at launch it can still be SET from the previous session -- the drainers clear it later in
        // bring-up. Honouring it immediately made this kernel emit exactly one sample and exit, on both
        // chips, deterministically (heartbeat: loops 1, emits 1). Waiting to see it clear once before
        // treating a set as "stop" makes a stale flag harmless without weakening real teardown.
        invalidate_l1_cache();
        const uint32_t term = kernel_profiler::profiler_control_buffer[kernel_profiler::PROFILER_TERMINATE];
        if (!armed) {
            armed = (term == 0u);
        } else if (term != 0u) {
            return;
        }
    }
#endif
}
