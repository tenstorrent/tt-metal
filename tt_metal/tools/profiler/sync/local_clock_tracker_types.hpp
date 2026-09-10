// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// L1 layout shared by the resident local clock tracker kernel and the host that drains it.
// No device-side includes, so the host can include it directly.
//
// WHAT THIS IS FOR. The device-to-device sync corrects the link at its own cadence. Between two of its
// rounds the local AICLK can run ~0.47% slow for ~1 ms -- an excursion that fits entirely inside one
// interval, so the wire path never sees it happen; it only observes that the offset has already moved.
// Measured: up to ~5 us, in roughly 1 idle interval in 4000. A core watching its wall clock against the
// eth tile's DVFS-immune 50 MHz counter at 1 us sees it 1000x sooner.
//
// Complementary, not a replacement: refclk is per-chip and free-running from its own reset, so it says
// nothing about chip-to-chip alignment. That still comes from the wire.
//
// THE KERNEL DOES NO ARITHMETIC. It writes raw (wall, refclk) pairs into a wrapping ring and the host
// fits the rate. An earlier version differenced against a hardcoded 27 cycles/tick and was wrong: the
// eth core's AICLK moves between 800 MHz (asic_fmin, chip idle) and 1350 MHz under load, so "deviation
// from nominal" is dominated by the DVFS frequency term -- hundreds of millions of cycles per second,
// burying the ~7000-cycle excursions entirely, and wrapping int32 into a sawtooth. The rate must be
// TRACKED, not assumed, and the host is where that belongs.

#pragma once

#include <cstdint>

namespace tt::tt_metal::local_clock {

constexpr uint32_t kMagic = 0x4C434C4B;  // 'LCLK'

// The eth tile's own 50 MHz counter, in its local register window (not a NoC access).
constexpr uint32_t kRefclkLoAddr = 0xFFB98850;
constexpr uint32_t kRefclkHiAddr = 0xFFB98854;

enum Status : uint32_t {
    ST_READY = 0x10,
    ST_RUNNING = 0x30,
    ST_STOPPED = 0x40,
};

enum Command : uint32_t {
    CMD_NONE = 0,
    CMD_GO = 1,
    CMD_STOP = 2,
};

// One raw sample. 16 B so host indexing is a shift.
struct Sample {
    uint32_t w_lo, w_hi;
    uint32_t rc_lo, rc_hi;
};

// Header. `write_idx` is MONOTONIC (never wrapped), so the host can tell the difference between
// "nothing new" and "the ring lapped me": if write_idx advanced by more than the ring holds, samples
// were lost and the host must say so rather than fitting a curve through a hole.
struct Hdr {
    uint32_t magic;
    uint32_t status;
    uint32_t write_idx_lo;  // monotonic count of samples written
    uint32_t write_idx_hi;
    uint32_t ring_entries;  // capacity, echoed so the host cannot disagree with the kernel about it
    uint32_t stride_ticks;
    uint32_t pad0, pad1;
};

}  // namespace tt::tt_metal::local_clock
