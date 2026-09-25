// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// A tile's wall clock read over the NoC: one raw 4 B read of its RISCV_DEBUG_REG_WALL_CLOCK_L, which the tile's NIU
// samples when the request arrives, bracketed by this core's own wall reads around nothing but the send store and
// the poll for the landing word. Shared by the tile clock network kernel (tile_sync.cpp) and the tile offsets test's
// reader (offset_reader.cpp).
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_sync.h"

namespace tile_read {

inline uint32_t coord(uint32_t xy) {
    return static_cast<uint32_t>(get_noc_addr(xy & 0xFFFFu, xy >> 16, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

// A 4 B read of `addr` on tile `coord`, landing at the scratch word congruent to it. It waits on the NIU's own count
// of responses, and keeps the firmware's count of issued reads, which the eth firmware syncs against at kernel exit.
inline uint32_t read(uint32_t noc, uint32_t coord, uint32_t addr, uint32_t scratch) {
    const uint32_t dst = scratch + (addr & 0x3Fu);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, 4);
    const uint32_t before = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
    while (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == before) {
    }
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
}

// A tile's whole wall clock, coarse by the reads' flight: the live high word before and after the low word, until
// the two agree. Only the clock's 2^32-tick turn is taken from it; the bracket medians carry the precision.
inline uint64_t wall64(uint32_t noc, uint32_t coord, uint32_t scratch) {
    for (;;) {
        const uint32_t hi0 = read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_1, scratch);
        const uint32_t lo = read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_L, scratch);
        if (read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_1, scratch) == hi0) {
            return (static_cast<uint64_t>(hi0) << 32) | lo;
        }
    }
}

inline uint64_t own_wall64() {
    volatile uint32_t* const lo = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint32_t* const hi = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_1);
    for (;;) {
        const uint32_t hi0 = *hi;
        const uint32_t l = *lo;
        if (*hi == hi0) {
            return (static_cast<uint64_t>(hi0) << 32) | l;
        }
    }
}

// `reps` brackets of tile `coord` on one NoC; each hands `f` the rep index, 2 * (tile wall - bracket midpoint) in
// wall ticks and the round trip (the bracket's width); `wall` is this core's own low wall-clock word. The request is
// programmed once and re-sent per rep (the NIU clears only NOC_CMD_CTRL on acceptance), and the bracket holds nothing
// but the send store and the poll: every instruction inside it is time the far end is allowed. The poll watches the
// word land in L1, which is closer than the NIU's response counter; the sentinel is the previous reading with its top
// bit flipped, a value the next reading can only take after advancing exactly 2^31 ticks, so a halted or slow clock
// cannot make it spin. The low words differ by the two tiles' offset modulo 2^32, anywhere in [-2^31, 2^31), so the
// doubled and summed values need 64 bits: in 32 they wrap for 3/4 of the offsets.
template <typename F>
__attribute__((noinline, cold)) inline void bracket(
    uint32_t noc, uint32_t coord, uint32_t scratch, volatile uint32_t* wall, uint32_t reps, F f) {
    uint32_t prev = read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_L, scratch);  // also leaves the request programmed
    volatile tt_l1_ptr uint32_t* const land =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (RISCV_DEBUG_REG_WALL_CLOCK_L & 0x3Fu));
    volatile uint32_t* const ctrl = reinterpret_cast<volatile uint32_t*>(
        (NCRISC_RD_CMD_BUF << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CMD_CTRL);
    for (uint32_t i = 0; i < reps; i++) {
        noc_reads_num_issued[noc] += 1;
        const uint32_t sentinel = prev ^ 0x80000000u;
        *land = sentinel;
        const uint32_t w0 = *wall;
        *ctrl = NOC_CTRL_SEND_REQ;
        do {
            invalidate_l1_cache();
        } while (*land == sentinel);
        const uint32_t w1 = *wall;
        const uint32_t v = *land;
        prev = v;
        f(i,
          2 * static_cast<int64_t>(static_cast<int32_t>(v - w0)) - static_cast<int32_t>(w1 - w0),
          static_cast<int64_t>(static_cast<int32_t>(w1 - w0)));
    }
}

constexpr uint32_t kWarmup = 16;
constexpr uint32_t kBins = kernel_profiler::kTileNetBins;

// The bin whose running count first reaches `rank` samples.
inline int32_t quantile(const volatile tt_l1_ptr uint32_t* hist, uint32_t rank) {
    uint32_t seen = 0;
    for (uint32_t b = 0; b < kBins; b++) {
        seen += hist[b];
        if (seen >= rank) {
            return static_cast<int32_t>(b) - static_cast<int32_t>(kBins / 2);
        }
    }
    return static_cast<int32_t>(kBins / 2) - 1;
}

// Tile `coord`'s clock against this core's from `reps` brackets on one NoC, written to `out`. The samples go into two
// histograms at `hist` (2 * kTileNetBins words: offsets, then round trips) centred on the medians of a short warm-up,
// whose first brackets run cold and land wide; the scratch is the same size for any number of reps, and a sample
// outside the window lands in its edge bin, which moves no median while such samples stay in the minority.
inline void measure(
    uint32_t noc,
    uint32_t coord,
    uint32_t scratch,
    uint32_t reps,
    volatile tt_l1_ptr uint32_t* hist_d,
    volatile tt_l1_ptr kernel_profiler::TileNetPartner& out) {
    volatile tt_l1_ptr uint32_t* hist_r = hist_d + kBins;
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    for (uint32_t b = 0; b < 2 * kBins; b++) {
        hist_d[b] = 0;
    }
    int64_t warm_d[kWarmup], warm_r[kWarmup];
    bracket(noc, coord, scratch, wall, kWarmup, [&](uint32_t i, int64_t d, int64_t r) {
        uint32_t j = i;
        for (; j > 0 && warm_d[j - 1] > d; j--) {
            warm_d[j] = warm_d[j - 1];
        }
        warm_d[j] = d;
        for (j = i; j > 0 && warm_r[j - 1] > r; j--) {
            warm_r[j] = warm_r[j - 1];
        }
        warm_r[j] = r;
    });
    const int64_t centre_d = warm_d[kWarmup / 2], centre_r = warm_r[kWarmup / 2];
    bracket(noc, coord, scratch, wall, reps, [&](uint32_t, int64_t d, int64_t r) {
        const int64_t bd = d - centre_d + kBins / 2, br = r - centre_r + kBins / 2;
        hist_d[bd < 0 ? 0 : bd >= kBins ? kBins - 1 : bd]++;
        hist_r[br < 0 ? 0 : br >= kBins ? kBins - 1 : br]++;
    });
    const uint64_t coarse = wall64(noc, coord, scratch) - own_wall64();
    out.median2 = static_cast<int32_t>(centre_d + quantile(hist_d, reps / 2 + 1));
    out.spread2 = quantile(hist_d, 3 * reps / 4 + 1) - quantile(hist_d, reps / 4 + 1);
    out.rtt = static_cast<int32_t>(centre_r + quantile(hist_r, reps / 2 + 1));
    out.coarse_lo = static_cast<uint32_t>(coarse);
    out.coarse_hi = static_cast<uint32_t>(coarse >> 32);
}

}  // namespace tile_read
