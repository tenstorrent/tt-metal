// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// A tile's wall clock read over the NoC: one raw 4 B read of its RISCV_DEBUG_REG_WALL_CLOCK_L, which the tile's NIU
// samples when the request arrives, bracketed by this core's own wall reads around nothing but the send store and
// the poll for the landing word. Shared by the idle-eth pusher and the worker tile network kernel.
#pragma once
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace tile_read {

constexpr uint32_t kWallLo = 0xFFB121F0u;      // RISCV_DEBUG_REG_WALL_CLOCK_L, latches the high word
constexpr uint32_t kWallHiLive = 0xFFB121F4u;  // RISCV_DEBUG_REG_WALL_CLOCK_1, the live high word

inline uint32_t coord(uint32_t xy) {
    return static_cast<uint32_t>(get_noc_addr(xy & 0xFFFFu, xy >> 16, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

// A 4 B read of `addr` on tile `coord`, landing at the scratch word congruent to it: programmed first, then sent
// and waited for as one step, so a bracket around the send alone holds the packet's flight and the two NIUs'
// handling and none of the programming.
inline volatile tt_l1_ptr uint32_t* arm(uint32_t noc, uint32_t coord, uint32_t addr, uint32_t bytes, uint32_t scratch) {
    const uint32_t dst = scratch + (addr & 0x3Fu);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, bytes);
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
}

// Waits on the NIU's own count of responses, and keeps the firmware's count of issued reads, which the eth
// firmware syncs against at kernel exit.
inline void send_wait(uint32_t noc) {
    const uint32_t before = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
    while (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == before) {
    }
}

inline uint32_t read(uint32_t noc, uint32_t coord, uint32_t addr, uint32_t scratch) {
    volatile tt_l1_ptr uint32_t* land = arm(noc, coord, addr, 4, scratch);
    send_wait(noc);
    invalidate_l1_cache();
    return *land;
}

// A tile's whole wall clock, coarse by the reads' flight: the live high word before and after the low word, until
// the two agree. Only the clock's 2^32-tick turn is taken from it; the bracket medians carry the precision.
inline uint64_t wall64(uint32_t noc, uint32_t coord, uint32_t scratch) {
    for (;;) {
        const uint32_t hi0 = read(noc, coord, kWallHiLive, scratch);
        const uint32_t lo = read(noc, coord, kWallLo, scratch);
        if (read(noc, coord, kWallHiLive, scratch) == hi0) {
            return (static_cast<uint64_t>(hi0) << 32) | lo;
        }
    }
}

inline uint64_t own_wall64() {
    volatile uint32_t* const lo = reinterpret_cast<volatile uint32_t*>(kWallLo);
    volatile uint32_t* const hi = reinterpret_cast<volatile uint32_t*>(kWallHiLive);
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
    uint32_t prev = read(noc, coord, kWallLo, scratch);  // also leaves the request programmed
    volatile tt_l1_ptr uint32_t* const land =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (kWallLo & 0x3Fu));
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

}  // namespace tile_read
