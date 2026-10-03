// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Partners share this tile's row or column. NoC 0 runs toward higher coordinates and NoC 1 toward lower, so a pair's
// two readings cross the same links in opposite directions.
#include <algorithm>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"

// A remote tile's NIU samples its RISCV_DEBUG_REG_WALL_CLOCK_L when the read request arrives.
namespace tile_read {

constexpr uint32_t kLandingOffsetMask = 0x3Fu;

// Keeps noc_reads_num_issued in step with the reads, as the watcher's check at every kernel's exit expects.
inline uint32_t read(uint32_t noc, uint32_t coord, uint32_t addr, uint32_t scratch) {
    const uint32_t dst = scratch + (addr & kLandingOffsetMask);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, 4);
    const uint32_t before = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
    while (NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED) == before) {
    }
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
}

template <typename Hi, typename Lo>
inline uint64_t wall64(Hi hi, Lo lo) {
    while (true) {
        const uint32_t first_hi = hi();
        const uint32_t low = lo();
        if (hi() == first_hi) {
            return (static_cast<uint64_t>(first_hi) << 32) | low;
        }
    }
}

inline uint64_t wall64(uint32_t noc, uint32_t coord, uint32_t scratch) {
    return wall64(
        [&] { return read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_1, scratch); },
        [&] { return read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_L, scratch); });
}

inline uint64_t own_wall64() {
    return wall64(
        [] { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_1); },
        [] { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); });
}

// The request is programmed once and re-sent each rep (the NIU only clears NOC_CMD_CTRL on acceptance). The bracket
// holds only the send and the poll, since every instruction inside it adds to the far end's allowance.
template <typename Sink>
__attribute__((noinline, cold)) inline void bracket(
    uint32_t noc, uint32_t coord, uint32_t scratch, volatile uint32_t* wall, uint32_t reps, Sink sink) {
    uint32_t prev = read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_L, scratch);
    volatile tt_l1_ptr uint32_t* const land =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (RISCV_DEBUG_REG_WALL_CLOCK_L & kLandingOffsetMask));
    volatile uint32_t* const ctrl =
        reinterpret_cast<volatile uint32_t*>(NOC_CMD_BUF_INSTANCE_OFFSET(noc, read_cmd_buf) + NOC_CMD_CTRL);
    for (uint32_t i = 0; i < reps; i++) {
        noc_reads_num_issued[noc] += 1;
        const uint32_t sentinel = prev ^ 0x80000000u;
        *land = sentinel;
        const uint32_t wall_before = *wall;
        *ctrl = NOC_CTRL_SEND_REQ;
        do {
            invalidate_l1_cache();
        } while (*land == sentinel);
        const uint32_t wall_after = *wall;
        const uint32_t partner_wall = *land;
        prev = partner_wall;
        sink(
            i,
            2 * static_cast<int64_t>(static_cast<int32_t>(partner_wall - wall_before)) -
                static_cast<int32_t>(wall_after - wall_before));
    }
}

constexpr uint32_t kWarmup = 16;
constexpr uint32_t kReps = 256;
constexpr uint32_t kBins = kernel_profiler::kTileNetBins;

// The bin holding the rank-th smallest reading, from the centre bin; rank is at most the histogram's count.
inline int32_t quantile(const volatile tt_l1_ptr uint32_t* hist, uint32_t rank) {
    uint32_t bin = 0;
    for (uint32_t seen = hist[0]; seen < rank; seen += hist[++bin]) {
    }
    return static_cast<int32_t>(bin) - static_cast<int32_t>(kBins / 2);
}

inline void measure(
    uint32_t noc,
    uint32_t coord,
    uint32_t scratch,
    volatile tt_l1_ptr uint32_t* hist,
    volatile tt_l1_ptr kernel_profiler::TileNetPartner& out) {
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    for (uint32_t bin = 0; bin < kBins; bin++) {
        hist[bin] = 0;
    }
    int64_t warm[kWarmup];
    bracket(noc, coord, scratch, wall, kWarmup, [&](uint32_t i, int64_t twice_offset) {
        uint32_t j = i;
        for (; j > 0 && warm[j - 1] > twice_offset; j--) {
            warm[j] = warm[j - 1];
        }
        warm[j] = twice_offset;
    });
    const int64_t centre = warm[kWarmup / 2];
    bracket(noc, coord, scratch, wall, kReps, [&](uint32_t, int64_t twice_offset) {
        hist[std::clamp<int64_t>(twice_offset - centre + kBins / 2, 0, kBins - 1)]++;
    });
    out.doubled_median = static_cast<int32_t>(centre + quantile(hist, kReps / 2 + 1));
    out.coarse = static_cast<int64_t>(wall64(noc, coord, scratch) - own_wall64());
}

}  // namespace tile_read

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t partner_count = get_arg_val<uint32_t>(1);
    constexpr uint32_t kFirstPartnerArg = 2;
    volatile tt_l1_ptr kernel_profiler::TileNetScratch* net_scratch =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::TileNetScratch*>(scratch);
    volatile tt_l1_ptr kernel_profiler::TileNetTable& table = net_scratch->table;
    table.ready = kernel_profiler::TileNetReady::Up;
    kernel_profiler::TileNetGo go;
    do {
        invalidate_l1_cache();
        go = table.go;
    } while (go == kernel_profiler::TileNetGo::Wait);
    if (go == kernel_profiler::TileNetGo::Measure) {
        for (uint32_t k = 0; k < partner_count; k++) {
            const auto partner =
                kernel_profiler::word_as<kernel_profiler::TileNetRead>(get_arg_val<uint32_t>(kFirstPartnerArg + k));
            tile_read::measure(
                partner.noc, NOC_XY_ENCODING(partner.x, partner.y), scratch, net_scratch->hist, table.partner[k]);
        }
        table.ready = kernel_profiler::TileNetReady::Done;
        do {
            invalidate_l1_cache();
        } while (table.go != kernel_profiler::TileNetGo::Exit);
    }
    // On a dispatch core the scratch is the profiler's ring space, which its first frame expects to be zero.
    volatile tt_l1_ptr uint32_t* scratch_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < sizeof(kernel_profiler::TileNetScratch) / 4; i++) {
        scratch_words[i] = 0;
    }
}
