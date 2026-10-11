// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Each tile has its own wall clock, offset from the others, and the streaming profiler stamps records with it. This
// kernel measures those offsets so every tile's records share one timeline. It runs once per device open, only with the
// streaming profiler enabled. Normal runs never launch it.
//
// A tile reads its own clock, reads a neighbour's clock over the NoC, then reads its own clock again. The neighbour's
// NIU reads its clock when the request arrives, so that reading falls between the tile's two. One reading on its own is
// off, because the request and the reply take different routes. Each NoC only moves data in one direction, so the reply
// has to go the long way around. The neighbour then reads this tile over the other NoC, which moves data in the
// opposite direction, so its request and reply take routes of the same lengths as ours. Both readings are off by the
// same amount, while the offset appears in them with opposite signs, so half the difference of the two readings is the
// offset.
//
// The timed reads re-send a programmed request instead of calling noc_async_read, to keep software out of that
// measurement. Time a tile spends in software before the send or after the reply counts like extra NoC delay. The
// pairing only cancels it if both tiles spend exactly the same time in software, and the more software sits between the
// two own-clock reads, the less that holds. noc_async_read and its barrier add a counter update, a command-buffer-ready
// poll and address writes before the send, and a call and a status-register poll after it. The re-send leaves only the
// send and an L1 poll.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"

namespace tile_read {

constexpr uint32_t kLandingOffsetMask = sizeof(kernel_profiler::TileSyncScratch::landing) - 1;
static_assert(sizeof(kernel_profiler::TileSyncScratch::landing) % NOC_DRAM_READ_ALIGNMENT_BYTES == 0);

// Reads the word at `addr` on the tile at `coord` over NoC `noc` and returns it.
inline uint32_t read(uint32_t noc, uint32_t coord, uint32_t addr, uint32_t scratch) {
    const uint32_t dst = scratch + (addr & kLandingOffsetMask);
    // This doesn't use noc_async_read_barrier, because the barrier made the tile offsets measurably less accurate.
    const uint32_t before = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
        noc, coord, uint64_t{addr}, dst, sizeof(uint32_t));
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

// Reads the partner's wall clock `reps` times, each between two reads of this tile's own wall clock. For rep i it calls
// on_reading(i, d), where d is the partner's reading minus the midpoint of the two own reads, doubled. The request is
// programmed once, by the read() before the loop, and re-sent each rep by writing NOC_CMD_CTRL alone, because accepting
// a request clears only NOC_CMD_CTRL and leaves the other command registers as programmed. Only the send and the poll
// sit between the two wall clock reads, because every instruction between them widens the window the partner's reading
// can fall in. It is marked cold because without it the tile offsets are measurably less accurate.
template <typename OnReading>
__attribute__((noinline, cold)) inline void bracket(
    uint32_t noc, uint32_t coord, uint32_t scratch, volatile uint32_t* wall, uint32_t reps, OnReading on_reading) {
    uint32_t prev = read(noc, coord, RISCV_DEBUG_REG_WALL_CLOCK_L, scratch);
    volatile tt_l1_ptr uint32_t* const land =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + (RISCV_DEBUG_REG_WALL_CLOCK_L & kLandingOffsetMask));
    volatile uint32_t* const ctrl =
        reinterpret_cast<volatile uint32_t*>(NOC_CMD_BUF_INSTANCE_OFFSET(noc, read_cmd_buf) + NOC_CMD_CTRL);
    constexpr uint32_t kFarFromPrevious = 0x80000000u;
    for (uint32_t i = 0; i < reps; i++) {
        // The re-send bypasses the NoC API, so this counts it as the API would, which keeps the issued count matching
        // the NoC's response count.
        noc_async_read_inc_num_issued(1, noc);
        const uint32_t sentinel = prev ^ kFarFromPrevious;
        *land = sentinel;
        const uint32_t wall_before = *wall;
        *ctrl = NOC_CTRL_SEND_REQ;
        do {
            invalidate_l1_cache();
        } while (*land == sentinel);
        const uint32_t wall_after = *wall;
        const uint32_t partner_wall = *land;
        prev = partner_wall;
        on_reading(
            i,
            2 * static_cast<int64_t>(static_cast<int32_t>(partner_wall - wall_before)) -
                static_cast<int32_t>(wall_after - wall_before));
    }
}

constexpr uint32_t kWarmup = 16;
constexpr uint32_t kReps = 256;
constexpr uint32_t kBins = kernel_profiler::kTileSyncBins;

// Returns the bin holding the rank-th smallest reading, as an offset from the centre bin. rank must be at most the
// histogram's count.
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
    volatile tt_l1_ptr kernel_profiler::TileSyncPartner& out) {
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    for (uint32_t bin = 0; bin < kBins; bin++) {
        hist[bin] = 0;
    }
    int64_t warm[kWarmup];
    bracket(noc, coord, scratch, wall, kWarmup, [&](uint32_t i, int64_t doubled_offset) {
        uint32_t j = i;
        for (; j > 0 && warm[j - 1] > doubled_offset; j--) {
            warm[j] = warm[j - 1];
        }
        warm[j] = doubled_offset;
    });
    const int64_t centre = warm[kWarmup / 2];
    bracket(noc, coord, scratch, wall, kReps, [&](uint32_t, int64_t doubled_offset) {
        hist[std::clamp<int64_t>(doubled_offset - centre + kBins / 2, 0, kBins - 1)]++;
    });
    out.doubled_median = static_cast<int32_t>(centre + quantile(hist, kReps / 2 + 1));
    out.whole_difference = static_cast<int64_t>(wall64(noc, coord, scratch) - own_wall64());
}

}  // namespace tile_read

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t partner_count = get_arg_val<uint32_t>(1);
    constexpr uint32_t kFirstPartnerArg = 2;
    volatile tt_l1_ptr kernel_profiler::TileSyncScratch* tile_scratch =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::TileSyncScratch*>(scratch);
    volatile tt_l1_ptr kernel_profiler::TileSyncTable& table = tile_scratch->table;
    table.ready = kernel_profiler::TileSyncReady::Up;
    kernel_profiler::TileSyncGo go;
    do {
        invalidate_l1_cache();
        go = table.go;
    } while (go == kernel_profiler::TileSyncGo::Wait);
    if (go == kernel_profiler::TileSyncGo::Measure) {
        // noc_read_with_state needs a NoC's read command buffer set up for reads, and a tile reads over both NoCs.
        for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
            noc_read_init_state<read_cmd_buf>(noc);
        }
        for (uint32_t k = 0; k < partner_count; k++) {
            const auto partner =
                kernel_profiler::word_as<kernel_profiler::TileSyncRead>(get_arg_val<uint32_t>(kFirstPartnerArg + k));
            tile_read::measure(
                partner.noc, NOC_XY_ENCODING(partner.x, partner.y), scratch, tile_scratch->hist, table.partner[k]);
        }
        std::atomic_thread_fence(std::memory_order_release);
        table.ready = kernel_profiler::TileSyncReady::Done;
        do {
            invalidate_l1_cache();
        } while (table.go != kernel_profiler::TileSyncGo::Exit);
    }
    // On a dispatch core the scratch is in the profiler's ring space, which must be zero for its first frame.
    volatile tt_l1_ptr uint32_t* scratch_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < sizeof(kernel_profiler::TileSyncScratch) / 4; i++) {
        scratch_words[i] = 0;
    }
}
