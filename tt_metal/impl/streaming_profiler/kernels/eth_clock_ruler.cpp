// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring_addr");

namespace eth_ptp = tt::tt_metal::eth_ptp;

constexpr uint32_t kHistory = 8;      // updates held back, so a change's run-up goes out dense
constexpr uint32_t kDenseAfter = 64;  // a change's dense run in arrivals: its kHistory run-up and the rest after it
constexpr int64_t kOffLine8 = 16;     // two ticks: a steady update sits within a few eighths of the line
constexpr uint32_t kStopPollMask = 255u;

// Covers the refclk's advance (20 ns, 16-27 cycles across the AICLK range) as well as a block; without that, the
// loop's own length phase-locks the passes to the refclk and each chip's updates settle on one AICLK cycle of the
// crossing.
constexpr uint32_t kRulerPadRange = 64;
constexpr int32_t kDropGap = -128;

// One 64-byte-aligned copy of the pass serves both calibration and sampling, so calibration measures exactly the reads
// that sample, whatever code surrounds them.
__attribute__((noinline, aligned(64))) uint32_t pass(uint32_t (&walls)[3], uint32_t (&refclks)[4]) {
    struct Block {
        uint32_t wall, first_refclk, second_refclk, last_refclk;
    };
    Block blocks[3];
    const auto read = [](Block& block) __attribute__((always_inline)) {
        block.first_refclk = eth_ptp::kRefclkLo.read();
        block.second_refclk = eth_ptp::kRefclkLo.read();
        block.wall = eth_ptp::kWallClockLo.read();
        block.last_refclk = eth_ptp::kRefclkLo.read();
    };
    uint32_t step_after = 0;
    eth_clock::disable_branch_predictor();
    read(blocks[2]);
    read(blocks[0]);
#pragma GCC unroll 20
    for (uint32_t k = 0; k < 20; k++) {
        Block& cur = blocks[(k + 1) % 3];
        const Block& prev = blocks[k % 3];
        const Block& prev2 = blocks[(k + 2) % 3];
        read(cur);
        if (prev.last_refclk != prev2.last_refclk) {
            walls[0] = prev2.wall;
            walls[1] = prev.wall;
            walls[2] = cur.wall;
            refclks[0] = prev2.last_refclk;
            refclks[1] = prev.first_refclk;
            refclks[2] = prev.second_refclk;
            refclks[3] = prev.last_refclk;
            step_after = k + 1;
            break;
        }
    }
    eth_clock::enable_branch_predictor();
    return step_after;
}
// Runs one pass and returns 1 if it wrote an update to slot. With period 0 an update is the length of the block after
// its step, which never follows the pad. Otherwise an update needs blocks period apart and a gap whose position isn't
// kDropGap, and holds the new refclk and the step's wall time in eighths: the block's wall read plus that position.
FORCE_INLINE uint32_t run(uint32_t period, eth_clock::GapPos8 pos8, volatile tt_l1_ptr uint32_t* slot, uint32_t& walk) {
    uint32_t walls[3], refclks[4];
    eth_clock::nops<kRulerPadRange>(eth_clock::draw(walk, kRulerPadRange));
    if (pass(walls, refclks) == 0) {
        return 0;
    }
    if (period == 0) {
        slot[0] = walls[2] - walls[1];
        return 1;
    }
    const uint32_t gap = refclks[1] != refclks[0] ? 0u : refclks[2] != refclks[1] ? 1u : 2u;
    const int32_t gap_pos8 = pos8.at(gap);
    if (walls[1] - walls[0] != period || walls[2] - walls[1] != period || gap_pos8 == kDropGap) {
        return 0;
    }
    slot[0] = refclks[gap + 1];
    slot[1] = (walls[1] << 3) + static_cast<uint32_t>(gap_pos8);
    return 1;
}

using Out = eth_clock::ClockPointWriter<kCtrlAddr, kSyncRingAddr>;
constexpr kernel_profiler::SyncMeta kMetaDense{.dense = 1, .kind = kernel_profiler::SyncKind::Ruler};
constexpr kernel_profiler::SyncMeta kMetaThin{.kind = kernel_profiler::SyncKind::Ruler};

// Decides which updates go out. Placement error peaks within a few microseconds of a clock change, so the updates
// around one go out dense (kDenseAfter). Elsewhere one in kSyncRulerKeepEvery does, and the host weights those by
// that. A change is a step more than kOffLine8 off the rate taken at the last change; that rate comes from the pair
// straddling it, so the update after a real change triggers once more and the rate settles on a clean pair.
struct Ruler {
    struct Update {
        uint64_t refclk, wall8;
    };
    struct Output {
        Out out;
        uint32_t thin = 0, dense_left = 0;
        int64_t wall_per_refclk_eighths = 0;
        __attribute__((noinline)) void release(const Update& update) {
            if (dense_left != 0) {
                out.add(update.refclk, update.wall8, 0, static_cast<uint32_t>(wall_per_refclk_eighths), kMetaDense);
            } else if (++thin == kernel_profiler::kSyncRulerKeepEvery) {
                thin = 0;
                out.add(update.refclk, update.wall8, 0, static_cast<uint32_t>(wall_per_refclk_eighths), kMetaThin);
            }
        }
    };
    // References, not members: release() takes the output's address and the history is indexed by a variable, so
    // either one held here would keep the whole Ruler in memory, counters included.
    Output& output;
    Update (&held)[kHistory];
    uint32_t held_count = 0, held_next = 0;

    FORCE_INLINE void on_update(uint64_t refclk, uint64_t wall8) {
        if (held_count != 0) {
            const Update& prev = held[(held_next + kHistory - 1) % kHistory];
            const auto refclk_step = static_cast<int64_t>(refclk - prev.refclk),
                       wall8_step = static_cast<int64_t>(wall8 - prev.wall8);
            const int64_t off_line8 = wall8_step - output.wall_per_refclk_eighths * refclk_step;
            if (output.wall_per_refclk_eighths == 0 || off_line8 > kOffLine8 || off_line8 < -kOffLine8) {
                if (output.wall_per_refclk_eighths != 0) {
                    output.dense_left = kDenseAfter;
                    release_held();
                }
                output.wall_per_refclk_eighths = (wall8_step + refclk_step / 2) / refclk_step;
            }
        }
        if (held_count == kHistory) {
            output.release(held[held_next]);
            held_count--;
        }
        held[held_next] = {refclk, wall8};
        held_next = (held_next + 1) % kHistory;
        held_count++;
        if (output.dense_left != 0) {
            output.dense_left--;
        }
    }
    FORCE_INLINE void release_held() {
        for (; held_count != 0; held_count--) {
            output.release(held[(held_next + kHistory - held_count) % kHistory]);
        }
    }
};

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    uint32_t walk = eth_ptp::kWallClockLo.read() | 1u;
    eth_clock::Calibration calibration = eth_clock::calibrate(
        [&](uint32_t period, eth_clock::GapPos8 pos8, volatile tt_l1_ptr uint32_t* slot) {
            return run(period, pos8, slot, walk);
        },
        ctrl);
    calibration.pos8 = eth_clock::GapPos8::of(kDropGap, calibration.pos8.at(1), kDropGap);
    Ruler::Output output;
    Ruler::Update held[kHistory];
    Ruler ruler{output, held};
    if (ctrl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        uint64_t refclk = start.refclk, wall8 = start.wall << 3;
        uint32_t iter = 0;
        while (true) {
            uint32_t update[2];
            if (run(calibration.period, calibration.pos8, update, walk) != 0) {
                refclk += update[0] - static_cast<uint32_t>(refclk);
                wall8 += update[1] - static_cast<uint32_t>(wall8);
                ruler.on_update(refclk, wall8);
            }
            if ((++iter & kStopPollMask) != 0u) {
                continue;
            }
            invalidate_l1_cache();
            if (ctrl->stop != 0u) {
                break;
            }
        }
        ruler.release_held();
    }
    output.out.close();
    std::atomic_thread_fence(std::memory_order_release);
    ctrl->done = kernel_profiler::kResidentDoneWord;
}
