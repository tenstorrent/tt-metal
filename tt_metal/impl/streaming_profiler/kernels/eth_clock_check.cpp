// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs on a spare idle eth tile when the sync check is on. The sync check measures the accuracy of the clock sync
// between chips. This kernel samples the refclk and the wall clock the way the wall-clock core's sampler does and sends
// the readings to the host.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring");

// How many recent updates are held back, so that the ones leading up to a change can be sent too.
constexpr uint32_t kHistory = 8;
// How many updates are sent around a change, including the kHistory before it.
constexpr uint32_t kDenseRun = 64;
// How far an update may land, in eighths of a cycle, from where the previous update and the rate measured at the last
// change predict, before it counts as a change. While AICLK is steady, an update is within a few eighths of that.
constexpr int32_t kChangeEighths = 16;
constexpr uint32_t kStopPollMask = 255u;

// The range, in nops, of the random pad before each pass. It covers a refclk tick (20 ns, which is 16 to 27 cycles
// across the AICLK range) as well as a block. With a smaller range, the loop's own length locks the passes to the
// refclk, so every reading lands on the same cycle relative to the refclk edge and their errors stop averaging out.
constexpr uint32_t kCheckPadRange = 64;

// Reads the refclk and the wall clock in blocks of four reads, up to 22 blocks, until the refclk steps, and returns
// whether it did. On a step, walls and refclks get the reads around it. Calibration and sampling call the same
// 64-byte-aligned copy of the pass, so calibration measures exactly the reads that sample, regardless of the code
// around them.
__attribute__((noinline, aligned(64))) bool pass(uint32_t (&walls)[3], uint32_t (&refclks)[4]) {
    eth_clock::Block blocks[3];
    const auto read = [](eth_clock::Block& block) __attribute__((always_inline)) {
        block.first_refclk = eth_ptp::kRefclkLo.read();
        block.second_refclk = eth_ptp::kRefclkLo.read();
        block.wall = eth_ptp::kWallClockLo.read();
        block.last_refclk = eth_ptp::kRefclkLo.read();
    };
    bool stepped = false;
    eth_clock::disable_branch_predictor();
    read(blocks[2]);
    read(blocks[0]);
#pragma GCC unroll 20
    for (uint32_t k = 0; k < 20; k++) {
        eth_clock::Block& cur = blocks[(k + 1) % 3];
        const eth_clock::Block& prev = blocks[k % 3];
        const eth_clock::Block& prev2 = blocks[(k + 2) % 3];
        read(cur);
        if (prev.last_refclk != prev2.last_refclk) {
            walls[0] = prev2.wall;
            walls[1] = prev.wall;
            walls[2] = cur.wall;
            refclks[0] = prev2.last_refclk;
            refclks[1] = prev.first_refclk;
            refclks[2] = prev.second_refclk;
            refclks[3] = prev.last_refclk;
            stepped = true;
            break;
        }
    }
    eth_clock::enable_branch_predictor();
    return stepped;
}
// Marks a gap whose updates run() drops. Real gaps sit within a few cycles of the wall read.
constexpr int32_t kDropGap = -128;
// Runs one pass and returns 1 if it wrote an update to slot. With period 0, for calibration, the update written is the
// length of the block after the step, which is never the block right after the pad. Otherwise an update is written only
// if its blocks are period apart and its gap's position isn't kDropGap. It holds the new refclk and the step's wall
// time in eighths, which is the block's wall read plus the gap's position.
FORCE_INLINE uint32_t
run(uint32_t period, eth_clock::GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot, uint32_t& random_state) {
    uint32_t walls[3], refclks[4];
    eth_clock::nops<kCheckPadRange>(4 * eth_clock::draw(random_state, kCheckPadRange));
    if (!pass(walls, refclks)) {
        return 0;
    }
    if (period == 0) {
        slot[0] = walls[2] - walls[1];
        return 1;
    }
    const uint32_t gap = refclks[1] != refclks[0] ? 0u : refclks[2] != refclks[1] ? 1u : 2u;
    const int32_t gap_position = gap_positions.at(gap);
    if (walls[1] - walls[0] != period || walls[2] - walls[1] != period || gap_position == kDropGap) {
        return 0;
    }
    slot[0] = refclks[gap + 1];
    slot[1] = (walls[1] << kernel_profiler::kWallEighthBits) + static_cast<uint32_t>(gap_position);
    return 1;
}

constexpr kernel_profiler::SyncMeta kMetaDense{.dense = 1, .kind = kernel_profiler::SyncKind::Check};
constexpr kernel_profiler::SyncMeta kMetaThin{.kind = kernel_profiler::SyncKind::Check};

// Decides which updates to send. The measured error peaks within a few microseconds of a clock change, so every update
// around a change is sent (kDenseRun of them). Elsewhere only one in kSyncCheckKeepEvery is sent. An update counts as a
// change when it is more than kChangeEighths off the rate measured at the last change. The rate at a change is measured
// from the two updates on either side of it, so it mixes the old and new rates. The next update is then off that mixed
// rate too and counts as a change again, and the rate is measured again from two updates that are both past the change.
struct Check {
    struct Output {
        eth_clock::WallClockWriter<kCtrlAddr, kSyncRingAddr> writer;
        uint32_t since_kept = 0, dense_left = 0;
        int32_t wall_per_refclk_eighths = 0;
        __attribute__((noinline)) void release(const kernel_profiler::SyncSample& update) {
            if (dense_left == 0) {
                if (++since_kept != kernel_profiler::kSyncCheckKeepEvery) {
                    return;
                }
                since_kept = 0;
            }
            writer.add(update.refclk, update.wall_eighths, 0, dense_left != 0 ? kMetaDense : kMetaThin);
        }
    };
    // The output and the history are held by reference because release() takes the output's address and the history
    // is indexed by a variable. Either one held as a member would force the whole Check, counters included, into memory
    // instead of registers.
    Output& output;
    kernel_profiler::SyncSample (&held)[kHistory];
    uint32_t held_count = 0, held_next = 0;

    FORCE_INLINE void on_update(uint32_t refclk, uint32_t wall_eighths) {
        if (held_count != 0) {
            const kernel_profiler::SyncSample& prev = held[(held_next + kHistory - 1) % kHistory];
            const auto refclk_step = static_cast<int32_t>(refclk - prev.refclk),
                       wall_eighths_step = static_cast<int32_t>(wall_eighths - prev.wall_eighths);
            const int32_t off_line_eighths = wall_eighths_step - output.wall_per_refclk_eighths * refclk_step;
            if (output.wall_per_refclk_eighths == 0 || std::abs(off_line_eighths) > kChangeEighths) {
                if (output.wall_per_refclk_eighths != 0) {
                    output.dense_left = kDenseRun;
                    release_held();
                }
                output.wall_per_refclk_eighths = (wall_eighths_step + refclk_step / 2) / refclk_step;
            }
        }
        if (held_count == kHistory) {
            output.release(held[held_next]);
            held_count--;
        }
        held[held_next] = {refclk, wall_eighths};
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
    uint32_t random_state = eth_ptp::kWallClockLo.read() | 1u;
    eth_clock::Calibration calibration = eth_clock::calibrate(
        [&](uint32_t period, eth_clock::GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot) {
            return run(period, gap_positions, slot, random_state);
        },
        ctrl);
    // Only updates caught between the two back-to-back refclk reads are kept. Readings from the other two gaps span the
    // wall read or the block boundary, and are up to 12 ns off under di/dt. Which gap catches an update is random,
    // so the kept readings still sample every moment equally.
    calibration.gap_positions = eth_clock::GapPositions::of(kDropGap, calibration.gap_positions.at(1), kDropGap);
    Check::Output output;
    kernel_profiler::SyncSample held[kHistory];
    Check check{output, held};
    if (ctrl->stop == 0u) {
        output.writer.wall_eighths = eth_ptp::read_instant().wall << kernel_profiler::kWallEighthBits;
        uint32_t iter = 0;
        while (true) {
            uint32_t update[2];
            if (run(calibration.period, calibration.gap_positions, update, random_state) != 0) {
                check.on_update(update[0], update[1]);
            }
            if ((++iter & kStopPollMask) != 0u) {
                continue;
            }
            invalidate_l1_cache();
            if (ctrl->stop != 0u) {
                break;
            }
        }
        check.release_held();
    }
    output.writer.close();
    std::atomic_thread_fence(std::memory_order_release);
    ctrl->done = kernel_profiler::kResidentDoneWord;
}
