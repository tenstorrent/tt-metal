// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Helpers for the clock sync's eth kernels: random draws, cycle-exact delays, branch prediction control, the
// calibration of a sampling pass, and the writer that packs clock points into sync records.
//
// A sampling pass reads the clocks in blocks of four loads (refclk, refclk, wall clock, refclk) until the refclk
// changes. The change falls in one of the block's three gaps, which lie before its first refclk read, between its first
// and second refclk reads, and around its wall clock read. The wall clock read times the change to within that gap.

#pragma once

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"

namespace eth_clock {
// Steps the xorshift generator in random_state and returns a uniform draw from [0, range). range must be at most 2^16.
FORCE_INLINE uint32_t draw(uint32_t& random_state, uint32_t range) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return ((random_state >> 16) * range) >> 16;
}
// Setting bit 1 of Blackhole's RISC configuration CSR (0x7c0) disables branch prediction. Passes run with it disabled
// so that a branch taken on the previous pass can't mispredict and change this pass's timing.
FORCE_INLINE void disable_branch_predictor() { asm volatile("csrrsi zero, 0x7c0, 2" ::: "memory"); }
FORCE_INLINE void enable_branch_predictor() { asm volatile("csrrci zero, 0x7c0, 2" ::: "memory"); }
// Runs `bytes` / 4 one-cycle nops by jumping into a run of Length of them. `bytes` must be a multiple of 4 and at most
// 4 * Length. A C loop's cost per iteration varies with its branch, so only jumping into a run of nops makes each nop
// exactly one cycle.
template <uint32_t Length>
FORCE_INLINE void nops(uint32_t bytes) {
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "auipc t0, 0\n\t"
        "sub t0, t0, %[bytes]\n\t"
        "jr 12 + %[n] * 4(t0)\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n\t"
        ".option pop"
        :
        : [bytes] "r"(bytes), [n] "i"(Length)
        : "t0", "memory");
}
// The four reads of one block of a pass, in the order the pass makes them.
struct Block {
    uint32_t first_refclk, second_refclk, wall, last_refclk;
};
// Where each of a block's three gaps sits, in eighths of a cycle from the block's wall clock read, one signed byte per
// gap. They are packed into one word because the sampling loop keeps them in a register.
struct GapPositions {
    uint32_t word = 0;
    static constexpr GapPositions of(int32_t gap0, int32_t gap1, int32_t gap2) {
        return {
            (static_cast<uint32_t>(gap0) & 0xFFu) | (static_cast<uint32_t>(gap1) & 0xFFu) << 8 |
            (static_cast<uint32_t>(gap2) & 0xFFu) << 16};
    }
    FORCE_INLINE constexpr int32_t at(uint32_t gap) const { return static_cast<int8_t>(word >> (8 * gap)); }
};
struct Calibration {
    uint32_t period;
    GapPositions gap_positions;
};
constexpr uint32_t kGaps = 3;
// Calibration works in 64ths of a cycle and rounds to the eighths GapPositions holds.
constexpr uint32_t kSixtyFourthsPerCycle = 64;
constexpr int32_t sixty_fourths_to_eighths(int32_t sixty_fourths) { return (sixty_fourths + 4) >> 3; }
// Calibrates the sampling pass, then waits for the host's go. Calibration measures the pass's period, which is the most
// common block length in wall ticks, and where each gap sits relative to the wall read. Passes start at a random phase,
// so a gap's width is the period times its share of the caught changes. `run_pass(period, gap_positions, slot)` runs
// one pass and returns 1 if it wrote an update.
template <class RunPass>
Calibration calibrate(RunPass run_pass, volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl) {
    constexpr uint32_t kPeriodCalibrationPasses = 1024;
    constexpr uint32_t kMinCalibrationUpdates = 1u << 16;
    constexpr uint32_t kMaxCalibrationUpdates = 1u << 20;
    constexpr uint32_t kPeriodBins = 64;
    constexpr uint32_t kControlPollMask = 1023u;
    Calibration calibration{};
    static uint32_t hist[kPeriodBins];
    uint32_t slot_words[2];
    volatile tt_l1_ptr uint32_t* slot = slot_words;
    for (uint32_t i = 0; i < kPeriodCalibrationPasses; i++) {
        if (run_pass(0, GapPositions{}, slot) != 0) {
            hist[slot[0] & (kPeriodBins - 1u)]++;
        }
    }
    for (uint32_t i = 1; i < kPeriodBins; i++) {
        calibration.period = hist[i] > hist[calibration.period] ? i : calibration.period;
    }
    uint32_t updates_per_gap[kGaps] = {}, total = 0;
    for (uint32_t i = 1; ctrl->stop == 0u; i++) {
        // These passes give gap p position p eighths, so an update's low three bits say which gap caught it.
        if (run_pass(calibration.period, GapPositions::of(0, 1, 2), slot) != 0) {
            updates_per_gap[slot[1] & 7u]++;
            total++;
        }
        if ((i & kControlPollMask) == 0u) {
            ctrl->heartbeat++;
            invalidate_l1_cache();
            if ((total >= kMinCalibrationUpdates && ctrl->go != 0u) || total >= kMaxCalibrationUpdates) {
                break;
            }
        }
    }
    while (ctrl->go == 0u && ctrl->stop == 0u) {
        ctrl->heartbeat++;
        invalidate_l1_cache();
    }
    // total is checked against the cap only every kControlPollMask + 1 passes, so it can exceed the cap by that much.
    static_assert(
        uint64_t{kSixtyFourthsPerCycle} * (kPeriodBins - 1) * (kMaxCalibrationUpdates + kControlPollMask) <=
        UINT32_MAX);
    int32_t width_sixty_fourths[kGaps];
    for (uint32_t gap = 0; gap < kGaps; gap++) {
        width_sixty_fourths[gap] =
            static_cast<int32_t>((kSixtyFourthsPerCycle * calibration.period * updates_per_gap[gap]) / total);
    }
    // The block's loads issue in consecutive cycles (measured), so the second refclk read is one cycle before the wall
    // read.
    const int32_t second_refclk_position_sixty_fourths = -static_cast<int32_t>(kSixtyFourthsPerCycle);
    const int32_t centre_sixty_fourths[kGaps] = {
        second_refclk_position_sixty_fourths - width_sixty_fourths[1] - width_sixty_fourths[0] / 2,
        second_refclk_position_sixty_fourths - width_sixty_fourths[1] / 2,
        second_refclk_position_sixty_fourths + width_sixty_fourths[2] / 2};
    calibration.gap_positions = GapPositions::of(
        sixty_fourths_to_eighths(centre_sixty_fourths[0]),
        sixty_fourths_to_eighths(centre_sixty_fourths[1]),
        sixty_fourths_to_eighths(centre_sixty_fourths[2]));
    return calibration;
}

// Packs a kernel's clock points into SyncWallClockRecords on its sync ring. Consecutive points with the same meta share
// a record, up to kSyncWallClockPoints of them. If the ring has no room for a record, the record is dropped and
// counted.
template <uint32_t CtrlAddr, uint32_t RingAddr>
class WallClockWriter {
public:
    // The full wall clock (in eighths) at the latest point. Points carry only their low word, and add() rebuilds each
    // point's full value from this one, so the kernel sets it to a wall clock read before the first point.
    uint64_t wall_eighths = 0;

    void close() {
        flush();
        ctrl()->sync.dropped = dropped;
    }
    // Adds a point from the low words of the refclk and the wall clock (in eighths) at one instant, and the rate there.
    __attribute__((noinline)) void add(
        uint32_t refclk, uint32_t wall_eighths_lo, uint32_t wall_per_refclk_eighths, kernel_profiler::SyncMeta meta) {
        wall_eighths += static_cast<int32_t>(wall_eighths_lo - static_cast<uint32_t>(wall_eighths));
        if (record.meta.count != 0 && record.meta.dense != meta.dense) {
            flush();
        }
        if (record.meta.count == 0) {
            record.meta = meta;
            record.first_wall_eighths_hi = static_cast<uint32_t>(wall_eighths >> 32);
        }
        const uint32_t i = record.meta.count++;
        record.wall_per_refclk_eighths[i] = static_cast<uint8_t>(wall_per_refclk_eighths);
        record.points[i] = {.refclk_lo = refclk, .wall_eighths_lo = wall_eighths_lo};
        if (record.meta.count == kernel_profiler::kSyncWallClockPoints) {
            flush();
        }
    }

private:
    static volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl() {
        return reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(CtrlAddr);
    }
    void flush() {
        if (record.meta.count == 0) {
            return;
        }
        invalidate_l1_cache();
        if (tail - ctrl()->sync.head >= kernel_profiler::kSyncRingRecords) {
            dropped++;
        } else {
            reinterpret_cast<tt_l1_ptr kernel_profiler::SyncRecord*>(RingAddr)[tail % kernel_profiler::kSyncRingRecords]
                .wall_clock = record;
            std::atomic_thread_fence(std::memory_order_release);
            ctrl()->sync.tail = ++tail;
        }
        record.meta.count = 0;
    }

    uint32_t tail = 0, dropped = 0;
    kernel_profiler::SyncWallClockRecord record{};
};

}  // namespace eth_clock
