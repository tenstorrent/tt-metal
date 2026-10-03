// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"

namespace eth_clock {
namespace eth_ptp = tt::tt_metal::eth_ptp;
FORCE_INLINE uint32_t xorshift(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
// Bit 1 of Blackhole's RISC configuration CSR, 0x7c0, disables branch prediction. A pass runs with it disabled, since a
// branch taken on the previous pass would otherwise mispredict on this one.
FORCE_INLINE void disable_branch_predictor() { asm volatile("csrrsi zero, 0x7c0, 2" ::: "memory"); }
FORCE_INLINE void enable_branch_predictor() { asm volatile("csrrci zero, 0x7c0, 2" ::: "memory"); }
// Uniform over [0, range), range at most 2^16.
FORCE_INLINE uint32_t draw(uint32_t& walk, uint32_t range) { return ((xorshift(walk) >> 16) * range) >> 16; }
// Runs `count` of Length nops, one cycle each; `count` is at most Length.
template <uint32_t Length>
FORCE_INLINE void nops(uint32_t count) {
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "slli t1, %[count], 2\n\t"
        "auipc t0, 0\n\t"
        "addi t0, t0, 16 + %[n] * 4\n\t"
        "sub t0, t0, t1\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n\t"
        ".option pop"
        :
        : [count] "r"(count), [n] "i"(Length)
        : "t0", "t1", "memory");
}
// Each of the three gaps' positions in eighths, one signed byte per gap in one word, since the stream takes it in a
// register.
struct GapPos8 {
    uint32_t word = 0;
    static constexpr GapPos8 of(int32_t gap0, int32_t gap1, int32_t gap2) {
        return {
            (static_cast<uint32_t>(gap0) & 0xFFu) | (static_cast<uint32_t>(gap1) & 0xFFu) << 8 |
            (static_cast<uint32_t>(gap2) & 0xFFu) << 16};
    }
    FORCE_INLINE constexpr int32_t at(uint32_t gap) const { return static_cast<int8_t>(word >> (8 * gap)); }
};
struct Calibration {
    uint32_t period;
    GapPos8 pos8;
};
// Measures the kernel's own pass: its block period, the most common wall-clock length of a block, and where each of the
// pass's three gaps between refclk reads sits against the block's wall read. Passes start at a random phase, so a gap
// catches updates in proportion to its width: the widths are the period split by each gap's share of the updates, laid
// out from the read just before the wall read. Then it waits for the host's go. `run_pass(period, pos8, slot)` runs one
// pass of the kernel's sampling reads and returns 1 if it wrote an update.
template <class RunPass>
Calibration calibrate(RunPass run_pass, volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl) {
    constexpr uint32_t kPeriodCalibrationPasses = 1024;
    constexpr uint32_t kMinCalibrationUpdates = 1u << 16;
    constexpr uint32_t kMaxCalibrationUpdates = 1u << 20;
    constexpr uint32_t kPeriodBins = 64;
    constexpr uint32_t kControlPollMask = 1023u;
    Calibration calibration{};
    // Static: it's larger than an idle ERISC kernel's 192 B guaranteed stack.
    static uint32_t hist[kPeriodBins];
    uint32_t slot_words[2];
    volatile tt_l1_ptr uint32_t* slot = slot_words;
    for (uint32_t i = 0; i < kPeriodCalibrationPasses; i++) {
        if (run_pass(0, GapPos8{}, slot) != 0) {
            hist[slot[0] & (kPeriodBins - 1u)]++;
        }
    }
    for (uint32_t i = 1; i < kPeriodBins; i++) {
        calibration.period = hist[i] > hist[calibration.period] ? i : calibration.period;
    }
    uint32_t updates_per_gap[3] = {}, total = 0;
    for (uint32_t i = 1; ctrl->stop == 0u; i++) {
        // Gap p sits p eighths in, so an update's low three bits name its gap.
        if (run_pass(calibration.period, GapPos8::of(0, 1, 2), slot) != 0) {
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
    // The cap is checked every kControlPollMask + 1 passes, so total stays small enough that this can't overflow.
    static_assert(uint64_t{64} * (kPeriodBins - 1) * (kMaxCalibrationUpdates + kControlPollMask) <= UINT32_MAX);
    int32_t width64[3];
    for (uint32_t gap = 0; gap < 3; gap++) {
        width64[gap] = static_cast<int32_t>((64u * calibration.period * updates_per_gap[gap]) / total);
    }
    const int32_t second_refclk_pos64 = -64;  // one cycle before the wall read
    const int32_t centre64[3] = {
        second_refclk_pos64 - width64[1] - width64[0] / 2,
        second_refclk_pos64 - width64[1] / 2,
        second_refclk_pos64 + width64[2] / 2};
    calibration.pos8 = GapPos8::of((centre64[0] + 4) >> 3, (centre64[1] + 4) >> 3, (centre64[2] + 4) >> 3);
    return calibration;
}

// Packs a kernel's clock points into SyncLocalRecords on its sync ring, which the eth relay drains. Consecutive points
// that share a meta go into one record, up to kSyncLocalPoints, while each one's step from the first fits a
// SyncLocalStep. A record the ring has no room for is dropped and counted.
template <uint32_t CtrlAddr, uint32_t RingAddr>
class ClockPointWriter {
public:
    void close() {
        flush();
        ctrl()->dropped_sync = dropped;
    }
    // A point: the refclk and the wall clock in eighths at one instant, the rate there, and the rate the record's later
    // points are measured against if this point starts one.
    __attribute__((noinline)) void add(
        uint64_t refclk,
        uint64_t wall8,
        uint32_t wall_per_refclk_eighths,
        uint32_t base_wall_per_refclk_eighths,
        kernel_profiler::SyncMeta record_meta) {
        if (points != 0) {
            if (kernel_profiler::word_of(record_meta) == kernel_profiler::word_of(meta)) {
                const int32_t wall_off = static_cast<int32_t>(
                    static_cast<uint32_t>(wall8) - static_cast<uint32_t>(first_wall8) -
                    rates.base_wall_per_refclk_eighths * static_cast<uint32_t>(refclk - first_refclk));
                if (kernel_profiler::sync_local_step_fits(refclk - first_refclk, wall_off)) {
                    from_first[points - 1] = kernel_profiler::word_of(kernel_profiler::SyncLocalStep{
                        .refclk_from_first = static_cast<uint32_t>(refclk - first_refclk), .wall_off = wall_off});
                    rates.wall_per_refclk_eighths[points] = static_cast<uint8_t>(wall_per_refclk_eighths);
                    if (++points == kernel_profiler::kSyncLocalPoints) {
                        flush();
                    }
                    return;
                }
            }
            flush();
        }
        first_refclk = refclk;
        first_wall8 = wall8;
        rates.wall_per_refclk_eighths[0] = static_cast<uint8_t>(wall_per_refclk_eighths);
        rates.base_wall_per_refclk_eighths = static_cast<uint8_t>(base_wall_per_refclk_eighths);
        meta = record_meta;
        points = 1;
    }

private:
    static volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl() {
        return reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(CtrlAddr);
    }
    void emit() {
        invalidate_l1_cache();
        if (tail - ctrl()->sync_head >= kernel_profiler::kSyncRingRecords) {
            dropped++;
            return;
        }
        volatile tt_l1_ptr kernel_profiler::SyncLocalRecord* record =
            &reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncRecord*>(
                 RingAddr)[tail % kernel_profiler::kSyncRingRecords]
                 .local;
        kernel_profiler::SyncMeta counted_meta = meta;
        counted_meta.count = points;
        record->meta = kernel_profiler::word_of(counted_meta);
        record->rates = kernel_profiler::word_of(rates);
        record->first_refclk = first_refclk;
        record->first_wall8 = first_wall8;
        for (uint32_t i = 0; i < kernel_profiler::kSyncLocalPoints - 1; i++) {
            record->from_first[i] = from_first[i];
        }
        std::atomic_thread_fence(std::memory_order_release);
        ctrl()->sync_tail = ++tail;
    }
    void flush() {
        if (points != 0) {
            emit();
            points = 0;
        }
    }

    uint32_t tail = 0, dropped = 0;
    uint32_t points = 0, from_first[kernel_profiler::kSyncLocalPoints - 1] = {};
    kernel_profiler::SyncMeta meta{};
    kernel_profiler::SyncLocalRates rates{};
    uint64_t first_refclk = 0, first_wall8 = 0;
};

}  // namespace eth_clock
