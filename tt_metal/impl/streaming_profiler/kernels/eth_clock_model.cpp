// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// AICLK only takes whole FBDIV values, each a multiple of the crystal the refclk counts: wall_per_refclk_eighths =
// FBDIV * 8 / (REFDIV * postdiv0). So wherever AICLK holds, the wall clock gains exactly wall_per_refclk_eighths / 8
// ticks per refclk tick, and its samples lie on one line of that slope to within a sample's width.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring_addr");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring_addr");

// A window becomes a point at least once a millisecond, well within a SyncLocalStep's 16-bit refclk field.
constexpr uint32_t kPointTicks = kernel_profiler::kEthRefclkHz / 1000;
// A sample is placed to within half its read pair, at most 1.5 cycles, so a sample twice that far from the line is off
// it.
constexpr int32_t kOffEighths = 24;
// The sampler sees nearly every refclk update, one per 80 ns (4 ticks). A step of one in wall_per_refclk_eighths adds
// an eighth of a cycle per refclk tick, so after eight samples (32 ticks) it is 32 eighths off the line, past
// kOffEighths. A sample only joins a window once the kHoldSamples samples after it are on the line too.
constexpr uint32_t kHoldSamples = 8;
// Off a line, every kGroupSamples consecutive samples (~320 ns, over which a clock walk bends the wall ~0.06 cycles)
// become one point, their centroid. A new line opens once kSteadyPoints such points (~4 us) fit one rate; a PLL step
// takes ~1.3 us, so it can't fit a line in that span.
constexpr uint32_t kGroupSamples = 4;
constexpr uint32_t kSteadyPoints = 12;
// A window holds at most one sample per refclk update (4 ticks), so at this size it always closes at kPointTicks first.
constexpr uint32_t kMaxWindowSamples = 1u << 14;
static_assert(kMaxWindowSamples > kPointTicks / tt::tt_metal::eth_ptp::kRefclkTicksPerUpdate);

using Out = eth_clock::ClockPointWriter<kCtrlAddr, kSyncRingAddr>;
constexpr kernel_profiler::SyncMeta kMetaLocal{.kind = kernel_profiler::SyncKind::Local};

struct Anchor {
    uint64_t refclk = 0, wall8 = 0;
    // The 64-bit count nearest `base` whose low word is `lo`.
    static uint64_t widen(uint64_t base, uint32_t lo) {
        return base +
               static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(lo - static_cast<uint32_t>(base))));
    }
    void rebase(uint32_t refclk_lo, uint32_t wall8_lo) {
        refclk = widen(refclk, refclk_lo);
        wall8 = widen(wall8, wall8_lo);
    }
};

using Sample = kernel_profiler::SyncSample;

struct Model {
    // The line the samples are on (rate 0 when there's none): through `at` at wall_per_refclk_eighths, with
    // the mean residue of its `count` samples so far.
    struct Line {
        uint32_t wall_per_refclk_eighths = 0;
        Sample at{};
        int32_t mean = 0;
        int64_t sum = 0;
        // 64-bit: at a sample per refclk update, 32 bits wrap after under six minutes on one line.
        uint64_t count = 0;
        FORCE_INLINE int32_t residue(Sample sample) const {
            return static_cast<int32_t>(
                (sample.wall8 - at.wall8) - wall_per_refclk_eighths * (sample.refclk - at.refclk));
        }
        FORCE_INLINE bool holds(int32_t sample_residue) const {
            return static_cast<uint32_t>(sample_residue - mean + kOffEighths) <= 2u * kOffEighths;
        }
    };
    // The samples on the line since the last point. sum_residue is their residues minus the line's mean; only a window
    // close moves the mean, so it fits in 32 bits.
    struct Window {
        uint32_t first_refclk = 0;
        uint32_t sum_refclk_from_first = 0;
        int32_t sum_residue = 0;
        uint32_t count = 0, size = 1;
        uint32_t last_refclk = 0;
        // Adds a sample on `line`, and returns whether the window is ready to close. The first sample's wall goes to
        // first_wall8, outside the window, so a copy of the window held in registers leaves it out.
        FORCE_INLINE bool add(Sample sample, const Line& line, uint32_t& first_wall8) {
            if (count == 0) {
                first_refclk = sample.refclk;
                first_wall8 = sample.wall8;
                sum_refclk_from_first = 0;
                sum_residue = 0;
            }
            const uint32_t refclk_from_first = sample.refclk - first_refclk;
            sum_refclk_from_first += refclk_from_first;
            sum_residue += line.residue(sample) - line.mean;
            last_refclk = sample.refclk;
            return ++count == size || refclk_from_first >= kPointTicks;
        }
    };
    // A held sample's residue is recomputed when it joins the window, since a line change empties the hold.
    struct Hold {
        Sample samples[kHoldSamples];
        uint32_t begin = 0, count = 0;
    };
    // Off a line: the samples since the last point, and the last kSteadyPoints points.
    struct Group {
        Sample first{}, last{};
        uint32_t sum_refclk_from_first = 0, sum_wall8_from_first = 0, count = 0;
    };
    struct Recent {
        Sample points[kSteadyPoints];
        uint32_t count = 0, head = 0;
    };

    // First, so the fast loop addresses its slots with no offset.
    Hold hold;
    Line line;
    // The latest line's or group's rate, which each record's later points are measured against.
    uint32_t current_wall_per_refclk_eighths = 0;
    Window window;
    uint32_t window_first_wall8 = 0;
    Sample off_sample{};
    bool has_off = false;
    Group group;
    Recent recent;
    Anchor anchor;
    Out out;

    void point(uint32_t refclk, uint32_t wall8, uint32_t wall_per_refclk_eighths) {
        out.add(
            Anchor::widen(anchor.refclk, refclk),
            Anchor::widen(anchor.wall8, wall8),
            wall_per_refclk_eighths,
            current_wall_per_refclk_eighths,
            kMetaLocal);
    }
    __attribute__((noinline)) void close_window() {
        const uint32_t count = window.count;
        const uint32_t mean_refclk_from_first = (window.sum_refclk_from_first + count / 2u) / count;
        const int32_t half_count = static_cast<int32_t>(count / 2u);
        // The window's error against its first sample: the sum of each sample's residue minus the first's.
        const int32_t error_sum =
            window.sum_residue -
            static_cast<int32_t>(count) * (line.residue(Sample{window.first_refclk, window_first_wall8}) - line.mean);
        const int32_t mean_error =
            (error_sum + (error_sum < 0 ? -half_count : half_count)) / static_cast<int32_t>(count);
        anchor.rebase(window.first_refclk, window_first_wall8);
        point(
            window.first_refclk + mean_refclk_from_first,
            window_first_wall8 + line.wall_per_refclk_eighths * mean_refclk_from_first +
                static_cast<uint32_t>(mean_error),
            line.wall_per_refclk_eighths);
        line.sum += static_cast<int64_t>(window.sum_residue) + static_cast<int64_t>(count) * line.mean;
        line.count += count;
        line.mean = static_cast<int32_t>(line.sum / static_cast<int64_t>(line.count));
        window.count = 0;
        window.size = window.size < kMaxWindowSamples ? window.size * 2u : window.size;
    }
    FORCE_INLINE void window_add(Sample sample) {
        if (window.add(sample, line, window_first_wall8)) {
            close_window();
        }
    }
    __attribute__((noinline)) void close_group() {
        const uint32_t span = group.last.refclk - group.first.refclk;
        if (span != 0) {
            current_wall_per_refclk_eighths = ((group.last.wall8 - group.first.wall8) + span / 2u) / span;
        }
        // Every group but the capture's last closes full, so the division is nearly always by the constant.
        const auto divide = [&](uint32_t sum) {
            return group.count == kGroupSamples ? sum / kGroupSamples : sum / group.count;
        };
        const uint32_t mean_refclk_from_first = divide(group.sum_refclk_from_first),
                       refclk_remainder = group.sum_refclk_from_first - mean_refclk_from_first * group.count;
        const uint32_t mean_wall8_from_first =
            divide(group.sum_wall8_from_first - current_wall_per_refclk_eighths * refclk_remainder + group.count / 2u);
        anchor.rebase(group.first.refclk, group.first.wall8);
        const Sample centroid{group.first.refclk + mean_refclk_from_first, group.first.wall8 + mean_wall8_from_first};
        point(centroid.refclk, centroid.wall8, 0);
        recent.points[(recent.head + recent.count) % kSteadyPoints] = centroid;
        if (recent.count < kSteadyPoints) {
            recent.count++;
        } else {
            recent.head = (recent.head + 1) % kSteadyPoints;
        }
        group.count = 0;
        if (recent.count == kSteadyPoints) {
            try_line();
        }
    }
    FORCE_INLINE void group_add(Sample sample) {
        if (group.count == 0) {
            group.first = sample;
            group.sum_refclk_from_first = 0;
            group.sum_wall8_from_first = 0;
        }
        group.sum_refclk_from_first += sample.refclk - group.first.refclk;
        group.sum_wall8_from_first += sample.wall8 - group.first.wall8;
        group.last = sample;
        if (++group.count == kGroupSamples) {
            close_group();
        }
    }
    __attribute__((noinline)) void try_line() {
        const Sample oldest = recent.points[recent.head];
        const Sample& newest = recent.points[(recent.head + kSteadyPoints - 1) % kSteadyPoints];
        const uint32_t span = newest.refclk - oldest.refclk;
        Line candidate{.wall_per_refclk_eighths = ((newest.wall8 - oldest.wall8) + span / 2u) / span, .at = oldest};
        int32_t residues[kSteadyPoints];
        int32_t sum = 0;
        for (uint32_t i = 0; i < kSteadyPoints; i++) {
            residues[i] = candidate.residue(recent.points[(recent.head + i) % kSteadyPoints]);
            sum += residues[i];
        }
        candidate.mean = sum / static_cast<int32_t>(kSteadyPoints);
        for (uint32_t i = 0; i < kSteadyPoints; i++) {
            if (!candidate.holds(residues[i])) {
                return;
            }
        }
        line.wall_per_refclk_eighths = candidate.wall_per_refclk_eighths;
        line.at = oldest;
        line.mean = candidate.mean;
        line.sum = sum;
        line.count = kSteadyPoints;
        current_wall_per_refclk_eighths = candidate.wall_per_refclk_eighths;
        window.count = 0;
        window.size = 1;
        hold.count = 0;
        has_off = false;
    }
    // End the line with a point where its samples last held it. Otherwise the chord from the last window's centroid, up
    // to half a window back, into the first group would leave the line well before the clock did.
    __attribute__((noinline)) void end_line(Sample sample) {
        if (window.count != 0) {
            close_window();
        }
        if (line.count > kSteadyPoints) {
            const uint32_t last_refclk = window.last_refclk;
            point(
                last_refclk,
                line.at.wall8 + line.wall_per_refclk_eighths * (last_refclk - line.at.refclk) +
                    static_cast<uint32_t>(line.mean),
                line.wall_per_refclk_eighths);
        }
        line.wall_per_refclk_eighths = 0;
        recent.count = 0;
        recent.head = 0;
        group.count = 0;
        for (uint32_t i = 0; i < hold.count; i++) {
            group_add(hold.samples[(hold.begin + i) % kHoldSamples]);
        }
        hold.count = 0;
        group_add(off_sample);
        group_add(sample);
        has_off = false;
    }
    FORCE_INLINE void on_sample(Sample sample) {
        if (line.wall_per_refclk_eighths == 0) {
            group_add(sample);
            return;
        }
        if (!line.holds(line.residue(sample))) {
            if (has_off) {
                end_line(sample);
                return;
            }
            off_sample = sample;
            has_off = true;
            return;
        }
        has_off = false;
        if (hold.count == kHoldSamples) {
            window_add(hold.samples[hold.begin]);
            hold.begin = (hold.begin + 1) % kHoldSamples;
            hold.count--;
        }
        hold.samples[(hold.begin + hold.count) % kHoldSamples] = sample;
        hold.count++;
    }
    // Feeds the samples [cursor, end) in ring order. On a line, with a full hold and nothing pending off it, the state
    // a sample touches stays in registers; anything else goes through on_sample.
    void feed(const volatile tt_l1_ptr Sample* cursor, const volatile tt_l1_ptr Sample* end) {
        const auto load = [](const volatile tt_l1_ptr Sample* slot) { return Sample{slot->refclk, slot->wall8}; };
        while (cursor != end) {
            if (line.wall_per_refclk_eighths == 0 || has_off || hold.count != kHoldSamples) {
                on_sample(load(cursor));
                cursor++;
                continue;
            }
            // Copies the loop keeps in registers. close_window() works on the members, so they sync around it.
            Line steady{.wall_per_refclk_eighths = line.wall_per_refclk_eighths, .at = line.at, .mean = line.mean};
            Window open = window;
            uint32_t hold_begin = hold.begin;
            bool left = false;
            // Load each sample's words and hold slot an iteration ahead, so their latency hides behind the math. The
            // one read past the end lands in L1 and is never used.
            Sample sample = load(cursor), held = hold.samples[hold_begin];
            for (; cursor != end; cursor++) {
                const Sample next_sample = load(cursor + 1);
                const uint32_t next_hold_begin = (hold_begin + 1) % kHoldSamples;
                const Sample next_held = hold.samples[next_hold_begin];
                if (!steady.holds(steady.residue(sample))) {
                    left = true;
                    break;
                }
                hold.samples[hold_begin] = sample;
                hold_begin = next_hold_begin;
                if (open.add(held, steady, window_first_wall8)) {
                    window = open;
                    close_window();
                    open.count = window.count;
                    open.size = window.size;
                    steady.mean = line.mean;
                }
                sample = next_sample;
                held = next_held;
            }
            window = open;
            hold.begin = hold_begin;
            if (left) {
                on_sample(load(cursor));
                cursor++;
            }
        }
    }
    void finish() {
        if (line.wall_per_refclk_eighths != 0) {
            for (uint32_t i = 0; i < hold.count; i++) {
                const Sample& sample = hold.samples[(hold.begin + i) % kHoldSamples];
                window_add(sample);
            }
            hold.count = 0;
            if (window.count != 0) {
                close_window();
            }
        } else if (group.count != 0) {
            close_group();
        }
    }
};

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncSampleRing*>(kSampleRingAddr);
    Model model;
    uint32_t next = 0;
    do {
        invalidate_l1_cache();
    } while (ring->tail == 0 && ring->done == 0);
    invalidate_l1_cache();
    model.anchor = Anchor{.refclk = ring->refclk, .wall8 = ring->wall8};
    constexpr uint32_t kChunkSamples = 64, kRingSamples = kernel_profiler::kSyncSampleRingSamples;
    uint32_t stop = 0;
    while (true) {
        invalidate_l1_cache();
        const bool done = ring->done != 0;
        stop = ctrl->stop != 0u ? kernel_profiler::kSyncHeadStop : stop;
        const uint32_t tail = ring->tail;
        const uint32_t end = next + std::min(tail - next, kChunkSamples);
        const uint32_t next_slot = next % kRingSamples;
        const uint32_t before_wrap = std::min(end - next, kRingSamples - next_slot);
        model.feed(&ring->samples[next_slot], &ring->samples[next_slot + before_wrap]);
        model.feed(&ring->samples[0], &ring->samples[end - next - before_wrap]);
        next = end;
        std::atomic_thread_fence(std::memory_order_release);
        ring->head = next + stop;
        if (done && next == tail) {
            break;
        }
    }
    model.finish();
    model.out.close();
    std::atomic_thread_fence(std::memory_order_release);
    ctrl->done = kernel_profiler::kResidentDoneWord;
}
