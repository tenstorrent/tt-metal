// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs on ERISC1 of the chip's wall-clock tile. eth_clock_sampler.cpp on ERISC0 writes a sample at every refclk update
// (every 80 ns), far too many to ship, so this kernel turns the samples into clock points for the host. A point is a
// refclk tick, the wall clock there and its rate, and there is at least one per millisecond. While AICLK is steady the
// samples lie on a straight line, so a point loses nothing. While it changes, a point averages kGroupSamples samples.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring");

// The most refclk ticks a window of samples covers before it closes into a point, so there is at least one point per
// millisecond.
constexpr uint32_t kPointTicks = kernel_profiler::kEthRefclkHz / 1000;
// How far from the line a sample can be, in eighths of a cycle, and still count as on it. A sample's own error is up to
// half the gap between the two reads around its refclk change, at most 1.5 cycles, and this is twice that.
constexpr int32_t kOffLineEighths = 24;
// How many samples after a sample must also be on the line before it joins a window. The sampler sees nearly every
// refclk update, one every 80 ns (4 ticks). A rate change of one eighth adds an eighth of a cycle per refclk tick, so
// eight samples (32 ticks) after a change the clock is 32 eighths off the line, past kOffLineEighths.
constexpr uint32_t kHoldSamples = 8;
// While AICLK is changing, every kGroupSamples consecutive samples (about 320 ns) are averaged into one point. AICLK's
// rate changes so little over 320 ns that the average is off by at most about 0.06 cycles. A new line opens once
// kSteadyPoints of these points (about 4 us) fit one rate. A PLL step takes about 1.3 us, which is too short for a step
// to be mistaken for a steady rate over 4 us.
constexpr uint32_t kGroupSamples = 4;
constexpr uint32_t kSteadyPoints = 12;
// The most samples a window holds. Each window that closes doubles the next one's size, up to this. A window holds at
// most one sample per refclk update (4 ticks), so a window this size always reaches kPointTicks and closes before it
// fills.
constexpr uint32_t kMaxWindowSamples = 1u << 14;
static_assert(kMaxWindowSamples > kPointTicks / kernel_profiler::kRefclkTicksPerUpdate);

FORCE_INLINE constexpr uint32_t divide_rounded(uint32_t numerator, uint32_t denominator) {
    return (numerator + denominator / 2u) / denominator;
}
constexpr kernel_profiler::SyncMeta kMetaWallClock{.kind = kernel_profiler::SyncKind::WallClock};

using Sample = kernel_profiler::SyncSample;

struct Model {
    // The line the samples are on, through `at` with slope wall_per_refclk_eighths, which is 0 when there is no line.
    // mean is the average residual of the line's `count` samples so far.
    struct Line {
        uint32_t wall_per_refclk_eighths = 0;
        Sample at{};
        int32_t mean = 0;
        int64_t sum = 0;
        // 64-bit because at one sample per refclk update, a 32-bit count wraps in under six minutes on one line.
        uint64_t count = 0;
        FORCE_INLINE int32_t residual(Sample sample) const {
            return static_cast<int32_t>(
                (sample.wall_eighths - at.wall_eighths) - wall_per_refclk_eighths * (sample.refclk - at.refclk));
        }
        FORCE_INLINE bool holds(int32_t sample_residual) const {
            return static_cast<uint32_t>(sample_residual - mean + kOffLineEighths) <= 2u * kOffLineEighths;
        }
    };
    // The samples on the line since the last point. sum_residual sums their residuals minus the line's mean, which only
    // moves when a window closes. Each term is then within about kOffLineEighths of zero, so the sum fits in 32 bits.
    struct Window {
        uint32_t first_refclk = 0;
        uint32_t sum_refclk_from_first = 0;
        int32_t sum_residual = 0;
        uint32_t count = 0, size = 1;
        uint32_t last_refclk = 0;
        // Adds a sample on `line` and returns whether the window is ready to close. The first sample's wall clock is
        // stored in first_wall_eighths, outside the window, so a register copy of the window doesn't carry it.
        FORCE_INLINE bool add(Sample sample, const Line& line, uint32_t& first_wall_eighths) {
            if (count == 0) {
                first_refclk = sample.refclk;
                first_wall_eighths = sample.wall_eighths;
                sum_refclk_from_first = 0;
                sum_residual = 0;
            }
            const uint32_t refclk_from_first = sample.refclk - first_refclk;
            sum_refclk_from_first += refclk_from_first;
            sum_residual += line.residual(sample) - line.mean;
            last_refclk = sample.refclk;
            return ++count == size || refclk_from_first >= kPointTicks;
        }
    };
    // Samples waiting to join the window. A line change empties the hold, so a held sample joins on the same line it
    // was checked against, and its residual is computed only then.
    struct Hold {
        Sample samples[kHoldSamples];
        uint32_t begin = 0, count = 0;
    };
    // While there is no line, Group holds the samples since the last point and Recent holds the last kSteadyPoints
    // points.
    struct Group {
        Sample first{}, last{};
        uint32_t sum_refclk_from_first = 0, sum_wall_eighths_from_first = 0, count = 0;
    };
    struct Recent {
        Sample points[kSteadyPoints];
        uint32_t count = 0, head = 0;
    };

    // hold is the first member, so the fast loop addresses its slots with no offset from the Model.
    Hold hold;
    Line line;
    Window window;
    uint32_t window_first_wall_eighths = 0;
    Sample off_line_candidate{};
    bool has_off_line_candidate = false;
    Group group;
    Recent recent;
    eth_clock::WallClockWriter<kCtrlAddr, kSyncRingAddr> writer;

    void emit_point(uint32_t refclk, uint32_t wall_eighths, uint32_t wall_per_refclk_eighths) {
        writer.add(refclk, wall_eighths, wall_per_refclk_eighths, kMetaWallClock);
    }
    __attribute__((noinline)) void close_window() {
        const uint32_t count = window.count;
        const uint32_t mean_refclk_from_first = divide_rounded(window.sum_refclk_from_first, count);
        const int32_t half_count = static_cast<int32_t>(count / 2u);
        // The sum over the window of each sample's residual minus the first sample's residual.
        const int32_t error_sum =
            window.sum_residual -
            static_cast<int32_t>(count) *
                (line.residual(Sample{window.first_refclk, window_first_wall_eighths}) - line.mean);
        const int32_t mean_error =
            (error_sum + (error_sum < 0 ? -half_count : half_count)) / static_cast<int32_t>(count);
        emit_point(
            window.first_refclk + mean_refclk_from_first,
            window_first_wall_eighths + line.wall_per_refclk_eighths * mean_refclk_from_first +
                static_cast<uint32_t>(mean_error),
            line.wall_per_refclk_eighths);
        line.sum += static_cast<int64_t>(window.sum_residual) + static_cast<int64_t>(count) * line.mean;
        line.count += count;
        line.mean = static_cast<int32_t>(line.sum / static_cast<int64_t>(line.count));
        window.count = 0;
        window.size = std::min(window.size * 2u, kMaxWindowSamples);
    }
    FORCE_INLINE void window_add(Sample sample) {
        if (window.add(sample, line, window_first_wall_eighths)) {
            close_window();
        }
    }
    __attribute__((noinline)) void close_group() {
        // A one-sample group has a span of 0 and no remainder, so its rate is never used.
        const uint32_t span = group.last.refclk - group.first.refclk;
        const uint32_t wall_per_refclk_eighths =
            span != 0 ? divide_rounded(group.last.wall_eighths - group.first.wall_eighths, span) : 0;
        // Every group except the capture's last one is full, so the division is nearly always by the constant.
        const auto divide = [&](uint32_t sum) {
            return group.count == kGroupSamples ? sum / kGroupSamples : sum / group.count;
        };
        const uint32_t mean_refclk_from_first = divide(group.sum_refclk_from_first),
                       refclk_remainder = group.sum_refclk_from_first - mean_refclk_from_first * group.count;
        const uint32_t mean_wall_eighths_from_first =
            divide(group.sum_wall_eighths_from_first - wall_per_refclk_eighths * refclk_remainder + group.count / 2u);
        const Sample centroid{
            group.first.refclk + mean_refclk_from_first, group.first.wall_eighths + mean_wall_eighths_from_first};
        emit_point(centroid.refclk, centroid.wall_eighths, 0);
        recent.points[(recent.head + recent.count) % kSteadyPoints] = centroid;
        if (recent.count < kSteadyPoints) {
            recent.count++;
        } else {
            recent.head = (recent.head + 1) % kSteadyPoints;
        }
        group.count = 0;
        if (recent.count == kSteadyPoints) {
            try_open_line();
        }
    }
    FORCE_INLINE void group_add(Sample sample) {
        if (group.count == 0) {
            group.first = sample;
            group.sum_refclk_from_first = 0;
            group.sum_wall_eighths_from_first = 0;
        }
        group.sum_refclk_from_first += sample.refclk - group.first.refclk;
        group.sum_wall_eighths_from_first += sample.wall_eighths - group.first.wall_eighths;
        group.last = sample;
        if (++group.count == kGroupSamples) {
            close_group();
        }
    }
    __attribute__((noinline)) void try_open_line() {
        const Sample oldest = recent.points[recent.head];
        const Sample& newest = recent.points[(recent.head + kSteadyPoints - 1) % kSteadyPoints];
        const uint32_t span = newest.refclk - oldest.refclk;
        Line candidate{
            .wall_per_refclk_eighths = divide_rounded(newest.wall_eighths - oldest.wall_eighths, span), .at = oldest};
        int32_t residuals[kSteadyPoints];
        int32_t sum = 0;
        for (uint32_t i = 0; i < kSteadyPoints; i++) {
            residuals[i] = candidate.residual(recent.points[(recent.head + i) % kSteadyPoints]);
            sum += residuals[i];
        }
        candidate.mean = sum / static_cast<int32_t>(kSteadyPoints);
        for (uint32_t i = 0; i < kSteadyPoints; i++) {
            if (!candidate.holds(residuals[i])) {
                return;
            }
        }
        line.wall_per_refclk_eighths = candidate.wall_per_refclk_eighths;
        line.at = oldest;
        line.mean = candidate.mean;
        line.sum = sum;
        line.count = kSteadyPoints;
        // While there is no line, the window and the hold are already empty.
        window.size = 1;
    }
    // Ends the line with a point at the last sample that joined the window, so the line's points reach that sample
    // instead of stopping at the middle of the last window.
    __attribute__((noinline)) void end_line(Sample sample) {
        if (window.count != 0) {
            close_window();
        }
        if (line.count > kSteadyPoints) {
            const uint32_t last_refclk = window.last_refclk;
            emit_point(
                last_refclk,
                line.at.wall_eighths + line.wall_per_refclk_eighths * (last_refclk - line.at.refclk) +
                    static_cast<uint32_t>(line.mean),
                line.wall_per_refclk_eighths);
        }
        line.wall_per_refclk_eighths = 0;
        recent.count = 0;
        for (uint32_t i = 0; i < hold.count; i++) {
            group_add(hold.samples[(hold.begin + i) % kHoldSamples]);
        }
        hold.count = 0;
        group_add(off_line_candidate);
        group_add(sample);
        has_off_line_candidate = false;
    }
    FORCE_INLINE void on_sample(Sample sample) {
        if (line.wall_per_refclk_eighths == 0) {
            group_add(sample);
            return;
        }
        if (!line.holds(line.residual(sample))) {
            if (has_off_line_candidate) {
                end_line(sample);
                return;
            }
            off_line_candidate = sample;
            has_off_line_candidate = true;
            return;
        }
        has_off_line_candidate = false;
        if (hold.count == kHoldSamples) {
            window_add(hold.samples[hold.begin]);
            hold.begin = (hold.begin + 1) % kHoldSamples;
            hold.count--;
        }
        hold.samples[(hold.begin + hold.count) % kHoldSamples] = sample;
        hold.count++;
    }
    // Feeds the samples [cursor, end) in ring order. While on a line, with a full hold and no off-line candidate
    // pending, the fast loop keeps the state a sample touches in registers. Every other case goes through on_sample.
    __attribute__((noinline)) void feed(const volatile tt_l1_ptr Sample* cursor, const volatile tt_l1_ptr Sample* end) {
        const auto load = [](const volatile tt_l1_ptr Sample* slot) {
            return Sample{slot->refclk, slot->wall_eighths};
        };
        while (cursor != end) {
            if (line.wall_per_refclk_eighths == 0 || has_off_line_candidate || hold.count != kHoldSamples) {
                on_sample(load(cursor));
                cursor++;
                continue;
            }
            // Register copies for the loop. close_window() works on the members, so the copies are written back
            // before it and reloaded after.
            Line steady{.wall_per_refclk_eighths = line.wall_per_refclk_eighths, .at = line.at, .mean = line.mean};
            Window open = window;
            uint32_t hold_begin = hold.begin;
            bool left_line = false;
            // Each sample and its hold slot are loaded an iteration ahead so the load latency hides behind the math.
            // The one read past the end stays inside L1, and its value is never used.
            Sample sample = load(cursor), held = hold.samples[hold_begin];
            for (; cursor != end; cursor++) {
                const Sample next_sample = load(cursor + 1);
                const uint32_t next_hold_begin = (hold_begin + 1) % kHoldSamples;
                const Sample next_held = hold.samples[next_hold_begin];
                if (!steady.holds(steady.residual(sample))) {
                    left_line = true;
                    break;
                }
                hold.samples[hold_begin] = sample;
                hold_begin = next_hold_begin;
                if (open.add(held, steady, window_first_wall_eighths)) {
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
            if (left_line) {
                on_sample(load(cursor));
                cursor++;
            }
        }
    }
    void finish() {
        if (line.wall_per_refclk_eighths != 0) {
            for (uint32_t i = 0; i < hold.count; i++) {
                window_add(hold.samples[(hold.begin + i) % kHoldSamples]);
            }
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
    model.writer.wall_eighths = ring->wall_eighths;
    constexpr uint32_t kChunkSamples = 64, kRingSamples = kernel_profiler::kSyncSampleRingSamples;
    static_assert(kRingSamples % kChunkSamples == 0);
    uint32_t head_stop = 0;
    while (true) {
        invalidate_l1_cache();
        const bool done = ring->done != 0;
        head_stop = ctrl->stop != 0u ? kernel_profiler::kSyncHeadStop : head_stop;
        const uint32_t tail = ring->tail;
        // A chunk ends at a multiple of kChunkSamples at the latest, so it never runs past the end of the ring.
        const uint32_t end = next + std::min(tail - next, kChunkSamples - next % kChunkSamples);
        const uint32_t slot = next % kRingSamples;
        model.feed(&ring->samples[slot], &ring->samples[slot + (end - next)]);
        next = end;
        std::atomic_thread_fence(std::memory_order_release);
        ring->head = next + head_stop;
        if (done && next == tail) {
            break;
        }
    }
    model.finish();
    model.writer.close();
    std::atomic_thread_fence(std::memory_order_release);
    ctrl->done = kernel_profiler::kResidentDoneWord;
}
