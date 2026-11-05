// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

// The Wormhole isolate mocks' MOP stream planning, free of hardware headers so a host test can tabulate it (perf.h emits it).
namespace llk_perf_mock
{
constexpr std::uint32_t REPLAY_HANDSHAKES = 32 / 2;        // the replay buffer's 32 instructions, a STALLWAIT and a flip each
constexpr std::uint32_t MOP_LOOP_MAX      = (1u << 7) - 1; // the MOP expander's 7 bit loop counters
constexpr std::uint32_t ZMASK_BITS        = 32;            // template 0's zmask: one handshake kind per iteration
constexpr std::uint32_t MAX_MOPS          = 32;            // template 1 MOPs one segments() pushes back to back
constexpr std::uint32_t REUSE_BURST       = 2;             // the matmul scheme's reused handshakes in a row

// A loop count for the MOP expander's 7 bit loop counters that divides n: n itself up to 127, else its largest power of two
// factor (at most 64), times 3 when that still fits and divides
constexpr std::uint32_t mop_loops(const std::uint32_t n)
{
    if (n <= MOP_LOOP_MAX)
    {
        return n;
    }
    std::uint32_t loops = std::min(n & (~n + 1), (MOP_LOOP_MAX + 1) / 2);
    if (loops * 3 <= MOP_LOOP_MAX && (n / loops) % 3 == 0)
    {
        loops *= 3;
    }
    return loops;
}

// How segments() runs periods x (a burst of `burst` handshakes and a run of `run` ones). fits is false when no plan exists.
// whole: the inner loop replays a period from the replay buffer; split: that period is the run's first period - REPLAY_HANDSHAKES
// handshakes and then the buffer; else the inner loop is the run and the burst is replayed around it. buffered is the replay
// buffer's handshakes, inner and outer the loop counts of a MOP and mops the MOPs pushed.
struct SegmentPlan
{
    bool fits;
    bool whole;
    bool split;
    std::uint32_t buffered;
    std::uint32_t inner;
    std::uint32_t outer;
    std::uint32_t mops;
};

// allow_split (the consumer only) also replays a period of up to 2 x REPLAY_HANDSHAKES - burst whole
constexpr SegmentPlan plan_segments(const bool allow_split, const std::uint32_t burst, const std::uint32_t run, const std::uint32_t periods)
{
    constexpr std::uint32_t HALF = REPLAY_HANDSHAKES;
    const std::uint32_t period   = burst + run;
    if (run == 0 || burst == 0 || burst > HALF || periods == 0 || (period > HALF && run > MOP_LOOP_MAX))
    {
        return {false, false, false, 0, 0, 0, 0};
    }
    const bool split          = allow_split && period > HALF && period + burst <= 2 * HALF;
    const bool whole          = period <= HALF || split;
    const std::uint32_t inner = whole ? mop_loops(periods) : run;
    const std::uint32_t outer = whole ? mop_loops(periods / inner) : mop_loops(periods);
    const std::uint32_t mops  = whole ? periods / inner / outer : periods / outer;
    if (mops > MAX_MOPS)
    {
        return {false, whole, split, 0, inner, outer, mops};
    }
    return {true, whole, split, whole ? (split ? HALF : period) : burst, inner, outer, mops};
}

// Which emitter runs a stream: segments (template 1 with the replay buffer), stream (template 0 with a zmask pattern), or one
// MOP per burst and run (a period longer than the zmask)
enum class Route
{
    NONE,
    SEGMENTS,
    STREAM,
    PER_MOP
};

// The matmul mocks' scheme (see matmul() in perf.h): per kt step, the reused tiles two at a time, each burst followed (unpack)
// or preceded (math) by the reloaded tiles. period is a kt step's handshakes, pattern its reuse ones (bit j of the period),
// steps the kt steps, and seg the segments plan with seg_periods periods of a burst of seg_burst reused handshakes.
struct MatmulPlan
{
    Route route;
    std::uint32_t reuse_loop;
    std::uint32_t reload_loop;
    std::uint32_t steps;
    std::uint32_t period;
    std::uint32_t pattern;
    std::uint32_t seg_burst;
    std::uint32_t seg_periods;
    SegmentPlan seg;
};

constexpr MatmulPlan plan_matmul(
    const bool produce,
    const std::uint32_t loop_factor,
    const std::uint32_t rt_dim,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim,
    const std::uint32_t num_blocks)
{
    MatmulPlan plan {Route::NONE, 0, 0, 0, 0, 0, 0, 0, {}};
    if (loop_factor == 0 || rt_dim == 0 || kt_dim == 0 || ct_dim == 0 || num_blocks == 0)
    {
        return plan; // no work
    }
    plan.reuse_loop  = std::min(ct_dim, rt_dim);
    plan.reload_loop = std::max(ct_dim, rt_dim);
    plan.steps       = loop_factor * num_blocks * kt_dim;
    for (std::uint32_t reuse_iter = 0; reuse_iter < plan.reuse_loop; reuse_iter += 2)
    {
        const std::uint32_t burst = std::min(REUSE_BURST, plan.reuse_loop - reuse_iter);
        if (plan.period + burst + plan.reload_loop <= ZMASK_BITS)
        {
            plan.pattern |= ((1u << burst) - 1) << (produce ? plan.period : plan.period + plan.reload_loop);
        }
        plan.period += burst + plan.reload_loop;
    }
    plan.seg_burst   = std::min(REUSE_BURST, plan.reuse_loop);
    plan.seg_periods = plan.steps * ((plan.reuse_loop + 1) / 2);
    if (plan.reuse_loop == 1 || plan.reuse_loop % 2 == 0) // an odd reuse count above 1 ends in a shorter burst than the replayed ones
    {
        plan.seg = plan_segments(!produce, plan.seg_burst, plan.reload_loop, plan.seg_periods);
    }
    plan.route = plan.seg.fits ? Route::SEGMENTS : (plan.period <= ZMASK_BITS ? Route::STREAM : Route::PER_MOP);
    return plan;
}

// periods x (paired handshakes of both sources, then b_only SrcB ones), in that order for the producer and the consumer:
// the stream's pattern, bit j of the period set for a SrcB handshake (the period is at most the zmask)
constexpr std::uint32_t plan_ab_then_b_pattern(const std::uint32_t paired, const std::uint32_t b_only)
{
    return b_only >= ZMASK_BITS ? ~0u : ((1u << b_only) - 1) << paired;
}

// periods x (one SrcB handshake and a_count SrcA ones): the producer gives B first, the consumer takes it back last. pattern
// is the stream's (bit j of the period of a_count + 1 handshakes: the SrcB one).
struct SrcbReusePlan
{
    Route route;
    std::uint32_t pattern;
    SegmentPlan seg;
};

constexpr SrcbReusePlan plan_srcb_reuse(const bool produce, const std::uint32_t a_count, const std::uint32_t periods)
{
    const SegmentPlan seg = plan_segments(false, 1, a_count, periods);
    if (seg.fits)
    {
        return {Route::SEGMENTS, 0, seg};
    }
    if (a_count < ZMASK_BITS)
    {
        return {Route::STREAM, produce ? 1u : 1u << a_count, seg};
    }
    return {Route::PER_MOP, 0, seg};
}
} // namespace llk_perf_mock
