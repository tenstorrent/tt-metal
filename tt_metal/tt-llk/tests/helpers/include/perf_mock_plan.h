// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

// The Wormhole isolate mocks' MOP stream planning, free of hardware headers so a host test can tabulate it (perf.h emits it).
// The planners are always_inline so a call with constant arguments folds into the emitted stream.
namespace llk_perf_mock
{
constexpr std::uint32_t REPLAY_HANDSHAKES = 32 / 2;        // the replay buffer's 32 instructions, a STALLWAIT and a flip each
constexpr std::uint32_t MOP_LOOP_MAX      = (1u << 7) - 1; // the MOP expander's 7 bit loop counters
constexpr std::uint32_t ZMASK_BITS        = 32;            // template 0's zmask: one handshake kind per iteration
constexpr std::uint32_t MAX_MOPS          = 32;            // template 1 MOPs one emit_segments() pushes back to back
constexpr std::uint32_t REUSE_BURST       = 2;             // the matmul scheme's reused handshakes in a row

// A loop count for the MOP expander's 7 bit loop counters that divides n: n itself up to 127, else its largest power of two
// factor (at most 64), times 3 when that still fits and divides
constexpr __attribute__((always_inline)) std::uint32_t mop_loops(const std::uint32_t n)
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

// How emit_segments() runs periods x (a burst of `burst` handshakes and a run of `run` ones). fits is false when no plan exists.
// whole: the inner loop replays a period from the replay buffer; split: that period is the run's first period - REPLAY_HANDSHAKES
// handshakes and then the buffer; else the inner loop is the run and the burst is replayed around it. buffered is the replay
// buffer's handshakes, inner and outer the loop counts of a MOP and mops the MOPs pushed.
struct SegmentPlan
{
    // words, not bool: packed bools make GCC spill the plan to a stack word and reload it, even in a constant call
    std::uint32_t fits;
    std::uint32_t whole;
    std::uint32_t split;
    std::uint32_t buffered;
    std::uint32_t inner;
    std::uint32_t outer;
    std::uint32_t mops;
};

// allow_split (the consumer only) also replays a period of up to 2 x REPLAY_HANDSHAKES - burst whole
constexpr __attribute__((always_inline)) SegmentPlan
plan_segments(const bool allow_split, const std::uint32_t burst, const std::uint32_t run, const std::uint32_t periods)
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

// One kt step of the matmul scheme: its handshakes (period) and the reuse ones among them (bit j of pattern)
struct MatmulStep
{
    std::uint32_t period;
    std::uint32_t pattern;
};

constexpr __attribute__((always_inline)) MatmulStep matmul_step(const bool produce, const std::uint32_t reuse_loop, const std::uint32_t reload_loop)
{
    MatmulStep step {0, 0};
    for (std::uint32_t reuse_iter = 0; reuse_iter < reuse_loop; reuse_iter += 2)
    {
        const std::uint32_t burst = std::min(REUSE_BURST, reuse_loop - reuse_iter);
        if (step.period + burst + reload_loop <= ZMASK_BITS)
        {
            step.pattern |= ((1u << burst) - 1) << (produce ? step.period : step.period + reload_loop);
        }
        step.period += burst + reload_loop;
    }
    return step;
}

// An odd reuse count above 1 ends in a shorter burst than the replayed ones, so only 1 or an even count can use the replay buffer
constexpr __attribute__((always_inline)) bool matmul_can_segment(const std::uint32_t reuse_loop)
{
    return reuse_loop == 1 || reuse_loop % 2 == 0;
}

constexpr __attribute__((always_inline)) MatmulPlan plan_matmul(
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
    plan.reuse_loop       = std::min(ct_dim, rt_dim);
    plan.reload_loop      = std::max(ct_dim, rt_dim);
    plan.steps            = loop_factor * num_blocks * kt_dim;
    const MatmulStep step = matmul_step(produce, plan.reuse_loop, plan.reload_loop);
    plan.period           = step.period;
    plan.pattern          = step.pattern;
    plan.seg_burst        = std::min(REUSE_BURST, plan.reuse_loop);
    plan.seg_periods      = plan.steps * ((plan.reuse_loop + 1) / 2);
    if (matmul_can_segment(plan.reuse_loop))
    {
        plan.seg = plan_segments(!produce, plan.seg_burst, plan.reload_loop, plan.seg_periods);
    }
    plan.route = plan.seg.fits ? Route::SEGMENTS : (plan.period <= ZMASK_BITS ? Route::STREAM : Route::PER_MOP);
    return plan;
}

// periods x (paired handshakes of both sources, then b_only SrcB ones), in that order for the producer and the consumer:
// the stream's pattern, bit j of the period set for a SrcB handshake (the period is at most the zmask)
constexpr __attribute__((always_inline)) std::uint32_t plan_ab_then_b_pattern(const std::uint32_t paired, const std::uint32_t b_only)
{
    return b_only == 0 ? 0u : (b_only >= ZMASK_BITS ? ~0u : ((1u << b_only) - 1) << paired); // no shift by 32 for (32, 0)
}

// periods x (one SrcB handshake and a_count SrcA ones): the producer gives B first, the consumer takes it back last. pattern
// is the stream's (bit j of the period of a_count + 1 handshakes: the SrcB one).
struct SrcbReusePlan
{
    Route route;
    std::uint32_t pattern;
    SegmentPlan seg;
};

// The stream's pattern for a_count < ZMASK_BITS: the SrcB handshake is first for the producer and last for the consumer
constexpr __attribute__((always_inline)) std::uint32_t srcb_reuse_pattern(const bool produce, const std::uint32_t a_count)
{
    return produce ? 1u : 1u << a_count;
}

constexpr __attribute__((always_inline)) SrcbReusePlan plan_srcb_reuse(const bool produce, const std::uint32_t a_count, const std::uint32_t periods)
{
    const SegmentPlan seg = plan_segments(false, 1, a_count, periods);
    if (seg.fits)
    {
        return {Route::SEGMENTS, 0, seg};
    }
    if (a_count < ZMASK_BITS)
    {
        return {Route::STREAM, srcb_reuse_pattern(produce, a_count), seg};
    }
    return {Route::PER_MOP, 0, seg};
}
} // namespace llk_perf_mock
