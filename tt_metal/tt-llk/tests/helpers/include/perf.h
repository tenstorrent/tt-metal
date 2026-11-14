// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "perf_mock_plan.h"
#include "tensor_shape.h"

// === L1 layout for perf tests ===
//
// Stimuli generator buffers and the perf-counter region are kept in this header
// so that the disjoint ranges are visible side-by-side and any future allocation
// has to be reconciled against both. The perf-counter macros are only defined when perf
// counters are compiled (the counter machinery in counters.h is gated by the same flag).

// FIXME: this shouldn't be statically allocated
constexpr std::uint32_t PERF_RING_TILES = 16; // Keep in sync with helpers/matmul_sweep.py
constexpr std::uint32_t PERF_INPUT_A    = 0x21000;
constexpr std::uint32_t PERF_INPUT_B    = PERF_INPUT_A + PERF_RING_TILES * 4096;
constexpr std::uint32_t PERF_INPUT_C    = PERF_INPUT_B + PERF_RING_TILES * 4096;
constexpr std::uint32_t PERF_OUTPUT     = PERF_INPUT_C + PERF_RING_TILES * 4096;

// Perf-counter shared config + per-zone data. Must stay below the profiler boundary at
// 0x16AFF0 (asserted in counters.h) and must not overlap the stimuli buffers above; both builds use it.
#define PERF_COUNTERS_BASE_ADDR         0x169000
#define PERF_COUNTERS_CONFIG_WORDS      200
#define PERF_COUNTERS_DATA_WORDS        200
#define PERF_COUNTERS_BANK_CYCLES_WORDS 5

constexpr std::uint32_t PERF_ADDRESS(std::uint32_t buffer, std::uint32_t tile)
{
    std::uint32_t address = buffer + (tile % PERF_RING_TILES) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                                          // Correct the L1 Address for Tensix
}

enum class PerfRunType
{
    L1_TO_L1,
    UNPACK_ISOLATE,
    MATH_ISOLATE,
    PACK_ISOLATE,
    L1_CONGESTION,
    SFPU_ISOLATE
};

// The run types whose source mock goes through the MOP expander: on Wormhole, MATH_ISOLATE's unpack mock and
// UNPACK_ISOLATE's math mock. Every other caller (L1_CONGESTION, Blackhole, Quasar) keeps the RISC-V loop mocks below.
template <PerfRunType RUN_TYPE>
constexpr bool perf_mock_mop()
{
#if defined(ARCH_WORMHOLE)
    return RUN_TYPE == PerfRunType::UNPACK_ISOLATE || RUN_TYPE == PerfRunType::MATH_ISOLATE;
#else
    return false;
#endif
}

#if defined(ARCH_WORMHOLE)
#include "ckernel_template.h"

// Wormhole isolate mocks: the partner that feeds (unpack mock) or drains (math mock) the measured thread hands its
// source handshakes to its own MOP expander: up to 128 per MOP (template 0), or a whole burst and run pattern in a few MOPs
// (template 1 with the burst in the replay buffer). Its RISC-V pushes a few instructions per MOP rather than looping once
// per handshake, so the measured thread waits only for the handshakes themselves. A handshake is the same STALLWAIT +
// SETDVALID / CLEARDVALID pair as in the RISC-V loop mocks below.
namespace llk_perf_mock
{
// The sources one handshake gives (SETDVALID) or takes back (CLEARDVALID): bit 0 SrcA, bit 1 SrcB
constexpr std::uint32_t SRC_A  = 1;
constexpr std::uint32_t SRC_B  = 2;
constexpr std::uint32_t SRC_AB = SRC_A | SRC_B;

// The producer (unpack mock) waits on TDMA for the sources to be cleared, the consumer (math mock) on MATH for them to be valid
template <bool PRODUCE>
constexpr std::uint32_t stall_unit()
{
    return PRODUCE ? ckernel::p_stall::STALL_TDMA : ckernel::p_stall::STALL_MATH;
}

template <bool PRODUCE>
constexpr std::uint32_t src_a_condition()
{
    return PRODUCE ? ckernel::p_stall::SRCA_CLR : ckernel::p_stall::SRCA_VLD;
}

// SrcB's condition is SrcA's one bit up, so the SRC_A / SRC_B bits times SrcA's condition give the condition for both
static_assert(ckernel::p_stall::SRCB_CLR == ckernel::p_stall::SRCA_CLR << 1 && ckernel::p_stall::SRCB_VLD == ckernel::p_stall::SRCA_VLD << 1);
// REPLAY_HANDSHAKES is two handshakes (STALLWAIT + flip) per replay buffer slot; fail on a buffer-size change
static_assert(2 * REPLAY_HANDSHAKES == REPLAY_BUF_SIZE, "REPLAY_HANDSHAKES must be half the replay buffer depth");

template <bool PRODUCE>
constexpr std::uint32_t wait_op(const std::uint32_t srcs)
{
    return TT_OP_STALLWAIT(stall_unit<PRODUCE>(), srcs * src_a_condition<PRODUCE>());
}

template <bool PRODUCE>
constexpr std::uint32_t flip_op(const std::uint32_t srcs)
{
    return PRODUCE ? TT_OP_SETDVALID(srcs) : TT_OP_CLEARDVALID(srcs, 0 /*reset*/);
}

// Template 0 with one handshake per iteration, of kind0 where the zmask bit is 0 and of kind1 where it is 1, and the
// zmask's high half (an instruction in the stream). Each mock programs it once, before its first MOP.
template <bool PRODUCE>
inline void program(const std::uint32_t kind0, const std::uint32_t kind1, const std::uint32_t mask_hi)
{
    // A0 and B one handshake, skip A and skip B the other; no A1-A3 (program() syncs on the MOPs in flight first)
    ckernel::ckernel_unpack_template(
        true /*unpackB*/, false /*unpackHalo*/, wait_op<PRODUCE>(kind0), TT_OP_NOP, TT_OP_NOP, TT_OP_NOP, wait_op<PRODUCE>(kind1), flip_op<PRODUCE>(kind0), flip_op<PRODUCE>(kind1))
        .program();
    TT_MOP_CFG(mask_hi);
}

// plan's periods x (a burst of `burst` handshakes of kind `once` and a run of `run` handshakes of kind `each`), the producer
// giving the burst first and the consumer taking it back last, in at most MAX_MOPS MOPs of template 1, pushed back to back: the
// mock's RISC-V is done pushing long before the stream ends. A period of up to REPLAY_HANDSHAKES is replayed whole from the
// replay buffer by the inner loop (up to 127 x 127 periods per MOP). SPLIT (the consumer only) also replays a period of up to
// 2 x REPLAY_HANDSHAKES - burst whole: its period - REPLAY_HANDSHAKES leading `each` handshakes, then the buffer's
// REPLAY_HANDSHAKES suffix. A longer period replays the burst around the run (up to 127 periods per MOP). The plan
// (plan_segments) must fit.
template <bool PRODUCE>
inline void emit_segments(const SegmentPlan& plan, const std::uint32_t once, const std::uint32_t burst, const std::uint32_t each, const std::uint32_t run)
{
    constexpr std::uint32_t HALF = REPLAY_HANDSHAKES;
    const std::uint32_t period   = burst + run;
    // the replay buffer, slots 0 ..: the whole period (split: HALF handshakes, the run's first period - HALF dropped), or the burst
    TT_REPLAY(0 /*start_idx*/, 2 * plan.buffered /*len*/, 0 /*execute_while_loading*/, 1 /*load_mode*/);
    for (std::uint32_t i = 0; i < plan.buffered; i++)
    {
        const std::uint32_t kind  = !plan.whole ? once : (PRODUCE ? (i < burst ? once : each) : (i < (plan.split ? HALF - burst : run) ? each : once));
        ckernel::instrn_buffer[0] = wait_op<PRODUCE>(kind);
        ckernel::instrn_buffer[0] = flip_op<PRODUCE>(kind);
    }
    const std::uint32_t replay = TT_OP_REPLAY(0 /*start_idx*/, 2 * plan.buffered /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
    if (plan.whole) // inner loop: the period, replayed (split: the run's first period - HALF, then the buffer); no start or end instructions
    {
        ckernel::ckernel_template tmpl(
            plan.outer, plan.inner, plan.split ? TT_OP_REPLAY(0 /*start_idx*/, 2 * (period - HALF) /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/) : replay, plan.split ? replay : TT_OP_NOP);
        tmpl.set_last_outer_loop_instr(replay);
        tmpl.set_last_inner_loop_instr(replay);
        tmpl.program(); // syncs on the MOPs in flight first
    }
    else // inner loop: the run, STALLWAIT and flip alternating; the burst replayed before (producer) or after (consumer) it
    {
        ckernel::ckernel_template tmpl(plan.outer, plan.inner, wait_op<PRODUCE>(each), flip_op<PRODUCE>(each));
        tmpl.set_start_op(PRODUCE ? replay : TT_OP_NOP);
        tmpl.set_end_op(PRODUCE ? TT_OP_NOP : replay);
        tmpl.program();
    }
    for (std::uint32_t i = 0; i < plan.mops; i++)
    {
        TTI_MOP(1 /*mop_type*/, 0 /*loop_count*/, 0 /*zmask_lo16*/);
    }
}

// count handshakes of one kind, 128 per MOP
template <bool PRODUCE>
inline void run(const std::uint32_t kind, std::uint32_t count)
{
    if (count == 0)
    {
        return; // like the loop mocks: no work leaves the MOP configuration alone
    }
    program<PRODUCE>(kind, kind, 0 /*mask_hi*/);
    for (; count >= MOP_LOOP_MAX + 1; count -= MOP_LOOP_MAX + 1)
    {
        TTI_MOP(0 /*mop_type*/, MOP_LOOP_MAX /*loop_count*/, 0 /*zmask_lo16*/);
    }
    if (count != 0)
    {
        TT_MOP(0 /*mop_type*/, count - 1 /*loop_count*/, 0 /*zmask_lo16*/);
    }
}

// periods x period handshakes, handshake j of each period of kind1 where bit j of pattern is set (period <= 32): as many
// whole periods per MOP as its 32 bit zmask holds
template <bool PRODUCE>
inline void stream(const std::uint32_t kind0, const std::uint32_t kind1, const std::uint32_t pattern, const std::uint32_t period, std::uint32_t periods)
{
    LLK_ASSERT(period >= 1 && period <= ZMASK_BITS, "a period fits the zmask");
    if (periods == 0 || period == 0 || period > ZMASK_BITS)
    {
        return; // period > ZMASK_BITS makes per_mop 0: guard so the loop below terminates even without LLK_ASSERT
    }
    const std::uint32_t per_mop = ZMASK_BITS / period;
    std::uint32_t mask          = 0;
    for (std::uint32_t i = 0; i < per_mop; i++)
    {
        mask |= pattern << (i * period);
    }
    program<PRODUCE>(kind0, kind1, mask >> 16);
    const std::uint32_t mop = TT_OP_MOP(0 /*mop_type*/, per_mop * period - 1 /*loop_count*/, mask & 0xFFFF /*zmask_lo16*/);
    for (; periods >= per_mop; periods -= per_mop)
    {
        ckernel::instrn_buffer[0] = mop;
    }
    if (periods != 0)
    {
        TT_MOP(0 /*mop_type*/, periods * period - 1 /*loop_count*/, mask & 0xFFFF /*zmask_lo16*/);
    }
}

// The matmul mocks' Wormhole scheme (see _perf_unpack_matmul_mock): per kt step, the reused tiles two at a time, each
// burst followed (unpack) or preceded (math) by the reloaded tiles. kind0 = reload, kind1 = reuse.
template <bool PRODUCE>
inline void matmul(
    const std::uint32_t loop_factor, const std::uint32_t rt_dim, const std::uint32_t kt_dim, const std::uint32_t ct_dim, const std::uint32_t num_blocks)
{
    const MatmulPlan plan = plan_matmul(PRODUCE, loop_factor, rt_dim, kt_dim, ct_dim, num_blocks);
    if (plan.route == Route::NONE)
    {
        return; // like the loop mocks: no work
    }
    LLK_ASSERT(
        static_cast<std::uint64_t>(loop_factor) * num_blocks * kt_dim * ((std::min(ct_dim, rt_dim) + 1) / 2) <= 0xFFFFFFFFu,
        "the handshake counts fit 32 bits");
    const std::uint32_t reuse  = ct_dim >= rt_dim ? SRC_B : SRC_A;
    const std::uint32_t reload = ct_dim >= rt_dim ? SRC_A : SRC_B;
    if (plan.route == Route::SEGMENTS)
    {
        return emit_segments<PRODUCE>(plan.seg, reuse, plan.seg_burst, reload, plan.reload_loop);
    }
    if (plan.route == Route::STREAM)
    {
        return stream<PRODUCE>(reload, reuse, plan.pattern, plan.period, plan.steps);
    }
    // a kt step longer than the zmask: one MOP per burst and one per reload run
    program<PRODUCE>(reload, reuse, 0);
    for (std::uint32_t step = 0; step < plan.steps; step++)
    {
        for (std::uint32_t reuse_iter = 0; reuse_iter < plan.reuse_loop; reuse_iter += 2)
        {
            const std::uint32_t burst = std::min(REUSE_BURST, plan.reuse_loop - reuse_iter);
            if constexpr (PRODUCE)
            {
                TT_MOP(0 /*mop_type*/, burst - 1 /*loop_count*/, (1u << burst) - 1 /*zmask_lo16*/);
            }
            for (std::uint32_t left = plan.reload_loop; left != 0;)
            {
                const std::uint32_t count = std::min(left, MOP_LOOP_MAX + 1);
                TT_MOP(0 /*mop_type*/, count - 1 /*loop_count*/, 0 /*zmask_lo16*/);
                left -= count;
            }
            if constexpr (!PRODUCE)
            {
                TT_MOP(0 /*mop_type*/, burst - 1 /*loop_count*/, (1u << burst) - 1 /*zmask_lo16*/);
            }
        }
    }
}

// periods x (paired handshakes of both sources, then b_only SrcB ones), the same order on both sides (period <= ZMASK_BITS)
template <bool PRODUCE>
inline void ab_then_b(const std::uint32_t paired, const std::uint32_t b_only, const std::uint32_t periods)
{
    LLK_ASSERT(paired + b_only <= ZMASK_BITS, "a period fits the zmask");
    if (paired + b_only != 0)
    {
        stream<PRODUCE>(SRC_AB, SRC_B, plan_ab_then_b_pattern(paired, b_only), paired + b_only, periods);
    }
}

// periods x (one SrcB handshake and a_count SrcA ones): the producer gives B first, the consumer takes it back last
template <bool PRODUCE>
inline void srcb_reuse(const std::uint32_t a_count, std::uint32_t periods)
{
    const SrcbReusePlan plan = plan_srcb_reuse(PRODUCE, a_count, periods);
    if (plan.route == Route::SEGMENTS)
    {
        return emit_segments<PRODUCE>(plan.seg, SRC_B, 1, SRC_A, a_count);
    }
    if (plan.route == Route::STREAM)
    {
        return stream<PRODUCE>(SRC_A, SRC_B, plan.pattern, a_count + 1, periods);
    }
    program<PRODUCE>(SRC_A, SRC_B, 0);
    for (; periods != 0; periods--)
    {
        if constexpr (PRODUCE)
        {
            TTI_MOP(0 /*mop_type*/, 0 /*loop_count*/, 1 /*zmask_lo16*/);
        }
        for (std::uint32_t left = a_count; left != 0;)
        {
            const std::uint32_t count = std::min(left, MOP_LOOP_MAX + 1);
            TT_MOP(0 /*mop_type*/, count - 1 /*loop_count*/, 0 /*zmask_lo16*/);
            left -= count;
        }
        if constexpr (!PRODUCE)
        {
            TTI_MOP(0 /*mop_type*/, 0 /*loop_count*/, 1 /*zmask_lo16*/);
        }
    }
}
} // namespace llk_perf_mock
#endif

// eltwise_binary_test.cpp's source mock as one stream: COL reuses a SrcB per face row, SCALAR per tile, NONE and ROW hand
// over both sources per face. Wormhole isolate mocks only.
template <bool PRODUCE, ckernel::BroadcastType BTYPE>
inline void _perf_mock_binary_(const std::uint32_t tiles, const std::uint32_t num_faces_r, const std::uint32_t num_faces_c)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (BTYPE == ckernel::BroadcastType::COL)
    {
        llk_perf_mock::srcb_reuse<PRODUCE>(num_faces_c, tiles * num_faces_r);
    }
    else if constexpr (BTYPE == ckernel::BroadcastType::SCALAR)
    {
        llk_perf_mock::srcb_reuse<PRODUCE>(num_faces_r * num_faces_c, tiles);
    }
    else
    {
        llk_perf_mock::run<PRODUCE>(llk_perf_mock::SRC_AB, tiles * num_faces_r * num_faces_c);
    }
#endif
}

// periods x (one SrcB handshake and a_count SrcA ones), the producer giving B first and the consumer taking it back last.
// Wormhole isolate mocks only.
template <bool PRODUCE>
inline void _perf_mock_srcb_reuse_(const std::uint32_t a_count, const std::uint32_t periods)
{
#if defined(ARCH_WORMHOLE)
    llk_perf_mock::srcb_reuse<PRODUCE>(a_count, periods);
#endif
}

inline void _perf_unpack_set_valid(std::uint32_t source)
{
    std::uint32_t set_a = source == ckernel::SrcA ? 1 : 0;
    std::uint32_t set_b = source == ckernel::SrcB ? 1 : 0;

    std::uint32_t wait_clear_a = set_a ? ckernel::p_stall::SRCA_CLR : 0;
    std::uint32_t wait_clear_b = set_b ? ckernel::p_stall::SRCB_CLR : 0;

#ifdef ARCH_QUASAR
    (void)wait_clear_a;
    (void)wait_clear_b;
    // UNPACR_NOP's set-dvalid-only form (encoding 2) stalling on UNP_STALL_UNP_WR waits for the bank the unpacker
    // writes next to be free and then sets it valid: the same handshake as the STALLWAIT and SETDVALID below, in one
    // instruction that also keeps the unpacker's bank sequencing
    if (set_a)
    {
        TT_UNPACR_NOP(ckernel::p_unpacr::UNP_A, 1 /*Set_Dvalid*/, ckernel::p_unpacr::UNP_STALL_UNP_WR, 0, 0, 2 /*set dvalid only*/);
    }
    if (set_b)
    {
        TT_UNPACR_NOP(ckernel::p_unpacr::UNP_B, 1 /*Set_Dvalid*/, ckernel::p_unpacr::UNP_STALL_UNP_WR, 0, 0, 2 /*set dvalid only*/);
    }

#else
    TT_STALLWAIT(ckernel::p_stall::STALL_TDMA, wait_clear_a | wait_clear_b);
    TT_SETDVALID((set_b << 1) | set_a);

#endif
}

inline void _perf_math_clear_valid(std::uint32_t source)
{
    std::uint32_t clear_a = source == ckernel::SrcA ? 1 : 0;
    std::uint32_t clear_b = source == ckernel::SrcB ? 1 : 0;

    std::uint32_t wait_valid_a = clear_a ? ckernel::p_stall::SRCA_VLD : 0;
    std::uint32_t wait_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;

#ifdef ARCH_QUASAR
    TT_STALLWAIT(ckernel::p_stall::STALL_MATH, wait_valid_a, wait_valid_b, 0);
    TT_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);

#else
    TT_STALLWAIT(ckernel::p_stall::STALL_MATH, wait_valid_a | wait_valid_b);
    TT_CLEARDVALID((clear_b << 1) | clear_a, 0);

#endif
}

// The source handshakes as a RISC-V loop, one per iteration. The _stream_ versions below take the run type and use the MOP
// stream where it serves it (perf_mock_mop); a caller that stays on the loop calls these by name.
template <bool set_a, bool set_b>
inline void _perf_unpack_loop_set_valid(std::uint32_t iterations)
{
    while (iterations-- > 0)
    {
        constexpr std::uint32_t cond_clear_a = set_a ? ckernel::p_stall::SRCA_CLR : 0;
        constexpr std::uint32_t cond_clear_b = set_b ? ckernel::p_stall::SRCB_CLR : 0;

#ifdef ARCH_QUASAR
        // See _perf_unpack_set_valid
        (void)cond_clear_a;
        (void)cond_clear_b;
        if constexpr (set_a)
        {
            TTI_UNPACR_NOP(ckernel::p_unpacr::UNP_A, 1 /*Set_Dvalid*/, ckernel::p_unpacr::UNP_STALL_UNP_WR, 0, 0, 2 /*set dvalid only*/);
        }
        if constexpr (set_b)
        {
            TTI_UNPACR_NOP(ckernel::p_unpacr::UNP_B, 1 /*Set_Dvalid*/, ckernel::p_unpacr::UNP_STALL_UNP_WR, 0, 0, 2 /*set dvalid only*/);
        }
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_TDMA, cond_clear_a | cond_clear_b);
        TTI_SETDVALID((set_b << 1) | set_a);
#endif
    }
}

template <bool clear_a, bool clear_b>
inline void _perf_math_loop_clear_valid(std::uint32_t iterations)
{
    while (iterations-- > 0)
    {
        constexpr std::uint32_t cond_valid_a = clear_a ? ckernel::p_stall::SRCA_VLD : 0;
        constexpr std::uint32_t cond_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a, cond_valid_b, 0);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a | cond_valid_b);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0);
#endif
    }
}

// iterations source handshakes for the run type RUN_TYPE: one MOP stream on Wormhole isolate runs, else the RISC-V loop
template <bool set_a, bool set_b, PerfRunType RUN_TYPE>
inline void _perf_unpack_stream_set_valid(std::uint32_t iterations)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::run<true>((set_b ? llk_perf_mock::SRC_B : 0) | (set_a ? llk_perf_mock::SRC_A : 0), iterations);
    }
#endif
    _perf_unpack_loop_set_valid<set_a, set_b>(iterations);
}

template <bool clear_a, bool clear_b, PerfRunType RUN_TYPE>
inline void _perf_math_stream_clear_valid(std::uint32_t iterations)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::run<false>((clear_b ? llk_perf_mock::SRC_B : 0) | (clear_a ? llk_perf_mock::SRC_A : 0), iterations);
    }
#endif
    _perf_math_loop_clear_valid<clear_a, clear_b>(iterations);
}

// periods x (paired handshakes of both sources, then b_only SrcB ones) for the run type RUN_TYPE: one MOP stream on Wormhole
// isolate runs, else the RISC-V loop
template <PerfRunType RUN_TYPE>
inline void _perf_unpack_stream_ab_then_b(const std::uint32_t paired, const std::uint32_t b_only, const std::uint32_t periods)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::ab_then_b<true>(paired, b_only, periods);
    }
#endif
    for (std::uint32_t period = 0; period < periods; ++period)
    {
        _perf_unpack_loop_set_valid<true, true>(paired);
        _perf_unpack_loop_set_valid<false, true>(b_only);
    }
}

template <PerfRunType RUN_TYPE>
inline void _perf_math_stream_ab_then_b(const std::uint32_t paired, const std::uint32_t b_only, const std::uint32_t periods)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::ab_then_b<false>(paired, b_only, periods);
    }
#endif
    for (std::uint32_t period = 0; period < periods; ++period)
    {
        _perf_math_loop_clear_valid<true, true>(paired);
        _perf_math_loop_clear_valid<false, true>(b_only);
    }
}

template <PerfRunType RUN_TYPE>
inline void _perf_unpack_matmul_mock(std::uint32_t loop_factor, std::uint32_t rt_dim, std::uint32_t kt_dim, std::uint32_t ct_dim, std::uint32_t num_blocks = 1)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::matmul<true>(loop_factor, rt_dim, kt_dim, ct_dim, num_blocks);
    }
#endif
    for (std::uint32_t loop = 0; loop < loop_factor; loop++)
    {
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            for (std::uint32_t j = 0; j < kt_dim; j++)
            {
                const std::uint32_t reuse_reg  = ct_dim >= rt_dim ? ckernel::SrcB : ckernel::SrcA;
                const std::uint32_t reuse_loop = std::min(ct_dim, rt_dim);

                const std::uint32_t reload_reg  = ct_dim >= rt_dim ? ckernel::SrcA : ckernel::SrcB;
                const std::uint32_t reload_loop = std::max(ct_dim, rt_dim);

#ifdef ARCH_WORMHOLE

                /*
                 * WORMHOLE SCHEME:
                 * Utilizes both source register banks to maximize throughput.
                 * IF CT_DIM >= RT_DIM ->
                 *   SRCB, SRCB, SRCA * CT_DIM, SRCB, SRCB, ...
                 * ELSE ->
                 *   SRCA, SRCA, SRCB * RT_DIM, SRCA, SRCA, ...
                 */

                std::uint32_t reuse_iter = 0;
                while (reuse_iter < reuse_loop)
                {
                    const std::uint32_t reuse_burst = std::min(static_cast<std::uint32_t>(2), reuse_loop - reuse_iter);

                    for (std::uint32_t i = 0; i < reuse_burst; i++)
                    {
                        _perf_unpack_set_valid(reuse_reg);
                    }

                    for (std::uint32_t i = 0; i < reload_loop; i++)
                    {
                        _perf_unpack_set_valid(reload_reg);
                    }

                    reuse_iter += reuse_burst;
                }
#endif

#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)

                /*
                 * BLACKHOLE/QUASAR SCHEME:
                 * Utilizes only one source register bank.
                 * IF CT_DIM >= RT_DIM ->
                 *   SRCB, SRCA * CT_DIM, SRCB, SRCA * CT_DIM, ...
                 * ELSE ->
                 *   SRCA, SRCB * RT_DIM, SRCA, SRCB * RT_DIM, ...
                 */

                for (std::uint32_t reuse_iter = 0; reuse_iter < reuse_loop; reuse_iter++)
                {
                    _perf_unpack_set_valid(reuse_reg);

                    for (std::uint32_t i = 0; i < reload_loop; i++)
                    {
                        _perf_unpack_set_valid(reload_reg);
                    }
                }
#endif
            }
        }
    }
}

template <PerfRunType RUN_TYPE>
inline void _perf_math_matmul_mock(std::uint32_t loop_factor, std::uint32_t rt_dim, std::uint32_t kt_dim, std::uint32_t ct_dim, std::uint32_t num_blocks = 1)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::matmul<false>(loop_factor, rt_dim, kt_dim, ct_dim, num_blocks);
    }
#endif
    for (std::uint32_t loop = 0; loop < loop_factor; loop++)
    {
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            for (std::uint32_t j = 0; j < kt_dim; j++)
            {
                const std::uint32_t reuse_reg  = ct_dim >= rt_dim ? ckernel::SrcB : ckernel::SrcA;
                const std::uint32_t reuse_loop = std::min(ct_dim, rt_dim);

                const std::uint32_t reload_reg  = ct_dim >= rt_dim ? ckernel::SrcA : ckernel::SrcB;
                const std::uint32_t reload_loop = std::max(ct_dim, rt_dim);

#ifdef ARCH_WORMHOLE

                /*
                 * WORMHOLE SCHEME:
                 * Utilizes both source register banks to maximize throughput.
                 * IF CT_DIM >= RT_DIM ->
                 *  SRCA * CT_DIM, SRCB, SRCB, SRCA * CT_DIM, ...
                 * ELSE ->
                 *  SRCB * RT_DIM, SRCA, SRCA, SRCB * RT_DIM, ...
                 */

                std::uint32_t reuse_iter = 0;
                while (reuse_iter < reuse_loop)
                {
                    const std::uint32_t reuse_burst = std::min(static_cast<std::uint32_t>(2), reuse_loop - reuse_iter);

                    for (std::uint32_t i = 0; i < reload_loop; i++)
                    {
                        _perf_math_clear_valid(reload_reg);
                    }

                    for (std::uint32_t i = 0; i < reuse_burst; i++)
                    {
                        _perf_math_clear_valid(reuse_reg);
                    }

                    reuse_iter += reuse_burst;
                }
#endif

#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)

                /*
                 * BLACKHOLE/QUASAR SCHEME:
                 * Utilizes only one source register bank.
                 * IF CT_DIM >= RT_DIM ->
                 *  SRCA * CT_DIM, SRCB, SRCA * CT_DIM, SRCB, ...
                 * ELSE ->
                 *  SRCB * RT_DIM, SRCA, SRCB * RT_DIM, SRCA, ...
                 */

                for (std::uint32_t reuse_iter = 0; reuse_iter < reuse_loop; reuse_iter++)
                {
                    for (std::uint32_t i = 0; i < reload_loop; i++)
                    {
                        _perf_math_clear_valid(reload_reg);
                    }

                    _perf_math_clear_valid(reuse_reg);
                }
#endif
            }
        }
    }
}

// Functional builds take the dimensions from runtime params, so the shape is const.
// Speed-of-light compiles those dimensions in, so the same declaration is constexpr.
#ifdef SPEED_OF_LIGHT
#define TEST_TENSOR_SHAPE(name, face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim) \
    constexpr ckernel::TensorShape name = ckernel::make_tensor_shape(face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
#else
#define TEST_TENSOR_SHAPE(name, face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim) \
    const ckernel::TensorShape name = ckernel::make_tensor_shape(face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim)
#endif
