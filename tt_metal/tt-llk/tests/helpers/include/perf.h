// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "ckernel.h"

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

// The PACK_ISOLATE and L1_CONGESTION pack loops release every dest section as tile_regs_release does in production
// (Wormhole only for now: without the drain its four packers can lock into their slow pattern for the whole loop)
#if defined(ARCH_WORMHOLE)
constexpr bool PERF_PACK_SECTION_RELEASE = true;
#else
constexpr bool PERF_PACK_SECTION_RELEASE = false;
#endif

// LLK_PERF_OOL threads run INIT and the code after the loop out of line at the end of the kernel code (sections.ld), by
// value and noipa, so changes to them cannot move the loop. The other threads keep INIT inline.
#if defined(LLK_DBG_BARRIER) && defined(LLK_PERF_INIT_ONLY) // INIT measurement build: INIT's body out of line, timed by counters.h init_zone
#define LLK_INIT_BEGIN \
    {                  \
        auto _llk_init_body_ = [=]() __attribute__((noipa, section(".llk_init_body")))
#define LLK_INIT_END                                     \
    ;                                                    \
    llk_perf::init_zone<PERF_RUN_TYPE>(_llk_init_body_); \
    }
#elif defined(LLK_DBG_BARRIER) && defined(LLK_PERF_OOL)
#define LLK_INIT_BEGIN [=]() __attribute__((noipa, section(".llk_init_text"), aligned(512)))
#define LLK_INIT_END   ()
#else
#define LLK_INIT_BEGIN
#define LLK_INIT_END
#endif
// The code after the measured loop (uninit): placed like INIT, but never timed as INIT in the INIT measurement build.
#if defined(LLK_DBG_BARRIER) && defined(LLK_PERF_OOL)
#define LLK_UNINIT_BEGIN [=]() __attribute__((noipa, section(".llk_init_text"), aligned(512)))
#define LLK_UNINIT_END   ()
#else
#define LLK_UNINIT_BEGIN
#define LLK_UNINIT_END
#endif

// Kernels with inline INIT: the code after the loop out of line, so it cannot move the loop, and opaque INIT copies of
// values INIT and the loop share, so INIT's code cannot change the loop's registers.
#if defined(LLK_DBG_BARRIER)
#define LLK_POST_LOOP_BEGIN [=]() __attribute__((noipa, section(".llk_init_text"), aligned(512)))
#define LLK_POST_LOOP_END   ()
#define LLK_PERF_OPAQUE(v)  asm volatile("" : "+r"(v))
#else
#define LLK_POST_LOOP_BEGIN
#define LLK_POST_LOOP_END
#define LLK_PERF_OPAQUE(v)
#endif

// experiment hooks (holdall, mockfix): LLK_HA_MOCK_NOPS_U / _M = k RISC nops per handshake of the RISC-V loop unpack / math
// mock; LLK_MF_RNOPS = k RISC nops before every MOP the Wormhole MOP mock pushes; LLK_MF_TNOPS: its kind0 handshakes carry
// three Tensix NOPs
#define LLK_HA_XSTR_(x) #x
#define LLK_HA_XSTR(x)  LLK_HA_XSTR_(x)
#if defined(LLK_HA_MOCK_NOPS_U)
#define LLK_HA_MOCK_U_() asm volatile(".rept " LLK_HA_XSTR(LLK_HA_MOCK_NOPS_U) "\n\tnop\n\t.endr")
#else
#define LLK_HA_MOCK_U_()
#endif
#if defined(LLK_HA_MOCK_NOPS_M)
#define LLK_HA_MOCK_M_() asm volatile(".rept " LLK_HA_XSTR(LLK_HA_MOCK_NOPS_M) "\n\tnop\n\t.endr")
#else
#define LLK_HA_MOCK_M_()
#endif
#if defined(LLK_MF_RNOPS)
#define LLK_MF_PUSH_() asm volatile(".rept " LLK_HA_XSTR(LLK_MF_RNOPS) "\n\tnop\n\t.endr")
#else
#define LLK_MF_PUSH_()
#endif

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
// Wormhole isolate mocks: the partner that feeds (unpack mock) or drains (math mock) the measured thread hands its
// source handshakes to its own MOP expander, up to 128 per MOP instruction. Its RISC-V then pushes a few instructions
// per MOP rather than looping once per handshake, so the measured thread waits only for the handshakes themselves.
// A handshake is the same STALLWAIT + SETDVALID / CLEARDVALID pair as in the RISC-V loop mocks below.
namespace llk_perf_mock
{
// The sources one handshake gives (SETDVALID) or takes back (CLEARDVALID): bit 0 SrcA, bit 1 SrcB
constexpr std::uint32_t SRC_A  = 1;
constexpr std::uint32_t SRC_B  = 2;
constexpr std::uint32_t SRC_AB = SRC_A | SRC_B;

template <bool PRODUCE>
constexpr std::uint32_t wait_op(const std::uint32_t srcs)
{
    using ckernel::p_stall;
    return PRODUCE ? TT_OP_STALLWAIT(p_stall::STALL_TDMA, ((srcs & SRC_A) ? p_stall::SRCA_CLR : 0) | ((srcs & SRC_B) ? p_stall::SRCB_CLR : 0))
                   : TT_OP_STALLWAIT(p_stall::STALL_MATH, ((srcs & SRC_A) ? p_stall::SRCA_VLD : 0) | ((srcs & SRC_B) ? p_stall::SRCB_VLD : 0));
}

template <bool PRODUCE>
constexpr std::uint32_t flip_op(const std::uint32_t srcs)
{
    return PRODUCE ? TT_OP_SETDVALID(srcs) : TT_OP_CLEARDVALID(srcs, 0);
}

// What this thread's MOP template 0 and zmask high half hold: 0x10 | kind1 << 2 | kind0, mask_hi << 16 (0: not set
// yet in this launch, the kernel's bss is cleared at start)
inline std::uint32_t programmed = 0;

// Template 0 with one handshake per iteration, of kind0 where the zmask bit is 0 and of kind1 where it is 1. The template
// is written once per launch, after the MOP expander is idle; the zmask's high half is an instruction in the stream.
template <bool PRODUCE>
inline void program(const std::uint32_t kind0, const std::uint32_t kind1, const std::uint32_t mask_hi)
{
    const std::uint32_t kinds = 0x10 | (kind1 << 2) | kind0;
    if ((programmed & 0xFFFF) != kinds)
    {
        volatile std::uint32_t* mop_cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
        ckernel::mop_sync();
#if defined(LLK_MF_TNOPS)
        mop_cfg[1] = 3; // experiment hook: A1-A3 = NOP
        mop_cfg[4] = TT_OP_NOP;
        mop_cfg[5] = TT_OP_NOP;
        mop_cfg[6] = TT_OP_NOP;
#elif defined(LLK_MF_LAG)
        mop_cfg[1] = 3; // experiment hook: A1 = a second STALLWAIT, met at once (no unpacker 0 instruction of this thread): +latch +lag
        mop_cfg[4] = TT_OP_STALLWAIT(PRODUCE ? ckernel::p_stall::STALL_TDMA : ckernel::p_stall::STALL_MATH, ckernel::p_stall::UNPACK0);
        mop_cfg[5] = TT_OP_NOP;
        mop_cfg[6] = TT_OP_NOP;
#else
        mop_cfg[1] = 1; // the B instruction, no A1-A3
#endif
        mop_cfg[2] = flip_op<PRODUCE>(kind0);
        mop_cfg[3] = wait_op<PRODUCE>(kind0);
        mop_cfg[7] = wait_op<PRODUCE>(kind1);
        mop_cfg[8] = flip_op<PRODUCE>(kind1);
        TT_MOP_CFG(mask_hi);
    }
    else if ((programmed >> 16) != mask_hi)
    {
        TT_MOP_CFG(mask_hi);
    }
    programmed = (mask_hi << 16) | kinds;
}

// count handshakes of one kind, 128 per MOP
template <bool PRODUCE>
inline void run(const std::uint32_t kind, std::uint32_t count)
{
    program<PRODUCE>(kind, kind, 0);
    for (; count >= 128; count -= 128)
    {
        LLK_MF_PUSH_();
        TTI_MOP(0, 127, 0);
    }
    if (count != 0)
    {
        LLK_MF_PUSH_();
        TT_MOP(0, count - 1, 0);
    }
}

// periods x period handshakes, handshake j of each period of kind1 where bit j of pattern is set (period <= 32): as many
// whole periods per MOP as its 32 bit zmask holds
template <bool PRODUCE>
inline void stream(const std::uint32_t kind0, const std::uint32_t kind1, const std::uint32_t pattern, const std::uint32_t period, std::uint32_t periods)
{
    const std::uint32_t per_mop = 32 / period;
    std::uint32_t mask          = 0;
    for (std::uint32_t i = 0; i < per_mop; i++)
    {
        mask |= pattern << (i * period);
    }
    program<PRODUCE>(kind0, kind1, mask >> 16);
    const std::uint32_t mop = TT_OP_MOP(0, per_mop * period - 1, mask & 0xFFFF);
    for (; periods >= per_mop; periods -= per_mop)
    {
        LLK_MF_PUSH_();
        ckernel::instrn_buffer[0] = mop;
    }
    if (periods != 0)
    {
        LLK_MF_PUSH_();
        TT_MOP(0, periods * period - 1, mask & 0xFFFF);
    }
}

// The matmul mocks' Wormhole scheme (see _perf_unpack_matmul_mock): per kt step, the reused tiles two at a time, each
// burst followed (unpack) or preceded (math) by the reloaded tiles. kind0 = reload, kind1 = reuse.
template <bool PRODUCE>
inline void matmul(
    const std::uint32_t loop_factor, const std::uint32_t rt_dim, const std::uint32_t kt_dim, const std::uint32_t ct_dim, const std::uint32_t num_blocks)
{
    const std::uint32_t reuse       = ct_dim >= rt_dim ? SRC_B : SRC_A;
    const std::uint32_t reload      = ct_dim >= rt_dim ? SRC_A : SRC_B;
    const std::uint32_t reuse_loop  = std::min(ct_dim, rt_dim);
    const std::uint32_t reload_loop = std::max(ct_dim, rt_dim);
    std::uint32_t pattern           = 0;
    std::uint32_t period            = 0;
    for (std::uint32_t reuse_iter = 0; reuse_iter < reuse_loop; reuse_iter += 2)
    {
        const std::uint32_t burst = std::min(static_cast<std::uint32_t>(2), reuse_loop - reuse_iter);
        if (period + burst + reload_loop <= 32)
        {
            pattern |= ((1u << burst) - 1) << (PRODUCE ? period : period + reload_loop);
        }
        period += burst + reload_loop;
    }
    const std::uint32_t steps = loop_factor * num_blocks * kt_dim;
    if (period <= 32)
    {
        return stream<PRODUCE>(reload, reuse, pattern, period, steps);
    }
    // a kt step longer than the zmask: one MOP per burst and one per reload run
    program<PRODUCE>(reload, reuse, 0);
    for (std::uint32_t step = 0; step < steps; step++)
    {
        for (std::uint32_t reuse_iter = 0; reuse_iter < reuse_loop; reuse_iter += 2)
        {
            const std::uint32_t burst = std::min(static_cast<std::uint32_t>(2), reuse_loop - reuse_iter);
            if constexpr (PRODUCE)
            {
                LLK_MF_PUSH_();
                TT_MOP(0, burst - 1, (1u << burst) - 1);
            }
            for (std::uint32_t left = reload_loop; left != 0;)
            {
                const std::uint32_t count = std::min(left, static_cast<std::uint32_t>(128));
                LLK_MF_PUSH_();
                TT_MOP(0, count - 1, 0);
                left -= count;
            }
            if constexpr (!PRODUCE)
            {
                LLK_MF_PUSH_();
                TT_MOP(0, burst - 1, (1u << burst) - 1);
            }
        }
    }
}

// periods x (one SrcB handshake and a_count SrcA ones): the producer gives B first, the consumer takes it back last
template <bool PRODUCE>
inline void srcb_reuse(const std::uint32_t a_count, std::uint32_t periods)
{
    if (a_count < 32)
    {
        return stream<PRODUCE>(SRC_A, SRC_B, PRODUCE ? 1 : 1u << a_count, a_count + 1, periods);
    }
    program<PRODUCE>(SRC_A, SRC_B, 0);
    for (; periods != 0; periods--)
    {
        if constexpr (PRODUCE)
        {
            LLK_MF_PUSH_();
            TTI_MOP(0, 0, 1);
        }
        for (std::uint32_t left = a_count; left != 0;)
        {
            const std::uint32_t count = std::min(left, static_cast<std::uint32_t>(128));
            LLK_MF_PUSH_();
            TT_MOP(0, count - 1, 0);
            left -= count;
        }
        if constexpr (!PRODUCE)
        {
            LLK_MF_PUSH_();
            TTI_MOP(0, 0, 1);
        }
    }
}
} // namespace llk_perf_mock
#endif

// Mocks of a source stream in which one SrcB is reused by a_count SrcA (per period: produce B then the A, consume the A
// then B). Only the Wormhole isolate mocks call it (perf_mock_mop).
template <bool PRODUCE>
inline void _perf_mock_srcb_reuse_(const std::uint32_t a_count, const std::uint32_t periods)
{
#if defined(ARCH_WORMHOLE)
    llk_perf_mock::srcb_reuse<PRODUCE>(a_count, periods);
#endif
}

// eltwise_binary_test.cpp's source mock as one stream, BTYPE = ckernel::BroadcastType's value: COL (1) reuses a SrcB per
// face row, SCALAR (3) per tile, NONE (0) and ROW (2) hand over both sources per face. Wormhole isolate mocks only.
template <bool PRODUCE, std::uint32_t BTYPE>
inline void _perf_mock_binary_(const std::uint32_t tiles, const std::uint32_t num_faces_r, const std::uint32_t num_faces_c)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (BTYPE == 1)
    {
        llk_perf_mock::srcb_reuse<PRODUCE>(num_faces_c, tiles * num_faces_r);
    }
    else if constexpr (BTYPE == 3)
    {
        llk_perf_mock::srcb_reuse<PRODUCE>(num_faces_r * num_faces_c, tiles);
    }
    else
    {
        llk_perf_mock::run<PRODUCE>(llk_perf_mock::SRC_AB, tiles * num_faces_r * num_faces_c);
    }
#endif
}

inline void _perf_unpack_set_valid(std::uint32_t source)
{
    LLK_HA_MOCK_U_();
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
    LLK_HA_MOCK_M_();
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

// RUN_TYPE: the run type the mock serves (perf_mock_mop); the default keeps the RISC-V loop
template <bool set_a, bool set_b, PerfRunType RUN_TYPE = PerfRunType::L1_CONGESTION>
inline void _perf_unpack_loop_set_valid(std::uint32_t iterations)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::run<true>((set_b ? llk_perf_mock::SRC_B : 0) | (set_a ? llk_perf_mock::SRC_A : 0), iterations);
    }
#endif
    while (iterations-- > 0)
    {
        LLK_HA_MOCK_U_();
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

template <bool clear_a, bool clear_b, PerfRunType RUN_TYPE = PerfRunType::L1_CONGESTION>
inline void _perf_math_loop_clear_valid(std::uint32_t iterations)
{
#if defined(ARCH_WORMHOLE)
    if constexpr (perf_mock_mop<RUN_TYPE>())
    {
        return llk_perf_mock::run<false>((clear_b ? llk_perf_mock::SRC_B : 0) | (clear_a ? llk_perf_mock::SRC_A : 0), iterations);
    }
#endif
    while (iterations-- > 0)
    {
        LLK_HA_MOCK_M_();
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

template <PerfRunType RUN_TYPE = PerfRunType::L1_CONGESTION>
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

template <PerfRunType RUN_TYPE = PerfRunType::L1_CONGESTION>
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

// experiment hook (nikola, 67d5f45f2a1): N NOPs in math_matmul's pack thread after INIT, before the TILE_LOOP zone
#if defined(LLK_NK_LOOPSHIFT)
#define LLK_NK_LS_XSTR_(x)      #x
#define LLK_NK_LS_XSTR(x)       LLK_NK_LS_XSTR_(x)
#define LLK_NK_LOOPSHIFT_HOOK() asm volatile(".rept " LLK_NK_LS_XSTR(LLK_NK_LOOPSHIFT) "\n\tnop\n\t.endr")
#else
#define LLK_NK_LOOPSHIFT_HOOK() \
    do                          \
    {                           \
    } while (0)
#endif
