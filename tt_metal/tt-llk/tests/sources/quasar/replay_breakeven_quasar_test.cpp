// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Micro-benchmark: when does the replay buffer pay off for an SFPU row loop on Quasar?
//
// Quasar cannot execute while recording (TEN-4690), so a rolled-up loop costs
// REPLAY(load) + L body instructions (not executed) + N REPLAYs, against N * L instructions
// pushed inline. Each config is timed on the math thread over TILES tiles of 4 faces:
//   INLINE          N * L instructions pushed directly
//   RECORD_REPLAY   record the body and replay it N times, on every call (what sfpi emits today)
//   REPLAY_ONLY     the body is recorded once before the timed region, each call is N REPLAYs
//   EMPTY           no body; the per-tile start / face-advance / done overhead alone
//   REAL_KERNEL     a production sfpi kernel (build with and without -mno-tt-tensix-optimize-replay)
//   RECORD_EXEC     execute-while-loading; hangs while the TEN-4690 erratum is present
// Results (wall-clock cycles) go to RESULT_ADDR in L1 for the host to read.
// Scratch benchmark, not a pass/fail LLK test.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

#ifdef LLK_TRISC_UNPACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_ops.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "params.h"
#include "quasar_test_common.h"

// Kept in a separate include block: the macros header above declares the op inits and must come
// before the kernel headers that define them (-Werror=redundant-decls otherwise).
#include "llk_sfpu/ckernel_sfpu_abs.h"
#include "llk_sfpu/ckernel_sfpu_cbrt.h"
#include "llk_sfpu/ckernel_sfpu_erf.h"
#include "llk_sfpu/ckernel_sfpu_mish.h"
#include "llk_sfpu/ckernel_sfpu_sign.h"
#include "llk_sfpu/ckernel_sfpu_softsign.h"
#include "llk_sfpu/ckernel_sfpu_square.h"

using namespace ckernel;
using namespace ckernel::math;

namespace
{
constexpr std::uint32_t RESULT_ADDR  = 0x160000;
constexpr std::uint32_t RESULT_MAGIC = 0x52504c59; // "RPLY"
constexpr std::uint32_t TILES        = 8;
constexpr std::uint32_t PASSES       = 2;

enum Variant : std::uint32_t
{
    INLINE        = 0,
    RECORD_REPLAY = 1,
    REPLAY_ONLY   = 2,
    EMPTY         = 3,
    REAL_KERNEL   = 4, // a production sfpi kernel; the build flag decides whether the compiler rolls it into replay
    RECORD_EXEC   = 5, // execute while loading (the mode TEN-4690 broke), then N - 1 REPLAYs
};

enum RealOp : int
{
    OP_ABS      = 0,
    OP_SQUARE   = 1,
    OP_SIGN     = 2,
    OP_SOFTSIGN = 3,
    OP_ERF      = 4,
    OP_MISH     = 5,
    OP_CBRT     = 6,
};

// Emit instruction words exactly as they appear in a disassembled kernel (the RISC-V encoding is the
// Tensix encoding rotated left by two).
template <std::uint32_t WORD>
__attribute__((always_inline)) inline void raw()
{
    TTI_INSN((WORD >> 2) | (WORD << 30));
}

template <std::uint32_t... WORDS>
__attribute__((always_inline)) inline void raws()
{
    (raw<WORDS>(), ...);
}

// One row-pair of an SFPU kernel, L instructions long: SFPLOAD, L - 3 ops chosen by MIX, SFPSTORE, INCRWC.
// MIX 5-9 are fixed sequences lifted from the sfpi-compiled mish kernel: 5 is the 32 instructions the
// compiler records for it, 6-9 are pieces of that window.
template <int L, int MIX>
__attribute__((always_inline)) inline void body()
{
    if constexpr (MIX == 5)
    {
        static_assert(L == 32);
        raws<
            0xc0838001,
            0xc4010401,
            0x5c004806,
            0xc4445695,
            0xc40aa8ed,
            0xc420fee1,
            0xc4c12d01,
            0x10080c02,
            0x3c000002,
            0xe40000d9,
            0xd72d0001,
            0xc509c801,
            0xc522fcc5,
            0x10010902,
            0xc54afa39,
            0xc562d6fd,
            0x10015002,
            0xc508bdb9,
            0xc520f025,
            0x10041042,
            0xc50ab6b5,
            0xc520f4a9,
            0x10041042,
            0xc50aa8a1,
            0xc520f8a9,
            0x10041042,
            0xc50bffed,
            0xc520fbfd,
            0x10041042,
            0x10042842,
            0x10042802,
            0xdc000105>();
        return;
    }
    if constexpr (MIX == 6)
    {
        // SFPLOADI pair into L4, then an SFPMAD that reads L4 and the previous SFPMAD's L1.
        static_assert(L == 12);
        raws<0xc508bdb9, 0xc520f025, 0x10041042, 0xc50ab6b5, 0xc520f4a9, 0x10041042, 0xc50aa8a1, 0xc520f8a9, 0x10041042, 0xc50bffed, 0xc520fbfd, 0x10041042>();
        return;
    }
    if constexpr (MIX == 7)
    {
        // SFPLOAD, SFPLOADI, SFPGT, four SFPLOADIs, SFPMAD.
        static_assert(L == 8);
        raws<0xc0838001, 0xc4010401, 0x5c004806, 0xc4445695, 0xc40aa8ed, 0xc420fee1, 0xc4c12d01, 0x10080c02>();
        return;
    }
    if constexpr (MIX == 8)
    {
        // SFPMAD, SFPNOP, SFPIADD, SFPADDI, then two SFPLOADI pairs each feeding an SFPMAD.
        static_assert(L == 10);
        raws<0x10080c02, 0x3c000002, 0xe40000d9, 0xd72d0001, 0xc509c801, 0xc522fcc5, 0x10010902, 0xc54afa39, 0xc562d6fd, 0x10015002>();
        return;
    }
    if constexpr (MIX == 9)
    {
        // Three back-to-back dependent SFPMADs and an SFPEXEXP, twice.
        static_assert(L == 8);
        raws<0x10041042, 0x10042842, 0x10042802, 0xdc000105, 0x10041042, 0x10042842, 0x10042802, 0xdc000105>();
        return;
    }
    if constexpr (L >= 3)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
#pragma GCC unroll 32
        for (int k = 0; k < L - 3; k++)
        {
            if constexpr (MIX == 0)
            {
                // LREG0 = LREG0 * LREG10 (1.0) + LREG9 (0.0)
                TTI_SFPMAD(p_sfpu::LREG0, 10 /*lreg_b*/, 9 /*lreg_c*/, p_sfpu::LREG0, 0 /*instr_mod1*/);
            }
            else if constexpr (MIX == 1)
            {
                TTI_SFPABS(p_sfpu::LREG0, p_sfpu::LREG0, 1 /*instr_mod1: float*/);
            }
            else if constexpr (MIX == 2)
            {
                // Constant loads, as sfpi emits them for a 32-bit float: FP16B half, then the upper half.
                if (k % 2 == 0)
                {
                    TTI_SFPLOADI(1 + (k / 2) % 5 /*lreg_ind*/, 2 /*instr_mod0*/, 0x3F80 /*imm16*/);
                }
                else
                {
                    TTI_SFPLOADI(1 + (k / 2) % 5 /*lreg_ind*/, 8 /*instr_mod0*/, 0x3F80 /*imm16*/);
                }
            }
            else if constexpr (MIX == 3)
            {
                // Independent SFPMADs: no instruction reads the result of the one before it.
                TTI_SFPMAD(p_sfpu::LREG0, 10 /*lreg_b*/, 9 /*lreg_c*/, 1 + k % 5 /*lreg_dest*/, 0 /*instr_mod1*/);
            }
            else
            {
                // Independent one-cycle ops.
                TTI_SFPMOV(p_sfpu::LREG0, 1 + k % 5 /*lreg_dest*/, 0 /*instr_mod1*/);
            }
        }
    }
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::FP32, ADDR_MOD_7, 0 /*done*/, 0 /*dest_reg_addr*/);
    TTI_INCRWC(0 /*rwc_cr*/, 0 /*rwc_a*/, 0 /*rwc_b*/, 2 /*rwc_d*/);
}

template <int L, int MIX>
__attribute__((always_inline)) inline void record()
{
    load_replay_buf<0 /*start*/, L /*len*/>([] { body<L, MIX>(); });
}

// One calculate_*() call: N row-pairs of one face.
template <std::uint32_t VARIANT, int L, int N, int MIX>
__attribute__((always_inline)) inline void face_call()
{
    if constexpr (VARIANT == INLINE)
    {
#pragma GCC unroll 32
        for (int n = 0; n < N; n++)
        {
            body<L, MIX>();
        }
    }
    else if constexpr (VARIANT == RECORD_REPLAY || VARIANT == REPLAY_ONLY)
    {
        if constexpr (VARIANT == RECORD_REPLAY)
        {
            record<L, MIX>();
        }
#pragma GCC unroll 32
        for (int n = 0; n < N; n++)
        {
            TTI_REPLAY(0 /*start_idx*/, L /*len*/, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
        }
    }
    else if constexpr (VARIANT == RECORD_EXEC)
    {
        TTI_REPLAY(0 /*start_idx*/, L /*len*/, 0 /*last*/, 0 /*set_mutex*/, 1 /*execute_while_loading*/, 1 /*load_mode*/);
        body<L, MIX>();
#pragma GCC unroll 32
        for (int n = 1; n < N; n++)
        {
            TTI_REPLAY(0 /*start_idx*/, L /*len*/, 0 /*last*/, 0 /*set_mutex*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
        }
    }
    else if constexpr (VARIANT == REAL_KERNEL)
    {
        // L carries the RealOp here.
        if constexpr (L == OP_ABS)
        {
            ckernel::sfpu::calculate_abs<false /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, true /*is_fp32_dest_acc_en*/>();
        }
        else if constexpr (L == OP_SQUARE)
        {
            ckernel::sfpu::calculate_square<8 /*ITERATIONS*/>();
        }
        else if constexpr (L == OP_SIGN)
        {
            ckernel::sfpu::calculate_sign<false /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/>(0 /*exponent_size_8*/);
        }
        else if constexpr (L == OP_SOFTSIGN)
        {
            ckernel::sfpu::calculate_softsign<false /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/>();
        }
        else if constexpr (L == OP_ERF)
        {
            ckernel::sfpu::calculate_erf<false /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/, true /*is_fp32_dest_acc_en*/>();
        }
        else if constexpr (L == OP_MISH)
        {
            ckernel::sfpu::calculate_mish<false /*APPROXIMATION_MODE*/, true /*is_fp32_dest_acc_en*/, 8 /*ITERATIONS*/>();
        }
        else
        {
            ckernel::sfpu::calculate_cube_root<false /*APPROXIMATION_MODE*/, true /*is_fp32_dest_acc_en*/, 8 /*ITERATIONS*/>();
        }
    }
}

template <int OP>
__attribute__((noinline)) void real_init()
{
    if constexpr (OP == OP_ABS)
    {
        ckernel::sfpu::abs_init();
    }
    else if constexpr (OP == OP_SQUARE)
    {
        ckernel::sfpu::init_square();
    }
    else if constexpr (OP == OP_SIGN)
    {
        ckernel::sfpu::sign_init();
    }
    else if constexpr (OP == OP_SOFTSIGN)
    {
        ckernel::sfpu::init_softsign<false /*APPROXIMATION_MODE*/>();
    }
    else if constexpr (OP == OP_ERF)
    {
        ckernel::sfpu::erf_init<false /*APPROXIMATION_MODE*/>();
    }
    else if constexpr (OP == OP_MISH)
    {
        ckernel::sfpu::mish_init<false /*APPROXIMATION_MODE*/>();
    }
    else
    {
        ckernel::sfpu::cube_root_init<false /*APPROXIMATION_MODE*/>();
    }
}

// The per-tile sequence of _llk_math_eltwise_unary_sfpu_params_ in VectorMode::RC.
template <std::uint32_t VARIANT, int L, int N, int MIX>
__attribute__((noinline)) void run_tiles(std::uint32_t tiles)
{
    for (std::uint32_t t = 0; t < tiles; t++)
    {
        _llk_math_eltwise_sfpu_start_(0 /*tile_index*/);
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            face_call<VARIANT, L, N, MIX>();
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
        _llk_math_eltwise_sfpu_done_();
    }
}

template <int L, int MIX>
__attribute__((noinline)) void record_once()
{
    record<L, MIX>();
}

inline void no_setup()
{
}

// Wait until every execution unit this thread drives is idle on two consecutive reads.
__attribute__((noinline)) void drain()
{
    constexpr std::uint32_t LOCAL_BUSY_MASK = 0x7FF;
    std::uint32_t idle_reads                = 0;
    while (idle_reads < 2)
    {
        idle_reads = (csr_read<CSR::tensix_busy_status>() & LOCAL_BUSY_MASK) == 0 ? idle_reads + 1 : 0;
    }
}

volatile std::uint32_t* out = reinterpret_cast<volatile std::uint32_t*>(RESULT_ADDR);
std::uint32_t out_idx       = 2; // [0] magic (written last), [1] record count

__attribute__((noinline)) void measure(std::uint32_t variant, std::uint32_t len, std::uint32_t iters, void (*setup)(), void (*run)(std::uint32_t))
{
    for (std::uint32_t pass = 0; pass < PASSES; pass++)
    {
        setup();
        drain();
        const std::uint32_t t_start = static_cast<std::uint32_t>(read_wall_clock());
        run(TILES);
        const std::uint32_t t_pushed = static_cast<std::uint32_t>(read_wall_clock());
        drain();
        const std::uint32_t t_done = static_cast<std::uint32_t>(read_wall_clock());

        out[out_idx++] = variant;
        out[out_idx++] = len;
        out[out_idx++] = iters;
        out[out_idx++] = t_pushed - t_start;
        out[out_idx++] = t_done - t_start;
    }
}

template <int OP>
void bench_real()
{
    measure(REAL_KERNEL, OP, 8, real_init<OP>, run_tiles<REAL_KERNEL, OP, 8, BENCH_MIX>);
}

template <int L, int N, int MIX>
void bench_exec()
{
    measure(RECORD_EXEC, L, N, no_setup, run_tiles<RECORD_EXEC, L, N, MIX>);
}

template <int L, int N, int MIX>
void bench()
{
    measure(INLINE, L, N, no_setup, run_tiles<INLINE, L, N, MIX>);
    measure(RECORD_REPLAY, L, N, no_setup, run_tiles<RECORD_REPLAY, L, N, MIX>);
    measure(REPLAY_ONLY, L, N, record_once<L, MIX>, run_tiles<REPLAY_ONLY, L, N, MIX>);
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
    _llk_math_srcAB_hw_configure_<true /*EN_32BIT_DEST*/>(DataFormat::Float32, DataFormat::Float32);
    _llk_math_eltwise_sfpu_init_();
    set_up_zero_dest_dvalid_handshake_for_math();
    set_up_zero_dest_dvalid_handshake_for_sfpu();

    measure(EMPTY, 0, 0, no_setup, run_tiles<EMPTY, 2, 1, BENCH_MIX>);

    if constexpr (BENCH_SET == 0)
    {
        bench<2, 8, BENCH_MIX>();
        bench<3, 8, BENCH_MIX>();
        bench<4, 8, BENCH_MIX>();
        bench<5, 8, BENCH_MIX>();
        bench<6, 8, BENCH_MIX>();
        bench<8, 8, BENCH_MIX>();
    }
    else if constexpr (BENCH_SET == 1)
    {
        bench<10, 8, BENCH_MIX>();
        bench<12, 8, BENCH_MIX>();
        bench<16, 8, BENCH_MIX>();
        bench<20, 8, BENCH_MIX>();
        bench<24, 8, BENCH_MIX>();
        bench<28, 8, BENCH_MIX>();
    }
    else if constexpr (BENCH_SET == 2)
    {
        bench<4, 1, BENCH_MIX>();
        bench<4, 2, BENCH_MIX>();
        bench<4, 4, BENCH_MIX>();
        bench<4, 16, BENCH_MIX>();
        bench<8, 1, BENCH_MIX>();
        bench<8, 2, BENCH_MIX>();
        bench<8, 4, BENCH_MIX>();
        bench<8, 16, BENCH_MIX>();
    }
    else if constexpr (BENCH_SET == 4)
    {
        bench_real<OP_ABS>();
        bench_real<OP_SQUARE>();
        bench_real<OP_SIGN>();
        bench_real<OP_SOFTSIGN>();
        bench_real<OP_ERF>();
        bench_real<OP_MISH>();
        bench_real<OP_CBRT>();
    }
    else if constexpr (BENCH_SET == 6)
    {
        bench<12, 8, 6>();
        bench<10, 8, 8>();
        bench<8, 8, 9>();
        bench<8, 8, 7>();
        bench<32, 8, 5>();
        bench<32, 1, 5>();
        bench<32, 2, 5>();
        TTI_SFPENCC(3 /*imm12_math*/, 10 /*instr_mod1*/); // back to the state sfpi kernels leave
    }
    else if constexpr (BENCH_SET == 7)
    {
        // Inline loop bodies of 320 to 576 instruction words: does a large unrolled body cost extra on the RISC-V side?
        bench<20, 16, BENCH_MIX>();
        bench<20, 24, BENCH_MIX>();
        bench<16, 32, BENCH_MIX>();
        bench<18, 32, BENCH_MIX>();
    }
    else if constexpr (BENCH_SET == 8)
    {
        // 640 to 896 instruction words.
        bench<20, 32, BENCH_MIX>();
        bench<24, 32, BENCH_MIX>();
        bench<28, 32, BENCH_MIX>();
    }
    else if constexpr (BENCH_SET == 5)
    {
        bench_exec<4, 8, BENCH_MIX>();
        bench_exec<16, 8, BENCH_MIX>();
    }
    else
    {
        bench<16, 1, BENCH_MIX>();
        bench<16, 2, BENCH_MIX>();
        bench<16, 4, BENCH_MIX>();
        bench<16, 16, BENCH_MIX>();
        bench<2, 1, BENCH_MIX>();
        bench<2, 2, BENCH_MIX>();
        bench<2, 4, BENCH_MIX>();
        bench<2, 16, BENCH_MIX>();
    }

    drain();
    out[1] = (out_idx - 2) / 5;
    out[0] = RESULT_MAGIC;
}

#endif
