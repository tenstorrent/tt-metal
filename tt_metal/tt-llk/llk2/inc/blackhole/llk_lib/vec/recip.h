// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: the reciprocal (llk_lib/vec/common.h has the shape of a function file).
//   vec_recip_init / vec_recip_iterations   1 / x in one of three forms, each a LOADMACRO schedule on silicon and a
//                                           plain load / store loop with DISABLE_SFPLOADMACRO:
//                                             fast  SFPARECIP alone, about 7 bits
//                                             bf16  SFPARECIP and one Newton correction on the bf16 LSB (a 16-bit Dest)
//                                             fp32  SFPARECIP and a cubic correction (a 32-bit Dest)
//                                           The macro sequences are written with vec/common.h's LOADMACRO encoders
//   vec_recip_newton, vec_recip_value       one value through SFPARECIP and Newton steps, for the functions built on
//                                           it (sigmoid, silu, gelu); vec_recip_newton_init programs its constant
// Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_recip.h, instruction for instruction.

#pragma once

#include <cstdint>

#include "llk_lib/platform/sfpu.h"
#include "llk_lib/vec/common.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

// Computes the reciprocal of a floating point value x.
// normalized=true requires a finite input with magnitude in [1, 2).
template <int max_iter = 2, bool normalized = false>
sfpi_inline sfpi::vFloat vec_recip_newton(const sfpi::vFloat x)
{
    // sfpi::approx_recip(x) will return ±0 for x = ±inf or x ≥ ±2**126, and ±inf for x = ±0.
    sfpi::vFloat y = sfpi::approx_recip(x);

    // Optionally improve the approximation using Newton-Raphson.
    if constexpr (max_iter > 0)
    {
        // Normally, t = 2.0 - x * y, but we negate this (and negate again using y = y * -t later).
        // On Blackhole, when x=0 and y=infinity (and vice versa), t=+NaN regardless of the operand signs.
        // Negating the meaning of t makes it easier to detect NaN using a trivial sign check t>=0.
        // Equivalently, we could use v_if (t >= 2.0) instead, but SFPI doesn't support SFPLE/SFPGT at the moment.
        sfpi::vFloat t = x * y - sfpi::vConstFloatPrgm0;

        if constexpr (normalized)
        {
            // Normalized operands cannot trigger the exceptional reciprocal cases.
            y = y * -t - 0.0f;
            if constexpr (max_iter > 1)
            {
                t = x * y - sfpi::vConstFloatPrgm0;
                y = y * -t - 0.0f;
            }
        }
        else if constexpr (max_iter > 1)
        {
            sfpi::vFloat y1 = y * -t - 0.0f;
            // If t=NaN, then t>=0.  This check consumes the SFPNOP slot of the preceding SFPMAD.
            v_if (t < 0)
            {
                t = x * y1 - sfpi::vConstFloatPrgm0;
                y = y1 * -t - 0.0f;
            }
            v_endif;
        }
        else
        {
            // If t=NaN, then t>=0.  This check cannot be hidden in a SFPNOP slot as it depends on the result of the
            // preceding SFPMAD.
            v_if (t < 0)
            {
                y = y * -t - 0.0f;
            }
            v_endif;
        }
    }

    return y;
}

// The fast form: SFPARECIP alone, about 7 bits. Throughput of 1c/32.
inline void vec_recip_fast_iterations(const int iterations)
{
    if constexpr (!vec_has_loadmacro)
    {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, addr_mod_hold, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, vec_addr_mod_step, 0);
    }
    }
    else
    {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOADMACRO((0 << 2) | 0, 0, vec_addr_mod_step, 0);
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    }
}

// The bf16 form: SFPARECIP and a Newton correction on the BF16 LSB.
inline void vec_recip_bf16_iterations(const int iterations)
{
    if constexpr (!vec_has_loadmacro)
    {
        TTI_SFPLOADI(p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_USHORT, 0x8000);

#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, addr_mod_hold, 0);
        TTI_SFPMAD(p_sfpu::LCONST_0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
        TTI_SFPSHFT((-16) & 0xFFF, p_sfpu::LREG1, p_sfpu::LREG1, 5);
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, vec_addr_mod_step, 0);
    }
    }
    else
    {
        constexpr int y = p_sfpu::LREG0;
        constexpr int x = p_sfpu::LREG1;

        // Macro template 0 uses SFPMAD_MOD1_INDIRECT_VD, so LREG7 selects where
        // the source x copy lands.
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_USHORT, x);

        // Pseudocode for the BF16 correction:
        //
        // y = load()
        // x = y
        // y = arecip(y)
        // y[15:0] = 0x8000
        // e = x * y - 1
        // t = e >> 16
        // y += t          # integer add, not FP32 add
        // store(y)
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TT_SFPLOADMACRO(
            /*lreg_ind*/ (0 << 2) | y,
            /*instr_mod0*/ InstrModLoadStore::DEFAULT,
            /*sfpu_addr_mode*/ addr_mod_hold,
            /*dest_reg_addr*/ 0);
        // Macro 0 schedules y = arecip(y) and x = y for the next SFPU issue.
        // Wait before writing y's low 16 bits directly.
        TTI_SFPNOP;
        // Keep the patch and correction in LReg space; macro store/reload
        // scheduling can read a just-written Dst block too soon on Blackhole.
        TTI_SFPLOADI(
            /*lreg_ind*/ y,
            /*instr_mod0*/ sfpi::SFPLOADI_MOD0_LOWER,
            /*imm16*/ 0x8000);
        TTI_SFPMAD(
            /*lreg_src_a*/ x,
            /*lreg_src_b*/ y,
            /*lreg_src_c*/ p_sfpu::LCONST_neg1,
            /*lreg_dest*/ x,
            /*instr_mod1*/ 0);
        TTI_SFPSHFT(
            /*imm12_math*/ (-16) & 0xFFF,
            /*lreg_c*/ x,
            /*lreg_dest*/ x,
            /*instr_mod1*/ 5);
        TTI_SFPIADD(
            /*imm12_math*/ 0,
            /*lreg_c*/ x,
            /*lreg_dest*/ y,
            /*instr_mod1*/ sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSTORE(
            /*lreg_ind*/ y,
            /*instr_mod0*/ InstrModLoadStore::DEFAULT,
            /*sfpu_addr_mode*/ vec_addr_mod_step,
            /*dest_reg_addr*/ 0);
    }

    TTI_SFPNOP;
    }
}

// The fp32 form's replay slots (vec_recip_fp32_init records them, vec_recip_fp32_iterations plays them):
//   0  LOADMACRO, sequence 0: [y], this vector
//   1  LOADMACRO, sequence 2: [t2], the previous vector's rows
//   2  LOADMACRO, sequence 1: [e], this vector
//   3  SFPMAD: e = 1 - e * y
//   4  LOADMACRO, sequence 3: [z], the previous vector's rows; steps the Dest row counter on
//   5  LOADMACRO, sequence 3: [z], not stepping. Recorded, and played by nothing (as in 1.0).
inline constexpr hal::replay::BufferRange vec_recip_fp32_recorded {0, 6};
inline constexpr hal::replay::BufferRange vec_recip_fp32_first {0, 4};   ///< slots 0..3: the first vector (no previous one to step past)
inline constexpr hal::replay::BufferRange vec_recip_fp32_steady {0, 5};  ///< slots 0..4: every other vector
inline constexpr hal::replay::BufferRange vec_recip_fp32_last_t2 {1, 1}; ///< slot 1: [t2] of the last vector
inline constexpr hal::replay::BufferRange vec_recip_fp32_last_z {3, 2};  ///< slots 3, 4: the SFPMAD and [z] of the last vector

// The fp32 form, with throughput of 5c/32.
inline void vec_recip_fp32_iterations(const int iterations)
{
    if constexpr (!vec_has_loadmacro)
    {
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, addr_mod_hold, 0);
        TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG2, 1); // SFPMAD_MOD1_NEGATE_VA
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG3, 0);
        // SFPMAD's result has a dynamic delay, and SFPSWAP is one of the consumers Blackhole's scoreboard does not
        // stall for (sfpi-gcc rvtt.md: xtt_delay dynamic; xtt_dynamic_bug BH on sfpswap). The compiler pads sfpi::
        // code for this, never a TTI_ word (rtl-rvtt-schedule.cc skips ttinsn), so the NOP is written here.
        // 1.0's ckernel_sfpu_recip.h has the same two lines without it.
        TTI_SFPNOP;
        TTI_SFPSWAP(0, p_sfpu::LCONST_1, p_sfpu::LREG3, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, vec_addr_mod_step, 0);
    }
    }
    else
    {
        // Pseudocode:
        //
        // y = arecip(x)
        // e = 1 - x*y
        // t = e * e + e
        // t2 = t * e + e    # e**3 + e**2 + e
        // t2 = min(t2, 1.0) # replace NaN with 1.0
        // y = t2 * y + y    # y = y * (e**3 + e**2 + e + 1)
        //                   # if y = ±0 or ±inf, then y = y+y
        //
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        //   | Load | Simple                 | MAD                     | Store   |
        // - | -----| ---------------------- | ----------------------- |-------- |
        // 0 | [y]  |                        |                         |         |
        // 1 |      | [y] = arecip(y)        |                         |         |
        // 2 | [e]  |                        |                         |         |
        // 3 |      | [e] L16 = arecip(e)    | e = mad(-e, y, 1.0)     |         |
        // 4 |      |                        |                         |         |
        // 0 |      |                        | [e] = mad(e, e, e)      | [e]     |
        // 1 | [t2] |                        |                         |         |
        // 2 |      |                        | [t2] = mad(t2, e, t2)   | [y] L16 |
        // 3 |      |                        |                         |         |
        // 4 | [z]  | [t2] = swap(t2, 1.0)   |                         |         |
        // 0 |      |                        |                         |         |
        // 1 |      |                        | [z] = mad(t2, z, z)     |         |
        // 2 |      |                        |                         |         |
        // 3 |      |                        |                         | [z]     |

        // The row label is the replay slot issued at that point (listed above); the table
        // follows one vector through three iterations.

        hal::replay::run<vec_recip_fp32_first>();
        // In slot 4's place for the first vector: there is no previous vector to finish, so a plain load (into L7, which
        // nothing here reads) makes the step of the Dest row counter.
        TTI_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, vec_addr_mod_step, 0);

#pragma GCC unroll 7
    for (int d = 0; d < iterations - 1; d++)
    {
        hal::replay::run<vec_recip_fp32_steady>();
    }

    // The last vector's second iteration, a NOP where each load of a next vector would be.
    TTI_SFPNOP;
    hal::replay::run<vec_recip_fp32_last_t2>();
    TTI_SFPNOP;
    hal::replay::run<vec_recip_fp32_last_z>();

    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    }
}

/// The fast form's LOADMACRO programming: one macro sequence, about 7 bits of precision, one instruction per iteration.
inline void vec_recip_fast_init()
{
    if constexpr (vec_has_loadmacro)
    {
        // Notation: [x] means scheduled by SFPLOADMACRO with VD=x.
        //
        // t | Load | Simple                | Store   |
        // - | ---- | --------------------- | ------- |
        // 0 | [x]  |                       |         |
        // 1 |      | [x] L16 = arecip([x]) |         |
        // 2 |      |                       | [x] L16 |

        // Programmable macro instruction 0: the reciprocal.
        TTI_SFPARECIP(0, 0, vec_macro_install_lreg(0), sfpi::SFPARECIP_MOD1_RECIP);

        // Macro sequence 0: the reciprocal of the loaded register at once, into the staging register (L16); the store
        // from there one instruction later.
        vec_set_macro_sequence<
            0,
            vec_macro_sequence {
                .simple = {.instruction = vec_macro_programmable(0), .staging = true},
                .store  = {.instruction = vec_macro_store, .delay = 1, .staging = true},
            }>();

        // Sequence 0's store takes the LOADMACRO's format; the simple unit's delays count instructions.
        vec_set_macro_control<vec_macro_control {.store_takes_load_format = 1u << 0, .delay_counts_instructions = vec_macro_simple_unit}>();
    }
}

/// The bf16 form's LOADMACRO programming: macro sequence 0 schedules the reciprocal and the copy of x; the rest of an
/// iteration is issued by vec_recip_bf16_iterations.
inline void vec_recip_bf16_init()
{
    if constexpr (vec_has_loadmacro)
    {
        // Programmable macro instruction 0: the reciprocal.
        TTI_SFPARECIP(
            /*imm12_math*/ 0,
            /*lreg_c*/ 0,
            /*lreg_dest*/ vec_macro_install_lreg(0),
            /*instr_mod1*/ sfpi::SFPARECIP_MOD1_RECIP);

        // Programmable macro instruction 1: a copy (0 * 0 + the loaded register) to the register L7 names.
        TTI_SFPMAD(
            /*lreg_src_a*/ p_sfpu::LCONST_0,
            /*lreg_src_b*/ p_sfpu::LCONST_0,
            /*lreg_src_c*/ 0,
            /*lreg_dest*/ vec_macro_install_lreg(1),
            /*instr_mod1*/ 8); // SFPMAD_MOD1_INDIRECT_VD

        // Macro sequence 0: [y]
        // Loads y, schedules y = arecip(y), and copies y to L[LREG7].
        vec_set_macro_sequence<
            0,
            vec_macro_sequence {
                .simple       = {.instruction = vec_macro_programmable(0)},
                .multiply_add = {.instruction = vec_macro_programmable(1)},
            }>();

        // No scheduled store, so the store format bits are unused. The simple and the multiply-add units' delays count
        // instructions.
        vec_set_macro_control<vec_macro_control {.delay_counts_instructions = vec_macro_simple_unit | vec_macro_multiply_add_unit}>();
    }
}

/// The fp32 form's LOADMACRO programming: four macro instructions, four macro sequences and the recorded iteration
/// (vec_recip_fp32_iterations has the schedule).
inline void vec_recip_fp32_init()
{
    if constexpr (vec_has_loadmacro)
    {
        constexpr int e  = p_sfpu::LREG0;
        constexpr int t2 = p_sfpu::LREG1;
        constexpr int z  = p_sfpu::LREG2;
        constexpr int y  = p_sfpu::LREG3;

        // Programmable macro instruction 0: the reciprocal.
        TTI_SFPARECIP(0, 0, vec_macro_install_lreg(0), sfpi::SFPARECIP_MOD1_RECIP);

        // Programmable macro instruction 1
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, 0, vec_macro_install_lreg(1), 0);

        // Programmable macro instruction 2
        // SFPMAD(VA=t2, VB=0 or VD, VC=VD or z)
        TTI_SFPMAD(t2, p_sfpu::LREG0, z, vec_macro_install_lreg(2), 0);

        // Programmable macro instruction 3
        TTI_SFPSWAP(0, p_sfpu::LCONST_1, vec_macro_install_lreg(3), sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

        // Macro sequence 0: [y]
        vec_set_macro_sequence<
            0,
            vec_macro_sequence {
                .simple = {.instruction = vec_macro_programmable(0)},
                .store  = {.instruction = vec_macro_store, .delay = 6, .staging = true},
            }>();

        // Macro sequence 1: [e]
        vec_set_macro_sequence<
            1,
            vec_macro_sequence {
                .simple       = {.instruction = vec_macro_programmable(0), .staging = true},
                .multiply_add = {.instruction = vec_macro_programmable(1), .delay = 2},
                .store        = {.instruction = vec_macro_store, .delay = 2},
            }>();

        // Macro sequence 2: [t2]. Its round and store units are empty, so the word fits SFPCONFIG's immediate.
        {
            constexpr vec_macro_sequence sequence {
                .simple       = {.instruction = vec_macro_programmable(3), .delay = 2, .loaded_is_vb = true},
                .multiply_add = {.instruction = vec_macro_programmable(2)},
            };
            TTI_SFPCONFIG(
                vec_macro_sequence_bits(sequence),
                static_cast<std::uint32_t>(platform::sfpu::ConfigTarget::MacroSequence2),
                static_cast<std::uint32_t>(platform::sfpu::ConfigSource::Immediate));
        }

        // Macro sequence 3: [z]
        // Keep the corrected result in z. Macro 0 still needs L16 for its delayed
        // store of the next vector's approximate reciprocal. Even with instruction-
        // counted delays, an issued MAD completes during scalar stalls (e.g. gcov),
        // so writing its result to L16 can clobber that value before macro 0 stores it.
        // The next load of z follows this macro's store, so z has no such overlap.
        vec_set_macro_sequence<
            3,
            vec_macro_sequence {
                .multiply_add = {.instruction = vec_macro_programmable(2), .delay = 1, .loaded_is_vb = true},
                .store        = {.instruction = vec_macro_store, .delay = 3},
            }>();

        // Every sequence's store takes the LOADMACRO's format; every unit's delays count instructions.
        vec_set_macro_control<vec_macro_control {.store_takes_load_format = 0xF, .delay_counts_instructions = vec_macro_every_unit}>();

        constexpr std::uint32_t prev_offset = -2 & 0x3ff;
        constexpr std::uint32_t offset      = 0;

        // The slots are listed at vec_recip_fp32_recorded.
        hal::replay::record<vec_recip_fp32_recorded>(
            [e, t2, z, y, offset, prev_offset]
            {
                TTI_SFPLOADMACRO((0 << 2) | (y & 3), 0, addr_mod_hold, offset | (y >> 2));
                TTI_SFPLOADMACRO((2 << 2) | (t2 & 3), 0, addr_mod_hold, prev_offset | (t2 >> 2));
                TTI_SFPLOADMACRO((1 << 2) | (e & 3), 0, addr_mod_hold, offset | (e >> 2));
                TTI_SFPMAD(p_sfpu::LREG0, y, p_sfpu::LCONST_1, 0, 1); // SFPMAD_MOD1_NEGATE_VA
                TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, vec_addr_mod_step, prev_offset | (z >> 2));
                TTI_SFPLOADMACRO((3 << 2) | (z & 3), 0, addr_mod_hold, prev_offset | (z >> 2));
            });
    }
}

/// 1 / in, one value: SFPARECIP alone (Fast) or with two Newton steps. Needs vec_recip_newton_init.
template <bool Fast = false>
sfpi_inline sfpi::vFloat vec_recip_value(const sfpi::vFloat in)
{
    return vec_recip_newton<Fast ? 0 : 2>(in);
}

/// Programs the Newton steps' constant (Prgm0 = 2.0); the fast form needs none.
template <bool Fast = false>
sfpi_inline void vec_recip_newton_init()
{
    if constexpr (!Fast)
    {
        sfpi::vConstFloatPrgm0 = 2.0f;
    }
}

template <bool Fast, bool Fp32, int Iterations = 8>
inline void vec_recip_iterations()
{
    if constexpr (Fast)
    {
        vec_recip_fast_iterations(Iterations);
    }
    else if constexpr (Fp32)
    {
        vec_recip_fp32_iterations(Iterations);
    }
    else
    {
        vec_recip_bf16_iterations(Iterations);
    }
}

template <bool Fast, bool Fp32>
void vec_recip_init()
{
    // The stepping address modifier the iterations name (Dest on by two rows) and the form's LOADMACRO programming:
    // its macro instructions, macro sequences, the control word and, for the fp32 form, replay slots 0..5. The
    // functions built on vec_recip_newton (sigmoid, silu, gelu) need none of this, only vec_recip_newton_init.
    set_addr_mod<vec_addr_mod_step, counter_stay, counter_stay, counter_step {.incr = vec_rows_per_iteration_step}>();
    vec_recip_newton_init<false>(); // set vConstFloatPrgm0 for vec_recip_newton
    if constexpr (Fast)
    {
        vec_recip_fast_init();
    }
    else if constexpr (Fp32)
    {
        vec_recip_fp32_init();
    }
    else
    {
        vec_recip_bf16_init();
    }
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD
