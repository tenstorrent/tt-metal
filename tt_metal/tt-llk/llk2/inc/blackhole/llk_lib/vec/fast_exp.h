// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: the fast exp, exp(scale x) after Schraudolph ("A Fast, Compact Approximation of the Exponential
// Function"), no input clamping. Unlike the other function files (llk_lib/vec/common.h) it walks whole tiles, and on
// silicon it is HAL calls throughout:
//   vec_fast_exp_init<Scale>()          its part of vec_configure: three constants and, on silicon, the macro
//                                       instructions, macro sequence 0, the LOADMACRO control word and the replay slots
//   vec_fast_exp_tiles(tiles)           silicon: the recorded body twice per tile, tile after tile, then the drain
//   vec_fast_exp_plain_iterations<N>()  the ttsim build (DISABLE_SFPLOADMACRO): the same arithmetic as a plain
//                                       load / store loop
// llk_lib/vec.h picks the walk (vec_function_tile).
//
// The arithmetic. An FP32's bits read as an integer are linear in its log2, so exp(x) = the float whose bits are
// i = A x + (B - C), A = 2^23 / ln 2, B = 127 * 2^23, C an error-minimizing adjustment. The SFPU rounds FP32 to a
// 16-bit integer only, so the arithmetic is done 15 bits lower (A = 2^8 / ln 2) and the integer shifted up by 15;
// the scale is folded into A. The sign: below about -88 i is negative and the result comes out negative instead of
// wrong-and-positive, so a ReLU after it gives 0 there; above 0.72 it saturates at exp(0.72).
//
// The two forms, instruction by instruction, on the 32 datums one load takes (L12 = A scale, L13 = B - C, L14 = 15):
//
//   plain loop (vec_fast_exp_plain_iterations)        macro form (vec_fast_exp_init, vec_fast_exp_tiles)
//   SFPLOAD    x -> L0                                LOADMACRO x -> L(i % 4); its sequence then runs, on that register,
//   SFPMAD     L12 * L0 + L13 -> L0                     macro instruction 5, at once
//   STOCH_RND  L0 -> a sign-magnitude 16-bit integer    macro instruction 6, delay 3
//   SFPSHFT2   L0 << L14 -> L1                        SFPSHFT2 -> L4, issued by hand two LOADMACROs later
//   SFPSETSGN  L1 with L0's sign -> L0                  macro instruction 7, delay 5, into the staging register
//   SFPSTORE   L0                                       macro instruction 3 (the store), delay 6, from the staging register
//
// In the macro form the loads take L0..L3 in turn, so four iterations are in flight, and the shift of an iteration
// sits between its round and its set-sign: two LOADMACROs after its own. The delays count instructions issued to the
// SFPU (the control word), so a stalled pipeline cannot move a macro instruction past the shift it waits for; for
// the same reason only one thread may issue SFPU instructions while a schedule is pending (llk_lib/vec/common.h).
// Ported from: ckernel_sfpu_exp.h (the same sequence bytes and drain).

#pragma once

#include <cstdint>

#include "hal/nop.h"
#include "hal/replay.h"
#include "llk_lib/math_common.h" // counter_step, set_addr_mod
#include "llk_lib/platform/sfpu.h"
#include "llk_lib/vec/common.h"
#include "wrapper/wrapper.h"

namespace llk2::llk_lib
{

/// Replay slots of the fast exp's body on the thread that drives the SFPU: 16 LOADMACRO + shift pairs, each pair one
/// iteration (vec_fast_exp_init records them, vec_fast_exp_tiles plays them).
inline constexpr hal::replay::BufferRange vec_fast_exp_body {0, 32};

/// The shift that turns the rounded 16-bit integer into FP32 bits (L14).
inline constexpr std::uint32_t vec_fast_exp_shift = 15;

/// One iteration of the fast exp's body: a LOADMACRO into Loaded, and the shift of the iteration two before it, whose
/// rounded value is in Shifted by now.
template <platform::sfpu::Lreg Loaded, platform::sfpu::Lreg Shifted>
inline __attribute__((always_inline)) void vec_fast_exp_iteration()
{
    using namespace platform::sfpu;
    load_macro<LoadMacro {.destination = Loaded, .sequence = MacroSequence::S0, .address_modifier = vec_addr_mod_step}>();
    shift_by_register<ShiftByRegister {.value = Shifted, .amount = Lreg::L14, .destination = Lreg::L4}>();
}

/// The fast exp's SFPU programming for exp(Scale x): L12..L14 and, on silicon, macro instructions 5..7, macro
/// sequence 0, the LOADMACRO control word and the recorded body (the file comment has the schedule). Without
/// SFPLOADMACRO (the ttsim build) only the three constants are programmed: the plain loop needs nothing else.
template <std::uint32_t Scale>
inline __attribute__((always_inline)) void vec_fast_exp_init()
{
    using namespace platform::sfpu;
    constexpr float ln2_recip = 1.4426950408889634f;
    constexpr float a         = 256.0f * ln2_recip;
    constexpr float b_minus_c = 32500.818359375f;
    constexpr float a_scaled  = a * __builtin_bit_cast(float, Scale);

    vec_set_constant<ConfigTarget::Constant12, __builtin_bit_cast(std::uint32_t, a_scaled)>();
    vec_set_constant<ConfigTarget::Constant13, __builtin_bit_cast(std::uint32_t, b_minus_c)>();
    vec_set_constant<ConfigTarget::Constant14, vec_fast_exp_shift>();

    if constexpr (vec_has_loadmacro)
    {
        // The three macro instructions. The loaded value is each one's second operand or its destination, as the
        // sequence below says; the registers named here are the rest.
        install_macro_instruction<
            MacroInstruction::M5,
            multiply_add_operation<MultiplyAdd {.multiplicand = Lreg::L12, .multiplier = Lreg::L0, .addend = Lreg::L13, .destination = Lreg::L0}>()>();
        install_macro_instruction<
            MacroInstruction::M6,
            round_to_integer_operation<RoundToInteger {.value = Lreg::L0, .destination = Lreg::L0, .target = RoundTarget::SignMagnitude16}>()>();
        install_macro_instruction<MacroInstruction::M7, set_sign_operation<SetSign {.magnitude = Lreg::L4, .sign_and_destination = Lreg::L0}>()>();

        // Macro sequence 0 (vec_macro_sequence, llk_lib/vec/common.h; programmable macro instruction n is macro
        // instruction 4 + n, the HAL's M4..M7).
        constexpr vec_macro_sequence sequence {
            // set-sign (macro instruction 7): the shifted bits (L4) with the sign of the loaded register, into the
            // staging register
            .simple = {.instruction = vec_macro_programmable(3), .delay = 5, .loaded_is_vb = true, .staging = true},
            // multiply-add (5) at once: the loaded register is the multiplier
            .multiply_add = {.instruction = vec_macro_programmable(1), .delay = 0, .loaded_is_vb = true},
            // round (6) the loaded register
            .round = {.instruction = vec_macro_programmable(2), .delay = 3},
            // store (3) from the staging register
            .store = {.instruction = vec_macro_store, .delay = 6, .staging = true},
        };
        load_immediate32<Lreg::L0, vec_macro_sequence_bits(sequence)>();
        configure<Configure {.target = ConfigTarget::MacroSequence0}>();

        // Every lane's LOADMACRO control: the stores' format bits 0, and each unit's delay counting issued instructions
        // only, so a stalled pipeline cannot move a macro instruction past the shift it waits for.
        constexpr vec_macro_control control {.delay_counts_instructions = vec_macro_every_unit};
        configure<Configure {.target = ConfigTarget::MacroControl, .source = ConfigSource::Immediate, .immediate = vec_macro_control_bits(control)}>();

        // The address modifier the body's LOADMACROs name: Dest on by two rows.
        set_addr_mod<vec_addr_mod_step, counter_stay, counter_stay, counter_step {.incr = vec_rows_per_iteration_step}>();

        // The body: 16 iterations. Iteration i loads into L(i % 4); the shift next to it is iteration i - 2's.
        hal::replay::record<vec_fast_exp_body>(
            [] __attribute__((always_inline))
            {
#pragma GCC unroll 4
                for (std::uint32_t i = 0; i < vec_fast_exp_body.count / 2 / 4; i++)
                {
                    vec_fast_exp_iteration<Lreg::L0, Lreg::L2>();
                    vec_fast_exp_iteration<Lreg::L1, Lreg::L3>();
                    vec_fast_exp_iteration<Lreg::L2, Lreg::L0>();
                    vec_fast_exp_iteration<Lreg::L3, Lreg::L1>();
                }
            });
        // The record above only records, so this NOP is the one SFPU instruction that runs between the writes of the
        // sequence and the control word and the first LOADMACRO. SFPLOADMACRO.md ("Instruction scheduling") asks for an
        // SFPNOP there: a LOADMACRO that is the very next SFPU instruction may see the old sequence or control word.
        hal::nop::vector();
    }
}

/// The fast exp on Tiles whole tiles that follow each other in Dest, from the Dest row counter on: the recorded body
/// (vec_fast_exp_body, 16 iterations) twice per tile, then the pipeline's drain. Each LOADMACRO loads four rows' even
/// or odd columns and its macro sequence multiplies, rounds and stores them; the shift between rounding and storing
/// is the SFPSHFT2 issued two LOADMACROs later. So the body's first two shifts belong to the two iterations before it
/// (the previous play's, or the previous tile's: the row counter and the pipeline run on from tile to tile), and only
/// the last two iterations of the whole run need their shifts issued by hand: the drain. The LOADMACROs step the Dest
/// row counter themselves, two rows each, through the stepping address modifier (vec_addr_mod_step), which
/// vec_fast_exp_init programs.
/// Needs what vec_fast_exp_init programmed: L12..L14, macro instructions 5..7, macro sequence 0 and the replay slots.
template <class Tiles>
inline __attribute__((always_inline)) void vec_fast_exp_tiles(const Tiles& tiles)
{
    using platform::sfpu::Lreg;
    constexpr std::uint32_t iterations_per_play = vec_fast_exp_body.count / 2; // a LOADMACRO and a shift each
    constexpr std::uint32_t plays_per_tile      = vec_iterations_per_face * faces_per_full_tile / iterations_per_play;
    for (std::uint32_t play = 0; play < plays_per_tile * wrapper::value(tiles); play++)
    {
        hal::replay::run<vec_fast_exp_body>();
    }
    // The drain: the shifts of the last two iterations (their rounded values are in L2 and L3), each behind a NOP
    // where the body has a LOADMACRO, then two more NOPs. The delays count SFPU instructions, so the pending
    // set-signs and stores run only as further SFPU instructions are issued.
    hal::nop::vector();
    platform::sfpu::shift_by_register<platform::sfpu::ShiftByRegister {.value = Lreg::L2, .amount = Lreg::L14, .destination = Lreg::L4}>();
    hal::nop::vector();
    platform::sfpu::shift_by_register<platform::sfpu::ShiftByRegister {.value = Lreg::L3, .amount = Lreg::L14, .destination = Lreg::L4}>();
    hal::nop::vector();
    hal::nop::vector();
}

#ifdef LLK2_SFPU_THREAD

/// The fast exp without SFPLOADMACRO (the ttsim build), Iterations iterations from the Dest row counter on: the plain
/// load / store loop of what vec_fast_exp_tiles runs from macros on silicon. Needs L12..L14 as vec_fast_exp_init
/// programs them.
template <int Iterations>
void vec_fast_exp_plain_iterations()
{
    for (int d = 0; d < Iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, addr_mod_hold, 0);
        TTI_SFPMAD(p_sfpu::LREG12, p_sfpu::LREG0, p_sfpu::LREG13, p_sfpu::LREG0, 0);
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT16);
        TTI_SFPSHFT2(p_sfpu::LREG0, p_sfpu::LREG14, p_sfpu::LREG1, 5); // lreg[1] = lreg[0] << 15
        TTI_SFPSETSGN(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);             // lreg[0] preserves sign, copies e/m from lreg[1]
        TTI_SFPSTORE(p_sfpu::LREG0, 0, addr_mod_hold, 0);
        sfpi::dst_reg++;
    }
}

#endif // LLK2_SFPU_THREAD

} // namespace llk2::llk_lib
