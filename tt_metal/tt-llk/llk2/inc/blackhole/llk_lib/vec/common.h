// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: what the SFPU function files (llk_lib/vec/<function>.h) share.
//
// The rule for the vector unit. llk-lib is written in HAL calls, or in SFPI for the body of a vector function. The
// SFPU's programming (lane control, address modifiers, counters, programmable constants, the replay buffer) is HAL
// calls, here and in the functions' inits. A function's arithmetic is SFPI: either SFPI C++ (sfpi::vFloat
// expressions, for which the compiler picks the registers) or, where a body is written instruction by instruction
// with its registers chosen by hand, the SFPI compiler's instruction builtins under their TTI_SFP* names
// (ckernel_ops.h), with the operand names of ckernel_instr_params.h (p_sfpu::LREG0, InstrModLoadStore).
//
// Every function file has the same parts:
//   vec_<function>_init<Fast, Fp32>()   the function's part of vec_configure: its programmable constants, lookup
//                                       tables, macro sequences and replay slots
//   vec_<function>_iterations<Fast, Fp32, Iterations>(scalars...)
//                                       the function on Iterations iterations from the Dest row counter on. One
//                                       iteration loads, computes and stores 32 datums (four rows' even or odd
//                                       columns) and moves the row counter on; 8 iterations are one face
//   helpers                             one value through the function (vec_<function>_value and the like), constants
// Fast is the function's fast (approximate) form, Fp32 a 32-bit Dest; a function with one form takes them and does
// not read them. The scalars of a call are FP32 bit patterns. llk_lib/vec.h picks the init and the iterations for a
// Plan's function (vec_function_init, vec_function_face).
// The exp family differs where it has to: the exact exp takes a scale (vec_exp_iterations<Fp32, Scaled,
// Iterations>(scale_bf16)), exp's half-face body runs 4 iterations and has no init, and the fast exp walks whole
// tiles (llk_lib/vec/fast_exp.h).
//
// This file also has the encoders for LOADMACRO programming (the fast exp and the reciprocals) and the rule about
// hoisting constants out of a loop (vec_hoisted_if).
//
// The SFPI code is compiled only into the threads that can drive the SFPU (math, pack): LLK2_SFPU_THREAD.

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ckernel.h"
#include "hal/cfg.h"
#include "hal/replay.h"
#include "llk_lib/math_common.h" // counter_step, set_addr_mod, reset_math_counters
#include "llk_lib/platform/sfpu.h"
#include "tensor/tensor.h" // max_face_rows, face_cols

#if defined(LLK_TRISC_MATH) || defined(TRISC_MATH) || defined(LLK_TRISC_PACK) || defined(TRISC_PACK)
#define LLK2_SFPU_THREAD 1
#include "sfpi.h"
#endif

namespace llk2::llk_lib
{

/// Whether this build may use SFPLOADMACRO schedules. Not with DISABLE_SFPLOADMACRO (TT_METAL_DISABLE_SFPLOADMACRO=1,
/// the ttsim build): the functions that have a LOADMACRO form (the fast exp, the exact fp32 exp, the reciprocals) then
/// run their plain load / store form, picked with `if constexpr` on this. So the LOADMACRO forms are covered on the
/// board only, and the plain forms on the simulator only.
#ifdef DISABLE_SFPLOADMACRO
inline constexpr bool vec_has_loadmacro = false;
#else
inline constexpr bool vec_has_loadmacro = true;
#endif

/// The address modifier slot of a store that moves the Dest row counter on by one iteration (two rows): the
/// functions written instruction by instruction program it in their init and name it in their last store.
inline constexpr std::uint8_t vec_addr_mod_step = 6;

/// Dest rows one iteration covers: what a stepping store moves the Dest row counter by.
inline constexpr std::uint32_t vec_rows_per_iteration_step = 2;

inline constexpr std::uint32_t vec_lanes               = 32; ///< SFPU lanes: datums one SFPI iteration loads, computes and stores
inline constexpr std::uint32_t vec_iterations_per_face = max_face_rows * face_cols / vec_lanes; ///< 8 iterations of 4 rows x 8 columns: one face

/// Every lane's control bits of the SFPU to 0: no lane disabled, no flag forced.
inline __attribute__((always_inline)) void vec_reset_lane_control()
{
    using namespace platform::sfpu;
    configure<Configure {.target = ConfigTarget::LaneControl, .source = ConfigSource::Immediate, .immediate = 0}>();
}

/// Programmable constant Target (L11..L14) = the 32-bit pattern Bits in every lane, through L0.
template <platform::sfpu::ConfigTarget Target, std::uint32_t Bits>
inline __attribute__((always_inline)) void vec_set_constant()
{
    using namespace platform::sfpu;
    load_immediate32<Lreg::L0, Bits>();
    configure<Configure {.target = Target}>();
}

/// The float an FP32 bit pattern holds. Not constexpr on purpose: the scalars of a vec_ call are run-time values.
inline float vec_float_of(std::uint32_t bits)
{
    return __builtin_bit_cast(float, bits);
}

/// The FP32 bit pattern of a float constant, and its two 16-bit halves as SFPLOADI takes them.
constexpr std::uint32_t vec_bits_of(float value)
{
    return __builtin_bit_cast(std::uint32_t, value);
}

constexpr std::uint16_t vec_lo16(float value)
{
    return static_cast<std::uint16_t>(vec_bits_of(value) & 0xFFFFu);
}

constexpr std::uint16_t vec_hi16(float value)
{
    return static_cast<std::uint16_t>(vec_bits_of(value) >> 16);
}

// ---- LOADMACRO programming ------------------------------------------------------------------------------------------
//
// An SFPLOADMACRO loads from Dest as an SFPLOAD does and schedules up to four more instructions, one for each sub-unit
// of the SFPU (simple, multiply-add, round, store), to run some instructions or cycles later (tt-isa-documentation,
// SFPLOADMACRO.md). What it schedules is one of four macro sequences: a 32-bit word, one byte per sub-unit
// (vec_macro_unit). A unit names a macro instruction: nothing, the store, or one of the four programmable ones, which
// a function installs by issuing the instruction with L12..L15 as its destination. One 12-bit control word
// (vec_macro_control) says how the delays count and which format the stores take.
//
// One thread at a time. A delay that counts instructions counts every instruction any thread issues to the SFPU, so
// an SFPU instruction from the other thread while a schedule is pending moves every pending macro instruction by
// one. A kernel that drives the SFPU from both threads hands it over between them (migrate, llk_lib/sync.h).

inline constexpr std::uint32_t vec_macro_nothing = 0; ///< macro instruction 0: the sub-unit gets nothing
inline constexpr std::uint32_t vec_macro_store   = 3; ///< macro instruction 3: SFPSTORE (fixed)

/// Programmable macro instruction Index (0..3) as a sequence unit names it: macro instructions 4..7 (the HAL's
/// platform::sfpu::MacroInstruction::M4..M7).
constexpr std::uint32_t vec_macro_programmable(std::uint32_t index)
{
    return 4 + index;
}

/// The destination register that installs an instruction as programmable macro instruction Index (0..3) instead of
/// executing it: L12..L15.
constexpr std::uint32_t vec_macro_install_lreg(std::uint32_t index)
{
    return 12 + index;
}

/// One unit of a macro sequence: what one sub-unit runs for each LOADMACRO of that sequence.
struct vec_macro_unit
{
    std::uint32_t instruction = vec_macro_nothing; ///< bits 2:0: vec_macro_store or vec_macro_programmable(n)
    std::uint32_t delay       = 0;                 ///< bits 5:3: instructions (or cycles: vec_macro_control) behind the LOADMACRO; 0 is at once
    bool loaded_is_vb         = false;             ///< bit 7: the loaded register is the instruction's VB operand, not its VC
    bool staging              = false;             ///< bit 6: the result goes to the staging register (LReg[16]), not to the loaded
                                                   ///< register; a store takes its value from there
};

constexpr std::uint32_t vec_macro_unit_bits(const vec_macro_unit& unit)
{
    return (unit.loaded_is_vb ? 0x80u : 0u) | (unit.staging ? 0x40u : 0u) | (unit.delay << 3) | unit.instruction;
}

/// One macro sequence: a unit for each sub-unit. A unit left out schedules nothing.
struct vec_macro_sequence
{
    vec_macro_unit simple       = {};
    vec_macro_unit multiply_add = {};
    vec_macro_unit round        = {};
    vec_macro_unit store        = {};
};

/// The 32-bit word of a macro sequence: the simple unit in the low byte, then multiply-add, round, store.
constexpr std::uint32_t vec_macro_sequence_bits(const vec_macro_sequence& sequence)
{
    return (vec_macro_unit_bits(sequence.store) << 24) | (vec_macro_unit_bits(sequence.round) << 16) | (vec_macro_unit_bits(sequence.multiply_add) << 8) |
           vec_macro_unit_bits(sequence.simple);
}

// The sub-units as bits of vec_macro_control::delay_counts_instructions.
inline constexpr std::uint32_t vec_macro_simple_unit       = 1u << 0;
inline constexpr std::uint32_t vec_macro_multiply_add_unit = 1u << 1;
inline constexpr std::uint32_t vec_macro_round_unit        = 1u << 2;
inline constexpr std::uint32_t vec_macro_store_unit        = 1u << 3;
inline constexpr std::uint32_t vec_macro_every_unit        = 0xF;

/// The LOADMACRO control word (one for all four sequences).
struct vec_macro_control
{
    std::uint32_t store_format              = 0; ///< bits 3:0: the format a scheduled store takes when its sequence's bit below is 0
    std::uint32_t store_takes_load_format   = 0; ///< bits 7:4, bit n for macro sequence n: its store takes the LOADMACRO's format
    std::uint32_t delay_counts_instructions = 0; ///< bits 11:8, one per sub-unit: its delays count instructions issued to the SFPU,
                                                 ///< not cycles
};

constexpr std::uint32_t vec_macro_control_bits(const vec_macro_control& control)
{
    return (control.delay_counts_instructions << 8) | (control.store_takes_load_format << 4) | control.store_format;
}

static_assert(vec_macro_unit_bits({.instruction = vec_macro_programmable(3), .delay = 5, .loaded_is_vb = true, .staging = true}) == 0xEF);
static_assert(vec_macro_control_bits({.store_takes_load_format = 1u << 0, .delay_counts_instructions = vec_macro_simple_unit}) == 0x110);

#ifdef LLK2_SFPU_THREAD

// Operand names of the SFPU instructions (ckernel_instr_params.h), for the bodies written instruction by instruction.
using ckernel::InstrModLoadStore;
using ckernel::p_sfpu;

/// Macro sequence Index (0..3) = Sequence, through L0, and the control word, for the functions that name their
/// instructions through the TTI_SFP* macros (the fast exp does the same through platform::sfpu).
template <std::uint32_t Index, vec_macro_sequence Sequence>
sfpi_inline void vec_set_macro_sequence()
{
    constexpr std::uint32_t bits = vec_macro_sequence_bits(Sequence);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
    TTI_SFPCONFIG(
        0, static_cast<std::uint32_t>(platform::sfpu::ConfigTarget::MacroSequence0) + Index, static_cast<std::uint32_t>(platform::sfpu::ConfigSource::L0));
}

template <vec_macro_control Control>
sfpi_inline void vec_set_macro_control()
{
    TTI_SFPCONFIG(
        vec_macro_control_bits(Control),
        static_cast<std::uint32_t>(platform::sfpu::ConfigTarget::MacroControl),
        static_cast<std::uint32_t>(platform::sfpu::ConfigSource::Immediate));
}

/// A polynomial by Horner's method: eval(x, c0, c1, ..., cn) = c0 + x (c1 + x (... + x cn)). Each coefficient is a
/// float (loaded where it is used), an sfpi::vFloat the caller holds, or a programmable constant register.
struct vec_polynomial
{
    template <typename U>
    sfpi_inline static constexpr auto eval(U x)
    {
        return U {0};
    }

    template <typename U, typename Coefficient0>
    sfpi_inline static constexpr auto eval([[maybe_unused]] U x, Coefficient0 coeff0)
    {
        return coeff0;
    }

    template <typename U, typename Coefficient0, typename... OtherCoefficients>
    sfpi_inline static constexpr auto eval(U x, Coefficient0 coeff0, OtherCoefficients... other_coefficients)
    {
        return coeff0 + x * eval(x, other_coefficients...);
    }
};

/// Hoisted constants. The SFPI compiler does not lift a float literal out of a loop by itself: each fp32 literal a
/// loop body uses costs an SFPLOADI pair per iteration. So a function that evaluates a constant-heavy body per
/// iteration builds its constants once before the loop, as sfpi::vFloats or in the programmable constants, and the
/// one-value helpers (vec_exp_21f, vec_exp_fp32, vec_expm1_clamped, vec_sigmoid_core, ...) take them as arguments.
/// SFPI cannot spill, so each function hoists only what fits beside its live data: one LReg too many is a compile
/// error ("too few lregs"), never a slowdown.
///
/// vec_hoisted_if: a loop constant a function keeps in an LReg when Cond, and a plain float otherwise. Functions
/// whose arms (bf16 / fp32 Dest) hoist different constants declare all of them before the loop with the arm's
/// condition, so the arm that never reads one pays nothing: a float is folded away, an sfpi::vFloat is an SFPLOADI
/// pair per call. Declare these non-const: a const vFloat that lives across a partially unrolled loop makes SFPI try
/// to spill it ("cannot write SFPU object to memory").
template <bool Cond>
using vec_hoisted_if = std::conditional_t<Cond, sfpi::vFloat, float>;

#endif // LLK2_SFPU_THREAD

} // namespace llk2::llk_lib
