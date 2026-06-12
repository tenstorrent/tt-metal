// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {
// fp16b bit patterns (upper 16 bits of the corresponding fp32), loaded as
// SFPLOADI immediates in MOD0_FLOATB mode.
constexpr std::uint32_t FP16B_ONE = 0x3F80;   // 1.0f
constexpr std::uint32_t FP16B_ZERO = 0x0000;  // 0.0f

// imm12_math[11]: makes SFPSETCC read the source as FP32/SMAG32 (sign-magnitude) rather than
// two's-complement INT32. SFPLOAD only ever produces FP32, SMAG32, or UINT32 in the LREG (signed
// integers load as sign-magnitude SMAG32, never 2's-complement), so this bit is set for every
// format we handle — both float and integer. The sign tests (SFPSETCC modes 0/4) read the sign
// bit regardless of this bit; it only governs the zero test, where SMAG32==0 correctly treats
// sign-magnitude ±0 as zero (2's-complement would miss -0 = 0x80000000).
constexpr std::uint32_t SFPSETCC_IMM_FP32 = 0x800;

/**
 * @brief Whether FMT is read/written as an integer (vs float) — drives the 1/0 result encoding.
 *
 * Also gates the sfpmem mode: integers take their explicit width from the canonical
 * @ref _sfpu_sfpmem_type_<FMT>() selector (Int32→INT32, Int16→INT16, Int8→INT8, UInt8→UINT8,
 * UInt16→UINT16), while floats use sfpmem::DEFAULT (the explicit fp16 modes decode the fp16b
 * boolean result as NaN).
 *
 * @note Int8/UInt8 use their native Quasar dest format (SMAG8 / UINT8) — these are real
 *       register-file formats, so the 8-bit datapath round-trips natively. UInt16 is the exception:
 *       it is not a Quasar register-file format at all (absent from the unpacker / SrcA-B / dest /
 *       packer encodings — see VALID_QUASAR_SRC/DEST_REG_FORMATS; Int8/UInt8 are present, hence
 *       native). It is therefore routed through the Int16/SMAG16 container — unpack and pack run in
 *       Int16 (the known-good 16-bit bit-passthrough path) and only the SFPU accesses the
 *       unsigned-16 semantics via sfpmem::UINT16. The caller must keep the unpack/pack/math formats
 *       at Int16 and select FMT=UInt16 only to pick that sfpmem mode.
 *
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32 / Int16 / Int8 signed, UInt16 / UInt8 unsigned.
 */
template <DataFormat FMT>
inline constexpr bool _zero_comp_is_int_() {
    return FMT == DataFormat::Int32 || FMT == DataFormat::Int16 || FMT == DataFormat::Int8 ||
           FMT == DataFormat::UInt16 || FMT == DataFormat::UInt8;
}

/**
 * @brief Number of instructions in the recorded replay body for a comparison mode.
 *
 * Format-independent (Int32 and float bodies have identical instruction counts): eqz/nez predicate
 * with a single SFPSETCC (7). The strict ltz/gtz AND a second SFPSETCC (8). gtez/ltez are the
 * lane-wise complements of ltz/gtz — they predicate the same strictly-signed lanes but default the
 * result to 1 and write 0 — so they share the strict body's length (8).
 *
 * @tparam COMP_MODE: Comparison-to-zero mode selecting the body to record.
 * @return Recorded body length in instructions.
 */
template <SfpuType COMP_MODE>
inline constexpr std::uint32_t _zero_comp_replay_len_() {
    if constexpr (
        COMP_MODE == SfpuType::less_than_zero || COMP_MODE == SfpuType::greater_than_zero ||
        COMP_MODE == SfpuType::greater_than_equal_zero || COMP_MODE == SfpuType::less_than_equal_zero) {
        return 8;
    }
    return 7;
}

/**
 * @brief Program ADDR_MOD_6 (dest.incr=2) so the replayed SFPSTORE advances the dest counter.
 *
 * Quasar's shared SFPU init only programs ADDR_MOD_7 (incr=0); this is additive and leaves
 * ADDR_MOD_7 in place for the body's SFPLOAD.
 *
 * @note Call once after @ref _llk_math_eltwise_sfpu_init_ and before @ref _calculate_zero_comp_.
 */
inline void _init_zero_comp_() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
}

/**
 * @brief Predicate (SFPSETCC) the lanes of LREG0 satisfying SETCC_MOD1.
 *
 * Reads LREG0 as FP32/SMAG32 (see @ref SFPSETCC_IMM_FP32) — correct for both float and the
 * sign-magnitude integers SFPLOAD produces.
 *
 * @tparam SETCC_MOD1: SFPSETCC mod1 predicate, e.g. sfpi::SFPSETCC_MOD1_LREG_EQ0.
 */
template <std::uint32_t SETCC_MOD1>
inline __attribute__((always_inline)) void _zero_comp_setcc_() {
    TTI_SFPSETCC(SFPSETCC_IMM_FP32, p_sfpu::LREG0, SETCC_MOD1);
}

/**
 * @brief Load the boolean result (0 or 1) into the result register LREG1 in FMT's encoding.
 *
 * Integer formats use SFPLOADI SHORT (INT16), which sign-extends the immediate into the full LREG
 * (1 -> 0x0000_0001) for a clean result at any store width. SHORT is used rather than USHORT
 * (UINT16) because USHORT left-shifts the immediate by 10 inside the LREG.
 *
 * @tparam FMT: SFPU DataFormat (sfpu_math); integer → integer 0/1, float → fp16b 0.0/1.0.
 * @tparam VALUE: false → 0, true → 1.
 */
template <DataFormat FMT, bool VALUE>
inline __attribute__((always_inline)) void _zero_comp_loadi_bool_() {
    if constexpr (_zero_comp_is_int_<FMT>()) {
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, VALUE ? 1 : 0);
    } else {
        TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, VALUE ? FP16B_ONE : FP16B_ZERO);
    }
}

/**
 * @brief Result value every lane is defaulted to before COMP_MODE's predication runs.
 *
 * eqz/nez/ltz/gtz default to 0 and write 1 into the matching lanes. gtez/ltez invert this: they
 * default to 1 and write 0 into the strictly-signed lanes (the complement of ltz/gtz).
 *
 * @tparam COMP_MODE: Comparison-to-zero mode.
 * @return true (lane default 1) for gtez/ltez, false (lane default 0) otherwise.
 */
template <SfpuType COMP_MODE>
inline constexpr bool _zero_comp_default_() {
    return COMP_MODE == SfpuType::greater_than_equal_zero || COMP_MODE == SfpuType::less_than_equal_zero;
}

/**
 * @brief Predicate the lanes satisfying COMP_MODE and write the result into them.
 *
 * Successive SFPSETCC calls AND-combine and SFPLOADI writes only the currently predicated lanes;
 * FMT selects the 1/0 encoding (via @ref _zero_comp_loadi_bool_), not the SFPSETCC sequence.
 * eqz/nez use a single test and write 1. The strict ltz/gtz AND a sign test with NE0 (excluding
 * ±0) and write 1. gtez/ltez are the lane-wise complements of ltz/gtz: the body defaults every
 * lane to 1 (see @ref _zero_comp_default_), so they predicate the same strictly-signed lanes and
 * write 0, leaving every other lane at 1. This folds the opposite-signed zero into the true set
 * for free — the NE0 test already excludes ±0 from the strict set, so the complement includes it.
 *
 * @tparam FMT: SFPU DataFormat (sfpu_math).
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 */
template <DataFormat FMT, SfpuType COMP_MODE>
struct zero_comp_fill;

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_EQ0>();  // == 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::not_equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();  // != 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::less_than_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_LT0>();  // negative (sign set, incl -0)
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();  // AND nonzero -> strictly < 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::greater_than_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_GTE0>();  // positive (sign clear, incl +0)
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();   // AND nonzero -> strictly > 0
        _zero_comp_loadi_bool_<FMT, true>();
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::greater_than_equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_LT0>();  // negative (sign set, incl -0)
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();  // AND nonzero -> strictly < 0
        _zero_comp_loadi_bool_<FMT, false>();               // write 0 there; the >= 0 lanes keep the default 1
    }
};

template <DataFormat FMT>
struct zero_comp_fill<FMT, SfpuType::less_than_equal_zero> {
    static inline __attribute__((always_inline)) void apply() {
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_GTE0>();  // positive (sign clear, incl +0)
        _zero_comp_setcc_<sfpi::SFPSETCC_MOD1_LREG_NE0>();   // AND nonzero -> strictly > 0
        _zero_comp_loadi_bool_<FMT, false>();                // write 0 there; the <= 0 lanes keep the default 1
    }
};

/**
 * @brief Compute the comparison-to-zero boolean (1/0) for one SFP-row pair.
 *
 * Load x, default every result lane to COMP_MODE's wide value (0, or 1 for the gtez/ltez
 * complement — see @ref _zero_comp_default_), predicate the lanes that satisfy COMP_MODE and write
 * the opposite value into them (see @ref zero_comp_fill), then store unconditionally. FMT selects
 * the sfpmem mode, the SFPSETCC interpretation, and the 1/0 encoding. The SFPSTORE uses ADDR_MOD_6
 * (dest.incr=2) so each replay advances the dest counter by one SFP-row pair while the load/store
 * offsets stay constant, letting the recorded instructions re-issue unchanged across iterations.
 *
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32, Int16, Int8, UInt16, UInt8, or a float format.
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 */
template <DataFormat FMT, SfpuType COMP_MODE>
inline __attribute__((always_inline)) void _zero_comp_body_() {
    // Integers take their explicit width from the canonical selector (Int32→INT32, Int16→INT16,
    // UInt16→UINT16). Floats must use DEFAULT: the comp result is written as an fp16b 1.0 bit
    // pattern, and the explicit fp16 sfpmem modes (FP16A/FP16B) decode that store back as NaN —
    // only the implied/DEFAULT format round-trips it correctly.
    constexpr std::uint32_t sfpmem = _zero_comp_is_int_<FMT>() ? _sfpu_sfpmem_type_<FMT>() : p_sfpu::sfpmem::DEFAULT;

    TTI_SFPLOAD(p_sfpu::LREG0, sfpmem, ADDR_MOD_7, 0 /* done */, 0 /* dest_reg */);  // load x from dest
    _zero_comp_loadi_bool_<FMT, _zero_comp_default_<COMP_MODE>()>();  // result lanes default to COMP_MODE's wide value
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, sfpi::SFPENCC_MOD1_EI_RI);  // enable CC + result=1: all lanes active

    zero_comp_fill<FMT, COMP_MODE>::apply();

    TTI_SFPENCC(sfpi::SFPENCC_IMM12_NEITHER, sfpi::SFPENCC_MOD1_EI_RI);  // disable CC, all lanes unconditional
    TTI_SFPSTORE(p_sfpu::LREG1, sfpmem, ADDR_MOD_6, 0 /* done */, 0 /* dest_reg */);  // store result; dest += 2 rows
}

/**
 * @brief Element-wise comparison-to-zero over a tile, written as 1/0 booleans.
 *
 * Records the per-row-pair body once into replay slots 0..len-1 (NoExec: the record pass does
 * not run the SFPU), then replays it ITERATIONS times. ADDR_MOD_6's dest.incr=2 on the SFPSTORE
 * advances the dest counter per replay, so the loop processes one SFP-row pair per iteration.
 *
 * @tparam APPROXIMATION_MODE: Unused (no approx path); retained for dispatcher signature symmetry.
 * @tparam FMT: SFPU DataFormat (sfpu_math): Int32, Int16, Int8, UInt16, UInt8, or a float format.
 * @tparam COMP_MODE: Comparison-to-zero mode, values =
 *         <equal_zero/not_equal_zero/less_than_zero/greater_than_zero/greater_than_equal_zero/less_than_equal_zero>
 * @tparam ITERATIONS: Number of SFP-row pairs to process (8 for a 32×16 face).
 * @note Requires @ref _init_zero_comp_ to have programmed ADDR_MOD_6 (dest.incr=2).
 */
template <bool APPROXIMATION_MODE, DataFormat FMT, SfpuType COMP_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_zero_comp_() {
    constexpr std::uint32_t replay_len = _zero_comp_replay_len_<COMP_MODE>();

    // Record the per-row-pair body once into replay slots 0..replay_len-1 (NoExec: the
    // record pass does not run the SFPU); each replay re-issues it while ADDR_MOD_6
    // advances the dest counter, so the loop processes one SFP-row pair per iteration.
    lltt::record(0, replay_len);
    _zero_comp_body_<FMT, COMP_MODE>();

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(0, replay_len);
    }
}

}  // namespace sfpu
}  // namespace ckernel
