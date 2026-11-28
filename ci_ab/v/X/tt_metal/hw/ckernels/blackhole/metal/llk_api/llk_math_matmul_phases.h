// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_size_mode.h"

// Fidelity phases carried at run time. In the high-fidelity matmul forms the requested fidelity sets only the MOP's inner loop
// (one pass per phase); the dropped phases of narrow operands add exactly 0 (matmul_phase_fidelity). Each math init whose cbs
// fold at compile time programs the count its operand formats need; any other init keeps the requested count. Calls run the MOP
// the last init left, so an init and the calls after it never disagree.
namespace ckernel {

// A kernel whose high-fidelity matmul MOPs may drop phases: HiFi3 or HiFi4 with a Bfp8_b or Bfp4_b CB, not bound by the kernel
// config buffer
constexpr bool matmul_kernel_phase_elidable() {
#if defined(UCK_CHLKC_MATH)
    if (MATH_FIDELITY != MathFidelity::HiFi3 && MATH_FIDELITY != MathFidelity::HiFi4) {
        return false;
    }
    if (LLK_SIZE_MODE) {
        return false;
    }
    for (const std::uint8_t f : unpack_src_format) {
        if (f == static_cast<std::uint8_t>(DataFormat::Bfp8_b) || f == static_cast<std::uint8_t>(DataFormat::Bfp4_b)) {
            return true;
        }
    }
#endif
    return false;
}

// The phases of a high-fidelity MOP that can change the result for in0 (SrcB) and in1 (SrcA): phase 1 needs the SrcA lo slice,
// 2 the SrcB lo slice, 3 both (tt_fpu_tile.sv); a dropped suffix only, and never below 2 (the form stays high fidelity)
constexpr std::uint32_t matmul_lo_phases(const std::uint32_t requested, const bool lo_a_zero, const bool lo_b_zero) {
    const std::uint32_t needed = (requested >= 4 && !lo_a_zero && !lo_b_zero) ? 4 : (requested >= 3 && !lo_b_zero) ? 3 : 2;
    return needed < requested ? needed : requested;
}

// A block-float in0 with a 16- or 32-bit in1 drops the SrcB lo phases only on blocks of more than one tile: minimal_matmul's
// 1x1 blocks of that pair read 0.3-0.8 % slower with them dropped (the unpack binds and the shorter math only polls)
constexpr bool matmul_phases_one_tile_keep(const bool lo_b_zero, const bool in1_bfp) { return lo_b_zero && !in1_bfp; }

constexpr std::uint32_t matmul_needed_phases(const std::uint32_t requested, const DataFormat in0, const DataFormat in1) {
    if (in0 == DataFormat::Invalid || in1 == DataFormat::Invalid) {
        return requested;
    }
    return matmul_lo_phases(
        requested, in1 == DataFormat::Bfp4_b, in0 == DataFormat::Bfp8_b || in0 == DataFormat::Bfp4_b);
}

// the CBs whose format zeroes the SrcA lo slice as in1 (Bfp4_b only) or is block float (the SrcB lo slice as in0)
constexpr std::uint64_t matmul_lo_zero_cbs(const bool src_a) {
    std::uint64_t cbs = 0;
#if defined(UCK_CHLKC_MATH)
    static_assert(sizeof(unpack_src_format) <= 64);
    for (std::uint32_t cb = 0; cb < sizeof(unpack_src_format); cb++) {
        const auto f = static_cast<DataFormat>(unpack_src_format[cb]);
        if (f == DataFormat::Bfp4_b || (!src_a && f == DataFormat::Bfp8_b)) {
            cbs |= std::uint64_t{1} << cb;
        }
    }
#endif
    return cbs;
}

// the math MOP's inner loop (ckernel_template::program: mop_cfg[1])
inline void matmul_set_mop_phases(const std::uint32_t phases) {
    volatile std::uint32_t* mop_cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
    mop_cfg[1] = phases;
}

// after an init of the requested count (an unkeyed init, a throttle switch back): the count of its cbs where they fold
// at compile time; others keep the requested count (sdpa's matmul_blocks)
inline __attribute__((always_inline)) void matmul_cb_phases(
    const std::uint32_t requested,
    const std::uint32_t in0_cb,
    const std::uint32_t in1_cb,
    const std::uint32_t ct_dim,
    const std::uint32_t rt_dim) {
    if (__builtin_constant_p(in0_cb) && __builtin_constant_p(in1_cb)) {
        constexpr std::uint64_t lo_a_cbs = matmul_lo_zero_cbs(true);
        constexpr std::uint64_t bfp_cbs = matmul_lo_zero_cbs(false);
        const bool lo_b_zero = (bfp_cbs >> in0_cb) & 1;
        const bool keep =
            matmul_phases_one_tile_keep(lo_b_zero, (bfp_cbs >> in1_cb) & 1) && ct_dim == 1 && rt_dim == 1;
        const std::uint32_t p = keep ? requested : matmul_lo_phases(requested, (lo_a_cbs >> in1_cb) & 1, lo_b_zero);
        if (p != requested) {
            matmul_set_mop_phases(p);
        }
    }
}

}  // namespace ckernel
