// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_binary_sfpu.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Copy one Dest tile onto another, bit-exact, using SFPLOAD/SFPSTORE.
 *
 * The SFPLOAD/SFPSTORE sfpmem mode is the canonical _sfpu_sfpmem_type_<DATA_FORMAT>()
 * mapping (Int32 → INT32 so TEN-4674 stays intact; UInt16/Int16/Int8/UInt8 keep their
 * dedicated modes; formats with no dedicated mode fall back to DEFAULT). Naming the
 * format avoids loading a narrow-integer payload as float, which is what DEFAULT
 * would do — DEFAULT only re-derives FP32/FP16A/FP16B from ALU_FORMAT_SPEC_REG.
 *
 * Dest offsets are in rows, relative to the dest counter
 * @ref _llk_math_eltwise_binary_sfpu_params_ sets. Each iteration covers one SFPU pass
 * (SFP_ROWS = 2); VectorMode::RC walks the four faces around this loop.
 *
 * @tparam DATA_FORMAT: Dest encoding; any DataFormat enumerator. Must match the
 *         format actually sitting in Dest (e.g. Float32 when dest-acc is on).
 * @tparam APPROXIMATION_MODE: unused, kept for ABI parity with the Blackhole kernel.
 * @tparam ITERATIONS: SFPU passes per face (default = SFPU_ITERATIONS).
 * @tparam TILE_SHAPE: Destination tile shape used to derive the tile stride.
 * @param dst_index_in: Dest tile index to copy from.
 * @param dst_index_out: Dest tile index to copy to; may alias @p dst_index_in.
 * @param dst_index_unused: Unused third argument. Same (in, out, unused) order as
 *        Blackhole, so the shared compute API can call this without an arch ifdef.
 * @note Stateless: no matching init is required beyond `_llk_math_eltwise_sfpu_init_()`.
 */
template <
    DataFormat DATA_FORMAT,
    [[maybe_unused]] bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline void copy_dest_value(
    const std::uint32_t dst_index_in,
    const std::uint32_t dst_index_out,
    [[maybe_unused]] const std::uint32_t dst_index_unused) {
    constexpr std::uint32_t SFPMEM_MODE = _sfpu_sfpmem_type_<DATA_FORMAT>();
    constexpr std::uint32_t tile_stride = 1U << trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t in_offset = dst_index_in * tile_stride;
    const std::uint32_t out_offset = dst_index_out * tile_stride;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const std::uint32_t row = static_cast<std::uint32_t>(d) * ckernel::math::SFP_ROWS;
        TT_SFPLOAD(p_sfpu::LREG0, SFPMEM_MODE, ADDR_MOD_7, 0 /* done */, in_offset + row);
        TT_SFPSTORE(p_sfpu::LREG0, SFPMEM_MODE, ADDR_MOD_7, 0 /* done */, out_offset + row);
    }
}

/**
 * @brief Format-agnostic Dest tile copy via sfpi vFloat (deprecated).
 *
 * @tparam APPROXIMATION_MODE: unused, kept for ABI parity with the Blackhole kernel.
 * @tparam ITERATIONS: SFPU passes per face (default = SFPU_ITERATIONS).
 * @tparam TILE_SHAPE: Destination tile shape used to derive the sfpi tile stride.
 * @param dst_index_in: Dest tile index to copy from.
 * @param dst_index_out: Dest tile index to copy to.
 * @param dst_index_unused: Unused third argument; the compute API passes 0 here.
 * @note Prefer the DataFormat-templated overload. Call `_llk_math_eltwise_sfpu_init_()`
 *       first.
 */
template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
[[deprecated("Use copy_dest_value<DataFormat, APPROXIMATION_MODE, ITERATIONS> instead")]]
inline void copy_dest_value(
    const std::uint32_t dst_index_in,
    const std::uint32_t dst_index_out,
    [[maybe_unused]] const std::uint32_t dst_index_unused) {
    // sfpi dst_reg indexes in SFP_ROWS (2) units, so the tile stride is dest-rows / 2
    // (32 for Tile32x32; same as BH/WH 64/SFP_DESTREG_STRIDE).
    constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] =
            sfpi::vFloat(sfpi::dst_reg[dst_index_in * dst_tile_size_sfpi]);
        sfpi::dst_reg++;
    }
}

/** @brief No-op init; Dest-to-Dest copy programs no SFPU constants. */
inline void copy_dest_value_init() {}

// Op class for copying one Dest tile onto another. Same name and leading template parameters as on
// Wormhole/Blackhole. Only run() needs DATA_FORMAT.
template <
    bool APPROXIMATION_MODE,
    DataFormat DATA_FORMAT = DataFormat::Invalid,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct CopyDestValue : SfpuBinaryOp<CopyDestValue<APPROXIMATION_MODE, DATA_FORMAT, ITERATIONS, SLOT>, SLOT> {
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_in, const std::uint32_t dst_index_out, const std::uint32_t dst_index_unused) {
        copy_dest_value<DATA_FORMAT, APPROXIMATION_MODE, ITERATIONS, SLOT>(
            dst_index_in, dst_index_out, dst_index_unused);
    }
    static inline __attribute__((always_inline)) void init_op() { copy_dest_value_init(); }
};

}  // namespace sfpu
}  // namespace ckernel
