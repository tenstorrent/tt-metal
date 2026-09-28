// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-harness mirror of tt_metal/hw/inc/api/compute/quantization.h.
//
// The quant family (quant / requant / dequant, int32 / uint8 / int8 outputs, int32 / int8 inputs)
// has no ckernel::BinaryOp and is reached in production only through the eleven `*_tile_init` /
// `*_tile` wrappers of quantization.h. This header re-creates those wrappers one to one, keyed by
// a QuantVariant that the Python harness selects through the SFPU_QUANT_VARIANT template
// parameter (helpers/test_variant_parameters.py::SFPU_QUANT_VARIANT). Every arm below names the
// same {quant,requant,dequant}_init and calculate_* instantiation as the production wrapper of
// the same name, so the tests and the perf sweep measure the shipped kernels, not a proxy.
//
// Blackhole only: the metal quant kernels live in the per-architecture llk_sfpu directory.

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_sfpu/ckernel_sfpu_quant.h"
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"

namespace test_utils
{

enum class QuantVariant : std::uint8_t
{
    QUANT,                     // quant_tile_init             / quant_tile
    QUANT_UINT8,               // quant_uint8_tile_init       / quant_tile
    QUANT_INT8,                // quant_int8_tile_init        / quant_int8_tile
    REQUANT,                   // requant_tile_init           / requant_tile
    REQUANT_UINT8,             // requant_uint8_tile_init     / requant_tile
    REQUANT_INT8,              // requant_int8_tile_init      / requant_int8_tile
    REQUANT_INT8_IN,           // requant_int8_in_tile_init   / requant_int8_in_tile
    REQUANT_INT8_IN_UINT8_OUT, // requant_int8_in_uint8_out_tile_init / requant_int8_in_tile
    REQUANT_INT8_IN_INT8_OUT,  // requant_int8_in_int8_out_tile_init  / requant_int8_in_int8_out_tile
    DEQUANT,                   // dequant_tile_init           / dequant_tile
    DEQUANT_INT8,              // dequant_int8_tile_init      / dequant_int8_tile
};

/**
 * @brief One-time init for a quant-family kernel; mirrors the `*_tile_init(zero_point)` wrappers.
 *
 * @param zero_point fp32 bit pattern of the zero-point. Like the production wrappers, DEQUANT expects the bits
 *        of -zero_point (its body computes (A + LREG2) * B).
 */
template <QuantVariant V, bool APPROXIMATION_MODE>
inline void quant_variant_init(const std::uint32_t zero_point)
{
    using namespace ckernel;
    if constexpr (V == QuantVariant::QUANT)
    {
        SFPU_BINARY_INIT_FN_ARGS(quant_int32, sfpu::quant_init, (APPROXIMATION_MODE), zero_point);
    }
    else if constexpr (V == QuantVariant::QUANT_UINT8)
    {
        SFPU_BINARY_INIT_FN_ARGS(quant_int32, sfpu::quant_init, (APPROXIMATION_MODE, false, DataFormat::UInt8), zero_point);
    }
    else if constexpr (V == QuantVariant::QUANT_INT8)
    {
        SFPU_BINARY_INIT_FN_ARGS(quant_int32, sfpu::quant_init, (APPROXIMATION_MODE, false, DataFormat::Int8), zero_point);
    }
    else if constexpr (V == QuantVariant::REQUANT)
    {
        SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROXIMATION_MODE), zero_point);
    }
    else if constexpr (V == QuantVariant::REQUANT_UINT8)
    {
        SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROXIMATION_MODE, false, DataFormat::UInt8), zero_point);
    }
    else if constexpr (V == QuantVariant::REQUANT_INT8)
    {
        SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROXIMATION_MODE, false, DataFormat::Int8), zero_point);
    }
    else if constexpr (V == QuantVariant::REQUANT_INT8_IN)
    {
        SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROXIMATION_MODE, false, DataFormat::Int32, true), zero_point);
    }
    else if constexpr (V == QuantVariant::REQUANT_INT8_IN_UINT8_OUT)
    {
        SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROXIMATION_MODE, false, DataFormat::UInt8, true), zero_point);
    }
    else if constexpr (V == QuantVariant::REQUANT_INT8_IN_INT8_OUT)
    {
        SFPU_BINARY_INIT_FN_ARGS(requant_int32, sfpu::requant_init, (APPROXIMATION_MODE, false, DataFormat::Int8, true), zero_point);
    }
    else if constexpr (V == QuantVariant::DEQUANT)
    {
        SFPU_BINARY_INIT_FN_ARGS(dequant_int32, sfpu::dequant_init, (APPROXIMATION_MODE), zero_point);
    }
    else
    {
        static_assert(V == QuantVariant::DEQUANT_INT8, "quant_variant_init: unhandled QuantVariant");
        SFPU_BINARY_INIT_FN_ARGS(dequant_int32, sfpu::dequant_init, (APPROXIMATION_MODE, false, true), zero_point);
    }
}

/**
 * @brief Run a quant-family kernel on one tile pair; mirrors the `*_tile(idst0, idst1, odst)` wrappers.
 */
template <QuantVariant V, bool APPROXIMATION_MODE, ckernel::DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE>
inline void quant_variant_call(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    using namespace ckernel;
    constexpr VectorMode vector_mode = VectorMode::RC;
    if constexpr (V == QuantVariant::QUANT || V == QuantVariant::QUANT_UINT8)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_quant_int32, (APPROXIMATION_MODE), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (V == QuantVariant::QUANT_INT8)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_quant_int32_int8_pack, (APPROXIMATION_MODE), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (V == QuantVariant::REQUANT || V == QuantVariant::REQUANT_UINT8)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_requant_int32, (APPROXIMATION_MODE), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (V == QuantVariant::REQUANT_INT8)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_requant_int32_int8_pack, (APPROXIMATION_MODE), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (V == QuantVariant::REQUANT_INT8_IN || V == QuantVariant::REQUANT_INT8_IN_UINT8_OUT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_requant_int32, (APPROXIMATION_MODE, 8, false, true), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else if constexpr (V == QuantVariant::REQUANT_INT8_IN_INT8_OUT)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE,
            DST_ACCUM_MODE,
            calculate_requant_int32_int8_pack,
            (APPROXIMATION_MODE, 8, true),
            dst_index_in0,
            dst_index_in1,
            dst_index_out,
            vector_mode);
    }
    else if constexpr (V == QuantVariant::DEQUANT)
    {
        SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_dequant_int32, (APPROXIMATION_MODE), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
    else
    {
        static_assert(V == QuantVariant::DEQUANT_INT8, "quant_variant_call: unhandled QuantVariant");
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_dequant_int32, (APPROXIMATION_MODE, 8, false, true), dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
}

} // namespace test_utils
