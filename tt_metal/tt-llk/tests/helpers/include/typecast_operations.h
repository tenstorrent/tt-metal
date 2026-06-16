// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_typecast.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

//
// LLK-level dispatch for the SFPU typecast kernels.
//
// This is a faithful, parametrized mirror of the production compute API
// `typecast_tile<IN, OUT>` / `typecast_tile_init<IN, OUT>` found in
//   tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h
// but rewritten to fit the tt-llk test harness:
//
//   * The ambient compute-kernel macros used by the production header
//     (`MATH(...)`, `DST_SYNC_MODE`, `DST_ACCUM_MODE`, `APPROX`) are replaced
//     by explicit template parameters so the test source can drive the SFPU
//     in any sync / accumulation / approximation mode.
//   * It calls the *exact same* `ckernel::sfpu::calculate_typecast_*` /
//     `init_typecast_*` primitives (the real LLK code under test) through the
//     same `SFPU_UNARY_CALL` / `SFPU_UNARY_INIT*` macros production uses, so
//     this exercises the kernels, not a re-implementation of them.
//
// Pairs that the hardware realises purely through unpacker / packer format
// conversion (block-float <-> float, etc.) intentionally have no SFPU call —
// they fall through to the empty `else` exactly like the production header.
//
// Keep this dispatch in lockstep with typecast.h when new pairs are added.
//

namespace test_utils
{
using namespace ckernel;
using namespace ckernel::sfpu;

template <DataFormat IN, DataFormat OUT, bool APPROX_MODE>
void call_unary_typecast_operation_init()
{
    if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt16 && (OUT == DataFormat::UInt32 || OUT == DataFormat::Int32))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_uint32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_uint16, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_uint16, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt16 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint16_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::Int32 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_int32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt32 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (
        (IN == DataFormat::Float32 || IN == DataFormat::Float16_b || IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b) && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_uint16, (APPROX_MODE));
    }
    else if constexpr (
        (IN == DataFormat::Float32 || IN == DataFormat::Float16_b || IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_fp32_to_uint8, (APPROX_MODE));
    }
    else if constexpr ((IN == DataFormat::Int32 || IN == DataFormat::UInt32 || IN == DataFormat::UInt16) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint_to_uint8, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp32, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt8 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_fp16b, (APPROX_MODE));
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_INIT_FN(typecast, sfpu::init_typecast_uint32_to_uint16, (APPROX_MODE));
    }
    else
    {
        // Pairs handled purely by unpacker/packer (no SFPU math) or that need
        // no per-op programming still issue the bare init so the SFPU is in a
        // defined state, exactly like the production `else` branch.
        SFPU_UNARY_INIT(typecast);
    }
}

template <DstSync DST_SYNC_MODE, bool DST_ACCUM_MODE, DataFormat IN, DataFormat OUT, bool APPROX_MODE, int ITERATIONS = 8>
void call_unary_typecast_operation(std::uint32_t dst_index)
{
    if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::Float32)
    {
        // no SFPU kernel needed, handled by packer
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Bfp8_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Bfp8_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float16_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Bfp8_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Int32)
    {
        // Calls same kernel as the UInt32 case.
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::Float16_b)
    {
        // no SFPU kernel needed, handled by unpacker
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::Bfp8_b)
    {
        // no SFPU kernel needed, handled by packer
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::Float32)
    {
        // no SFPU kernel needed, handled by unpacker/packer
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Bfp8_b)
    {
        // no SFPU kernel needed, handled by packer
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt16 && OUT == DataFormat::Bfp4_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint16_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::Int32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_int32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Int32 && OUT == DataFormat::Bfp4_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_int32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::UInt32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt32 && OUT == DataFormat::Bfp4_b)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::Float16_b)
    {
        // no SFPU kernel needed, handled by unpacker
    }
    else if constexpr (IN == DataFormat::Float16_b && OUT == DataFormat::Bfp4_b)
    {
        // no SFPU kernel needed, handled by packer
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::Bfp8_b)
    {
        // no SFPU kernel needed, handled by unpacker
    }
    else if constexpr (IN == DataFormat::Bfp8_b && OUT == DataFormat::Bfp4_b)
    {
        // no SFPU kernel needed, handled by packer
    }
    else if constexpr (IN == DataFormat::Bfp4_b && OUT == DataFormat::Float32)
    {
        // no SFPU kernel needed, handled by unpacker/packer
    }
    else if constexpr (IN == DataFormat::Float32 && OUT == DataFormat::Bfp4_b)
    {
        // no SFPU kernel needed, handled by packer
    }
    else if constexpr (
        (IN == DataFormat::Float32 || IN == DataFormat::Float16_b || IN == DataFormat::Bfp8_b || IN == DataFormat::Bfp4_b) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_fp32_to_uint8, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr ((IN == DataFormat::Int32 || IN == DataFormat::UInt32 || IN == DataFormat::UInt16) && OUT == DataFormat::UInt8)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint_to_uint8, (APPROX_MODE, ITERATIONS, (IN == DataFormat::UInt16)), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::Float32)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp32, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && (OUT == DataFormat::Float16_b || OUT == DataFormat::Bfp8_b || OUT == DataFormat::Bfp4_b))
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_fp16b, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
    else if constexpr (IN == DataFormat::UInt8 && (OUT == DataFormat::Int32 || OUT == DataFormat::UInt32))
    {
        // No SFPU kernel needed.
    }
    else if constexpr (IN == DataFormat::UInt8 && OUT == DataFormat::UInt16)
    {
        SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_typecast_uint32_to_uint16, (APPROX_MODE, ITERATIONS), dst_index, VectorMode::RC);
    }
}

} // namespace test_utils
