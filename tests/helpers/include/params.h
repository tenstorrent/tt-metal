// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdarg>
#include <type_traits>

// Include auto-generated build configuration
#include "build.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_binary.h"
#include "ckernel_sfpu_log.h"
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_square.h"
#include "data_format_inference.h"
#include "llk_defs.h"
#include "llk_sfpu_types.h"
#include "perf.h"
#include "tensix_types.h"

inline uint32_t L1_ADDRESS(const volatile void* buffer)
{
    return (reinterpret_cast<uint32_t>(buffer) / 16) - 1;
}

namespace
{
constexpr std::underlying_type_t<DataFormat> get_data_format(DataFormat format)
{
    return static_cast<std::underlying_type_t<DataFormat>>(format);
}
} // namespace

constexpr bool dest_acc_en_input =
#ifdef DEST_ACC
    true;
#else
    false;
#endif

constexpr bool unpack_to_dest = UNPACKING_TO_DEST;

#if DATA_FORMAT_INFERENCE_MODEL
constexpr bool is_fp32_dest_acc_en =
    dest_acc_en_input || is_format_combination_outlier(static_cast<DataFormat>(UNPACK_A_IN), static_cast<DataFormat>(PACK_OUT), dest_acc_en_input);
constexpr FormatConfig pipeline_formats = get_data_formats(static_cast<DataFormat>(UNPACK_A_IN), static_cast<DataFormat>(PACK_OUT), dest_acc_en_input);
constexpr auto UNPACK_A_OUT             = static_cast<uint32_t>(pipeline_formats.unpack_dst);
constexpr auto UNPACK_B_IN              = static_cast<uint32_t>(pipeline_formats.unpack_src);
constexpr auto UNPACK_B_OUT             = static_cast<uint32_t>(pipeline_formats.unpack_dst);
constexpr auto PACK_IN                  = static_cast<uint32_t>(pipeline_formats.pack_src);
constexpr auto MATH_FORMAT              = static_cast<uint32_t>(pipeline_formats.unpack_dst);
#else
constexpr bool is_fp32_dest_acc_en = dest_acc_en_input;
#endif

#ifdef ELTWISE_BINARY_ADD
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWADD;
#endif
#ifdef ELTWISE_BINARY_SUB
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWSUB;
#endif
#ifdef ELTWISE_BINARY_MUL
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;
#endif
// TO BE IMPLEMENTED IN LLKs
#ifdef ELTWISE_BINARY_DIV
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWDIV;
#endif
#ifdef ELTWISE_BINARY_LESS
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWLESS;
#endif

#ifdef SFPU_ELWADD
constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::ADD;
#endif
#ifdef SFPU_ELWSUB
constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::SUB;
#endif
#ifdef SFPU_ELWMUL
constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::MUL;
#endif
#ifdef SFPU_OP_XLOGY
constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::XLOGY;
#endif
#ifdef SFPU_OP_RSHFT
constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::RSHFT;
#endif
#ifdef SFPU_OP_LSHFT
constexpr auto SFPU_BINARY_OPERATION = ckernel::BinaryOp::LSHFT;
#endif

#ifdef SFPU_OP_SQRT
constexpr auto SFPU_OPERATION = SfpuType::sqrt;
#endif
#ifdef SFPU_OP_LOG
constexpr auto SFPU_OPERATION = SfpuType::log;
#endif
#ifdef SFPU_OP_SQUARE
constexpr auto SFPU_OPERATION = SfpuType::square;
#endif
#ifdef SFPU_OP_SINE
constexpr auto SFPU_OPERATION = SfpuType::sine;
#endif
#ifdef SFPU_OP_COSINE
constexpr auto SFPU_OPERATION = SfpuType::cosine;
#endif
#ifdef SFPU_OP_ABS
constexpr auto SFPU_OPERATION = SfpuType::abs;
#endif
#ifdef SFPU_OP_RECIPROCAL
constexpr auto SFPU_OPERATION = SfpuType::reciprocal;
#endif
#ifdef SFPU_OP_CELU
constexpr auto SFPU_OPERATION = SfpuType::celu;
#endif

inline void process_addresses(volatile uint32_t* buffer_Dest[], int n, int first, ...)
{
    buffer_Dest[0] = reinterpret_cast<volatile uint32_t*>(first);

    va_list args;
    va_start(args, first);
    for (int i = 1; i < n; ++i)
    {
        int num        = va_arg(args, int);
        buffer_Dest[i] = reinterpret_cast<volatile uint32_t*>(num);
    }
    va_end(args);
}
