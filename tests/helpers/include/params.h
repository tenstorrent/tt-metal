// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef PARAMS_H
#define PARAMS_H

#include <cstdarg>
#include <cstdint>

#define L1_ADDRESS(buffer) ((reinterpret_cast<uint32_t>(buffer) / 16) - 1)

#ifdef LLK_TRISC_UNPACK

#ifdef FORMAT_FLOAT16_B
#define DATA_FORMAT (uint32_t)DataFormat::Float16_b
#endif
#ifdef FORMAT_FLOAT16
#define DATA_FORMAT (uint32_t)DataFormat::Float16
#endif
#ifdef FORMAT_FLOAT32
#define DATA_FORMAT (uint32_t)DataFormat::Float32
#endif
#ifdef FORMAT_INT32
#define DATA_FORMAT (uint32_t)DataFormat::Int32
#endif
#ifdef FORMAT_BFP8_B
#define DATA_FORMAT (uint32_t)DataFormat::Bfp8_b
#endif

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_FLOAT16_B
#define DATA_FORMAT (uint32_t)DataFormat::Float16_b
#endif
#ifdef FORMAT_FLOAT16
#define DATA_FORMAT (uint32_t)DataFormat::Float16
#endif
#ifdef FORMAT_FLOAT32
#define DATA_FORMAT (uint32_t)DataFormat::Float32
#endif
#ifdef FORMAT_INT32
#define DATA_FORMAT (uint32_t)DataFormat::Int32
#endif
#ifdef FORMAT_BFP8_B
#define DATA_FORMAT (uint32_t)DataFormat::Bfp8_b
#endif

#ifdef ELTWISE_BINARY_ADD
#define ELTWISE_BINARY_OP EltwiseBinaryType::ELWADD
#endif
#ifdef ELTWISE_BINARY_SUB
#define ELTWISE_BINARY_OP EltwiseBinaryType::ELWSUB
#endif
#ifdef ELTWISE_BINARY_MUL
#define ELTWISE_BINARY_OP EltwiseBinaryType::ELWMUL
#endif
// TO BE IMPLEMENTED IN LLKs
#ifdef ELTWISE_BINARY_DIV
#define ELTWISE_BINARY_OP EltwiseBinaryType::ELWDIV
#endif
#ifdef ELTWISE_BINARY_LESS
#define ELTWISE_BINARY_OP EltwiseBinaryType::ELWLESS
#endif

// SFPU operation macros

#ifdef SFPU_OP_SQRT
#define SFPU_OPERATION SfpuType::sqrt
#define SFPU_CALLS              \
    _init_sqrt_<APPROX_MODE>(); \
    _calculate_sqrt_<APPROX_MODE, 0, 10>(10);
#endif
#ifdef SFPU_OP_LOG
#define SFPU_OPERATION SfpuType::log
#define SFPU_CALLS             \
    _init_log_<APPROX_MODE>(); \
    _calculate_log_<APPROX_MODE, false, 10>(10, 0);
#endif
#ifdef SFPU_OP_SQUARE
#define SFPU_OPERATION SfpuType::square
#define SFPU_CALLS     _calculate_square_<APPROX_MODE, 10>(10);
#endif

#endif

#ifdef LLK_TRISC_PACK

inline void process_addresses(volatile uint32_t* buffer_Dest[], int n, int first, ...)
{
    buffer_Dest[0] = (volatile uint32_t*)first;

    va_list args;
    va_start(args, first);
    for (int i = 1; i < n; ++i)
    {
        int num        = va_arg(args, int);
        buffer_Dest[i] = (volatile uint32_t*)num;
    }
    va_end(args);
}

#ifdef FORMAT_FLOAT16_B
#define DATA_FORMAT (uint32_t)DataFormat::Float16_b
#endif
#ifdef FORMAT_FLOAT16
#define DATA_FORMAT (uint32_t)DataFormat::Float16
#endif
#ifdef FORMAT_FLOAT32
#define DATA_FORMAT (uint32_t)DataFormat::Float32
#endif
#ifdef FORMAT_INT32
#define DATA_FORMAT (uint32_t)DataFormat::Int32
#endif
#ifdef FORMAT_BFP8_B
#define DATA_FORMAT (uint32_t)DataFormat::Bfp8_b
#endif

#endif

#endif
