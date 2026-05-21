// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AUTO-GENERATED CONFIGURATION HEADER. DO NOT EDIT MANUALLY!

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include "llk_defs.h"
#include "llk_sfpu_types.h"
#include "operand.h"
#include "perf.h"
#include "tensix_types.h"
#define RUNTIME_PARAMETERS [[maybe_unused]] const struct RuntimeParams&
constexpr bool l1_acc_en      = 0;
constexpr bool unpack_to_dest = false;

// Formats struct
struct FormatConfig
{
    const std::uint32_t unpack_A_src;
    const std::uint32_t unpack_B_src;
    const std::uint32_t unpack_S_src;
    const std::uint32_t unpack_A_dst;
    const std::uint32_t unpack_B_dst;
    const std::uint32_t unpack_S_dst;
    const std::uint32_t math;
    const std::uint32_t pack_src;
    const std::uint32_t pack_dst;
    const std::uint32_t pack_S_src;
    const std::uint32_t pack_S_dst;

    constexpr FormatConfig(
        std::uint32_t unpack_A_src_,
        std::uint32_t unpack_B_src_,
        std::uint32_t unpack_S_src_,
        std::uint32_t unpack_A_dst_,
        std::uint32_t unpack_B_dst_,
        std::uint32_t unpack_S_dst_,
        std::uint32_t math_,
        std::uint32_t pack_src_,
        std::uint32_t pack_dst_,
        std::uint32_t pack_S_src_,
        std::uint32_t pack_S_dst_) :
        unpack_A_src(unpack_A_src_),
        unpack_B_src(unpack_B_src_),
        unpack_S_src(unpack_S_src_),
        unpack_A_dst(unpack_A_dst_),
        unpack_B_dst(unpack_B_dst_),
        unpack_S_dst(unpack_S_dst_),
        math(math_),
        pack_src(pack_src_),
        pack_dst(pack_dst_),
        pack_S_src(pack_S_src_),
        pack_S_dst(pack_S_dst_)
    {
    }
};

constexpr bool is_fp32_dest_acc_en         = true;
constexpr std::uint32_t TILE_SIZE_PACK     = 68;
constexpr std::uint32_t TILE_SIZE_UNPACK_A = 68;
constexpr std::uint32_t TILE_SIZE_UNPACK_B = 68;
constexpr Operand buffer_A(0x21000, 1088);
constexpr Operand buffer_B(0x25400, 1088);
constexpr Operand buffer_Res(0x29800, 1088);
constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::HiFi2;

// Math operation configuration
constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;

constexpr auto PERF_RUN_TYPE     = PerfRunType::PACK_ISOLATE;
constexpr std::uint32_t TILE_CNT = 16;
// Data formats inferred by Python inference model
// Format data for single L1-to-L1 iteration
constexpr auto UNPACK_A_IN  = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto UNPACK_B_IN  = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto UNPACK_S_IN  = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto UNPACK_A_OUT = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto UNPACK_B_OUT = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto UNPACK_S_OUT = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto MATH_FORMAT  = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto PACK_IN      = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto PACK_OUT     = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto PACK_S_IN    = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr auto PACK_S_OUT   = ckernel::to_underlying(DataFormat::Bfp8_b);
constexpr FormatConfig formats =
    FormatConfig(UNPACK_A_IN, UNPACK_B_IN, UNPACK_S_IN, UNPACK_A_OUT, UNPACK_B_OUT, UNPACK_S_OUT, MATH_FORMAT, PACK_IN, PACK_OUT, PACK_S_IN, PACK_S_OUT);

struct RuntimeParams
{
};
