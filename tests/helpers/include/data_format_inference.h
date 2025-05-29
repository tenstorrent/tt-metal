// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tensix_types.h"

#if defined(ARCH_WORMHOLE) && defined(ARCH_BLACKHOLE)
#error "Only one of ARCH_WORMHOLE or ARCH_BLACKHOLE can be defined"
#elif defined(ARCH_WORMHOLE)
constexpr bool is_blackhole = false;
constexpr bool is_wormhole  = true;
#elif defined(ARCH_BLACKHOLE)
constexpr bool is_blackhole = true;
constexpr bool is_wormhole  = false;
#else
#error "You must define either ARCH_WORMHOLE or ARCH_BLACKHOLE"
#endif

struct FormatConfig
{
    const uint32_t unpack_src;
    const uint32_t unpack_dst;
    const uint32_t pack_src;
    const uint32_t pack_dst;
};

constexpr bool is_exponentB(uint32_t format)
{
    return (
        format == static_cast<uint32_t>(DataFormat::Float16_b) || format == static_cast<uint32_t>(DataFormat::Bfp8_b) ||
        format == static_cast<uint32_t>(DataFormat::Tf32));
}

constexpr bool is_format_combination_outlier(uint32_t input, uint32_t output, bool is_fp32_dest_acc_en)
{
    return (is_exponentB(input) && output == (uint32_t)DataFormat::Float16 && !is_fp32_dest_acc_en);
}

constexpr FormatConfig get_data_formats(uint32_t input, uint32_t output, bool is_fp32_dest_acc_en)
{
    uint32_t unpack_in  = input;
    uint32_t unpack_out = input;
    uint32_t pack_out   = output;
    uint32_t pack_in    = 0;

    if (input == (uint32_t)DataFormat::Float16 && output == (uint32_t)DataFormat::Bfp8_b && !is_fp32_dest_acc_en)
    {
        pack_in = static_cast<uint32_t>(DataFormat::Bfp8);
    }
    else if (is_wormhole && is_fp32_dest_acc_en && output == (uint32_t)DataFormat::Float16)
    {
        pack_in = static_cast<uint32_t>(DataFormat::Float32); // Gasket in wormhole cannot convert fp32 to fp16, and since dest accumulation turns on for
                                                              // outlier cases we have fp32 in dest, so gasket cannot convert it to fp16, packer must do that
    }
    else if (is_format_combination_outlier(input, output, is_fp32_dest_acc_en))
    {
        pack_in = (is_wormhole) ? (uint32_t)DataFormat::Float32 : output;
    }
    else
    {
        pack_in = is_fp32_dest_acc_en ? output : input;
    }

    return {unpack_in, unpack_out, pack_in, pack_out};
}
