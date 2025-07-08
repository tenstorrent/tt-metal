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
    DataFormat unpack_src;
    DataFormat unpack_dst;
    DataFormat pack_src;
    DataFormat pack_dst;
};

constexpr bool is_exponentB(DataFormat format)
{
    return (format == DataFormat::Float16_b || format == DataFormat::Bfp8_b || format == DataFormat::Tf32);
}

constexpr bool is_format_combination_outlier(DataFormat input, DataFormat output, bool is_fp32_dest_acc_en)
{
    return (is_exponentB(input) && output == DataFormat::Float16 && !is_fp32_dest_acc_en);
}

constexpr FormatConfig get_data_formats(DataFormat input, DataFormat output, bool is_fp32_dest_acc_en)
{
    DataFormat unpack_in  = input;
    DataFormat unpack_out = input;
    DataFormat pack_out   = output;
    DataFormat pack_in    = DataFormat::Invalid; // Invalid format as placeholder

    if (input == DataFormat::Float32 && !UNPACKING_TO_DEST)
    {
        if (is_fp32_dest_acc_en)
        {
            unpack_out = DataFormat::Tf32;
        }
        else
        {
            if (is_exponentB(output) || output == DataFormat::Float32)
            {
                unpack_out = DataFormat::Float16_b; // If output Float32 or Float16_b
            }
            else
            {
                unpack_out = DataFormat::Float16; // Tilize to Float16
            }
        }

        if (is_fp32_dest_acc_en || is_exponentB(output))
        {
            pack_in = output;
        }
        else
        {
            pack_in = unpack_out;
        }
    }
    else if (input == DataFormat::Float16 && output == DataFormat::Bfp8_b && !is_fp32_dest_acc_en)
    {
        pack_in = DataFormat::Bfp8;
    }
    else if (is_format_combination_outlier(input, output, is_fp32_dest_acc_en))
    {
        pack_in = (is_wormhole) ? DataFormat::Float32 : output;
    }
    else
    {
        pack_in = is_fp32_dest_acc_en ? output : input;
    }

    if (is_wormhole && is_fp32_dest_acc_en && output == DataFormat::Float16)
    {
        pack_in = DataFormat::Float32; // Gasket in wormhole cannot convert fp32 to fp16, and since dest accumulation turns on for
                                       // outlier cases we have fp32 in dest, so gasket cannot convert it to fp16, packer must do that
    }

    return {unpack_in, unpack_out, pack_in, pack_out};
    // return {unpack_in, unpack_out, pack_in, pack_out};
}
