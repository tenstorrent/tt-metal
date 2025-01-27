// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data_format.hpp"

#include <iostream>       // for basic_ostream
#include <map>            // for operator!=
#include <set>            // for set
#include <string>         // for char_traits
#include <unordered_map>  // for unordered_map

#include "fmt/base.h"                      // for format_string
#include <assert.hpp>      // for tt_throw, TT_FATAL
#include <base_types.hpp>  // for UnpackToDestMode
#include <circular_buffer_constants.h>

namespace tt {

static const std::set<DataFormat> ALL_VALID_FORMATS = {
    DataFormat::Bfp8,      DataFormat::Bfp8_b,   DataFormat::Bfp4,      DataFormat::Bfp4_b,  DataFormat::Bfp2,
    DataFormat::Bfp2_b,    DataFormat::Float16,  DataFormat::Float16_b, DataFormat::Float32, DataFormat::RawUInt32,
    DataFormat::RawUInt16, DataFormat::RawUInt8, DataFormat::Tf32,      DataFormat::Lf8,     DataFormat::Fp8_e4m3,
    DataFormat::Int8,      DataFormat::Int32,    DataFormat::UInt8,     DataFormat::UInt32,  DataFormat::UInt16,
};

static const std::unordered_map<DataFormat, DataFormat> CONVERT_EXP_WIDTH = {
    {DataFormat::Bfp8, DataFormat::Bfp8_b},
    {DataFormat::Bfp8_b, DataFormat::Bfp8},
    {DataFormat::Bfp4, DataFormat::Bfp4_b},
    {DataFormat::Bfp4_b, DataFormat::Bfp4},
    {DataFormat::Bfp2, DataFormat::Bfp2_b},
    {DataFormat::Bfp2_b, DataFormat::Bfp2},
    {DataFormat::Float16, DataFormat::Float16_b},
    {DataFormat::Float16_b, DataFormat::Float16},
};

bool is_bfp_format(DataFormat data_format) {
    return (
        (data_format == DataFormat::Bfp8_b) || (data_format == DataFormat::Bfp8) ||
        (data_format == DataFormat::Bfp4_b) || (data_format == DataFormat::Bfp4) ||
        (data_format == DataFormat::Bfp2_b) || (data_format == DataFormat::Bfp2));
}

bool is_exp_b_format(DataFormat data_format) {
    return (
        (data_format == DataFormat::Tf32 || data_format == DataFormat::Float16_b) ||
        (data_format == DataFormat::Bfp8_b) || (data_format == DataFormat::Bfp4_b) ||
        (data_format == DataFormat::Bfp2_b));
}

ExpPrecision get_exp_precison(DataFormat data_format) {
    return (is_exp_b_format(data_format) ? ExpPrecision::B : ExpPrecision::A);
}

void dump_data_formats(DataFormat data_format[NUM_CIRCULAR_BUFFERS]) {
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        std::cout << "Operand idx " << i << ": " << data_format[i] << "," << std::endl;
    }
}

DataFormat check_consistent_format_across_buffers(DataFormat data_format[NUM_CIRCULAR_BUFFERS]) {
    DataFormat last_valid_format = DataFormat::Invalid;
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        // Special case where Float32 can pair with any exponent precision, skip checking
        if ((data_format[i] == DataFormat::Float32) || (data_format[i] == DataFormat::RawUInt32) ||
            (data_format[i] == DataFormat::UInt32) || (data_format[i] == DataFormat::RawUInt16) ||
            (data_format[i] == DataFormat::RawUInt8) || (data_format[i] == DataFormat::UInt16) ||
            (data_format[i] == DataFormat::UInt8) || (data_format[i] == DataFormat::Int32)) {
            continue;
        }

        if (data_format[i] != DataFormat::Invalid) {
            TT_FATAL(
                ALL_VALID_FORMATS.find(data_format[i]) != ALL_VALID_FORMATS.end(),
                "Format = {} not supported",
                data_format[i]);

            if (last_valid_format != DataFormat::Invalid) {
                TT_FATAL(
                    is_exp_b_format(data_format[i]) == is_exp_b_format(last_valid_format),
                    "All input data-formats must have the same exponent format.");
                // dump_data_formats(data_format);
                last_valid_format = data_format[i];

            } else {
                last_valid_format = data_format[i];
            }
        }
    }
    return last_valid_format;
}

DataFormat check_valid_formats_in_out_data_formats(DataFormat data_format[NUM_CIRCULAR_BUFFERS]) {
    DataFormat last_valid_format = DataFormat::Invalid;
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        if (data_format[i] != DataFormat::Invalid) {
            TT_FATAL(
                ALL_VALID_FORMATS.find(data_format[i]) != ALL_VALID_FORMATS.end(),
                "Format = {} not supported",
                data_format[i]);
            last_valid_format = data_format[i];
        }
    }
    return last_valid_format;
}

ExpPrecision get_data_exp_precision(DataFormat data_formats[NUM_CIRCULAR_BUFFERS]) {
    DataFormat last_valid_format = check_consistent_format_across_buffers(data_formats);
    return get_exp_precison(last_valid_format);
}

std::vector<DataFormat> get_unpack_src_formats(DataFormat data_formats[NUM_CIRCULAR_BUFFERS]) {
    std::vector<DataFormat> unpack_src_format;
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        DataFormat src_format = data_formats[i];
        if (src_format == DataFormat::RawUInt32 || src_format == DataFormat::RawUInt16 ||
            src_format == DataFormat::RawUInt8) {
            switch (src_format) {
                case DataFormat::RawUInt32: src_format = DataFormat::Float32; break;
                case DataFormat::RawUInt16: src_format = DataFormat::Float16; break;
                default: src_format = DataFormat::Lf8; break;
            }
        }
        unpack_src_format.push_back(src_format);
    }
    return unpack_src_format;
}

const DataFormat get_single_unpack_dst_format(
    const DataFormat src_format, const DataFormat pack_format, const DataFormat unpack_conditional_dst_format) {
    DataFormat dst_format = src_format;
    if (src_format == DataFormat::Float32) {
        TT_FATAL(
            (unpack_conditional_dst_format == DataFormat::Float16) ||
                (unpack_conditional_dst_format == DataFormat::Float16_b) ||
                (unpack_conditional_dst_format == DataFormat::Tf32) ||
                (unpack_conditional_dst_format == DataFormat::Float32),
            "fp32 conditional format can only be fp16a/b or fp32");
        dst_format = unpack_conditional_dst_format;
    } else if (is_bfp_format(src_format)) {
        dst_format = is_exp_b_format(src_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
    }

    return dst_format;
}

bool is_all_fp32_formats(const DataFormat data_format[NUM_CIRCULAR_BUFFERS]) {
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        if (data_format[i] != DataFormat::Invalid && data_format[i] != DataFormat::Float32) {
            return false;
        }
    }
    return true;
}

std::vector<DataFormat> get_unpack_dst_formats(
    DataFormat buf_formats[NUM_CIRCULAR_BUFFERS],
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    std::vector<UnpackToDestMode> unpack_to_dest_mode,
    bool int_fpu_en) {
    if (!unpack_to_dest_mode.empty()) {
        TT_FATAL(
            unpack_to_dest_mode.size() == NUM_CIRCULAR_BUFFERS, "unpack_to_dest_mode vector must have 32 elements");
    }

    std::vector<DataFormat> unpack_dst_format;

    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        DataFormat src_format = buf_formats[i];
        if (src_format == DataFormat::RawUInt32 || src_format == DataFormat::RawUInt16 ||
            src_format == DataFormat::RawUInt8) {
            switch (src_format) {
                case DataFormat::RawUInt32: src_format = DataFormat::Float32; break;
                case DataFormat::RawUInt16: src_format = DataFormat::Float16; break;
                default: src_format = DataFormat::Lf8; break;
            }
            unpack_dst_format.push_back(src_format);
        } else if (int_fpu_en) {
            unpack_dst_format.push_back(src_format);
        } else {
            if (buf_formats[i] == DataFormat::Float32 && !unpack_to_dest_mode.empty() &&
                unpack_to_dest_mode[i] != UnpackToDestMode::Default) {
                unpack_dst_format.push_back(
                    get_single_unpack_dst_format(src_format, DataFormat::Invalid, DataFormat::Float32));
            } else {
                unpack_dst_format.push_back(
                    get_single_unpack_dst_format(src_format, DataFormat::Invalid, unpack_conditional_dst_format));
            }
        }
    }
    return unpack_dst_format;
}

const DataFormat get_single_pack_src_format(
    DataFormat data_format,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool bfp8_pack_precise,
    bool int_fpu_en,
    tt::ARCH arch) {
    if (data_format == DataFormat::Fp8_e4m3) {
        TT_FATAL(arch == tt::ARCH::BLACKHOLE, "Fp8 E4M3 mode only available in Blackhole");
    }

    DataFormat pack_src_format;
    const ExpPrecision input_exp_width = get_exp_precison(data_format);
    const ExpPrecision output_exp_width = get_exp_precison(data_format);
    const ExpPrecision fp32_condition_exp_width = get_exp_precison(unpack_conditional_dst_format);

    bool is_input_or_output_float32 = data_format == DataFormat::Float32;
    bool condition_exp_float32_match_output =
        is_input_or_output_float32 && fp32_condition_exp_width == output_exp_width;

    if (data_format == DataFormat::RawUInt32 || data_format == DataFormat::RawUInt16 ||
        data_format == DataFormat::RawUInt8) {
        switch (data_format) {
            case DataFormat::RawUInt32: pack_src_format = DataFormat::Float32; break;
            case DataFormat::RawUInt16: pack_src_format = DataFormat::Float16; break;
            default: pack_src_format = DataFormat::Lf8; break;
        }
    } else if (data_format == DataFormat::UInt16) {
        pack_src_format = data_format;
    } else if (data_format == DataFormat::Invalid) {
        pack_src_format = DataFormat::Invalid;
    } else if (data_format == DataFormat::Fp8_e4m3) {
        pack_src_format = DataFormat::Float16;
    } else if (fp32_dest_acc_en) {
        TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Dest Fp32 mode is not supported for arch grayskull");

        if (is_bfp_format(data_format)) {
            pack_src_format = bfp8_pack_precise
                                  ? DataFormat::Float32
                                  : (is_exp_b_format(data_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8);
        } else if (is_exp_b_format(data_format) || (data_format == DataFormat::Float32)) {
            pack_src_format = data_format;
        } else if (data_format == DataFormat::Float16) {
            pack_src_format = DataFormat::Float16_b;
        } else if (data_format == DataFormat::UInt32) {
            pack_src_format = DataFormat::UInt32;
        } else if (data_format == DataFormat::Int32) {
            pack_src_format = DataFormat::Int32;
        } else if (data_format == DataFormat::UInt16) {
            pack_src_format = DataFormat::UInt16;
        } else if (data_format == DataFormat::UInt8) {
            pack_src_format = DataFormat::UInt8;
        } else {
            TT_THROW("No valid conversion from fp32 dest to output format = {}", data_format);
        }
    } else if (int_fpu_en) {
        TT_THROW("Integer math is not supported");
        // TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Integer math is not supported for arch grayskull");
        // If output is integer, then pack_src_format is integer as conversion in packer is not supported
        // If output if float, then pack_src_format is Float32 as sfpu outut if Float32
        if (tt::is_integer_format(data_format)) {
            pack_src_format = data_format;
        } else {
            pack_src_format = DataFormat::Float32;
        }
    } else if (tt::is_integer_format(data_format)) {
        pack_src_format = data_format;
    } else if (
        (!is_input_or_output_float32 && input_exp_width == output_exp_width) || condition_exp_float32_match_output ||
        data_format == DataFormat::Float32) {
        if (is_input_or_output_float32) {
            // Assert that pack_src_format has same exp width as input format
            TT_FATAL(
                (unpack_conditional_dst_format == DataFormat::Float16_b ||
                 unpack_conditional_dst_format == DataFormat::Float16),
                "fp32 conditional format can only be fp16a/b");

            if (data_format != DataFormat::Float32) {
                TT_FATAL(
                    (input_exp_width == fp32_condition_exp_width),
                    "Input format exponent width = {}, must match pack src format exponent width = {}",
                    data_format,
                    unpack_conditional_dst_format);
            }
            pack_src_format = unpack_conditional_dst_format;
        } else if (is_bfp_format(data_format)) {
            pack_src_format = bfp8_pack_precise
                                  ? (is_exp_b_format(data_format) ? DataFormat::Float16_b : DataFormat::Float16)
                                  : (is_exp_b_format(data_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8);
        } else {
            pack_src_format = data_format;
        }
    } else {
        // Inputs and outputs are different exponent widths, gs/wha0 only support this mode for fp16
        if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            TT_FATAL(
                (data_format == DataFormat::Float16_b) || (data_format == DataFormat::Float16),
                "Exponent width conversion is only supported for float16 formats for grayskull/wormhole_a0");
        }

        // Pack_src_format is the same data format as output data format, but with same exponent width as input data
        // format A/B format mixing only occurs at packer level
        DataFormat pack_src_format_tmp = data_format;

        if (is_bfp_format(data_format)) {
            pack_src_format_tmp = bfp8_pack_precise
                                      ? (is_exp_b_format(data_format) ? DataFormat::Float16_b : DataFormat::Float16)
                                      : (is_exp_b_format(data_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8);
        }

        if (pack_src_format_tmp != DataFormat::Float32) {
            pack_src_format = CONVERT_EXP_WIDTH.at(pack_src_format_tmp);
            if (data_format != DataFormat::Float32) {
                TT_FATAL(
                    input_exp_width == get_exp_precison(pack_src_format),
                    "Input format exponent width = {}, must match pack src format exponent width = {}",
                    data_format,
                    pack_src_format);
            }
        } else {
            pack_src_format = pack_src_format_tmp;
        }
    }
    return pack_src_format;
}

std::vector<DataFormat> get_pack_src_formats(
    DataFormat data_formats[NUM_CIRCULAR_BUFFERS],
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool bfp8_pack_precise,
    bool int_fpu_en,
    tt::ARCH arch) {
    std::vector<DataFormat> pack_src_formats;
    DataFormat pack_src_format;
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        pack_src_format = get_single_pack_src_format(
            data_formats[i], unpack_conditional_dst_format, fp32_dest_acc_en, bfp8_pack_precise, int_fpu_en, arch);
        pack_src_formats.push_back(pack_src_format);
    }

    return pack_src_formats;
}

std::vector<DataFormat> get_pack_dst_formats(DataFormat buf_formats[NUM_CIRCULAR_BUFFERS]) {
    std::vector<DataFormat> pack_dst_format;
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        DataFormat dst_format = buf_formats[i];
        if (dst_format == DataFormat::RawUInt32 || dst_format == DataFormat::RawUInt16 ||
            dst_format == DataFormat::RawUInt8) {
            switch (dst_format) {
                case DataFormat::RawUInt32: dst_format = DataFormat::Float32; break;
                case DataFormat::RawUInt16: dst_format = DataFormat::Float16; break;
                default: dst_format = DataFormat::Lf8; break;
            }
        }
        pack_dst_format.push_back(dst_format);
    }
    return pack_dst_format;
}

}  // namespace tt
