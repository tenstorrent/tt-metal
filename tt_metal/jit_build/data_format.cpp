// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "data_format.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include <unordered_map>
#include <set>
namespace tt {

static const std::set<DataFormat> ALL_VALID_FORMATS = {
    DataFormat::Bfp8,
    DataFormat::Bfp8_b,
    DataFormat::Bfp4,
    DataFormat::Bfp4_b,
    DataFormat::Bfp2,
    DataFormat::Bfp2_b,
    DataFormat::Float16,
    DataFormat::Float16_b,
    DataFormat::Float32,
    DataFormat::RawUInt32,
    DataFormat::RawUInt16,
    DataFormat::RawUInt8,
    DataFormat::Tf32,
    DataFormat::Lf8,
    DataFormat::Fp8_e4m3,
    DataFormat::Int8,
    DataFormat::Int32,
    DataFormat::UInt8,
    DataFormat::UInt32,
    DataFormat::UInt16,
};

static const std::unordered_map<DataFormat, DataFormat> CONVERT_EXP_WIDTH = {
    {DataFormat::Bfp8,      DataFormat::Bfp8_b },
    {DataFormat::Bfp8_b,    DataFormat::Bfp8   },
    {DataFormat::Bfp4,      DataFormat::Bfp4_b },
    {DataFormat::Bfp4_b,    DataFormat::Bfp4   },
    {DataFormat::Bfp2,      DataFormat::Bfp2_b },
    {DataFormat::Bfp2_b,    DataFormat::Bfp2   },
    {DataFormat::Float16,   DataFormat::Float16_b},
    {DataFormat::Float16_b, DataFormat::Float16  },
};

bool is_bfp_format(DataFormat data_format) {
    return(
        (data_format == DataFormat::Bfp8_b)
        || (data_format == DataFormat::Bfp8)
        || (data_format == DataFormat::Bfp4_b)
        || (data_format == DataFormat::Bfp4)
        || (data_format == DataFormat::Bfp2_b)
        || (data_format == DataFormat::Bfp2));
}

bool is_exp_b_format(DataFormat data_format) {
    return(
        (data_format == DataFormat::Tf32
        || data_format == DataFormat::Float16_b)
        || (data_format == DataFormat::Bfp8_b)
        || (data_format == DataFormat::Bfp4_b)
        || (data_format == DataFormat::Bfp2_b));
}

ExpPrecision get_exp_precison(DataFormat data_format) {
    return (is_exp_b_format(data_format) ? ExpPrecision::B : ExpPrecision::A);
}

void dump_data_formats(DataFormat data_format[NUM_OPERANDS]) {
    for (int i = 0; i < NUM_OPERANDS; i++) {
        std::cout << "Operand idx " << i << ": " << data_format[i] << "," <<std::endl;
    }
}

DataFormat check_consistent_format_within_operand(DataFormat data_format[NUM_OPERANDS]) {
    DataFormat last_valid_format = DataFormat::Invalid;
    for (int i = 0; i < NUM_OPERANDS; i++) {
        // Special case where Float32 can pair with any exponent precision, skip checking
        if ((data_format[i] == DataFormat::Float32) || (data_format[i] == DataFormat::RawUInt32) ||
            (data_format[i] == DataFormat::UInt32) || (data_format[i] == DataFormat::RawUInt16) ||
            (data_format[i] == DataFormat::RawUInt8) || (data_format[i] == DataFormat::UInt16) ||
            (data_format[i] == DataFormat::UInt8) || (data_format[i] == DataFormat::Int32)) {
            continue;
        }

        if (data_format[i] != DataFormat::Invalid) {
            TT_FATAL(ALL_VALID_FORMATS.find(data_format[i]) != ALL_VALID_FORMATS.end(),
                "Format = {} not supported", data_format[i]);

            if(last_valid_format != DataFormat::Invalid) {
                TT_FATAL(is_exp_b_format(data_format[i]) == is_exp_b_format(last_valid_format),
                    "All input data-formats must have the same exponent format.");
                //dump_data_formats(data_format);
                last_valid_format = data_format[i];

            } else {
                last_valid_format = data_format[i];
            }
        }
    }
    return last_valid_format;
}

DataFormat check_same_format_within_operand(DataFormat data_format[NUM_OPERANDS]) {
    DataFormat last_valid_format = DataFormat::Invalid;
    for (int i = 0; i < NUM_OPERANDS; i++) {
        if (data_format[i] != DataFormat::Invalid && last_valid_format != DataFormat::Invalid) {
            // TT_FATAL(data_format[i] == last_valid_format,
            //     "Not all buffer data-formats within this operand are the same");

            // dump_data_formats(data_format);
        } else if (data_format[i] != DataFormat::Invalid && last_valid_format == DataFormat::Invalid) {
            last_valid_format = data_format[i];
        }
    }
    return last_valid_format;
}

DataFormat check_valid_formats_within_operand(DataFormat data_format[NUM_OPERANDS]) {
    DataFormat last_valid_format = DataFormat::Invalid;
    for (int i = 0; i < NUM_OPERANDS; i++) {
        if (data_format[i] != DataFormat::Invalid) {
            TT_FATAL(ALL_VALID_FORMATS.find(data_format[i]) != ALL_VALID_FORMATS.end(),
                "Format = {} not supported", data_format[i]);
            last_valid_format = data_format[i];
        }
    }
    return last_valid_format;
}

// Checks consistency between input operand data-formats.
// Data-formats for all input operands must have the same -b- exponent precision type.
void check_consistent_format_across_input_operands(DataFormat input_format[NUM_OPERANDS], DataFormat param_format[NUM_OPERANDS]) {

    DataFormat last_input_valid_format = check_consistent_format_within_operand(input_format);
    DataFormat last_param_valid_format = check_consistent_format_within_operand(param_format);
    if (last_input_valid_format != DataFormat::Invalid && last_param_valid_format != DataFormat::Invalid) {
        TT_FATAL(is_exp_b_format(last_input_valid_format) == is_exp_b_format(last_param_valid_format),
            "Formats don't have same exponent width");
    }
}

DataFormat get_pack_data_format(DataFormat output_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]) {
    DataFormat output_format = check_same_format_within_operand(output_formats);
    if (output_format == DataFormat::Invalid) {
        DataFormat intermed_format = check_same_format_within_operand(intermed_formats);
        return intermed_format;
    } else {
        return output_format;
    }
}

ExpPrecision get_data_exp_precision(DataFormat data_formats[NUM_OPERANDS]) {
    DataFormat last_valid_format = check_consistent_format_within_operand(data_formats);
    return get_exp_precison(last_valid_format);
}

ExpPrecision get_input_data_exp_precision(DataFormat input_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]) {
    DataFormat last_valid_format = check_consistent_format_within_operand(input_formats);
    if (last_valid_format == DataFormat::Invalid) {
        last_valid_format = check_consistent_format_within_operand(intermed_formats);
    }

    return get_exp_precison(last_valid_format);
}

void check_valid_in_out_data_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat output_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]) {

    //inputs must have same exp_width, save last found formats with exp width
    DataFormat last_valid_input_format = check_consistent_format_within_operand(input_formats);
    DataFormat last_valid_param_format = check_consistent_format_within_operand(param_formats);
    if (last_valid_input_format != DataFormat::Invalid && last_valid_param_format != DataFormat::Invalid) {
        TT_FATAL((is_exp_b_format(last_valid_input_format) == is_exp_b_format(last_valid_param_format)),
            "Input format = {} and Param format = {} must have same exp width", last_valid_input_format, last_valid_param_format);
    }

    //If intermediate buffers are used, check they have same exp width as inputs
    DataFormat last_valid_intermed_format = check_consistent_format_within_operand(intermed_formats);
    if (last_valid_input_format != DataFormat::Invalid && last_valid_intermed_format != DataFormat::Invalid) {
        TT_FATAL(is_exp_b_format(last_valid_input_format) == is_exp_b_format(last_valid_intermed_format),
            "Input format = {} and Intermed format = {} must have same exp width", last_valid_input_format, last_valid_intermed_format);
    }

    // MT: Why is this check not enabled?
    // DataFormat last_out_valid_format = check_valid_formats_within_operand(output_formats);
    // TT_FATAL((last_out_valid_format != DataFormat::Invalid), "Output format not found");
}

std::vector<DataFormat> get_unpack_src_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]) {
    std::vector<DataFormat> unpack_src_format;
    for (int i=0 ; i<NUM_OPERANDS ; i++) {
        DataFormat src_format = input_formats[i];
        if (src_format == DataFormat::RawUInt32 || src_format == DataFormat::RawUInt16 || src_format == DataFormat::RawUInt8) {
            switch (src_format) {
               case DataFormat::RawUInt32: src_format = DataFormat::Float32; break;
               case DataFormat::RawUInt16: src_format = DataFormat::Float16; break;
               default: src_format = DataFormat::Lf8; break;
            }
        }
        unpack_src_format.push_back(src_format);
    }
    for (int i=0 ; i<NUM_OPERANDS ; i++) {
        DataFormat src_format = param_formats[i];
        unpack_src_format.push_back(src_format);
    }
    for (int i=0 ; i<NUM_OPERANDS ; i++) {
        DataFormat src_format = intermed_formats[i];
        unpack_src_format.push_back(src_format);
    }
    return unpack_src_format;
}

const DataFormat get_single_unpack_dst_format(const DataFormat src_format, const DataFormat pack_format, const DataFormat unpack_conditional_dst_format){

    DataFormat dst_format = src_format;
    if (src_format == DataFormat::Float32){
        TT_FATAL((unpack_conditional_dst_format == DataFormat::Float16) || (unpack_conditional_dst_format == DataFormat::Float16_b) || (unpack_conditional_dst_format == DataFormat::Tf32) || (unpack_conditional_dst_format == DataFormat::Float32),
                    "fp32 conditional format can only be fp16a/b or fp32");
        dst_format = unpack_conditional_dst_format;
    } else if (is_bfp_format(src_format)) {
        dst_format = is_exp_b_format(src_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
    }

    return dst_format;
}

bool is_all_fp32_formats(const DataFormat data_format[NUM_OPERANDS]) {
    for (int i = 0; i < NUM_OPERANDS; i++) {
        if (data_format[i] != DataFormat::Invalid && data_format[i] != DataFormat::Float32) {
            return false;
        }
    }
    return true;
}

std::vector<DataFormat> get_unpack_dst_formats(
    DataFormat input_formats[NUM_OPERANDS],
    DataFormat param_formats[NUM_OPERANDS],
    DataFormat intermed_formats[NUM_OPERANDS],
    DataFormat output_formats[NUM_OPERANDS],
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    std::vector<UnpackToDestMode> unpack_to_dest_mode,
    bool int_fpu_en)
{
    if (!unpack_to_dest_mode.empty()) {
        TT_FATAL(unpack_to_dest_mode.size() == NUM_CIRCULAR_BUFFERS, "unpack_to_dest_mode vector must have 32 elements");
    }

    DataFormat pack_format = get_pack_data_format(output_formats, intermed_formats);
    ExpPrecision input_precision = get_data_exp_precision(input_formats);

    std::vector<DataFormat> unpack_dst_format;

    const bool en_unpack_tf32 = fp32_dest_acc_en && (tt::is_all_fp32_formats(input_formats) || (input_precision == ExpPrecision::B));
    DataFormat unpack_cond_dst_format = en_unpack_tf32 ? DataFormat::Tf32 : unpack_conditional_dst_format;
    for (int i=0 ; i<NUM_OPERANDS ; i++) {
        DataFormat src_format = input_formats[i];
        if (src_format == DataFormat::RawUInt32 || src_format == DataFormat::RawUInt16 || src_format == DataFormat::RawUInt8) {
            switch (src_format) {
               case DataFormat::RawUInt32: src_format = DataFormat::Float32; break;
               case DataFormat::RawUInt16: src_format = DataFormat::Float16; break;
               default: src_format = DataFormat::Lf8; break;
            }
            unpack_dst_format.push_back(src_format);
        } else if (int_fpu_en) {
            unpack_dst_format.push_back(src_format);
        } else {
            if (input_formats[i] == DataFormat::Float32 && !unpack_to_dest_mode.empty() && unpack_to_dest_mode[i] != UnpackToDestMode::Default) {
                unpack_dst_format.push_back(get_single_unpack_dst_format(input_formats[i], pack_format, DataFormat::Float32));
            } else {
                unpack_dst_format.push_back(get_single_unpack_dst_format(input_formats[i], pack_format, unpack_cond_dst_format));
            }
        }
    }
    for (int i=0 ; i<NUM_OPERANDS ; i++) {
        if (param_formats[i] == DataFormat::Float32 && !unpack_to_dest_mode.empty() && unpack_to_dest_mode[NUM_OPERANDS+i] != UnpackToDestMode::Default) {
            unpack_dst_format.push_back(get_single_unpack_dst_format(param_formats[i], pack_format, DataFormat::Float32));
        } else {
            unpack_dst_format.push_back(get_single_unpack_dst_format(param_formats[i], pack_format, unpack_cond_dst_format));
        }
    }
    for (int i=0 ; i<NUM_OPERANDS ; i++) {
        if (intermed_formats[i] == DataFormat::Float32 && !unpack_to_dest_mode.empty() && unpack_to_dest_mode[3*NUM_OPERANDS+i] != UnpackToDestMode::Default) {
            unpack_dst_format.push_back(get_single_unpack_dst_format(intermed_formats[i], pack_format, DataFormat::Float32));
        } else {
            unpack_dst_format.push_back(get_single_unpack_dst_format(intermed_formats[i], pack_format, unpack_cond_dst_format));
        }
    }
    return unpack_dst_format;
}

const DataFormat get_single_pack_src_format(
    DataFormat input_format,
    DataFormat output_format,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool int_fpu_en,
    tt::ARCH arch) {

    if(input_format == DataFormat::Fp8_e4m3) {
        TT_FATAL(arch == tt::ARCH::BLACKHOLE, "Fp8 E4M3 mode only available in Blackhole");
    }

    DataFormat pack_src_format;
    const ExpPrecision input_exp_width = get_exp_precison(input_format);
    const ExpPrecision output_exp_width = get_exp_precison(output_format);
    const ExpPrecision fp32_condition_exp_width = get_exp_precison(unpack_conditional_dst_format);

    bool is_input_or_output_float32 = input_format == DataFormat::Float32 || output_format == DataFormat::Float32;
    bool condition_exp_float32_match_output = is_input_or_output_float32 && fp32_condition_exp_width == output_exp_width;

    if (input_format == DataFormat::RawUInt32 || input_format == DataFormat::RawUInt16 || input_format == DataFormat::RawUInt8) {
        switch (input_format) {
            case DataFormat::RawUInt32: pack_src_format = DataFormat::Float32; break;
            case DataFormat::RawUInt16: pack_src_format = DataFormat::Float16; break;
            default: pack_src_format = DataFormat::Lf8; break;
        }
    } else if (input_format == DataFormat::UInt16) {
        pack_src_format = output_format;
    } else if ((input_format == DataFormat::Invalid) || (output_format == DataFormat::Invalid)) {
        pack_src_format =  DataFormat::Invalid;
    } else if (input_format == DataFormat::Fp8_e4m3) {
        pack_src_format =  DataFormat::Float16;
    } else if (fp32_dest_acc_en) {
        TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Dest Fp32 mode is not supported for arch grayskull");

        if (is_bfp_format(output_format)) {
            pack_src_format = DataFormat::Bfp8_b;
        } else if(is_exp_b_format(output_format) || (output_format == DataFormat::Float32)) {
            pack_src_format = output_format;
        } else if(output_format == DataFormat::Float16){
            pack_src_format = DataFormat::Float16_b;
        } else if(output_format == DataFormat::UInt32){
            pack_src_format = DataFormat::UInt32;
        } else if(output_format == DataFormat::Int32){
            pack_src_format = DataFormat::Int32;
        } else if(output_format == DataFormat::UInt16){
            pack_src_format = DataFormat::UInt16;
        } else if(output_format == DataFormat::UInt8){
            pack_src_format = DataFormat::UInt8;
        } else {
            TT_THROW("No valid conversion from fp32 dest to output format = {}", output_format);
        }
    } else if (int_fpu_en) {
        TT_THROW("Integer math is not supported");
        // TT_FATAL(arch != tt::ARCH::GRAYSKULL, "Integer math is not supported for arch grayskull");
        // If output is integer, then pack_src_format is integer as conversion in packer is not supported
        // If output if float, then pack_src_format is Float32 as sfpu outut if Float32
        if (tt::is_integer_format(output_format)) {
            pack_src_format = output_format;
        } else {
            pack_src_format = DataFormat::Float32;
        }
    } else if (tt::is_integer_format(output_format)) {
        pack_src_format = output_format;
    } else if ( (!is_input_or_output_float32 && input_exp_width == output_exp_width ) || condition_exp_float32_match_output || output_format == DataFormat::Float32) {
        if (is_input_or_output_float32) {
            //Assert that pack_src_format has same exp width as input format
            TT_FATAL((unpack_conditional_dst_format == DataFormat::Float16_b || unpack_conditional_dst_format == DataFormat::Float16),
                        "fp32 conditional format can only be fp16a/b");

            if(input_format != DataFormat::Float32) {
                TT_FATAL((input_exp_width == fp32_condition_exp_width),
                    "Input format exponent width = {}, must match pack src format exponent width = {}", input_format, unpack_conditional_dst_format);
            }
            pack_src_format = unpack_conditional_dst_format;
        } else if (is_bfp_format(output_format)) {
            pack_src_format = is_exp_b_format(output_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
        } else {
            pack_src_format = output_format;
        }
    } else {
        //Inputs and outputs are different exponent widths, gs/wha0 only support this mode for fp16
        if(arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
            TT_FATAL((output_format == DataFormat::Float16_b) || (output_format == DataFormat::Float16),
                "Exponent width conversion is only supported for float16 formats for grayskull/wormhole_a0");
        }

        //Pack_src_format is the same data format as output data format, but with same exponent width as input data format
        //A/B format mixing only occurs at packer level
        DataFormat pack_src_format_tmp = output_format;

        if (is_bfp_format(output_format)) {
            pack_src_format_tmp = is_exp_b_format(output_format) ? DataFormat::Bfp8_b : DataFormat::Bfp8;
        }

        if (pack_src_format_tmp != DataFormat::Float32) {
            pack_src_format = CONVERT_EXP_WIDTH.at(pack_src_format_tmp);
            if (input_format != DataFormat::Float32) {
                TT_FATAL(input_exp_width == get_exp_precison(pack_src_format),
                    "Input format exponent width = {}, must match pack src format exponent width = {}", input_format, pack_src_format);
            }
        } else {
            pack_src_format = pack_src_format_tmp;
        }
    }
    return pack_src_format;
}

std::vector<DataFormat> get_pack_src_formats(
    DataFormat input_formats[NUM_OPERANDS],
    DataFormat param_formats[NUM_OPERANDS],
    DataFormat intermed_formats[NUM_OPERANDS],
    DataFormat output_formats[NUM_OPERANDS],
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool int_fpu_en,
    tt::ARCH arch
) {
    DataFormat pack_output_format = get_pack_data_format(output_formats, intermed_formats);

    std::vector<DataFormat> pack_src_formats;
    DataFormat pack_src_format;
    for (int i = 0; i < NUM_OPERANDS; i++) {
        pack_src_format = get_single_pack_src_format(input_formats[i], pack_output_format, unpack_conditional_dst_format, fp32_dest_acc_en, int_fpu_en, arch);
        pack_src_formats.push_back(pack_src_format);
    }

    // Intermediates
    for (int i = 0; i < NUM_OPERANDS; i++) {
        //Intermediates can be inputs & outputs to same op, provide same format per operand id
        pack_src_format = get_single_pack_src_format(intermed_formats[i], intermed_formats[i], unpack_conditional_dst_format, fp32_dest_acc_en, int_fpu_en, arch);
        pack_src_formats.push_back(pack_src_format);
    }
    return pack_src_formats;
}

std::vector<DataFormat> get_pack_dst_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS], DataFormat output_formats[NUM_OPERANDS]) {
    DataFormat pack_format = get_pack_data_format(output_formats, intermed_formats);

    std::vector<DataFormat> pack_dst_format;
    for (int i = 0; i < NUM_OPERANDS; i++) {
        if (i == 0) {
            pack_dst_format.push_back(pack_format);
        } else {
            pack_dst_format.push_back(output_formats[i]);
        }
    }

    // Intermediates
    for (int i = 0; i < NUM_OPERANDS; i++) {
        pack_dst_format.push_back(intermed_formats[i]);
    }
    return pack_dst_format;
}

}
