#include "data_format.hpp"
#include "common/assert.hpp"

namespace tt {
bool is_valid_conversion(DataFormat input_format, DataFormat output_format) {
    auto input_iter = ALL_INVALID_FORMAT_CONVERSIONS.find(input_format);
    if (input_iter == ALL_INVALID_FORMAT_CONVERSIONS.end()) {
        std::cout << "Data-Format Error: ";
        std::cout << "The invalid output (packer) data-formats are not specified in ALL_INVALID_FORMAT_CONVERSIONS for this input data-format" << std::endl;
        std::cout << "Please add the invalid input to output conversions to ALL_INVALID_FORMAT_CONVERSIONS." << std::endl;
        TT_ASSERT(false);
    }

    std::vector<DataFormat> invalid_conversions = input_iter->second;
    for (auto &invalid_output: invalid_conversions) {
        if (output_format == invalid_output) {
            std::cout << "Data-Format Error: ";
            std::cout << "Invalid input to output data-format conversion." << std::endl;
            std::cout << "Input format = " << input_format << " Output format = " << output_format << std::endl;
            return false;
        }
    }
    return true;
}

bool is_exp_b_format(DataFormat data_format) {
    return(
        (data_format == DataFormat::Float16_b)
        || (data_format == DataFormat::Bfp8_b)
        || (data_format == DataFormat::Bfp4_b)
        || (data_format == DataFormat::Bfp2_b));
}

ExpPrecision get_exp_precison(DataFormat data_format) {
    return (is_exp_b_format(data_format) ? ExpPrecision::B : ExpPrecision::A);
}

void dump_data_formats(DataFormat data_format[8]) {
    for (int i = 0; i < 8; i++) {
        std::cout << "Operand idx " << i << ": " << data_format[i] << "," <<std::endl;
    }
}

// Checks the input operand data-formats for consistency.
// All buffers must have the same -b- exponent precision type.
// Returns the last valid data-format between operand buffers.
// This data-format will be used to check consistency across operands
DataFormat check_consistent_format_within_operand(DataFormat data_format[8], bool is_param) {
    DataFormat last_valid_format = DataFormat::Invalid;
    for (int i = 0; i < 8; i++) {
        // Special case where Float32 can pair with any exponent precision, skip checking
        if ((data_format[i] == DataFormat::Float32) || (data_format[i] == DataFormat::Tf32) ||
            (data_format[i] == DataFormat::RawUInt32) || (data_format[i] == DataFormat::RawUInt16) || (data_format[i] == DataFormat::RawUInt8)){
            continue;
        }

        if (data_format[i] != DataFormat::Invalid && last_valid_format != DataFormat::Invalid) {
            if (is_exp_b_format(data_format[i]) != is_exp_b_format(last_valid_format)) {
                std::cout << "Data-Format Error: ";
                std::cout << "All input data-formats must have the same exponent b format." << std::endl;
                if (is_param) {
                    std::cout << "Param data formats:" << std::endl;
                } else {
                    std::cout << "Input data formats:" << std::endl;
                }
                dump_data_formats(data_format);
                TT_ASSERT(false);
            }
            last_valid_format = data_format[i];

        } else if (data_format[i] != DataFormat::Invalid && last_valid_format == DataFormat::Invalid) {
            last_valid_format = data_format[i];
        }
    }
    return last_valid_format;
}

// Checks to see if all buffers within an operand have the same data-format.
// Returns that data format to compare with other operands.
// Must use this for output operands and intermediate buffers.
DataFormat check_same_format_within_operand(DataFormat data_format[8]) {
    DataFormat last_valid_format = DataFormat::Invalid;
    for (int i = 0; i < 8; i++) {
        if (data_format[i] != DataFormat::Invalid && last_valid_format != DataFormat::Invalid) {
            if (data_format[i] != last_valid_format) {
                std::cout << "Data-Format Error: ";
                std::cout << "Not all buffer data-formats within this operand are the same." << std::endl;
                dump_data_formats(data_format);
                TT_ASSERT(false);
            }
        } else if (data_format[i] != DataFormat::Invalid && last_valid_format == DataFormat::Invalid) {
            last_valid_format = data_format[i];
        }
    }
    return last_valid_format;
}

// Checks consistency between input operand data-formats.
// Data-formats for all input operands must have the same -b- exponent precision type.
void check_consistent_format_across_input_operands(DataFormat input_format[8], DataFormat param_format[8]) {

    DataFormat last_input_valid_format = check_consistent_format_within_operand(input_format, false);
    DataFormat last_param_valid_format = check_consistent_format_within_operand(param_format, true);
    if (last_input_valid_format != DataFormat::Invalid && last_param_valid_format != DataFormat::Invalid) {
        TT_ASSERT(is_exp_b_format(last_input_valid_format) == is_exp_b_format(last_param_valid_format));
    }
}

bool is_all_fp32_formats(const DataFormat data_format[8]) {
    for (int i = 0; i < 8; i++) {
        if (data_format[i] != DataFormat::Invalid && data_format[i] != DataFormat::Float32 && data_format[i] != DataFormat::Tf32) {
            return false;
        }
    }
    return true;
}

DataFormat get_pack_data_format(DataFormat output_formats[8], DataFormat intermed_formats[8]) {
    DataFormat output_format = check_same_format_within_operand(output_formats);
    if (output_format == DataFormat::Invalid) {
        DataFormat intermed_format = check_same_format_within_operand(intermed_formats);
        return intermed_format;
    } else {
        return output_format;
    }
}

ExpPrecision get_data_exp_precision(DataFormat data_formats[8]) {
    DataFormat last_valid_format = check_consistent_format_within_operand(data_formats, false);
    return get_exp_precison(last_valid_format);
}

void check_input_to_output_valid_conversion(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat output_formats[8], DataFormat intermed_formats[8]) {
    DataFormat output_format = get_pack_data_format(output_formats, intermed_formats);
    for (int i = 0; i < 8; i++) {
        if (input_formats[i] != DataFormat::Invalid) {
            TT_ASSERT(is_valid_conversion(input_formats[i], output_format));
        }
    }
    for (int i = 0; i < 8; i++) {
        if (param_formats[i] != DataFormat::Invalid) {
            TT_ASSERT(is_valid_conversion(param_formats[i], output_format));
        }
    }
}

//  This pass checks
//      1- Correct data-format conversions from input to the output of the op.
//      2- All buffers within output and intermediate operands must have the same data-format.
//      3- The input buffers must all have the same -b- exponent precision.
//          (Check TODO. Ideally applies to only operands that are not unpacked by the same unpacker.)
//      4- If we use the intermediates, all input and output buffers must have the same data-format.
//          (Check TODO. Ideally applies to only the input buffers that get unpacked by the same unpacker that unpack intermediates.)

//  TODO: when input buffer to unpacker index map is available,
//  All buffers that get unpacked by the same unpacker must have the same data-format.
//  All input buffers that get unpacked by different unpackers must only have the same -b- precision.

void check_valid_in_out_data_formats(DataFormat input_formats[8], DataFormat output_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8]) {

    // std::cout << "input format = " << std::endl;
    // dump_data_formats(input_formats);
    // std::cout << "output format = " << std::endl;
    // dump_data_formats(output_formats);
    // std::cout << "param format = " << std::endl;
    // dump_data_formats(param_formats);
    // std::cout << "intermed format = " << std::endl;
    // dump_data_formats(intermed_formats);

    // All intermed buffers must have the same data-format.
    DataFormat intermed_format = check_same_format_within_operand(intermed_formats);
    if (intermed_format != DataFormat::Invalid) {
        // If intermed buffers are being used, input/output/intermed all need to have the same -b- precision.
        check_consistent_format_across_input_operands(input_formats, param_formats);
        check_consistent_format_across_input_operands(input_formats, intermed_formats);
        check_consistent_format_across_input_operands(input_formats, output_formats);
        // If intermed buffers are being used, it is also the output buffer for accumulate ops, hence formats must match
        DataFormat intermed_format = check_same_format_within_operand(intermed_formats);
        DataFormat output_format = check_same_format_within_operand(output_formats);
        TT_ASSERT((intermed_format != DataFormat::Invalid) || (output_format != DataFormat::Invalid));
    } else {
        // If intermed buffers are not being used, input buffers only need to have the same -b- precision.
        check_consistent_format_across_input_operands(input_formats, param_formats);
        check_same_format_within_operand(output_formats);
        check_input_to_output_valid_conversion(input_formats, param_formats, output_formats, intermed_formats);
    }
}

const DataFormat get_single_pack_src_format(DataFormat input_format, DataFormat output_format) {
    uint32_t in_exp_width = (((uint32_t)input_format) >> 2) & 0x1;
    uint32_t out_exp_width = (((uint32_t)output_format) >> 2) & 0x1;
    if ((input_format == DataFormat::Invalid) || (output_format == DataFormat::Invalid)) {
        return DataFormat::Invalid;
    }

    if (input_format == DataFormat::Float32) {
        if (out_exp_width) {
            return DataFormat::Float16_b;
        } else {
            return DataFormat::Float16;
        }
    } else if (output_format == DataFormat::Float32) {
        if (in_exp_width) {
            return DataFormat::Float16_b;
        } else {
            return DataFormat::Float16;
        }
    } else if (input_format == DataFormat::Bfp4 || input_format == DataFormat::Bfp4_b
                || input_format == DataFormat::Bfp2 || input_format == DataFormat::Bfp2_b) {
        if (out_exp_width) {
            return DataFormat::Bfp8_b;
        } else {
            return DataFormat::Bfp8;
        }
    } else if (output_format == DataFormat::Bfp4 || output_format == DataFormat::Bfp4_b
                || output_format == DataFormat::Bfp2 || output_format == DataFormat::Bfp2_b) {
        if (in_exp_width) {
            return DataFormat::Bfp8_b;
        } else {
            return DataFormat::Bfp8;
        }
    }
    else {
        if (in_exp_width != out_exp_width) {
            TT_ASSERT(
                ((output_format == DataFormat::Float16_b) || (output_format == DataFormat::Float16)) &&
                "Exponent width conversion is only supported for float16 formats");
            if (in_exp_width) {
                return DataFormat::Float16_b;
            } else {
                return DataFormat::Float16;
            }
        } else {
            return output_format;
        }
    }
}

std::vector<DataFormat> get_unpack_src_formats(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8]) {
    std::vector<DataFormat> unpack_src_format;
    for (int i=0 ; i<8 ; i++) {
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
    for (int i=0 ; i<8 ; i++) {
        DataFormat src_format = param_formats[i];
        unpack_src_format.push_back(src_format);
    }
    for (int i=0 ; i<8 ; i++) {
        DataFormat src_format = intermed_formats[i];
        unpack_src_format.push_back(src_format);
    }
    return unpack_src_format;
}

const DataFormat get_single_unpack_dst_format(const DataFormat src_format, const DataFormat pack_format, const DataFormat unpack_conditional_dst_format){

    DataFormat dst_format = src_format;
    if (src_format == DataFormat::Float32){
        TT_ASSERT((unpack_conditional_dst_format == DataFormat::Float16) || (unpack_conditional_dst_format == DataFormat::Float16_b) || (unpack_conditional_dst_format == DataFormat::Tf32));
        dst_format = unpack_conditional_dst_format;
    } else if (src_format == DataFormat::Bfp4 || pack_format == DataFormat::Bfp4
                || src_format == DataFormat::Bfp2 || pack_format == DataFormat::Bfp2) {
        dst_format = DataFormat::Bfp8;
    } else if (src_format ==  DataFormat::Bfp4_b || pack_format ==  DataFormat::Bfp4_b
                || src_format ==  DataFormat::Bfp2_b || pack_format ==  DataFormat::Bfp2_b) {
        dst_format = DataFormat::Bfp8_b;
    }
    return dst_format;
}

std::vector<DataFormat> get_unpack_dst_formats(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8], DataFormat output_formats[8], DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en) {
    DataFormat pack_format = get_pack_data_format(output_formats, intermed_formats);

    std::vector<DataFormat> unpack_dst_format;

    DataFormat unpack_cond_dst_format = (tt::is_all_fp32_formats(input_formats) && fp32_dest_acc_en) ? DataFormat::Tf32 : unpack_conditional_dst_format;
    for (int i=0 ; i<8 ; i++) {
        DataFormat src_format = input_formats[i];
        if (src_format == DataFormat::RawUInt32 || src_format == DataFormat::RawUInt16 || src_format == DataFormat::RawUInt8) {
            switch (src_format) {
               case DataFormat::RawUInt32: src_format = DataFormat::Float32; break;
               case DataFormat::RawUInt16: src_format = DataFormat::Float16; break;
               default: src_format = DataFormat::Lf8; break;
            }
            unpack_dst_format.push_back(src_format);
        } else {
            unpack_dst_format.push_back(get_single_unpack_dst_format(input_formats[i], pack_format, unpack_cond_dst_format));
        }
    }
    unpack_cond_dst_format = (tt::is_all_fp32_formats(param_formats) && fp32_dest_acc_en) ? DataFormat::Tf32 : unpack_conditional_dst_format;
    for (int i=0 ; i<8 ; i++) {
        unpack_dst_format.push_back(get_single_unpack_dst_format(param_formats[i], pack_format, unpack_cond_dst_format));
    }
    unpack_cond_dst_format = (tt::is_all_fp32_formats(intermed_formats) && fp32_dest_acc_en) ? DataFormat::Tf32 : unpack_conditional_dst_format;
    for (int i=0 ; i<8 ; i++) {
        unpack_dst_format.push_back(get_single_unpack_dst_format(intermed_formats[i], pack_format, unpack_cond_dst_format));
    }
    return unpack_dst_format;
}

std::vector<DataFormat> get_pack_src_formats(
    DataFormat input_formats[8],
    DataFormat param_formats[8],
    DataFormat intermed_formats[8],
    DataFormat output_formats[8],
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en
) {
    DataFormat pack_format = get_pack_data_format(output_formats, intermed_formats);

    std::vector<DataFormat> pack_src_formats;
    for (int i = 0; i < 8; i++) {
        DataFormat unpack_format = input_formats[i];
        DataFormat pack_src_format = DataFormat::Invalid;

        if (unpack_format == DataFormat::RawUInt32 || unpack_format == DataFormat::RawUInt16 || unpack_format == DataFormat::RawUInt8) {
            switch (unpack_format) {
               case DataFormat::RawUInt32: unpack_format = DataFormat::Float32; break;
               case DataFormat::RawUInt16: unpack_format = DataFormat::Float16; break;
               default: unpack_format = DataFormat::Lf8; break;
            }
            pack_src_format = unpack_format;
        } else {
            if (unpack_format == DataFormat::Float32 || pack_format == DataFormat::Float32) {
               unpack_format = unpack_conditional_dst_format;
            }
            // std::cout << "DEBUG PCK" << i << " unp " << unpack_format << " pck " << pack_format << std::endl;
            pack_src_format = fp32_dest_acc_en ? DataFormat::Float32 : get_single_pack_src_format(unpack_format, pack_format);
        }
        pack_src_formats.push_back(pack_src_format);
    }

    // Intermediates
    for (int i = 0; i < 8; i++) {
        DataFormat unpack_format = intermed_formats[i];
        if (unpack_format == DataFormat::Float32 || pack_format == DataFormat::Float32) {
            unpack_format = unpack_conditional_dst_format;
        }
        // std::cout << "DEBUG INT" << i << " unp " << unpack_format << " pck " << pack_format << std::endl;
        DataFormat pack_src_format = fp32_dest_acc_en ? DataFormat::Float32 : get_single_pack_src_format(unpack_format, pack_format);
        pack_src_formats.push_back(pack_src_format);
    }
    return pack_src_formats;
}

std::vector<DataFormat> get_pack_dst_formats(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8], DataFormat output_formats[8]) {
    DataFormat pack_format = get_pack_data_format(output_formats, intermed_formats);

    std::vector<DataFormat> pack_dst_format;
    for (int i = 0; i < 8; i++) {
        if (i == 0) {
            pack_dst_format.push_back(pack_format);
        } else {
            pack_dst_format.push_back(output_formats[i]);
        }
    }

    // Intermediates
    for (int i = 0; i < 8; i++) {
        pack_dst_format.push_back(intermed_formats[i]);
    }
    return pack_dst_format;
}

}
