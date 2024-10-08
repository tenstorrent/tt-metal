// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "common/base.hpp"

namespace tt {

static constexpr uint NUM_OPERANDS = 8;

enum class ExpPrecision : uint8_t
{
  A = 0,
  B = 1,
};

bool is_valid_conversion(DataFormat input_format, DataFormat output_format);
bool is_exp_b_format(DataFormat data_format);
ExpPrecision get_exp_precison(DataFormat data_format);
void dump_data_formats(DataFormat data_format[NUM_OPERANDS]);

/*
 * Checks operand data formats for same exponent width format
 * Returns the last valid data-format between operand buffers.
 */
DataFormat check_consistent_format_within_operand(DataFormat data_format[NUM_OPERANDS]);

/*
 * Checks operand data formats for same data format
 * Returns the last valid data-format in operand buffers.
 */
DataFormat check_same_format_within_operand(DataFormat data_format[NUM_OPERANDS]);

/*
 * Checks consistency between input operand data-formats.
 * Data-formats for all input operands must have the same exponent width precision type.
 */
void check_consistent_format_across_input_operands(DataFormat input_format[NUM_OPERANDS], DataFormat param_format[NUM_OPERANDS]);
DataFormat check_valid_formats_within_operand(DataFormat data_format[NUM_OPERANDS]);
DataFormat get_pack_data_format(DataFormat output_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]);
ExpPrecision get_data_exp_precision(DataFormat data_formats[NUM_OPERANDS]);
ExpPrecision get_input_data_exp_precision(DataFormat input_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]);


// Checks if all formats in format array are fp32/tf32/invalid, then data can be unpacked as tf32 for fp32 accumulation
bool is_all_fp32_formats(const DataFormat data_format[NUM_OPERANDS]);

/*
*This pass checks
*      1- The input buffers must all have the same exponent precision.
*      2- Intermediate buffers must also have same exponent precision as inputs.
*      3- Output buffers can have different exponent width formats
*      4- Check all buffers have valid supported formats
*/
void check_valid_in_out_data_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat output_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]);
const DataFormat get_single_pack_src_format(DataFormat input_format, DataFormat output_format, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, tt::ARCH arch);

std::vector<DataFormat> get_unpack_src_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS]);
std::vector<DataFormat> get_unpack_dst_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS], DataFormat output_formats[NUM_OPERANDS], DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, std::vector<UnpackToDestMode> unpack_to_dest_mode, bool int_fpu_en = false);
std::vector<DataFormat> get_pack_src_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS], DataFormat output_formats[NUM_OPERANDS], DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, bool int_fpu_en = false, tt::ARCH arch = tt::ARCH::GRAYSKULL);
std::vector<DataFormat> get_pack_dst_formats(DataFormat input_formats[NUM_OPERANDS], DataFormat param_formats[NUM_OPERANDS], DataFormat intermed_formats[NUM_OPERANDS], DataFormat output_formats[NUM_OPERANDS]);

}
