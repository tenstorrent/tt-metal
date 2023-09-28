/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "common/tt_backend_api_types.hpp"

namespace tt {

// A map of invalid Input->Output data-format conversions.
// Input data-format is for the operands that get unpacked.
// Output data-format is for the operands that packer packs into.
static const std::map<DataFormat, std::vector<DataFormat>> ALL_INVALID_FORMAT_CONVERSIONS = {
    {DataFormat::Bfp8,      {DataFormat::Bfp8_b }},
    {DataFormat::Bfp8_b,    {DataFormat::Bfp8   }},
    {DataFormat::Bfp4,      {DataFormat::Bfp8_b }},
    {DataFormat::Bfp4_b,    {DataFormat::Bfp8   }},
    {DataFormat::Bfp2,      {DataFormat::Bfp8_b }},
    {DataFormat::Bfp2_b,    {DataFormat::Bfp8   }},
    {DataFormat::Float16,   {DataFormat::Bfp8_b }},
    {DataFormat::Float16_b, {DataFormat::Bfp8   }},
    {DataFormat::Float32,   {DataFormat::Invalid }},
    {DataFormat::Tf32,      {DataFormat::Invalid }}, // for TF 32, math dest must in fp32, hence output can be anything
    {DataFormat::RawUInt32, {DataFormat::Invalid }},
    {DataFormat::RawUInt16, {DataFormat::Invalid }},
    {DataFormat::UInt32,    {DataFormat::Invalid }},
};

enum class ExpPrecision : uint8_t
{
  A = 0,
  B = 1,
};

bool is_valid_conversion(DataFormat input_format, DataFormat output_format);
bool is_exp_b_format(DataFormat data_format);
ExpPrecision get_exp_precison(DataFormat data_format);
void dump_data_formats(DataFormat data_format[8]);

// Checks the input operand data-formats for consistency.
// All buffers must have the same -b- exponent precision type.
// Returns the last valid data-format between operand buffers.
// This data-format will be used to check consistency across operands
DataFormat check_consistent_format_within_operand(DataFormat data_format[8], bool is_param);

// Checks to see if all buffers within an operand have the same data-format.
// Returns that data format to compare with other operands.
// Must use this for output operands and intermediate buffers.
DataFormat check_same_format_within_operand(DataFormat data_format[8]);

// Checks consistency between input operand data-formats.
// Data-formats for all input operands must have the same -b- exponent precision type.
void check_consistent_format_across_input_operands(DataFormat input_format[8], DataFormat param_format[8]);
DataFormat get_pack_data_format(DataFormat output_formats[8], DataFormat intermed_formats[8]);
ExpPrecision get_data_exp_precision(DataFormat data_formats[8]);
void check_input_to_output_valid_conversion(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat output_formats[8], DataFormat intermed_formats[8]);

// Checks if all formats in format array are fp32/tf32/invalid, then data can be unpacked as tf32 for fp32 accumulation
bool is_all_fp32_formats(const DataFormat data_format[8]);

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

void check_valid_in_out_data_formats(DataFormat input_formats[8], DataFormat output_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8]);
const DataFormat get_single_pack_src_format(DataFormat input_format, DataFormat output_format);

std::vector<DataFormat> get_unpack_src_formats(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8]);
std::vector<DataFormat> get_unpack_dst_formats(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8], DataFormat output_formats[8], DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en);
std::vector<DataFormat> get_pack_src_formats(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8], DataFormat output_formats[8], DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en);
std::vector<DataFormat> get_pack_dst_formats(DataFormat input_formats[8], DataFormat param_formats[8], DataFormat intermed_formats[8], DataFormat output_formats[8]);

}
