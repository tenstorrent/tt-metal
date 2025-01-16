// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>
#include "tt_backend_api_types.hpp"              // for DataFormat
#include "umd/device/types/arch.h"                      // for ARCH
#include "circular_buffer_constants.h"  // for NUM_CIRCULAR_BUFFERS

enum class UnpackToDestMode : std::uint8_t;

namespace tt {

enum class ExpPrecision : std::uint8_t {
    A = 0,
    B = 1,
};

bool is_valid_conversion(DataFormat input_format, DataFormat output_format);
bool is_exp_b_format(DataFormat data_format);
ExpPrecision get_exp_precison(DataFormat data_format);
void dump_data_formats(DataFormat data_format[NUM_CIRCULAR_BUFFERS]);

/*
 * Checks operand data formats for same exponent width format
 * Returns the last valid data-format between operand buffers.
 */
DataFormat check_consistent_format_across_buffers(DataFormat data_format[NUM_CIRCULAR_BUFFERS]);

/*
 * Checks operand data formats for same data format
 * Returns the last valid data-format in operand buffers.
 */
DataFormat check_same_format_across_buffers(DataFormat data_format[NUM_CIRCULAR_BUFFERS]);

DataFormat check_valid_formats_in_out_data_formats(DataFormat data_format[NUM_CIRCULAR_BUFFERS]);
ExpPrecision get_data_exp_precision(DataFormat data_formats[NUM_CIRCULAR_BUFFERS]);

// Checks if all formats in format array are fp32/tf32/invalid, then data can be unpacked as tf32 for fp32 accumulation
bool is_all_fp32_formats(const DataFormat data_format[NUM_CIRCULAR_BUFFERS]);

const DataFormat get_single_pack_src_format(
    DataFormat input_format, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, tt::ARCH arch);

std::vector<DataFormat> get_unpack_src_formats(DataFormat buf_formats[NUM_CIRCULAR_BUFFERS]);
std::vector<DataFormat> get_unpack_dst_formats(
    DataFormat buf_formats[NUM_CIRCULAR_BUFFERS],
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    std::vector<UnpackToDestMode> unpack_to_dest_mode,
    bool int_fpu_en = false);
std::vector<DataFormat> get_pack_src_formats(
    DataFormat buf_formats[NUM_CIRCULAR_BUFFERS],
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool bfp8_pack_precise,
    bool int_fpu_en = false,
    tt::ARCH arch = tt::ARCH::GRAYSKULL);
std::vector<DataFormat> get_pack_dst_formats(DataFormat buf_formats[NUM_CIRCULAR_BUFFERS]);

}  // namespace tt
