// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <tt-metalium/tt_backend_api_types.hpp>     // for DataFormat
#include <tt-umd/types/arch.hpp>                // for ARCH

namespace tt::tt_metal {
enum class UnpackToDestMode : std::uint8_t;
}  // namespace tt::tt_metal

namespace tt {

enum class ExpPrecision : std::uint8_t {
    A = 0,
    B = 1,
};

DataFormat check_valid_formats_in_out_data_formats(std::span<const DataFormat> data_format);
ExpPrecision get_data_exp_precision(std::span<const DataFormat> data_formats);

// Checks if all formats in format array are fp32/tf32/invalid, then data can be unpacked as tf32 for fp32 accumulation
bool is_all_fp32_formats(std::span<const DataFormat> data_format);

// True for any OCP MX block-scaled format.
bool is_mx_format(DataFormat data_format);

// Quasar has no UInt32 unpack/pack encoding: the 32-bit integer path carries it as Int32 (same bits).
DataFormat quasar_canonical_format(DataFormat data_format);

// Formats are canonicalised for Quasar (see quasar_canonical_format) when arch == QUASAR.
std::vector<DataFormat> get_unpack_src_formats(
    std::span<const DataFormat> data_formats, tt::ARCH arch = tt::ARCH::WORMHOLE_B0);
std::vector<DataFormat> get_unpack_dst_formats(
    std::span<const DataFormat> buf_formats,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode,
    bool int_fpu_en = false,
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0);

// (Quasar only) Kernel-wide UnpackToDestEn: any explicit to-dest mode entry (TT_FATAL when the DEST is 32-bit and that
// operand is not Float32 / Int32 / UInt32), or, with the 32-bit DEST enabled, any Int32/UInt32 operand.
bool quasar_unpack_to_dest_en(
    std::span<const DataFormat> buf_formats,
    const std::vector<tt::tt_metal::UnpackToDestMode>& unpack_to_dest_mode,
    bool fp32_dest_acc_en);

[[deprecated("use quasar_unpack_to_dest_en")]]
bool any_unpack_to_dest(const std::vector<tt::tt_metal::UnpackToDestMode>& unpack_to_dest_mode);

std::vector<DataFormat> get_pack_src_formats(
    std::span<const DataFormat> data_formats,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool bfp8_pack_precise,
    bool int_fpu_en = false,
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0);
std::vector<DataFormat> get_pack_dst_formats(
    std::span<const DataFormat> buf_formats, tt::ARCH arch = tt::ARCH::WORMHOLE_B0);

}  // namespace tt
