// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <tt-metalium/tt_backend_api_types.hpp>     // for DataFormat
#include <umd/device/types/arch.hpp>                // for ARCH

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

std::vector<DataFormat> get_unpack_src_formats(std::span<const DataFormat> data_formats);
std::vector<DataFormat> get_unpack_dst_formats(
    std::span<const DataFormat> buf_formats,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode,
    bool int_fpu_en = false);

// True if any data-flow buffer is configured to unpack directly to Dest (i.e. any entry is not
// UnpackToDestMode::Default). Used to derive the (Quasar-only) kernel-wide UnpackToDestEn sync
// selector from the same per-operand vector that drives the unpack dst formats, so the two agree.
// Quasar has no performance penalty for unpacking to Dest for any data format, unlike WH/BH
// which restrict it to 32-bit formats; the routing is captured kernel-wide, not per-operand.
bool any_unpack_to_dest(const std::vector<tt::tt_metal::UnpackToDestMode>& unpack_to_dest_mode);

std::vector<DataFormat> get_pack_src_formats(
    std::span<const DataFormat> data_formats,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool bfp8_pack_precise,
    bool int_fpu_en = false,
    tt::ARCH arch = tt::ARCH::WORMHOLE_B0);
std::vector<DataFormat> get_pack_dst_formats(std::span<const DataFormat> buf_formats);

}  // namespace tt
