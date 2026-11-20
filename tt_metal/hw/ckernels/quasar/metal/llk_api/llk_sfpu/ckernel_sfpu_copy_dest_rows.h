// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The row-granular DEST copies (copy_dest_rows / copy_dest_row) are not ported to Quasar. The Compute API keeps its
// entry points on every arch so api/compute/copy_dest_rows.h stays includable; a kernel that calls one fails to build.

#include <cstdint>

#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

[[gnu::error("copy_dest_rows / copy_dest_row are not supported on Quasar")]] void
copy_dest_rows_is_not_supported_on_quasar();

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void copy_dest_band(
    [[maybe_unused]] std::uint32_t dst_index_in,
    [[maybe_unused]] std::uint32_t dst_index_out,
    [[maybe_unused]] std::uint32_t unused,
    [[maybe_unused]] std::uint32_t in_band,
    [[maybe_unused]] std::uint32_t out_band) {
    copy_dest_rows_is_not_supported_on_quasar();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void copy_dest_row(
    [[maybe_unused]] std::uint32_t dst_index_in,
    [[maybe_unused]] std::uint32_t dst_index_out,
    [[maybe_unused]] std::uint32_t unused,
    [[maybe_unused]] std::uint32_t from_row,
    [[maybe_unused]] std::uint32_t to_row) {
    copy_dest_rows_is_not_supported_on_quasar();
}

}  // namespace sfpu
}  // namespace ckernel
