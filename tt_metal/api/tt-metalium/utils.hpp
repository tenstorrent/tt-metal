// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal::detail {

/**
 * Returns tile size of given data format in bytes
 *
 * Return value: uint32_t
 *
 * | Argument    | Description    | Type                | Valid Range | Required |
 * |-------------|----------------|---------------------|-------------|----------|
 * | data_format | Format of data | tt::DataFormat enum |             | Yes      |
 */
inline uint32_t TileSize(const DataFormat& data_format) { return tt::tile_size(data_format); }

inline DeviceAddr SizeBytesPerBank(
    DeviceAddr size_bytes, DeviceAddr page_size_bytes, uint32_t num_banks, uint32_t alignment_bytes) {
    TT_ASSERT(
        page_size_bytes == 0 ? size_bytes == 0 : size_bytes % page_size_bytes == 0,
        "Page size {} should be divisible by buffer size {}",
        page_size_bytes,
        size_bytes);
    DeviceAddr num_pages = page_size_bytes == 0 ? 0 : size_bytes / page_size_bytes;
    DeviceAddr num_equally_distributed_pages = num_pages == 0 ? 0 : 1 + ((num_pages - 1) / num_banks);
    return num_equally_distributed_pages * round_up(page_size_bytes, static_cast<DeviceAddr>(alignment_bytes));
}

}  // namespace tt::tt_metal::detail
