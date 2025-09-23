// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal::detail {

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
