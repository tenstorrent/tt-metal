// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/core_subset_write/buffer_write.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace tt::tt_metal::experimental::core_subset_write {

void WriteToBuffer(Buffer& buffer, tt::stl::Span<const uint8_t> host_buffer, const CoreRangeSet& logical_core_filter) {
    tt::tt_metal::detail::WriteToBuffer(buffer, host_buffer, &logical_core_filter);
}

}  // namespace tt::tt_metal::experimental::core_subset_write
