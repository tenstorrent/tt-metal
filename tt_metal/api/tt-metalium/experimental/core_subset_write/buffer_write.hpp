// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal::experimental::core_subset_write {

// EXPERIMENTAL: may evolve into an overload of WriteToBuffer.
// Writes host bytes to `buffer`, applying the write only to logical cores contained in
// `logical_core_filter`. Empty filter skips all cores; nullptr is not used (pass empty set for no-op).
void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer, const CoreRangeSet& logical_core_filter);

// As above, but writing only the page-aligned sub-region `region` of `buffer`. `host_buffer` holds the
// data for `region` only; the region offset is applied on the device side.
void WriteToBuffer(
    Buffer& buffer,
    ttsl::Span<const uint8_t> host_buffer,
    const BufferRegion& region,
    const CoreRangeSet& logical_core_filter);

}  // namespace tt::tt_metal::experimental::core_subset_write
