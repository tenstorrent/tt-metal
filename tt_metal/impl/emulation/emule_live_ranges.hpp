// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Per-device live-buffer registries consumed by the emule sanitizers.
// snapshot() returns packed (low << 32) | high pairs — flat uint64_t scan
// in JIT translation units, no std::vector layout dependency. See SANITIZERS.md.

#include <cstdint>
#include <vector>

namespace tt::tt_metal::emule {

class LiveL1Ranges {
public:
    static void add(int device_id, uint32_t start, uint32_t end);
    static void remove(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

class LiveDramRanges {
public:
    static void add(int device_id, uint32_t start, uint32_t end);
    static void remove(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

// Padded-tensor layout descriptors declared via Buffer::set_logical_size /
// set_padded_layout. Tensor padding is 2-D, not a single trailing band: a tensor
// whose logical extent is smaller than its padded (tile/alignment-rounded) extent
// has padding both at the right edge of every data row AND in the trailing rows
// (an "L" shape). The descriptor carries the 2-D layout so the kernel-side check
// can decide, per access, whether a byte falls in the data region or the padding,
// via closed-form modulo math (see __emule_offset_in_padding in the jit_hw
// headers). `layout`: 0 = row-major, 1 = tiled (32x32 tiles stored as 4 16x16
// nfaces, see include/jit_hw/nfaces.h). Dimensions are in elements; `padded_cols`
// is the physical row width (row-major) or tile-grid width (tiled), and must be a
// multiple of 32 for the tiled layout.
//
// N-D / batched tensors: an [..., H, W] tensor tiled on its last two dims is stored
// as G stacked pages, each a padded H_padded x W_padded matrix; the pad pattern
// repeats with period H_padded down the rows. `padded_page_rows` carries that
// period so the kernel resets the row test per page (`row % padded_page_rows >=
// logical_rows`, with logical_rows = the per-page logical height). All dims above
// the last two just contribute more pages of the same period, so one modulo covers
// any dimensionality. A single 2-D matrix is the G==1 case: padded_page_rows = its
// padded height (0 is also accepted and means "no row paging").
//
// snapshot() emits 4 packed uint64 words per descriptor (see emule_live_ranges.cpp).
class LiveL1PaddingRanges {
public:
    static void set(
        int device_id,
        uint32_t start,
        uint32_t physical_end,
        uint8_t layout,
        uint32_t elem_size,
        uint32_t logical_rows,
        uint32_t logical_cols,
        uint32_t padded_cols,
        uint32_t padded_page_rows);
    static void clear(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

}  // namespace tt::tt_metal::emule
