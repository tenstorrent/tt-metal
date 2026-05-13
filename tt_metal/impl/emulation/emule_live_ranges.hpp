// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

namespace tt::tt_metal::emule {

// Per-device live L1 buffer extents, populated by Buffer::allocate_impl /
// deallocate_impl. Consumed by emulated_program_runner before each kernel
// launch to plumb the array down to the kernel thread, where the inline
// sanitizer in __emule_local_l1_to_ptr checks accesses against it. Ranges are
// packed as (start << 32) | end so the kernel-side check is a flat scan over
// uint64_t without needing the std::vector layout in JIT translation units.
class LiveL1Ranges {
public:
    static void add(int device_id, uint32_t start, uint32_t end);
    static void remove(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

// DRAM mirror of LiveL1Ranges. Same packing, same per-device keying. Tracks
// the global DRAM extent of every allocated BufferType::DRAM buffer so the
// OOB sanitizer inside __emule_dram_ptr can validate that an offset falls in
// some allocated tensor. Stored as (start << 32) | end — DRAM offsets in
// emule are bounded by the bridge's mmap size (well under 2^32 today), so
// uint32_t per endpoint is sufficient.
class LiveDramRanges {
public:
    static void add(int device_id, uint32_t start, uint32_t end);
    static void remove(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

// Padding-region registry for the Tensor-Padding-Violation sanitizer.
// Populated by Buffer::set_logical_size when a caller declares a logical size
// smaller than the physical allocation; removed by Buffer::deallocate_impl.
// Each entry describes the padded byte range [logical_end, physical_end) of
// one L1 buffer — the sanitizer inside __emule_local_l1_to_ptr aborts on any
// access that lands inside one of these ranges. set() is keyed by the
// buffer's physical start address so re-declaring the logical size for the
// same buffer overwrites the previous entry; clear() also keys by start.
// snapshot() returns packed (logical_end << 32) | physical_end pairs so the
// kernel-side scan is a flat array of uint64_t (matching LiveL1Ranges).
class LiveL1PaddingRanges {
public:
    static void set(int device_id, uint32_t start, uint32_t logical_end, uint32_t physical_end);
    static void clear(int device_id, uint32_t start);
    static std::vector<uint64_t> snapshot(int device_id);
};

}  // namespace tt::tt_metal::emule
