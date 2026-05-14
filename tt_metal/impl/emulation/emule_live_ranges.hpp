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

}  // namespace tt::tt_metal::emule
