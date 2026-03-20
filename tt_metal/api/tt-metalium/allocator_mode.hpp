// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

enum class AllocatorMode {
    LOCKSTEP,  // Default: single allocator, all cores get the same address
    HYBRID,    // N+1 allocators: supports both lockstep and per-core addresses
};

}  // namespace tt::tt_metal
