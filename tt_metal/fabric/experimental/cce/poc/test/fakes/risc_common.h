// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

// Copy the cached view of the range to SRAM.
void flush_l2_cache_range(uintptr_t start_addr, size_t size);
// Copy SRAM to the cached view of the range.
void invalidate_l2_cache_range(uintptr_t start_addr, size_t size);
inline void invalidate_l1_cache() {}
