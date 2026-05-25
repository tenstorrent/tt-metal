// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side emule sanitizers + master switch. See SANITIZERS.md.

#include <buffer.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace tt::tt_metal::emule {

// Re-read every call: caching breaks combined test runs that toggle the var.
inline bool emule_asan_enabled() {
    const char* v = std::getenv("TT_METAL_EMULE_ASAN");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

inline void check_buffer_allocated(const tt::tt_metal::Buffer& buffer, const char* op) {
    if (!emule_asan_enabled()) {
        return;
    }
    if (!buffer.is_allocated()) {
        fprintf(
            stderr,
            "[ASAN ERROR] Use-After-Free: %s called on Buffer (unique_id=%zu, size=%lu, type=%d) "
            "that is not currently allocated (either deallocated or never allocated). "
            "This would access reclaimed device memory and corrupt unrelated allocations on silicon.\n",
            op,
            buffer.unique_id(),
            static_cast<unsigned long>(buffer.size()),
            static_cast<int>(buffer.buffer_type()));
        std::abort();
    }
}

inline void check_host_l1_alignment(uint32_t address, const char* op) {
    if (!emule_asan_enabled()) {
        return;
    }
    if (address % 4 != 0) {
        fprintf(
            stderr,
            "[ASAN ERROR] L1 Alignment: %s host address 0x%x must be 4-byte aligned\n",
            op,
            address);
        std::abort();
    }
}

inline void check_host_dram_alignment(uint32_t address, const char* op) {
    if (!emule_asan_enabled()) {
        return;
    }
    if (address % 32 != 0) {
        fprintf(
            stderr,
            "[ASAN ERROR] DRAM Alignment: %s host address 0x%x must be 32-byte aligned (WH)\n",
            op,
            address);
        std::abort();
    }
}

}  // namespace tt::tt_metal::emule
