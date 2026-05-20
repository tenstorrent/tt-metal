// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <buffer.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace tt::tt_metal::emule {

// Master ASAN switch for all emule sanitizers (host + kernel side). Off by default.
// Re-read on every call — see emule_strict_cb_boundary_enabled comment in
// emulated_program_runner.cpp for why caching breaks combined test runs.
inline bool emule_asan_enabled() {
    const char* v = std::getenv("TT_METAL_EMULE_ASAN");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// Host-side use-after-free / use-before-allocation sanitizer for Buffer access.
// Catches the case where a caller holds a live Buffer object (or shared_ptr to one)
// whose backing device memory has been reclaimed by the allocator (DeallocateBuffer,
// allocator reset, etc.) and then issues a host-side read/write through it. Without
// this check, address() still returns the cached stale address and the write silently
// stomps memory that the allocator may have already re-issued to another buffer.
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

// Host→device alignment sanitizers. Device/kernel-side accesses are checked
// inside tt-emule's Core::l1_ptr; these guard the symmetric host entry points
// (WriteToDevice{L1,DRAMChannel}, ReadFromDevice{L1,DRAMChannel}) so misaligned
// addresses passed from host code are caught before they reach the cluster
// write path. Alignment values match WH/N150.
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
