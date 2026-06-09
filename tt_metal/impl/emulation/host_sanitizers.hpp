// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side emule sanitizers + master switch. See SANITIZERS.md.

#include <buffer.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Unified ASAN trace. Forward-declared (not #included) because this header is
// compiled into the tt_metal target too, which does not carry the tt-emule
// include path. The single definition lives in emulated_program_runner.cpp
// (EMULE_ASAN_IMPLEMENTATION), linked into the same libtt_metal.
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...);

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
        __emule_asan_panic(
            "[ASAN ERROR] Use-After-Free: %s called on Buffer (unique_id=%zu, size=%lu, type=%d) "
            "that is not currently allocated (either deallocated or never allocated). "
            "This would access reclaimed device memory and corrupt unrelated allocations on silicon.\n",
            op,
            buffer.unique_id(),
            static_cast<unsigned long>(buffer.size()),
            static_cast<int>(buffer.buffer_type()));
    }
}

// `alignment` is the transfer's real requirement from Cluster::get_alignment_requirements(device, size)
// (DMA alignment when DMA-backed, else 1). The host->L1 data path
// (Cluster::write_core -> UMD write_to_device) accepts byte-granular writes and
// WriteToBuffer applies no separate floor, so a hardcoded word/NoC alignment
// here would false-positive legitimate writes (e.g. row-major remainders).
// Register pokes that genuinely need 4-byte alignment go through write_reg,
// not this path.
inline void check_host_l1_alignment(uint32_t address, uint32_t alignment, const char* op) {
    if (!emule_asan_enabled()) {
        return;
    }
    if (alignment > 1 && address % alignment != 0) {
        __emule_asan_panic(
            "[ASAN ERROR] L1 Alignment: %s host address 0x%x must be %u-byte aligned\n", op, address, alignment);
    }
}

// `alignment` is the transfer's real requirement, obtained at the call site
// from Cluster::get_alignment_requirements(device, size): the DMA alignment
// when a DMA engine backs the transfer, otherwise 1. A value of 1 means the
// host/UMD poke path imposes no alignment (e.g. emule's memory-backed I/O, or
// the unaligned row-major page remainders that WriteToBuffer issues by design),
// so the check must be a no-op. Passing the value in keeps this header free of
// MetalContext/Cluster includes.
inline void check_host_dram_alignment(uint32_t address, uint32_t alignment, const char* op) {
    if (!emule_asan_enabled()) {
        return;
    }
    if (alignment > 1 && address % alignment != 0) {
        __emule_asan_panic(
            "[ASAN ERROR] DRAM Alignment: %s host address 0x%x must be %u-byte aligned\n", op, address, alignment);
    }
}

}  // namespace tt::tt_metal::emule
