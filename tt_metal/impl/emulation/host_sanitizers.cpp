// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Emule-only implementation of the host-side sanitizer facade (host_sanitizers.hpp).
// Compiled into libtt_metal ONLY when TT_METAL_USE_EMULE is set, so it is the single
// place that references __emule_asan_panic / MetalContext — a non-emule build never
// carries an unresolved panic reference. See SANITIZER_CHECKS.md.

#include "host_sanitizers.hpp"

#include <cstddef>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/tt_cluster.hpp"
#include "emule_live_ranges.hpp"

// Single definition lives in emule_asan_panic.cpp, linked into the same libtt_metal.
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...);

namespace tt::tt_metal::emule {

namespace {
// The transfer's real alignment requirement: DMA alignment when a DMA engine
// backs the transfer, otherwise 1 (the host/UMD poke path imposes none — e.g.
// emule's memory-backed I/O, or the unaligned row-major page remainders that
// WriteToBuffer issues by design). Resolved here, behind the enabled-check, so
// it never runs on the production host path.
uint32_t host_alignment_requirement(const IDevice* device, uint32_t size) {
    return MetalContext::instance().get_cluster().get_alignment_requirements(device->id(), size);
}
}  // namespace

void check_buffer_allocated(const Buffer& buffer, const char* op) {
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

void check_host_l1_alignment(const IDevice* device, uint32_t address, uint32_t size, const char* op) {
    if (!emule_asan_enabled()) {
        return;
    }
    // Query the real requirement (1 on the byte-granular host->L1 path) rather than
    // hardcoding word/NoC alignment, which would false-positive legitimate writes.
    const uint32_t alignment = host_alignment_requirement(device, size);
    if (alignment > 1 && address % alignment != 0) {
        __emule_asan_panic(
            "[ASAN ERROR] L1 Alignment: %s host address 0x%x must be %u-byte aligned\n", op, address, alignment);
    }
}

void check_host_dram_alignment(const IDevice* device, uint32_t address, uint32_t size, const char* op) {
    if (!emule_asan_enabled()) {
        return;
    }
    const uint32_t alignment = host_alignment_requirement(device, size);
    if (alignment > 1 && address % alignment != 0) {
        __emule_asan_panic(
            "[ASAN ERROR] DRAM Alignment: %s host address 0x%x must be %u-byte aligned\n", op, address, alignment);
    }
}

void check_program_metadata_size(Program& program) {
    // The CB/L1 validators only fire when an L1 tensor pins
    // lowest_occupied_compute_l1_address; on a freshly-initialized device a
    // program can still statically exceed the reserved KERNEL_CONFIG window.
    const auto& hal = MetalContext::instance().hal();
    const auto& metadata_sizes = program.impl().get_program_config_sizes();
    for (uint32_t pct_index = 0; pct_index < hal.get_programmable_core_type_count(); pct_index++) {
        HalProgrammableCoreType pct = hal.get_programmable_core_type(pct_index);
        uint32_t metadata_size = metadata_sizes[pct_index];
        // TENSIX disallows hal.get_dev_size(KERNEL_CONFIG); its window is dynamic
        // = DEFAULT_UNRESERVED_base - KERNEL_CONFIG_base. Other core types report a
        // static KERNEL_CONFIG size. Mirrors program_dispatch::initialize_worker_config_buf_mgr.
        uint32_t window_size;
        if (pct == HalProgrammableCoreType::TENSIX) {
            uint32_t kc_base = static_cast<uint32_t>(hal.get_dev_addr(pct, HalL1MemAddrType::KERNEL_CONFIG));
            uint32_t unreserved_base =
                static_cast<uint32_t>(hal.get_dev_addr(pct, HalL1MemAddrType::DEFAULT_UNRESERVED));
            window_size = unreserved_base - kc_base;
        } else {
            window_size = hal.get_dev_size(pct, HalL1MemAddrType::KERNEL_CONFIG);
        }
        if (metadata_size > window_size) {
            TT_THROW(
                "Program metadata size {} exceeds reserved KERNEL_CONFIG window {} for programmable core type {}",
                metadata_size,
                window_size,
                pct_index);
        }
    }
}

void report_metadata_overflow(bool is_emulated, const char* what) {
    if (is_emulated && emule_asan_enabled()) {
        __emule_asan_panic("[ASAN ERROR] Metadata Overflow: Program metadata exceeds reserved L1 region — %s\n", what);
    }
}

void register_logical_size(const Buffer& buffer, DeviceAddr logical_size) {
    if (!emule_asan_enabled()) {
        return;
    }
    TT_FATAL(
        logical_size <= buffer.size(),
        "logical_size ({}) must not exceed buffer size ({})",
        logical_size,
        buffer.size());
    if (!buffer.is_allocated()) {
        return;
    }
    if (buffer.buffer_type() != BufferType::L1 && buffer.buffer_type() != BufferType::L1_SMALL) {
        return;
    }
    uint32_t start = static_cast<uint32_t>(buffer.address());
    uint32_t physical_end = static_cast<uint32_t>(buffer.address() + buffer.size());
    if (logical_size == buffer.size()) {
        LiveL1PaddingRanges::clear(buffer.device()->id(), start);
    } else {
        uint32_t logical_end = static_cast<uint32_t>(buffer.address() + logical_size);
        LiveL1PaddingRanges::set(buffer.device()->id(), start, logical_end, physical_end);
    }
}

}  // namespace tt::tt_metal::emule
