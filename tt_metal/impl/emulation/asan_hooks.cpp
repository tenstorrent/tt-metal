// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/emulation/asan_hooks.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/asan_interface.h>
#endif
#endif

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_page_mapping.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>

#include "impl/context/metal_context.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/asan_bridge.h"

namespace tt::tt_metal::emulation {

namespace {

// True iff we're running in the emulated cluster. All entry points short-
// circuit when this is false so non-emulated builds pay only a single load.
bool is_emulated_target() {
    return MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Emule;
}

// Mirrors get_sw_emulated_chip() in emulated_program_runner.cpp — go through
// the cluster's UMD driver and dynamic_cast to the emule chip type.
tt::umd::SWEmuleChip* sw_emulated_chip_for(IDevice* device) {
    if (!device) {
        return nullptr;
    }
    auto& cluster = MetalContext::instance().get_cluster();
    auto* umd_cluster = cluster.get_driver().get();
    if (!umd_cluster) {
        return nullptr;
    }

    // Try direct lookup. Works when Buffer::device() is the underlying
    // physical IDevice. Fails for MeshBuffer-allocated buffers where
    // Buffer::device() is the MeshDevice (whose id() doesn't match a UMD
    // chip id — e.g. MeshDevice id 1 wrapping chip id 0).
    auto try_lookup = [&](int chip_id) -> tt::umd::SWEmuleChip* {
        try {
            return dynamic_cast<tt::umd::SWEmuleChip*>(umd_cluster->get_chip(chip_id));
        } catch (...) {
            return nullptr;
        }
    };
    if (auto* sw = try_lookup(device->id()); sw != nullptr) {
        return sw;
    }

    // Fallback: iterate every registered UMD chip and return the first that's
    // an SWEmuleChip. In single-chip emulation that's chip 0; multi-chip
    // emulation isn't currently supported (every chip would need its own
    // SWEmuleChip lookup tied to a specific buffer).
    for (auto chip_id : umd_cluster->get_target_device_ids()) {
        if (auto* sw = try_lookup(chip_id); sw != nullptr) {
            return sw;
        }
    }
    return nullptr;
}

struct PoisonRange {
    CoreCoord logical_core;  // logical worker (L1) or logical DRAM channel core
    bool is_dram;
    DeviceAddr offset;
    std::size_t size;
};

// Enumerate the (core, offset, size) ranges that this buffer occupies. The
// caller is responsible for translating logical → virtual for L1 cores.
//
// `base_address`: the bank-manager-assigned address. Pass the local from
// AllocatorImpl::allocate_buffer when called at allocate-time (where
// Buffer::address_ has not yet been written and Buffer::address() throws).
// Pass 0 / unused when called at deallocate-time — the function falls back to
// buffer->address() (valid by then). HYBRID per-core allocations ignore this
// argument entirely; per_core_addresses_ is consulted directly.
std::vector<PoisonRange> enumerate_buffer_ranges(const Buffer* buffer, DeviceAddr base_address) {
    std::vector<PoisonRange> ranges;
    if (!buffer) {
        return ranges;
    }

    const auto buffer_type = buffer->buffer_type();
    const bool is_dram = buffer->is_dram();
    const std::size_t per_bank_size = static_cast<std::size_t>(buffer->aligned_size_per_bank());
    if (per_bank_size == 0) {
        return ranges;
    }

    // Case 1: HYBRID per-core-address allocation. Each entry in
    // per_core_addresses_ is its own (core, address) range. base_address is
    // unused — per_core_addresses_ is the source of truth.
    if (experimental::per_core_allocation::is_per_core_allocation(*buffer)) {
        const auto& per_core = experimental::per_core_allocation::get_per_core_addresses(*buffer);
        ranges.reserve(per_core.size());
        for (const auto& [core, addr] : per_core) {
            ranges.push_back(PoisonRange{core, /*is_dram=*/false, addr, per_bank_size});
        }
        return ranges;
    }

    // The address Buffer::address() returns is what hashes into bank_offset
    // arithmetic for the interleaved/sharded paths. At alloc-time the caller
    // passes base_address; at dealloc-time we read it from the buffer.
    const DeviceAddr addr = base_address != 0 ? base_address : buffer->address();

    // Case 2: Sharded L1 (non-HYBRID). Walk the page-mapping per core and sum
    // contiguous device-page ranges to get the actual byte count owned by each
    // shard. This avoids the over-unpoisoning of partial shards that the
    // earlier per_bank_size fallback caused. get_buffer_page_mapping is a
    // logically-const lazy build but missing the const qualifier upstream;
    // const_cast is the standard workaround.
    if (!is_dram && is_sharded(buffer->buffer_layout()) && buffer->has_shard_spec()) {
        const auto& mapping = const_cast<Buffer*>(buffer)->get_buffer_page_mapping();
        if (mapping == nullptr) {
            return ranges;
        }
        const auto page_size = static_cast<std::size_t>(buffer->aligned_page_size());
        ranges.reserve(mapping->all_cores.size());
        for (std::size_t i = 0; i < mapping->all_cores.size(); ++i) {
            std::size_t total_pages = 0;
            for (const auto& range : mapping->core_page_mappings[i]) {
                total_pages += range.num_pages;
            }
            if (total_pages == 0) {
                continue;
            }
            ranges.push_back(PoisonRange{mapping->all_cores[i], /*is_dram=*/false, addr, total_pages * page_size});
        }
        return ranges;
    }

    // Case 3: Interleaved DRAM / banked L1. Walk every bank, computing each
    // bank's address by replicating Buffer::translate_page_address arithmetic
    // (= addr + allocator->get_bank_offset(buffer_type, bank_id)). Buffer::page_address
    // would do the same but it asserts on a non-finalized buffer at alloc-time.
    auto* device = buffer->device();
    if (!device) {
        return ranges;
    }
    const auto& allocator = device->allocator();
    if (!allocator) {
        return ranges;
    }
    const uint32_t num_banks = allocator->get_num_banks(buffer_type);
    ranges.reserve(num_banks);
    for (uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        const DeviceAddr bank_addr = addr + allocator->get_bank_offset(buffer_type, bank_id);
        const CoreCoord logical_core = allocator->get_logical_core_from_bank_id(bank_id);
        ranges.push_back(PoisonRange{logical_core, is_dram, bank_addr, per_bank_size});
    }
    return ranges;
}

// Resolve a PoisonRange to a host-pointer base. Returns nullptr if the chip
// cannot resolve the range to a backing core (in which case the caller skips
// this range).
uint8_t* resolve_range_host_base(tt::umd::SWEmuleChip* sw_chip, IDevice* device, const PoisonRange& r) {
    if (!sw_chip || !device) {
        return nullptr;
    }
    // SWEmuleChip::core_for_logical takes a tt::umd::CoreCoord but only
    // reads .x / .y — the CoreType / CoordSystem fields are unused. We
    // construct a UMD CoreCoord from the bare {x,y}.
    CoreCoord metal_coord = r.logical_core;
    if (!r.is_dram) {
        // L1 case: translate logical → virtual worker NOC coord. The chip
        // helper deliberately does not pull in tt-metalium for this.
        metal_coord = device->virtual_core_from_logical_core(r.logical_core, ::tt::CoreType::WORKER);
    }
    tt::umd::CoreCoord umd_coord(
        metal_coord.x,
        metal_coord.y,
        r.is_dram ? ::tt::CoreType::DRAM : ::tt::CoreType::WORKER,
        r.is_dram ? ::tt::CoordSystem::LOGICAL : ::tt::CoordSystem::NOC0);
    auto* core = sw_chip->core_for_logical(umd_coord, r.is_dram);
    if (!core) {
        return nullptr;
    }
    return core->l1_data() + static_cast<std::size_t>(r.offset);
}

}  // namespace

// Set TT_EMULE_ASAN_TRACE=1 in the environment to get one stderr line per
// hook invocation showing the buffer pointer, host base, and size of each
// poison/unpoison operation. Useful for verifying the alloc and dealloc
// hooks actually run for a given test.
//
// The hook is invoked OUTSIDE AllocatorImpl::mutex_ because the enumerator
// calls back into get_logical_core_from_bank_id() and get_bank_offset()
// which both lock that same mutex — re-entry deadlocks. allocate_buffer
// and deallocate_buffer release the lock before calling the hook.
namespace {
bool asan_trace_enabled() {
    static const bool enabled = []() {
        const char* s = std::getenv("TT_EMULE_ASAN_TRACE");
        return s != nullptr && s[0] != '\0' && !(s[0] == '0' && s[1] == '\0');
    }();
    return enabled;
}
}  // namespace

void on_buffer_allocated(const Buffer* buffer, DeviceAddr base_address) {
    if (!is_emulated_target() || !buffer) {
        return;
    }
    auto* sw_chip = sw_emulated_chip_for(buffer->device());
    if (!sw_chip) {
        return;
    }
    const auto ranges = enumerate_buffer_ranges(buffer, base_address);
    for (const auto& r : ranges) {
        if (auto* base = resolve_range_host_base(sw_chip, buffer->device(), r)) {
            __emule_buffer_alloc(base, r.size);
            if (asan_trace_enabled()) {
                std::fprintf(
                    stderr,
                    "[ASAN_TRACE] alloc   buffer=%p base=%p size=%zu offset=0x%lx core=(%u,%u) is_dram=%d\n",
                    static_cast<const void*>(buffer),
                    static_cast<const void*>(base),
                    r.size,
                    static_cast<unsigned long>(r.offset),
                    static_cast<unsigned>(r.logical_core.x),
                    static_cast<unsigned>(r.logical_core.y),
                    r.is_dram ? 1 : 0);
            }
        }
    }
}

void on_buffer_deallocated(const Buffer* buffer) {
    if (!is_emulated_target() || !buffer) {
        return;
    }
    auto* sw_chip = sw_emulated_chip_for(buffer->device());
    if (!sw_chip) {
        return;
    }
    // Pass 0: enumerate_buffer_ranges falls back to buffer->address(), which
    // is finalized by the time deallocate runs.
    const auto ranges = enumerate_buffer_ranges(buffer, /*base_address=*/0);
    if (asan_trace_enabled()) {
        std::fprintf(
            stderr,
            "[ASAN_TRACE] dealloc buffer=%p is_dram=%d num_ranges=%zu\n",
            static_cast<const void*>(buffer),
            buffer->is_dram() ? 1 : 0,
            ranges.size());
    }
    for (const auto& r : ranges) {
        auto* base = resolve_range_host_base(sw_chip, buffer->device(), r);
        if (!base && asan_trace_enabled()) {
            std::fprintf(
                stderr,
                "[ASAN_TRACE] dealloc range UNRESOLVED core=(%u,%u) is_dram=%d offset=0x%lx size=%zu\n",
                static_cast<unsigned>(r.logical_core.x),
                static_cast<unsigned>(r.logical_core.y),
                r.is_dram ? 1 : 0,
                static_cast<unsigned long>(r.offset),
                r.size);
        }
        if (base) {
            __emule_buffer_free(base, r.size);
            if (asan_trace_enabled()) {
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
                int probe_after = __asan_address_is_poisoned(base);
#else
                int probe_after = -1;
#endif
#else
                int probe_after = -1;
#endif
                std::fprintf(
                    stderr,
                    "[ASAN_TRACE] dealloc buffer=%p base=%p size=%zu offset=0x%lx core=(%u,%u) is_dram=%d "
                    "immediately_poisoned=%d\n",
                    static_cast<const void*>(buffer),
                    static_cast<const void*>(base),
                    r.size,
                    static_cast<unsigned long>(r.offset),
                    static_cast<unsigned>(r.logical_core.x),
                    static_cast<unsigned>(r.logical_core.y),
                    r.is_dram ? 1 : 0,
                    probe_after);
            }
        }
    }
}

void on_allocator_configured(IDevice* device) {
    if (!is_emulated_target() || !device) {
        return;
    }
    auto* sw_chip = sw_emulated_chip_for(device);
    if (!sw_chip) {
        return;
    }
    const auto& allocator = device->allocator();
    if (!allocator) {
        return;
    }
    sw_chip->initialize_asan_poison(
        static_cast<uint32_t>(allocator->get_base_allocator_addr(HalMemType::L1)),
        static_cast<uint32_t>(allocator->get_base_allocator_addr(HalMemType::DRAM)));
}

}  // namespace tt::tt_metal::emulation
