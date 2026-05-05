// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/hal_types.hpp>

namespace tt::tt_metal {
class Buffer;
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal::emulation {

// Called from AllocatorImpl::allocate_buffer after the buffer's address is
// computed but BEFORE Buffer::address_ has been stored back. base_address is
// the result of bank_manager->allocate_buffer; for HYBRID per-core allocations
// (where each core has its own address) this is unused and the hook reads
// buffer->per_core_addresses_ directly (already populated via
// set_per_core_addresses by the time the hook fires). For each (core, offset,
// size) triple the buffer covers, resolves the emulated tt_emule::Core* via
// SWEmuleChip and invokes __emule_buffer_alloc to unpoison. No-op when the
// cluster's target type is not Emule.
void on_buffer_allocated(const Buffer* buffer, DeviceAddr base_address);

// Symmetric: called from AllocatorImpl::deallocate_buffer (and
// ::deallocate_buffers, which iterates allocated_buffers_ before mass-
// deallocating). Calls __emule_buffer_free per range.
void on_buffer_deallocated(const Buffer* buffer);

// Convenience hook called once per device after the AllocatorConfig is
// known. Forwards l1_unreserved_base / dram_unreserved_base to the chip
// via SWEmuleChip::initialize_asan_poison so existing cores get their
// allocator-managed regions initial-poisoned. No-op for non-emulated
// targets. Idempotent — safe to call multiple times.
void on_allocator_configured(IDevice* device);

}  // namespace tt::tt_metal::emulation
