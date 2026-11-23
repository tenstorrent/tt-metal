// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

namespace tt::tt_metal {

// Bump allocator over the aux region of CQ sysmem: the tail after all per-CQ issue/completion buffers, in the hugepage
// or in the DRAM-backed CQ staging buffer. It hands out device-visible host memory, e.g. D2H socket buffers when the
// host can't pin memory. Regions are never freed; they live as long as the device.
class AuxRegionAllocator {
public:
    // Bytes each CQ gives up to the aux region; the code that lays out a device's CQs calls it once per device.
    static uint32_t reserved_bytes_per_cq(uint8_t num_hw_cqs, bool with_d2h_fallback_reserve);

    // An empty allocator: allocate() returns {nullptr, 0}.
    AuxRegionAllocator() = default;

    // host_base and device_base address the same size bytes from the host and from the device.
    // has_d2h_fallback_reserve: size includes the D2H fallback reserve (see reserved_bytes_per_cq).
    AuxRegionAllocator(char* host_base, uint32_t device_base, uint32_t size, bool has_d2h_fallback_reserve);

    // Not copyable: a copy would hand out the same bytes again. A move leaves the source empty.
    AuxRegionAllocator(const AuxRegionAllocator&) = delete;
    AuxRegionAllocator& operator=(const AuxRegionAllocator&) = delete;
    AuxRegionAllocator(AuxRegionAllocator&& other) noexcept;
    AuxRegionAllocator& operator=(AuxRegionAllocator&& other) noexcept;
    ~AuxRegionAllocator() = default;

    // Returns the host pointer and device address of a zeroed region of at least size bytes (rounded up to 64 B),
    // or {nullptr, 0} if the allocator is empty. Fails if the region is exhausted.
    std::pair<void*, uint32_t> allocate(uint32_t size);

    uint32_t size() const { return size_; }

    uint32_t used() const { return used_; }

    // Whether the region includes the D2H fallback reserve. A D2H socket that needs the fallback (see
    // Cluster::needs_d2h_hugepage_fallback) requires it.
    bool has_d2h_fallback_reserve() const { return has_d2h_fallback_reserve_; }

private:
    static constexpr uint32_t kMinAlignment = 64;

    char* host_base_ = nullptr;
    uint32_t device_base_ = 0;
    uint32_t size_ = 0;
    uint32_t used_ = 0;
    bool has_d2h_fallback_reserve_ = false;
};

}  // namespace tt::tt_metal
