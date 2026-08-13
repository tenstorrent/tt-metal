// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llrt/metal_soc_descriptor.hpp"
#include <tt_backend_api_types.hpp>
#include <unordered_map>

#include <umd/device/cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

struct metal_SocDescriptor;

namespace ll_api {

// LIM aperture covered by the per-tile L2CPU static TLB that
// configure_static_tlbs() programs on Blackhole. The window is anchored at the
// LIM base rather than 0, so callers writing through it must convert absolute
// LIM addresses to window-relative offsets. Anything addressed through that
// window must lie entirely within [kL2cpuLimBase, kL2cpuLimTlbEnd).
inline constexpr uint64_t kL2cpuLimBase = 0x08000000ULL;
inline constexpr uint64_t kL2cpuLimTlbSize = 2ULL * 1024 * 1024;
inline constexpr uint64_t kL2cpuLimTlbEnd = kL2cpuLimBase + kL2cpuLimTlbSize;

void configure_static_tlbs(
    tt::ARCH arch,
    tt::ChipId mmio_device_id,
    const metal_SocDescriptor& sdesc,
    tt::umd::Cluster& device_driver,
    bool include_dram_tlbs = true);

}  // namespace ll_api
