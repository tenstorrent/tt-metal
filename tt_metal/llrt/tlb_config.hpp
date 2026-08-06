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

// L2CPU (X280) LIM aperture covered by the per-tile static TLB that
// configure_static_tlbs() programs on Blackhole.
//
// Unlike Tensix/ETH -- whose windows are anchored at NOC address 0 because their
// L1 lives in [0, 2 MiB) -- LIM starts at 0x08000000, so a window anchored at 0
// would cover no usable LIM address. The window is therefore anchored at the LIM
// base, and callers writing through it must convert absolute LIM addresses to
// window-relative offsets (device_addr - tlb->get_base_address()).
//
// Anything addressing LIM through that window -- the H2D/D2H socket config
// buffers and the HOST_PUSH data FIFO -- must lie entirely within
// [kL2cpuLimBase, kL2cpuLimTlbEnd).
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
