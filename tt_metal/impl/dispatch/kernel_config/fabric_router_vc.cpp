// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <tt_metal.hpp>

#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/dispatch_settings.hpp>
#include <tt-metalium/device_pool.hpp>
#include "assert.hpp"
#include "dispatch/kernel_config/dispatch.hpp"
#include "dispatch/kernel_config/prefetch.hpp"

#include "fabric_router_vc.hpp"

namespace tt::tt_metal {

void FabricRouterVC::GenerateStaticConfigs() {}

void FabricRouterVC::GenerateDependentConfigs() {
    // Provide router details to upstream and downstream kernels
    TT_ASSERT(
        upstream_kernels_.size() == 1 && downstream_kernels_.size() == 1,
        "Fabric Router VC requires exactly 1 upstream and 1 downstream");
    auto us_kernel = upstream_kernels_.at(0);
    auto ds_kernel = downstream_kernels_.at(0);

    // Upstream can be PREFETCH_H or DISPATCH_D
    // Downstream can be PREFETCH_D or DISPATCH_H
    // 4 Combinations
    if (auto prefetch_h_us = dynamic_cast<PrefetchKernel*>(us_kernel);
        auto prefetch_d_ds = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
        return;
    }

    if (auto dispatch_d_us = dynamic_cast<DispatchKernel*>(us_kernel);
        auto dispatch_h_ds = dynamic_cast<DispatchKernel*>(ds_kernel)) {
        return;
    }
}

void FabricRouterVC::CreateKernel() {}

void FabricRouterVC::ConfigureCore() {}

std::pair<tt::tt_fabric::routing_plane_id_t, CoreCoord> FabricRouterVC::get_closest_router(
    const IDevice* src, const IDevice* dst) {
    const auto& control_plane = tt::DevicePool::instance().get_control_plane();
    const auto& [src_mesh_id, src_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(src->id());
    const auto& [dst_mesh_id, dst_chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(dst->id());

    const auto& routers = control_plane->get_routers_to_chip(src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
    TT_ASSERT(!routers.empty(), "no fabric routers available from device {} to device {}", src->id(), dst->id());

    // Use first router on the list
    return routers.front();
}

}  // namespace tt::tt_metal
