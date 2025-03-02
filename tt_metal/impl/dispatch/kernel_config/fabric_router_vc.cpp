// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <iostream>
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
    const auto& control_plane = tt::DevicePool::instance().get_control_plane();
    auto us_kernel = upstream_kernels_.at(0);
    auto ds_kernel = downstream_kernels_.at(0);

    // Upstream can be PREFETCH_H or DISPATCH_D
    // Downstream can be PREFETCH_D or DISPATCH_H
    // 4 Combinations
    const auto& [src_mesh_id, src_chip_id] =
        control_plane->get_mesh_chip_id_from_physical_chip_id(us_kernel->GetDeviceId());
    const auto& [dst_mesh_id, dst_chip_id] =
        control_plane->get_mesh_chip_id_from_physical_chip_id(ds_kernel->GetDeviceId());
    const auto& routers = control_plane->get_routers_to_chip(src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
    const auto& [routing_plane, fabric_router] = routers.front();
    if (auto prefetch_h_us = dynamic_cast<PrefetchKernel*>(us_kernel);
        auto prefetch_d_ds = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
        std::cout << fmt::format("Fabric Router VC - Sender Path (Prefetch H to Prefetch D)\n");
    }

    if (auto dispatch_d_us = dynamic_cast<DispatchKernel*>(us_kernel);
        auto dispatch_h_ds = dynamic_cast<DispatchKernel*>(ds_kernel)) {
        std::cout << fmt::format("Fabric Router VC - Return Path (Dispatch D to Dispatch H)\n");
    }

    // Downstream path. src -> dst
    us_kernel->UpdateArgsForFabric(fabric_router, src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
    // Upstream path. dst -> src
    ds_kernel->UpdateArgsForFabric(fabric_router, dst_mesh_id, dst_chip_id, src_mesh_id, src_chip_id);

    TT_FATAL(false, "FabricRouterVC is not implemented for this path\n");
}

void FabricRouterVC::CreateKernel() {}

void FabricRouterVC::ConfigureCore() {}

}  // namespace tt::tt_metal
