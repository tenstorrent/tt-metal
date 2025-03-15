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
        upstream_kernels_.size() == downstream_kernels_.size(),
        "Fabric Router VC requires upstream.size() == downstream.size()");
    const auto& control_plane = tt::Cluster::instance().get_control_plane();
    TT_FATAL(control_plane, "Control plane is nullptr. Is fabric initialized yet?");

    // Zip upstream and downstream kernels together
    for (int i = 0; i < upstream_kernels_.size(); ++i) {
        auto us_kernel = upstream_kernels_.at(i);
        auto ds_kernel = downstream_kernels_.at(i);

        // Upstream can be PREFETCH_H or DISPATCH_D
        // Downstream can be PREFETCH_D or DISPATCH_H
        // 4 Combinations
        const auto& [src_mesh_id, src_chip_id] =
            control_plane->get_mesh_chip_id_from_physical_chip_id(us_kernel->GetDeviceId());
        const auto& [dst_mesh_id, dst_chip_id] =
            control_plane->get_mesh_chip_id_from_physical_chip_id(ds_kernel->GetDeviceId());
        const auto& routers = control_plane->get_routers_to_chip(src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
        const auto& [routing_plane, fabric_router] = routers.front();

        const auto& routers_reversed =
            control_plane->get_routers_to_chip(dst_mesh_id, dst_chip_id, src_mesh_id, src_chip_id);
        const auto& [routing_plane_rev, fabric_router_rev] = routers_reversed.front();
        bool valid_path{false};

        if (auto prefetch_us = dynamic_cast<PrefetchKernel*>(us_kernel);
            auto prefetch_ds = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
            valid_path = true;
        }

        if (auto dispatch_us = dynamic_cast<DispatchKernel*>(us_kernel);
            auto dispatch_ds = dynamic_cast<DispatchKernel*>(ds_kernel)) {
            valid_path = true;
        }

        TT_FATAL(valid_path, "FabricRouterVC is not implemented for this path\n");

        // Downstream path. src -> dst
        us_kernel->UpdateArgsForFabric(fabric_router, src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
        // Upstream path. dst -> src
        ds_kernel->UpdateArgsForFabric(fabric_router_rev, dst_mesh_id, dst_chip_id, src_mesh_id, src_chip_id);
    }
}

void FabricRouterVC::CreateKernel() {}

void FabricRouterVC::ConfigureCore() {}

}  // namespace tt::tt_metal
