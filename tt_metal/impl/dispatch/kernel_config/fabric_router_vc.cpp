// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "assert.hpp"
#include "control_plane.hpp"
#include "dispatch/kernel_config/dispatch.hpp"
#include "dispatch/kernel_config/fd_kernel.hpp"
#include "dispatch/kernel_config/prefetch.hpp"
#include "fabric_router_vc.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

void FabricRouterVC::GenerateStaticConfigs() { kernel_type_ = FDKernelType::VIRTUAL; }

void FabricRouterVC::GenerateDependentConfigs() {
    // Provide router details to upstream and downstream kernels
    TT_ASSERT(
        upstream_kernels_.size() == downstream_kernels_.size(),
        "Fabric Router VC requires upstream.size() == downstream.size()");
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& control_plane = cluster.get_control_plane();
    TT_FATAL(
        cluster.get_fabric_config() != FabricConfig::DISABLED && control_plane,
        "Control plane is nullptr. Is fabric initialized yet?");

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
        const auto& router_chans =
            control_plane->get_forwarding_eth_chans_to_chip(src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
        TT_ASSERT(
            !router_chans.empty(),
            "No routers for (mesh {}, chip {}) to (mesh {}, chip{})",
            src_mesh_id,
            src_chip_id,
            dst_mesh_id,
            dst_chip_id);
        const auto& fabric_router =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                us_kernel->GetDeviceId(), *router_chans.begin());

        const auto& router_chans_rev =
            control_plane->get_forwarding_eth_chans_to_chip(dst_mesh_id, dst_chip_id, src_mesh_id, src_chip_id);
        TT_ASSERT(
            !router_chans_rev.empty(),
            "No routers for return path (mesh {}, chip {}) to (mesh {}, chip{})",
            dst_mesh_id,
            dst_chip_id,
            src_mesh_id,
            src_chip_id);
        const auto& fabric_router_rev =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_eth_core_from_channel(
                ds_kernel->GetDeviceId(), *router_chans_rev.begin());

        bool valid_path{false};
        if (auto prefetch_us = dynamic_cast<PrefetchKernel*>(us_kernel);
            auto prefetch_ds = dynamic_cast<PrefetchKernel*>(ds_kernel)) {
            valid_path = true;
        }

        if (auto dispatch_us = dynamic_cast<DispatchKernel*>(us_kernel);
            auto dispatch_ds = dynamic_cast<DispatchKernel*>(ds_kernel)) {
            valid_path = true;
        }

        TT_FATAL(valid_path, "FabricRouterVC is not implemented for this path");

        // Get outbound ethernet channels
        auto us_outbound_eth_channels = cluster.get_fabric_ethernet_channels(src_chip_id);
        auto ds_outbound_eth_channels = cluster.get_fabric_ethernet_channels(dst_chip_id);
        TT_FATAL(!us_outbound_eth_channels.empty(), "No outbound ethernet channels for upstream kernel");
        TT_FATAL(!ds_outbound_eth_channels.empty(), "No outbound ethernet channels for downstream kernel");

        // Downstream path. src -> dst
        us_kernel->UpdateArgsForFabric(
            fabric_router, *us_outbound_eth_channels.begin(), src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
        ds_kernel->UpdateArgsForFabric(
            fabric_router_rev, *ds_outbound_eth_channels.begin(), src_mesh_id, src_chip_id, dst_mesh_id, dst_chip_id);
    }
}

void FabricRouterVC::CreateKernel() {}

void FabricRouterVC::ConfigureCore() {}

}  // namespace tt::tt_metal
