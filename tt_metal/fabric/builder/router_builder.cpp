// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/router_builder.hpp"
#include "fabric_context.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/api/tt-metalium/data_types.hpp"

namespace tt::tt_fabric::builder {

RouterBuilder::RouterBuilder(const tt::tt_fabric::FabricEriscDatamoverBuilder& erisc_builder) :
    erisc_kernel_builder(erisc_builder) {}
RouterBuilder::RouterBuilder(
    const tt::tt_fabric::FabricEriscDatamoverBuilder& erisc_builder,
    const tt::tt_fabric::FabricTensixDatamoverBuilder& tensix_builder) :
    erisc_kernel_builder(erisc_builder), tensix_kernel_builder(tensix_builder) {}

void RouterBuilder::create_kernels(
    tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, const InitConfig& init_config, chan_id_t eth_chan) {
    if (tensix_kernel_builder.has_value()) {
        create_tensix_kernel(device, program);
    }
    create_erisc_kernel(device, program, init_config, eth_chan);
}

void RouterBuilder::create_erisc_kernel(
    tt::tt_metal::IDevice* device, tt::tt_metal::Program& program, const InitConfig& init_config, chan_id_t eth_chan) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    auto& fabric_context = control_plane.get_fabric_context();

    std::map<std::string, std::string> defines = {};
    if (fabric_context.is_2D_routing_enabled()) {
        defines["FABRIC_2D"] = "";
    }

    auto& edm_builder = this->erisc_kernel_builder;
    edm_builder.set_wait_for_host_signal(true);
    const std::vector<uint32_t> rt_args = edm_builder.get_runtime_args();
    for (uint32_t risc_id = 0; risc_id < edm_builder.get_configured_risc_count(); risc_id++) {
        std::vector<uint32_t> ct_args = edm_builder.get_compile_time_args(risc_id);

        const auto is_master_risc_core = eth_chan == init_config.master_router_chan && (risc_id == 0);
        ct_args.push_back(is_master_risc_core);
        ct_args.push_back(init_config.master_router_chan);
        ct_args.push_back(init_config.num_local_fabric_routers);
        ct_args.push_back(init_config.router_channels_mask);

        auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
        auto kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
            eth_logical_core,
            tt::tt_metal::EthernetConfig{
                .noc = edm_builder.config.risc_configs[risc_id].get_configured_noc(),
                .processor = static_cast<tt::tt_metal::DataMovementProcessor>(risc_id),
                .compile_args = ct_args,
                .defines = defines,
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

        tt::tt_metal::SetRuntimeArgs(program, kernel, eth_logical_core, rt_args);
    }

    log_debug(
        tt::LogMetal,
        "Building fabric router -> device (phys): {}, (logical): {}, channel: {}, num_local_fabric_routers: {}",
        device->id(),
        control_plane.get_fabric_node_id_from_physical_chip_id(device->id()).chip_id,
        eth_chan,
        num_local_fabric_routers);
}

void RouterBuilder::create_tensix_kernel(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
    TT_FATAL(tensix_kernel_builder.has_value(), "Tensix kernel builder not found");
    tensix_kernel_builder->create_and_compile(device, program);
}

}  // namespace tt::tt_fabric::builder
