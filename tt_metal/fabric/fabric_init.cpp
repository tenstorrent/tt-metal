// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "metal_soc_descriptor.h"

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_fabric {

std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program(tt::tt_metal::IDevice* device) {
    auto fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();

    // Use FabricBuilder to orchestrate the build
    FabricBuilder builder(device, *fabric_program_ptr, fabric_context);

    // Execute build phases
    builder.discover_channels();
    builder.create_routers();
    if (!builder.has_routers()) {
        return nullptr;
    }

<<<<<<< HEAD
    builder.connect_routers();
    builder.compile_ancillary_kernels();
    builder.create_kernels();
=======
    // Compile all fabric tensix builders through router builders
    if (tt::tt_metal::MetalContext::instance().get_fabric_tensix_config() !=
        tt::tt_fabric::FabricTensixConfig::DISABLED) {
        for (auto& [eth_chan, router_builder] : router_builders) {
            if (router_builder->has_tensix_builder()) {
                router_builder->get_tensix_builder().create_and_compile(*fabric_program_ptr);
            }
        }
    }

    // for now it doesnt matter which channel is the master, so just pick the 1st in the map
    auto master_router_chan = router_builders.begin()->first;
    fabric_context.set_fabric_master_router_chan(device->id(), master_router_chan);

    uint32_t router_channels_mask = 0;
    for (const auto& [router_chan, _] : router_builders) {
        router_channels_mask += 0x1 << (uint32_t)router_chan;
    }

    std::map<std::string, std::string> defines = {};
    if (fabric_context.is_2D_routing_enabled()) {
        defines["FABRIC_2D"] = "";
    }

    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto num_enabled_eth_cores = router_builders.size();
    const auto num_enabled_risc_cores =
        router_builders.begin()->second->get_configured_risc_count();  // same across all eth cores
    size_t num_local_fabric_routers = num_enabled_eth_cores;
    for (auto& [eth_chan, router_builder] : router_builders) {
        auto& edm_builder = router_builder->get_erisc_builder();
        edm_builder.set_wait_for_host_signal(true);
        const std::vector<uint32_t> rt_args = edm_builder.get_runtime_args();
        for (uint32_t risc_id = 0; risc_id < num_enabled_risc_cores; risc_id++) {
            std::vector<uint32_t> ct_args = edm_builder.get_compile_time_args(risc_id);

            const auto is_master_risc_core = eth_chan == master_router_chan && (risc_id == 0);
            ct_args.push_back(is_master_risc_core);
            ct_args.push_back(master_router_chan);
            ct_args.push_back(num_local_fabric_routers);
            ct_args.push_back(router_channels_mask);

            auto proc = static_cast<tt::tt_metal::DataMovementProcessor>(risc_id);
            if (tt::tt_metal::MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE &&
                tt::tt_metal::MetalContext::instance().rtoptions().get_enable_2_erisc_mode() &&
                num_enabled_risc_cores == 1) {
                // Force fabric to run on erisc1 due to stack usage exceeded with MUX on erisc0
                proc = tt::tt_metal::DataMovementProcessor::RISCV_1;
            }
            auto eth_logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
            auto kernel = tt::tt_metal::CreateKernel(
                *fabric_program_ptr,
                "tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp",
                eth_logical_core,
                tt::tt_metal::EthernetConfig{
                    .noc = edm_builder.config.risc_configs[risc_id].get_configured_noc(),
                    .processor = proc,
                    .compile_args = ct_args,
                    .defines = defines,
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

            tt::tt_metal::SetRuntimeArgs(*fabric_program_ptr, kernel, eth_logical_core, rt_args);
        }
        log_debug(
            tt::LogMetal,
            "Building fabric router -> device (phys): {}, (logical): {}, channel: {}, num_local_fabric_routers: {}",
            device->id(),
            control_plane.get_fabric_node_id_from_physical_chip_id(device->id()).chip_id,
            eth_chan,
            num_local_fabric_routers);
    }
>>>>>>> 4ff0149b82 (Added NeighborExchange topology to Fabric)

    // Compile the program
    tt::tt_metal::detail::CompileProgram(
        device, *fabric_program_ptr, tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch());

    return fabric_program_ptr;
}

std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(tt::tt_metal::IDevice* device) {
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        return create_and_compile_tt_fabric_program(device);
    }
    return nullptr;
}

void configure_fabric_cores(tt::tt_metal::IDevice* device) {
    auto soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device->id());
    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto addresses_to_clear = builder_context.get_fabric_router_addresses_to_clear();
    const auto& router_config = builder_context.get_fabric_router_config();
    std::vector<uint32_t> router_zero_buf(router_config.router_buffer_clear_size_words, 0);
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
    }
}

}  // namespace tt::tt_fabric
