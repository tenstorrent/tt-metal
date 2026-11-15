// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"
#include "tt_metal/fabric/fabric_init.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/context_types.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/metal_soc_descriptor.hpp"

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_fabric {

std::unique_ptr<FabricBuildSession> begin_fabric_build_session(tt::tt_metal::MetalEnvImpl& env_impl) {
    if (!tt_fabric::is_tt_fabric_config(env_impl.get_fabric_config())) {
        return nullptr;
    }
    auto& fabric_context = env_impl.get_control_plane().get_fabric_context();
    fabric_context.get_builder_context().clear_router_channel_layouts();
    return create_fabric_build_session(fabric_context);
}

std::optional<FabricDeviceBuild> build_fabric_routers(
    tt::tt_metal::MetalEnvImpl& env_impl,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    tt::tt_metal::IDevice* device,
    const FabricBuildSession& session) {
    auto& fabric_context = env_impl.get_control_plane().get_fabric_context();
    FabricDeviceBuild build;
    build.program = std::make_unique<tt::tt_metal::Program>(
        std::make_shared<tt::tt_metal::detail::ProgramImpl>(tt::tt_metal::extract_context_id(device)));

    // The launch's session picks the builder: the registered plugin's, or the default FabricBuilder.
    build.builder = session.make_device_builder(
        device, *build.program, fabric_context, tt::tt_metal::get_core_type_from_config(dispatch_core_config));
    TT_FATAL(build.builder != nullptr, "Fabric build session returned a null FabricBuilder");

    build.builder->discover_channels();
    build.builder->create_routers();
    if (!build.builder->has_routers()) {
        return std::nullopt;
    }
    build.builder->connect_routers();
    build.builder->compile_ancillary_kernels();
    return build;
}

void exchange_fabric_router_channel_layouts(tt::tt_metal::MetalEnvImpl& env_impl) {
    env_impl.get_control_plane().get_fabric_context().get_builder_context().exchange_router_channel_layouts(
        env_impl.full_world_distributed_context());
}

std::unique_ptr<tt::tt_metal::Program> compile_fabric_program(
    tt::tt_metal::MetalEnvImpl& env_impl, tt::tt_metal::IDevice* device, FabricDeviceBuild build) {
    build.builder->create_kernels();
    build.program->impl().compile(
        device, env_impl.get_control_plane().get_fabric_context().get_rtoptions().get_fast_dispatch());
    return std::move(build.program);
}

void configure_fabric_cores(tt::tt_metal::MetalEnvImpl& env_impl, tt::tt_metal::IDevice* device) {
    auto soc_desc = env_impl.get_cluster().get_soc_desc(device->id());
    const auto& control_plane = env_impl.get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto router_chans_and_direction = control_plane.get_active_fabric_eth_channels(fabric_node_id);
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& builder_context = fabric_context.get_builder_context();
    const auto addresses_to_clear = builder_context.get_fabric_router_addresses_to_clear();
    const auto& router_config = builder_context.get_fabric_router_config();
    std::vector<uint32_t> router_zero_buf(router_config.router_buffer_clear_size_words, 0);
    // The router trusts a sender channel's connection semaphore to read "unused" at start, which only the last run's
    // teardown guarantees. A router's channels move with its own sender counts, so start every run from zero.
    std::vector<uint32_t> sender_bookkeeping_zeros(
        (router_config.sender_channels_bookkeeping_end - router_config.sender_channels_bookkeeping_start) /
            sizeof(uint32_t),
        0);
    for (const auto& [router_chan, _] : router_chans_and_direction) {
        auto router_logical_core = soc_desc.get_eth_core_for_channel(router_chan, CoordSystem::LOGICAL);
        for (const auto& address : addresses_to_clear) {
            tt::tt_metal::detail::WriteToDeviceL1(device, router_logical_core, address, router_zero_buf, CoreType::ETH);
        }
        tt::tt_metal::detail::WriteToDeviceL1(
            device,
            router_logical_core,
            router_config.sender_channels_bookkeeping_start,
            sender_bookkeeping_zeros,
            CoreType::ETH);
    }
}

}  // namespace tt::tt_fabric
