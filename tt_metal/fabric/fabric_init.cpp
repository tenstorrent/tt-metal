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
#include "distributed_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <string>

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
    if (env_impl.get_control_plane().fabric_uses_flat_addressing()) {
        // Horizon uses NOC crossbar for inter-chip communication; no ERISC fabric firmware needed.
        return std::nullopt;
    }
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

void check_fabric_launch_headers(const std::vector<RouterChannelLayouts::LaunchHeader>& headers) {
    std::string failed, launches;
    bool mismatch = false;
    for (size_t rank = 0; rank < headers.size(); ++rank) {
        const auto& header = headers[rank];
        if (header.status != 0) {
            failed += fmt::format("{}{}", failed.empty() ? "" : ", ", rank);
        }
        mismatch |= header.launch_generation != headers[0].launch_generation;
        launches += fmt::format(" rank {}: config {} launch {};", rank, header.fabric_config, header.launch_generation);
    }
    TT_FATAL(failed.empty(), "fabric launch: rank {} failed the phase-1 build (see its log)", failed);
    TT_FATAL(!mismatch, "fabric launch: ranks disagree on the launch:{}", launches);
}

void exchange_fabric_router_channel_layouts(tt::tt_metal::MetalEnvImpl& env_impl, std::exception_ptr phase1_error) {
    const RouterChannelLayouts::LaunchHeader header{
        .status = phase1_error ? 1u : 0u, .fabric_config = static_cast<uint32_t>(env_impl.get_fabric_config())};
    const auto& context = env_impl.full_world_distributed_context();
    if (phase1_error) {
        if (*context.size() == 1) {
            std::rethrow_exception(phase1_error);
        }
        try {
            std::rethrow_exception(phase1_error);
        } catch (const std::exception& e) {
            log_error(tt::LogFabric, "fabric launch: phase-1 build failed: {}", e.what());
        } catch (...) {
            log_error(tt::LogFabric, "fabric launch: phase-1 build failed with an unknown exception");
        }
    }
    env_impl.get_control_plane().get_fabric_context().get_builder_context().exchange_router_channel_layouts(
        context, header, check_fabric_launch_headers);
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
    if (soc_desc.arch == tt::ARCH::QUASAR &&
        soc_desc.get_arch_descriptor().get_ip_variant() == tt::IpVariant::HORIZON) {
        return;  // Horizon uses crossbar; no ERISC cores to configure.
    }
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
