// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "tt_metal/fabric/fabric_init.hpp"

#include "tt_metal.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_builder.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"

// hack for test_basic_fabric_apis.cpp
// https://github.com/tenstorrent/tt-metal/issues/20000
// TODO: delete this once tt_fabric_api.h fully support low latency feature
extern "C" bool isFabricUnitTest() __attribute__((weak));
bool isFabricUnitTest() { return false; }

namespace tt::tt_fabric {

// FabricBuildBarrier — shared state for the internal barrier in compile_fabric.
// Defined here because FabricBuildPhase1Result contains unique_ptr<FabricBuilder>
// which requires the complete type (available in this TU via fabric_builder.hpp).
FabricBuildBarrier::FabricBuildBarrier(const std::vector<tt::tt_metal::IDevice*>& devices) :
    all_devices(devices), phase1_results(devices.size()) {}
FabricBuildBarrier::~FabricBuildBarrier() = default;

namespace {

FabricBuildPhase1Result fabric_build_phase1(tt::tt_metal::IDevice* device) {
    auto fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();

    auto builder = std::make_unique<FabricBuilder>(device, *fabric_program_ptr, fabric_context);

    builder->discover_channels();
    builder->create_routers();

    if (!builder->has_routers()) {
        return FabricBuildPhase1Result{nullptr, nullptr, false};
    }

    return FabricBuildPhase1Result{std::move(builder), std::move(fabric_program_ptr), true};
}

std::unique_ptr<tt::tt_metal::Program> fabric_build_phase2(
    tt::tt_metal::IDevice* device, FabricBuildPhase1Result phase1_result) {
    if (!phase1_result.has_routers || !phase1_result.builder) {
        return nullptr;
    }

    auto& builder = *phase1_result.builder;
    auto& program = *phase1_result.program;

    // Update remote allocators from peer state (published during the barrier between phases)
    builder.update_remote_allocators();

    builder.connect_routers();
    builder.compile_ancillary_kernels();
    builder.create_kernels();

    tt::tt_metal::detail::CompileProgram(
        device, program, tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch());

    return std::move(phase1_result.program);
}

// Single-phase path: full build (used by TERMINATE_FABRIC and when all_devices is empty)
std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program_single_phase(
    tt::tt_metal::IDevice* device) {
    auto fabric_program_ptr = std::make_unique<tt::tt_metal::Program>();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();

    FabricBuilder builder(device, *fabric_program_ptr, fabric_context);

    builder.discover_channels();
    builder.create_routers();
    if (!builder.has_routers()) {
        return nullptr;
    }

    builder.connect_routers();
    builder.compile_ancillary_kernels();
    builder.create_kernels();

    tt::tt_metal::detail::CompileProgram(
        device, *fabric_program_ptr, tt::tt_metal::MetalContext::instance().rtoptions().get_fast_dispatch());

    return fabric_program_ptr;
}

// Two-phase path: uses barrier to coordinate with other devices.
// Each thread calls this for its own device. The barrier ensures all devices
// complete phase 1 before any proceeds to phase 2.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_tt_fabric_program_two_phase(
    tt::tt_metal::IDevice* device, const std::vector<tt::tt_metal::IDevice*>& all_devices) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto& fabric_context = control_plane.get_fabric_context();
    auto& builder_context = fabric_context.get_builder_context();

    // Find this device's index
    size_t device_index = 0;
    for (size_t i = 0; i < all_devices.size(); ++i) {
        if (all_devices[i]->id() == device->id()) {
            device_index = i;
            break;
        }
    }

    auto& barrier = builder_context.get_or_create_build_barrier(all_devices);

    // Phase 1: discover channels, create routers with servicing-aware allocators
    barrier.phase1_results[device_index] = fabric_build_phase1(device);

    // Signal phase 1 completion and wait for all devices
    size_t completed = barrier.phase1_count.fetch_add(1) + 1;
    if (completed == all_devices.size()) {
        // Last thread: collect and publish allocator state from all builders
        for (auto& result : barrier.phase1_results) {
            if (result.builder) {
                auto states = result.builder->collect_published_allocator_state();
                for (auto& [key, state] : states) {
                    builder_context.publish_allocator_state(key.first, key.second, std::move(state));
                }
            }
        }

        // TODO: MPI exchange for multi-host (Galaxy) — exchange published allocator state
        // with other hosts using DistributedContext::broadcast() following the pattern
        // in control_plane.cpp::collect_and_merge_router_port_directions_from_all_hosts().
        // For now, single-host mode works because all devices are local.

        // Release all waiting threads
        {
            std::lock_guard<std::mutex> lock(barrier.barrier_mutex);
            barrier.phase1_published = true;
        }
        barrier.barrier_cv.notify_all();
    } else {
        // Wait for the last thread to finish publishing
        std::unique_lock<std::mutex> lock(barrier.barrier_mutex);
        barrier.barrier_cv.wait(lock, [&] { return barrier.phase1_published; });
    }

    // Phase 2: update remote allocators from peer state, connect, compile
    return fabric_build_phase2(device, std::move(barrier.phase1_results[device_index]));
}

}  // anonymous namespace

std::unique_ptr<tt::tt_metal::Program> create_and_compile_fabric_program(
    tt::tt_metal::IDevice* device, const std::vector<tt::tt_metal::IDevice*>& all_devices) {
    auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (!tt_fabric::is_tt_fabric_config(fabric_config)) {
        return nullptr;
    }

    if (all_devices.empty()) {
        return create_and_compile_tt_fabric_program_single_phase(device);
    }
    return create_and_compile_tt_fabric_program_two_phase(device, all_devices);
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
