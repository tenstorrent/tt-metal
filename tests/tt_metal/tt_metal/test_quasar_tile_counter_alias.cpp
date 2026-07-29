// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <cstdint>
#include <iostream>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "impl/context/metal_context.hpp"
#include "internal/tt-2xx/quasar/overlay/meta/registers/overlay_reg_defines_core.h"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {
namespace {

constexpr std::uint32_t kTensixOnlyTileCounter = 16;
constexpr std::uint32_t kPostedIncrement = 7;

std::uint32_t read_core_register(IDevice* device, const CoreCoord& logical_core, std::uint32_t address) {
    const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
    return MetalContext::instance().get_cluster().read_core(
        device->id(), virtual_core, address, sizeof(std::uint32_t))[0];
}

void write_core_register(IDevice* device, const CoreCoord& logical_core, std::uint32_t address, std::uint32_t value) {
    const CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);
    MetalContext::instance().get_cluster().write_core_immediate(
        &value, sizeof(value), tt_cxy_pair(device->id(), virtual_core), address);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixTileCounter16DoesNotAliasOverlayTileCounter0) {
    if (!MetalContext::instance().rtoptions().is_simulator_or_emulated()) {
        GTEST_SKIP() << "This low-level RTL regression test only runs in simulation or emulation.";
    }

    auto mesh_device = devices_[0];
    IDevice* device = mesh_device->get_devices()[0];
    constexpr CoreCoord logical_core{0, 0};

    // TC16 is Tensix-only, so updating it must not change any of the 16 overlay counters.
    write_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__RESET_REG_ADDR, 1);
    ASSERT_EQ(
        read_core_register(
            device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR),
        0u);

    const experimental::NodeCoord node{0, 0};
    const experimental::KernelSpecName kernel_name{"tile_counter_overlay_alias"};
    experimental::KernelSpec kernel{
        .unique_id = kernel_name,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/tile_counter_overlay_alias.cpp",
        .num_threads = 1,
        .hw_config = experimental::ComputeHardwareConfig{},
    };
    experimental::ProgramSpec spec{
        .name = "tile_counter_overlay_alias",
        .kernels = {kernel},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main",
            .kernels = {kernel_name},
            .target_nodes = node,
        }},
    };

    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    experimental::ProgramRunArgs run_args;
    run_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{.kernel = kernel_name}};
    experimental::SetProgramRunArgs(program, run_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, true);

    const Hal& hal = MetalContext::instance().hal();
    const std::uint32_t tensix_tc16_tiles_available_addr = hal.get_neo_tile_counters_base_addr() +
                                                           kTensixOnlyTileCounter * hal.get_neo_tile_counters_size() +
                                                           hal.get_neo_tile_counters_tiles_available_offset();
    const std::uint32_t tensix_tc16_tiles_available =
        read_core_register(device, logical_core, tensix_tc16_tiles_available_addr);
    const std::uint32_t overlay_tc0_posted = read_core_register(
        device, logical_core, TT_OVERLAY_LLK_TILE_COUNTERS_TT_LLK_INTERFACE_TILE_COUNTERS_0__READ_POSTED_REG_ADDR);

    std::cout << "TC alias repro: T6 TC16 tiles_available=" << tensix_tc16_tiles_available
              << ", overlay TC0 posted=" << overlay_tc0_posted << std::endl;

    ASSERT_EQ(tensix_tc16_tiles_available, kPostedIncrement);
    EXPECT_EQ(overlay_tc0_posted, 0u)
        << "T6 TC16 incorrectly reached overlay TC0 (the 5-bit selector was truncated to 4 bits)";
}

}  // namespace
}  // namespace tt::tt_metal
