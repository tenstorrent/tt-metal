// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Horizon x2 fabric loopback test.
//
// Requires a horizon_x2 emulator build (TT_METAL_SIMULATOR set to an emu-horizon-x2-* binary)
// with both chips visible.  SetFabricConfig(FABRIC_1D) populates horizon_routing_l1_info_t on
// every chip's Tensix cores via the flat addresses from get_flat_address().
//
// Test layout:
//   Chip 0, core (0,0): sender kernel — writes a value to chip 1, then spins for the echo.
//   Chip 1, core (0,0): loopback kernel — spins until data arrives, writes it back to chip 0.
//
// The round-trip verifies that:
//   (a) SetFabricConfig correctly populates the flat-address routing table on both chips.
//   (b) noc_async_write with the looked-up flat address reaches the correct remote L1.

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Opens chips 0 and 1 with Fabric 1D initialised.
// Fabric config must be set before device creation so that control_plane
// populates horizon_routing_l1_info_t during device open.
class HorizonX2FabricFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void create_devices() override { AnyDispatchMeshDeviceSingleCardFixture::create_devices({0, 1}); }

    void SetUp() override {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
        QuasarMeshDeviceSingleCardFixture::SetUp();
    }

    void TearDown() override {
        QuasarMeshDeviceSingleCardFixture::TearDown();
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
    }
};

bool horizon_x2_available() {
    return std::getenv("TT_METAL_SIMULATOR") != nullptr &&
           MetalContext::instance().get_cluster().number_of_devices() >= 2;
}

// One-word (4-byte) L1 buffer, sharded to core (0,0).
std::shared_ptr<distributed::MeshBuffer> make_word_buffer(distributed::MeshDevice& dev) {
    const ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange({0, 0}, {0, 0})), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = sizeof(uint32_t),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = sizeof(uint32_t)}, local_cfg, &dev);
}

}  // namespace

// Chip 0 sends a value to chip 1; chip 1 echoes it back; chip 0 verifies the round-trip.
TEST_F(HorizonX2FabricFixture, FabricLoopback) {
    if (!horizon_x2_available()) {
        GTEST_SKIP() << "needs a horizon_x2 emulator build with both chips visible";
    }

    auto& cluster = MetalContext::instance().get_cluster();
    auto chip0 = devices_[0];  // chip 0 — sender
    auto chip1 = devices_[1];  // chip 1 — loopback

    // ------------------------------------------------------------------
    // Allocate L1 buffers (one word each, core (0,0)).
    //   send_buf  — chip 0, holds the outgoing value
    //   echo_buf  — chip 0, receives the loopback echo
    //   recv_buf  — chip 1, receives the initial write from chip 0
    // ------------------------------------------------------------------
    auto send_buf = make_word_buffer(*chip0);
    auto echo_buf = make_word_buffer(*chip0);
    auto recv_buf = make_word_buffer(*chip1);

    const uint32_t SEND_VALUE = 0xC0FFEE;

    // Initialise: echo and recv buffers to 0 so the kernels can spin-detect the write.
    distributed::EnqueueWriteMeshBuffer(chip0->mesh_command_queue(), echo_buf, {0u}, true);
    distributed::EnqueueWriteMeshBuffer(chip1->mesh_command_queue(), recv_buf, {0u}, true);

    // ------------------------------------------------------------------
    // Resolve NOC0 coordinates for core (0,0) on each chip.
    // The routing table is indexed by NOC0 coords, so we pass those to the kernels.
    // ------------------------------------------------------------------
    const CoreCoord chip0_noc0 = cluster.get_virtual_coordinate_from_logical_coordinates(0, {0, 0}, CoreType::WORKER);
    const CoreCoord chip1_noc0 = cluster.get_virtual_coordinate_from_logical_coordinates(1, {0, 0}, CoreType::WORKER);

    std::cout << "[LOOPBACK] chip0 core(0,0) NOC0=(" << chip0_noc0.x << "," << chip0_noc0.y
              << ")  chip1 core(0,0) NOC0=(" << chip1_noc0.x << "," << chip1_noc0.y << ")" << std::endl;

    // ------------------------------------------------------------------
    // Build programs.
    // ------------------------------------------------------------------
    const experimental::NodeCoord node{0, 0};

    // -- Loopback kernel (chip 1) --
    const experimental::KernelSpecName RECV_K{"loopback_recv"};
    experimental::ProgramSpec recv_spec{
        .name = "horizon_loopback_recv",
        .kernels = {experimental::KernelSpec{
            .unique_id = RECV_K,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_fabric_loopback_receiver.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names = {"recv_addr", "echo_l1_addr", "echo_noc_x", "echo_noc_y", "src_dev_id"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = {RECV_K}, .target_nodes = node}},
    };
    Program recv_prog = experimental::MakeProgramFromSpec(*chip1, recv_spec);
    experimental::ProgramRunArgs recv_args;
    recv_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = RECV_K,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {
                {"recv_addr", recv_buf->address()},
                {"echo_l1_addr", echo_buf->address()},
                {"echo_noc_x", static_cast<uint32_t>(chip0_noc0.x)},
                {"echo_noc_y", static_cast<uint32_t>(chip0_noc0.y)},
                {"src_dev_id", 0u},  // chip 0 is device 0
            }),
    }};
    experimental::SetProgramRunArgs(recv_prog, recv_args);

    // -- Sender kernel (chip 0) --
    const experimental::KernelSpecName SEND_K{"loopback_send"};
    experimental::ProgramSpec send_spec{
        .name = "horizon_loopback_send",
        .kernels = {experimental::KernelSpec{
            .unique_id = SEND_K,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_fabric_loopback_sender.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"value", "src_addr", "echo_addr", "dst_l1_addr", "dst_noc_x", "dst_noc_y", "dst_dev_id"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = {SEND_K}, .target_nodes = node}},
    };
    Program send_prog = experimental::MakeProgramFromSpec(*chip0, send_spec);
    experimental::ProgramRunArgs send_args;
    send_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = SEND_K,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {
                {"value", SEND_VALUE},
                {"src_addr", send_buf->address()},
                {"echo_addr", echo_buf->address()},
                {"dst_l1_addr", recv_buf->address()},
                {"dst_noc_x", static_cast<uint32_t>(chip1_noc0.x)},
                {"dst_noc_y", static_cast<uint32_t>(chip1_noc0.y)},
                {"dst_dev_id", 1u},  // chip 1 is device 1
            }),
    }};
    experimental::SetProgramRunArgs(send_prog, send_args);

    // ------------------------------------------------------------------
    // Dispatch: loopback first (non-blocking), then sender (blocking).
    // The sender spins until the echo arrives, so it won't exit until
    // chip 1 has completed the round-trip.
    // ------------------------------------------------------------------
    distributed::MeshWorkload recv_wl, send_wl;
    recv_wl.add_program(distributed::MeshCoordinateRange(chip1->shape()), std::move(recv_prog));
    send_wl.add_program(distributed::MeshCoordinateRange(chip0->shape()), std::move(send_prog));

    std::cout << "[LOOPBACK] dispatching loopback kernel on chip 1..." << std::endl;
    distributed::EnqueueMeshWorkload(chip1->mesh_command_queue(), recv_wl, false);

    std::cout << "[LOOPBACK] dispatching sender kernel on chip 0..." << std::endl;
    distributed::EnqueueMeshWorkload(chip0->mesh_command_queue(), send_wl, true);  // blocking

    std::cout << "[LOOPBACK] sender done — reading echo buffer from chip 0..." << std::endl;

    // ------------------------------------------------------------------
    // Verify: the echo buffer on chip 0 should hold SEND_VALUE.
    // ------------------------------------------------------------------
    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(chip0->mesh_command_queue(), result, echo_buf, true);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], SEND_VALUE) << "loopback mismatch: got 0x" << std::hex << result[0] << " expected 0x"
                                     << SEND_VALUE;

    std::cout << "[LOOPBACK] echo value 0x" << std::hex << result[0] << " — test "
              << (result[0] == SEND_VALUE ? "PASSED" : "FAILED") << std::endl;
}
