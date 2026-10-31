// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Horizon x2 fabric unicast write test.
//
// Requires a horizon_x2 emulator build (TT_METAL_SIMULATOR set to an emu-horizon-x2-* binary)
// with both chips visible.  SetFabricConfig(FABRIC_1D) populates fabric_over_noc_routing_l1_info_t on
// every chip's Tensix cores via the flat addresses from get_flat_address().
//
// Test layout:
//   Chip 0, core (0,0): sender kernel — fills a buffer with a pattern and calls
//                        fabric_unicast_noc_unicast_write() to push it to chip 1.
//   Chip 1, core (0,0): checker kernel — spins until data arrives, verifies the pattern,
//                        writes pass (1) or fail (0xDEAD0000|i) to a result word.
//
// The test verifies that:
//   (a) SetFabricConfig correctly populates the flat-address routing table on both chips.
//   (b) fabric_unicast_noc_unicast_write (mesh/experimental API) reaches the correct remote L1.
//   (c) Multi-word payloads are transferred correctly.

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"
#include <umd/device/tt_device/tt_device.hpp>

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
// populates fabric_over_noc_routing_l1_info_t during device open.

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

// One-word (4-byte) L1 buffer, sharded to the given core (default (0,0)).
std::shared_ptr<distributed::MeshBuffer> make_word_buffer(distributed::MeshDevice& dev, CoreCoord core = {0, 0}) {
    const ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange(core, core)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = sizeof(uint32_t),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = sizeof(uint32_t)}, local_cfg, &dev);
}

// N-word L1 buffer, sharded to the given core (default (0,0)).  Each word is a separate page.
std::shared_ptr<distributed::MeshBuffer> make_n_word_buffer(
    distributed::MeshDevice& dev, uint32_t n, CoreCoord core = {0, 0}) {
    const ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange(core, core)), {n, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {n, 1});
    const distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = sizeof(uint32_t),
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = n * sizeof(uint32_t)}, local_cfg, &dev);
}

void run_fabric_unicast_write_test(
    distributed::MeshDevice* chip0,
    distributed::MeshDevice* chip1,
    tt::Cluster& cluster,
    uint32_t num_words,
    uint32_t base_value,
    ChipId chip1_id = 1,
    CoreCoord checker_core = {0, 0}) {
    // ------------------------------------------------------------------
    // Allocate L1 buffers.
    //   src_buf   — chip 0, holds the outgoing payload
    //   recv_buf  — chip 1, destination for the fabric write
    //   result_buf — chip 1, one word: 1 = pass, 0xDEAD0000|i = fail at word i
    // ------------------------------------------------------------------
    auto src_buf = make_n_word_buffer(*chip0, num_words);
    auto recv_buf = make_n_word_buffer(*chip1, num_words, checker_core);
    auto result_buf = make_word_buffer(*chip1, checker_core);

    // Zero-initialise receive and result buffers so kernels can spin-detect.
    distributed::EnqueueWriteMeshBuffer(
        chip1->mesh_command_queue(), recv_buf, std::vector<uint32_t>(num_words, 0), true);
    distributed::EnqueueWriteMeshBuffer(chip1->mesh_command_queue(), result_buf, std::vector<uint32_t>{0u}, true);

    // ------------------------------------------------------------------
    // Resolve NOC0 coordinates for core (0,0) on chip 1.
    // ------------------------------------------------------------------
    const CoreCoord chip1_noc0 =
        cluster.get_virtual_coordinate_from_logical_coordinates(chip1_id, checker_core, CoreType::WORKER);

    std::cout << "[UNICAST_WRITE] chip1 core(0,0) NOC0=(" << chip1_noc0.x << "," << chip1_noc0.y << ")" << std::endl;

    const experimental::NodeCoord node{0, 0};

    // ------------------------------------------------------------------
    // Checker kernel (chip 1) — dispatch non-blocking.
    // ------------------------------------------------------------------
    const experimental::KernelSpecName CHECKER_K{"unicast_checker"};
    experimental::ProgramSpec checker_spec{
        .name = "horizon_unicast_checker",
        .kernels = {experimental::KernelSpec{
            .unique_id = CHECKER_K,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_fabric_unicast_write_checker.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"recv_l1_addr", "num_words", "base_value", "result_l1_addr"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main", .kernels = {CHECKER_K}, .target_nodes = checker_core}},
    };
    Program checker_prog = experimental::MakeProgramFromSpec(*chip1, checker_spec);
    experimental::ProgramRunArgs checker_args;
    checker_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = CHECKER_K,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            checker_core,
            {
                {"recv_l1_addr", recv_buf->address()},
                {"num_words", num_words},
                {"base_value", base_value},
                {"result_l1_addr", result_buf->address()},
            }),
    }};
    experimental::SetProgramRunArgs(checker_prog, checker_args);

    distributed::MeshWorkload checker_wl;
    checker_wl.add_program(distributed::MeshCoordinateRange(chip1->shape()), std::move(checker_prog));

    // ------------------------------------------------------------------
    // Sender kernel (chip 0) — dispatch non-blocking.
    // ------------------------------------------------------------------
    const experimental::KernelSpecName SENDER_K{"unicast_sender"};
    experimental::ProgramSpec sender_spec{
        .name = "horizon_unicast_sender",
        .kernels = {experimental::KernelSpec{
            .unique_id = SENDER_K,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_fabric_unicast_write_sender.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"num_words", "src_l1_addr", "dst_l1_addr", "dst_noc_x", "dst_noc_y", "dst_dev_id", "base_value"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = {SENDER_K}, .target_nodes = node}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(*chip0, sender_spec);
    experimental::ProgramRunArgs sender_args;
    sender_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER_K,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {
                {"num_words", num_words},
                {"src_l1_addr", src_buf->address()},
                {"dst_l1_addr", recv_buf->address()},
                {"dst_noc_x", static_cast<uint32_t>(chip1_noc0.x)},
                {"dst_noc_y", static_cast<uint32_t>(chip1_noc0.y)},
                {"dst_dev_id", static_cast<uint32_t>(chip1_id)},
                {"base_value", base_value},
            }),
    }};
    experimental::SetProgramRunArgs(sender_prog, sender_args);

    distributed::MeshWorkload sender_wl;
    sender_wl.add_program(distributed::MeshCoordinateRange(chip0->shape()), std::move(sender_prog));

    // Single queue (chip0==chip1): dispatch sender first so data exists when checker runs.
    if (chip0 == chip1) {
        std::cout << "[UNICAST_WRITE] single-device: dispatching sender first, then checker..." << std::endl;
        distributed::EnqueueMeshWorkload(chip0->mesh_command_queue(), sender_wl, false);
        distributed::EnqueueMeshWorkload(chip1->mesh_command_queue(), checker_wl, false);
    } else {
        std::cout << "[UNICAST_WRITE] dispatching checker on chip 1, sender on chip 0..." << std::endl;
        distributed::EnqueueMeshWorkload(chip1->mesh_command_queue(), checker_wl, false);
        distributed::EnqueueMeshWorkload(chip0->mesh_command_queue(), sender_wl, false);
    }

    // ------------------------------------------------------------------
    // Wait: sender first (ensures noc_async_write_barrier ran), then checker.
    // ------------------------------------------------------------------
    std::cout << "[UNICAST_WRITE] Finish(chip0)..." << std::endl;
    distributed::Finish(chip0->mesh_command_queue());
    std::cout << "[UNICAST_WRITE] Finish(chip1)..." << std::endl;
    distributed::Finish(chip1->mesh_command_queue());

    // ------------------------------------------------------------------
    // Verify.
    // ------------------------------------------------------------------
    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(chip1->mesh_command_queue(), result, result_buf, true);

    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result[0], 1u) << "unicast write check failed: result=0x" << std::hex << result[0];

    std::cout << "[UNICAST_WRITE] result=0x" << std::hex << result[0] << " — test "
              << (result[0] == 1u ? "PASSED" : "FAILED") << std::endl;
}

bool horizon_ttsim_available() {
    return std::getenv("TT_METAL_SIMULATOR") != nullptr &&
           MetalContext::instance().get_cluster().number_of_devices() >= 1;
}

// Opens MMIO device 0 only. No SetFabricConfig — routing table bypassed by TT_METAL_TTSIM.
class HorizonTtsimFabricFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void create_devices() override { AnyDispatchMeshDeviceSingleCardFixture::create_devices({0}); }
};

}  // namespace

TEST_F(HorizonX2FabricFixture, FabricUnicastWrite_SingleWord) {
    if (!horizon_x2_available()) {
        GTEST_SKIP() << "needs a horizon_x2 emulator build with both chips visible";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    run_fabric_unicast_write_test(devices_[0].get(), devices_[1].get(), cluster, 1, 0xC0FFEE00);
}

TEST_F(HorizonX2FabricFixture, FabricUnicastWrite_MultiWord) {
    if (!horizon_x2_available()) {
        GTEST_SKIP() << "needs a horizon_x2 emulator build with both chips visible";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    run_fabric_unicast_write_test(devices_[0].get(), devices_[1].get(), cluster, 16, 0xABCD0000);
}

TEST_F(HorizonTtsimFabricFixture, TtsimUnicastWrite_SingleWord) {
    if (!horizon_ttsim_available()) {
        GTEST_SKIP() << "set TT_METAL_SIMULATOR=libttsim_qsr.so to enable";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    // chip0 == chip1: both point to device 0. Checker on core (1,0), sender on (0,0).
    run_fabric_unicast_write_test(
        devices_[0].get(),
        devices_[0].get(),
        cluster,
        1,
        0xC0FFEE00,
        /*chip1_id=*/0,
        /*checker_core=*/{1, 0});
}

TEST_F(HorizonTtsimFabricFixture, TtsimUnicastWrite_MultiWord) {
    if (!horizon_ttsim_available()) {
        GTEST_SKIP() << "set TT_METAL_SIMULATOR=libttsim_qsr.so to enable";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    run_fabric_unicast_write_test(
        devices_[0].get(),
        devices_[0].get(),
        cluster,
        16,
        0xABCD0000,
        /*chip1_id=*/0,
        /*checker_core=*/{1, 0});
}
