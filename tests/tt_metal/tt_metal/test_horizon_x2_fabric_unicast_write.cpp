// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Horizon x2 fabric unicast write test — two separate MeshDevices.
//
// SetFabricConfig(FABRIC_1D) is called before device open so that control_plane writes
// fabric_over_noc_routing_l1_info_t to ROUTING_TABLE_BASE on every Tensix core.  No EDM
// or mux kernels are needed: Horizon uses an always-on AXI crossbar, not ERISC routers.
//
// Test layout:
//   src chip, core (0,0): sender — calls fabric_unicast_noc_unicast_write() to push the
//                          payload (pre-filled by the host) to the target chip's recv_buf.
//   tgt chip, core (0,0): checker — spins until recv_buf[0] != 0, verifies all words,
//                          writes 1u (pass) or 0xDEAD0000|i (fail) to result_buf.
// Host asserts result_buf[0] == 1u.

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <cstdint>
#include <cstdlib>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t FABRIC_UNICAST_2DEV_WORDS = 16;
constexpr uint32_t FABRIC_UNICAST_2DEV_BYTES = FABRIC_UNICAST_2DEV_WORDS * sizeof(uint32_t);

std::shared_ptr<distributed::MeshBuffer> fabric_unicast_2dev_l1_buffer(distributed::MeshDevice& device) {
    const ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange({0, 0}, {0, 0})), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = FABRIC_UNICAST_2DEV_BYTES,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = FABRIC_UNICAST_2DEV_BYTES}, local_cfg, &device);
}

bool fabric_unicast_2dev_available() {
    return std::getenv("TT_METAL_SIMULATOR") != nullptr &&
           MetalContext::instance().get_cluster().number_of_devices() >= 2;
}

class FabricUnicastWrite2DevBothChipsFixture : public QuasarMeshDeviceSingleCardFixture {
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

void expect_fabric_unicast_write_with_checker(
    distributed::MeshDevice& src_device, distributed::MeshDevice& tgt_device, ChipId source, ChipId target) {
    auto& cluster = MetalContext::instance().get_cluster();
    distributed::MeshCommandQueue& src_cq = src_device.mesh_command_queue();
    distributed::MeshCommandQueue& tgt_cq = tgt_device.mesh_command_queue();

    auto src_buf = fabric_unicast_2dev_l1_buffer(src_device);
    auto recv_buf = fabric_unicast_2dev_l1_buffer(tgt_device);
    auto result_buf = fabric_unicast_2dev_l1_buffer(tgt_device);

    const uint32_t base_value = 0xC0DE0000u | (source << 12) | (target << 8);
    std::vector<uint32_t> pattern(FABRIC_UNICAST_2DEV_WORDS), zeros(FABRIC_UNICAST_2DEV_WORDS, 0);
    for (uint32_t i = 0; i < FABRIC_UNICAST_2DEV_WORDS; i++) {
        pattern[i] = base_value | i;
    }
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);

    const CoreCoord tgt_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(target, {0, 0}, CoreType::WORKER);
    cluster.write_core(zeros.data(), FABRIC_UNICAST_2DEV_BYTES, tt_cxy_pair(target, tgt_core), recv_buf->address());
    cluster.write_core(zeros.data(), FABRIC_UNICAST_2DEV_BYTES, tt_cxy_pair(target, tgt_core), result_buf->address());

    // Sender kernel on source chip.
    const experimental::NodeCoord src_node{0, 0};
    const experimental::KernelSpecName SENDER{"fabric_unicast_write_sender"};
    experimental::ProgramSpec sender_spec{
        .name = "horizon_fabric_unicast_write_sender",
        .kernels = {experimental::KernelSpec{
            .unique_id = SENDER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_fabric_unicast_write_sender.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names = {"src_l1_addr", "dst_l1_addr", "dst_noc_x", "dst_noc_y", "dst_dev_id", "size"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = {SENDER}, .target_nodes = src_node}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(src_device, sender_spec);
    experimental::ProgramRunArgs sender_run_args;
    sender_run_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            src_node,
            {{"src_l1_addr", src_buf->address()},
             {"dst_l1_addr", recv_buf->address()},
             {"dst_noc_x", static_cast<uint32_t>(tgt_core.x)},
             {"dst_noc_y", static_cast<uint32_t>(tgt_core.y)},
             {"dst_dev_id", static_cast<uint32_t>(target)},
             {"size", FABRIC_UNICAST_2DEV_BYTES}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, sender_run_args);
    distributed::MeshWorkload sender_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(src_device.shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, sender_workload, true);

    // Checker kernel on target chip.
    const experimental::NodeCoord tgt_node{0, 0};
    const experimental::KernelSpecName CHECKER{"fabric_unicast_write_checker"};
    experimental::ProgramSpec checker_spec{
        .name = "horizon_fabric_unicast_write_checker",
        .kernels = {experimental::KernelSpec{
            .unique_id = CHECKER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_fabric_unicast_write_checker.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"recv_l1_addr", "num_words", "base_value", "result_l1_addr"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = {CHECKER}, .target_nodes = tgt_node}},
    };
    Program checker_prog = experimental::MakeProgramFromSpec(tgt_device, checker_spec);
    experimental::ProgramRunArgs checker_run_args;
    checker_run_args.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = CHECKER,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            tgt_node,
            {{"recv_l1_addr", recv_buf->address()},
             {"num_words", FABRIC_UNICAST_2DEV_WORDS},
             {"base_value", base_value},
             {"result_l1_addr", result_buf->address()}}),
    }};
    experimental::SetProgramRunArgs(checker_prog, checker_run_args);
    distributed::MeshWorkload checker_workload;
    checker_workload.add_program(distributed::MeshCoordinateRange(tgt_device.shape()), std::move(checker_prog));
    distributed::EnqueueMeshWorkload(tgt_cq, checker_workload, true);

    std::vector<uint32_t> result(1, 0);
    cluster.read_core(result.data(), sizeof(uint32_t), tt_cxy_pair(target, tgt_core), result_buf->address());
    EXPECT_EQ(result[0], 1u) << "checker failed on chip " << target << ": 0x" << std::hex << result[0];
}

}  // namespace

TEST_F(FabricUnicastWrite2DevBothChipsFixture, FabricUnicastWrite) {
    if (!fabric_unicast_2dev_available()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_fabric_unicast_write_with_checker(*devices_[0], *devices_[1], 0, 1);
}

TEST_F(FabricUnicastWrite2DevBothChipsFixture, FabricUnicastWriteReverse) {
    if (!fabric_unicast_2dev_available()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_fabric_unicast_write_with_checker(*devices_[1], *devices_[0], 1, 0);
}
