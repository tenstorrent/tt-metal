// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LOCAL horizon_x2 spike (not for upstream): tests flat-address writes between two chips,
// each opened as a separate MeshDevice. A sender kernel on the source chip writes via flat
// address to the target chip's L1; a checker kernel on the target spins until the data arrives,
// verifies all words, and writes pass/fail to a result buffer. Host asserts result == 1u.

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <umd/device/tt_device/tt_device.hpp>

#include <cstdint>
#include <cstdlib>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t FLAT_WRITE_2DEV_WORDS = 16;
constexpr uint32_t FLAT_WRITE_2DEV_BYTES = FLAT_WRITE_2DEV_WORDS * sizeof(uint32_t);

std::shared_ptr<distributed::MeshBuffer> flat_write_2dev_l1_buffer(distributed::MeshDevice& device) {
    const ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange({0, 0}, {0, 0})), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = FLAT_WRITE_2DEV_BYTES,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    return distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = FLAT_WRITE_2DEV_BYTES}, local_cfg, &device);
}

bool flat_write_2dev_two_chips() {
    return std::getenv("TT_METAL_SIMULATOR") != nullptr &&
           MetalContext::instance().get_cluster().number_of_devices() >= 2;
}

// Opens chip 0 and chip 1 as separate unit MeshDevices.
// After SetUp: devices_[0] = chip 0, devices_[1] = chip 1.
class FlatWrite2DevBothChipsFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void create_devices() override { AnyDispatchMeshDeviceSingleCardFixture::create_devices({0, 1}); }
};

// Sends FLAT_WRITE_2DEV_WORDS words from src_device (chip @p source) to tgt_device (chip @p target)
// via flat address, then runs a checker kernel on the target to verify. Asserts checker result == 1u.
void expect_flat_write_2dev_with_checker(
    distributed::MeshDevice& src_device,
    distributed::MeshDevice& tgt_device,
    ChipId source,
    ChipId target,
    size_t access_point_index,
    bool posted) {
    auto& cluster = MetalContext::instance().get_cluster();
    distributed::MeshCommandQueue& src_cq = src_device.mesh_command_queue();
    distributed::MeshCommandQueue& tgt_cq = tgt_device.mesh_command_queue();

    auto src_buf = flat_write_2dev_l1_buffer(src_device);
    auto recv_buf = flat_write_2dev_l1_buffer(tgt_device);
    auto result_buf = flat_write_2dev_l1_buffer(tgt_device);

    const uint32_t base_value = 0xC0DE0000u | (source << 12) | (target << 8);
    std::vector<uint32_t> pattern(FLAT_WRITE_2DEV_WORDS), zeros(FLAT_WRITE_2DEV_WORDS, 0);
    for (uint32_t i = 0; i < FLAT_WRITE_2DEV_WORDS; i++) {
        pattern[i] = base_value | i;
    }
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);

    const CoreCoord tgt_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(target, {0, 0}, CoreType::WORKER);
    cluster.write_core(zeros.data(), FLAT_WRITE_2DEV_BYTES, tt_cxy_pair(target, tgt_core), recv_buf->address());
    cluster.write_core(zeros.data(), FLAT_WRITE_2DEV_BYTES, tt_cxy_pair(target, tgt_core), result_buf->address());

    tt::umd::TTDevice* target_tt_device = cluster.get_driver()->get_tt_device(target);
    const std::vector<tt::umd::AccessPointId> access_points = target_tt_device->get_access_points();
    ASSERT_GT(access_points.size(), access_point_index);
    const std::optional<uint64_t> flat = target_tt_device->get_flat_address(
        access_points[access_point_index],
        tt::umd::CoreCoord(0, 0, CoreType::TENSIX, CoordSystem::LITERAL),
        tt::umd::WindowFlags::UnicastWrite,
        tt::umd::IoOrdering::Posted,
        recv_buf->address());
    ASSERT_TRUE(flat.has_value()) << "chip " << target << " has no flat address";

    // Sender kernel on source chip.
    const experimental::NodeCoord src_node{0, 0};
    const experimental::KernelSpecName SENDER{"flat_write_2dev"};
    experimental::ProgramSpec sender_spec{
        .name = "flat_address_write_2_device",
        .kernels = {experimental::KernelSpec{
            .unique_id = SENDER,
            .source =
                OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/flat_address_write_2_device.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"src", "dst_lo", "dst_hi", "size", "posted"}},
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
            {{"src", src_buf->address()},
             {"dst_lo", static_cast<uint32_t>(*flat)},
             {"dst_hi", static_cast<uint32_t>(*flat >> 32)},
             {"size", FLAT_WRITE_2DEV_BYTES},
             {"posted", posted ? 1u : 0u}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, sender_run_args);
    distributed::MeshWorkload sender_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(src_device.shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, sender_workload, true);

    // Checker kernel on target chip — spins until first word arrives, verifies all words, writes 1u/fail.
    const experimental::NodeCoord tgt_node{0, 0};
    const experimental::KernelSpecName CHECKER{"flat_write_2dev_checker"};
    experimental::ProgramSpec checker_spec{
        .name = "flat_address_write_2_device_checker",
        .kernels = {experimental::KernelSpec{
            .unique_id = CHECKER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/flat_address_write_2_device_checker.cpp",
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
             {"num_words", FLAT_WRITE_2DEV_WORDS},
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

TEST_F(FlatWrite2DevBothChipsFixture, FlatWrite2DevPosted) {
    if (!flat_write_2dev_two_chips()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_flat_write_2dev_with_checker(*devices_[0], *devices_[1], 0, 1, 0, true);
}

TEST_F(FlatWrite2DevBothChipsFixture, FlatWrite2DevNonPosted) {
    if (!flat_write_2dev_two_chips()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_flat_write_2dev_with_checker(*devices_[0], *devices_[1], 0, 1, 0, false);
}

// Chip 0's second access point, NOC2AXI 1 at 0x11_...: chip 1's kernels can't use 0x10_...,
// which is tt-metal's own REMOTE window.
TEST_F(FlatWrite2DevBothChipsFixture, FlatWrite2DevReversePosted) {
    if (!flat_write_2dev_two_chips()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_flat_write_2dev_with_checker(*devices_[1], *devices_[0], 1, 0, 1, true);
}
