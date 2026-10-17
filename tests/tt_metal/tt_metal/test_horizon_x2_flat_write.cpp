// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LOCAL horizon_x2 spike (not for upstream): a kernel on one chip writes to another chip's Tensix
// L1 through that chip's flat address (UMD TTDevice::get_flat_address), and the host checks that it
// landed there and not on the writer's own chip. Needs a horizon_x2 run dir with both chips visible
// (TT_VISIBLE_DEVICES unset), TT_UMD_NOC_ATT=horizon_x2, TT_METAL_NOC_ATT=horizon_2x3, and ATT
// tables with the cross-chip tile routes (tt-umd-simulators emu/horizon_x2/horizon_x2_att.yaml).

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <umd/device/tt_device/tt_device.hpp>

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

constexpr uint32_t FLAT_WRITE_WORDS = 16;
constexpr uint32_t FLAT_WRITE_BYTES = FLAT_WRITE_WORDS * sizeof(uint32_t);

std::shared_ptr<distributed::MeshBuffer> flat_write_l1_buffer(distributed::MeshDevice& device) {
    const ShardSpecBuffer shard_spec(CoreRangeSet(CoreRange({0, 0}, {0, 0})), {1, 1}, ShardOrientation::ROW_MAJOR,
                                     {1, 1}, {1, 1});
    const distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = FLAT_WRITE_BYTES,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
    };
    return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = FLAT_WRITE_BYTES}, local_cfg, &device);
}

// A kernel on @p device's (chip @p source) Tensix (0,0) writes a pattern to Tensix (0,0) of chip
// @p target, at the offset of a guard buffer, through the target's access point @p access_point_index.
void expect_flat_write_lands(
    distributed::MeshDevice& device, ChipId source, ChipId target, size_t access_point_index, bool posted) {
    auto& cluster = MetalContext::instance().get_cluster();
    distributed::MeshCommandQueue& cq = device.mesh_command_queue();

    auto src = flat_write_l1_buffer(device);
    auto guard = flat_write_l1_buffer(device);
    std::vector<uint32_t> pattern(FLAT_WRITE_WORDS), zeros(FLAT_WRITE_WORDS, 0), before(FLAT_WRITE_WORDS, 0xDEADBEEF);
    for (uint32_t i = 0; i < FLAT_WRITE_WORDS; i++) {
        pattern[i] = 0xC0DE0000 | (source << 12) | (target << 8) | i;
    }
    distributed::EnqueueWriteMeshBuffer(cq, src, pattern, true);
    distributed::EnqueueWriteMeshBuffer(cq, guard, zeros, true);

    const uint64_t offset = guard->address();
    const CoreCoord target_core = cluster.get_virtual_coordinate_from_logical_coordinates(target, {0, 0}, CoreType::WORKER);
    cluster.write_core(before.data(), FLAT_WRITE_BYTES, tt_cxy_pair(target, target_core), offset);
    std::vector<uint32_t> readback(FLAT_WRITE_WORDS);
    cluster.read_core(readback.data(), FLAT_WRITE_BYTES, tt_cxy_pair(target, target_core), offset);
    ASSERT_EQ(readback, before) << "host can't pre-clear chip " << target;

    tt::umd::TTDevice* target_device = cluster.get_driver()->get_tt_device(target);
    const std::vector<tt::umd::AccessPointId> access_points = target_device->get_access_points();
    ASSERT_GT(access_points.size(), access_point_index);
    const std::optional<uint64_t> flat = target_device->get_flat_address(
        access_points[access_point_index],
        tt::umd::CoreCoord(0, 0, CoreType::TENSIX, CoordSystem::LITERAL),
        tt::umd::WindowFlags::UnicastWrite,
        tt::umd::IoOrdering::Posted,
        offset);
    ASSERT_TRUE(flat.has_value()) << "chip " << target << " has no flat address";
    std::cout << "[FLAT] chip " << source << " -> chip " << target << ": access point "
              << static_cast<uint32_t>(access_points[access_point_index]) << ", offset 0x" << std::hex << offset
              << ", flat address 0x" << *flat << ", src 0x" << src->address() << std::dec
              << (posted ? ", posted" : ", non-posted") << std::endl;

    const experimental::NodeCoord node{0, 0};
    const experimental::KernelSpecName KERNEL{"flat_write"};
    experimental::ProgramSpec spec{
        .name = "flat_address_write",
        .kernels = {experimental::KernelSpec{
            .unique_id = KERNEL,
            .source = OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/flat_address_write.cpp",
            .num_threads = 1,
            .runtime_arg_schema = {.runtime_arg_names = {"src", "dst_lo", "dst_hi", "size", "posted"}},
            .hw_config = experimental::DataMovementHardwareConfig{},
        }},
        .work_units = {experimental::WorkUnitSpec{.name = "main", .kernels = {KERNEL}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(device, spec);
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = KERNEL,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            node,
            {{"src", src->address()},
             {"dst_lo", static_cast<uint32_t>(*flat)},
             {"dst_hi", static_cast<uint32_t>(*flat >> 32)},
             {"size", FLAT_WRITE_BYTES},
             {"posted", posted ? 1u : 0u}}),
    }};
    experimental::SetProgramRunArgs(program, params);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::cout << "[FLAT] kernel done" << std::endl;

    // A posted write may still be in flight when the kernel ends: poll briefly.
    for (int attempt = 0; attempt < 20; attempt++) {
        cluster.read_core(readback.data(), FLAT_WRITE_BYTES, tt_cxy_pair(target, target_core), offset);
        if (readback == pattern) {
            break;
        }
    }
    std::cout << "[FLAT] chip " << target << " word 0 now 0x" << std::hex << readback[0] << std::dec << std::endl;
    EXPECT_EQ(readback, pattern) << "the write did not land on chip " << target;

    std::vector<uint32_t> local;
    distributed::EnqueueReadMeshBuffer(cq, local, guard, true);
    std::cout << "[FLAT] chip " << source << " guard word 0 0x" << std::hex << local[0] << std::dec << std::endl;
    EXPECT_EQ(local, zeros) << "the write landed on chip " << source << " itself";
}

bool flat_write_two_chips() {
    return std::getenv("TT_METAL_SIMULATOR") != nullptr &&
           MetalContext::instance().get_cluster().number_of_devices() >= 2;
}

// Opens chip 1 instead of chip 0, for the reverse direction.
class HorizonX2Chip1Fixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void create_devices() override { AnyDispatchMeshDeviceSingleCardFixture::create_devices({1}); }
};

}  // namespace

// Chip 1's first access point is its NOC2AXI 0, at 0x20_...
TEST_F(QuasarMeshDeviceSingleCardFixture, HorizonX2FlatWritePosted) {
    if (!flat_write_two_chips()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_flat_write_lands(*devices_[0], 0, 1, 0, true);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, HorizonX2FlatWriteNonPosted) {
    if (!flat_write_two_chips()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_flat_write_lands(*devices_[0], 0, 1, 0, false);
}

// Chip 0's second access point, its NOC2AXI 1 at 0x11_...: chip 1's kernels can't use 0x10_..., which
// is tt-metal's own REMOTE window.
TEST_F(HorizonX2Chip1Fixture, HorizonX2FlatWriteReversePosted) {
    if (!flat_write_two_chips()) {
        GTEST_SKIP() << "needs a horizon_x2 build with both chips visible";
    }
    expect_flat_write_lands(*devices_[0], 1, 0, 1, true);
}
