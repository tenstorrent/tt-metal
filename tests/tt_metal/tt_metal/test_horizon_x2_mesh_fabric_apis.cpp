// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Tests for the Horizon x2 mesh fabric APIs: stateful unicast write and scatter write.
// Uses FabricUnicastWrite2DevBothChipsFixture (SetFabricConfig(FABRIC_2D) + both chips open).
// All verification is done host-side via cluster.read_core() after the sender workload
// completes with noc_async_write_barrier() — no checker kernel needed.

#include "common/device_fixture.hpp"
#include "context/metal_context.hpp"

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <cstdint>
#include <filesystem>
#include <cstdlib>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// Reuse fixture from test_horizon_x2_fabric_unicast_write.cpp.
// (Duplicate definition: each TU gets its own copy of the fixture class.)
class HorizonX2MeshFabricApiFixture : public QuasarMeshDeviceSingleCardFixture {
protected:
    void create_devices() override { AnyDispatchMeshDeviceSingleCardFixture::create_devices({0, 1}); }

    void SetUp() override {
        if (std::getenv("TT_METAL_SIMULATOR") == nullptr ||
            MetalContext::instance().get_cluster().number_of_devices() < 2) {
            GTEST_SKIP() << "needs horizon_x2 emulator with both chips visible";
        }
        auto mgd_path = (std::filesystem::path(MetalContext::instance().rtoptions().get_root_dir()) /
                         "tt_metal/fabric/mesh_graph_descriptors/horizon_x2_mesh_graph_descriptor.textproto")
                            .string();
        MetalContext::instance().set_custom_fabric_topology(
            mgd_path,
            {{tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{0}, 0), 0},
             {tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{0}, 1), 1}});
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
        QuasarMeshDeviceSingleCardFixture::SetUp();
    }

    void TearDown() override { QuasarMeshDeviceSingleCardFixture::TearDown(); }
};

namespace {

constexpr uint32_t TEST_WORDS = 16;
constexpr uint32_t TEST_BYTES = TEST_WORDS * sizeof(uint32_t);
constexpr uint32_t HALF_WORDS = TEST_WORDS / 2;
constexpr uint32_t HALF_BYTES = HALF_WORDS * sizeof(uint32_t);

std::shared_ptr<distributed::MeshBuffer> make_l1_buffer(
    distributed::MeshDevice& device, const CoreCoord& core = CoreCoord{0, 0}) {
    const ShardSpecBuffer shard_spec(
        CoreRangeSet(CoreRange(core, core)), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});
    const distributed::DeviceLocalBufferConfig local_cfg{
        .page_size = TEST_BYTES,
        .buffer_type = BufferType::L1,
        .sharding_args = BufferShardingArgs(shard_spec, TensorMemoryLayout::HEIGHT_SHARDED),
        .bottom_up = true,
    };
    return distributed::MeshBuffer::create(distributed::ReplicatedBufferConfig{.size = TEST_BYTES}, local_cfg, &device);
}

// Zero a buffer in the target chip's L1 via cluster write.
void zero_l1(ChipId chip, const CoreCoord& virtual_core, uint32_t addr) {
    std::vector<uint32_t> zeros(TEST_WORDS, 0);
    MetalContext::instance().get_cluster().write_core(zeros.data(), TEST_BYTES, tt_cxy_pair(chip, virtual_core), addr);
}

// Read TEST_WORDS from an L1 address and return them.
std::vector<uint32_t> read_l1(ChipId chip, const CoreCoord& virtual_core, uint32_t addr) {
    std::vector<uint32_t> result(TEST_WORDS, 0xdeadbeef);
    MetalContext::instance().get_cluster().read_core(result.data(), TEST_BYTES, tt_cxy_pair(chip, virtual_core), addr);
    return result;
}

// Read HALF_WORDS from an L1 address.
std::vector<uint32_t> read_l1_half(ChipId chip, const CoreCoord& virtual_core, uint32_t addr) {
    std::vector<uint32_t> result(HALF_WORDS, 0xdeadbeef);
    MetalContext::instance().get_cluster().read_core(result.data(), HALF_BYTES, tt_cxy_pair(chip, virtual_core), addr);
    return result;
}

// Build a test pattern for the given chip + iter.
std::vector<uint32_t> make_pattern(uint32_t base, uint32_t n = TEST_WORDS) {
    std::vector<uint32_t> pat(n);
    for (uint32_t i = 0; i < n; ++i) {
        pat[i] = base | i;
    }
    return pat;
}

}  // namespace

// ===========================================================================
// Test 1: Stateful unicast write — set_state once, with_state twice.
// The sender kernel writes the same payload to two different L1 addresses on
// chip 1.  After the sender completes (noc_async_write_barrier), the host
// reads both addresses and verifies.
// ===========================================================================
TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshUnicastWriteStateful) {
    auto& cluster = MetalContext::instance().get_cluster();
    ChipId src_chip = 0, tgt_chip = 1;
    CoreCoord logical_core{0, 0};
    const CoreCoord tgt_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(tgt_chip, logical_core, CoreType::WORKER);

    auto src_buf = make_l1_buffer(*devices_[src_chip], logical_core);
    auto recv_buf_a = make_l1_buffer(*devices_[tgt_chip], logical_core);
    auto recv_buf_b = make_l1_buffer(*devices_[tgt_chip], logical_core);

    const uint32_t base = 0xC0FE0000u;
    auto pattern = make_pattern(base);
    std::vector<uint32_t> zeros(TEST_WORDS, 0);

    distributed::MeshCommandQueue& src_cq = devices_[src_chip]->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);
    zero_l1(tgt_chip, tgt_virt, recv_buf_a->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_b->address());

    const KernelSpecName SENDER{"mesh_unicast_write_stateful_sender"};
    ProgramSpec sender_spec{
        .name = "horizon_mesh_unicast_write_stateful_sender",
        .kernels = {KernelSpec{
            .unique_id = SENDER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_mesh_unicast_write_stateful_sender.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"src_l1_addr", "dst_noc_x", "dst_noc_y", "dst_dev_id", "dst_l1_addr_a", "dst_l1_addr_b", "size"}},
            .hw_config = DataMovementHardwareConfig{},
        }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {SENDER}, .target_nodes = logical_core}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(*devices_[src_chip], sender_spec);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            logical_core,
            {{"src_l1_addr", src_buf->address()},
             {"dst_noc_x", static_cast<uint32_t>(tgt_virt.x)},
             {"dst_noc_y", static_cast<uint32_t>(tgt_virt.y)},
             {"dst_dev_id", static_cast<uint32_t>(tgt_chip)},
             {"dst_l1_addr_a", recv_buf_a->address()},
             {"dst_l1_addr_b", recv_buf_b->address()},
             {"size", TEST_BYTES}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, run_args);
    distributed::MeshWorkload sender_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(devices_[src_chip]->shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, sender_workload, true);

    auto got_a = read_l1(tgt_chip, tgt_virt, recv_buf_a->address());
    auto got_b = read_l1(tgt_chip, tgt_virt, recv_buf_b->address());
    EXPECT_EQ(got_a, pattern) << "recv_buf_a mismatch (stateful unicast write)";
    EXPECT_EQ(got_b, pattern) << "recv_buf_b mismatch (stateful unicast write)";
}

// ===========================================================================
// Test 2: Stateless scatter write.
// The sender kernel splits the payload into two equal halves and writes each
// to a different L1 address on chip 1.
// ===========================================================================
TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshScatterWriteStateless) {
    auto& cluster = MetalContext::instance().get_cluster();
    ChipId src_chip = 0, tgt_chip = 1;
    CoreCoord logical_core{0, 0};
    const CoreCoord tgt_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(tgt_chip, logical_core, CoreType::WORKER);

    auto src_buf = make_l1_buffer(*devices_[src_chip], logical_core);
    auto recv_buf_a = make_l1_buffer(*devices_[tgt_chip], logical_core);
    auto recv_buf_b = make_l1_buffer(*devices_[tgt_chip], logical_core);

    const uint32_t base = 0xBEEF0000u;
    auto pattern = make_pattern(base);
    distributed::MeshCommandQueue& src_cq = devices_[src_chip]->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);
    zero_l1(tgt_chip, tgt_virt, recv_buf_a->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_b->address());

    const KernelSpecName SENDER{"mesh_scatter_write_sender"};
    ProgramSpec sender_spec{
        .name = "horizon_mesh_scatter_write_sender",
        .kernels = {KernelSpec{
            .unique_id = SENDER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_mesh_scatter_write_sender.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"src_l1_addr",
                      "dst_noc_x",
                      "dst_noc_y",
                      "dst_dev_id",
                      "dst_l1_addr_a",
                      "dst_l1_addr_b",
                      "chunk_size_a",
                      "total_size"}},
            .hw_config = DataMovementHardwareConfig{},
        }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {SENDER}, .target_nodes = logical_core}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(*devices_[src_chip], sender_spec);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            logical_core,
            {{"src_l1_addr", src_buf->address()},
             {"dst_noc_x", static_cast<uint32_t>(tgt_virt.x)},
             {"dst_noc_y", static_cast<uint32_t>(tgt_virt.y)},
             {"dst_dev_id", static_cast<uint32_t>(tgt_chip)},
             {"dst_l1_addr_a", recv_buf_a->address()},
             {"dst_l1_addr_b", recv_buf_b->address()},
             {"chunk_size_a", HALF_BYTES},
             {"total_size", TEST_BYTES}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, run_args);
    distributed::MeshWorkload sender_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(devices_[src_chip]->shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, sender_workload, true);

    // First half of pattern at addr_a, second half at addr_b.
    auto got_a = read_l1_half(tgt_chip, tgt_virt, recv_buf_a->address());
    auto got_b = read_l1_half(tgt_chip, tgt_virt, recv_buf_b->address());
    auto expected_a = make_pattern(base, HALF_WORDS);
    auto expected_b = make_pattern(base, HALF_WORDS);
    for (uint32_t i = 0; i < HALF_WORDS; ++i) {
        expected_b[i] = base | (HALF_WORDS + i);
    }
    EXPECT_EQ(got_a, expected_a) << "recv_buf_a mismatch (scatter write stateless)";
    EXPECT_EQ(got_b, expected_b) << "recv_buf_b mismatch (scatter write stateless)";
}

// ===========================================================================
// Test 3: Stateful scatter write — set_state once, with_state twice.
// First with_state sends chunks to (addr_a, addr_b).
// Second with_state updates DstAddrs to (addr_c, addr_d) and sends again.
// ===========================================================================
TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshScatterWriteStateful) {
    auto& cluster = MetalContext::instance().get_cluster();
    ChipId src_chip = 0, tgt_chip = 1;
    CoreCoord logical_core{0, 0};
    const CoreCoord tgt_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(tgt_chip, logical_core, CoreType::WORKER);

    auto src_buf = make_l1_buffer(*devices_[src_chip], logical_core);
    auto recv_buf_a = make_l1_buffer(*devices_[tgt_chip], logical_core);
    auto recv_buf_b = make_l1_buffer(*devices_[tgt_chip], logical_core);
    auto recv_buf_c = make_l1_buffer(*devices_[tgt_chip], logical_core);
    auto recv_buf_d = make_l1_buffer(*devices_[tgt_chip], logical_core);

    const uint32_t base = 0xFAB10000u;
    auto pattern = make_pattern(base);
    distributed::MeshCommandQueue& src_cq = devices_[src_chip]->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);
    zero_l1(tgt_chip, tgt_virt, recv_buf_a->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_b->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_c->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_d->address());

    const KernelSpecName SENDER{"mesh_scatter_write_stateful_sender"};
    ProgramSpec sender_spec{
        .name = "horizon_mesh_scatter_write_stateful_sender",
        .kernels = {KernelSpec{
            .unique_id = SENDER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_mesh_scatter_write_stateful_sender.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"src_l1_addr",
                      "dst_noc_x",
                      "dst_noc_y",
                      "dst_dev_id",
                      "dst_l1_addr_a",
                      "dst_l1_addr_b",
                      "dst_l1_addr_c",
                      "dst_l1_addr_d",
                      "chunk_size_0",
                      "total_size"}},
            .hw_config = DataMovementHardwareConfig{},
        }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {SENDER}, .target_nodes = logical_core}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(*devices_[src_chip], sender_spec);
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            logical_core,
            {{"src_l1_addr", src_buf->address()},
             {"dst_noc_x", static_cast<uint32_t>(tgt_virt.x)},
             {"dst_noc_y", static_cast<uint32_t>(tgt_virt.y)},
             {"dst_dev_id", static_cast<uint32_t>(tgt_chip)},
             {"dst_l1_addr_a", recv_buf_a->address()},
             {"dst_l1_addr_b", recv_buf_b->address()},
             {"dst_l1_addr_c", recv_buf_c->address()},
             {"dst_l1_addr_d", recv_buf_d->address()},
             {"chunk_size_0", HALF_BYTES},
             {"total_size", TEST_BYTES}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, run_args);
    distributed::MeshWorkload sender_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(devices_[src_chip]->shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, sender_workload, true);

    auto got_a = read_l1_half(tgt_chip, tgt_virt, recv_buf_a->address());
    auto got_b = read_l1_half(tgt_chip, tgt_virt, recv_buf_b->address());
    auto got_c = read_l1_half(tgt_chip, tgt_virt, recv_buf_c->address());
    auto got_d = read_l1_half(tgt_chip, tgt_virt, recv_buf_d->address());

    auto first_half = make_pattern(base, HALF_WORDS);
    auto second_half = make_pattern(base, HALF_WORDS);
    for (uint32_t i = 0; i < HALF_WORDS; ++i) {
        second_half[i] = base | (HALF_WORDS + i);
    }

    EXPECT_EQ(got_a, first_half) << "recv_buf_a mismatch (scatter stateful, first send, chunk 0)";
    EXPECT_EQ(got_b, second_half) << "recv_buf_b mismatch (scatter stateful, first send, chunk 1)";
    EXPECT_EQ(got_c, first_half) << "recv_buf_c mismatch (scatter stateful, second send, chunk 0)";
    EXPECT_EQ(got_d, second_half) << "recv_buf_d mismatch (scatter stateful, second send, chunk 1)";
}

// ===========================================================================
// Test 4: ConnectionManager unicast write — two route_ids, both pointing to
// chip 1.  route_id=0 writes to recv_buf_0, route_id=1 writes to recv_buf_1.
// Verifies that connection_manager.get(route_id) correctly indexes each slot.
// ===========================================================================
TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshConnectionManagerUnicastWrite) {
    auto& cluster = MetalContext::instance().get_cluster();
    ChipId src_chip = 0, tgt_chip = 1;
    CoreCoord logical_core{0, 0};
    const CoreCoord tgt_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(tgt_chip, logical_core, CoreType::WORKER);

    auto src_buf = make_l1_buffer(*devices_[src_chip], logical_core);
    auto recv_buf_0 = make_l1_buffer(*devices_[tgt_chip], logical_core);
    auto recv_buf_1 = make_l1_buffer(*devices_[tgt_chip], logical_core);

    const uint32_t base = 0xC0FE0100u;
    auto pattern = make_pattern(base);
    distributed::MeshCommandQueue& src_cq = devices_[src_chip]->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);
    zero_l1(tgt_chip, tgt_virt, recv_buf_0->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_1->address());

    // Fixed args (indices 0-5); CM args (indices 6-19) appended by helper below.
    std::vector<uint32_t> sender_runtime_args = {
        src_buf->address(),
        static_cast<uint32_t>(tgt_virt.x),
        static_cast<uint32_t>(tgt_virt.y),
        recv_buf_0->address(),
        recv_buf_1->address(),
        TEST_BYTES,
    };

    auto src_node = tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{0}, src_chip);
    auto dst_node = tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{0}, tgt_chip);

    const KernelSpecName SENDER{"mesh_cm_unicast_write_sender"};
    KernelSpec sender_kernel_spec{
        .unique_id = SENDER,
        .source = OVERRIDE_KERNEL_PREFIX
        "tests/tt_metal/tt_metal/test_kernels/dataflow/"
        "horizon_mesh_connection_manager_unicast_write_sender.cpp",
        .num_threads = 1,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {
                     // Fixed args (0-5)
                     "src_l1_addr",
                     "dst_noc_x",
                     "dst_noc_y",
                     "dst_l1_addr_0",
                     "dst_l1_addr_1",
                     "size",
                     // CM slot 0: tag, eth_ch, teardown_sem, buf_idx (6-9)
                     "cm_s0_w0",
                     "cm_s0_w1",
                     "cm_s0_w2",
                     "cm_s0_w3",
                     // CM slot 1: (10-13)
                     "cm_s1_w0",
                     "cm_s1_w1",
                     "cm_s1_w2",
                     "cm_s1_w3",
                     // FABRIC_2D suffix: my_chip, my_mesh (14-15)
                     "cm_my_chip",
                     "cm_my_mesh",
                     // Per-slot dst ids: slot 0 (16-17), slot 1 (18-19)
                     "cm_s0_dst_dev",
                     "cm_s0_dst_mesh",
                     "cm_s1_dst_dev",
                     "cm_s1_dst_mesh",
                 }},
        .hw_config = DataMovementHardwareConfig{},
    };

    // Appends 14 CM RT arg words (8 zeros for ETH fields + 6 for FABRIC_2D suffix)
    // and injects API_TYPE_Mesh + FABRIC_2D into sender_kernel_spec.compiler_options.defines.
    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        *devices_[src_chip],
        src_node,
        {dst_node, dst_node},
        {},
        sender_kernel_spec,
        logical_core,
        sender_runtime_args,
        tt::tt_fabric::FabricApiType::Mesh);

    ProgramSpec sender_spec{
        .name = "horizon_mesh_cm_unicast_write",
        .kernels = {std::move(sender_kernel_spec)},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {SENDER}, .target_nodes = logical_core}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(*devices_[src_chip], sender_spec);

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            logical_core, {{"src_l1_addr", sender_runtime_args[0]},    {"dst_noc_x", sender_runtime_args[1]},
                           {"dst_noc_y", sender_runtime_args[2]},      {"dst_l1_addr_0", sender_runtime_args[3]},
                           {"dst_l1_addr_1", sender_runtime_args[4]},  {"size", sender_runtime_args[5]},
                           {"cm_s0_w0", sender_runtime_args[6]},       {"cm_s0_w1", sender_runtime_args[7]},
                           {"cm_s0_w2", sender_runtime_args[8]},       {"cm_s0_w3", sender_runtime_args[9]},
                           {"cm_s1_w0", sender_runtime_args[10]},      {"cm_s1_w1", sender_runtime_args[11]},
                           {"cm_s1_w2", sender_runtime_args[12]},      {"cm_s1_w3", sender_runtime_args[13]},
                           {"cm_my_chip", sender_runtime_args[14]},    {"cm_my_mesh", sender_runtime_args[15]},
                           {"cm_s0_dst_dev", sender_runtime_args[16]}, {"cm_s0_dst_mesh", sender_runtime_args[17]},
                           {"cm_s1_dst_dev", sender_runtime_args[18]}, {"cm_s1_dst_mesh", sender_runtime_args[19]}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, run_args);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(devices_[src_chip]->shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, workload, true);

    auto got_0 = read_l1(tgt_chip, tgt_virt, recv_buf_0->address());
    auto got_1 = read_l1(tgt_chip, tgt_virt, recv_buf_1->address());
    EXPECT_EQ(got_0, pattern) << "recv_buf_0 mismatch (ConnectionManager route_id=0)";
    EXPECT_EQ(got_1, pattern) << "recv_buf_1 mismatch (ConnectionManager route_id=1)";
}
