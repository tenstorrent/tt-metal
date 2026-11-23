// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Horizon x2 mesh fabric API tests — unified common helpers.
// All tests run under FABRIC_2D (the production Horizon fabric config).
// Uses sender + checker kernel pattern matching NightlyFabric2DFixture.

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

constexpr uint32_t NOC_UNICAST_WRITE = 0u;
constexpr uint32_t NOC_UNICAST_SCATTER_WRITE = 4u;

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

void zero_l1(ChipId chip, const CoreCoord& virtual_core, uint32_t addr) {
    std::vector<uint32_t> zeros(TEST_WORDS, 0);
    MetalContext::instance().get_cluster().write_core(zeros.data(), TEST_BYTES, tt_cxy_pair(chip, virtual_core), addr);
}

std::vector<uint32_t> make_pattern(uint32_t base, uint32_t n = TEST_WORDS) {
    std::vector<uint32_t> pat(n);
    for (uint32_t i = 0; i < n; ++i) {
        pat[i] = base | i;
    }
    return pat;
}

// ─────────────────────────────────────────────────────────────────────────────
// HorizonFabricUnicastCommon
//
// Unified helper for unicast write and scatter write tests.
// noc_packet_type: 0 = NOC_UNICAST_WRITE, 4 = NOC_UNICAST_SCATTER_WRITE.
// with_state: false = stateless API, true = set_state + with_state.
//
// Two-kernel pattern: sender runs blocking on src_device, then checker runs
// blocking on tgt_device. Host reads result_buf to confirm pass.
// ─────────────────────────────────────────────────────────────────────────────
void HorizonFabricUnicastCommon(
    distributed::MeshDevice& src_device,
    distributed::MeshDevice& tgt_device,
    ChipId src_chip,
    ChipId tgt_chip,
    uint32_t noc_packet_type,
    bool with_state) {
    auto& cluster = MetalContext::instance().get_cluster();
    distributed::MeshCommandQueue& src_cq = src_device.mesh_command_queue();
    distributed::MeshCommandQueue& tgt_cq = tgt_device.mesh_command_queue();

    const CoreCoord logical_core{0, 0};
    const CoreCoord tgt_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(tgt_chip, logical_core, CoreType::WORKER);

    auto src_buf = make_l1_buffer(src_device, logical_core);
    auto recv_buf_a = make_l1_buffer(tgt_device, logical_core);
    auto recv_buf_b = make_l1_buffer(tgt_device, logical_core);
    auto result_buf = make_l1_buffer(tgt_device, logical_core);

    const uint32_t base = 0xC0FE0000u | (src_chip << 8) | tgt_chip;
    auto pattern = make_pattern(base);
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);
    zero_l1(tgt_chip, tgt_virt, recv_buf_a->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_b->address());
    zero_l1(tgt_chip, tgt_virt, result_buf->address());

    const uint32_t chunk_size_a = (noc_packet_type == NOC_UNICAST_SCATTER_WRITE) ? HALF_BYTES : TEST_BYTES;
    const uint32_t chunk_words_a = (noc_packet_type == NOC_UNICAST_SCATTER_WRITE) ? HALF_WORDS : TEST_WORDS;

    // ── Sender ───────────────────────────────────────────────────────────────
    const KernelSpecName SENDER{"mesh_unicast_write_sender"};
    ProgramSpec sender_spec{
        .name = "horizon_mesh_unicast_write_sender",
        .kernels = {KernelSpec{
            .unique_id = SENDER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_mesh_unicast_write_sender.cpp",
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
                      "total_size",
                      "noc_packet_type",
                      "with_state"}},
            .hw_config = DataMovementHardwareConfig{},
        }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {SENDER}, .target_nodes = logical_core}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(src_device, sender_spec);
    ProgramRunArgs sender_run;
    sender_run.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            logical_core,
            {{"src_l1_addr", src_buf->address()},
             {"dst_noc_x", static_cast<uint32_t>(tgt_virt.x)},
             {"dst_noc_y", static_cast<uint32_t>(tgt_virt.y)},
             {"dst_dev_id", static_cast<uint32_t>(tgt_chip)},
             {"dst_l1_addr_a", recv_buf_a->address()},
             {"dst_l1_addr_b", recv_buf_b->address()},
             {"chunk_size_a", chunk_size_a},
             {"total_size", TEST_BYTES},
             {"noc_packet_type", noc_packet_type},
             {"with_state", static_cast<uint32_t>(with_state)}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, sender_run);
    distributed::MeshWorkload sender_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(src_device.shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, sender_workload, true);

    // ── Checker ──────────────────────────────────────────────────────────────
    const KernelSpecName CHECKER{"mesh_unicast_write_checker"};
    ProgramSpec checker_spec{
        .name = "horizon_mesh_unicast_write_checker",
        .kernels = {KernelSpec{
            .unique_id = CHECKER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_mesh_unicast_write_checker.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"recv_l1_addr_a",
                      "recv_l1_addr_b",
                      "num_words",
                      "base_value",
                      "result_l1_addr",
                      "chunk_words_a",
                      "noc_packet_type"}},
            .hw_config = DataMovementHardwareConfig{},
        }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {CHECKER}, .target_nodes = logical_core}},
    };
    Program checker_prog = experimental::MakeProgramFromSpec(tgt_device, checker_spec);
    ProgramRunArgs checker_run;
    checker_run.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = CHECKER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            logical_core,
            {{"recv_l1_addr_a", recv_buf_a->address()},
             {"recv_l1_addr_b", recv_buf_b->address()},
             {"num_words", TEST_WORDS},
             {"base_value", base},
             {"result_l1_addr", result_buf->address()},
             {"chunk_words_a", chunk_words_a},
             {"noc_packet_type", noc_packet_type}}),
    }};
    experimental::SetProgramRunArgs(checker_prog, checker_run);
    distributed::MeshWorkload checker_workload;
    checker_workload.add_program(distributed::MeshCoordinateRange(tgt_device.shape()), std::move(checker_prog));
    distributed::EnqueueMeshWorkload(tgt_cq, checker_workload, true);

    std::vector<uint32_t> result(1, 0);
    cluster.read_core(result.data(), sizeof(uint32_t), tt_cxy_pair(tgt_chip, tgt_virt), result_buf->address());
    EXPECT_EQ(result[0], 1u) << "checker failed: 0x" << std::hex << result[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// HorizonConnectionManagerUnicastWriteCommon
//
// CM unicast write helper. Runs the existing CM sender kernel then runs the
// unified checker twice — once per recv buffer — to verify both CM slots.
// ─────────────────────────────────────────────────────────────────────────────
void HorizonConnectionManagerUnicastWriteCommon(
    distributed::MeshDevice& src_device, distributed::MeshDevice& tgt_device, ChipId src_chip, ChipId tgt_chip) {
    auto& cluster = MetalContext::instance().get_cluster();
    distributed::MeshCommandQueue& src_cq = src_device.mesh_command_queue();
    distributed::MeshCommandQueue& tgt_cq = tgt_device.mesh_command_queue();

    const CoreCoord logical_core{0, 0};
    const CoreCoord tgt_virt =
        cluster.get_virtual_coordinate_from_logical_coordinates(tgt_chip, logical_core, CoreType::WORKER);

    auto src_buf = make_l1_buffer(src_device, logical_core);
    auto recv_buf_0 = make_l1_buffer(tgt_device, logical_core);
    auto recv_buf_1 = make_l1_buffer(tgt_device, logical_core);
    auto result_buf_0 = make_l1_buffer(tgt_device, logical_core);
    auto result_buf_1 = make_l1_buffer(tgt_device, logical_core);

    const uint32_t base = 0xC0FE0100u;
    auto pattern = make_pattern(base);
    distributed::EnqueueWriteMeshBuffer(src_cq, src_buf, pattern, true);
    zero_l1(tgt_chip, tgt_virt, recv_buf_0->address());
    zero_l1(tgt_chip, tgt_virt, recv_buf_1->address());
    zero_l1(tgt_chip, tgt_virt, result_buf_0->address());
    zero_l1(tgt_chip, tgt_virt, result_buf_1->address());

    std::vector<uint32_t> sender_rt_args = {
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
                     "src_l1_addr", "dst_noc_x",     "dst_noc_y",      "dst_l1_addr_0", "dst_l1_addr_1",
                     "size",        "cm_s0_w0",      "cm_s0_w1",       "cm_s0_w2",      "cm_s0_w3",
                     "cm_s1_w0",    "cm_s1_w1",      "cm_s1_w2",       "cm_s1_w3",      "cm_my_chip",
                     "cm_my_mesh",  "cm_s0_dst_dev", "cm_s0_dst_mesh", "cm_s1_dst_dev", "cm_s1_dst_mesh",
                 }},
        .hw_config = DataMovementHardwareConfig{},
    };

    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        src_device,
        src_node,
        {dst_node, dst_node},
        {},
        sender_kernel_spec,
        logical_core,
        sender_rt_args,
        tt::tt_fabric::FabricApiType::Mesh);

    ProgramSpec sender_spec{
        .name = "horizon_mesh_cm_unicast_write",
        .kernels = {std::move(sender_kernel_spec)},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {SENDER}, .target_nodes = logical_core}},
    };
    Program sender_prog = experimental::MakeProgramFromSpec(src_device, sender_spec);
    ProgramRunArgs sender_run;
    sender_run.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
        .kernel = SENDER,
        .runtime_arg_values = MakeRuntimeArgsForSingleNode(
            logical_core, {{"src_l1_addr", sender_rt_args[0]},    {"dst_noc_x", sender_rt_args[1]},
                           {"dst_noc_y", sender_rt_args[2]},      {"dst_l1_addr_0", sender_rt_args[3]},
                           {"dst_l1_addr_1", sender_rt_args[4]},  {"size", sender_rt_args[5]},
                           {"cm_s0_w0", sender_rt_args[6]},       {"cm_s0_w1", sender_rt_args[7]},
                           {"cm_s0_w2", sender_rt_args[8]},       {"cm_s0_w3", sender_rt_args[9]},
                           {"cm_s1_w0", sender_rt_args[10]},      {"cm_s1_w1", sender_rt_args[11]},
                           {"cm_s1_w2", sender_rt_args[12]},      {"cm_s1_w3", sender_rt_args[13]},
                           {"cm_my_chip", sender_rt_args[14]},    {"cm_my_mesh", sender_rt_args[15]},
                           {"cm_s0_dst_dev", sender_rt_args[16]}, {"cm_s0_dst_mesh", sender_rt_args[17]},
                           {"cm_s1_dst_dev", sender_rt_args[18]}, {"cm_s1_dst_mesh", sender_rt_args[19]}}),
    }};
    experimental::SetProgramRunArgs(sender_prog, sender_run);
    distributed::MeshWorkload sender_workload;
    sender_workload.add_program(distributed::MeshCoordinateRange(src_device.shape()), std::move(sender_prog));
    distributed::EnqueueMeshWorkload(src_cq, sender_workload, true);

    // Checker spec reused for both recv buffers.
    const KernelSpecName CHECKER{"mesh_unicast_write_checker"};
    ProgramSpec checker_spec{
        .name = "horizon_mesh_unicast_write_checker",
        .kernels = {KernelSpec{
            .unique_id = CHECKER,
            .source = OVERRIDE_KERNEL_PREFIX
            "tests/tt_metal/tt_metal/test_kernels/dataflow/horizon_mesh_unicast_write_checker.cpp",
            .num_threads = 1,
            .runtime_arg_schema =
                {.runtime_arg_names =
                     {"recv_l1_addr_a",
                      "recv_l1_addr_b",
                      "num_words",
                      "base_value",
                      "result_l1_addr",
                      "chunk_words_a",
                      "noc_packet_type"}},
            .hw_config = DataMovementHardwareConfig{},
        }},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {CHECKER}, .target_nodes = logical_core}},
    };

    auto run_checker = [&](uint32_t recv_addr, uint32_t result_addr) {
        Program prog = experimental::MakeProgramFromSpec(tgt_device, checker_spec);
        ProgramRunArgs run;
        run.kernel_run_args = {ProgramRunArgs::KernelRunArgs{
            .kernel = CHECKER,
            .runtime_arg_values = MakeRuntimeArgsForSingleNode(
                logical_core,
                {{"recv_l1_addr_a", recv_addr},
                 {"recv_l1_addr_b", recv_addr},
                 {"num_words", TEST_WORDS},
                 {"base_value", base},
                 {"result_l1_addr", result_addr},
                 {"chunk_words_a", TEST_WORDS},
                 {"noc_packet_type", NOC_UNICAST_WRITE}}),
        }};
        experimental::SetProgramRunArgs(prog, run);
        distributed::MeshWorkload wl;
        wl.add_program(distributed::MeshCoordinateRange(tgt_device.shape()), std::move(prog));
        distributed::EnqueueMeshWorkload(tgt_cq, wl, true);
    };

    run_checker(recv_buf_0->address(), result_buf_0->address());
    run_checker(recv_buf_1->address(), result_buf_1->address());

    std::vector<uint32_t> r0(1, 0), r1(1, 0);
    cluster.read_core(r0.data(), sizeof(uint32_t), tt_cxy_pair(tgt_chip, tgt_virt), result_buf_0->address());
    cluster.read_core(r1.data(), sizeof(uint32_t), tt_cxy_pair(tgt_chip, tgt_virt), result_buf_1->address());
    EXPECT_EQ(r0[0], 1u) << "CM checker buf_0 failed: 0x" << std::hex << r0[0];
    EXPECT_EQ(r1[0], 1u) << "CM checker buf_1 failed: 0x" << std::hex << r1[0];
}

}  // namespace

// ── Unicast write ─────────────────────────────────────────────────────────────

TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshUnicastWriteStateless) {
    HorizonFabricUnicastCommon(*devices_[0], *devices_[1], 0, 1, NOC_UNICAST_WRITE, false);
}

TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshUnicastWriteStateful) {
    HorizonFabricUnicastCommon(*devices_[0], *devices_[1], 0, 1, NOC_UNICAST_WRITE, true);
}

// ── Scatter write ─────────────────────────────────────────────────────────────

TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshScatterWriteStateless) {
    HorizonFabricUnicastCommon(*devices_[0], *devices_[1], 0, 1, NOC_UNICAST_SCATTER_WRITE, false);
}

TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshScatterWriteStateful) {
    HorizonFabricUnicastCommon(*devices_[0], *devices_[1], 0, 1, NOC_UNICAST_SCATTER_WRITE, true);
}

// ── Connection Manager ────────────────────────────────────────────────────────

TEST_F(HorizonX2MeshFabricApiFixture, HorizonX2MeshConnectionManagerUnicastWrite) {
    HorizonConnectionManagerUnicastWriteCommon(*devices_[0], *devices_[1], 0, 1);
}
