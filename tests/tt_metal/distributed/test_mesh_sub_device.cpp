// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include "llrt.hpp"
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/tt_metal/dispatch/sub_device_test_utils.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>

#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/program/sub_device_setup_batch.hpp"

namespace tt::tt_metal::distributed {
class SubDeviceSetupCacheTestAccessor {
public:
    static auto keys(MeshCommandQueue& queue) {
        auto& cq = dynamic_cast<FDMeshCommandQueue&>(queue);
        std::vector<std::vector<std::pair<CoreRangeSet, uint32_t>>> keys;
        keys.reserve(cq.sub_device_setup_commands_.size());
        for (const auto& entry : cq.sub_device_setup_commands_) {
            keys.push_back(entry.core_mapping);
        }
        return keys;
    }
    static constexpr size_t capacity() { return FDMeshCommandQueue::max_sub_device_setup_cache_entries; }
};
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::distributed::test {
namespace {

namespace tt::tt_metal {
using MeshSubDeviceTestSuite = GenericMeshDeviceFixture;

// Two hardware CQs plus a trace region, so a trace replay can be left running on one CQ while the
// host keeps issuing on the other.
class MeshSubDeviceMultiCQTraceTestSuite : public MeshDeviceFixtureBase {
protected:
    MeshSubDeviceMultiCQTraceTestSuite() :
        MeshDeviceFixtureBase(Config{.num_cqs = 2, .trace_region_size = (16 << 20)}) {}
};

TEST(SubDeviceSetupBatchTest, PreservesGroupsAtFetchBoundary) {
    using program_dispatch::setup_batch::append_setup_commands;
    // One cache line per group; two groups fit exactly in this synthetic fetch limit.
    const vector_aligned<uint32_t> first(16, 1), second(16, 2), third(16, 3);
    std::vector<vector_aligned<uint32_t>> batches;
    append_setup_commands(batches, first, 128);
    append_setup_commands(batches, second, 128);
    ASSERT_EQ(batches.size(), 1);
    EXPECT_EQ(batches[0].size() * sizeof(uint32_t), 128);
    append_setup_commands(batches, third, 128);
    ASSERT_EQ(batches.size(), 2);
    vector_aligned<uint32_t> expected(first);
    expected.insert(expected.end(), second.begin(), second.end());
    EXPECT_EQ(batches[0], expected);
    EXPECT_EQ(batches[1], third);
    const auto saved = batches;
    const vector_aligned<uint32_t> oversized(48, 4);
    EXPECT_ANY_THROW(append_setup_commands(batches, oversized, 128));
    EXPECT_EQ(batches, saved);
}

TEST(SubDeviceSetupBatchTest, ResetAndFirstSetupFetchBoundary) {
    using program_dispatch::setup_batch::can_combine_setup;
    EXPECT_TRUE(can_combine_setup(64, 64, 128));
    EXPECT_FALSE(can_combine_setup(64, 128, 128));
    EXPECT_TRUE(can_combine_setup(128, 0, 128));
    EXPECT_FALSE(can_combine_setup(192, 0, 128));
}

TEST_F(MeshSubDeviceTestSuite, SetupCacheEvictsLeastRecentlyUsedConfiguration) {
    using Accessor = SubDeviceSetupCacheTestAccessor;
    std::vector<SubDeviceManagerId> managers;
    auto& cq = mesh_device_->mesh_command_queue();
    const auto capacity = Accessor::capacity();
    // Distinct single-core partitions also exercise changing mailbox mappings with identical counts.
    for (size_t i = 0; i < capacity; ++i) {
        const CoreCoord core(i % 4, i / 4);
        managers.push_back(mesh_device_->create_sub_device_manager(
            {SubDevice(std::array{CoreRangeSet(CoreRange(core, core))})}, 3200));
        mesh_device_->load_sub_device_manager(managers.back());
    }
    const auto full = Accessor::keys(cq);
    ASSERT_EQ(full.size(), capacity);
    mesh_device_->load_sub_device_manager(managers.front());
    const auto promoted = Accessor::keys(cq);
    EXPECT_EQ(promoted.front(), full.back());
    EXPECT_EQ(promoted.back(), full[capacity - 2]);

    // Cycle through removed managers: their historical entries must not accumulate.
    for (size_t i = capacity; i < 2 * capacity; ++i) {
        const CoreCoord core(i % 4, i / 4);
        const auto manager =
            mesh_device_->create_sub_device_manager({SubDevice(std::array{CoreRangeSet(CoreRange(core, core))})}, 3200);
        mesh_device_->load_sub_device_manager(manager);
        const auto keys = Accessor::keys(cq);
        ASSERT_EQ(keys.size(), capacity);
        if (i == capacity) {
            EXPECT_EQ(keys[1], promoted.front());
            EXPECT_EQ(std::find(keys.begin(), keys.end(), promoted.back()), keys.end());
        }
        mesh_device_->clear_loaded_sub_device_manager();
        mesh_device_->remove_sub_device_manager(manager);
    }
    // Rebuild an evicted configuration and verify its setup still dispatches successfully.
    mesh_device_->load_sub_device_manager(managers.front());
    EXPECT_EQ(Accessor::keys(cq).front(), full.back());
    Finish(cq);
    mesh_device_->clear_loaded_sub_device_manager();
    for (auto manager : managers) {
        mesh_device_->remove_sub_device_manager(manager);
    }
}

TEST_F(MeshSubDeviceTestSuite, SyncWorkloadsOnSubDevice) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});

    uint32_t num_iters = 5;
    auto sub_device_manager = mesh_device_->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    mesh_device_->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(mesh_device_.get(), sub_device_1, sub_device_2);

    MeshCoordinateRange devices(mesh_device_->shape());
    auto waiter_mesh_workload = MeshWorkload();
    auto syncer_mesh_workload = MeshWorkload();
    auto incrementer_mesh_workload = MeshWorkload();
    waiter_mesh_workload.add_program(devices, std::move(waiter_program));
    syncer_mesh_workload.add_program(devices, std::move(syncer_program));
    incrementer_mesh_workload.add_program(devices, std::move(incrementer_program));
    for (uint32_t i = 0; i < num_iters; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({{SubDeviceId{0}}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload, false);
        mesh_device_->reset_sub_device_stall_group();
    }
    Finish(mesh_device_->mesh_command_queue());
}

TEST_F(MeshSubDeviceTestSuite, DataCopyOnSubDevices) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(CoreRange({1, 1}, {1, 1}))});
    SubDevice sub_device_3(std::array{CoreRangeSet(CoreRange({2, 2}, {2, 2}))});

    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    uint32_t num_tiles = 32;
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size * num_tiles, .buffer_type = BufferType::DRAM, .bottom_up = true};

    ReplicatedBufferConfig global_buffer_config{
        .size = single_tile_size * num_tiles,
    };
    // Create IO Buffers
    auto input_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto output_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Create and Load SubDeviceConfig on the mesh
    auto sub_device_manager = mesh_device_->create_sub_device_manager({sub_device_1, sub_device_2, sub_device_3}, 3200);
    mesh_device_->load_sub_device_manager(sub_device_manager);

    auto syncer_coord = sub_device_1.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_phys = mesh_device_->worker_core_from_logical_core(syncer_coord);
    auto datacopy_coord = sub_device_2.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto datacopy_core = CoreRangeSet(CoreRange(datacopy_coord, datacopy_coord));
    auto datacopy_core_phys = mesh_device_->worker_core_from_logical_core(datacopy_coord);

    auto all_cores = syncer_core.merge(datacopy_core);
    auto global_sem = GlobalSemaphore(*mesh_device_, all_cores, 0);

    Program sync_and_incr_program = CreateProgram();
    auto sync_kernel = CreateKernel(
        sync_and_incr_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/sync_and_increment.cpp",
        syncer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 3> sync_rt_args = {global_sem.address(), datacopy_core_phys.x, datacopy_core_phys.y};
    SetRuntimeArgs(sync_and_incr_program, sync_kernel, syncer_core, sync_rt_args);

    Program datacopy_program = CreateProgram();
    auto datacopy_kernel = CreateKernel(
        datacopy_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/sync_and_datacopy.cpp",
        datacopy_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 6> datacopy_rt_args = {
        global_sem.address(), 0, 0, input_buf->address(), output_buf->address(), num_tiles};
    SetRuntimeArgs(datacopy_program, datacopy_kernel, datacopy_core, datacopy_rt_args);
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(single_tile_size * num_tiles, {{src0_cb_index, DataFormat::UInt32}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(datacopy_program, datacopy_core, cb_src0_config);

    auto syncer_mesh_workload = MeshWorkload();
    auto datacopy_mesh_workload = MeshWorkload();
    MeshCoordinateRange devices(mesh_device_->shape());

    syncer_mesh_workload.add_program(devices, std::move(sync_and_incr_program));
    datacopy_mesh_workload.add_program(devices, std::move(datacopy_program));

    for (int i = 0; i < 50; i++) {
        mesh_device_->set_sub_device_stall_group({{SubDeviceId{2}}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, false);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), datacopy_mesh_workload, false);

        std::vector<uint32_t> src_vec(input_buf->size() / sizeof(uint32_t));
        std::iota(src_vec.begin(), src_vec.end(), i);
        // Block after this write on host, since the global semaphore update starting the
        // program goes through an independent path (UMD) and can go out of order wrt the
        // buffer data
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), input_buf, src_vec, true);

        for (auto* device : mesh_device_->get_devices()) {
            MetalContext::instance().get_cluster().write_core(
                device->id(), syncer_core_phys, std::vector<uint32_t>{1}, global_sem.address());
        }
        mesh_device_->reset_sub_device_stall_group();
        for (std::size_t logical_x = 0; logical_x < output_buf->device()->num_cols(); logical_x++) {
            for (std::size_t logical_y = 0; logical_y < output_buf->device()->num_rows(); logical_y++) {
                std::vector<uint32_t> dst_vec;
                ReadShard(
                    mesh_device_->mesh_command_queue(), dst_vec, output_buf, MeshCoordinate(logical_y, logical_x));
                EXPECT_EQ(dst_vec, src_vec);
            }
        }
    }
}

TEST_F(MeshSubDeviceTestSuite, SubDeviceSwitching) {
    // Sub Devices for config 0
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    // Sub Devices for config 1
    SubDevice sub_device_3(std::array{CoreRangeSet(CoreRange({3, 3}, {5, 5}))});
    SubDevice sub_device_4(std::array{CoreRangeSet(std::vector{CoreRange({0, 0}, {0, 0}), CoreRange({1, 1}, {1, 1})})});
    // Initialize different SubDeviceManagers
    auto sub_device_manager_0 = mesh_device_->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    auto sub_device_manager_1 = mesh_device_->create_sub_device_manager({sub_device_3, sub_device_4}, 3200);

    // Initialize programs on different SubDevices
    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(mesh_device_.get(), sub_device_1, sub_device_2);

    auto [waiter_program_1, syncer_program_1, incrementer_program_1, global_sem_1] =
        create_basic_sync_program(mesh_device_.get(), sub_device_3, sub_device_4);

    uint32_t num_iters = 100;
    // Create MeshWorkloads corresponding to different SubDevice configs,
    // so we can single-shot dispatch to the entire Mesh
    MeshCoordinateRange devices(mesh_device_->shape());
    auto waiter_mesh_workload = MeshWorkload();
    auto syncer_mesh_workload = MeshWorkload();
    auto incrementer_mesh_workload = MeshWorkload();
    waiter_mesh_workload.add_program(devices, std::move(waiter_program));
    syncer_mesh_workload.add_program(devices, std::move(syncer_program));
    incrementer_mesh_workload.add_program(devices, std::move(incrementer_program));

    auto waiter_mesh_workload_1 = MeshWorkload();
    auto syncer_mesh_workload_1 = MeshWorkload();
    auto incrementer_mesh_workload_1 = MeshWorkload();
    waiter_mesh_workload_1.add_program(devices, std::move(waiter_program_1));
    syncer_mesh_workload_1.add_program(devices, std::move(syncer_program_1));
    incrementer_mesh_workload_1.add_program(devices, std::move(incrementer_program_1));

    // Load SubDevice configs, run corresponding workloads, reset ... repeat
    for (uint32_t i = 0; i < num_iters; i++) {
        mesh_device_->load_sub_device_manager(sub_device_manager_0);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({{SubDeviceId{0}}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload, false);
        mesh_device_->reset_sub_device_stall_group();

        mesh_device_->load_sub_device_manager(sub_device_manager_1);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload_1, false);
        mesh_device_->set_sub_device_stall_group({{SubDeviceId{0}}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload_1, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload_1, false);
        mesh_device_->reset_sub_device_stall_group();
    }
    Finish(mesh_device_->mesh_command_queue());
}

// Switching sub device managers resets the worker GO mailboxes and remaps each core's go message index,
// state that is shared by every hardware CQ. Both managers below cover the same cores in the opposite
// order, so the switch flips the go message index of every core the trace runs on. A switch that lands
// while the trace is still replaying on the other CQ leaves those cores polling a mailbox slot the trace
// never signals: the workers stop launching and the replay never completes. Regressions here hang rather
// than fail.
TEST_F(MeshSubDeviceMultiCQTraceTestSuite, SubDeviceSwitchingWhileOtherCQReplaysTrace) {
    constexpr uint32_t k_local_l1_size = 3200;
    // Enough device side work that the host reaches the manager switch below mid replay.
    constexpr uint32_t k_delay_iters = 4000000;
    constexpr uint32_t k_workloads_in_trace = 8;

    CoreRangeSet cores_0(CoreRange({0, 0}, {1, 1}));
    CoreRangeSet cores_1(CoreRange({2, 2}, {3, 3}));
    auto sub_device_manager_0 = mesh_device_->create_sub_device_manager(
        {SubDevice(std::array{cores_0}), SubDevice(std::array{cores_1})}, k_local_l1_size);
    auto sub_device_manager_1 = mesh_device_->create_sub_device_manager(
        {SubDevice(std::array{cores_1}), SubDevice(std::array{cores_0})}, k_local_l1_size);
    mesh_device_->load_sub_device_manager(sub_device_manager_0);

    Program delay_program = CreateProgram();
    auto delay_kernel = CreateKernel(
        delay_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/delay.cpp",
        cores_0,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 1> delay_rt_args = {k_delay_iters};
    SetRuntimeArgs(delay_program, delay_kernel, cores_0, delay_rt_args);

    MeshCoordinateRange devices(mesh_device_->shape());
    auto delay_mesh_workload = MeshWorkload();
    delay_mesh_workload.add_program(devices, std::move(delay_program));

    // Compile the workload before capturing it.
    auto& trace_cq = mesh_device_->mesh_command_queue(1);
    EnqueueMeshWorkload(trace_cq, delay_mesh_workload, true);

    auto trace_id = BeginTraceCapture(mesh_device_.get(), 1);
    for (uint32_t i = 0; i < k_workloads_in_trace; i++) {
        EnqueueMeshWorkload(trace_cq, delay_mesh_workload, false);
    }
    mesh_device_->end_mesh_trace(1, trace_id);

    // Repeat to exercise both newly built and cached manager setup commands while CQ1 is busy.
    for (uint32_t i = 0; i < 3; ++i) {
        mesh_device_->replay_mesh_trace(1, trace_id, false);
        mesh_device_->load_sub_device_manager(sub_device_manager_1);
        Finish(mesh_device_->mesh_command_queue(0));
        Finish(trace_cq);
        // The trace belongs to the manager it was captured under.
        mesh_device_->load_sub_device_manager(sub_device_manager_0);
    }
    mesh_device_->release_mesh_trace(trace_id);
}

TEST_F(MeshSubDeviceTestSuite, SubDeviceBasicProgramsReuse) {
    constexpr uint32_t k_num_iters = 5;
    constexpr uint32_t k_local_l1_size = 3200;

    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});
    // sub-device 3 and 4 are supersets of sub-device 1 and 2 respectively
    SubDevice sub_device_3(std::array{CoreRangeSet(std::vector{CoreRange({0, 0}, {2, 2}), CoreRange({5, 5}, {5, 5})})});
    SubDevice sub_device_4(std::array{
        CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4}), CoreRange({6, 6}, {6, 6})})});
    auto sub_device_manager_1 = mesh_device_->create_sub_device_manager({sub_device_1, sub_device_2}, k_local_l1_size);
    auto sub_device_manager_2 = mesh_device_->create_sub_device_manager({sub_device_4, sub_device_3}, k_local_l1_size);
    mesh_device_->load_sub_device_manager(sub_device_manager_1);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(mesh_device_.get(), sub_device_1, sub_device_2);
    MeshCoordinateRange devices(mesh_device_->shape());
    auto waiter_mesh_workload = MeshWorkload();
    auto syncer_mesh_workload = MeshWorkload();
    auto incrementer_mesh_workload = MeshWorkload();
    waiter_mesh_workload.add_program(devices, std::move(waiter_program));
    syncer_mesh_workload.add_program(devices, std::move(syncer_program));
    incrementer_mesh_workload.add_program(devices, std::move(incrementer_program));

    // Run programs on sub-device manager 1
    for (uint32_t i = 0; i < k_num_iters; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({{SubDeviceId{0}}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload, false);
        mesh_device_->reset_sub_device_stall_group();
    }
    Finish(mesh_device_->mesh_command_queue());

    // Rerun programs on sub-device manager 2
    mesh_device_->load_sub_device_manager(sub_device_manager_2);
    for (uint32_t i = 0; i < k_num_iters; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({{SubDeviceId{1}}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload, false);
        mesh_device_->reset_sub_device_stall_group();
    }
    Finish(mesh_device_->mesh_command_queue());
}
}  // namespace tt::tt_metal
}  // namespace
}  // namespace tt::tt_metal::distributed::test
