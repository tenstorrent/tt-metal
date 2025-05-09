// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
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
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
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
#include <tt-metalium/util.hpp>

namespace tt::tt_metal::distributed::test {
namespace {

namespace tt::tt_metal {
using MeshSubDeviceTestSuite = GenericMeshDeviceFixture;

TEST_F(MeshSubDeviceTestSuite, SyncWorkloadsOnSubDevice) {
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});

    uint32_t num_iters = 5;
    auto sub_device_manager = mesh_device_->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    mesh_device_->load_sub_device_manager(sub_device_manager);

    auto [waiter_program, syncer_program, incrementer_program, global_sem] =
        create_basic_sync_program(mesh_device_.get(), sub_device_1, sub_device_2);

    MeshCoordinateRange devices(mesh_device_->shape());
    auto waiter_mesh_workload = CreateMeshWorkload();
    auto syncer_mesh_workload = CreateMeshWorkload();
    auto incrementer_mesh_workload = CreateMeshWorkload();
    AddProgramToMeshWorkload(waiter_mesh_workload, std::move(waiter_program), devices);
    AddProgramToMeshWorkload(syncer_mesh_workload, std::move(syncer_program), devices);
    AddProgramToMeshWorkload(incrementer_mesh_workload, std::move(incrementer_program), devices);
    for (uint32_t i = 0; i < num_iters; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({SubDeviceId{0}});
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

    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::UInt32);
    uint32_t num_tiles = 32;
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size * num_tiles,
        .buffer_type = BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};

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
    auto global_sem = CreateGlobalSemaphore(mesh_device_.get(), all_cores, 0);

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
    CBHandle cb_src0 = CreateCircularBuffer(datacopy_program, datacopy_core, cb_src0_config);

    auto syncer_mesh_workload = CreateMeshWorkload();
    auto datacopy_mesh_workload = CreateMeshWorkload();
    MeshCoordinateRange devices(mesh_device_->shape());

    AddProgramToMeshWorkload(syncer_mesh_workload, std::move(sync_and_incr_program), devices);
    AddProgramToMeshWorkload(datacopy_mesh_workload, std::move(datacopy_program), devices);

    for (int i = 0; i < 50; i++) {
        mesh_device_->set_sub_device_stall_group({SubDeviceId{2}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, false);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), datacopy_mesh_workload, false);

        std::vector<uint32_t> src_vec(input_buf->size() / sizeof(uint32_t));
        std::iota(src_vec.begin(), src_vec.end(), i);
        // Block after this write on host, since the global semaphore update starting the
        // program goes through an independent path (UMD) and can go out of order wrt the
        // buffer data
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), input_buf, src_vec, true);

        for (auto device : mesh_device_->get_devices()) {
            llrt::write_hex_vec_to_core(
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
    auto waiter_mesh_workload = CreateMeshWorkload();
    auto syncer_mesh_workload = CreateMeshWorkload();
    auto incrementer_mesh_workload = CreateMeshWorkload();
    AddProgramToMeshWorkload(waiter_mesh_workload, std::move(waiter_program), devices);
    AddProgramToMeshWorkload(syncer_mesh_workload, std::move(syncer_program), devices);
    AddProgramToMeshWorkload(incrementer_mesh_workload, std::move(incrementer_program), devices);

    auto waiter_mesh_workload_1 = CreateMeshWorkload();
    auto syncer_mesh_workload_1 = CreateMeshWorkload();
    auto incrementer_mesh_workload_1 = CreateMeshWorkload();
    AddProgramToMeshWorkload(waiter_mesh_workload_1, std::move(waiter_program_1), devices);
    AddProgramToMeshWorkload(syncer_mesh_workload_1, std::move(syncer_program_1), devices);
    AddProgramToMeshWorkload(incrementer_mesh_workload_1, std::move(incrementer_program_1), devices);

    // Load SubDevice configs, run corresponding workloads, reset ... repeat
    for (uint32_t i = 0; i < num_iters; i++) {
        mesh_device_->load_sub_device_manager(sub_device_manager_0);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({SubDeviceId{0}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload, false);
        mesh_device_->reset_sub_device_stall_group();

        mesh_device_->load_sub_device_manager(sub_device_manager_1);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload_1, false);
        mesh_device_->set_sub_device_stall_group({SubDeviceId{0}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload_1, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload_1, false);
        mesh_device_->reset_sub_device_stall_group();
    }
    Finish(mesh_device_->mesh_command_queue());
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
    auto waiter_mesh_workload = CreateMeshWorkload();
    auto syncer_mesh_workload = CreateMeshWorkload();
    auto incrementer_mesh_workload = CreateMeshWorkload();
    AddProgramToMeshWorkload(waiter_mesh_workload, std::move(waiter_program), devices);
    AddProgramToMeshWorkload(syncer_mesh_workload, std::move(syncer_program), devices);
    AddProgramToMeshWorkload(incrementer_mesh_workload, std::move(incrementer_program), devices);

    // Run programs on sub-device manager 1
    for (uint32_t i = 0; i < k_num_iters; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({SubDeviceId{0}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload, false);
        mesh_device_->reset_sub_device_stall_group();
    }
    Finish(mesh_device_->mesh_command_queue());

    // Rerun programs on sub-device manager 2
    mesh_device_->load_sub_device_manager(sub_device_manager_2);
    for (uint32_t i = 0; i < k_num_iters; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_mesh_workload, false);
        mesh_device_->set_sub_device_stall_group({SubDeviceId{1}});
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, true);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_mesh_workload, false);
        mesh_device_->reset_sub_device_stall_group();
    }
    Finish(mesh_device_->mesh_command_queue());
}
}
}  // namespace
}  // namespace tt::tt_metal::distributed::test
