// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <boost/move/utility_core.hpp>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt_stl/indestructible.hpp>
#include <array>
#include <cstdlib>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "env_lib.hpp"
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/strong_type.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include "tests/tt_metal/distributed/utils.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/tt_metal/dispatch/sub_device_test_utils.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal::distributed::test {
namespace {

// Helper functions that return MeshCoordinateRange spanning various parts of the T3000 device.
const MeshCoordinateRange& t3k_bottom_row() {
    static tt::stl::Indestructible<MeshCoordinateRange> bottom_row(MeshCoordinate{1, 0}, MeshCoordinate{1, 3});
    return bottom_row.get();
}

const MeshCoordinateRange& t3k_top_row() {
    static tt::stl::Indestructible<MeshCoordinateRange> top_row(MeshCoordinate{0, 0}, MeshCoordinate{0, 3});
    return top_row.get();
}

const MeshCoordinateRange& t3k_full_grid() {
    static tt::stl::Indestructible<MeshCoordinateRange> full_grid(MeshCoordinate{0, 0}, MeshCoordinate{1, 3});
    return full_grid.get();
}

const MeshCoordinateRange& tg_full_grid() {
    static tt::stl::Indestructible<MeshCoordinateRange> full_grid(MeshCoordinate{0, 0}, MeshCoordinate{3, 7});
    return full_grid.get();
}

std::vector<MeshCoordinateRange> tg_all_devices() {
    std::vector<MeshCoordinateRange> devices = {};
    for (const auto& coord : tg_full_grid()) {
        devices.push_back(MeshCoordinateRange(coord, coord));
    }
    return devices;
}

// Define custom fixtures initializing a trace region on the MeshDevice
class MeshTraceTestSuite : public MeshDeviceFixtureBase {
protected:
    MeshTraceTestSuite() : MeshDeviceFixtureBase(Config{.num_cqs = 1, .trace_region_size = (64 << 20)}) {}
};

class MeshTraceTest2x4 : public MeshDeviceFixtureBase {
protected:
    MeshTraceTest2x4() :
        MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{2, 4}, .trace_region_size = (64 << 20)}) {}
};

class MeshTraceTest4x8 : public MeshDeviceFixtureBase {
protected:
    MeshTraceTest4x8() :
        MeshDeviceFixtureBase(Config{.mesh_shape = MeshShape{4, 8}, .trace_region_size = (64 << 20)}) {}
};

TEST_F(MeshTraceTestSuite, Sanity) {
    auto random_seed = 10;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    uint32_t num_workloads_per_trace = 5;
    uint32_t num_traces = 4;
    uint32_t num_iters = 10;
    uint32_t num_trace_setup_teardown_loops = 10;

    MeshCoordinateRange all_devices(mesh_device_->shape());
    for (int outer_loop = 0; outer_loop < num_trace_setup_teardown_loops; outer_loop++) {
        std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};
        for (int i = 0; i < num_workloads_per_trace * num_traces; i++) {
            auto workload = std::make_shared<MeshWorkload>();
            auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
                1, mesh_device_->compute_with_storage_grid_size(), seed);
            workload->add_program(all_devices, std::move(*programs[0]));
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
            mesh_workloads.push_back(workload);
        }

        std::vector<MeshTraceId> trace_ids = {};
        for (int trace_idx = 0; trace_idx < num_traces; trace_idx++) {
            auto trace_id = BeginTraceCapture(mesh_device_.get(), 0);
            for (int workload_idx = 0; workload_idx < num_workloads_per_trace; workload_idx++) {
                EnqueueMeshWorkload(
                    mesh_device_->mesh_command_queue(),
                    *mesh_workloads[(trace_idx * num_workloads_per_trace) + workload_idx],
                    false);
            }
            mesh_device_->end_mesh_trace(0, trace_id);
            trace_ids.push_back(trace_id);
        }

        for (int i = 0; i < num_iters; i++) {
            for (auto trace_id : trace_ids) {
                mesh_device_->replay_mesh_trace(0, trace_id, false);
            }
        }
        Finish(mesh_device_->mesh_command_queue());

        for (auto trace_id : trace_ids) {
            mesh_device_->release_mesh_trace(trace_id);
        }
    }
}

TEST_F(MeshTraceTest2x4, EltwiseBinaryMeshTrace) {
    std::vector<std::shared_ptr<MeshBuffer>> src0_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> src1_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> intermed_bufs_0 = {};
    std::vector<std::shared_ptr<MeshBuffer>> intermed_bufs_1 = {};
    std::vector<std::shared_ptr<MeshBuffer>> output_bufs = {};

    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();

    // Separate Mesh into top and bottom rows
    const MeshCoordinateRange& row_0 = t3k_top_row();
    const MeshCoordinateRange& row_1 = t3k_bottom_row();
    // Separate Mesh into 3 columns
    MeshCoordinateRange col_0({0, 0}, {1, 1});
    MeshCoordinateRange col_1({0, 2}, {1, 2});
    MeshCoordinateRange col_2({0, 3}, {1, 3});

    // Create first workload: running addition on top row and multiplication on bottom row
    auto programs = tt::tt_metal::distributed::test::utils::create_eltwise_bin_programs(
        mesh_device_, src0_bufs, src1_bufs, intermed_bufs_0);
    auto mesh_workload = MeshWorkload();
    mesh_workload.add_program(row_0, std::move(*programs[0]));
    mesh_workload.add_program(row_1, std::move(*programs[1]));
    // Create second workload: running addition on top row (src1 + intermed0) and multiplication on
    // bottom row (src1 * intermed0)
    auto programs_1 = tt::tt_metal::distributed::test::utils::create_eltwise_bin_programs(
        mesh_device_, intermed_bufs_0, src1_bufs, intermed_bufs_1);
    auto mesh_workload_1 = MeshWorkload();
    mesh_workload_1.add_program(row_0, std::move(*programs_1[1]));
    mesh_workload_1.add_program(row_1, std::move(*programs_1[0]));
    // Create third workload: running addition on 1st col (src1 + intermed1), multiplication on
    // second col (src1 * intermed1) and subtraction on the third col( src1 - intermed1)
    auto programs_2 = tt::tt_metal::distributed::test::utils::create_eltwise_bin_programs(
        mesh_device_, intermed_bufs_1, src1_bufs, output_bufs);
    auto mesh_workload_2 = MeshWorkload();
    mesh_workload_2.add_program(col_0, std::move(*programs_2[0]));
    mesh_workload_2.add_program(col_1, std::move(*programs_2[1]));
    mesh_workload_2.add_program(col_2, std::move(*programs_2[2]));

    // Initialize inputs
    std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(src0_bufs[0]->size(), 2);
    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(src1_bufs[0]->size(), 3);
    // Write inputs for all cores across the Mesh
    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), src0_bufs[(col_idx * worker_grid_size.y) + row_idx], src0_vec);
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), src1_bufs[(col_idx * worker_grid_size.y) + row_idx], src1_vec);
        }
    }
    // Compile workloads
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload_1, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload_2, false);
    // Capture trace
    auto trace_id = BeginTraceCapture(mesh_device_.get(), 0);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload_1, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload_2, false);
    mesh_device_->end_mesh_trace(0, trace_id);

    // Run workload multiple times
    for (int i = 0; i < 1000; i++) {
        mesh_device_->replay_mesh_trace(0, trace_id, false);
    }
    // Verify outputs
    std::vector<uint32_t> expected_values = {18, 18, 45, 12, 12, 12, 27, 6};
    for (std::size_t logical_y = 0; logical_y < mesh_device_->num_rows(); logical_y++) {
        for (std::size_t logical_x = 0; logical_x < mesh_device_->num_cols(); logical_x++) {
            for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                    std::vector<bfloat16> dst_vec = {};
                    ReadShard(
                        mesh_device_->mesh_command_queue(),
                        dst_vec,
                        output_bufs[(col_idx * worker_grid_size.y) + row_idx],
                        MeshCoordinate(logical_y, logical_x));
                    auto expected_value = expected_values[logical_x + (logical_y * mesh_device_->num_cols())];
                    for (auto val : dst_vec) {
                        EXPECT_EQ(static_cast<float>(val), expected_value);
                    }
                }
            }
        }
    }
    mesh_device_->release_mesh_trace(trace_id);
}

TEST_F(MeshTraceTestSuite, SyncWorkloadsOnSubDeviceTrace) {
    if (mesh_device_->num_devices() == 1) {
        GTEST_SKIP() << "Skipping test for a unit-size mesh device";
    }
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {2, 2}))});
    SubDevice sub_device_2(std::array{CoreRangeSet(std::vector{CoreRange({3, 3}, {3, 3}), CoreRange({4, 4}, {4, 4})})});

    uint32_t num_iters = 5;
    auto sub_device_manager = mesh_device_->create_sub_device_manager({sub_device_1, sub_device_2}, 3200);
    mesh_device_->load_sub_device_manager(sub_device_manager);

    // Create three variants of the same program set - will be traced on the Mesh differently
    auto [waiter_program_0, syncer_program_0, incrementer_program_0, global_sem_0] =
        create_basic_sync_program(mesh_device_.get(), sub_device_1, sub_device_2);

    auto [waiter_program_1, syncer_program_1, incrementer_program_1, global_sem_1] =
        create_basic_sync_program(mesh_device_.get(), sub_device_1, sub_device_2);

    auto [waiter_program_2, syncer_program_2, incrementer_program_2, global_sem_2] =
        create_basic_sync_program(mesh_device_.get(), sub_device_1, sub_device_2);

    uint32_t num_cols_in_workload = mesh_device_->num_cols() / 2;
    // Left column - first MeshWorkload set
    MeshCoordinateRange left_col({0, 0}, {mesh_device_->num_rows() - 1, num_cols_in_workload - 1});
    // Right column - second MeshWorkload set
    MeshCoordinateRange right_col(
        {0, num_cols_in_workload}, {mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1});
    // All devices: third MeshWorkload set
    MeshCoordinateRange all_devices(mesh_device_->shape());

    // Initialize and construct all MeshWorkloads running on different SubDevices
    auto waiter_0 = MeshWorkload();
    auto syncer_0 = MeshWorkload();
    auto incrementer_0 = MeshWorkload();

    auto waiter_1 = MeshWorkload();
    auto syncer_1 = MeshWorkload();
    auto incrementer_1 = MeshWorkload();

    auto waiter_2 = MeshWorkload();
    auto syncer_2 = MeshWorkload();
    auto incrementer_2 = MeshWorkload();

    waiter_0.add_program(left_col, std::move(waiter_program_0));
    syncer_0.add_program(left_col, std::move(syncer_program_0));
    incrementer_0.add_program(left_col, std::move(incrementer_program_0));

    waiter_1.add_program(right_col, std::move(waiter_program_1));
    syncer_1.add_program(right_col, std::move(syncer_program_1));
    incrementer_1.add_program(right_col, std::move(incrementer_program_1));

    waiter_2.add_program(all_devices, std::move(waiter_program_2));
    syncer_2.add_program(all_devices, std::move(syncer_program_2));
    incrementer_2.add_program(all_devices, std::move(incrementer_program_2));

    // Compile all MeshWorkloads
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_0, false);
    mesh_device_->set_sub_device_stall_group({{SubDeviceId{0}}});
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_0, true);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_0, false);
    mesh_device_->reset_sub_device_stall_group();
    Finish(mesh_device_->mesh_command_queue());

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_1, false);
    mesh_device_->set_sub_device_stall_group({{SubDeviceId{0}}});
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_1, true);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_1, false);
    mesh_device_->reset_sub_device_stall_group();
    Finish(mesh_device_->mesh_command_queue());

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_2, false);
    mesh_device_->set_sub_device_stall_group({{SubDeviceId{0}}});
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_2, true);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_2, false);
    mesh_device_->reset_sub_device_stall_group();
    Finish(mesh_device_->mesh_command_queue());

    // Capture trace
    auto trace_id = BeginTraceCapture(mesh_device_.get(), 0);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_0, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_0, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_0, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_1, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_1, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_1, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), waiter_2, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_2, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), incrementer_2, false);
    mesh_device_->end_mesh_trace(0, trace_id);

    // Run trace on all SubDevices in the Mesh
    for (uint32_t i = 0; i < num_iters; i++) {
        mesh_device_->replay_mesh_trace(0, trace_id, false);
    }
    Finish(mesh_device_->mesh_command_queue());
    mesh_device_->release_mesh_trace(trace_id);
}

TEST_F(MeshTraceTestSuite, DataCopyOnSubDevicesTrace) {
    if (mesh_device_->num_devices() == 1) {
        GTEST_SKIP() << "Skipping test for a unit-size mesh device";
    }
    // Create 4 SubDevices
    SubDevice sub_device_1(std::array{CoreRangeSet(CoreRange({0, 0}, {0, 0}))});  // Sync with host
    SubDevice sub_device_2(std::array{CoreRangeSet(CoreRange({1, 1}, {1, 1}))});  // Run datacopy
    SubDevice sub_device_3(std::array{CoreRangeSet(
        CoreRange({2, 2}, {2, 2}))});  // Dummy - use this for blocking operations when using persistent kernels
    SubDevice sub_device_4(std::array{CoreRangeSet(CoreRange({3, 3}, {3, 3}))});  // Run addition

    // Create and Load SubDeviceConfig on the mesh
    auto sub_device_manager =
        mesh_device_->create_sub_device_manager({sub_device_1, sub_device_2, sub_device_3, sub_device_4}, 3200);
    mesh_device_->load_sub_device_manager(sub_device_manager);

    // Create IO Buffers
    uint32_t single_tile_size = ::tt::tile_size(DataFormat::UInt32);
    uint32_t num_tiles = 32;
    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size * num_tiles, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = true};

    ReplicatedBufferConfig global_buffer_config{
        .size = single_tile_size * num_tiles,
    };
    auto input_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());
    auto output_buf = MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get());

    // Query coords for syncer, datacopy and addition workloads
    auto syncer_coord = sub_device_1.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto syncer_core = CoreRangeSet(CoreRange(syncer_coord, syncer_coord));
    auto syncer_core_phys = mesh_device_->worker_core_from_logical_core(syncer_coord);
    auto datacopy_coord = sub_device_2.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto datacopy_core = CoreRangeSet(CoreRange(datacopy_coord, datacopy_coord));
    auto datacopy_core_phys = mesh_device_->worker_core_from_logical_core(datacopy_coord);
    auto add_coord = sub_device_4.cores(HalProgrammableCoreType::TENSIX).ranges().at(0).start_coord;
    auto add_core = CoreRangeSet(CoreRange(add_coord, add_coord));
    auto add_core_phys = mesh_device_->worker_core_from_logical_core(add_coord);

    // Create global semaphore for syncing between programs
    auto all_cores = syncer_core.merge(datacopy_core).merge(add_core);
    auto global_sem = CreateGlobalSemaphore(mesh_device_.get(), all_cores, 0);

    // Program syncs with host and notifies downstream datacopy or addition program
    Program sync_and_incr_program = CreateProgram();
    auto sync_kernel = CreateKernel(
        sync_and_incr_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/sync_and_increment.cpp",
        syncer_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 3> sync_rt_args = {global_sem.address(), datacopy_core_phys.x, datacopy_core_phys.y};
    SetRuntimeArgs(sync_and_incr_program, sync_kernel, syncer_core, sync_rt_args);
    // Program copies data from dram once notified
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
    // Program copies data from DRAM, does addition in RISC once notified
    Program add_program = CreateProgram();
    auto add_kernel = CreateKernel(
        add_program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/sync_and_add.cpp",
        datacopy_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 9> add_rt_args = {
        global_sem.address(),
        0,
        0,
        input_buf->address(),
        output_buf->address(),
        num_tiles,
        add_core_phys.x,
        add_core_phys.y,
        1};
    SetRuntimeArgs(add_program, add_kernel, datacopy_core, add_rt_args);
    CreateCircularBuffer(add_program, datacopy_core, cb_src0_config);
    // Same program as above, but runs on different SubDevice. Reads from DRAM, once
    // notified by previous program
    Program add_program_2 = CreateProgram();
    auto add_kernel_2 = CreateKernel(
        add_program_2,
        "tests/tt_metal/tt_metal/test_kernels/misc/sub_device/sync_and_add.cpp",
        add_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    std::array<uint32_t, 9> add_rt_args_2 = {
        global_sem.address(), 0, 0, output_buf->address(), output_buf->address(), num_tiles, 0, 0, 2};
    SetRuntimeArgs(add_program_2, add_kernel_2, add_core, add_rt_args_2);
    CreateCircularBuffer(add_program_2, add_core, cb_src0_config);

    uint32_t num_cols_in_workload = mesh_device_->num_cols() / 2;
    MeshCoordinateRange devices(mesh_device_->shape());
    MeshCoordinateRange left_col({0, 0}, {mesh_device_->num_rows() - 1, num_cols_in_workload - 1});
    MeshCoordinateRange right_col(
        {0, num_cols_in_workload}, {mesh_device_->num_rows() - 1, mesh_device_->num_cols() - 1});

    // Create and initialize MeshWorkloads
    auto syncer_mesh_workload = MeshWorkload();
    auto datacopy_mesh_workload = MeshWorkload();
    auto add_mesh_workload = MeshWorkload();
    // Sync program goes to entire Mesh
    syncer_mesh_workload.add_program(devices, std::move(sync_and_incr_program));
    // Datacopy goes to top row
    datacopy_mesh_workload.add_program(left_col, std::move(datacopy_program));
    // First addition goes to bottom row
    datacopy_mesh_workload.add_program(right_col, std::move(add_program));
    // Second addition goes to bottom row
    add_mesh_workload.add_program(right_col, std::move(add_program_2));

    // Compile and load workloads
    mesh_device_->set_sub_device_stall_group({{SubDeviceId{2}}});
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), datacopy_mesh_workload, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), add_mesh_workload, false);

    for (auto* device : mesh_device_->get_devices()) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            device->id(), syncer_core_phys, std::vector<uint32_t>{1}, global_sem.address());
    }

    // Capture Trace
    auto trace_id = BeginTraceCapture(mesh_device_.get(), 0);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), syncer_mesh_workload, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), datacopy_mesh_workload, false);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), add_mesh_workload, false);
    mesh_device_->end_mesh_trace(0, trace_id);
    // Run trace and verify outputs
    for (int i = 0; i < 50; i++) {
        mesh_device_->replay_mesh_trace(0, trace_id, false);

        std::vector<uint32_t> src_vec(input_buf->size() / sizeof(uint32_t));
        std::iota(src_vec.begin(), src_vec.end(), i);
        // Block after this write on host, since the global semaphore update starting the
        // program goes through an independent path (UMD) and can go out of order wrt the
        // buffer data
        mesh_device_->set_sub_device_stall_group({{SubDeviceId{2}}});
        EnqueueWriteMeshBuffer(mesh_device_->mesh_command_queue(), input_buf, src_vec, true);

        for (auto* device : mesh_device_->get_devices()) {
            tt::tt_metal::MetalContext::instance().get_cluster().write_core(
                device->id(), syncer_core_phys, std::vector<uint32_t>{1}, global_sem.address());
        }
        mesh_device_->reset_sub_device_stall_group();
        for (const auto& device_coord : left_col) {
            std::vector<uint32_t> dst_vec;
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, output_buf, device_coord);
            EXPECT_EQ(dst_vec, src_vec);
        }

        for (const auto& device_coord : right_col) {
            std::vector<uint32_t> dst_vec;
            ReadShard(mesh_device_->mesh_command_queue(), dst_vec, output_buf, device_coord);
            for (int j = 0; j < dst_vec.size(); j++) {
                EXPECT_EQ(dst_vec[j], src_vec[j] + 3);
            }
        }
    }
    mesh_device_->release_mesh_trace(trace_id);
}

TEST_F(MeshTraceTestSuite, MeshTraceAsserts) {
    auto random_seed = 10;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);
    MeshCoordinateRange all_devices(mesh_device_->shape());
    auto workload = std::make_shared<MeshWorkload>();
    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        1, mesh_device_->compute_with_storage_grid_size(), seed);
    workload->add_program(all_devices, std::move(*programs[0]));
    auto trace_id = BeginTraceCapture(mesh_device_.get(), 0);
    EXPECT_THROW(EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, true), std::runtime_error);
    EXPECT_THROW(Finish(mesh_device_->mesh_command_queue()), std::runtime_error);
    mesh_device_->end_mesh_trace(0, trace_id);
}

TEST_F(MeshTraceTest2x4, NonConvexGridTrace) {
    // Create a non-convex grid pattern by using non-contiguous device ranges
    MeshCoordinateRange top_left_corner({0, 0}, {0, 1});      // Devices (0,0) to (0,1)
    MeshCoordinateRange bottom_right_corner({1, 2}, {1, 3});  // Devices (1,2) to (1,3)

    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        2, mesh_device_->compute_with_storage_grid_size(), 0);

    auto mesh_workload = std::make_shared<MeshWorkload>();
    mesh_workload->add_program(top_left_corner, std::move(*programs[0]));
    mesh_workload->add_program(bottom_right_corner, std::move(*programs[1]));

    // Compile workload first
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());

    // Capture trace with non-convex grid
    auto trace_id = BeginTraceCapture(mesh_device_.get(), 0);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *mesh_workload, false);
    mesh_device_->end_mesh_trace(0, trace_id);

    // Replay trace multiple times to ensure it works correctly
    for (int i = 0; i < 100; i++) {
        mesh_device_->replay_mesh_trace(0, trace_id, false);
    }
    Finish(mesh_device_->mesh_command_queue());

    // Cleanup
    mesh_device_->release_mesh_trace(trace_id);
}

// Sweep Tests on T3K and TG
void run_heterogenous_trace_sweep(
    const std::shared_ptr<MeshDevice>& mesh_device,
    const std::vector<std::vector<MeshCoordinateRange>>& workload_grids) {
    auto random_seed = 10;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    uint32_t num_workloads = 10;

    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    for (const auto& workload_grid : workload_grids) {
        for (int i = 0; i < num_workloads; i++) {
            auto workload = std::make_shared<MeshWorkload>();
            for (const auto& program_grid : workload_grid) {
                auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
                    1, mesh_device->compute_with_storage_grid_size(), seed);
                workload->add_program(program_grid, std::move(*programs[0]));
            }
            EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *workload, false);
            mesh_workloads.push_back(workload);
        }
    }
    auto trace_id = BeginTraceCapture(mesh_device.get(), 0);
    for (auto& workload : mesh_workloads) {
        EnqueueMeshWorkload(mesh_device->mesh_command_queue(), *workload, false);
    }
    mesh_device->end_mesh_trace(0, trace_id);
    for (int i = 0; i < 50; i++) {
        mesh_device->replay_mesh_trace(0, trace_id, false);
    }
    Finish(mesh_device->mesh_command_queue());
    mesh_device->release_mesh_trace(trace_id);
}

class MeshTraceSweepTest2x4 : public MeshTraceTest2x4,
                              public testing::WithParamInterface<std::vector<std::vector<MeshCoordinateRange>>> {};

class MeshTraceSweepTest4x8 : public MeshTraceTest4x8,
                              public testing::WithParamInterface<std::vector<std::vector<MeshCoordinateRange>>> {};

TEST_P(MeshTraceSweepTest2x4, Sweep) { run_heterogenous_trace_sweep(mesh_device_, GetParam()); }

TEST_P(MeshTraceSweepTest4x8, Sweep) { run_heterogenous_trace_sweep(mesh_device_, GetParam()); }

INSTANTIATE_TEST_SUITE_P(
    MeshTraceSweepTest2x4Tests,
    MeshTraceSweepTest2x4,
    ::testing::Values(
        std::vector<std::vector<MeshCoordinateRange>>({
            {t3k_full_grid()},
            // Run on single center column:
            {MeshCoordinateRange({0, 1}, {1, 1})},
            // Run on single device - top row, center:
            {MeshCoordinateRange({0, 2}, {0, 2})},
            // Run on bottom right device:
            {MeshCoordinateRange({1, 3}, {1, 3})},
            // Run on top left device:
            {MeshCoordinateRange({0, 0}, {0, 0})},
            {t3k_full_grid()},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            {t3k_full_grid()},
            // Split grid into 4 columns:
            {MeshCoordinateRange({0, 1}, {1, 1}),
             MeshCoordinateRange({0, 2}, {1, 2}),
             MeshCoordinateRange({0, 3}, {1, 3}),
             MeshCoordinateRange({0, 0}, {1, 0})},
            // Split grid into 2 rows:
            {t3k_top_row(), t3k_bottom_row()},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            {t3k_full_grid()},
            // Split grid into 2 rows:
            {t3k_top_row(), t3k_bottom_row()},
            // Split grid into 2 columns:
            {MeshCoordinateRange({0, 0}, {1, 1}),  //
             MeshCoordinateRange({0, 2}, {1, 3})},
            // Split grid into 3 columns:
            {MeshCoordinateRange({0, 0}, {1, 1}),  //
             MeshCoordinateRange({0, 2}, {1, 2}),  //
             MeshCoordinateRange({0, 3}, {1, 3})},
            // Split grid into 4 columns:
            {MeshCoordinateRange({0, 0}, {1, 0}),  //
             MeshCoordinateRange({0, 1}, {1, 1}),  //
             MeshCoordinateRange({0, 2}, {1, 2}),  //
             MeshCoordinateRange({0, 3}, {1, 3})},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            {t3k_full_grid()},
            // Run on individual devices:
            {MeshCoordinateRange({0, 0}, {0, 0}),
             MeshCoordinateRange({0, 1}, {0, 1}),
             MeshCoordinateRange({0, 2}, {0, 2}),
             MeshCoordinateRange({0, 3}, {0, 3}),
             MeshCoordinateRange({1, 0}, {1, 0}),
             MeshCoordinateRange({1, 1}, {1, 1}),
             MeshCoordinateRange({1, 2}, {1, 2}),
             MeshCoordinateRange({1, 3}, {1, 3})},
            // Run on 2 center columns:
            {MeshCoordinateRange({0, 1}, {1, 2})},
            // Run on single center column:
            {MeshCoordinateRange({0, 2}, {1, 2})},
            // Run on 2 devices on the bottom row:
            {MeshCoordinateRange({1, 1}, {1, 2})},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            // Split grid into 4 columns:
            {MeshCoordinateRange({0, 0}, {1, 0}),
             MeshCoordinateRange({0, 1}, {1, 1}),
             MeshCoordinateRange({0, 2}, {1, 2}),
             MeshCoordinateRange({0, 3}, {1, 3})},
            // Split grid into 2 rows:
            {t3k_top_row(), t3k_bottom_row()},
            {t3k_full_grid()},
            {t3k_top_row()},
            {t3k_bottom_row()},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            {t3k_top_row()},
            {t3k_bottom_row()},
            // Run on left most column only:
            {MeshCoordinateRange({0, 0}, {1, 0})},
            // Run on right most 3-columns only:
            {MeshCoordinateRange({0, 1}, {1, 3})},
            // Run on left most 2-columns only:
            {MeshCoordinateRange({0, 0}, {1, 1})},
            // Full grid:
            {MeshCoordinateRange({0, 0}, {1, 3})},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            // Run on individual devices:
            {MeshCoordinateRange({0, 0}, {0, 0}),
             MeshCoordinateRange({0, 1}, {0, 1}),
             MeshCoordinateRange({0, 2}, {0, 2}),
             MeshCoordinateRange({0, 3}, {0, 3}),
             MeshCoordinateRange({1, 0}, {1, 0}),
             MeshCoordinateRange({1, 1}, {1, 1}),
             MeshCoordinateRange({1, 2}, {1, 2}),
             MeshCoordinateRange({1, 3}, {1, 3})},
            {t3k_top_row()},
            {t3k_bottom_row()},
            {t3k_full_grid()},
        })));

INSTANTIATE_TEST_SUITE_P(
    MeshTraceSweepTest4x8Tests,
    MeshTraceSweepTest4x8,
    ::testing::Values(
        std::vector<std::vector<MeshCoordinateRange>>({
            // Run on full grid
            {tg_full_grid()},
            // Run on top two rows
            {MeshCoordinateRange({0, 0}, {1, 7})},
            // Run on bottom two rows:
            {MeshCoordinateRange({2, 0}, {3, 7})},
            // Run on middle column
            {MeshCoordinateRange({0, 1}, {3, 2})},
            {tg_full_grid()},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            // Split into 4 columns
            {MeshCoordinateRange({0, 0}, {3, 1}),
             MeshCoordinateRange({0, 2}, {3, 3}),
             MeshCoordinateRange({0, 4}, {3, 5}),
             MeshCoordinateRange({0, 6}, {3, 7})},
            // Run on full grid
            {tg_full_grid()},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({
            // Run on full grid
            {tg_full_grid()},
            // Run on middle columns
            {MeshCoordinateRange({0, 1}, {3, 2})},
            // Split into 4 rows
            {MeshCoordinateRange({0, 0}, {0, 7}),
             MeshCoordinateRange({1, 0}, {1, 7}),
             MeshCoordinateRange({2, 0}, {2, 7}),
             MeshCoordinateRange({3, 0}, {3, 7})},
            // Split into 8 columns
            {MeshCoordinateRange({0, 0}, {3, 0}),
             MeshCoordinateRange({0, 1}, {3, 1}),
             MeshCoordinateRange({0, 2}, {3, 2}),
             MeshCoordinateRange({0, 3}, {3, 3}),
             MeshCoordinateRange({0, 4}, {3, 4}),
             MeshCoordinateRange({0, 5}, {3, 5}),
             MeshCoordinateRange({0, 6}, {3, 6}),
             MeshCoordinateRange({0, 7}, {3, 7})},
        }),
        std::vector<std::vector<MeshCoordinateRange>>({// Run on full grid
                                                       {tg_full_grid()},
                                                       // Run on top left device
                                                       {MeshCoordinateRange({0, 0}, {0, 0})},
                                                       // Run on individual devices
                                                       tg_all_devices(),
                                                       // Run on top 3 rows (split into 2)
                                                       {MeshCoordinateRange({0, 0}, {0, 7}),
                                                        MeshCoordinateRange({1, 0}, {2, 7})},
                                                       // Run on all columns but the first
                                                       {MeshCoordinateRange({0, 1}, {3, 7})}}),
        std::vector<std::vector<MeshCoordinateRange>>({// Run on individual devices
                                                       tg_all_devices(),
                                                       // Run on full grid
                                                       {tg_full_grid()},
                                                       // Run on middle columns
                                                       {MeshCoordinateRange({0, 1}, {3, 3})}})));

}  // namespace
}  // namespace tt::tt_metal::distributed::test
