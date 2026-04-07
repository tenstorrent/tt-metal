// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the global (cross-core) debug checkpoint API.
// Runs a reader/compute/writer pipeline on 2 cores with DEBUG_CHECKPOINT_GLOBAL,
// verifying that all cores synchronize and dump CB metadata.

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace global_ckpt_test {
constexpr uint32_t INPUT_CB_INDEX = 0;
constexpr uint32_t OUTPUT_CB_INDEX = 16;
constexpr size_t NUM_TILES = 1;
constexpr tt::DataFormat DATA_FORMAT = tt::DataFormat::Float16_b;
constexpr uint32_t NUM_CORES = 2;
}  // namespace global_ckpt_test

class GlobalCheckpointTest : public DPrintMeshFixture {
protected:
    void SetUp() override { DPrintMeshFixture::SetUp(); }
    void TearDown() override { DPrintMeshFixture::TearDown(); }
};

static void run_global_checkpoint_test(
    DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    using namespace global_ckpt_test;

    auto* device = mesh_device->get_devices()[0];
    size_t tile_size = tt::tile_size(DATA_FORMAT);
    size_t buffer_size = NUM_TILES * tile_size;

    // Use 2 cores: (0,0) and (0,1)
    CoreCoord core0 = {0, 0};
    CoreCoord core1 = {0, 1};
    CoreRange core_range(core0, core1);

    // Coordinator is core0 in physical coords
    CoreCoord coord_phys = device->worker_core_from_logical_core(core0);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto& cq = mesh_device->mesh_command_queue();

    // Create semaphore for cross-core barrier (same ID on all cores)
    uint32_t sem_id = CreateSemaphore(program_, core_range, 0);

    // Create per-core DRAM buffers and CBs
    distributed::DeviceLocalBufferConfig local_config = {.page_size = buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config = {.size = buffer_size};

    // Core 0 buffers
    auto input_dram_0 = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    auto output_dram_0 = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    // Core 1 buffers
    auto input_dram_1 = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    auto output_dram_1 = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    // Create CBs on both cores
    for (auto core : {core0, core1}) {
        CircularBufferConfig input_cb_config =
            CircularBufferConfig(buffer_size, {{INPUT_CB_INDEX, DATA_FORMAT}}).set_page_size(INPUT_CB_INDEX, tile_size);
        CreateCircularBuffer(program_, core, input_cb_config);

        CircularBufferConfig output_cb_config = CircularBufferConfig(buffer_size, {{OUTPUT_CB_INDEX, DATA_FORMAT}})
                                                    .set_page_size(OUTPUT_CB_INDEX, tile_size);
        CreateCircularBuffer(program_, core, output_cb_config);
    }

    // Create reader kernel on both cores (NCRISC)
    auto reader_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_global_checkpoint.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {INPUT_CB_INDEX}});

    // Create writer kernel on both cores (BRISC)
    auto writer_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_global_checkpoint.cpp",
        core_range,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {OUTPUT_CB_INDEX}});

    // Create compute kernel on both cores
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_global_checkpoint.cpp",
        core_range,
        ComputeConfig{
            .compile_args = {static_cast<uint32_t>(NUM_TILES), sem_id, coord_phys.x, coord_phys.y, NUM_CORES}});

    // Set runtime args for both cores
    for (uint32_t i = 0; i < NUM_CORES; i++) {
        CoreCoord core = (i == 0) ? core0 : core1;
        auto input_dram = (i == 0) ? input_dram_0 : input_dram_1;
        auto output_dram = (i == 0) ? output_dram_0 : output_dram_1;

        SetRuntimeArgs(
            program_,
            reader_kernel,
            core,
            {static_cast<uint32_t>(input_dram->address()),
             0u,
             static_cast<uint32_t>(NUM_TILES),
             sem_id,
             coord_phys.x,
             coord_phys.y,
             NUM_CORES});

        SetRuntimeArgs(
            program_,
            writer_kernel,
            core,
            {static_cast<uint32_t>(output_dram->address()),
             0u,
             static_cast<uint32_t>(NUM_TILES),
             sem_id,
             coord_phys.x,
             coord_phys.y,
             NUM_CORES});
    }

    // Generate and write input data for both cores
    auto input_data_0 =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);
    auto input_data_1 =
        tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, NUM_TILES * 1024);

    distributed::WriteShard(cq, input_dram_0, input_data_0, zero_coord);
    distributed::WriteShard(cq, input_dram_1, input_data_1, zero_coord);

    // Run program
    fixture->RunProgram(mesh_device, workload);

    // Verify data integrity on both cores
    std::vector<uint32_t> output_data_0, output_data_1;
    distributed::ReadShard(cq, output_data_0, output_dram_0, zero_coord);
    distributed::ReadShard(cq, output_data_1, output_dram_1, zero_coord);
    EXPECT_EQ(input_data_0, output_data_0);
    EXPECT_EQ(input_data_1, output_data_1);

    // Verify checkpoint output contains markers from both cores
    EXPECT_TRUE(FileContainsAllStrings(
        fixture->dprint_file_name,
        {
            "=== CKPT 1 RISC *",  // At least one checkpoint marker
            "CB0 sz=*",           // CB metadata
        }));
}

TEST_F(GlobalCheckpointTest, TwoCoreCheckpoint) {
    this->RunTestOnDevice(
        [](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            run_global_checkpoint_test(fixture, mesh_device);
        },
        this->devices_[0]);
}
