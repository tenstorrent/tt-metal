// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for the debug checkpoint API (api/debug/checkpoint.h).
// Verifies that DEBUG_CHECKPOINT correctly synchronizes all RISCs and dumps CB metadata.

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

namespace ckpt_test {
constexpr uint32_t INPUT_CB_INDEX = 0;
constexpr uint32_t OUTPUT_CB_INDEX = 16;
constexpr size_t NUM_TILES = 1;
constexpr tt::DataFormat DATA_FORMAT = tt::DataFormat::Float16_b;
}  // namespace ckpt_test

class CheckpointTest : public DPrintMeshFixture {
protected:
    void SetUp() override { DPrintMeshFixture::SetUp(); }
    void TearDown() override { DPrintMeshFixture::TearDown(); }
};

// Runs a program with reader -> compute (with checkpoint) -> writer on a single core.
// Returns true if the DPRINT output contains expected checkpoint strings.
static void run_checkpoint_test(
    DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    CoreCoord core = {0, 0};
    size_t tile_size = tt::tile_size(ckpt_test::DATA_FORMAT);
    size_t buffer_size = ckpt_test::NUM_TILES * tile_size;

    // Create program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto& cq = mesh_device->mesh_command_queue();

    // Create DRAM buffers
    distributed::DeviceLocalBufferConfig local_config = {.page_size = buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config = {.size = buffer_size};
    auto input_dram = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    auto output_dram = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    // Create circular buffers
    CircularBufferConfig input_cb_config =
        CircularBufferConfig(buffer_size, {{ckpt_test::INPUT_CB_INDEX, ckpt_test::DATA_FORMAT}})
            .set_page_size(ckpt_test::INPUT_CB_INDEX, tile_size);
    CreateCircularBuffer(program_, core, input_cb_config);

    CircularBufferConfig output_cb_config =
        CircularBufferConfig(buffer_size, {{ckpt_test::OUTPUT_CB_INDEX, ckpt_test::DATA_FORMAT}})
            .set_page_size(ckpt_test::OUTPUT_CB_INDEX, tile_size);
    CreateCircularBuffer(program_, core, output_cb_config);

    // Create reader kernel (NCRISC) with checkpoint support
    auto reader_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_checkpoint.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {ckpt_test::INPUT_CB_INDEX}});

    // Create writer kernel (BRISC) with checkpoint support
    auto writer_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_checkpoint.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {ckpt_test::OUTPUT_CB_INDEX}});

    // Create compute kernel with checkpoint
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_checkpoint.cpp",
        core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(ckpt_test::NUM_TILES)}});

    // Set runtime args
    SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(input_dram->address()), 0u, static_cast<uint32_t>(ckpt_test::NUM_TILES)});
    SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {static_cast<uint32_t>(output_dram->address()), 0u, static_cast<uint32_t>(ckpt_test::NUM_TILES)});

    // Generate and write input data
    auto input_data = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, ckpt_test::NUM_TILES * 1024);
    distributed::WriteShard(cq, input_dram, input_data, zero_coord);

    // Run program
    fixture->RunProgram(mesh_device, workload);

    // Read output and verify data moved correctly
    std::vector<uint32_t> output_data;
    distributed::ReadShard(cq, output_data, output_dram, zero_coord);
    EXPECT_EQ(input_data, output_data);

    // Verify checkpoint output in DPRINT log.
    // The checkpoint should produce output from multiple RISCs containing:
    // - Checkpoint header markers "=== CKPT 1 RISC <idx> ==="
    // - CB metadata lines for configured CBs
    // The exact RISC indices depend on which threads are active, but we should see
    // at least the compute thread markers and CB metadata.
    EXPECT_TRUE(FileContainsAllStrings(
        fixture->dprint_file_name,
        {
            "=== CKPT 1 RISC *",  // Checkpoint marker (glob pattern)
            "CB0 sz=*",           // Input CB metadata
        }));
}

TEST_F(CheckpointTest, BasicCheckpoint) {
    this->RunTestOnDevice(
        [](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            run_checkpoint_test(fixture, mesh_device);
        },
        this->devices_[0]);
}
