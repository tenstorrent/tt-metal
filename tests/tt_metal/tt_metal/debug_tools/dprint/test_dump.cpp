// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests for standalone debug dump utilities (api/debug/dump.h).

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

namespace dump_test {
constexpr uint32_t INPUT_CB_INDEX = 0;
constexpr uint32_t OUTPUT_CB_INDEX = 16;
constexpr size_t NUM_TILES = 1;
constexpr tt::DataFormat DATA_FORMAT = tt::DataFormat::Float16_b;
}  // namespace dump_test

class DumpTest : public DPrintMeshFixture {
protected:
    void SetUp() override { DPrintMeshFixture::SetUp(); }
    void TearDown() override { DPrintMeshFixture::TearDown(); }
};

static void run_dump_cb_test(DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    CoreCoord core = {0, 0};
    size_t tile_size = tt::tile_size(dump_test::DATA_FORMAT);
    size_t buffer_size = dump_test::NUM_TILES * tile_size;

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto& cq = mesh_device->mesh_command_queue();

    // DRAM buffers
    distributed::DeviceLocalBufferConfig local_config = {.page_size = buffer_size, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig buffer_config = {.size = buffer_size};
    auto input_dram = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());
    auto output_dram = distributed::MeshBuffer::create(buffer_config, local_config, mesh_device.get());

    // Circular buffers
    CircularBufferConfig input_cb_config =
        CircularBufferConfig(buffer_size, {{dump_test::INPUT_CB_INDEX, dump_test::DATA_FORMAT}})
            .set_page_size(dump_test::INPUT_CB_INDEX, tile_size);
    CreateCircularBuffer(program_, core, input_cb_config);

    CircularBufferConfig output_cb_config =
        CircularBufferConfig(buffer_size, {{dump_test::OUTPUT_CB_INDEX, dump_test::DATA_FORMAT}})
            .set_page_size(dump_test::OUTPUT_CB_INDEX, tile_size);
    CreateCircularBuffer(program_, core, output_cb_config);

    // Reader (NCRISC)
    auto reader_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {dump_test::INPUT_CB_INDEX}});

    // Writer (BRISC)
    auto writer_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {dump_test::OUTPUT_CB_INDEX}});

    // Compute with debug_dump_cb
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_dump_cb.cpp",
        core,
        ComputeConfig{.compile_args = {static_cast<uint32_t>(dump_test::NUM_TILES)}});

    // Runtime args
    SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(input_dram->address()), 0u, static_cast<uint32_t>(dump_test::NUM_TILES)});
    SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {static_cast<uint32_t>(output_dram->address()), 0u, static_cast<uint32_t>(dump_test::NUM_TILES)});

    // Input data
    auto input_data = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, dump_test::NUM_TILES * 1024);
    distributed::WriteShard(cq, input_dram, input_data, zero_coord);

    // Run
    fixture->RunProgram(mesh_device, workload);

    // Verify data integrity
    std::vector<uint32_t> output_data;
    distributed::ReadShard(cq, output_data, output_dram, zero_coord);
    EXPECT_EQ(input_data, output_data);

    // Verify DPRINT output contains CB dump
    EXPECT_TRUE(FileContainsAllStrings(
        fixture->dprint_file_name,
        {
            "CB0 sz=*",  // CB metadata
            "[0]*",      // Hex data line
        }));
}

TEST_F(DumpTest, DumpCB) {
    this->RunTestOnDevice(
        [](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            run_dump_cb_test(fixture, mesh_device);
        },
        this->devices_[0]);
}
