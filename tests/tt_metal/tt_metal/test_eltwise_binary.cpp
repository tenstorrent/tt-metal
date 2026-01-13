// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/command_queue_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <array>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include "test_gold_impls.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

const char* op_id_to_op_define[] = {"add_tiles", "sub_tiles", "mul_tiles"};
const char* op_id_to_op_type_define[] = {
    "EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWSUB", "EltwiseBinaryType::ELWMUL"};
const char* op_id_to_op_name[] = {"ADD", "SUB", "MUL"};

void run_eltwise_binary_test(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, distributed::MeshCommandQueue& cq, int eltwise_op) {
    bool multibank = true;

    log_info(LogTest, "====================================================================");
    log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);

    Program program = CreateProgram();
    distributed::MeshWorkload mesh_workload;
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    uint32_t page_size = single_tile_size;
    if (!multibank) {
        page_size = dram_buffer_size;
    }

    distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_size,
        .buffer_type = BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig buffer_config{
        .size = dram_buffer_size,
    };
    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
    uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(src1_dram_buffer).append_to(reader_compile_time_args);
    auto binary_reader_kernel = CreateKernel(
        program,
        multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp"
                  : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
    auto unary_writer_kernel = CreateKernel(
        program,
        multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                  : "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {};

    std::map<std::string, std::string> binary_defines = {
        {"ELTWISE_OP", op_id_to_op_define[eltwise_op]}, {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]}};
    auto eltwise_binary_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

    SetRuntimeArgs(program, eltwise_binary_kernel, core, {2048, 1});

    const std::array<uint32_t, 7> reader_args = {
        dram_buffer_src0_addr, 0, num_tiles, dram_buffer_src1_addr, 0, num_tiles, 0};

    const std::array<uint32_t, 3> writer_args = {dram_buffer_dst_addr, 0, num_tiles};

    SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
    SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

    mesh_workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

    // Execute
    std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);

    std::vector<uint32_t> src1_vec;
    if (eltwise_op == static_cast<int>(EltwiseOp::MUL)) {
        src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1.0f);
    } else {
        src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
    }
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    std::vector<uint32_t> result_vec;
    distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0));

    // Validation
    EXPECT_EQ(src0_vec, result_vec);
}

}  // namespace

TEST_F(UnitMeshCQSingleCardFixture, EltwiseBinaryAdd) {
    run_eltwise_binary_test(devices_[0], devices_[0]->mesh_command_queue(), static_cast<int>(EltwiseOp::ADD));
}

TEST_F(UnitMeshCQSingleCardFixture, EltwiseBinarySub) {
    run_eltwise_binary_test(devices_[0], devices_[0]->mesh_command_queue(), static_cast<int>(EltwiseOp::SUB));
}

TEST_F(UnitMeshCQSingleCardFixture, EltwiseBinaryMul) {
    run_eltwise_binary_test(devices_[0], devices_[0]->mesh_command_queue(), static_cast<int>(EltwiseOp::MUL));
}
