// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <stdint.h>
#include <stdlib.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_gold_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/distributed.hpp>

#include <tt-metalium/tt_metal.hpp>
#include "test_common.hpp"

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);

    uint32_t reader_input = 0;
    std::tie(reader_input, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--reader", 0);
    uint32_t writer_input = 0;
    std::tie(writer_input, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--writer", 0);
    uint32_t compute_input = 0;
    std::tie(compute_input, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--compute", 0);
    bool measure_cb_timings = false;
    std::tie(measure_cb_timings, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--measure-cb-timings");
    bool use_zone_counter = false;
    std::tie(use_zone_counter, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--use-zone-counter");
    uint32_t num_tiles = 2048;
    std::tie(num_tiles, input_args) =
        test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tiles", 2048);
    bool disable_cb_operation = false;
    std::tie(disable_cb_operation, input_args) =
        test_args::has_command_option_and_remaining_args(input_args, "--disable-cb-operation");

    bool pass = true;
    bool multibank = true;

    const char* op_id_to_op_define[] = {"add_tiles", "sub_tiles", "mul_tiles"};
    const char* op_id_to_op_type_define[] = {
        "EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWSUB", "EltwiseBinaryType::ELWMUL"};
    const char* op_id_to_op_name[] = {"ADD", "SUB", "MUL"};
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    auto& cq = mesh_device->mesh_command_queue();

    distributed::MeshWorkload mesh_workloads[] = {
        distributed::CreateMeshWorkload(), distributed::CreateMeshWorkload(), distributed::CreateMeshWorkload()};
    auto ops = {EltwiseOp::Enum::ADD};  // EltwiseOp::all();
    for (auto eltwise_op : ops) {
        log_info(LogTest, "====================================================================");
        log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);

        try {
            ////////////////////////////////////////////////////////////////////////////
            //                      Application Setup
            ////////////////////////////////////////////////////////////////////////////
            tt_metal::Program program = tt_metal::CreateProgram();
            auto& mesh_workload = mesh_workloads[eltwise_op];
            CoreCoord core = {0, 0};

            uint32_t single_tile_size = 2 * 1024;
            uint32_t dram_buffer_size =
                single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
            uint32_t page_size = single_tile_size;
            if (not multibank) {
                page_size = dram_buffer_size;
            }

            distributed::DeviceLocalBufferConfig device_local_config{
                .page_size = page_size,
                .buffer_type = tt_metal::BufferType::DRAM,
            };

            distributed::ReplicatedBufferConfig buffer_config{
                .size = dram_buffer_size,
            };
            auto src0_dram_buffer =
                distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
            uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
            auto src1_dram_buffer =
                distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
            uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
            auto dst_dram_buffer =
                distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
            uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

            uint32_t src0_cb_index = tt::CBIndex::c_0;
            uint32_t num_input_tiles = 2;
            tt_metal::CircularBufferConfig cb_src0_config =
                tt_metal::CircularBufferConfig(
                    num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src0_cb_index, single_tile_size);
            tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t src1_cb_index = tt::CBIndex::c_1;
            tt_metal::CircularBufferConfig cb_src1_config =
                tt_metal::CircularBufferConfig(
                    num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src1_cb_index, single_tile_size);
            tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

            uint32_t ouput_cb_index = tt::CBIndex::c_16;
            uint32_t num_output_tiles = 2;
            tt_metal::CircularBufferConfig cb_output_config =
                tt_metal::CircularBufferConfig(
                    num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(ouput_cb_index, single_tile_size);
            tt_metal::CreateCircularBuffer(program, core, cb_output_config);

            std::map<std::string, std::string> reader_defines;
            uint32_t reader_wait_time = 0;
            if (reader_input == 9999) {
                reader_defines["READER_NOC"] = "1";
            } else if (reader_input > 0) {
                reader_defines["READER_RISCV_WAIT"] = "1";
                reader_wait_time = reader_input;
            }
            if (!measure_cb_timings) {
                reader_defines["MEASURE_CB_TIMINGS_SKIP"] = "1";
            }
            if (use_zone_counter) {
                reader_defines["USE_ZONE_COUNTER"] = "1";
            }
            if (disable_cb_operation) {
                reader_defines["DISABLE_CB_OPERATION"] = "1";
            }

            std::map<std::string, std::string> writer_defines;
            uint32_t writer_wait_time = 0;
            if (writer_input == 9999) {
                writer_defines["WRITER_NOC"] = "1";
            } else if (writer_input > 0) {
                writer_defines["WRITER_RISCV_WAIT"] = "1";
                writer_wait_time = writer_input;
            }
            if (!measure_cb_timings) {
                writer_defines["MEASURE_CB_TIMINGS_SKIP"] = "1";
            }
            if (use_zone_counter) {
                writer_defines["USE_ZONE_COUNTER"] = "1";
            }
            if (disable_cb_operation) {
                writer_defines["DISABLE_CB_OPERATION"] = "1";
            }

            auto binary_reader_kernel = tt_metal::CreateKernel(
                program,
                multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp"
                          : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
                core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .defines = reader_defines});

            auto unary_writer_kernel = tt_metal::CreateKernel(
                program,
                multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                          : "tt_metal/kernels/dataflow/writer_unary.cpp",
                core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt_metal::NOC::RISCV_0_default,
                    .defines = writer_defines});

            vector<uint32_t> compute_kernel_args = {};

            std::map<std::string, std::string> compute_defines = {
                {"ELTWISE_OP", op_id_to_op_define[eltwise_op]},
                {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]}};

            uint32_t compute_wait_time = 0;
            if (compute_input == 9999) {
                compute_defines["COMPUTE_PROCESS"] = "1";
            } else if (compute_input > 0) {
                compute_defines["COMPUTE_RISCV_WAIT"] = "1";
                compute_wait_time = compute_input;
            }
            if (!measure_cb_timings) {
                compute_defines["MEASURE_CB_TIMINGS_SKIP"] = "1";
            }
            if (use_zone_counter) {
                compute_defines["USE_ZONE_COUNTER"] = "1";
            }
            if (disable_cb_operation) {
                compute_defines["DISABLE_CB_OPERATION"] = "1";
            }

            auto eltwise_binary_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/kernels/compute/eltwise_binary.cpp",
                core,
                tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = compute_defines});

            SetRuntimeArgs(program, eltwise_binary_kernel, core, {num_tiles, 1, 0, compute_wait_time});

            const std::array<uint32_t, 8> reader_args = {
                dram_buffer_src0_addr, 0, num_tiles, dram_buffer_src1_addr, 0, num_tiles, 0, reader_wait_time};

            const std::array<uint32_t, 4> writer_args = {dram_buffer_dst_addr, 0, num_tiles, writer_wait_time};

            SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
            SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

            distributed::AddProgramToMeshWorkload(
                mesh_workload, std::move(program), distributed::MeshCoordinateRange(mesh_device->shape()));
            ////////////////////////////////////////////////////////////////////////////
            //                      Compile Application
            ////////////////////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////////////////////
            //                      Execute Application
            ////////////////////////////////////////////////////////////////////////////
            std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
                dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
            distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, false);

            std::vector<uint32_t> src1_vec;
            if (eltwise_op == EltwiseOp::MUL) {
                // TODO(AP): this doesn't provide very good coverage
                // switch to a better test with different values like in reduce
                src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1.0f);
            } else {
                src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
            }
            distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, false);

            distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
            std::vector<uint32_t> result_vec;
            distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0));

            ////////////////////////////////////////////////////////////////////////////
            //                      Validation & Teardown
            ////////////////////////////////////////////////////////////////////////////

            pass &= (src0_vec == result_vec);

        } catch (const std::exception& e) {
            pass = false;
            // Capture the exception error message
            log_error(LogTest, "{}", e.what());
            // Capture system call errors that may have returned from driver/kernel
            log_error(LogTest, "System error message: {}", std::strerror(errno));
        }
    }  // for EltwiseOp::all()
    mesh_device->close();

    // if (pass) {
    //     log_info(LogTest, "Test Passed");
    // } else {
    //     TT_THROW("Test Failed");
    // }

    // TT_FATAL(pass, "Error");

    return 0;
}
