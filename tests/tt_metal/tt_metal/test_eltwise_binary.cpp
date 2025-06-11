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
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

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
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    CommandQueue& cq = device->command_queue();

    Program programs[] = {tt_metal::CreateProgram(), tt_metal::CreateProgram(), tt_metal::CreateProgram()};

    auto ops = EltwiseOp::all();
    for (auto eltwise_op : ops) {
        log_info(LogTest, "====================================================================");
        log_info(LogTest, "======= Running eltwise_binary test for op={}", op_id_to_op_name[eltwise_op]);

        try {
            ////////////////////////////////////////////////////////////////////////////
            //                      Application Setup
            ////////////////////////////////////////////////////////////////////////////
            // tt_metal::Program program = tt_metal::CreateProgram();
            tt_metal::Program& program = programs[eltwise_op];

            CoreCoord core = {0, 0};

            uint32_t single_tile_size = 2 * 1024;
            uint32_t num_tiles = 2048;
            uint32_t dram_buffer_size =
                single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
            uint32_t page_size = single_tile_size;
            if (not multibank) {
                page_size = dram_buffer_size;
            }
            tt_metal::InterleavedBufferConfig dram_config{
                .device = device,
                .size = dram_buffer_size,
                .page_size = page_size,
                .buffer_type = tt_metal::BufferType::DRAM};

            auto src0_dram_buffer = CreateBuffer(dram_config);
            uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
            auto src1_dram_buffer = CreateBuffer(dram_config);
            uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();
            auto dst_dram_buffer = CreateBuffer(dram_config);

            uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

            uint32_t src0_cb_index = tt::CBIndex::c_0;
            uint32_t num_input_tiles = 2;
            tt_metal::CircularBufferConfig cb_src0_config =
                tt_metal::CircularBufferConfig(
                    num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src0_cb_index, single_tile_size);
            auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t src1_cb_index = tt::CBIndex::c_1;
            tt_metal::CircularBufferConfig cb_src1_config =
                tt_metal::CircularBufferConfig(
                    num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src1_cb_index, single_tile_size);
            auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

            uint32_t ouput_cb_index = tt::CBIndex::c_16;
            uint32_t num_output_tiles = 2;
            tt_metal::CircularBufferConfig cb_output_config =
                tt_metal::CircularBufferConfig(
                    num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(ouput_cb_index, single_tile_size);
            auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

            auto binary_reader_kernel = tt_metal::CreateKernel(
                program,
                multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp"
                          : "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
                core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

            auto unary_writer_kernel = tt_metal::CreateKernel(
                program,
                multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                          : "tt_metal/kernels/dataflow/writer_unary.cpp",
                core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            vector<uint32_t> compute_kernel_args = {};

            bool fp32_dest_acc_en = false;
            bool math_approx_mode = false;
            std::map<std::string, std::string> binary_defines = {
                {"ELTWISE_OP", op_id_to_op_define[eltwise_op]},
                {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]}};
            auto eltwise_binary_kernel = tt_metal::CreateKernel(
                program,
                "tt_metal/kernels/compute/eltwise_binary.cpp",
                core,
                tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

            SetRuntimeArgs(program, eltwise_binary_kernel, core, {2048, 1});

            ////////////////////////////////////////////////////////////////////////////
            //                      Compile Application
            ////////////////////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////////////////////
            //                      Execute Application
            ////////////////////////////////////////////////////////////////////////////
            std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
                dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());

            EnqueueWriteBuffer(cq, std::ref(src0_dram_buffer), src0_vec, false);

            std::vector<uint32_t> src1_vec;
            if (eltwise_op == EltwiseOp::MUL) {
                // TODO(AP): this doesn't provide very good coverage
                // switch to a better test with different values like in reduce
                src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1.0f);
            } else {
                src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
            }

            EnqueueWriteBuffer(cq, std::ref(src1_dram_buffer), src1_vec, false);

            const std::array<uint32_t, 7> reader_args = {
                dram_buffer_src0_addr,
                0,
                num_tiles,
                dram_buffer_src1_addr,
                0,
                num_tiles,
                0};

            const std::array<uint32_t, 3> writer_args = {
                dram_buffer_dst_addr, 0, num_tiles};

            SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
            SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

            EnqueueProgram(cq, program, false);
            std::vector<uint32_t> result_vec;
            EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

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

    pass &= tt_metal::CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
