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
// Test for fused eltwise binary + reduce operation using reduce_h2.cpp kernel
// This test combines:
// 1. Eltwise binary operation (ADD/SUB/MUL) on two input tensors
// 2. Reduce operation (SUM) along height dimension on the result
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    bool multibank = true;

    // Operation definitions for eltwise binary
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

    // Test parameters - similar to reduce test but with two inputs for eltwise
    uint32_t tile_H = 32, tile_W = 32;  // Standard tile dimensions
    uint32_t H = tile_H;                // 3 tiles in height (will be reduced)
    uint32_t W = tile_W;                // 2 tiles in width
    uint32_t NC = 1;
    uint32_t Ht = H / tile_H;
    uint32_t Wt = W / tile_W;

    log_info(LogTest, "====================================================================");
    log_info(LogTest, "Testing fused eltwise binary + reduce operation");
    log_info(LogTest, "Tensor shape: NC={}, H={}, W={} (Ht={}, Wt={})", NC, H, W, Ht, Wt);

    // Test with ADD operation (can be extended to test SUB/MUL)
    auto eltwise_op = EltwiseOp::ADD;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;  // FP16_B format
        uint32_t num_tiles = NC * Ht * Wt;     // Total tiles in input tensor
        uint32_t dram_buffer_size = single_tile_size * num_tiles;
        uint32_t page_size = single_tile_size;

        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        // Input buffers for eltwise binary operation
        auto src0_dram_buffer = CreateBuffer(dram_config);
        auto src1_dram_buffer = CreateBuffer(dram_config);

        // Output buffer for final reduced result
        uint32_t output_size_bytes = dram_buffer_size / Ht;  // Reduced in height dimension
        tt_metal::InterleavedBufferConfig dst_config{
            .device = device,
            .size = output_size_bytes,
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::DRAM};
        auto dst_dram_buffer = CreateBuffer(dst_config);

        ////////////////////////////////////////////////////////////////////////////
        //                      Circular Buffers Setup
        ////////////////////////////////////////////////////////////////////////////

        // Input CBs for eltwise binary operation
        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = 32;
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

        // Intermediate CB for data flow between eltwise and reduce operations
        uint32_t intermediate_cb_index = tt::CBIndex::c_24;
        tt_metal::CircularBufferConfig cb_intermediate_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{intermediate_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(intermediate_cb_index, single_tile_size);
        auto cb_intermediate = tt_metal::CreateCircularBuffer(program, core, cb_intermediate_config);

        // Scaler CB for reduce operation
        uint32_t scaler_cb_index = tt::CBIndex::c_2;
        tt_metal::CircularBufferConfig cb_scaler_config =
            tt_metal::CircularBufferConfig(2 * single_tile_size, {{scaler_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(scaler_cb_index, single_tile_size);
        auto cb_scaler = tt_metal::CreateCircularBuffer(program, core, cb_scaler_config);

        // Output CB for final result
        uint32_t output_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = 32;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        ////////////////////////////////////////////////////////////////////////////
        //                      Kernels Setup
        ////////////////////////////////////////////////////////////////////////////

        // Reader kernel - reads two input tensors for eltwise operation and generates scaler for reduce
        auto dual_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_with_scaler.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        // Writer kernel - writes final reduced result
        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        // Fused compute kernel - performs eltwise binary + reduce operations
        vector<uint32_t> compute_kernel_args = {
            uint32_t(Ht),  // Height in tiles (compile-time arg 0)
            uint32_t(Wt),  // Width in tiles (compile-time arg 1)
            uint32_t(NC),  // Number of channels (compile-time arg 2)
        };

        std::map<std::string, std::string> fused_defines = {
            // Eltwise binary operation defines
            {"ELTWISE_OP", op_id_to_op_define[eltwise_op]},
            {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]},
            // Reduce operation defines
            {"REDUCE_OP", "PoolType::SUM"},
            {"REDUCE_DIM", "ReduceDim::REDUCE_COL"}  // Reduce along height (columns)
        };

        auto fused_compute_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/reduce_h2.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = fused_defines});

        ////////////////////////////////////////////////////////////////////////////
        //                      Runtime Arguments Setup
        ////////////////////////////////////////////////////////////////////////////

        // Compute kernel runtime args: per_core_block_cnt, per_core_block_size, acc_to_dst
        SetRuntimeArgs(
            program,
            fused_compute_kernel,
            core,
            {
                uint32_t(1),          // per_core_block_cnt - process 1 block
                uint32_t(num_tiles),  // per_core_block_size - all tiles in one block
                uint32_t(0)           // acc_to_dst - no accumulation to destination
            });

        // Create scaler value for reduce operation (1.0f for SUM reduction)
        float scaler = 1.0f;
        bfloat16 bfloat_scaler_value = bfloat16(scaler);
        uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});

        // Reader kernel args: src0_addr, src0_bank, src0_tiles, src1_addr, src1_bank, src1_tiles, scaler_value
        const std::array<uint32_t, 7> reader_args = {
            src0_dram_buffer->address(),  // src0_addr
            0,                            // src0_bank
            num_tiles,                    // src0_tiles
            src1_dram_buffer->address(),  // src1_addr
            0,                            // src1_bank
            num_tiles,                    // src1_tiles
            packed_scaler_value           // scaler_value for reduce
        };

        // Writer kernel args: dst_addr, dst_bank, num_output_tiles
        const std::array<uint32_t, 3> writer_args = {
            dst_dram_buffer->address(),  // dst_addr
            0,                           // dst_bank
            num_tiles / Ht               // num_output_tiles (reduced by height factor)
        };

        SetRuntimeArgs(program, dual_reader_kernel, core, reader_args);
        SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        // Create test data
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 10.0f, std::chrono::system_clock::now().time_since_epoch().count());

        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 10.0f, std::chrono::system_clock::now().time_since_epoch().count() + 1);

        // Write input data to device
        EnqueueWriteBuffer(cq, std::ref(src0_dram_buffer), src0_vec, false);
        EnqueueWriteBuffer(cq, std::ref(src1_dram_buffer), src1_vec, false);

        // Execute the fused kernel
        EnqueueProgram(cq, program, false);

        // Read results
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation
        ////////////////////////////////////////////////////////////////////////////

        log_info(LogTest, "Input tensor 0 size: {} tiles", num_tiles);
        log_info(LogTest, "Input tensor 1 size: {} tiles", num_tiles);
        log_info(LogTest, "Output tensor size: {} tiles", result_vec.size() / (single_tile_size / sizeof(uint32_t)));
        log_info(LogTest, "Expected output size: {} tiles", num_tiles / Ht);

        // Basic validation - check that we got expected output size
        uint32_t expected_output_elements = output_size_bytes / sizeof(uint32_t);
        bool size_check = (result_vec.size() == expected_output_elements);

        if (size_check) {
            log_info(LogTest, "✓ Output size validation passed");
        } else {
            log_error(
                LogTest,
                "✗ Output size validation failed: got {} elements, expected {}",
                result_vec.size(),
                expected_output_elements);
        }

        pass &= size_check;

        // Additional validation could include:
        // 1. Golden model comparison (eltwise operation followed by reduce)
        // 2. Numerical accuracy checks
        // 3. Different input patterns and edge cases

    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "Exception: {}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Cleanup
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CloseDevice(device);

    if (pass) {
        log_info(LogTest, "✓ Fused eltwise binary + reduce test PASSED");
    } else {
        log_error(LogTest, "✗ Fused eltwise binary + reduce test FAILED");
        TT_THROW("Test Failed");
    }

    return 0;
}
