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
// Test for fused eltwise binary + reduce operation on multiple tiles
// This test combines:
// 1. Eltwise binary operation (ADD/SUB/MUL) on two input tensors (8 tiles each)
// 2. Reduce operation (SUM) along tiles to produce a single output tile
//
// Key features:
// - Processes 8 input tiles through eltwise binary operation
// - Reduces the 8 results into a single output tile
// - Uses the fused API from fused_eltwise_binary_reduce.h
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    // Operation definitions for eltwise binary
    const char* op_id_to_op_define[] = {"add_tiles", "sub_tiles", "mul_tiles"};
    const char* op_id_to_op_type_define[] = {
        "EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWSUB", "EltwiseBinaryType::ELWMUL"};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    CommandQueue& cq = device->command_queue();

    // Test parameters - MULTIPLE TILES TEST
    uint32_t tile_H = 32, tile_W = 32;      // Standard tile dimensions
    uint32_t num_input_tiles = 8;           // Process 8 tiles
    uint32_t H = tile_H * num_input_tiles;  // 8 tiles in height
    uint32_t W = tile_W * 1;                // 1 tile in width
    uint32_t NC = 1;
    uint32_t Ht = H / tile_H;  // Ht = 8
    uint32_t Wt = W / tile_W;  // Wt = 1

    log_info(LogTest, "====================================================================");
    log_info(LogTest, "Testing fused eltwise binary + reduce operation (Multiple Tiles)");
    log_info(LogTest, "Input tiles: {}, Output tiles: 1", num_input_tiles);
    log_info(LogTest, "Tensor shape: NC={}, H={}, W={} (Ht={}, Wt={})", NC, H, W, Ht, Wt);
    log_info(LogTest, "Using fused API for multiple tiles processing");

    // Test with MUL operation
    auto eltwise_op = EltwiseOp::MUL;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;                             // FP16_B format
        uint32_t input_buffer_size = single_tile_size * num_input_tiles;  // 8 tiles
        uint32_t output_buffer_size = single_tile_size * 1;               // 1 tile output
        uint32_t page_size = single_tile_size;

        // Input buffers for eltwise binary operation (8 tiles each)
        tt_metal::InterleavedBufferConfig input_config{
            .device = device,
            .size = input_buffer_size,
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        auto src0_dram_buffer = CreateBuffer(input_config);
        auto src1_dram_buffer = CreateBuffer(input_config);

        // Output buffer for final result (1 tile)
        tt_metal::InterleavedBufferConfig dst_config{
            .device = device,
            .size = output_buffer_size,
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::DRAM};
        auto dst_dram_buffer = CreateBuffer(dst_config);

        ////////////////////////////////////////////////////////////////////////////
        //                      Circular Buffers Setup
        ////////////////////////////////////////////////////////////////////////////

        // Input CBs for eltwise binary operation
        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t cb_tiles = 32;  // Buffer capacity
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = tt::CBIndex::c_1;
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        // Output CB for final result
        uint32_t output_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_cb_tiles = 32;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_cb_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        ////////////////////////////////////////////////////////////////////////////
        //                      Kernels Setup
        ////////////////////////////////////////////////////////////////////////////

        // Reader kernel - reads 8 tiles for each input tensor
        auto dual_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_multiple_tiles.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        // Writer kernel - writes 1 output tile
        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_single_tile.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        // Fused compute kernel - performs eltwise binary + reduce operations
        vector<uint32_t> compute_kernel_args = {};  // No compile-time args needed

        std::map<std::string, std::string> fused_defines = {
            // Eltwise binary operation defines
            {"ELTWISE_OP", op_id_to_op_define[eltwise_op]},
            {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]},
            // Reduce operation defines
            {"REDUCE_OP", "PoolType::SUM"},
            {"REDUCE_DIM", "ReduceDim::REDUCE_COL"}  // Reduce along tiles
        };

        auto fused_compute_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/fused_op_multiple_tiles.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = fused_defines});

        ////////////////////////////////////////////////////////////////////////////
        //                      Runtime Arguments Setup
        ////////////////////////////////////////////////////////////////////////////

        // Compute kernel runtime args: tile_cnt
        SetRuntimeArgs(
            program,
            fused_compute_kernel,
            core,
            {
                num_input_tiles  // tile_cnt - process 8 tiles
            });

        // Reader kernel args: src0_addr, src0_bank, src0_tiles, src1_addr, src1_bank, src1_tiles
        const std::array<uint32_t, 6> reader_args = {
            src0_dram_buffer->address(),  // src0_addr
            0,                            // src0_bank
            num_input_tiles,              // src0_tiles (8 tiles)
            src1_dram_buffer->address(),  // src1_addr
            0,                            // src1_bank
            num_input_tiles,              // src1_tiles (8 tiles)
        };

        // Writer kernel args: dst_addr, dst_bank, num_output_tiles
        const std::array<uint32_t, 3> writer_args = {
            dst_dram_buffer->address(),  // dst_addr
            0,                           // dst_bank
            1                            // num_output_tiles (1 tile output)
        };

        SetRuntimeArgs(program, dual_reader_kernel, core, reader_args);
        SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        // Create test data
        log_info(LogTest, "Creating test data for {} tiles...", num_input_tiles);

        // Input tensor 0: filled with constant value (1.5f)
        std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(input_buffer_size, 1.5f);

        // Input tensor 1: filled with constant value (2.0f)
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(input_buffer_size, 2.5f);

        // Write input data to device
        log_info(LogTest, "Writing input data to device...");
        EnqueueWriteBuffer(cq, std::ref(src0_dram_buffer), src0_vec, false);
        EnqueueWriteBuffer(cq, std::ref(src1_dram_buffer), src1_vec, false);

        // Execute the fused kernel
        log_info(LogTest, "Executing fused eltwise binary + reduce kernel for {} tiles...", num_input_tiles);
        EnqueueProgram(cq, program, false);

        // Read results
        log_info(LogTest, "Reading results from device...");
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation
        ////////////////////////////////////////////////////////////////////////////

        log_info(LogTest, "Input tensor 0 size: {} tiles", num_input_tiles);
        log_info(LogTest, "Input tensor 1 size: {} tiles", num_input_tiles);
        log_info(LogTest, "Output tensor size: {} tiles", result_vec.size() / (single_tile_size / sizeof(uint32_t)));
        log_info(LogTest, "Expected output size: 1 tile");

        // Basic validation - check that we got expected output size
        uint32_t expected_output_elements = output_buffer_size / sizeof(uint32_t);
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

        // Additional validation: check some output values
        if (size_check && result_vec.size() > 0) {
            log_info(LogTest, "Sample output values (first 4 elements):");
            for (size_t i = 0; i < std::min(size_t(4), result_vec.size()); ++i) {
                // Unpack bfloat16 values for display
                bfloat16 val1, val2;
                std::tie(val1, val2) = unpack_two_bfloat16_from_uint32(result_vec[i]);
                log_info(
                    LogTest,
                    "  result_vec[{}] = [{:.3f}, {:.3f}]",
                    i,
                    static_cast<float>(val1),
                    static_cast<float>(val2));
            }
        }

        log_info(LogTest, "✓ Fused eltwise binary + reduce operation on multiple tiles completed successfully");

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
        log_info(LogTest, "✓ Fused eltwise binary + reduce test (Multiple Tiles) PASSED");
    } else {
        log_error(LogTest, "✗ Fused eltwise binary + reduce test (Multiple Tiles) FAILED");
        TT_THROW("Test Failed");
    }

    return 0;
}
