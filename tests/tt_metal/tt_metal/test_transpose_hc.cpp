// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "test_gold_impls.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include "impl/data_format/bfloat16_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <vector>

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

using namespace tt;

//////////////////////////////////////////////////////////////////////////////////////////
// Reference CPU implementation of transpose_HC
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: tests transpose kernel for HC dimensions
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    bool pass = true;
    constexpr bool multibank = true;

    const auto* const slow_dispatch_mode = ::getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        constexpr uint32_t subtile_elements = 16U;
        constexpr uint32_t subtile_line_bytes = subtile_elements * 2U;  // FP16 is 2 bytes
        constexpr uint32_t tile_elements = 32U;
        constexpr uint32_t tile_size = tile_elements * tile_elements;

        // shape of this vector is [N, C, H, W]
        // It is laid out in memory as addr = W + ShapeW * H + ShapeW * ShapeH * C + ShapeW * SHapeH * ShapeC * N
        const std::vector<uint32_t> shape = {2U, tile_elements * 3U, tile_elements * 5U, tile_elements * 2U};
        const uint32_t num_elements =
            std::accumulate(std::cbegin(shape), std::cend(shape), 1U, [](const uint32_t left, const uint32_t right) {
                return left * right;
            });
        const uint32_t num_tensor_tiles = num_elements / tile_size;

        const uint32_t single_tile_bytes = 2U * tile_size;  // FP16_B is 2 bytes
        const uint32_t dram_buffer_bytes =
            single_tile_bytes * num_tensor_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        const uint32_t page_size = (!multibank) ? dram_buffer_bytes : single_tile_bytes;

        const tt_metal::InterleavedBufferConfig dram_config = {
            .device = device,
            .size = dram_buffer_bytes,
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        auto src0_dram_buffer = CreateBuffer(dram_config);
        const uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
        auto dst_dram_buffer = CreateBuffer(dram_config);
        const uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();
        const uint32_t alignment = dst_dram_buffer->alignment();
        const bool misaligned = alignment > subtile_line_bytes;

        const uint32_t src0_cb_index = 0U;
        const uint32_t num_buffer_tiles = 2U;
        // this buffer is used in transpose_hc.cpp NCRISC kernel
        const tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_bytes);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        const uint32_t output_cb_index = tt::CBIndex::c_16;
        // this buffer is used in writer_unary.cpp BRISC kernel
        const tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_buffer_tiles * single_tile_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_bytes);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        // need some scratch memory here - if we need data from a misaligned address then we need to read from the
        // nearest aligned address and then copy the data to the correct location
        if (misaligned) {
            const uint32_t src1_cb_index = 1U;
            tt::tt_metal::CircularBufferConfig cb_src1_config =
                tt::tt_metal::CircularBufferConfig(alignment, {{src1_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src1_cb_index, alignment);
            tt::tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
        }

        const uint32_t W = shape[3U], H = shape[2U], C = shape[1U], N = shape[0U];
        const uint32_t HW = H * W;
        const uint32_t CHW = C * H * W;
        std::vector<uint32_t> reader_compile_time_args;
        reader_compile_time_args.emplace_back(alignment);
        tt::tt_metal::TensorAccessorArgs(src0_dram_buffer).append_to(reader_compile_time_args);
        auto reader_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/transpose_hc_8bank.cpp"
                      : "tests/tt_metal/tt_metal/test_kernels/dataflow/transpose_hc.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(dst_dram_buffer).append_to(writer_compile_time_args);
        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                      : "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_kernel_args = {num_tensor_tiles};

        tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 100U, 0x1234);
        auto src_4f_16 = u16_from_u32_vector(src0_vec);
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

        tt_metal::SetRuntimeArgs(program, reader_kernel, core, {dram_buffer_src0_addr, 0U, W, H, C, HW, N, CHW});

        tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, 0U, num_tensor_tiles});

        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        int argfail = -1;
        auto comparison_function = [](float a, float b) {
            const float rtol{0.001f};
            const float atol{1e-3f};
            const float maxabs{std::fmaxf(std::abs(a), std::abs(b))};
            float absdiff{std::abs(a - b)};
            const auto result{(absdiff <= atol) || (absdiff < rtol * maxabs)};
            if (!result) {
                absdiff *= 1.0f;  // breakpoint spot
            }
            return result;
        };

        // recover a linear view of input vector for consumption by gold_ function
        std::vector<uint16_t> src_linear =
            convert_layout<uint16_t>(src_4f_16, shape, TensorLayoutType::TILED_NFACES, TensorLayoutType::LIN_ROW_MAJOR);
        std::vector<uint16_t> gold_reduced = gold_transpose_hc(src_linear, shape);  // result is uint16_t untilized

        // Tilize from row major and convert to pairs (uint32_t)
        std::vector<uint32_t> shapeR = {shape[0U], shape[2U], shape[1U], shape[3U]};
        const auto gold_16_4f = convert_layout<uint16_t>(
            gold_reduced, shapeR, TensorLayoutType::LIN_ROW_MAJOR, TensorLayoutType::TILED_NFACES);
        const auto gold_4f_u32 = u32_from_u16_vector(gold_16_4f);
        const auto u16_result = u16_from_u32_vector(result_vec);

        pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
        if (!pass) {
            log_error(LogTest, "Failure position={}", argfail);
        }

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return EXIT_SUCCESS;
}
