// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <magic_enum/magic_enum.hpp>

using namespace tt;
using namespace tt::tt_metal;

/*
 * 1. Host creates two vectors of data.
 * 2. Device eltwise adds them together.
 * 3. Intermediate result read back to host.
 * 6. Read result back and compare to golden.
 * */

int main() {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        /*
         * Silicon accelerator setup
         */
        constexpr int device_id = 0;
        IDevice* device = CreateDevice(device_id);

        /*
         * Setup program to execute along with its buffers and kernels to use
         */
        CommandQueue& cq = device->command_queue();

        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t single_tile_size = 2 * 1024;
        constexpr uint32_t num_tiles = 64;
        constexpr uint32_t block_size = 8;
        constexpr uint32_t dram_buffer_size =
            single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
        std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
        std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
        // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
        uint32_t src0_bank_id = 0;
        uint32_t src1_bank_id = 0;
        uint32_t dst_bank_id = 0;
        /*
         * Use circular buffers to set input and output buffers that the
         * compute engine will use.
         */
        constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
        constexpr uint32_t num_input_tiles = block_size * 2;
        CircularBufferConfig cb_src0_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
        CircularBufferConfig cb_src1_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
        constexpr uint32_t num_output_tiles = block_size * 2;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(output_cb_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        /*
         * Specify data movement kernels for reading/writing data to/from
         * DRAM.
         */
        KernelHandle binary_reader_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_dummy.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        KernelHandle unary_writer_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_dummy.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        /*
         * Set the parameters that the compute kernel will use.
         */
        std::vector<uint32_t> compute_kernel_args = {};

        constexpr bool fp32_dest_acc_en = false;
        constexpr bool math_approx_mode = false;

        /*
         * Use the add_tiles operation available in the eltwise_binary
         * compute kernel.
         */
        KernelHandle eltwise_binary_kernel_id = CreateKernel(
            program,
            "tt_metal/kernels/compute/single_thread_eltwise_binary_with_read_write.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args});

        /*
         * Create source data and write to DRAM.
         */
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 1, std::chrono::system_clock::now().time_since_epoch().count());

        EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);

        constexpr float val_to_add = -2.0f;
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_add);

        EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

        /*
         * Configure program and runtime kernel arguments, then execute.
         */

        SetRuntimeArgs(
            program,
            binary_reader_kernel_id,
            core,
            {src0_dram_buffer->address(), src0_bank_id, src1_dram_buffer->address(), src1_bank_id, num_tiles});

        SetRuntimeArgs(
            program,
            eltwise_binary_kernel_id,
            core,
            {src0_dram_buffer->address(),
             src0_bank_id,
             src1_dram_buffer->address(),
             src1_bank_id,
             num_tiles / block_size,
             block_size,
             dst_dram_buffer->address(),
             dst_bank_id});

        SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_bank_id, num_tiles});

        EnqueueProgram(cq, program, false);
        Finish(cq);

        /*
         * Read in result into a host vector.
         */
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

        auto transform_to_golden = [](const bfloat16& a) { return bfloat16(a.to_float() + val_to_add); };
        std::vector<uint32_t> golden_vec =
            pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(src0_vec, transform_to_golden));

        constexpr float abs_tolerance = 0.01f;
        constexpr float rel_tolerance = 0.001f;
        auto comparison_function = [](const float a, const float b) {
            return is_close(a, b, rel_tolerance, abs_tolerance);
        };

        pass &= packed_uint32_t_vector_comparison(golden_vec, result_vec, comparison_function);

        pass &= CloseDevice(device);

    } catch (const std::exception& e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
