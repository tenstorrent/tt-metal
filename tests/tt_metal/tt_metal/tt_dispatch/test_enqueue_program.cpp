// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/program/program_pool.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;
uint32_t NUM_TILES = 2048;

tt_metal::ScopedProgramHandle generate_eltwise_unary_program(tt_metal::Device *device) {
    // TODO(agrebenisan): This is directly copy and pasted from test_eltwise_binary.
    // We need to think of a better way to generate test data, so this section needs to be heavily refactored.

    auto program = tt_metal::CreateScopedProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = NUM_TILES;
    uint32_t dram_buffer_size =
        single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t page_size = single_tile_size;

    tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = dram_buffer_size,
                    .page_size = page_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    auto src0_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();


    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig src_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, src_cb_config);

    uint32_t ouput_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    vector<uint32_t> compute_kernel_args = {
        NUM_TILES,  // per_core_block_cnt
        1,          // per_core_block_size
    };

    auto eltwise_binary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    return std::move(program);
}

void test_enqueue_program(std::function<tt_metal::ScopedProgramHandle(tt_metal::Device *device)> create_program) {
    int device_id = 0;
    tt_metal::Device *device = tt_metal::CreateDevice(device_id);


    auto program = create_program(device);

    CoreCoord worker_core(0, 0);
    vector<uint32_t> inp = create_random_vector_of_bfloat16(NUM_TILES * 2048, 100, 0);

    vector<uint32_t> out_vec;
    {
        CommandQueue& cq = device->command_queue();

        // Enqueue program inputs
        Buffer buf(device, NUM_TILES * 2048, 2048, BufferType::DRAM);
        Buffer out(device, NUM_TILES * 2048, 2048, BufferType::DRAM);

        // Absolutely disgusting way to query for the kernel I want to set runtime args for... needs to be cleaned up
        auto* program_ptr = ProgramPool::instance().get_program(program);
        const KernelGroup *kernel_group = program_ptr->kernels_on_core(worker_core, CoreType::WORKER);
        SetRuntimeArgs(program, kernel_group->riscv0_id.value(), worker_core, {out.address(), 0, 0, NUM_TILES});
        SetRuntimeArgs(program, kernel_group->riscv1_id.value(), worker_core, {buf.address(), 0, 0, NUM_TILES});

        EnqueueWriteBuffer(cq, std::ref(buf), inp, false);
        EnqueueProgram(cq, program, false);

        EnqueueReadBuffer(cq, std::ref(out), out_vec, true);
    }

    TT_FATAL(out_vec == inp, "Error");
    tt_metal::CloseDevice(device);
}

int main() {
    // test_program_to_device_map();
    test_enqueue_program(generate_eltwise_unary_program);
    // test_enqueue_program(generate_simple_brisc_program);

    // test_relay_program_to_dram(generate_simple_brisc_program);
    // test_compare_program_binaries_with_bins_on_disk();
}
