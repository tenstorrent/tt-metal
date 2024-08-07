// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "tt_metal/detail/tt_metal.hpp"
using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    // Initialize Program and Device
    constexpr CoreCoord core = {5, 0};
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue &cq = device->command_queue();
    Program program_no_barriers = CreateProgram();
    Program program_both_barriers = CreateProgram();
    constexpr uint32_t single_tile_size = 1024;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> collect_dram_buffer = CreateBuffer(dram_config);

    constexpr uint32_t cb_write_index = CB::c_in0;
    CircularBufferConfig cb_write_config = CircularBufferConfig(single_tile_size, {{cb_write_index, tt::DataFormat::Float16_b}}).set_page_size(cb_write_index, single_tile_size);
    CBHandle cb_write = tt_metal::CreateCircularBuffer(program_no_barriers, core, cb_write_config);
    tt_metal::CreateCircularBuffer(program_both_barriers, core, cb_write_config);

    constexpr uint32_t cb_read_index = CB::c_in1;
    CircularBufferConfig cb_read_config = CircularBufferConfig(single_tile_size, {{cb_read_index, tt::DataFormat::Float16_b}}).set_page_size(cb_read_index, single_tile_size);
    CBHandle cb_read = tt_metal::CreateCircularBuffer(program_no_barriers, core, cb_read_config);
    tt_metal::CreateCircularBuffer(program_both_barriers, core, cb_read_config);

    std::vector<uint32_t> initial_int_value(1, 0);
    EnqueueWriteBuffer(cq, collect_dram_buffer, initial_int_value, false);

    KernelHandle kernel_no_barriers = CreateKernel(
        program_no_barriers,
        "tt_metal/programming_examples/one_to_one_datastream/kernels/dataflow/int_num_gen_and_reader_kernel_no_barriers.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle kernel_both_barriers = CreateKernel(
        program_both_barriers,
        "tt_metal/programming_examples/one_to_one_datastream/kernels/dataflow/int_num_gen_and_reader_kernel_both_barriers.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(program_no_barriers, kernel_no_barriers, core, {collect_dram_buffer->address()});
    SetRuntimeArgs(program_both_barriers, kernel_both_barriers, core, {collect_dram_buffer->address()});

    auto measure_time = [&](Program &program, int runs) -> uint64_t {
        uint64_t total_time = 0;
        for (int i = 0; i < runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            EnqueueProgram(cq, program, false);
            Finish(cq);
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        }
        return total_time;
    };

    std::ofstream csv_file("timing_results.csv");
    csv_file << "Runs,No Barriers (µs),Both Barriers (µs)\n";

    for (int runs = 100; runs <= 2500; runs += 100) {
        double time_no_barriers = static_cast<double>(measure_time(program_no_barriers, runs)) / runs;
        double time_both_barriers = static_cast<double>(measure_time(program_both_barriers, runs)) / runs;
        csv_file << runs << "," << time_no_barriers << "," << time_both_barriers << "\n";
    }

    csv_file.close();
    CloseDevice(device);

    std::cout << "Timing results have been written to timing_results.csv\n";
    return 0;
}
