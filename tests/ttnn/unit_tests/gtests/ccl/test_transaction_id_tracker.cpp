// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "kernel_types.hpp"
#include "tt_metal.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include <tt-metalium/circular_buffer_types.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/assert.hpp>

int main(int argc, char* argv[]) {
    using namespace tt::tt_metal;
    using tt::tt_metal::distributed::MeshCoordinate;
    using tt::tt_metal::distributed::MeshDevice;
    using tt::tt_metal::distributed::MeshDeviceConfig;
    using tt::tt_metal::distributed::MeshShape;

    // Parse arguments
    TT_FATAL(argc == 8, "Expected 6 arguments, got {}", argc);
    uint32_t page_size_bytes = std::stoi(argv[1]);
    uint32_t num_pages = std::stoi(argv[2]);
    bool is_dram = std::stoi(argv[3]) != 0;
    bool do_write_barrier_instead_of_writes_sent = std::stoi(argv[4]) != 0;
    uint32_t num_reader_transaction_ids = std::stoi(argv[5]);
    uint32_t num_writer_transaction_ids = std::stoi(argv[6]);
    bool do_correctness_check = std::stoi(argv[7]) != 0;

    log_info(tt::LogTest, "Running with:");
    log_info(tt::LogTest, "  page_size_bytes: {}", page_size_bytes);
    log_info(tt::LogTest, "  num_pages: {}", num_pages);
    log_info(tt::LogTest, "  is_dram: {}", (is_dram ? "true" : "false"));
    log_info(
        tt::LogTest,
        "  do_write_barrier_instead_of_writes_sent: {}",
        (do_write_barrier_instead_of_writes_sent ? "true" : "false"));
    log_info(tt::LogTest, "  num_reader_transaction_ids: {}", num_reader_transaction_ids);
    log_info(tt::LogTest, "  num_writer_transaction_ids: {}", num_writer_transaction_ids);

    // Create device
    auto mesh_device = MeshDevice::create(MeshDeviceConfig{.mesh_shape = MeshShape(1, 1)});
    auto mesh_device_view = mesh_device->get_view();
    auto device = mesh_device_view.get_device(MeshCoordinate{0, 0});

    // Calculate tensor size
    uint32_t tensor_size_bytes = page_size_bytes * num_pages;

    // Create a program
    auto program = CreateProgram();

    // Create buffers
    BufferType buffer_type = is_dram ? BufferType::DRAM : BufferType::L1;

    auto input_buffer_config = InterleavedBufferConfig{
        .device = device,
        .size = tensor_size_bytes,
        .page_size = page_size_bytes,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED};

    auto output_buffer_config = input_buffer_config;

    auto input_buffer = CreateBuffer(input_buffer_config);
    auto output_buffer = CreateBuffer(output_buffer_config);

    // Create a list of pages_per_push/pages_per_pop values
    // We'll create random values between 1 and 10, up to 10 different values
    // std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    // std::uniform_int_distribution<int> dist(1, 10);
    uint32_t max_pages_per_group = 5;
    uint32_t num_page_groups = std::min(max_pages_per_group, num_pages / 2);

    std::vector<uint32_t> pages_per_push_array(num_page_groups);
    for (uint32_t i = 0; i < num_page_groups; ++i) {
        pages_per_push_array[i] = i + 1;
    }

    // Adjust to ensure total matches num_pages
    const uint32_t sum_pages = std::accumulate(pages_per_push_array.begin(), pages_per_push_array.end(), 0);

    // Create circular buffer for communication between kernels
    // This must be large enough to hold at least all the pages
    uint32_t cb_size = sum_pages * 2 * page_size_bytes;

    constexpr tt::DataFormat cb_df = tt::DataFormat::Bfp8;
    CoreCoord core = CoreCoord(0, 0);
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{src0_cb_index, cb_df}})
            .set_page_size(src0_cb_index, page_size_bytes);

    auto cb_handle = CreateCircularBuffer(program, core, cb_src0_config);

    // Create kernels
    std::vector<uint32_t> reader_compile_args = {num_reader_transaction_ids, is_dram};
    std::vector<uint32_t> writer_compile_args = {
        num_writer_transaction_ids, is_dram, do_write_barrier_instead_of_writes_sent};

    auto reader_kernel = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/transaction_id_tracker_reader.cpp",
        core,
        ReaderDataMovementConfig(reader_compile_args));
    auto writer_kernel = CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/transaction_id_tracker_writer.cpp",
        core,
        WriterDataMovementConfig(writer_compile_args));

    std::vector<uint32_t> reader_runtime_args = {
        src0_cb_index,  // CB ID
        input_buffer->address(),
        page_size_bytes,
        num_pages,
        !do_correctness_check,
        static_cast<uint32_t>(pages_per_push_array.size())};
    std::copy(pages_per_push_array.begin(), pages_per_push_array.end(), std::back_inserter(reader_runtime_args));

    std::vector<uint32_t> writer_runtime_args = {
        src0_cb_index,  // CB ID
        output_buffer->address(),
        page_size_bytes,
        num_pages,
        static_cast<uint32_t>(pages_per_push_array.size())};
    std::copy(pages_per_push_array.begin(), pages_per_push_array.end(), std::back_inserter(writer_runtime_args));

    SetRuntimeArgs(program, reader_kernel, core, reader_runtime_args);
    SetRuntimeArgs(program, writer_kernel, core, writer_runtime_args);

    // Prepare data for input buffer - sequential values
    std::vector<uint32_t> input_data(tensor_size_bytes / sizeof(uint32_t));
    std::iota(input_data.begin(), input_data.end(), 0);

    // Golden tensor - copy of input data for validation
    std::vector<uint32_t> golden_data = input_data;

    // Output data - initialize with zeros
    std::vector<uint32_t> output_data(tensor_size_bytes / sizeof(uint32_t), 0);

    constexpr bool blocking_true = true;
    // Copy input data to device
    EnqueueWriteBuffer(device->command_queue(), input_buffer, input_data.data(), blocking_true);
    EnqueueWriteBuffer(device->command_queue(), output_buffer, output_data.data(), blocking_true);
    Finish(device->command_queue());
    // Launch program
    try {
        detail::CompileProgram(device, program);
    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Exception during compilation: {}", e.what());
        mesh_device->close();
        return -1;
    }

    log_info(tt::LogTest, "Launching program");
    EnqueueProgram(device->command_queue(), program, blocking_true);
    Finish(device->command_queue());
    // Copy output back to host
    EnqueueReadBuffer(device->command_queue(), output_buffer, output_data, blocking_true);

    DumpDeviceProfileResults(device, program);

    // Validate results
    bool match = true;
    if (do_correctness_check) {
        for (size_t i = 0; i < golden_data.size(); ++i) {
            if (output_data[i] != golden_data[i]) {
                log_error(
                    tt::LogTest, "Mismatch at index {}: expected {} but got {}", i, golden_data[i], output_data[i]);
                match = false;
                break;
            }
        }
    }
    if (!match) {
        log_error(tt::LogTest, "Output does not match golden data!");
        mesh_device->close();
        return -1;
    }

    // Clean up
    mesh_device->close();

    return 0;
}
