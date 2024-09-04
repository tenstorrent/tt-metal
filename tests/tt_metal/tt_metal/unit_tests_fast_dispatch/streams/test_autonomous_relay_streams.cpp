// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <tuple>

#include "device/tt_arch_types.h"
#include "gtest/gtest.h"
#include "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/common/command_queue_fixture.hpp"
#include "tt_metal/common/logger.hpp"
#include "impl/device/device.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
// #include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/detail/persistent_kernel_cache.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using tt::tt_metal::Device;

constexpr uint32_t num_sizes = 8;
namespace tt {

namespace tt_metal {

struct hop_eth_sockets {
    chip_id_t receiver_device_id;
    CoreCoord receiver_core;
    chip_id_t sender_device_id;
    CoreCoord sender_core;
};

struct stream_config_t {
    uint32_t buffer_addr;
    uint32_t buffer_size;  // in bytes
    uint32_t tile_header_buffer_addr;
    uint32_t tile_header_num_msgs;
    uint32_t tile_header_buffer_size;  // in bytes
};

struct stream_builder_spec_t {
    uint32_t buffer_size_bytes;
    uint32_t tile_header_buffer_size_bytes;
};

constexpr uint32_t relay_stream_id = 32;
constexpr uint32_t tile_header_size = 32;  // needs to provide noc word alignment
// constexpr uint32_t tile_header_size = 16;
constexpr uint32_t noc_word_size = 16;

// Reads data from input
std::vector<uint32_t> get_sender_reader_rt_args(
    Device* device,
    uint32_t input_buffer_addr,
    uint32_t page_size_plus_header,
    uint32_t num_messages_to_read,
    std::array<uint32_t, num_sizes> const& sub_sizes) {
    auto args = std::vector<uint32_t>{input_buffer_addr, page_size_plus_header, num_messages_to_read};
    for (auto const& sub_size : sub_sizes) {
        args.push_back(sub_size);
    }
    return args;
}
// sender stream data mover kernel
std::vector<uint32_t> get_sender_writer_rt_args(
    Device* device,
    uint32_t num_messages,
    uint32_t relay_done_semaphore,
    CoreCoord const& relay_core,
    uint32_t sender_noc_id,
    stream_config_t const& sender_stream_config,
    stream_config_t const& relay_stream_config,
    CoreCoord const& other_relay_to_notify_when_done,
    uint32_t other_relay_done_semaphore,
    uint32_t sender_wait_for_receiver_semaphore,
    uint32_t first_relay_remote_src_start_phase_id,
    uint32_t hang_toggle_id) {
    return std::vector<uint32_t>{
        num_messages,

        relay_stream_id,
        sender_stream_config.buffer_addr,
        sender_stream_config.buffer_size,
        sender_stream_config.tile_header_buffer_addr,
        relay_stream_config.tile_header_num_msgs,

        static_cast<uint32_t>(device->worker_core_from_logical_core(relay_core).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(relay_core).y),
        relay_stream_id,
        sender_noc_id,

        relay_stream_config.buffer_addr,
        relay_stream_config.buffer_size,
        relay_stream_config.tile_header_buffer_addr,

        relay_done_semaphore,
        static_cast<uint32_t>(device->worker_core_from_logical_core(other_relay_to_notify_when_done).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(other_relay_to_notify_when_done).y),
        other_relay_done_semaphore,

        static_cast<uint32_t>(sender_wait_for_receiver_semaphore),
        first_relay_remote_src_start_phase_id,
        hang_toggle_id};
}

std::vector<uint32_t> get_relay_rt_args(
    Device* device,
    uint32_t relay_stream_overlay_blob_addr,
    uint32_t relay_done_semaphore,
    CoreCoord const& sender_core,
    CoreCoord const& receiver_core,
    uint32_t sender_noc_id,
    uint32_t receiver_noc_id,
    // stream_config_t const& sender_stream_config,
    stream_config_t const& relay_stream_config,
    stream_config_t const& receiver_stream_config,
    uint32_t remote_src_start_phase_addr,
    uint32_t dest_remote_src_start_phase_addr,
    bool is_first_relay_in_chain) {
    return std::vector<uint32_t>{
        static_cast<uint32_t>(relay_stream_overlay_blob_addr),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(relay_stream_config.buffer_addr),
        static_cast<uint32_t>(relay_stream_config.buffer_size),
        static_cast<uint32_t>(relay_stream_config.tile_header_buffer_addr),
        static_cast<uint32_t>(relay_stream_config.tile_header_num_msgs),

        // noc0 address
        static_cast<uint32_t>(device->worker_core_from_logical_core(sender_core).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(sender_core).y),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(sender_noc_id),

        static_cast<uint32_t>(device->worker_core_from_logical_core(receiver_core).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(receiver_core).y),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(receiver_noc_id),
        static_cast<uint32_t>(receiver_stream_config.buffer_addr),
        static_cast<uint32_t>(receiver_stream_config.buffer_size),
        static_cast<uint32_t>(receiver_stream_config.tile_header_buffer_addr),

        static_cast<uint32_t>(relay_done_semaphore),
        static_cast<uint32_t>(is_first_relay_in_chain ? 1 : 0),

        remote_src_start_phase_addr,
        dest_remote_src_start_phase_addr};
}

// Receiver stream data mover kernel
std::vector<uint32_t> get_receiver_reader_rt_args(
    Device* device,
    uint32_t num_messages,
    uint32_t relay_done_semaphore,
    CoreCoord const& relay_core,
    uint32_t receiver_noc_id,
    stream_config_t const& relay_stream_config,
    stream_config_t const& receiver_stream_config,
    CoreCoord const& other_relay_core_to_notify_when_done,
    uint32_t other_relay_done_semaphore,
    CoreCoord const& sender_core,
    uint32_t sender_receiver_semaphore,
    uint32_t remote_src_start_phase_id) {
    return std::vector<uint32_t>{
        static_cast<uint32_t>(num_messages),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(receiver_stream_config.buffer_addr),
        static_cast<uint32_t>(receiver_stream_config.buffer_size),
        static_cast<uint32_t>(receiver_stream_config.tile_header_buffer_addr),
        static_cast<uint32_t>(receiver_stream_config.tile_header_num_msgs),
        static_cast<uint32_t>(device->worker_core_from_logical_core(relay_core).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(relay_core).y),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(receiver_noc_id),
        static_cast<uint32_t>(relay_stream_config.buffer_addr),
        static_cast<uint32_t>(relay_stream_config.buffer_size),
        static_cast<uint32_t>(relay_stream_config.tile_header_buffer_addr),

        static_cast<uint32_t>(relay_done_semaphore),
        static_cast<uint32_t>(device->worker_core_from_logical_core(other_relay_core_to_notify_when_done).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(other_relay_core_to_notify_when_done).y),
        other_relay_done_semaphore,

        static_cast<uint32_t>(device->worker_core_from_logical_core(sender_core).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(sender_core).y),
        sender_receiver_semaphore,
        remote_src_start_phase_id};
}
std::vector<uint32_t> get_receiver_writer_rt_args(
    Device* device, uint32_t output_buffer_addr, uint32_t page_size, uint32_t num_messages_to_read) {
    return std::vector<uint32_t>{output_buffer_addr, page_size, num_messages_to_read};
}

// TODO: randomize each noc for testing purposes
void build_and_run_autonomous_stream_test(
    std::vector<Program>& programs,
    std::vector<Device*> const& devices,
    std::size_t num_messages,
    std::size_t page_size,
    uint32_t tile_header_buffer_num_messages,
    stream_builder_spec_t const& sender_stream_spec,
    stream_builder_spec_t const& relay_stream_spec,
    stream_builder_spec_t const& receiver_stream_spec,
    bool enable_page_size_variations,
    std::array<uint32_t, num_sizes> const& sub_sizes,
    std::size_t num_loop_iterations) {
    TT_ASSERT(programs.size() == 0);
    // Make configurable
    const uint32_t read_write_cb_num_pages = 8;
    const uint32_t page_size_plus_header = page_size + tile_header_size;

    const uint32_t sender_stream_buffer_num_pages = sender_stream_spec.buffer_size_bytes / page_size;
    const uint32_t relay_stream_buffer_num_pages = relay_stream_spec.buffer_size_bytes / page_size;
    const uint32_t receiver_stream_buffer_num_pages = receiver_stream_spec.buffer_size_bytes / page_size;

    const uint32_t sender_stream_buffer_size_bytes = sender_stream_buffer_num_pages * page_size_plus_header;
    const uint32_t relay_stream_buffer_size_bytes = relay_stream_buffer_num_pages * page_size_plus_header;
    const uint32_t receiver_stream_buffer_size_bytes = receiver_stream_buffer_num_pages * page_size_plus_header;
    uint32_t stream_tile_header_buffer_size_bytes = tile_header_buffer_num_messages * tile_header_size;
    uint32_t relay_stream_overlay_blob_size_bytes = 256;

    programs.emplace_back();
    Device* device = devices.at(0);
    Program& program = programs.at(0);
    log_trace(tt::LogTest, "Device ID: {}", device->id());

    CoreCoord sender_core = CoreCoord(0, 0);
    CoreCoord first_relay_core = CoreCoord(1, 0);
    CoreCoord second_relay_core = CoreCoord(2, 0);
    CoreCoord receiver_core = CoreCoord(3, 0);

    log_trace(
        tt::LogTest,
        "sender_core: x={}, y={}",
        device->physical_core_from_logical_core(sender_core, CoreType::WORKER).x,
        device->physical_core_from_logical_core(sender_core, CoreType::WORKER).y);
    log_trace(
        tt::LogTest,
        "first_relay_core: x={}, y={}",
        device->physical_core_from_logical_core(first_relay_core, CoreType::WORKER).x,
        device->physical_core_from_logical_core(first_relay_core, CoreType::WORKER).y);
    log_trace(
        tt::LogTest,
        "second_relay_core: x={}, y={}",
        device->physical_core_from_logical_core(second_relay_core, CoreType::WORKER).x,
        device->physical_core_from_logical_core(second_relay_core, CoreType::WORKER).y);
    log_trace(
        tt::LogTest,
        "receiver_core: x={}, y={}",
        device->physical_core_from_logical_core(receiver_core, CoreType::WORKER).x,
        device->physical_core_from_logical_core(receiver_core, CoreType::WORKER).y);

    // Input DRAM buffer creation
    uint32_t buffer_size_bytes = num_messages * page_size;
    auto inputs = test_utils::generate_uniform_random_vector<uint32_t>(0, 100, buffer_size_bytes / sizeof(uint32_t));
    std::iota(inputs.begin(), inputs.end(), 1);
    // for (auto i = 0; i < inputs.size(); i += page_size) {
    //     for (auto ii = 0; ii < std::min<std::size_t>(page_size, inputs.size() - i); ii++) {
    //         inputs.at(i + ii) = i + 1;
    //     }
    // }

    auto zeroes_buffer = std::vector<uint32_t>(buffer_size_bytes / sizeof(uint32_t), 0);
    std::vector<uint32_t> outputs(buffer_size_bytes / sizeof(uint32_t), 0);
    log_trace(tt::LogTest, "outputs.size(): {}", outputs.size());
    log_trace(tt::LogTest, "inputs.size(): {}", inputs.size());
    auto input_buffer = CreateBuffer(
        InterleavedBufferConfig{device, static_cast<uint32_t>(num_messages * page_size), page_size, BufferType::DRAM});
    auto output_buffer = CreateBuffer(
        InterleavedBufferConfig{device, static_cast<uint32_t>(num_messages * page_size), page_size, BufferType::DRAM});

    tt_metal::EnqueueWriteBuffer(device->command_queue(), input_buffer, inputs, false);
    // Explicitly overwrite to 0 in case of left over state from prior run(s)
    tt_metal::EnqueueWriteBuffer(device->command_queue(), output_buffer, zeroes_buffer, true);
    const uint32_t dram_input_buf_base_addr = input_buffer->address();

    // For overlay blob on relay core
    constexpr uint32_t dummy_cb_index3 = CB::c_in3;
    auto const& relay_stream_overlay_blob_buffer_cb_config =
        tt_metal::CircularBufferConfig(
            relay_stream_overlay_blob_size_bytes, {{dummy_cb_index3, tt::DataFormat::Float16_b}})
            .set_page_size(dummy_cb_index3, relay_stream_overlay_blob_size_bytes);
    auto first_relay_stream_overlay_blob_cb =
        CreateCircularBuffer(program, first_relay_core, relay_stream_overlay_blob_buffer_cb_config);
    auto second_relay_stream_overlay_blob_cb =
        CreateCircularBuffer(program, second_relay_core, relay_stream_overlay_blob_buffer_cb_config);

    // Sender/Receiver CBs for pulling in/pushing out stimulus data taht we can output compare
    constexpr uint32_t cb_index = CB::c_in0;
    const uint32_t cb_size = page_size_plus_header * read_write_cb_num_pages;
    auto const& cb_config = tt_metal::CircularBufferConfig(cb_size, {{cb_index, tt::DataFormat::Float16_b}})
                                .set_page_size(cb_index, page_size_plus_header);
    auto sender_cb = CreateCircularBuffer(program, sender_core, cb_config);
    auto receiver_cb = CreateCircularBuffer(program, receiver_core, cb_config);

    // Stream Tile Header Buffers
    constexpr uint32_t dummy_cb_index2 = CB::c_in2;
    auto const& stream_tile_header_buffer_cb_config =
        tt_metal::CircularBufferConfig(
            stream_tile_header_buffer_size_bytes, {{dummy_cb_index2, tt::DataFormat::Float16_b}})
            .set_page_size(dummy_cb_index2, stream_tile_header_buffer_size_bytes);
    auto sender_stream_tile_header_buffer_cb =
        CreateCircularBuffer(program, sender_core, stream_tile_header_buffer_cb_config);
    auto first_relay_stream_tile_header_buffer_cb =
        CreateCircularBuffer(program, first_relay_core, stream_tile_header_buffer_cb_config);
    auto second_relay_stream_tile_header_buffer_cb =
        CreateCircularBuffer(program, second_relay_core, stream_tile_header_buffer_cb_config);
    auto receiver_stream_tile_header_buffer_cb =
        CreateCircularBuffer(program, receiver_core, stream_tile_header_buffer_cb_config);

    constexpr uint32_t dummy_cb_index = CB::c_in1;
    auto const& sender_stream_buffer_cb_config =
        tt_metal::CircularBufferConfig(sender_stream_buffer_size_bytes, {{dummy_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dummy_cb_index, sender_stream_buffer_size_bytes);
    auto const& relay_stream_buffer_cb_config =
        tt_metal::CircularBufferConfig(relay_stream_buffer_size_bytes, {{dummy_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dummy_cb_index, relay_stream_buffer_size_bytes);
    auto const& receiver_stream_buffer_cb_config =
        tt_metal::CircularBufferConfig(receiver_stream_buffer_size_bytes, {{dummy_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dummy_cb_index, receiver_stream_buffer_size_bytes);
    auto sender_stream_buffer_cb = CreateCircularBuffer(program, sender_core, sender_stream_buffer_cb_config);
    auto first_relay_stream_buffer_cb = CreateCircularBuffer(program, first_relay_core, relay_stream_buffer_cb_config);
    auto second_relay_stream_buffer_cb =
        CreateCircularBuffer(program, second_relay_core, relay_stream_buffer_cb_config);
    auto receiver_stream_buffer_cb = CreateCircularBuffer(program, receiver_core, receiver_stream_buffer_cb_config);

    program.allocate_circular_buffers();

    uint32_t sender_stream_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, sender_stream_buffer_cb)->address();
    uint32_t first_relay_stream_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, first_relay_stream_buffer_cb)->address();
    uint32_t second_relay_stream_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, second_relay_stream_buffer_cb)->address();
    uint32_t receiver_stream_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, receiver_stream_buffer_cb)->address();
    uint32_t sender_stream_tile_header_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, sender_stream_tile_header_buffer_cb)->address();
    uint32_t first_relay_stream_tile_header_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, first_relay_stream_tile_header_buffer_cb)->address();
    uint32_t second_relay_stream_tile_header_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, second_relay_stream_tile_header_buffer_cb)->address();
    uint32_t receiver_stream_tile_header_buffer_addr =
        tt_metal::detail::GetCircularBuffer(program, receiver_stream_tile_header_buffer_cb)->address();
    uint32_t first_relay_stream_overlay_blob_addr =
        tt_metal::detail::GetCircularBuffer(program, first_relay_stream_overlay_blob_cb)->address();
    uint32_t second_relay_stream_overlay_blob_addr =
        tt_metal::detail::GetCircularBuffer(program, second_relay_stream_overlay_blob_cb)->address();

    uint32_t receiver_cb_address = tt_metal::detail::GetCircularBuffer(program, receiver_cb)->address();
    log_trace(tt::LogTest, "receiver_cb_address: {}", receiver_cb_address);

    TT_ASSERT(sender_stream_buffer_size_bytes % page_size_plus_header == 0);
    TT_ASSERT(relay_stream_buffer_size_bytes % page_size_plus_header == 0);
    TT_ASSERT(receiver_stream_buffer_size_bytes % page_size_plus_header == 0);
    log_trace(
        tt::LogTest, "first_relay_stream_tile_header_buffer_addr: {}", first_relay_stream_tile_header_buffer_addr);
    log_trace(
        tt::LogTest, "second_relay_stream_tile_header_buffer_addr: {}", second_relay_stream_tile_header_buffer_addr);
    stream_config_t sender_stream_config = stream_config_t{
        sender_stream_buffer_addr,
        sender_stream_buffer_size_bytes,
        sender_stream_tile_header_buffer_addr,
        tile_header_buffer_num_messages,
        stream_tile_header_buffer_size_bytes};
    stream_config_t first_relay_stream_config = stream_config_t{
        first_relay_stream_buffer_addr,
        relay_stream_buffer_size_bytes,
        first_relay_stream_tile_header_buffer_addr,
        tile_header_buffer_num_messages,
        stream_tile_header_buffer_size_bytes};
    stream_config_t second_relay_stream_config = stream_config_t{
        second_relay_stream_buffer_addr,
        relay_stream_buffer_size_bytes,
        second_relay_stream_tile_header_buffer_addr,
        tile_header_buffer_num_messages,
        stream_tile_header_buffer_size_bytes};
    stream_config_t receiver_stream_config = stream_config_t{
        receiver_stream_buffer_addr,
        receiver_stream_buffer_size_bytes,
        receiver_stream_tile_header_buffer_addr,
        tile_header_buffer_num_messages,
        stream_tile_header_buffer_size_bytes};

    // TODO: CreateSemaphore api was updated to return an id. Kernels used in this test have not been updated
    uint32_t sender_receiver_semaphore_sender_id = CreateSemaphore(program, sender_core, 0, CoreType::WORKER);
    uint32_t remote_sender_hang_toggle_semaphore_id = CreateSemaphore(program, sender_core, 0, CoreType::WORKER);
    uint32_t first_relay_done_semaphore_id = CreateSemaphore(program, first_relay_core, 0, CoreType::WORKER);
    uint32_t second_relay_done_semaphore_id = CreateSemaphore(program, second_relay_core, 0, CoreType::WORKER);

    uint32_t first_relay_remote_src_start_phase_id = CreateSemaphore(program, first_relay_core, 0, CoreType::WORKER);
    uint32_t second_relay_remote_src_start_phase_id = CreateSemaphore(program, second_relay_core, 0, CoreType::WORKER);
    uint32_t receiver_remote_src_start_phase_id = CreateSemaphore(program, receiver_core, 0, CoreType::WORKER);

    auto sender_noc_id = tt_metal::NOC::NOC_0;
    auto relay_to_relay_data_noc_id = tt_metal::NOC::NOC_0;
    // remote deceiver doesn't handshake properly with noc_1
    auto receiver_noc_id = tt_metal::NOC::NOC_0;
    std::vector<uint32_t> const& sender_reader_rt_args =
        get_sender_reader_rt_args(device, input_buffer->address(), page_size_plus_header, num_messages, sub_sizes);
    std::vector<uint32_t> const& sender_writer_rt_args = get_sender_writer_rt_args(
        device,
        num_messages,
        first_relay_done_semaphore_id,
        first_relay_core,
        sender_noc_id,
        sender_stream_config,
        first_relay_stream_config,
        second_relay_core,
        second_relay_done_semaphore_id,
        sender_receiver_semaphore_sender_id,
        first_relay_remote_src_start_phase_id,
        remote_sender_hang_toggle_semaphore_id);

    log_trace(tt::LogTest, "first_relay_stream_config");
    log_trace(tt::LogTest, "\tfirst_relay_stream_config.buffer_addr: {}", first_relay_stream_config.buffer_addr);
    log_trace(tt::LogTest, "\tfirst_relay_stream_config.buffer_size: {}", first_relay_stream_config.buffer_size);
    log_trace(
        tt::LogTest,
        "\tfirst_relay_stream_config.tile_header_buffer_addr: {}",
        first_relay_stream_config.tile_header_buffer_addr);
    log_trace(
        tt::LogTest,
        "\tfirst_relay_stream_config.tile_header_num_msgs: {}",
        first_relay_stream_config.tile_header_num_msgs);
    log_trace(
        tt::LogTest,
        "\tfirst_relay_stream_config.tile_header_buffer_size: {}",
        first_relay_stream_config.tile_header_buffer_size);
    log_trace(tt::LogTest, "second_relay_stream_config");
    log_trace(tt::LogTest, "\tsecond_relay_stream_config.buffer_addr: {}", second_relay_stream_config.buffer_addr);
    log_trace(tt::LogTest, "\tsecond_relay_stream_config.buffer_size: {}", second_relay_stream_config.buffer_size);
    log_trace(
        tt::LogTest,
        "\tsecond_relay_stream_config.tile_header_buffer_addr: {}",
        second_relay_stream_config.tile_header_buffer_addr);
    log_trace(
        tt::LogTest,
        "\tsecond_relay_stream_config.tile_header_num_msgs: {}",
        second_relay_stream_config.tile_header_num_msgs);
    log_trace(
        tt::LogTest,
        "\tsecond_relay_stream_config.tile_header_buffer_size: {}",
        second_relay_stream_config.tile_header_buffer_size);

    // Need to figure out the noc IDs between the first and second relay. Also double check the
    std::vector<uint32_t> const first_relay_rt_args = get_relay_rt_args(
        device,
        first_relay_stream_overlay_blob_addr,
        first_relay_done_semaphore_id,
        sender_core,
        second_relay_core,
        sender_noc_id,
        relay_to_relay_data_noc_id,
        /*sender_stream_config,*/ first_relay_stream_config,
        second_relay_stream_config,
        first_relay_remote_src_start_phase_id,
        second_relay_remote_src_start_phase_id,
        true);
    std::vector<uint32_t> const second_relay_rt_args = get_relay_rt_args(
        device,
        second_relay_stream_overlay_blob_addr,
        second_relay_done_semaphore_id,
        first_relay_core,
        receiver_core,
        relay_to_relay_data_noc_id,
        receiver_noc_id,
        /*first_relay_stream_config,*/ second_relay_stream_config,
        receiver_stream_config,
        second_relay_remote_src_start_phase_id,
        receiver_remote_src_start_phase_id,
        false);

    std::vector<uint32_t> const& receiver_reader_rt_args = get_receiver_reader_rt_args(
        device,
        num_messages,
        second_relay_done_semaphore_id,
        second_relay_core,
        receiver_noc_id,
        second_relay_stream_config,
        receiver_stream_config,
        first_relay_core,
        first_relay_done_semaphore_id,
        sender_core,
        sender_receiver_semaphore_sender_id,
        receiver_remote_src_start_phase_id);
    std::vector<uint32_t> const& receiver_writer_rt_args =
        get_receiver_writer_rt_args(device, output_buffer->address(), page_size_plus_header, num_messages);

    auto sender_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay_remote_sender_reader.cpp",
        sender_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {tile_header_size, static_cast<uint32_t>(enable_page_size_variations ? 1 : 0)}});
    auto sender_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay_remote_sender.cpp",
        sender_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_1,  // to keep noc coords simple (no calculating noc1 coords)
            .compile_args = {}});

    auto first_relay_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay.cpp",
        first_relay_core,
        tt_metal::DataMovementConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {}});

    auto second_relay_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay.cpp",
        second_relay_core,
        tt_metal::DataMovementConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {}});

    auto receiver_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay_remote_receiver.cpp",
        receiver_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0, .compile_args = {}});
    auto receiver_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay_remote_receiver_writer.cpp",
        receiver_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_1,  // to keep noc coords simple (no calculating noc1 coords)
            .compile_args = {tile_header_size}});

    log_trace(tt::LogTest, "sender_reader_rt_args: ");
    for (auto const& arg : sender_reader_rt_args) {
        log_trace(tt::LogTest, "\t{}", arg);
    }
    tt_metal::SetRuntimeArgs(program, sender_reader_kernel, sender_core, sender_reader_rt_args);

    log_trace(tt::LogTest, "sender_writer_rt_args: ");
    for (auto const& arg : sender_writer_rt_args) {
        log_trace(tt::LogTest, "\t{}", arg);
    }
    tt_metal::SetRuntimeArgs(program, sender_writer_kernel, sender_core, sender_writer_rt_args);

    log_trace(tt::LogTest, "first_relay_rt_args: ");
    for (auto const& arg : first_relay_rt_args) {
        log_trace(tt::LogTest, "\t{}", arg);
    }
    tt_metal::SetRuntimeArgs(program, first_relay_kernel, first_relay_core, first_relay_rt_args);

    log_trace(tt::LogTest, "second_relay_rt_args: ");
    for (auto const& arg : second_relay_rt_args) {
        log_trace(tt::LogTest, "\t{}", arg);
    }
    tt_metal::SetRuntimeArgs(program, second_relay_kernel, second_relay_core, second_relay_rt_args);

    log_trace(tt::LogTest, "receiver_reader_rt_args: ");
    for (auto const& arg : receiver_reader_rt_args) {
        log_trace(tt::LogTest, "\t{}", arg);
    }
    tt_metal::SetRuntimeArgs(program, receiver_reader_kernel, receiver_core, receiver_reader_rt_args);

    log_trace(tt::LogTest, "receiver_writer_rt_args: ");
    for (auto const& arg : receiver_writer_rt_args) {
        log_trace(tt::LogTest, "\t{}", arg);
    }
    tt_metal::SetRuntimeArgs(program, receiver_writer_kernel, receiver_core, receiver_writer_rt_args);

    tt::tt_metal::detail::CompileProgram(device, program);
    for (std::size_t i = 0; i < num_loop_iterations; i++) {
        log_debug(tt::LogTest, "Enqueing Program");
        tt_metal::EnqueueProgram(device->command_queue(), program, true);
        log_debug(tt::LogTest, "Calling Finish");
        tt_metal::Finish(device->command_queue());
        if (i == 0) {
            log_debug(tt::LogTest, "Reading Output Buffer");
            tt_metal::EnqueueReadBuffer(device->command_queue(), output_buffer, outputs, true);
        }
    }

    log_debug(tt::LogTest, "outputs.size(): {}", outputs.size());
    log_debug(tt::LogTest, "inputs.size(): {}", inputs.size());
    log_debug(tt::LogTest, "Comparing Outputs");
    TT_ASSERT(inputs.size() == outputs.size());
    if (enable_page_size_variations) {
        uint32_t page_size_words = page_size / sizeof(uint32_t);
        bool matches = true;
        std::size_t size = outputs.size();
        uint32_t sub_size_i = 0;
        uint32_t page_idx = 0;
        for (auto i = 0; i < size; i += page_size_words) {
            std::size_t n_elems = page_size_words - (sub_sizes.at(sub_size_i) * noc_word_size / sizeof(uint32_t));
            sub_size_i = (sub_size_i + 1) % num_sizes;
            bool printed_page_info = false;
            for (auto ii = 0; ii < n_elems; ii++) {
                bool match = outputs.at(i + ii) == inputs.at(i + ii);
                if (!match) {
                    if (!printed_page_info) {
                        printed_page_info = true;
                        log_error(tt::LogTest, "Output Mismatch");
                    }
                    log_trace(
                        tt::LogTest,
                        "Mismatch at index {}:  {} (expected) != {} (actual)",
                        i + ii,
                        inputs.at(i + ii),
                        outputs.at(i + ii));
                    matches = false;
                }
            }
            page_idx++;
        }
        TT_ASSERT(matches);
    } else {
        bool matches = true;
        bool printed = false;
        TT_ASSERT(inputs.size() == outputs.size());
        for (std::size_t i = 0; i < inputs.size(); i++) {
            if (inputs.at(i) != outputs.at(i)) {
                if (!printed) {
                    log_error(tt::LogTest, "Output Mismatch");
                    printed = true;
                }
                matches = false;
                log_trace(
                    tt::LogTest, "Mismatch at index {}:  {} (expected) != {} (actual)", i, inputs.at(i), outputs.at(i));
            }
        }
        TT_ASSERT(matches);
    }
}

}  // namespace tt_metal

}  // namespace tt

TEST_F(CommandQueueFixture, DISABLED_TestAutonomousRelayStreams) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return;
    }
    std::srand(0);

    uint32_t num_loop_iterations = 2;
    uint32_t num_messages_to_send = 1'000'000;
    uint32_t tx_rx_stream_buffer_size_bytes = 16 * 1024;
    uint32_t relay_stream_buffer_size_bytes = 16 * 1024;
    uint32_t tile_header_buffer_num_messages = 1024;
    uint32_t page_size = 4096;
    uint32_t enable_variable_sized_messages = 1;

    auto sender_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto relay_stream_spec =
        tt::tt_metal::stream_builder_spec_t{relay_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto receiver_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};

    std::array<uint32_t, num_sizes> sub_sizes = std::array<uint32_t, num_sizes>{0, 3, 4, 7, 0, 2, 10, 1};

    std::vector<Program> programs;
    tt::tt_metal::build_and_run_autonomous_stream_test(
        programs,
        {device_},
        num_messages_to_send,
        page_size,
        tile_header_buffer_num_messages,
        sender_stream_spec,
        relay_stream_spec,
        receiver_stream_spec,
        enable_variable_sized_messages == 1,
        sub_sizes,
        num_loop_iterations);

    return;
}

TEST_F(CommandQueueFixture, DISABLED_TestAutonomousRelayStreamsSmallPackets) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return;
    }
    std::srand(0);

    uint32_t num_loop_iterations = 10;
    uint32_t num_messages_to_send = 1'000'000;
    uint32_t tx_rx_stream_buffer_size_bytes = 16 * 1024;
    uint32_t relay_stream_buffer_size_bytes = 16 * 1024;
    uint32_t tile_header_buffer_num_messages = 1024;
    uint32_t page_size = 128;
    uint32_t enable_variable_sized_messages = 1;

    auto sender_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto relay_stream_spec =
        tt::tt_metal::stream_builder_spec_t{relay_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto receiver_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};

    std::array<uint32_t, num_sizes> sub_sizes = std::array<uint32_t, num_sizes>{0, 3, 4, 7, 0, 2, 5, 1};

    std::vector<Program> programs;
    tt::tt_metal::build_and_run_autonomous_stream_test(
        programs,
        {device_},
        num_messages_to_send,
        page_size,
        tile_header_buffer_num_messages,
        sender_stream_spec,
        relay_stream_spec,
        receiver_stream_spec,
        enable_variable_sized_messages == 1,
        sub_sizes,
        num_loop_iterations);

    return;
}

TEST_F(CommandQueueFixture, DISABLED_TestAutonomousRelayStreamsLoopingShort) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return;
    }
    std::srand(0);

    uint32_t num_loop_iterations = 50;
    uint32_t num_messages_to_send = 1'000'000;
    uint32_t tx_rx_stream_buffer_size_bytes = 16 * 1024;
    uint32_t relay_stream_buffer_size_bytes = 16 * 1024;
    uint32_t tile_header_buffer_num_messages = 1024;
    uint32_t page_size = 4096;
    uint32_t enable_variable_sized_messages = 1;

    auto sender_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto relay_stream_spec =
        tt::tt_metal::stream_builder_spec_t{relay_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto receiver_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};

    std::array<uint32_t, num_sizes> sub_sizes = std::array<uint32_t, num_sizes>{0, 3, 4, 7, 0, 2, 10, 1};

    std::vector<Program> programs;
    tt::tt_metal::build_and_run_autonomous_stream_test(
        programs,
        {device_},
        num_messages_to_send,
        page_size,
        tile_header_buffer_num_messages,
        sender_stream_spec,
        relay_stream_spec,
        receiver_stream_spec,
        enable_variable_sized_messages == 1,
        sub_sizes,
        num_loop_iterations);

    return;
}

// Too long to run in post commit and these kernels are currently only live in these unit tests anyways
// so we just enable a couple of the unit tests to ensure nobody accidentally introduces compile errors
// or anything like that
TEST_F(CommandQueueFixture, DISABLED_TestAutonomousRelayStreamsLoopingRandomShort) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    // if (num_devices != 8) {
    //     log_info(tt::LogTest, "Need at least 2 devices to run this test");
    //     return;
    // }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return;
    }
    std::srand(0);

    uint32_t num_loop_iterations = 500;
    uint32_t num_messages_to_send = 1'000'000;
    uint32_t tx_rx_stream_buffer_size_bytes = 16 * 1024;
    uint32_t relay_stream_buffer_size_bytes = 16 * 1024;
    uint32_t tile_header_buffer_num_messages = 1024;
    uint32_t page_size = 4096;
    uint32_t enable_variable_sized_messages = 1;

    auto sender_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto relay_stream_spec =
        tt::tt_metal::stream_builder_spec_t{relay_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto receiver_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};

    for (std::size_t i = 0; i < num_loop_iterations; i++) {
        std::array<uint32_t, num_sizes> sub_sizes = {};
        for (auto i = 0; i < num_sizes; i++) {
            sub_sizes.at(i) = std::rand() % (page_size / noc_word_size);
            EXPECT_TRUE(sub_sizes.at(i) < (page_size / noc_word_size));
        }
        std::vector<Program> programs;
        log_info(tt::LogTest, "Iteration: {}", i);
        tt::tt_metal::build_and_run_autonomous_stream_test(
            programs,
            {device_},
            num_messages_to_send,
            page_size,
            tile_header_buffer_num_messages,
            sender_stream_spec,
            relay_stream_spec,
            receiver_stream_spec,
            enable_variable_sized_messages == 1,
            sub_sizes,
            1);
    }
    return;
}

// Too long to run in post commit and these kernels are currently only live in these unit tests anyways
// so we just enable a couple of the unit tests to ensure nobody accidentally introduces compile errors
// or anything like that
TEST_F(CommandQueueFixture, DISABLED_TestAutonomousRelayStreamsLoopingLong) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    // if (num_devices != 8) {
    //     log_info(tt::LogTest, "Need at least 2 devices to run this test");
    //     return;
    // }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return;
    }
    std::srand(0);

    uint32_t num_loop_iterations = 1'000;
    uint32_t num_messages_to_send = 1'000'000;
    uint32_t tx_rx_stream_buffer_size_bytes = 16 * 1024;
    uint32_t relay_stream_buffer_size_bytes = 16 * 1024;
    uint32_t tile_header_buffer_num_messages = 1024;
    uint32_t page_size = 4096;
    uint32_t enable_variable_sized_messages = 1;

    auto sender_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto relay_stream_spec =
        tt::tt_metal::stream_builder_spec_t{relay_stream_buffer_size_bytes, tile_header_buffer_num_messages};
    auto receiver_stream_spec =
        tt::tt_metal::stream_builder_spec_t{tx_rx_stream_buffer_size_bytes, tile_header_buffer_num_messages};

    std::array<uint32_t, num_sizes> sub_sizes = std::array<uint32_t, num_sizes>{0, 3, 4, 7, 0, 2, 10, 1};

    std::vector<Program> programs;
    tt::tt_metal::build_and_run_autonomous_stream_test(
        programs,
        {device_},
        num_messages_to_send,
        page_size,
        tile_header_buffer_num_messages,
        sender_stream_spec,
        relay_stream_spec,
        receiver_stream_spec,
        enable_variable_sized_messages == 1,
        sub_sizes,
        num_loop_iterations);

    return;
}

// Too long to run in post commit and these kernels are currently only live in these unit tests anyways
// so we just enable a couple of the unit tests to ensure nobody accidentally introduces compile errors
// or anything like that
TEST_F(CommandQueueFixture, DISABLED_TestAutonomousRelayStreamsSweep) {
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return;
    }

    // Create array of size `num_sizes` of random integers using c++ random
    std::array<uint32_t, num_sizes> sub_sizes_global = {};
    std::srand(0);
    for (auto i = 0; i < num_sizes; i++) {
        sub_sizes_global.at(i) = std::rand();
    }

    uint32_t num_loop_iterations = 10;
    std::vector<uint32_t> message_counts = {1'000'000};
    std::vector<uint32_t> fw_stream_buffer_sizes = {2 * 1024, 8 * 1024, 16 * 1024, 32 * 1024};
    std::vector<uint32_t> relay_stream_buffer_sizes = {8 * 1024, 16 * 1024, 24 * 1024};
    std::vector<uint32_t> phase_message_counts = {
        // 32, // Hangs on handshake on phase range wrap, or 25th run, whichever comes first
        // 64, // Hangs on handshake on phase range wrap, or 25th run, whichever comes first
        128,  // works with 16KB buffer
        256,  // works with 16KB buffer
        1024  // works with 16KB buffer
    };
    // std::vector<uint32_t> page_size = {2048, 4096};
    std::vector<uint32_t> page_size = {4096};
    for (auto num_messages : message_counts) {
        for (auto fw_stream_buffer_size : fw_stream_buffer_sizes) {
            for (auto relay_stream_buffer_size : relay_stream_buffer_sizes) {
                // auto fw_stream_buffer_size = relay_stream_buffer_size;
                for (auto tile_header_buffer_num_messages : phase_message_counts) {
                    for (auto page_size : page_size) {
                        if (page_size > fw_stream_buffer_size) {
                            continue;
                        }
                        if (page_size > relay_stream_buffer_size) {
                            continue;
                        }
                        uint32_t enable_variable_sized_messages = 1;

                        log_info(
                            tt::LogTest,
                            "num_messages: {}, fw_stream_buffer_size: {}, relay_stream_buffer_size: {}, "
                            "tile_header_buffer_num_messages: {}, page_size: {}, enable_variable_sized_messages: {}",
                            num_messages,
                            fw_stream_buffer_size,
                            relay_stream_buffer_size,
                            tile_header_buffer_num_messages,
                            page_size,
                            enable_variable_sized_messages);

                        auto sender_stream_spec =
                            tt::tt_metal::stream_builder_spec_t{fw_stream_buffer_size, tile_header_buffer_num_messages};
                        auto relay_stream_spec = tt::tt_metal::stream_builder_spec_t{
                            relay_stream_buffer_size, tile_header_buffer_num_messages};
                        auto receiver_stream_spec =
                            tt::tt_metal::stream_builder_spec_t{fw_stream_buffer_size, tile_header_buffer_num_messages};

                        std::array<uint32_t, num_sizes> sub_sizes = {};
                        for (auto i = 0; i < num_sizes; i++) {
                            sub_sizes.at(i) = sub_sizes_global.at(i) % (page_size / noc_word_size);
                            EXPECT_TRUE(sub_sizes.at(i) < (page_size / noc_word_size));
                        }

                        std::vector<Program> programs;
                        tt::tt_metal::build_and_run_autonomous_stream_test(
                            programs,
                            {device_},
                            num_messages,
                            page_size,
                            tile_header_buffer_num_messages,
                            sender_stream_spec,
                            relay_stream_spec,
                            receiver_stream_spec,
                            enable_variable_sized_messages == 1,
                            sub_sizes,
                            num_loop_iterations);
                    }
                }
            }
        }
    }

    return;
}
