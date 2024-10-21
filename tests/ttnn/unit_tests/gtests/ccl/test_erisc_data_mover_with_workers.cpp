
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>

#include "gtest/gtest.h"

#include "device/tt_arch_types.h"
// #include "tt_backend_api_types.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp"

// #include "impl/kernels/kernel_types.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

// Taken from ccl_common... some dependency annoyance to deal with so just copying it here for now... resolve before merging
namespace ttnn {
namespace ccl {
void set_edm_runtime_args(
    tt_metal::Program& program,
    KernelHandle edm_kernel_handle,
    ccl::EriscDatamoverBuilder const& edm_builder,
    CoreCoord const& eth_core
) {
    std::vector<uint32_t> const& edm_clockwise_kernel_rt_args = edm_builder.emit_runtime_args();
    tt_metal::SetRuntimeArgs(program, edm_kernel_handle, eth_core, edm_clockwise_kernel_rt_args);

    std::stringstream ss;
    ss << "EDM ARGS:\n";
    for (auto const& s : edm_clockwise_kernel_rt_args) {
        ss << "\t" << s << "\n";
    }
    log_info(tt::LogOp, "{}", ss.str());
}

} // namespace ccl
} // namespace ttnn


class N300TestDevice {
   public:
    N300TestDevice() : device_open(false) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 2 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<chip_id_t> ids(num_devices_,0);
            std::iota(ids.begin(), ids.end(), 0);
            devices_ = tt::tt_metal::detail::CreateDevices(ids);

        } else {
            TT_THROW("This suite can only be run on N300 Wormhole devices");
        }
        device_open = true;
    }
    ~N300TestDevice() {
        if (device_open) {
            TearDown();
        }
    }

    void TearDown() {
        device_open = false;
        for (auto [device_id, device_ptr] : devices_) {
            tt::tt_metal::CloseDevice(device_ptr);
        }
    }

    std::map<chip_id_t, Device *> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

   private:
    bool device_open;
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    BufferType input_buffer_type;   // = BufferType::L1;
    BufferType output_buffer_type;  // = BufferType::L1;
    tt::DataFormat l1_data_format;  // = tt::DataFormat::Float16_b;
};

struct KernelXY {
    uint16_t x;
    uint16_t y;

    uint32_t to_uint32() const { return y << 16 | x; }
};

void generate_receiver_worker_kernels(
    Program &program,
    Device *device,
    CoreCoord const& worker_core,
    CoreCoord const& edm_core,
    ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& edm_channel,
    uint32_t page_size,
    uint32_t num_pages,
    std::size_t num_buffers_per_edm_channel,
    uint32_t num_pages_per_edm_buffer,
    uint32_t worker_semaphore_address,
    uint32_t dram_output_buffer_base_addr, // remote_output_buffers.at(i)->address();
    bool dest_is_dram,
    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode
) {
    // Just want a dummy DF
    uint32_t src0_cb_index = CB::c_in0;
    tt::DataFormat df = page_size == 1024 ? tt::DataFormat::Bfp8 :
                        page_size == 2048 ? tt::DataFormat::Float16 :
                                                         tt::DataFormat::Float32;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{src0_cb_index, df}})
		.set_page_size(src0_cb_index, page_size);

    CBHandle receiver_workers_cb = CreateCircularBuffer(program, worker_core, cb_src0_config);
    std::vector<uint32_t> receiver_worker_writer_compile_args{
        dest_is_dram,  //
        num_pages,     //
        page_size,
        num_pages_per_edm_buffer};
    std::vector<uint32_t> receiver_worker_writer_runtime_args{dram_output_buffer_base_addr};
    log_info(tt::LogTest, "\tReceiverWriter CT Args");
    for (auto const& arg : receiver_worker_writer_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tReceiverWriter RT Args");
    for (auto const& arg : receiver_worker_writer_runtime_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }


    std::vector<uint32_t> receiver_worker_receiver_compile_args{
        edm_channel.eth_buffer_l1_address,
        edm_channel.eth_semaphore_l1_address,
        num_buffers_per_edm_channel,
        edm_termination_mode
    };
    std::vector<uint32_t> receiver_worker_receiver_runtime_args{
        num_pages_per_edm_buffer,
        num_pages,
        page_size,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).x,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).y,
        worker_semaphore_address,
        num_buffers_per_edm_channel};
    log_info(tt::LogTest, "\tReceiverReader CT Args");
    for (auto const& arg : receiver_worker_receiver_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tReceiverReader RT Args");
    for (auto const& arg : receiver_worker_receiver_runtime_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }


    auto receiver_worker_receiver_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/erisc_datamover_receiver_worker_reader.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = receiver_worker_receiver_compile_args});
    auto receiver_worker_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/erisc_datamover_receiver_worker_sender.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = receiver_worker_writer_compile_args});
    tt_metal::SetRuntimeArgs(
        program,
        receiver_worker_receiver_kernel,
        worker_core,
        receiver_worker_receiver_runtime_args);
    tt_metal::SetRuntimeArgs(
        program,
        receiver_worker_writer_kernel,
        worker_core,
        receiver_worker_writer_runtime_args);
}

void generate_sender_worker_kernels(
    Program &program,
    Device *device,
    CoreCoord const& worker_core,
    CoreCoord const& edm_core,
    ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& edm_channel,
    uint32_t page_size,
    uint32_t num_pages_total,
    std::size_t num_buffers_per_edm_channel,
    uint32_t num_pages_per_edm_buffer,
    uint32_t worker_semaphore_address,
    uint32_t dram_output_buffer_base_addr, // remote_output_buffers.at(i)->address();
    bool src_is_dram,
    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode
) {
    std::vector<uint32_t> sender_worker_reader_compile_args{
        src_is_dram,  //
        num_pages_total,     //
        page_size,
        num_pages_per_edm_buffer};
    std::vector<uint32_t> sender_worker_reader_runtime_args{dram_output_buffer_base_addr};

    log_info(tt::LogTest, "\tSenderReader CT Args");
    for (auto const& arg : sender_worker_reader_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tSenderReader RT Args");
    for (auto const& arg : sender_worker_reader_runtime_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }

    std::vector<uint32_t> sender_worker_writer_compile_args{
        num_pages_per_edm_buffer,
        num_pages_total,
        page_size,
        num_buffers_per_edm_channel,
        edm_termination_mode
    };
    std::vector<uint32_t> sender_worker_writer_runtime_args{
        edm_channel.eth_buffer_l1_address,
        edm_channel.eth_semaphore_l1_address,
        worker_semaphore_address,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).x,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).y,
        num_buffers_per_edm_channel
    };
    uint32_t src0_cb_index = CB::c_in0;
    log_info(tt::LogTest, "\tSenderWriter CT Args");
    for (auto const& arg : sender_worker_writer_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tSenderWriter RT Args");
    for (auto const& arg : sender_worker_writer_runtime_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    // Just want a dummy DF
    tt::DataFormat df = page_size == 1024 ? tt::DataFormat::Bfp8 :
                        page_size == 2048 ? tt::DataFormat::Float16 :
                                                         tt::DataFormat::Float32;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{src0_cb_index, df}})
		.set_page_size(src0_cb_index, page_size);
    CBHandle sender_workers_cb = CreateCircularBuffer(program, worker_core, cb_src0_config);
    auto sender_worker_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/erisc_datamover_sender_worker_reader.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = sender_worker_reader_compile_args});
    auto sender_worker_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/ccl/kernels/erisc_datamover_sender_worker_sender.cpp",
        worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = sender_worker_writer_compile_args});
    tt_metal::SetRuntimeArgs(
        program,
        sender_worker_reader_kernel,
        worker_core,
        sender_worker_reader_runtime_args);
    tt_metal::SetRuntimeArgs(
        program,
        sender_worker_writer_kernel,
        worker_core,
        sender_worker_writer_runtime_args);
}



bool RunWriteBWTest(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const uint32_t num_local_sender_channels,
    const uint32_t num_remote_sender_channels,

    // default is 1.
    // 2 means channel is double buffered
    // 3 means channel is triple buffered
    // ... and so on
    std::size_t num_buffers_per_edm_channel,

    const uint32_t page_size,
    const uint32_t num_pages_total,
    bool src_is_dram,
    bool dest_is_dram,

    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode
) {

    std::size_t tensor_size_bytes = num_pages_total * page_size;

    tt_metal::Program sender_program{};
    tt_metal::Program receiver_program{};

    std::vector<CoreCoord> worker_cores;
    {
        std::size_t row = 0;
        std::size_t col = 0;
        for (uint32_t i = 0; i < num_local_sender_channels + num_remote_sender_channels; i++) {
            worker_cores.push_back(CoreCoord(col, row));
            col++;
            if (col == 8) {
                col = 0;
                row++;
            }
        }
    }

    std::vector<uint32_t> local_worker_semaphore_addresses;
    std::vector<uint32_t> remote_worker_semaphore_addresses;
    for (auto const& worker_core : worker_cores) {
        local_worker_semaphore_addresses.push_back(tt::tt_metal::CreateSemaphore(sender_program, worker_core, 0));
        remote_worker_semaphore_addresses.push_back(tt::tt_metal::CreateSemaphore(receiver_program, worker_core, 0));
        log_info(tt::LogTest, "worker_core=(x={},y={}), local_worker_semaphore_address={}, remote_worker_semaphore_address={}",
                  worker_core.x, worker_core.y, local_worker_semaphore_addresses.back(), remote_worker_semaphore_addresses.back());
    }

    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, tensor_size_bytes / sizeof(uint32_t));
    std::iota(inputs.begin(), inputs.end(), 0);

    BankedConfig test_config = BankedConfig{
        .num_pages = num_pages_total,
        .size_bytes = tensor_size_bytes,
        .page_size_bytes = page_size,
        .input_buffer_type = src_is_dram ? BufferType::DRAM : BufferType::L1,
        .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};

    auto local_input_buffer = CreateBuffer(InterleavedBufferConfig{
        sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});
    auto remote_input_buffer = CreateBuffer(InterleavedBufferConfig{
        receiver_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});
    bool input_is_dram = test_config.input_buffer_type == BufferType::DRAM;

    tt_metal::detail::WriteToBuffer(local_input_buffer, inputs);
    tt_metal::detail::WriteToBuffer(remote_input_buffer, inputs);

    std::vector<uint32_t> local_input_buffer_addresses(num_local_sender_channels, local_input_buffer->address());
    std::vector<uint32_t> remote_input_buffer_addresses(num_remote_sender_channels, remote_input_buffer->address());

    ////////////////////////////////////////////////////////////////////////////
    //   EMPTY INITIALIZE THE OUTPUT CB
    ////////////////////////////////////////////////////////////////////////////

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    std::vector<std::shared_ptr<Buffer>> local_output_buffers;
    std::vector<std::shared_ptr<Buffer>> remote_output_buffers;

    for (std::size_t i = 0; i < num_local_sender_channels; i++) {
        auto output_buffer = CreateBuffer(InterleavedBufferConfig{
            receiver_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});
        remote_output_buffers.push_back(output_buffer);
    }
    for (std::size_t i = 0; i < num_remote_sender_channels; i++) {
        auto output_buffer = CreateBuffer(InterleavedBufferConfig{
            sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});
        local_output_buffers.push_back(output_buffer);
    }

    bool output_is_dram = test_config.output_buffer_type == BufferType::DRAM;
    for (auto buffer_id : local_output_buffers) {
        tt_metal::detail::WriteToBuffer(buffer_id, all_zeros);
    }
    for (auto buffer_id : remote_output_buffers) {
        tt_metal::detail::WriteToBuffer(buffer_id, all_zeros);
    }

    uint32_t erisc_handshake_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    uint32_t chip0_next_buffer_address = erisc_handshake_address + 16;
    std::vector<uint32_t> chip0_edm_args = {erisc_handshake_address};
    uint32_t chip0_sender_channels_offset = 0;
    uint32_t chip0_arg_sender_num_channels = 1;

    ////////////////////////////////////////////////////////////////////////////
    // EDM Builder Setup
    ////////////////////////////////////////////////////////////////////////////

    ttnn::ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode = ttnn::ccl::EriscDataMoverBufferSharingMode::NOT_SHARED;

    const std::size_t num_edm_channels = num_local_sender_channels + num_remote_sender_channels;
    // TODO: Allow an override of EDM buffer size
    auto local_chip_edm_builder = ttnn::ccl::create_erisc_datamover_builder(
        num_edm_channels, page_size, num_buffers_per_edm_channel, buffer_sharing_mode, edm_termination_mode);
    auto remote_chip_edm_builder = ttnn::ccl::create_erisc_datamover_builder(
        num_edm_channels, page_size, num_buffers_per_edm_channel, buffer_sharing_mode, edm_termination_mode);

    const uint32_t num_bytes_per_send = local_chip_edm_builder.get_eth_buffer_size_bytes();
    const uint32_t pages_per_send = num_bytes_per_send / page_size;
    TT_ASSERT(num_bytes_per_send > 0);
    TT_ASSERT(num_bytes_per_send >= page_size);
    TT_ASSERT(num_bytes_per_send >= page_size);
    const uint32_t num_messages_to_send = (((num_pages_total * page_size) - 1) / num_bytes_per_send) + 1;
    log_info(tt::LogTest, "num_bytes_per_send={}", num_bytes_per_send);
    log_info(tt::LogTest, "page_size={}", page_size);
    log_info(tt::LogTest, "pages_per_send={}", pages_per_send);
    log_info(tt::LogTest, "num_messages_to_send={}", num_messages_to_send);
    std::vector<uint32_t> num_messages_to_send_over_channel(num_edm_channels, num_messages_to_send);

    std::vector<CoreCoord> local_sender_workers;
    std::vector<CoreCoord> remote_receiver_workers;
    std::vector<CoreCoord> remote_sender_workers;
    std::vector<CoreCoord> local_receiver_workers;

    // setup edm channels
    std::vector<ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface> local_edm_channels;
    std::vector<ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface> remote_edm_channels;
    for (uint32_t i = 0; i < num_local_sender_channels; i++) {
        auto const& worker_core_local_chip = ttnn::ccl::WorkerXY(
                            sender_device->worker_core_from_logical_core(worker_cores.at(i)).x,
                            sender_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        auto const& worker_core_remote_chip = ttnn::ccl::WorkerXY(
                            receiver_device->worker_core_from_logical_core(worker_cores.at(i)).x,
                            receiver_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& local_sender_channel_buffer = local_chip_edm_builder.add_sender_channel(
            local_worker_semaphore_addresses.at(i),
            num_messages_to_send_over_channel.at(i),
            {worker_core_local_chip});
        local_edm_channels.push_back(local_sender_channel_buffer);
        ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& remote_receiver_channel_buffer = remote_chip_edm_builder.add_receiver_channel(
            remote_worker_semaphore_addresses.at(i),
            num_messages_to_send_over_channel.at(i),
            {worker_core_remote_chip});
        remote_edm_channels.push_back(remote_receiver_channel_buffer);
    }
    for (uint32_t i = num_local_sender_channels; i < num_local_sender_channels + num_remote_sender_channels; i++) {
        auto const& worker_core_remote_chip = ttnn::ccl::WorkerXY(
                            receiver_device->worker_core_from_logical_core(worker_cores.at(i)).x,
                            receiver_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        auto const& worker_core_local_chip = ttnn::ccl::WorkerXY(
                            sender_device->worker_core_from_logical_core(worker_cores.at(i)).x,
                            sender_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& local_receiver_channel_buffer = local_chip_edm_builder.add_receiver_channel(
            local_worker_semaphore_addresses.at(i),
            num_messages_to_send_over_channel.at(i),
            {worker_core_remote_chip});
        local_edm_channels.push_back(local_receiver_channel_buffer);
        ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface const& remote_sender_channel_buffer = remote_chip_edm_builder.add_sender_channel(
            remote_worker_semaphore_addresses.at(i),
            num_messages_to_send_over_channel.at(i),
            {worker_core_local_chip});
        remote_edm_channels.push_back(remote_sender_channel_buffer);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build Workers
    ////////////////////////////////////////////////////////////////////////////
    log_info(tt::LogTest, "Generating local_sender -> remote_receiver workers");
    for (uint32_t i = 0; i < num_local_sender_channels; i++) {
        auto const& worker_core = worker_cores.at(i);
        log_info(tt::LogTest, "Worker {}. On Core x={},y={}", i, worker_core.x, worker_core.y);
        generate_sender_worker_kernels(
            sender_program,
            sender_device,
            worker_core,
            eth_sender_core,
            local_edm_channels.at(i),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            local_worker_semaphore_addresses.at(i),
            local_input_buffer_addresses.at(i),
            src_is_dram,
            edm_termination_mode
        );
        generate_receiver_worker_kernels(
            receiver_program,
            receiver_device,
            worker_core,
            eth_receiver_core,
            remote_edm_channels.at(i),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            remote_worker_semaphore_addresses.at(i),
            remote_output_buffers.at(i)->address(),
            dest_is_dram,
            edm_termination_mode
        );

    }
    log_info(tt::LogTest, "Generating remote_sender -> local_receiver workers");
    for (uint32_t i = 0; i < num_remote_sender_channels; i++) {
        log_info(tt::LogTest, "Worker {}", i);
        auto const& worker_core = worker_cores.at(i + num_local_sender_channels);
        generate_sender_worker_kernels(
            receiver_program,
            receiver_device,
            worker_core,
            eth_receiver_core,
            remote_edm_channels.at(i + num_local_sender_channels),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            remote_worker_semaphore_addresses.at(i + num_local_sender_channels),
            remote_input_buffer_addresses.at(i),
            src_is_dram,
            edm_termination_mode
        );

        generate_receiver_worker_kernels(
            sender_program,
            sender_device,
            worker_core,
            eth_sender_core,
            local_edm_channels.at(i + num_local_sender_channels),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            local_worker_semaphore_addresses.at(i + num_local_sender_channels),
            local_output_buffers.at(i)->address(),
            dest_is_dram,
            edm_termination_mode
        );
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build EDMs
    ////////////////////////////////////////////////////////////////////////////
    auto local_edm_kernel = ttnn::ccl::generate_edm_kernel(
        sender_program,
        sender_device,
        local_chip_edm_builder,
        eth_sender_core,
        NOC::NOC_0);
    set_edm_runtime_args(
        sender_program,
        local_edm_kernel,
        local_chip_edm_builder,
        eth_sender_core
    );

    auto remote_edm_kernel = ttnn::ccl::generate_edm_kernel(
        receiver_program,
        receiver_device,
        remote_chip_edm_builder,
        eth_receiver_core,
        NOC::NOC_0);
    set_edm_runtime_args(
        receiver_program,
        remote_edm_kernel,
        remote_chip_edm_builder,
        eth_receiver_core
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    try {
        tt::tt_metal::detail::CompileProgram(sender_device, sender_program);
        tt::tt_metal::detail::CompileProgram(receiver_device, receiver_program);
    } catch (std::exception& e) {
        log_error("Failed compile: {}", e.what());
        throw e;
    }

    log_info(tt::LogTest, "Running...");

    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(sender_device, sender_program); });
        std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(receiver_device, receiver_program); });

        th2.join();
        th1.join();
    } else {
        tt_metal::EnqueueProgram(sender_device->command_queue(), sender_program, false);
        tt_metal::EnqueueProgram(receiver_device->command_queue(), receiver_program, false);

        log_debug(tt::LogTest, "Calling Finish");
        tt_metal::Finish(sender_device->command_queue());
        tt_metal::Finish(receiver_device->command_queue());
    }
    // tt::tt_metal::detail::DumpDeviceProfileResults(receiver_device);
    // tt::tt_metal::detail::DumpDeviceProfileResults(sender_device);
    log_info(tt::LogTest, "Reading back outputs");

    auto is_output_correct = [&all_zeros, &inputs](std::shared_ptr<Buffer> output_buffer) {
        constexpr bool debug_mode = false;
        std::vector<uint32_t> readback_data_vec;  // init to 0 data for easier debug
        readback_data_vec.reserve(all_zeros.size());
        std::fill(readback_data_vec.begin(), readback_data_vec.end(), 0);

        tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
        log_info(tt::LogTest, "Checking outputs");
        if (readback_data_vec.size() != inputs.size()) {
            log_error(tt::LogTest, "Output size mismatch: expected {} got {}", inputs.size(), readback_data_vec.size());
            return false;
        }
        bool pass = (readback_data_vec == inputs);
        TT_ASSERT(
            std::any_of(inputs.begin(), inputs.end(), [](uint32_t x) { return x != 0; }),
            "Input buffer expected to not be all 0");
        if (not pass) {
            log_error("Output mismatch");
            if (debug_mode) {
                std::size_t num_printed_mismatches = 0;
                for (size_t i = 0; i < readback_data_vec.size() && num_printed_mismatches < 64; i++) {
                    if (readback_data_vec[i] != inputs[i]) {
                        log_error("[{}]: expected {} got {}", i, inputs[i], readback_data_vec[i]);
                        num_printed_mismatches++;
                    }
                }
                log_error("... (remaining mismatches omitted)");
            }
        }
        return pass;
    };

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr(enable_check) {
        for (auto const& output_buffer : local_output_buffers) {
            pass &= is_output_correct(output_buffer);
        }
        for (auto const& output_buffer : remote_output_buffers) {
            pass &= is_output_correct(output_buffer);
        }
    }


    return pass;
}

int TestEntrypoint(
    const uint32_t num_local_sender_channels,
    const uint32_t num_remote_sender_channels,
    // default is 1.
    // 2 means channel is double buffered
    // 3 means channel is triple buffered
    // ... and so on
    std::size_t num_buffers_per_edm_channel,
    const uint32_t page_size,
    const uint32_t num_pages_total,
    const bool src_is_dram,
    const bool dest_is_dram,
    ttnn::ccl::EriscDataMoverTerminationMode termination_mode
) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        log_info("This test can only be run on n300 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return 0;
    }

    N300TestDevice test_fixture;

    const auto& device_0 = test_fixture.devices_.at(0);

    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    auto eth_sender_core_iter_end = active_eth_cores.end();
    chip_id_t device_id = std::numeric_limits<chip_id_t>::max();
    tt_xy_pair eth_receiver_core;
    bool initialized = false;
    tt_xy_pair eth_sender_core;
    do {
        TT_FATAL(eth_sender_core_iter != eth_sender_core_iter_end, "Error");
        std::tie(device_id, eth_receiver_core) = device_0->get_connected_ethernet_core(*eth_sender_core_iter);
        eth_sender_core = *eth_sender_core_iter;
        eth_sender_core_iter++;
    } while (device_id != 1);
    TT_ASSERT(device_id == 1);
    const auto& device_1 = test_fixture.devices_.at(device_id);

    bool success = false;
    try {
        success = RunWriteBWTest(
            device_0,
            device_1,

            eth_sender_core,
            eth_receiver_core,

            num_local_sender_channels, // from args
            num_remote_sender_channels, // from args
            num_buffers_per_edm_channel, // from args

            page_size,
            num_pages_total,
            src_is_dram,
            dest_is_dram,

            termination_mode);
    } catch (std::exception& e) {
        log_error("Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

////////////////////////////////////////////////////////////////////
///  MESSAGE COUNT TERMINATION MODE
////////////////////////////////////////////////////////////////////

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_0ChannelForward_1ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 0;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}


TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}


TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_3BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 3;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_2ChannelForward_2ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 2;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}


TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_4ChannelForward_4ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 4;
    const uint32_t num_remote_sender_channels = 4;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_4ChannelForward_4ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 4;
    const uint32_t num_remote_sender_channels = 4;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}



////////////////////////////////////////////////////////////////////
///  WORKER_INITIATED_TERMINATION_MODE
////////////////////////////////////////////////////////////////////

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_3BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 3;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

TEST(WorkerEdmDatapath, DISABLED_MergedPayloadAndSignal_4ChannelForward_4ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 4;
    const uint32_t num_remote_sender_channels = 4;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const bool merge_message_and_signal = true;
    auto termination_mode = ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED;

    auto result = TestEntrypoint(
        num_local_sender_channels,
        num_remote_sender_channels,
        // default is 1.
        // 2 means channel is double buffered
        // 3 means channel is triple buffered
        // ... and so on
        num_buffers_per_edm_channel,
        page_size,
        num_pages_total,
        src_is_dram,
        dest_is_dram,
        termination_mode
    );
    ASSERT_EQ(result, 0);
}

// EnablePersistentKernelCache
