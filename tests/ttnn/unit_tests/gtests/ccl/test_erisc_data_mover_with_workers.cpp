
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cstdint>
// #include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "gtest/gtest.h"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/types.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>
#include "common/tt_backend_api_types.hpp"

// #include <tt-metalium/kernel_types.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

// Taken from ccl_common... some dependency annoyance to deal with so just copying it here for now... resolve before
// merging

namespace ttnn::ccl {
void set_edm_runtime_args(
    tt_metal::Program& program,
    tt_metal::KernelHandle edm_kernel_handle,
    const ccl::EriscDatamoverBuilder& edm_builder,
    const CoreCoord& eth_core) {
    const std::vector<uint32_t>& edm_clockwise_kernel_rt_args = edm_builder.get_runtime_args();
    tt_metal::SetRuntimeArgs(program, edm_kernel_handle, eth_core, edm_clockwise_kernel_rt_args);

    std::stringstream ss;
    ss << "EDM ARGS:\n";
    for (const auto& s : edm_clockwise_kernel_rt_args) {
        ss << "\t" << s << "\n";
    }
    log_info(tt::LogOp, "{}", ss.str());
}

}  // namespace ttnn::ccl

class N300TestDevice {
public:
    N300TestDevice() : num_devices_(tt::tt_metal::GetNumAvailableDevices()) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 2 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<ChipId> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);
            devices_ = distributed::MeshDevice::create_unit_meshes(ids);

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
        for (const auto& [device_id, device_ptr] : devices_) {
            device_ptr->close();
        }
    }

    std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

private:
    bool device_open{false};
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    tt_metal::BufferType input_buffer_type;   // = BufferType::L1;
    tt_metal::BufferType output_buffer_type;  // = BufferType::L1;
    tt::DataFormat l1_data_format;            // = tt::DataFormat::Float16_b;
};

struct KernelXY {
    uint16_t x;
    uint16_t y;

    uint32_t to_uint32() const { return y << 16 | x; }
};

void generate_receiver_worker_kernels(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const CoreCoord& worker_core,
    const CoreCoord& edm_core,
    const ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface& edm_channel,
    uint32_t page_size,
    uint32_t num_pages,
    std::size_t num_buffers_per_edm_channel,
    uint32_t num_pages_per_edm_buffer,
    uint32_t worker_semaphore_address,
    uint32_t dram_output_buffer_base_addr,  // remote_output_buffers.at(i)->address();
    distributed::MeshBuffer* dst_buffer,
    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    // Just want a dummy DF
    uint32_t src0_cb_index = CBIndex::c_0;
    tt::DataFormat df;
    if (page_size == 1024) {
        df = tt::DataFormat::Bfp8;
    } else if (page_size == 2048) {
        df = tt::DataFormat::Float16;
    } else {
        df = tt::DataFormat::Float32;
    }
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_size);

    CreateCircularBuffer(program, worker_core, cb_src0_config);
    std::vector<uint32_t> receiver_worker_writer_compile_args{
        num_pages,  //
        page_size,
        num_pages_per_edm_buffer};
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(receiver_worker_writer_compile_args);
    std::vector<uint32_t> receiver_worker_writer_runtime_args{dram_output_buffer_base_addr};
    log_info(tt::LogTest, "\tReceiverWriter CT Args");
    for (const auto& arg : receiver_worker_writer_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tReceiverWriter RT Args");
    for (const auto& arg : receiver_worker_writer_runtime_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }

    std::vector<uint32_t> receiver_worker_receiver_compile_args{
        edm_channel.eth_buffer_l1_address,
        edm_channel.eth_semaphore_l1_address,
        num_buffers_per_edm_channel,
        edm_termination_mode};
    std::vector<uint32_t> receiver_worker_receiver_runtime_args{
        num_pages_per_edm_buffer,
        num_pages,
        page_size,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).x,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).y,
        worker_semaphore_address,
        num_buffers_per_edm_channel};
    log_info(tt::LogTest, "\tReceiverReader CT Args");
    for (const auto& arg : receiver_worker_receiver_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tReceiverReader RT Args");
    for (const auto& arg : receiver_worker_receiver_runtime_args) {
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
        program, receiver_worker_receiver_kernel, worker_core, receiver_worker_receiver_runtime_args);
    tt_metal::SetRuntimeArgs(program, receiver_worker_writer_kernel, worker_core, receiver_worker_writer_runtime_args);
}

void generate_sender_worker_kernels(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    const CoreCoord& worker_core,
    const CoreCoord& edm_core,
    const ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface& edm_channel,
    uint32_t page_size,
    uint32_t num_pages_total,
    std::size_t num_buffers_per_edm_channel,
    uint32_t num_pages_per_edm_buffer,
    uint32_t worker_semaphore_address,
    uint32_t dram_output_buffer_base_addr,  // remote_output_buffers.at(i)->address();
    distributed::MeshBuffer* src_buffer,
    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode) {
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    std::vector<uint32_t> sender_worker_reader_compile_args{
        num_pages_total,  //
        page_size,
        num_pages_per_edm_buffer};
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(sender_worker_reader_compile_args);
    std::vector<uint32_t> sender_worker_reader_runtime_args{dram_output_buffer_base_addr};

    log_info(tt::LogTest, "\tSenderReader CT Args");
    for (const auto& arg : sender_worker_reader_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tSenderReader RT Args");
    for (const auto& arg : sender_worker_reader_runtime_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }

    std::vector<uint32_t> sender_worker_writer_compile_args{
        num_pages_per_edm_buffer, num_pages_total, page_size, num_buffers_per_edm_channel, edm_termination_mode};
    std::vector<uint32_t> sender_worker_writer_runtime_args{
        edm_channel.eth_buffer_l1_address,
        edm_channel.eth_semaphore_l1_address,
        worker_semaphore_address,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).x,
        (uint32_t)device->ethernet_core_from_logical_core(edm_core).y,
        num_buffers_per_edm_channel};
    uint32_t src0_cb_index = CBIndex::c_0;
    log_info(tt::LogTest, "\tSenderWriter CT Args");
    for (const auto& arg : sender_worker_writer_compile_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    log_info(tt::LogTest, "\tSenderWriter RT Args");
    for (const auto& arg : sender_worker_writer_runtime_args) {
        log_info(tt::LogTest, "\t\t{}", arg);
    }
    // Just want a dummy DF
    tt::DataFormat df;
    if (page_size == 1024) {
        df = tt::DataFormat::Bfp8;
    } else if (page_size == 2048) {
        df = tt::DataFormat::Float16;
    } else {
        df = tt::DataFormat::Float32;
    }
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(2 * num_pages_per_edm_buffer * page_size, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, page_size);
    CreateCircularBuffer(program, worker_core, cb_src0_config);
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
    tt_metal::SetRuntimeArgs(program, sender_worker_reader_kernel, worker_core, sender_worker_reader_runtime_args);
    tt_metal::SetRuntimeArgs(program, sender_worker_writer_kernel, worker_core, sender_worker_writer_runtime_args);
}

bool RunWriteBWTest(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& sender_mesh_device,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& receiver_mesh_device,

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

    ttnn::ccl::EriscDataMoverTerminationMode edm_termination_mode) {
    std::size_t tensor_size_bytes = num_pages_total * page_size;

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload sender_workload;
    tt_metal::Program sender_program_{};
    sender_workload.add_program(device_range, std::move(sender_program_));
    auto& sender_program = sender_workload.get_programs().at(device_range);

    distributed::MeshWorkload receiver_workload;
    tt_metal::Program receiver_program_{};
    receiver_workload.add_program(device_range, std::move(receiver_program_));
    auto& receiver_program = receiver_workload.get_programs().at(device_range);

    auto& sender_cq = sender_mesh_device->mesh_command_queue();
    auto& receiver_cq = receiver_mesh_device->mesh_command_queue();

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
    for (const auto& worker_core : worker_cores) {
        local_worker_semaphore_addresses.push_back(tt::tt_metal::CreateSemaphore(sender_program, worker_core, 0));
        remote_worker_semaphore_addresses.push_back(tt::tt_metal::CreateSemaphore(receiver_program, worker_core, 0));
        log_info(
            tt::LogTest,
            "worker_core=(x={},y={}), local_worker_semaphore_address={}, remote_worker_semaphore_address={}",
            worker_core.x,
            worker_core.y,
            local_worker_semaphore_addresses.back(),
            remote_worker_semaphore_addresses.back());
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
        .input_buffer_type = src_is_dram ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1,
        .output_buffer_type = dest_is_dram ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};

    distributed::DeviceLocalBufferConfig local_input_config{
        .page_size = test_config.page_size_bytes,
        .buffer_type = test_config.input_buffer_type,
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = test_config.size_bytes,
    };
    auto local_input_buffer =
        distributed::MeshBuffer::create(buffer_config, local_input_config, sender_mesh_device.get());
    auto remote_input_buffer =
        distributed::MeshBuffer::create(buffer_config, local_input_config, receiver_mesh_device.get());

    distributed::WriteShard(sender_cq, local_input_buffer, inputs, zero_coord);
    distributed::WriteShard(receiver_cq, remote_input_buffer, inputs, zero_coord);

    std::vector<uint32_t> local_input_buffer_addresses(num_local_sender_channels, local_input_buffer->address());
    std::vector<uint32_t> remote_input_buffer_addresses(num_remote_sender_channels, remote_input_buffer->address());

    ////////////////////////////////////////////////////////////////////////////
    //   EMPTY INITIALIZE THE OUTPUT CB
    ////////////////////////////////////////////////////////////////////////////

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    std::vector<std::shared_ptr<distributed::MeshBuffer>> local_output_buffers;
    std::vector<std::shared_ptr<distributed::MeshBuffer>> remote_output_buffers;

    distributed::DeviceLocalBufferConfig local_output_config{
        .page_size = test_config.page_size_bytes,
        .buffer_type = test_config.output_buffer_type,
    };

    for (std::size_t i = 0; i < num_local_sender_channels; i++) {
        auto output_buffer =
            distributed::MeshBuffer::create(buffer_config, local_output_config, receiver_mesh_device.get());
        remote_output_buffers.push_back(output_buffer);
    }
    for (std::size_t i = 0; i < num_remote_sender_channels; i++) {
        auto output_buffer =
            distributed::MeshBuffer::create(buffer_config, local_output_config, sender_mesh_device.get());
        local_output_buffers.push_back(output_buffer);
    }

    for (const auto& buffer_id : local_output_buffers) {
        distributed::WriteShard(sender_cq, buffer_id, all_zeros, zero_coord);
    }
    for (const auto& buffer_id : remote_output_buffers) {
        distributed::WriteShard(receiver_cq, buffer_id, all_zeros, zero_coord);
    }

    uint32_t erisc_handshake_address = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    std::vector<uint32_t> chip0_edm_args = {erisc_handshake_address};

    ////////////////////////////////////////////////////////////////////////////
    // EDM Builder Setup
    ////////////////////////////////////////////////////////////////////////////

    ttnn::ccl::EriscDataMoverBufferSharingMode buffer_sharing_mode =
        ttnn::ccl::EriscDataMoverBufferSharingMode::NOT_SHARED;

    const std::size_t num_edm_channels = num_local_sender_channels + num_remote_sender_channels;
    // TODO: Allow an override of EDM buffer size
    auto local_chip_edm_builder = ttnn::ccl::create_erisc_datamover_builder(
        num_edm_channels, page_size, num_buffers_per_edm_channel, buffer_sharing_mode, edm_termination_mode);
    auto remote_chip_edm_builder = ttnn::ccl::create_erisc_datamover_builder(
        num_edm_channels, page_size, num_buffers_per_edm_channel, buffer_sharing_mode, edm_termination_mode);

    const uint32_t num_bytes_per_send = local_chip_edm_builder.get_eth_buffer_size_bytes();
    const uint32_t pages_per_send = num_bytes_per_send / page_size;
    TT_FATAL(num_bytes_per_send > 0, "num_bytes_per_send must be greater than 0");
    TT_FATAL(num_bytes_per_send >= page_size, "num_bytes_per_send must be at least page_size");
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
        const auto& worker_core_local_chip = ttnn::ccl::WorkerXY(
            sender_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).x,
            sender_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        const auto& worker_core_remote_chip = ttnn::ccl::WorkerXY(
            receiver_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).x,
            receiver_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        const ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface& local_sender_channel_buffer =
            local_chip_edm_builder.add_sender_channel(
                local_worker_semaphore_addresses.at(i),
                num_messages_to_send_over_channel.at(i),
                {worker_core_local_chip});
        local_edm_channels.push_back(local_sender_channel_buffer);
        const ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface& remote_receiver_channel_buffer =
            remote_chip_edm_builder.add_receiver_channel(
                remote_worker_semaphore_addresses.at(i),
                num_messages_to_send_over_channel.at(i),
                {worker_core_remote_chip});
        remote_edm_channels.push_back(remote_receiver_channel_buffer);
    }
    for (uint32_t i = num_local_sender_channels; i < num_local_sender_channels + num_remote_sender_channels; i++) {
        const auto& worker_core_remote_chip = ttnn::ccl::WorkerXY(
            receiver_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).x,
            receiver_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        const auto& worker_core_local_chip = ttnn::ccl::WorkerXY(
            sender_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).x,
            sender_mesh_device->worker_core_from_logical_core(worker_cores.at(i)).y);
        const ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface& local_receiver_channel_buffer =
            local_chip_edm_builder.add_receiver_channel(
                local_worker_semaphore_addresses.at(i),
                num_messages_to_send_over_channel.at(i),
                {worker_core_remote_chip});
        local_edm_channels.push_back(local_receiver_channel_buffer);
        const ttnn::ccl::EriscDatamoverBuilder::ChannelBufferInterface& remote_sender_channel_buffer =
            remote_chip_edm_builder.add_sender_channel(
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
        const auto& worker_core = worker_cores.at(i);
        log_info(tt::LogTest, "Worker {}. On Core x={},y={}", i, worker_core.x, worker_core.y);
        generate_sender_worker_kernels(
            sender_workload,
            sender_mesh_device,
            worker_core,
            eth_sender_core,
            local_edm_channels.at(i),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            local_worker_semaphore_addresses.at(i),
            local_input_buffer_addresses.at(i),
            local_input_buffer.get(),
            edm_termination_mode);
        generate_receiver_worker_kernels(
            receiver_workload,
            receiver_mesh_device,
            worker_core,
            eth_receiver_core,
            remote_edm_channels.at(i),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            remote_worker_semaphore_addresses.at(i),
            remote_output_buffers.at(i)->address(),
            remote_output_buffers.at(i).get(),
            edm_termination_mode);
    }
    log_info(tt::LogTest, "Generating remote_sender -> local_receiver workers");
    for (uint32_t i = 0; i < num_remote_sender_channels; i++) {
        log_info(tt::LogTest, "Worker {}", i);
        const auto& worker_core = worker_cores.at(i + num_local_sender_channels);
        generate_sender_worker_kernels(
            receiver_workload,
            receiver_mesh_device,
            worker_core,
            eth_receiver_core,
            remote_edm_channels.at(i + num_local_sender_channels),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            remote_worker_semaphore_addresses.at(i + num_local_sender_channels),
            remote_input_buffer_addresses.at(i),
            local_input_buffer.get(),
            edm_termination_mode);

        generate_receiver_worker_kernels(
            sender_workload,
            sender_mesh_device,
            worker_core,
            eth_sender_core,
            local_edm_channels.at(i + num_local_sender_channels),
            page_size,
            num_pages_total,
            num_buffers_per_edm_channel,
            pages_per_send,
            local_worker_semaphore_addresses.at(i + num_local_sender_channels),
            local_output_buffers.at(i)->address(),
            local_output_buffers.at(i).get(),
            edm_termination_mode);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Build EDMs
    ////////////////////////////////////////////////////////////////////////////
    auto local_edm_kernel = ttnn::ccl::generate_edm_kernel(
        sender_program,
        sender_mesh_device->get_devices()[0],
        local_chip_edm_builder,
        eth_sender_core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::NOC_0);
    set_edm_runtime_args(sender_program, local_edm_kernel, local_chip_edm_builder, eth_sender_core);

    auto remote_edm_kernel = ttnn::ccl::generate_edm_kernel(
        receiver_program,
        receiver_mesh_device->get_devices()[0],
        remote_chip_edm_builder,
        eth_receiver_core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::NOC_0);
    set_edm_runtime_args(receiver_program, remote_edm_kernel, remote_chip_edm_builder, eth_receiver_core);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    log_info(tt::LogTest, "Compiling and Running...");

    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread th2 = std::thread([&] { distributed::EnqueueMeshWorkload(sender_cq, sender_workload, false); });
        std::thread th1 = std::thread([&] { distributed::EnqueueMeshWorkload(receiver_cq, receiver_workload, false); });

        th2.join();
        th1.join();
    } else {
        distributed::EnqueueMeshWorkload(sender_cq, sender_workload, false);
        distributed::EnqueueMeshWorkload(receiver_cq, receiver_workload, false);

        log_debug(tt::LogTest, "Calling Finish");
        distributed::Finish(sender_cq);
        distributed::Finish(receiver_cq);
    }
    // tt::tt_metal::detail::ReadDeviceProfilerResults(receiver_device);
    // tt::tt_metal::detail::ReadDeviceProfilerResults(sender_device);
    log_info(tt::LogTest, "Reading back outputs");

    auto is_output_correct = [&all_zeros, &inputs](const std::shared_ptr<distributed::MeshBuffer>& output_buffer) {
        constexpr bool debug_mode = false;
        std::vector<uint32_t> readback_data_vec(all_zeros.size());  // init to 0 data for easier debug
        std::fill(readback_data_vec.begin(), readback_data_vec.end(), 0);

        distributed::ReadShard(
            output_buffer->device()->mesh_command_queue(),
            readback_data_vec,
            output_buffer,
            distributed::MeshCoordinate(0, 0));
        log_info(tt::LogTest, "Checking outputs");
        if (readback_data_vec.size() != inputs.size()) {
            log_error(tt::LogTest, "Output size mismatch: expected {} got {}", inputs.size(), readback_data_vec.size());
            return false;
        }
        bool pass = (readback_data_vec == inputs);
        TT_FATAL(
            std::any_of(inputs.begin(), inputs.end(), [](uint32_t x) { return x != 0; }),
            "Input buffer expected to not be all 0");
        if (not pass) {
            log_error(tt::LogTest, "Output mismatch");
            if (debug_mode) {
                std::size_t num_printed_mismatches = 0;
                for (size_t i = 0; i < readback_data_vec.size() && num_printed_mismatches < 64; i++) {
                    if (readback_data_vec[i] != inputs[i]) {
                        log_error(tt::LogTest, "[{}]: expected {} got {}", i, inputs[i], readback_data_vec[i]);
                        num_printed_mismatches++;
                    }
                }
                log_error(tt::LogTest, "... (remaining mismatches omitted)");
            }
        }
        return pass;
    };

    bool pass = true;
    constexpr bool enable_check = true;
    if constexpr (enable_check) {
        for (const auto& output_buffer : local_output_buffers) {
            pass &= is_output_correct(output_buffer);
        }
        for (const auto& output_buffer : remote_output_buffers) {
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
    ttnn::ccl::EriscDataMoverTerminationMode termination_mode) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        log_info(tt::LogTest, "This test can only be run on n300 devices");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return 0;
    }

    N300TestDevice test_fixture;

    const auto& mesh_device_0 = test_fixture.devices_.at(0);
    auto* device_0 = mesh_device_0->get_devices()[0];

    const auto& active_eth_cores = device_0->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    auto eth_sender_core_iter_end = active_eth_cores.end();
    ChipId device_id = std::numeric_limits<ChipId>::max();
    tt_xy_pair eth_receiver_core;
    tt_xy_pair eth_sender_core;
    do {
        TT_FATAL(eth_sender_core_iter != eth_sender_core_iter_end, "Error");
        std::tie(device_id, eth_receiver_core) = device_0->get_connected_ethernet_core(*eth_sender_core_iter);
        eth_sender_core = *eth_sender_core_iter;
        eth_sender_core_iter++;
    } while (device_id != 1);
    TT_FATAL(device_id == 1, "Expected device_id to be 1");
    const auto& mesh_device_1 = test_fixture.devices_.at(device_id);

    bool success = false;
    try {
        success = RunWriteBWTest(
            mesh_device_0,
            mesh_device_1,

            eth_sender_core,
            eth_receiver_core,

            num_local_sender_channels,    // from args
            num_remote_sender_channels,   // from args
            num_buffers_per_edm_channel,  // from args

            page_size,
            num_pages_total,
            src_is_dram,
            dest_is_dram,

            termination_mode);
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Caught exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}

////////////////////////////////////////////////////////////////////
///  MESSAGE COUNT TERMINATION MODE
////////////////////////////////////////////////////////////////////

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_0ChannelForward_1ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 0;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_3BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 3;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_2ChannelForward_2ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 2;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_4ChannelForward_4ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 4;
    const uint32_t num_remote_sender_channels = 4;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_4ChannelForward_4ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_MessageCountTermination) {
    const uint32_t num_local_sender_channels = 4;
    const uint32_t num_remote_sender_channels = 4;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

////////////////////////////////////////////////////////////////////
///  WORKER_INITIATED_TERMINATION_MODE
////////////////////////////////////////////////////////////////////

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_1BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 1;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_0ChannelsReverse_3BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 0;
    const uint32_t num_buffers_per_edm_channel = 3;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_1ChannelForward_1ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 1;
    const uint32_t num_remote_sender_channels = 1;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}

TEST(
    WorkerEdmDatapath,
    DISABLED_MergedPayloadAndSignal_4ChannelForward_4ChannelsReverse_2BufferPerChannel_2048PageSize_100kPages_WorkerInitiatedTermination) {
    const uint32_t num_local_sender_channels = 4;
    const uint32_t num_remote_sender_channels = 4;
    const uint32_t num_buffers_per_edm_channel = 2;
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 100000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
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
        termination_mode);
    ASSERT_EQ(result, 0);
}
