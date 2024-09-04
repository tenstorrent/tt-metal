
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>

#include "device/tt_arch_types.h"
#include "tt_backend_api_types.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

// #include "impl/kernels/kernel_types.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

class N300TestDevice {
   public:
    N300TestDevice() : device_open(false) {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            TT_THROW("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() == 2 and
            tt::tt_metal::GetNumPCIeDevices() == 1) {
            for (unsigned int id = 0; id < num_devices_; id++) {
                auto* device = tt::tt_metal::CreateDevice(id);
                devices_.push_back(device);
            }
            tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);

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
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

   private:
    bool device_open;
};

struct BankedConfig {
    size_t num_pages;
    size_t size_bytes;
    size_t page_size_bytes;
    tt_metal::BufferType input_buffer_type;   // = BufferType::L1;
    tt_metal::BufferType output_buffer_type;  // = BufferType::L1;
    tt::DataFormat l1_data_format;  // = tt::DataFormat::Float16_b;
};

struct KernelXY {
    uint16_t x;
    uint16_t y;

    uint32_t to_uint32() const { return y << 16 | x; }
};

bool RunWriteBWTest(
    std::string const &sender_side_reader_worker_kernel_path,
    std::string const &sender_side_writer_worker_kernel_path,
    std::string const &receiver_side_reader_worker_kernel_path,
    std::string const &receiver_side_writer_worker_kernel_path,

    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const size_t src_eth_l1_byte_address,
    const size_t dst_eth_l1_byte_address,

    const size_t precomputed_source_addresses_buffer_address,
    const size_t precomputed_source_addresses_buffer_size,

    const uint32_t eth_l1_staging_buffer_size,
    const uint32_t eth_max_concurrent_sends,
    const uint32_t input_buffer_page_size,
    const uint32_t input_buffer_size_bytes,
    bool source_is_dram,
    bool dest_is_dram

) {
    // number of bytes to send per eth send (given that eth l1 buf size not
    // guaranteed to be multiple of page size, we won't send the left over
    // bytes at the end
    const uint32_t pages_per_send = eth_l1_staging_buffer_size / input_buffer_page_size;
    const uint32_t num_bytes_per_send = pages_per_send * input_buffer_page_size;
    const uint32_t num_pages = ((input_buffer_size_bytes - 1) / input_buffer_page_size) + 1;  // includes padding
    const uint32_t num_messages_to_send = ((input_buffer_size_bytes - 1) / num_bytes_per_send) + 1;

    TT_ASSERT(
        precomputed_source_addresses_buffer_address < std::numeric_limits<uint32_t>::max(),
        "precomputed_source_addresses_buffer_address is too large");

    bool pass = true;
    log_debug(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {}",
        input_buffer_size_bytes,
        sender_device->id(),
        eth_sender_core.str(),
        src_eth_l1_byte_address,
        receiver_device->id(),
        eth_receiver_core.str(),
        dst_eth_l1_byte_address);
    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, input_buffer_size_bytes / sizeof(uint32_t));

    // Clear expected value at ethernet L1 address

    BankedConfig test_config = BankedConfig{
        .num_pages = num_pages,
        .size_bytes = input_buffer_size_bytes,
        .page_size_bytes = input_buffer_page_size,
        .input_buffer_type = source_is_dram ? BufferType::DRAM : BufferType::L1,
        .output_buffer_type = dest_is_dram ? BufferType::DRAM : BufferType::L1,
        .l1_data_format = tt::DataFormat::Float16_b};
    auto input_buffer = CreateBuffer(InterleavedBufferConfig{
        sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});

    bool input_is_dram = test_config.input_buffer_type == BufferType::DRAM;
    tt_metal::detail::WriteToBuffer(input_buffer, inputs);
    const uint32_t dram_input_buf_base_addr = input_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    //   EMPTY INITIALIZE THE OUTPUT CB
    ////////////////////////////////////////////////////////////////////////////

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    auto output_buffer = CreateBuffer(InterleavedBufferConfig{
        receiver_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});

    bool output_is_dram = test_config.output_buffer_type == BufferType::DRAM;
    tt_metal::detail::WriteToBuffer(output_buffer, all_zeros);
    const uint32_t dram_output_buffer_base_addr = output_buffer->address();

    // TODO(snijjar): Find a cleaner way to do this
    uint32_t erisc_handshake_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    // MAKE GLOBAL FOR NOW
    auto chip0_sender_worker_core = CoreCoord(0, 0);

    uint32_t chip0_next_buffer_address = erisc_handshake_address + 16;
    std::vector<uint32_t> chip0_edm_args = {erisc_handshake_address};
    uint32_t chip0_sender_channels_offset = 0;
    uint32_t chip0_arg_sender_num_channels = 1;

    ////////////////////////////////////////////////////////////////////////////
    //                  WORKER CB CONFIG
    ////////////////////////////////////////////////////////////////////////////
    uint32_t src0_cb_index = CB::c_in0;

    // Just want a dummy DF
    tt::DataFormat df = input_buffer_page_size == 1024 ? tt::DataFormat::Bfp8 :
                        input_buffer_page_size == 2048 ? tt::DataFormat::Float16 :
                                                         tt::DataFormat::Float32;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(2 * pages_per_send * input_buffer_page_size, {{src0_cb_index, df}})
		.set_page_size(src0_cb_index, input_buffer_page_size);

    ////////////////////////////////////////////////////////////////////////////
    //                               Device 0
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program sender_program = tt_metal::Program();
    uint32_t chip0_worker_semaphore_id = tt::tt_metal::CreateSemaphore(sender_program, chip0_sender_worker_core, 0);

    chip0_edm_args.push_back(chip0_sender_channels_offset);

    // 3) sender num channels (How many erisc channels to use. ALso how many buffers to instantiate)
    log_debug(tt::LogTest, "------------------ Device 0 args ----------------");
    //---------------------
    //                              Device 0 - SENDER
    log_debug(tt::LogTest, "\t-- sender --");
    log_debug(tt::LogTest, "\tchip0_sender_channels_offset: {}", chip0_sender_channels_offset);
    uint32_t chip0_sender_num_channels = chip0_arg_sender_num_channels;
    log_debug(tt::LogTest, "\tchip0_sender_num_channels: {}", chip0_sender_num_channels);
    uint32_t chip0_receiver_channels_offset = chip0_sender_channels_offset + chip0_sender_num_channels;
    uint32_t chip0_sender_erisc_sender_buffer_address = chip0_next_buffer_address;
    uint32_t chip0_eth_sender_l1_base_addr = 0;
    uint32_t chip0_eth_sender_l1_sem_addr = 0;
    for (uint32_t sc = 0; sc < chip0_sender_num_channels; sc++) {
        //    Informs how many times to iterate through the next group of args
        //    4) sender_buffer_address
        chip0_edm_args.push_back(chip0_next_buffer_address);
        chip0_eth_sender_l1_base_addr = chip0_next_buffer_address;
        log_debug(tt::LogTest, "\t\tsender_buffer_address: {}", chip0_next_buffer_address);
        //    5) sender_num_messages_to_send
        chip0_edm_args.push_back(num_messages_to_send);
        log_debug(tt::LogTest, "\t\tchip0_sender_num_messages_to_send: {}", num_messages_to_send);
        //    6) sender_channel_size
        chip0_edm_args.push_back(eth_l1_staging_buffer_size);
        log_debug(tt::LogTest, "\t\tchip0_sender_channel_size: {}", eth_l1_staging_buffer_size);
        chip0_next_buffer_address =
            chip0_next_buffer_address + eth_l1_staging_buffer_size + 16 - (eth_l1_staging_buffer_size % 16);
        //    7) sender_semaphores_base_address
        // ... Any off by one errors and I'm toast :)
        //     erisc local copy of semaphore that workers update remotely
        chip0_edm_args.push_back(chip0_next_buffer_address);
        log_debug(tt::LogTest, "\t\tsender erisc l1 semaphore address: {}", chip0_next_buffer_address);
        chip0_eth_sender_l1_sem_addr = chip0_next_buffer_address;
        chip0_next_buffer_address += 16;
        //    8) worker_semaphore_address
        //       worker local L1 semaphores that erisc updates
        chip0_edm_args.push_back(chip0_worker_semaphore_id);
        log_debug(tt::LogTest, "\t\tworker_semaphores_base_address: {}", chip0_worker_semaphore_id);
        //    9) sender_num_workers
        //       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
        const uint32_t chip0_num_workers_on_channel = 1;
        chip0_edm_args.push_back(chip0_num_workers_on_channel);
        log_debug(tt::LogTest, "\t\tchip0_num_workers_on_channel: {}", chip0_num_workers_on_channel);
        for (uint32_t w = 0; w < chip0_num_workers_on_channel; w++) {
            //       10) worker_coord(s)
            auto worker_noc_coord = sender_device->physical_core_from_logical_core(chip0_sender_worker_core, CoreType::WORKER);
            chip0_edm_args.push_back(KernelXY{
                static_cast<uint16_t>(worker_noc_coord.x), static_cast<uint16_t>(worker_noc_coord.y)}
                                         .to_uint32());
            log_debug(
                tt::LogTest,
                "\t\t\tchip0_sender_noc_xy: x={},y={}",
                worker_noc_coord.x,
                worker_noc_coord.y);
        }
    }
    TT_ASSERT(chip0_eth_sender_l1_sem_addr != 0);
    TT_ASSERT(chip0_eth_sender_l1_base_addr != 0);

    //---------------------
    //                              Device 0 - RECEIVER
    log_debug(tt::LogTest, "\t-- receiver --");
    log_debug(tt::LogTest, "\tchip0_receiver_channels_offset: {}", chip0_receiver_channels_offset);
    chip0_edm_args.push_back(chip0_receiver_channels_offset);
    uint32_t chip0_receiver_num_channels = 0;
    // chip0_edm_args.push_back(chip0_receiver_num_channels);
    log_debug(tt::LogTest, "\tchip0_receiver_num_channels: {}", chip0_receiver_num_channels);
    //---------------------

    uint32_t num_pages_per_l1_buffer = num_bytes_per_send / input_buffer_page_size;
    TT_ASSERT(num_messages_to_send * num_pages_per_l1_buffer >= num_pages);

    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {
                uint32_t(1),  // enable sender side
                uint32_t(0),   // enable receiver side
                static_cast<uint32_t>(chip0_sender_num_channels),
                static_cast<uint32_t>(0)
            }});

    // chip 0 sender_worker_sender
    std::vector<uint32_t> chip0_sender_worker_sender_compile_args{
        pages_per_send,
        num_pages,
        input_buffer_page_size};
    std::vector<uint32_t> chip0_sender_worker_sender_runtime_args{
        chip0_sender_erisc_sender_buffer_address,
        // ERISC's semaphore address
        chip0_eth_sender_l1_sem_addr,
        // worker L1 semaphore address. Sender writes to this address to signal the worker
        // that the buffer is empty and available to write into
        chip0_worker_semaphore_id,
        uint32_t(sender_device->ethernet_core_from_logical_core(eth_sender_core).x),
        uint32_t(sender_device->ethernet_core_from_logical_core(eth_sender_core).y)
        };
    // TODO
    std::vector<uint32_t> chip0_sender_worker_reader_compile_args{
        input_is_dram,
        num_pages,
        input_buffer_page_size
        };
    // TODO
    std::vector<uint32_t> chip0_sender_worker_reader_runtime_args{
        dram_input_buf_base_addr};

    CBHandle cb_src0_sender_workers = CreateCircularBuffer(sender_program, chip0_sender_worker_core, cb_src0_config);
    auto device_0_edm_sender_worker_reader_kernel = tt_metal::CreateKernel(
        sender_program,
        sender_side_reader_worker_kernel_path,
        chip0_sender_worker_core,
        tt_metal::DataMovementConfig {
            .processor = DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = chip0_sender_worker_reader_compile_args});
    auto device_0_edm_sender_worker_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        sender_side_writer_worker_kernel_path,
        chip0_sender_worker_core,
        tt_metal::DataMovementConfig {
            .processor = DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::NOC_0,//  ::RISCV_1_default,
            .compile_args = chip0_sender_worker_sender_compile_args});

    tt_metal::SetRuntimeArgs(sender_program, eth_sender_kernel, eth_sender_core, chip0_edm_args);
    tt_metal::SetRuntimeArgs(
        sender_program,
        device_0_edm_sender_worker_sender_kernel,
        chip0_sender_worker_core,
        chip0_sender_worker_sender_runtime_args);
    tt_metal::SetRuntimeArgs(
        sender_program, device_0_edm_sender_worker_reader_kernel, chip0_sender_worker_core, chip0_sender_worker_reader_runtime_args);

    ////////////////////////////////////////////////////////////////////////////
    //                              Device 1
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program receiver_program = tt_metal::Program();
    auto chip1_receiver_worker_core = CoreCoord(0, 0);
    uint32_t chip1_worker_semaphore_id = tt::tt_metal::CreateSemaphore(receiver_program, chip1_receiver_worker_core, 0);

    uint32_t chip1_receiver_num_channels = chip0_arg_sender_num_channels;
    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/op_library/ccl/edm/erisc_datamover.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {
                static_cast<uint32_t>(0),  // enable sender side
                static_cast<uint32_t>(1),   // enable receiver side
                static_cast<uint32_t>(0),
                static_cast<uint32_t>(chip1_receiver_num_channels)
            }});

    //                              Device 1 - RECEIVER
    log_debug(tt::LogTest, "------------------ Device 1 args ----------------");
    uint32_t chip1_sender_num_channels = 0;
    uint32_t chip1_next_buffer_address = erisc_handshake_address + 16;
    std::vector<uint32_t> chip1_edm_args = {erisc_handshake_address};

    //                              Device 1 - SENDER
    uint32_t chip1_sender_channel_size = 0;
    uint32_t chip1_sender_num_messages_to_send = 0;
    uint32_t chip1_sender_channels_offset = chip1_receiver_num_channels;
    log_debug(tt::LogTest, "\t-- sender --");
    chip1_edm_args.push_back(chip1_sender_channels_offset);
    log_debug(tt::LogTest, "\tchip1_sender_channels_offset: {}", chip1_sender_channels_offset);
    // chip1_edm_args.push_back(chip1_sender_num_channels);
    log_debug(tt::LogTest, "\tchip1_sender_num_channels: {}", chip1_sender_num_channels);
    CoreCoord chip1_sender_noc_xy(0,0);
    for (uint32_t sc = 0; sc < chip1_sender_num_channels; sc++) {
        TT_ASSERT(chip1_sender_noc_xy.x != 0 && chip1_sender_noc_xy.y != 0);
        //    Informs how many times to iterate through the next group of args
        //    4) sender_buffer_address
        chip1_edm_args.push_back(chip1_next_buffer_address);
        log_debug(tt::LogTest, "\t\tsender_buffer_address: {}", chip1_next_buffer_address);
        //    5) sender_num_messages_to_send
        chip1_edm_args.push_back(chip1_sender_num_messages_to_send);
        log_debug(tt::LogTest, "\t\tchip1_sender_num_messages_to_send: {}", chip1_sender_num_messages_to_send);
        //    6) sender_channel_size
        chip1_edm_args.push_back(chip1_sender_channel_size);
        log_debug(tt::LogTest, "\t\tchip1_sender_channel_size: {}", chip1_sender_channel_size);
        chip1_next_buffer_address =
            chip1_next_buffer_address + chip1_sender_channel_size + 16 - (chip1_sender_channel_size % 16);
        //    7) sender_semaphores_base_address
        // ... Any off by one errors and I'm toast :)
        chip1_edm_args.push_back(chip1_next_buffer_address);
        log_debug(tt::LogTest, "\t\tsender_semaphores_base_address: {}", chip1_next_buffer_address);
        chip1_next_buffer_address += 16;
        //    8) worker_semaphore_address
        chip1_edm_args.push_back(chip1_worker_semaphore_id);
        log_debug(tt::LogTest, "\t\tworker_semaphores_base_id: {}", chip1_worker_semaphore_id);
        //    9) sender_num_workers
        //       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
        const uint32_t chip1_num_workers_on_channel = 1;
        chip1_edm_args.push_back(chip1_num_workers_on_channel);
        log_debug(tt::LogTest, "\t\tchip1_num_workers_on_channel: {}", chip1_num_workers_on_channel);
        for (uint32_t w = 0; w < chip1_num_workers_on_channel; w++) {
            //       10) worker_coord(s)
            auto worker_noc_coord = receiver_device->physical_core_from_logical_core(chip1_sender_noc_xy, CoreType::WORKER);
            chip1_edm_args.push_back(
                KernelXY{static_cast<uint16_t>(worker_noc_coord.x), static_cast<uint16_t>(worker_noc_coord.y)}
                    .to_uint32());
            log_debug(
                tt::LogTest,
                "\t\t\tchip1_num_workers_on_channel: x={},y={}",
                worker_noc_coord.x,
                worker_noc_coord.y);
        }
    }


    log_debug(tt::LogTest, "\t-- receiver --");
    uint32_t chip1_receiver_channels_offset = chip0_sender_channels_offset;
    uint32_t chip1_eth_receiver_l1_base_addr = 0;
    uint32_t chip1_eth_receiver_l1_sem_addr = 0;
    chip1_edm_args.push_back(chip1_receiver_channels_offset);
    log_debug(tt::LogTest, "\tchip1_receiver_channels_offset: {}", chip1_receiver_channels_offset);
    // chip1_edm_args.push_back(chip1_receiver_num_channels);
    log_debug(tt::LogTest, "\tchip1_receiver_num_channels: {}", chip1_receiver_num_channels);
    for (uint32_t sc = 0; sc < chip1_receiver_num_channels; sc++) {
        //    Informs how many times to iterate through the next group of args
        //    4) sender_buffer_address
        chip1_edm_args.push_back(chip1_next_buffer_address);
        chip1_eth_receiver_l1_base_addr = chip1_next_buffer_address;
        log_debug(tt::LogTest, "\t\tchip1_next_buffer_address: {}", chip1_next_buffer_address);
        //    5) sender_num_messages_to_send
        chip1_edm_args.push_back(num_messages_to_send);
        log_debug(tt::LogTest, "\t\tchip1_receiver_num_messages_to_send: {}", num_messages_to_send);
        //    6) sender_channel_size
        chip1_edm_args.push_back(eth_l1_staging_buffer_size);
        log_debug(tt::LogTest, "\t\tchip1_receiver_channel_size: {}", eth_l1_staging_buffer_size);
        chip1_next_buffer_address =
            chip1_next_buffer_address + eth_l1_staging_buffer_size + 16 - (eth_l1_staging_buffer_size % 16);
        //    7) sender_semaphores_base_address
        // ... Any off by one errors and I'm toast :)
        chip1_edm_args.push_back(chip1_next_buffer_address);
        log_debug(tt::LogTest, "\t\treceiver erisc semaphores_base_address: {}", chip1_next_buffer_address);
        chip1_eth_receiver_l1_sem_addr = chip1_next_buffer_address;
        chip1_next_buffer_address += 16;
        //    8) worker_semaphore_address
        chip1_edm_args.push_back(chip1_worker_semaphore_id);
        log_debug(tt::LogTest, "\t\tworker_semaphores_base_address: {}", chip1_worker_semaphore_id);
        //    9) sender_num_workers
        //       Informs how many worker X/Y coords to accept in the next loop. Each X/Y pair is 2 uint16s
        const uint32_t chip1_num_workers_on_channel = 1;
        chip1_edm_args.push_back(chip1_num_workers_on_channel);
        log_debug(tt::LogTest, "\t\tnum_workers: {}", chip1_num_workers_on_channel);
        for (uint32_t w = 0; w < chip1_num_workers_on_channel; w++) {
            //       10) worker_coord(s)
            auto worker_noc_coord = receiver_device->physical_core_from_logical_core(chip1_receiver_worker_core, CoreType::WORKER);
            chip1_edm_args.push_back(
                KernelXY{static_cast<uint16_t>(worker_noc_coord.x), static_cast<uint16_t>(worker_noc_coord.y)}
                    .to_uint32());
            log_debug(
                tt::LogTest, "\t\t\tchip1_receiver_noc_xy: x={},y={}", worker_noc_coord.x, worker_noc_coord.y);
        }
    }
    TT_ASSERT(chip1_eth_receiver_l1_base_addr != 0);
    TT_ASSERT(chip1_eth_receiver_l1_sem_addr != 0);


    tt_metal::SetRuntimeArgs(receiver_program, eth_receiver_kernel, eth_receiver_core, chip1_edm_args);
    std::vector<uint32_t> chip1_receiver_worker_sender_compile_args{
        dest_is_dram,  //
        num_pages,     //
        input_buffer_page_size};
    std::vector<uint32_t> chip1_receiver_worker_sender_runtime_args{dram_output_buffer_base_addr};
    std::vector<uint32_t> chip1_receiver_worker_receiver_compile_args{
        chip1_eth_receiver_l1_base_addr,  //
        chip1_eth_receiver_l1_sem_addr    //
    };
    std::vector<uint32_t> chip1_receiver_worker_receiver_runtime_args{
        num_pages_per_l1_buffer,
        num_pages,
        input_buffer_page_size,
        (uint32_t)receiver_device->ethernet_core_from_logical_core(eth_receiver_core).x,
        (uint32_t)receiver_device->ethernet_core_from_logical_core(eth_receiver_core).y,
        chip1_worker_semaphore_id};

    CBHandle cb_src0_receiver_workers = CreateCircularBuffer(receiver_program, chip1_receiver_worker_core, cb_src0_config);
    auto device_1_edm_receiver_worker_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        receiver_side_reader_worker_kernel_path,
        chip1_receiver_worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = chip1_receiver_worker_receiver_compile_args});
    auto device_1_edm_receiver_worker_sender_kernel = tt_metal::CreateKernel(
        receiver_program,
        receiver_side_writer_worker_kernel_path,
        chip1_receiver_worker_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = chip1_receiver_worker_sender_compile_args});
    tt_metal::SetRuntimeArgs(
        receiver_program,
        device_1_edm_receiver_worker_receiver_kernel,
        chip1_receiver_worker_core,
        chip1_receiver_worker_receiver_runtime_args);
    tt_metal::SetRuntimeArgs(
        receiver_program,
        device_1_edm_receiver_worker_sender_kernel,
        chip1_receiver_worker_core,
        chip1_receiver_worker_sender_runtime_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    try {
        tt::tt_metal::detail::CompileProgram(sender_device, sender_program);
        tt::tt_metal::detail::CompileProgram(receiver_device, receiver_program);
    } catch (std::exception& e) {
        std::cout << "Failed compile: " << e.what() << std::endl;
        throw e;
    }

    std::cout << "Running..." << std::endl;

    std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(receiver_device, receiver_program); });
    std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(sender_device, sender_program); });

    th2.join();
    th1.join();

    std::vector<uint32_t> readback_data_vec =
        std::vector<uint32_t>(all_zeros.size(), -1);  // init to 0 data for easier debug
    tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
    pass &= (readback_data_vec == inputs);
    TT_ASSERT(
        std::any_of(inputs.begin(), inputs.end(), [](uint32_t x) { return x != 0; }),
        "Input buffer expected to not be all 0");
    if (not pass) {
        std::cout << "Mismatch output mismatch" << std::endl;
        std::size_t num_printed_mismatches = 0;
        for (size_t i = 0; i < readback_data_vec.size() && num_printed_mismatches < 64; i++) {
            if (readback_data_vec[i] != inputs[i]) {
                std::cout << "[" << i << "]: expected " << inputs[i] << " got " << readback_data_vec[i] << std::endl;
                num_printed_mismatches++;
            }
        }
        std::cout << "... (remaining mismatches omitted)" << std::endl;
    }
    return pass;
}

int main(int argc, char** argv) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops
    assert(argc == 12);
    std::string const& sender_side_reader_worker_kernel_path = argv[1];
    std::string const& sender_side_writer_worker_kernel_path = argv[2];
    std::string const& receiver_side_reader_worker_kernel_path = argv[3];
    std::string const& receiver_side_writer_worker_kernel_path = argv[4];
    const uint32_t eth_l1_staging_buffer_size = std::stoi(argv[5]);
    const uint32_t eth_max_concurrent_sends = std::stoi(argv[6]);
    const uint32_t input_buffer_page_size = std::stoi(argv[7]);
    const uint32_t input_buffer_size_bytes = std::stoi(argv[8]);
    const bool source_is_dram = std::stoi(argv[9]) == 1;
    const bool dest_is_dram = std::stoi(argv[10]) == 1;
    const uint32_t precomputed_source_addresses_buffer_size = std::stoi(argv[9]);

    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices != 2) {
        std::cout << "Need at least 2 devices to run this test" << std::endl;
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        std::cout << "Test must be run on WH" << std::endl;
        return 0;
    }

    N300TestDevice test_fixture;


    const auto& device_0 = test_fixture.devices_.at(0);
    const auto& device_1 = test_fixture.devices_.at(1);
    const size_t precomputed_source_addresses_buffer_address = (size_t) nullptr;
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 32;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE + 32;

    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
    assert(active_eth_cores.size() > 0);
    auto eth_sender_core_iter = active_eth_cores.begin();
    assert(eth_sender_core_iter != active_eth_cores.end());
    const auto& eth_sender_core = *eth_sender_core_iter;
    auto [device_id, eth_receiver_core] = device_0->get_connected_ethernet_core(eth_sender_core);

    bool success = false;
    try {
        success = RunWriteBWTest(
            sender_side_reader_worker_kernel_path,
            sender_side_writer_worker_kernel_path,
            receiver_side_reader_worker_kernel_path,
            receiver_side_writer_worker_kernel_path,
            device_0,
            device_1,
            eth_sender_core,
            eth_receiver_core,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            precomputed_source_addresses_buffer_address,
            precomputed_source_addresses_buffer_size,
            eth_l1_staging_buffer_size,
            eth_max_concurrent_sends,
            input_buffer_page_size,
            input_buffer_size_bytes,
            source_is_dram,
            dest_is_dram);
    } catch (std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
        test_fixture.TearDown();
        return -1;
    }

    test_fixture.TearDown();

    return success ? 0 : -1;
}


// EnablePersistentKernelCache
