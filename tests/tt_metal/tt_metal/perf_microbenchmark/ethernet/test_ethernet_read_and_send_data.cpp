
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>

#include "tt_metal/common/core_coord.h"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/program/program_pool.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

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
    tt_metal::BufferType input_buffer_type;// = tt_metal::BufferType::L1;
    tt_metal::BufferType output_buffer_type;// = tt_metal::BufferType::L1;
    tt::DataFormat l1_data_format;// = tt::DataFormat::Float16_b;
};

bool RunWriteBWTest(
    std::string const& sender_kernel_path,
    std::string const& receiver_kernel_path,
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,

    const size_t eth_channel_sync_ack_addr,
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

    TT_ASSERT(precomputed_source_addresses_buffer_address < std::numeric_limits<uint32_t>::max(), "precomputed_source_addresses_buffer_address is too large");

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
    std::cout << "num_messages_to_send: " << num_messages_to_send << std::endl;
    // Generate inputs
    ////////////////////////////////////////////////////////////////////////////
    //   SETUP THE INPUT CB
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "Generating vector" << std::endl;
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, input_buffer_size_bytes / sizeof(uint32_t));

    // Clear expected value at ethernet L1 address

    BankedConfig test_config =
        BankedConfig{
            .num_pages = num_pages,
            .size_bytes = input_buffer_size_bytes,
            .page_size_bytes = input_buffer_page_size,
            .input_buffer_type = source_is_dram ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1,
            .output_buffer_type = dest_is_dram ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1,
            .l1_data_format = tt::DataFormat::Float16_b};
    auto input_buffer =
            CreateBuffer(
                InterleavedBufferConfig{sender_device, test_config.size_bytes, test_config.page_size_bytes, test_config.input_buffer_type});

    bool input_is_dram = test_config.input_buffer_type == tt_metal::BufferType::DRAM;
    tt_metal::detail::WriteToBuffer(input_buffer, inputs);
    const uint32_t dram_input_buf_base_addr = input_buffer->address();

    ////////////////////////////////////////////////////////////////////////////
    //   EMPTY INITIALIZE THE OUTPUT CB
    ////////////////////////////////////////////////////////////////////////////

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    auto output_buffer =
            CreateBuffer(
                InterleavedBufferConfig{receiver_device, test_config.size_bytes, test_config.page_size_bytes, test_config.output_buffer_type});

    bool output_is_dram = test_config.output_buffer_type == tt_metal::BufferType::DRAM;
    tt_metal::detail::WriteToBuffer(output_buffer, all_zeros);
    const uint32_t dram_output_buffer_base_addr = output_buffer->address();



    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    auto sender_program = tt_metal::CreateScopedProgram();

    uint32_t num_pages_per_l1_buffer = num_bytes_per_send / input_buffer_page_size;
    TT_ASSERT(num_messages_to_send * num_pages_per_l1_buffer >= num_pages);
    std::cout << "eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE: " << eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE << std::endl;
    std::cout << "src_eth_l1_byte_address: " << src_eth_l1_byte_address << std::endl;
    auto eth_sender_kernel = tt_metal::CreateKernel(
        sender_program,
        sender_kernel_path,
        eth_sender_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {
                uint32_t(num_bytes_per_send),         // 0
                uint32_t(num_bytes_per_send >> 4),    // 1
                uint32_t(num_messages_to_send),       // 2
                uint32_t(eth_max_concurrent_sends),   // 3
                uint32_t(source_is_dram)              // 4
                }
            // .compile_args = {uint32_t(256), uint32_t(256 >> 4)}
            });

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            uint32_t(src_eth_l1_byte_address),
            uint32_t(dst_eth_l1_byte_address),
            uint32_t(dram_input_buf_base_addr),
            uint32_t(input_buffer_page_size),
            uint32_t(num_pages),
            uint32_t(precomputed_source_addresses_buffer_address),
            uint32_t(precomputed_source_addresses_buffer_size)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                           Receiver Device
    ////////////////////////////////////////////////////////////////////////////
    auto receiver_program = tt_metal::CreateScopedProgram();

    auto eth_receiver_kernel = tt_metal::CreateKernel(
        receiver_program,
        receiver_kernel_path,
        eth_receiver_core,
        tt_metal::EthernetConfig{
            .noc = tt_metal::NOC::NOC_0,
            .compile_args = {
                uint32_t(num_bytes_per_send),         // 0
                uint32_t(num_bytes_per_send >> 4),    // 1
                uint32_t(num_messages_to_send),       // 2
                uint32_t(eth_max_concurrent_sends),   // 3
                uint32_t(dest_is_dram)                // 4
            }});  // probably want to use NOC_1 here
            // .compile_args = {uint32_t(256), uint32_t(256 >> 4)}});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            uint32_t(eth_channel_sync_ack_addr),
            uint32_t(dst_eth_l1_byte_address),
            uint32_t(src_eth_l1_byte_address),
            dram_output_buffer_base_addr,
            input_buffer_page_size,
            num_pages
        });

    std::cout << "dram_output_buffer_base_addr: " << dram_output_buffer_base_addr << std::endl;
    std::cout << "dram_input_buf_base_addr: " << dram_input_buf_base_addr << std::endl;
    std::cout << "input_buffer_page_size: " << input_buffer_page_size << std::endl;
    std::cout << "num_pages: " << num_pages << std::endl;


    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto* sender_program_ptr = tt::tt_metal::ProgramPool::instance().get_program(sender_program);
    auto* receiver_program_ptr = tt::tt_metal::ProgramPool::instance().get_program(receiver_program);
    try {
        tt::tt_metal::detail::CompileProgram(sender_device, *sender_program_ptr);
        tt::tt_metal::detail::CompileProgram(receiver_device, *receiver_program_ptr);
    } catch (std::exception& e) {
        std::cout << "Failed compile: " << e.what() << std::endl;
        throw e;
    }

    std::cout << "Running..." << std::endl;

    std::thread th2 = std::thread([&] {
        tt_metal::detail::LaunchProgram(receiver_device, *receiver_program_ptr);
    });
    std::thread th1 = std::thread([&] {
        tt_metal::detail::LaunchProgram(sender_device, *sender_program_ptr);
    });

    th2.join();
    std::cout << "receiver done" << std::endl;
    th1.join();
    std::cout << "sender done" << std::endl;

    std::vector<uint32_t> readback_data_vec = std::vector<uint32_t>(all_zeros.size(), -1); // init to 0 data for easier debug
    tt_metal::detail::ReadFromBuffer(output_buffer, readback_data_vec);
    pass &= (readback_data_vec == inputs);
    TT_ASSERT(std::any_of(inputs.begin(), inputs.end(), [](uint32_t x) { return x != 0; }), "Input buffer expected to not be all 0");
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
    assert (argc == 10);
    std::string const& sender_kernel_path = argv[1];
    std::string const& receiver_kernel_path = argv[2];
    const uint32_t eth_l1_staging_buffer_size = std::stoi(argv[3]);
    const uint32_t eth_max_concurrent_sends = std::stoi(argv[4]);
    const uint32_t input_buffer_page_size = std::stoi(argv[5]);
    const uint32_t input_buffer_size_bytes = std::stoi(argv[6]);
    const bool source_is_dram = std::stoi(argv[7]) == 1;
    const bool dest_is_dram = std::stoi(argv[8]) == 1;
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

    std::cout << "precomputed_source_addresses_buffer_size: " << precomputed_source_addresses_buffer_size << std::endl;
    const auto& device_0 = test_fixture.devices_.at(0);
    const auto& device_1 = test_fixture.devices_.at(1);
    const size_t precomputed_source_addresses_buffer_address = (size_t)nullptr;
    const size_t eth_channel_sync_ack_addr = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t src_eth_l1_byte_address = eth_channel_sync_ack_addr + 16;
    const size_t dst_eth_l1_byte_address = eth_channel_sync_ack_addr + 16;

    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
    assert (active_eth_cores.size() > 0);
    auto eth_sender_core_iter = active_eth_cores.begin();
    assert (eth_sender_core_iter != active_eth_cores.end());
    // eth_sender_core_iter++;
    // assert (eth_sender_core_iter != active_eth_cores.end());
    const auto& eth_sender_core = *eth_sender_core_iter;
    auto [device_id, eth_receiver_core] = device_0->get_connected_ethernet_core(eth_sender_core);

    // std::cout << "SENDER CORE: (x=" << eth_sender_core.x << ", y=" << eth_sender_core.y << ")" << std::endl;
    // std::cout << "RECEIVER CORE: (x=" << eth_receiver_core.x << ", y=" << eth_receiver_core.y << ")" << std::endl;

    // std::cout << "BW TEST: " << 64 << ", num_messages_to_send: " << num_messages_to_send << std::endl;
    bool success = false;
    try {
        success = RunWriteBWTest(
            sender_kernel_path,
            receiver_kernel_path,
            device_0,
            device_1,
            eth_sender_core,
            eth_receiver_core,
            eth_channel_sync_ack_addr,
            src_eth_l1_byte_address,
            dst_eth_l1_byte_address,
            precomputed_source_addresses_buffer_address,
            precomputed_source_addresses_buffer_size,
            eth_l1_staging_buffer_size,
            eth_max_concurrent_sends,
            input_buffer_page_size,
            input_buffer_size_bytes,
            source_is_dram,
            dest_is_dram
            );
    } catch (std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
        test_fixture.TearDown();
        return -1;
    }


    test_fixture.TearDown();

    return success ? 0 : -1;

}
