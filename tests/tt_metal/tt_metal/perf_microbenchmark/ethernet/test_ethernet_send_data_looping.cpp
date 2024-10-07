
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "impl/device/device.hpp"
#include "llrt/llrt.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/program/program_pool.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

using namespace tt;
using namespace tt::test_utils;

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

bool RunWriteBWTest(
    std::string const& sender_kernel_path,
    std::string const& receiver_kernel_path,
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,

    bool source_is_dram,
    bool dest_is_dram,

    const uint32_t size_in_bytes,
    const size_t src_eth_l1_byte_address,
    const size_t dst_eth_l1_byte_address,
    const CoreCoord& eth_sender_core,
    const CoreCoord& eth_receiver_core,
    const uint32_t num_messages_to_send,
    const uint32_t max_num_transaction_buffers,
    uint32_t num_bytes_per_send = 16

    ) {
    bool pass = true;
    log_debug(
        tt::LogTest,
        "Sending {} bytes from device {} eth core {} addr {} to device {} eth core {} addr {}",
        size_in_bytes,
        sender_device->id(),
        eth_sender_core.str(),
        src_eth_l1_byte_address,
        receiver_device->id(),
        eth_receiver_core.str(),
        dst_eth_l1_byte_address);

    // Initialize L1s:
    std::vector<uint32_t> zeros = std::vector<uint32_t>(32, 0);
    llrt::write_hex_vec_to_core(
        sender_device->id(),
        sender_device->ethernet_core_from_logical_core(eth_sender_core),
        zeros,
        src_eth_l1_byte_address);
    llrt::write_hex_vec_to_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        zeros,
        dst_eth_l1_byte_address);

    // Generate inputs
    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, size_in_bytes / sizeof(uint32_t));
    llrt::write_hex_vec_to_core(
        sender_device->id(),
        sender_device->ethernet_core_from_logical_core(eth_sender_core),
        inputs,
        src_eth_l1_byte_address);

    // Clear expected value at ethernet L1 address
    std::vector<uint32_t> all_zeros(inputs.size(), 0);
    llrt::write_hex_vec_to_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        all_zeros,
        dst_eth_l1_byte_address);

    ////////////////////////////////////////////////////////////////////////////
    //                      Sender Device
    ////////////////////////////////////////////////////////////////////////////
    auto sender_program = tt_metal::CreateScopedProgram();

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
                uint32_t(max_num_transaction_buffers),// 3
                uint32_t(source_is_dram)              // 4
                }
            });

    tt_metal::SetRuntimeArgs(
        sender_program,
        eth_sender_kernel,
        eth_sender_core,
        {
            uint32_t(src_eth_l1_byte_address),
            uint32_t(dst_eth_l1_byte_address)
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
                uint32_t(max_num_transaction_buffers),// 3
                uint32_t(dest_is_dram)                // 4
            }});  // probably want to use NOC_1 here

    tt_metal::SetRuntimeArgs(
        receiver_program,
        eth_receiver_kernel,
        eth_receiver_core,
        {
            uint32_t(src_eth_l1_byte_address),
            uint32_t(dst_eth_l1_byte_address)
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto* sender_program_ptr = tt::tt_metal::ProgramPool::instance().get_program(sender_program);
    auto* receiver_program_ptr = tt::tt_metal::ProgramPool::instance().get_program(receiver_program);
    tt::tt_metal::detail::CompileProgram(sender_device, *sender_program_ptr);
    tt::tt_metal::detail::CompileProgram(receiver_device, *receiver_program_ptr);

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

    auto readback_vec = llrt::read_hex_vec_from_core(
        receiver_device->id(),
        receiver_device->ethernet_core_from_logical_core(eth_receiver_core),
        dst_eth_l1_byte_address,
        size_in_bytes);
    pass &= (readback_vec == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << eth_receiver_core.str() << std::endl;
        std::cout << readback_vec[0] << std::endl;
    }
    return pass;
}


int main(int argc, char** argv) {
    // argv[0]: program
    // argv[1]: buffer_size_bytes
    // argv[2]: num_loops
    assert (argc == 6);
    const uint32_t buffer_size_bytes = std::stoi(argv[1]);
    const uint32_t num_messages_to_send = std::stoi(argv[2]);
    const uint32_t max_num_transaction_buffers = std::stoi(argv[3]);
    std::string const& sender_kernel_path = argv[4];
    std::string const& receiver_kernel_path = argv[5];

    bool source_is_dram = true;
    bool dest_is_dram = true;

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
    const size_t src_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
    const size_t dst_eth_l1_byte_address = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;

    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
    assert (active_eth_cores.size() > 0);
    auto eth_sender_core_iter = active_eth_cores.begin();
    assert (eth_sender_core_iter != active_eth_cores.end());
    eth_sender_core_iter++;
    assert (eth_sender_core_iter != active_eth_cores.end());
    const auto& eth_sender_core = *eth_sender_core_iter;
    auto [device_id, eth_receiver_core] = device_0->get_connected_ethernet_core(eth_sender_core);

    // std::cout << "SENDER CORE: (x=" << eth_sender_core.x << ", y=" << eth_sender_core.y << ")" << std::endl;
    // std::cout << "RECEIVER CORE: (x=" << eth_receiver_core.x << ", y=" << eth_receiver_core.y << ")" << std::endl;

    // std::cout << "BW TEST: " << 64 << ", num_messages_to_send: " << num_messages_to_send << std::endl;
    RunWriteBWTest(
        sender_kernel_path,
        receiver_kernel_path,
        device_0,
        device_1,
        source_is_dram,
        dest_is_dram,
        buffer_size_bytes,
        src_eth_l1_byte_address,
        dst_eth_l1_byte_address,
        eth_sender_core,
        eth_receiver_core,
        num_messages_to_send,
        max_num_transaction_buffers,
        buffer_size_bytes);

    test_fixture.TearDown();

}
