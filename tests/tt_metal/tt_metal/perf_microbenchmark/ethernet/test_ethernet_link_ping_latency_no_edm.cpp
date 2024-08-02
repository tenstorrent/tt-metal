
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <tuple>
#include <map>

#include "device/tt_arch_types.h"
#include "impl/device/device.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "tt_backend_api_types.hpp"
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

#include "tt_metal/detail/persistent_kernel_cache.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;


class N300TestDevice {
   public:
    N300TestDevice() : device_open(false) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

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

struct ChipSenderReceiverEthCore {
    CoreCoord sender_core;
    CoreCoord receiver_core;
};

std::tuple<Program,Program> build(
    Device *device0,
    Device *device1,
    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t num_channels,
    KernelHandle &local_kernel,
    KernelHandle &remote_kernel
) {
    Program program0;
    Program program1;

    std::vector<uint32_t> const& ct_args = {num_channels};

    // Kernel Setup

    auto rt_args = [&]() -> std::vector<uint32_t> {
        return std::vector<uint32_t> {
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
            static_cast<uint32_t>(num_samples),
            static_cast<uint32_t>(sample_page_size)};
    };

    local_kernel = tt_metal::CreateKernel(
        program0,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/ethernet_ping_latency_ubench_sender.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig {
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program0, local_kernel, eth_sender_core, rt_args());

    remote_kernel = tt_metal::CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/ethernet_ping_latency_ubench_receiver.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig {
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program1, remote_kernel, eth_receiver_core, rt_args());

    // // Launch
    try {
        tt::tt_metal::detail::CompileProgram(device0, program0);
        tt::tt_metal::detail::CompileProgram(device1, program1);
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Failed compile: {}", e.what());
        throw e;
    }

    return std::tuple<Program,Program>{std::move(program0),std::move(program1)};
}

void run(
    Device *device0,
    Device *device1,
    Program &program0,
    Program &program1,
    KernelHandle local_kernel,
    KernelHandle remote_kernel,

    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t max_channels_per_direction
) {
    auto rt_args = [&]() -> std::vector<uint32_t> {
        return std::vector<uint32_t> {
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
            static_cast<uint32_t>(num_samples),
            static_cast<uint32_t>(sample_page_size)};
    };
    log_trace(tt::LogTest, "Running...");

    tt_metal::SetRuntimeArgs(program0, local_kernel, eth_sender_core, rt_args());
    tt_metal::SetRuntimeArgs(program1, remote_kernel, eth_receiver_core, rt_args());

    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(device0, program0); });
        std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(device1, program1); });

        th2.join();
        th1.join();
    } else {

        tt_metal::EnqueueProgram(device0->command_queue(), program0, false);
        tt_metal::EnqueueProgram(device1->command_queue(), program1, false);

        std::cout << "Calling Finish" << std::endl;
        tt_metal::Finish(device0->command_queue());
        tt_metal::Finish(device1->command_queue());
    }
    tt::tt_metal::detail::DumpDeviceProfileResults(device0);
    tt::tt_metal::detail::DumpDeviceProfileResults(device1);
}


int main(int argc, char** argv) {
    // argv[0]: program
    // argv[1]: num_samples
    // argv[2]: sample_page_size
    // argv[3]: max_channels_per_direction
    assert(argc >= 4);
    std::size_t arg_idx = 1;
    std::size_t num_sample_counts = std::stoi(argv[arg_idx++]);
    log_trace(tt::LogTest, "num_sample_counts: {}", std::stoi(argv[arg_idx]));
    std::vector<std::size_t> sample_counts;
    for (std::size_t i = 0; i < num_sample_counts; i++) {
        sample_counts.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "sample_counts[{}]: {}", i, sample_counts.back());
    }

    std::size_t num_sample_sizes = std::stoi(argv[arg_idx++]);
    std::vector<std::size_t> sample_sizes;
    log_trace(tt::LogTest, "num_sample_sizes: {}", num_sample_sizes);
    for (std::size_t i = 0; i < num_sample_sizes; i++) {
        sample_sizes.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "sample_sizes[{}]: {}", i, sample_sizes.back());
    }

    std::size_t num_channel_counts = std::stoi(argv[arg_idx++]);
    std::vector<std::size_t> channel_counts;
    log_trace(tt::LogTest, "num_channel_counts: {}", num_channel_counts);
    for (std::size_t i = 0; i < num_channel_counts; i++) {
        channel_counts.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "channel_counts[{}]: {}", i, channel_counts.back());
    }

    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        log_info(tt::LogTest, "Need at least 2 devices to run this test");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return 0;
    }

    std::cout << "setting up test fixture" << std::endl;
    N300TestDevice test_fixture;
    std::cout << "done setting up test fixture" << std::endl;

    const auto& device_0 = test_fixture.devices_.at(0);
    std::cout << "1" << std::endl;
    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
    std::cout << "2" << std::endl;
    auto eth_sender_core_iter = active_eth_cores.begin();
    auto eth_sender_core_iter_end = active_eth_cores.end();
    chip_id_t device_id = std::numeric_limits<chip_id_t>::max();
    std::cout << "3" << std::endl;
    tt_xy_pair eth_receiver_core;
    bool initialized = false;
    tt_xy_pair eth_sender_core;
    std::cout << "4" << std::endl;
    do {
        TT_ASSERT(eth_sender_core_iter != eth_sender_core_iter_end);
        std::cout << "4a" << std::endl;
        std::tie(device_id, eth_receiver_core) = device_0->get_connected_ethernet_core(*eth_sender_core_iter);
        std::cout << "4b" << std::endl;
        eth_sender_core = *eth_sender_core_iter;
        eth_sender_core_iter++;
    } while (device_id != 1);
    std::cout << "5" << std::endl;
    TT_ASSERT(device_id == 1);
    std::cout << "6" << std::endl;
    const auto& device_1 = test_fixture.devices_.at(device_id);
    std::cout << "7" << std::endl;
    // Add more configurations here until proper argc parsing added
    bool success = false;
    success = true;
    std::cout << "STARTING" << std::endl;
    try {
        for (auto num_samples : sample_counts) {
            for (auto sample_page_size : sample_sizes) {
                for (auto max_channels_per_direction : channel_counts) {
                    log_info(tt::LogTest, "num_samples: {}, sample_page_size: {}, num_channels_per_direction: {}", num_samples, sample_page_size, max_channels_per_direction);
                    KernelHandle local_kernel;
                    KernelHandle remote_kernel;
                    try {
                        auto [program0,program1] = build(
                            device_0,
                            device_1,
                            eth_sender_core,
                            eth_receiver_core,
                            num_samples,
                            sample_page_size,
                            max_channels_per_direction,
                            local_kernel,
                            remote_kernel
                        );
                        run(
                            device_0,
                            device_1,
                            program0,
                            program1,
                            local_kernel,
                            remote_kernel,

                            eth_sender_core,
                            eth_receiver_core,
                            num_samples,
                            sample_page_size,
                            max_channels_per_direction
                        );
                    } catch (std::exception& e) {
                        log_error(tt::LogTest, "Caught exception: {}", e.what());
                        test_fixture.TearDown();
                        return -1;
                    }
                }
            }
        }
    } catch (std::exception e) {
        test_fixture.TearDown();
        return -1;
    }


    return success ? 0 : -1;
}
