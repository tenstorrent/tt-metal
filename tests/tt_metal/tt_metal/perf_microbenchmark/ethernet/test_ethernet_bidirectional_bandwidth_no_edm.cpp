
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <fmt/base.h>
#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "df/float32.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "umd/device/types/arch.h"
#include "umd/device/types/xy_pair.h"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

class N300TestDevice {
public:
    N300TestDevice() : device_open(false) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 2 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<chip_id_t> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);
            devices_ = tt::tt_metal::detail::CreateDevices({0, 1, 2, 3, 4, 5, 6, 7});

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

    std::map<chip_id_t, tt_metal::IDevice*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

private:
    bool device_open;
};

struct ChipSenderReceiverEthCore {
    CoreCoord sender_core;
    CoreCoord receiver_core;
};

std::tuple<tt_metal::Program, tt_metal::Program> build(
    tt_metal::IDevice* device0,
    tt_metal::IDevice* device1,
    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t max_channels_per_direction,
    tt_metal::KernelHandle& local_kernel,
    tt_metal::KernelHandle& remote_kernel) {
    tt_metal::Program program0;
    tt_metal::Program program1;

    std::vector<uint32_t> const& ct_args = {};
    constexpr std::size_t num_links = 0;

    // Kernel Setup

    auto rt_args = [&](bool send_channels_at_offset_0) -> std::vector<uint32_t> {
        return std::vector<uint32_t>{
            static_cast<uint32_t>(tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED)),
            static_cast<uint32_t>(num_samples),
            static_cast<uint32_t>(sample_page_size),
            static_cast<uint32_t>(max_channels_per_direction),
            static_cast<uint32_t>(send_channels_at_offset_0)};
    };

    local_kernel = tt_metal::CreateKernel(
        program0,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/ethernet_bidirectional_ubench.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});

    remote_kernel = tt_metal::CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/ethernet_bidirectional_ubench.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});

    // Launch
    try {
        tt::tt_metal::detail::CompileProgram(device0, program0);
        tt::tt_metal::detail::CompileProgram(device1, program1);
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Failed compile: {}", e.what());
        throw e;
    }

    return std::tuple<tt_metal::Program, tt_metal::Program>{std::move(program0), std::move(program1)};
}

void run(
    tt_metal::IDevice* device0,
    tt_metal::IDevice* device1,
    tt_metal::Program& program0,
    tt_metal::Program& program1,
    tt_metal::KernelHandle local_kernel,
    tt_metal::KernelHandle remote_kernel,

    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t max_channels_per_direction) {
    auto rt_args = [&](bool send_channels_at_offset_0) -> std::vector<uint32_t> {
        return std::vector<uint32_t>{
            static_cast<uint32_t>(tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
                tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED)),
            static_cast<uint32_t>(num_samples),
            static_cast<uint32_t>(sample_page_size),
            static_cast<uint32_t>(max_channels_per_direction),
            static_cast<uint32_t>(send_channels_at_offset_0)};
    };
    log_trace(tt::LogTest, "Running...");

    tt_metal::SetRuntimeArgs(program0, local_kernel, eth_sender_core, rt_args(true));
    tt_metal::SetRuntimeArgs(program1, remote_kernel, eth_receiver_core, rt_args(false));

    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(device0, program0); });
        std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(device1, program1); });

        th2.join();
        th1.join();
    } else {
        tt_metal::EnqueueProgram(device0->command_queue(), program0, false);
        tt_metal::EnqueueProgram(device1->command_queue(), program1, false);

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
    TT_ASSERT(num_sample_counts > 0);
    log_trace(tt::LogTest, "num_sample_counts: {}", std::stoi(argv[arg_idx]));
    std::vector<std::size_t> sample_counts;
    for (std::size_t i = 0; i < num_sample_counts; i++) {
        sample_counts.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "sample_counts[{}]: {}", i, sample_counts.back());
    }

    std::size_t num_sample_sizes = std::stoi(argv[arg_idx++]);
    std::vector<std::size_t> sample_sizes;
    TT_ASSERT(num_sample_sizes > 0);
    log_trace(tt::LogTest, "num_sample_sizes: {}", num_sample_sizes);
    for (std::size_t i = 0; i < num_sample_sizes; i++) {
        sample_sizes.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "sample_sizes[{}]: {}", i, sample_sizes.back());
    }

    std::size_t num_channel_counts = std::stoi(argv[arg_idx++]);
    std::vector<std::size_t> channel_counts;
    TT_ASSERT(num_channel_counts > 0);
    log_trace(tt::LogTest, "num_channel_counts: {}", num_channel_counts);
    for (std::size_t i = 0; i < num_channel_counts; i++) {
        channel_counts.push_back(std::stoi(argv[arg_idx++]));
        log_trace(tt::LogTest, "channel_counts[{}]: {}", i, channel_counts.back());
    }

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        log_trace(tt::LogTest, "Need at least 2 devices to run this test");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_trace(tt::LogTest, "Test must be run on WH");
        return 0;
    }

    N300TestDevice test_fixture;

    const auto& device_0 = test_fixture.devices_.at(2);
    auto const& active_eth_cores = device_0->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    TT_ASSERT(eth_sender_core_iter != active_eth_cores.end());
    auto eth_sender_core = *eth_sender_core_iter;

    auto [device_id, eth_receiver_core] = device_0->get_connected_ethernet_core(eth_sender_core);
    const auto& device_1 = test_fixture.devices_.at(device_id);
    bool success = false;
    success = true;
    try {
        for (auto num_samples : sample_counts) {
            for (auto sample_page_size : sample_sizes) {
                for (auto max_channels_per_direction : channel_counts) {
                    log_trace(
                        tt::LogTest,
                        "num_samples: {}, sample_page_size: {}, num_channels_per_direction: {}",
                        num_samples,
                        sample_page_size,
                        max_channels_per_direction);
                    tt_metal::KernelHandle local_kernel;
                    tt_metal::KernelHandle remote_kernel;
                    try {
                        auto [program0, program1] = build(
                            device_0,
                            device_1,
                            eth_sender_core,
                            eth_receiver_core,
                            num_samples,
                            sample_page_size,
                            max_channels_per_direction,
                            local_kernel,
                            remote_kernel);
                        run(device_0,
                            device_1,
                            program0,
                            program1,
                            local_kernel,
                            remote_kernel,

                            eth_sender_core,
                            eth_receiver_core,
                            num_samples,
                            sample_page_size,
                            max_channels_per_direction);
                    } catch (std::exception& e) {
                        log_error(tt::LogTest, "Caught exception: {}", e.what());
                        test_fixture.TearDown();
                        return -1;
                    }
                }
            }
        }
    } catch (std::exception& e) {
        test_fixture.TearDown();
        return -1;
    }

    return success ? 0 : -1;
}
