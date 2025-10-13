
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
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "df/float32.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

class N300TestDevice {
public:
    N300TestDevice() : num_devices_(tt::tt_metal::GetNumAvailableDevices()), device_open(false) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 2 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<chip_id_t> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);
            auto reserved_devices = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
                ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1);
            for (const auto& [id, device] : reserved_devices) {
                devices_[id] = device;
            }
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

    std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
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
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device0,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& device1,
    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t num_channels,
    tt_metal::KernelHandle& local_kernel,
    tt_metal::KernelHandle& remote_kernel) {
    tt_metal::Program program0;
    tt_metal::Program program1;

    std::vector<uint32_t> const& ct_args = {num_channels};

    // Kernel Setup
    uint32_t erisc_unreserved_base = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    auto rt_args = [&]() -> std::vector<uint32_t> {
        return std::vector<uint32_t>{
            erisc_unreserved_base, static_cast<uint32_t>(num_samples), static_cast<uint32_t>(sample_page_size)};
    };

    local_kernel = tt_metal::CreateKernel(
        program0,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/ethernet_ping_latency_ubench_sender.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program0, local_kernel, eth_sender_core, rt_args());

    remote_kernel = tt_metal::CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/ethernet_ping_latency_ubench_receiver.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = ct_args});
    tt_metal::SetRuntimeArgs(program1, remote_kernel, eth_receiver_core, rt_args());

    return std::tuple<tt_metal::Program, tt_metal::Program>{std::move(program0), std::move(program1)};
}

void run(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device0,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device1,
    tt_metal::Program& program0,
    tt_metal::Program& program1,
    tt_metal::KernelHandle local_kernel,
    tt_metal::KernelHandle remote_kernel,

    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t max_channels_per_direction) {
    uint32_t erisc_unreserved_base = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
    auto rt_args = [&]() -> std::vector<uint32_t> {
        return std::vector<uint32_t>{
            erisc_unreserved_base, static_cast<uint32_t>(num_samples), static_cast<uint32_t>(sample_page_size)};
    };
    log_trace(tt::LogTest, "Running...");

    tt_metal::SetRuntimeArgs(program0, local_kernel, eth_sender_core, rt_args());
    tt_metal::SetRuntimeArgs(program1, remote_kernel, eth_receiver_core, rt_args());

    tt::tt_metal::distributed::MeshCoordinate zero_coord0 =
        tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(device0->shape().dims());
    tt::tt_metal::distributed::MeshCoordinateRange device_range0 =
        tt::tt_metal::distributed::MeshCoordinateRange(zero_coord0, zero_coord0);

    tt::tt_metal::distributed::MeshCoordinate zero_coord1 =
        tt::tt_metal::distributed::MeshCoordinate::zero_coordinate(device1->shape().dims());
    tt::tt_metal::distributed::MeshCoordinateRange device_range1 =
        tt::tt_metal::distributed::MeshCoordinateRange(zero_coord1, zero_coord1);

    tt::tt_metal::distributed::MeshWorkload mesh_workload0;
    mesh_workload0.add_program(device_range0, std::move(program0));

    tt::tt_metal::distributed::MeshWorkload mesh_workload1;
    mesh_workload1.add_program(device_range1, std::move(program1));

    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        // For slow dispatch mode, use threads with mesh workloads
        std::thread th2 = std::thread([&] {
            tt::tt_metal::distributed::EnqueueMeshWorkload(device0->mesh_command_queue(), mesh_workload0, true);
        });
        std::thread th1 = std::thread([&] {
            tt::tt_metal::distributed::EnqueueMeshWorkload(device1->mesh_command_queue(), mesh_workload1, true);
        });

        th2.join();
        th1.join();
    } else {
        tt::tt_metal::distributed::EnqueueMeshWorkload(device0->mesh_command_queue(), mesh_workload0, false);
        tt::tt_metal::distributed::EnqueueMeshWorkload(device1->mesh_command_queue(), mesh_workload1, false);

        std::cout << "Calling Finish" << std::endl;
        tt::tt_metal::distributed::Finish(device0->mesh_command_queue());
        tt::tt_metal::distributed::Finish(device1->mesh_command_queue());
    }
    tt::tt_metal::ReadMeshDeviceProfilerResults(*device0);
    tt::tt_metal::ReadMeshDeviceProfilerResults(*device1);
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

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        log_info(tt::LogTest, "Need at least 2 devices to run this test");
        return 0;
    }
    if (arch != tt::ARCH::WORMHOLE_B0) {
        log_info(tt::LogTest, "Test must be run on WH");
        return 0;
    }

    std::cout << "setting up test fixture" << std::endl;
    N300TestDevice test_fixture;
    std::cout << "done setting up test fixture" << std::endl;

    const auto& device_0 = test_fixture.devices_.at(0);
    const auto& active_eth_cores = device_0->get_devices()[0]->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    auto eth_sender_core_iter_end = active_eth_cores.end();
    chip_id_t device_id = std::numeric_limits<chip_id_t>::max();
    tt_xy_pair eth_receiver_core;
    tt_xy_pair eth_sender_core;
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    do {
        TT_FATAL(eth_sender_core_iter != eth_sender_core_iter_end, "No active ethernet core found for device 0");
        if (cluster.is_ethernet_link_up(device_0->get_devices()[0]->id(), *eth_sender_core_iter)) {
            std::tie(device_id, eth_receiver_core) =
                device_0->get_devices()[0]->get_connected_ethernet_core(*eth_sender_core_iter);
            eth_sender_core = *eth_sender_core_iter;
        }
        eth_sender_core_iter++;
    } while (device_id == std::numeric_limits<chip_id_t>::max() ||
             !test_fixture.devices_.at(device_id)->get_devices()[0]->is_mmio_capable());
    TT_FATAL(device_id != std::numeric_limits<chip_id_t>::max(), "No valid receiver device found to connect to");
    auto* receiver_device = test_fixture.devices_.at(device_id)->get_devices()[0];
    TT_FATAL(receiver_device->is_mmio_capable(), "Receiver device is not mmio capable");
    const auto& device_1 = test_fixture.devices_.at(device_id);

    // Add more configurations here until proper argc parsing added
    bool success = false;
    success = true;
    std::cout << "STARTING" << std::endl;
    try {
        for (auto num_samples : sample_counts) {
            for (auto sample_page_size : sample_sizes) {
                for (auto max_channels_per_direction : channel_counts) {
                    log_info(
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
