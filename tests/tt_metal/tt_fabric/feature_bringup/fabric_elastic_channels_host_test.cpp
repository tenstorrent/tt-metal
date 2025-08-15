// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <fmt/base.h>
#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>

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
#include <chrono>
#include <cstring>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "umd/device/types/arch.h"
#include "umd/device/types/xy_pair.h"

#include <tt-metalium/kernel_types.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

using namespace tt;
using namespace tt::test_utils;

// Timing stats structure matching kernel side
struct TimingStats {
    uint64_t total_acquire_cycles = 0;
    uint64_t total_release_cycles = 0;
    uint64_t total_test_cycles = 0;
    uint32_t total_misc_cycles = 0;
    uint32_t acquire_count = 0;
    uint32_t release_count = 0;
    uint32_t misc_count = 0;
};

class N300TestDevice {
public:
    N300TestDevice() : device_open(false) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 2 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<chip_id_t> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);
            devices_ = tt::tt_metal::detail::CreateDevices(ids);

        } else {
            TT_THROW("This suite can only be run on N300 Wormhole devices");
        }
        device_open = true;
    }
    ~N300TestDevice() {
        if (device_open) {
            log_info(tt::LogAlways, "Tearing down devices");
            TearDown();
        }
        log_info(tt::LogAlways, "Tearing down devices complete");
    }

    void TearDown() {
        device_open = false;
        tt::tt_metal::detail::CloseDevices(devices_);
    }

    std::map<chip_id_t, tt_metal::IDevice*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

private:
    bool device_open;
};

struct TestConfig {
    uint32_t n_chunks;
    uint32_t chunk_n_pkts;
    uint32_t rx_chunk_n_pkts;
    uint32_t packet_size;
    bool bidirectional_mode;
    uint32_t message_size;
    uint32_t total_messages;
    uint32_t n_workers;
    uint32_t fabric_mcast_factor;
};


struct DeviceTestResources {
    tt_metal::IDevice* device;
    CoreRangeSet worker_cores;
    std::vector<CoreCoord> worker_cores_vec;
    CoreCoord eth_core;
    tt_metal::Program program;
    uint32_t worker_ack_semaphore_id = std::numeric_limits<uint32_t>::max();
    uint32_t worker_new_chunk_semaphore_id = std::numeric_limits<uint32_t>::max();
    uint32_t worker_src_buffer_address = std::numeric_limits<uint32_t>::max();
};


struct TestResources {
    DeviceTestResources local_device;
    DeviceTestResources remote_device;
};

void create_worker_kernels(
    TestResources& test_resources,
    std::vector<tt_metal::KernelHandle>& local_sender_worker_kernels,
    std::vector<tt_metal::KernelHandle>& remote_sender_worker_kernels,
    const TestConfig& config) {
    std::vector<uint32_t> ct_args = {
        config.n_chunks,
        config.chunk_n_pkts,
    };

    auto local_sender_worker_kernel = tt_metal::CreateKernel(
        test_resources.local_device.program,
        "tests/tt_metal/tt_fabric/feature_bringup/kernels/fabric_elastic_channel_sender_worker.cpp",
        test_resources.local_device.worker_cores,
        tt::tt_metal::WriterDataMovementConfig(ct_args));

    for (size_t i = 0; i < config.n_workers; i++) {
        local_sender_worker_kernels.push_back(local_sender_worker_kernel);
    }
    log_info(tt::LogAlways, "Local sender worker kernel: {}", local_sender_worker_kernel);

    auto remote_sender_worker_kernel = tt_metal::CreateKernel(
        test_resources.remote_device.program,
        "tests/tt_metal/tt_fabric/feature_bringup/kernels/fabric_elastic_channel_sender_worker.cpp",
        test_resources.remote_device.worker_cores,
        tt::tt_metal::WriterDataMovementConfig(ct_args));

    for (size_t i = 0; i < config.n_workers; i++) {
        remote_sender_worker_kernels.push_back(remote_sender_worker_kernel);
    }
    log_info(tt::LogAlways, "Remote sender worker kernel: {}", remote_sender_worker_kernel);
}

void build(
    TestResources& test_resources,
    const TestConfig& config,
    tt_metal::KernelHandle& local_erisc_kernel,
    tt_metal::KernelHandle& remote_erisc_kernel,
    std::vector<tt_metal::KernelHandle>& local_sender_worker_kernels,
    std::vector<tt_metal::KernelHandle>& remote_sender_worker_kernels) {
    // Compile-time arguments - match kernel expectations:
    // get_compile_time_arg_val(0) = N_CHUNKS
    // get_compile_time_arg_val(1) = RX_N_PKTS
    // get_compile_time_arg_val(2) = CHUNK_N_PKTS
    // get_compile_time_arg_val(3) = PACKET_SIZE
    // get_compile_time_arg_val(4) = BIDIRECTIONAL_MODE
    // get_compile_time_arg_val(5) = N_SRC_CHANS
    uint32_t rx_n_pkts = config.rx_chunk_n_pkts;
    std::vector<uint32_t> erisc_kernel_compile_time_args = {
        config.n_chunks,            // N_CHUNKS
        rx_n_pkts,                  // RX_N_PKTS
        config.chunk_n_pkts,        // CHUNK_N_PKTS
        config.packet_size,         // PACKET_SIZE
        config.bidirectional_mode,  // BIDIRECTIONAL_MODE
        config.n_workers            // N_SRC_CHANS
    };

    auto erisc_kernel_name = config.n_workers > 0
                                 ? "tests/tt_metal/tt_fabric/feature_bringup/kernels/"
                                   "fabric_elastic_channels_erisc_forward_worker_traffic.cpp"
                                 : "tests/tt_metal/tt_fabric/feature_bringup/fabric_elastic_channels_test.cpp";
    log_info(tt::LogAlways, "Erisc kernel name: {}", erisc_kernel_name);
    local_erisc_kernel = tt_metal::CreateKernel(
        test_resources.local_device.program,
        erisc_kernel_name,
        test_resources.local_device.eth_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_1, .compile_args = erisc_kernel_compile_time_args});

    remote_erisc_kernel = tt_metal::CreateKernel(
        test_resources.remote_device.program,
        erisc_kernel_name,
        test_resources.remote_device.eth_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::NOC_1, .compile_args = erisc_kernel_compile_time_args});

    log_info(tt::LogAlways, "Local erisc kernel: {}", local_erisc_kernel);
    log_info(tt::LogAlways, "Remote erisc kernel: {}", remote_erisc_kernel);

    if (config.n_workers > 0) {
        create_worker_kernels(test_resources, local_sender_worker_kernels, remote_sender_worker_kernels, config);
    }

    // Compile programs
    try {
        tt::tt_metal::detail::CompileProgram(test_resources.local_device.device, test_resources.local_device.program);
        tt::tt_metal::detail::CompileProgram(test_resources.remote_device.device, test_resources.remote_device.program);
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Failed compile: {}", e.what());
        throw e;
    }
}

void set_worker_runtime_args(
    DeviceTestResources& device_resources,
    std::vector<tt_metal::KernelHandle>& worker_kernels,
    const TestConfig& config) {
    auto eth_core_virtual = device_resources.device->ethernet_core_from_logical_core(device_resources.eth_core);

    for (size_t i = 0; i < device_resources.worker_cores_vec.size(); i++) {
        auto worker_core = device_resources.worker_cores_vec[i];

        TT_FATAL(device_resources.worker_ack_semaphore_id != std::numeric_limits<uint32_t>::max(), "worker_ack_semaphore_id is not set");
        TT_FATAL(device_resources.worker_new_chunk_semaphore_id != std::numeric_limits<uint32_t>::max(), "worker_new_chunk_semaphore_id is not set");
        TT_FATAL(device_resources.worker_src_buffer_address != std::numeric_limits<uint32_t>::max(), "worker_src_buffer_address is not set");

        std::vector<uint32_t> rt_args = {
            config.total_messages,
            device_resources.worker_src_buffer_address,
            eth_core_virtual.x,
            eth_core_virtual.y,
            config.message_size,
            device_resources.worker_new_chunk_semaphore_id,
            device_resources.worker_ack_semaphore_id,
            i
        };

        tt_metal::SetRuntimeArgs(device_resources.program, worker_kernels.at(i), worker_core, rt_args);
    }
}


void set_worker_runtime_args(
    TestResources& test_resources,
    std::vector<tt_metal::KernelHandle>& local_sender_worker_kernels,
    std::vector<tt_metal::KernelHandle>& remote_sender_worker_kernels,
    const TestConfig& config) {

    set_worker_runtime_args(test_resources.local_device, local_sender_worker_kernels, config);
    set_worker_runtime_args(test_resources.remote_device, remote_sender_worker_kernels, config);
}

TimingStats read_timing_stats(tt_metal::IDevice* device, CoreCoord core, uint32_t handshake_addr) {
    TimingStats stats;

    // Read timing stats from L1 memory - use same address as calculated in host
    uint32_t timing_stats_addr = handshake_addr + 0x800;

    std::vector<uint32_t> timing_data;
    timing_data.resize(sizeof(TimingStats) / sizeof(uint32_t));
    tt_metal::detail::ReadFromDeviceL1(
        device, core, timing_stats_addr, sizeof(TimingStats), timing_data, CoreType::ETH);

    // Copy data to stats structure
    std::memcpy(&stats, timing_data.data(), sizeof(TimingStats));

    return stats;
}

void run_test(
    TestResources& test_resources,
    tt_metal::KernelHandle local_erisc_kernel,
    tt_metal::KernelHandle remote_erisc_kernel,
    std::vector<tt_metal::KernelHandle>& local_sender_worker_kernels,
    std::vector<tt_metal::KernelHandle>& remote_sender_worker_kernels,
    const TestConfig& config) {
    auto rt_args = [&](bool send_channels_at_offset_0, DeviceTestResources& device_resources) -> std::vector<uint32_t> {
        uint32_t base_addr = static_cast<uint32_t>(tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED));

        // Calculate buffer sizes and addresses
        uint32_t send_buffer_size = config.n_chunks * config.chunk_n_pkts * config.packet_size;
        uint32_t send_buffer_base = base_addr + 0x1000;  // Start after handshake area

        // Align receive buffer start to 16B boundary after send buffers
        uint32_t recv_buffer_base = tt::align(send_buffer_base + send_buffer_size, 16);

        // Calculate timing stats address in unreserved space before send buffers
        uint32_t timing_stats_addr = base_addr + 0x800;  // Place in unreserved space before send buffers

        std::vector<uint32_t> rt_args = {
            base_addr,  // handshake_addr
            config.total_messages,
            config.message_size,
            static_cast<uint32_t>(send_channels_at_offset_0),
            send_buffer_base,
            recv_buffer_base,
            timing_stats_addr,
            device_resources.worker_src_buffer_address,
            config.fabric_mcast_factor};

        for (size_t i = 0; i < config.n_workers; i++) {
            auto translated =
                device_resources.device->worker_core_from_logical_core(device_resources.worker_cores_vec[i]);
            rt_args.push_back(translated.x);
        }
        for (size_t i = 0; i < config.n_workers; i++) {
            auto translated =
                device_resources.device->worker_core_from_logical_core(device_resources.worker_cores_vec[i]);
            rt_args.push_back(translated.y);
        }
        for (size_t i = 0; i < config.n_workers; i++) {
            rt_args.push_back(device_resources.worker_ack_semaphore_id);
        }
        for (size_t i = 0; i < config.n_workers; i++) {
            rt_args.push_back(device_resources.worker_new_chunk_semaphore_id);
        }
        for (size_t i = 0; i < config.n_workers; i++) {
            rt_args.push_back(device_resources.worker_src_buffer_address);
        }

        return rt_args;
    };

    log_info(tt::LogAlways, "Running Fabric Elastic Channels Test...");
    log_info(
        tt::LogTest,
        "Config: n_chunks={}, chunk_n_pkts={}, packet_size={}, bidirectional={}, message_size={}, total_messages={}, "
        "num_workers={}",
        config.n_chunks,
        config.chunk_n_pkts,
        config.packet_size,
        config.bidirectional_mode,
        config.message_size,
        config.total_messages,
        config.n_workers);

    tt_metal::SetRuntimeArgs(test_resources.local_device.program, local_erisc_kernel, test_resources.local_device.eth_core, rt_args(true, test_resources.local_device));
    tt_metal::SetRuntimeArgs(test_resources.remote_device.program, remote_erisc_kernel, test_resources.remote_device.eth_core, rt_args(false, test_resources.remote_device));

    if (config.n_workers > 0) {
        log_info(tt::LogAlways, "Setting worker runtime args");
        set_worker_runtime_args(test_resources, local_sender_worker_kernels, remote_sender_worker_kernels, config);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    log_info(tt::LogAlways, "Launching programs");
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(test_resources.local_device.device, test_resources.local_device.program); });
        std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(test_resources.remote_device.device, test_resources.remote_device.program); });

        th2.join();
        th1.join();
    } else {
        tt_metal::EnqueueProgram(test_resources.local_device.device->command_queue(), test_resources.local_device.program, false);
        tt_metal::EnqueueProgram(test_resources.remote_device.device->command_queue(), test_resources.remote_device.program, false);

        tt_metal::Finish(test_resources.local_device.device->command_queue());
        tt_metal::Finish(test_resources.remote_device.device->command_queue());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Read timing statistics from both devices
    uint32_t handshake_addr = static_cast<uint32_t>(tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED));
    TimingStats stats0 = read_timing_stats(test_resources.local_device.device, test_resources.local_device.eth_core, handshake_addr);
    TimingStats stats1 = read_timing_stats(test_resources.remote_device.device, test_resources.remote_device.eth_core, handshake_addr);

    // Report results
    log_info(tt::LogAlways, "Test completed in {} microseconds", duration.count());

    log_info(tt::LogAlways, "Device 0 (Sender) Timing Stats:");
    log_info(tt::LogAlways, "  Acquire operations: {}", stats0.acquire_count);
    log_info(tt::LogAlways, "  Total acquire cycles: {}", stats0.total_acquire_cycles);
    log_info(
        tt::LogTest,
        "  Average acquire cycles: {:.2f}",
        stats0.acquire_count > 0 ? (double)stats0.total_acquire_cycles / stats0.acquire_count : 0.0);
    log_info(tt::LogAlways, "  Release operations: {}", stats0.release_count);
    log_info(tt::LogAlways, "  Total release cycles: {}", stats0.total_release_cycles);
    log_info(
        tt::LogTest,
        "  Average release cycles: {:.2f}",
        stats0.release_count > 0 ? (double)stats0.total_release_cycles / stats0.release_count : 0.0);
    log_info(tt::LogAlways, "  Misc operations: {}", stats0.misc_count);
    log_info(tt::LogAlways, "  Total misc cycles: {}", stats0.total_misc_cycles);
    log_info(
        tt::LogTest,
        "  Average misc cycles: {:.2f}",
        stats0.misc_count > 0 ? (double)stats0.total_misc_cycles / stats0.misc_count : 0.0);

    if (config.bidirectional_mode) {
        log_info(tt::LogAlways, "Device 1 (Receiver) Timing Stats:");
        log_info(tt::LogAlways, "  Acquire operations: {}", stats1.acquire_count);
        log_info(tt::LogAlways, "  Total acquire cycles: {}", stats1.total_acquire_cycles);
        log_info(
            tt::LogTest,
            "  Average acquire cycles: {:.2f}",
            stats1.acquire_count > 0 ? (double)stats1.total_acquire_cycles / stats1.acquire_count : 0.0);
        log_info(tt::LogAlways, "  Release operations: {}", stats1.release_count);
        log_info(tt::LogAlways, "  Total release cycles: {}", stats1.total_release_cycles);
        log_info(
            tt::LogTest,
            "  Average release cycles: {:.2f}",
            stats1.release_count > 0 ? (double)stats1.total_release_cycles / stats1.release_count : 0.0);
        log_info(tt::LogAlways, "  Misc operations: {}", stats1.misc_count);
        log_info(tt::LogAlways, "  Total misc cycles: {}", stats1.total_misc_cycles);
        log_info(
            tt::LogTest,
            "  Average misc cycles: {:.2f}",
            stats1.misc_count > 0 ? (double)stats1.total_misc_cycles / stats1.misc_count : 0.0);
    }

    // Calculate throughput metrics
    uint32_t total_messages_sent = config.total_messages * config.n_workers;
    auto total_cycles = stats0.total_test_cycles;
    auto test_seconds = (double)total_cycles / 1000000000;
    double throughput_msgs_per_sec = (double)total_messages_sent / test_seconds;
    double throughput_bytes_per_sec = throughput_msgs_per_sec * config.message_size;
    double throughput_GB_s = throughput_bytes_per_sec / 1e9;
    log_info(tt::LogAlways, "Total_cycles: {}", total_cycles);

    log_info(tt::LogAlways, "Performance Metrics:");
    log_info(tt::LogAlways, "  Total messages sent: {}", total_messages_sent);
    log_info(tt::LogAlways, "  Throughput: {:.2f} messages/second", throughput_msgs_per_sec);
    log_info(tt::LogAlways, "  Throughput: {:.2f} GB/s", throughput_GB_s);
}

TestResources create_test_resources(
    tt_metal::IDevice* device_0,
    tt_metal::IDevice* device_1,
    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    const TestConfig& config) {
    TestResources resources;
    resources.local_device.device = device_0;
    resources.remote_device.device = device_1;
    resources.local_device.eth_core = eth_sender_core;
    resources.remote_device.eth_core = eth_receiver_core;

    for (auto &device_resource_reference : {std::ref(resources.local_device), std::ref(resources.remote_device)}) {
        auto& device_resource = device_resource_reference.get();
        auto& worker_cores = device_resource.worker_cores;
        auto& worker_cores_vec = device_resource.worker_cores_vec;
        worker_cores_vec.reserve(config.n_workers);
        worker_cores = CoreRange(CoreCoord(0, 0), CoreCoord(0, config.n_workers - 1));
        worker_cores_vec = corerange_to_cores(worker_cores);

        device_resource.worker_ack_semaphore_id = tt_metal::CreateSemaphore(device_resource.program, device_resource.worker_cores, 0, CoreType::WORKER);
        device_resource.worker_new_chunk_semaphore_id = tt_metal::CreateSemaphore(device_resource.program, device_resource.worker_cores, 0, CoreType::WORKER);

        auto worker_src_buffer = tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
            .device = device_resource.device,
            .size = config.n_workers * config.message_size,
            .page_size = config.n_workers * config.message_size,
            .buffer_type = tt::tt_metal::BufferType::L1
        });
        auto erisc_write_out_dest_buffer = tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
            .device = device_resource.device,
            .size = config.n_workers * config.message_size,
            .page_size = config.n_workers * config.message_size,
            .buffer_type = tt::tt_metal::BufferType::L1
        });
        device_resource.worker_src_buffer_address = worker_src_buffer->address();
    }

    return resources;
}

void validate_test_config(const TestConfig& config) {
    if (config.message_size > config.packet_size) {
        log_error(
            tt::LogTest,
            "Message size ({}) cannot be larger than packet size ({})",
            config.message_size,
            config.packet_size);
        exit(-1);
    }

    if (config.n_workers > config.n_chunks) {
        log_error(
            tt::LogTest,
            "Number of workers ({}) cannot be greater than number of chunks ({})",
            config.n_workers,
            config.n_chunks);
        exit(-1);
    }

    if (config.chunk_n_pkts == 0 || config.chunk_n_pkts == 0 || config.n_chunks == 0 || config.packet_size == 0 ||
        config.message_size == 0 || config.total_messages == 0 || config.n_workers == 0) {
        log_error(
            tt::LogTest,
            "Invalid test config. Found a zero value. The following are all expected to be non-zero: chunk_n_pkts={}, "
            "packet_size={}, message_size={}, total_messages={}, n_workers={}",
            config.chunk_n_pkts,
            config.packet_size,
            config.message_size,
            config.total_messages,
            config.n_workers);
        exit(-1);
    }

    if (config.fabric_mcast_factor > 10) {
        log_error(tt::LogTest, "Fabric mcast factor ({}) cannot be greater than 10", config.fabric_mcast_factor);
        exit(-1);
    }
}

int main(int argc, char** argv) {
    // Command line argument parsing
    // Usage: fabric_elastic_channels_host_test <n_chunks> <chunk_n_pkts> <packet_size> <bidirectional> <message_size>
    // <total_messages>
    if (argc != 10) {
        log_error(
            tt::LogTest,
            "Usage: {} <n_chunks> <chunk_n_pkts> <rx_chunk_n_pkts> <packet_size> <bidirectional> <message_size> "
            "<total_messages> <n_workers> <fabric_mcast_factor>",
            argv[0]);
        return -1;
    }

    TestConfig config;
    size_t arg_idx = 1;
    config.n_chunks = std::stoi(argv[arg_idx++]);
    config.chunk_n_pkts = std::stoi(argv[arg_idx++]);
    config.rx_chunk_n_pkts = std::stoi(argv[arg_idx++]);
    config.packet_size = std::stoi(argv[arg_idx++]);
    config.bidirectional_mode = std::stoi(argv[arg_idx++]) != 0;
    config.message_size = std::stoi(argv[arg_idx++]);
    config.total_messages = std::stoi(argv[arg_idx++]);
    config.n_workers = std::stoi(argv[arg_idx++]);
    config.fabric_mcast_factor = std::stoi(argv[arg_idx++]);

    if (config.packet_size % 16 != 0) {
        log_warning(tt::LogTest, "Packet size is not aligned to 16 bytes. Aligning to 16 bytes.");
        config.packet_size = tt::align(config.packet_size, 16);
    }

    validate_test_config(config);

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        log_error(tt::LogTest, "Need at least 2 devices to run this test");
        return -1;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_error(tt::LogTest, "Test must be run on WH");
        return -1;
    }

    N300TestDevice test_fixture;

    const auto& device_0 = test_fixture.devices_.at(0);
    const auto& active_eth_cores = device_0->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    TT_FATAL(eth_sender_core_iter != active_eth_cores.end(), "No active ethernet cores found");
    while (eth_sender_core_iter != active_eth_cores.end() and
           not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(device_0->id(), *eth_sender_core_iter)) {
        eth_sender_core_iter++;
    }
    TT_FATAL(eth_sender_core_iter != active_eth_cores.end(), "No active ethernet cores found");
    auto eth_sender_core = *eth_sender_core_iter;
    TT_FATAL(device_0->is_active_ethernet_core(eth_sender_core), "Not an active ethernet core {}", eth_sender_core);

    auto [device_id, eth_receiver_core] = device_0->get_connected_ethernet_core(eth_sender_core);
    const auto& device_1 = test_fixture.devices_.at(device_id);

    bool success = false;
    try {
        log_info(tt::LogAlways, "Building programs...");
        tt_metal::KernelHandle local_kernel;
        tt_metal::KernelHandle remote_kernel;

        std::vector<tt_metal::KernelHandle> local_sender_worker_kernels;
        std::vector<tt_metal::KernelHandle> remote_sender_worker_kernels;

        auto test_resources = create_test_resources(device_0, device_1, eth_sender_core, eth_receiver_core, config);

        build(
            test_resources,
            config,
            local_kernel,
            remote_kernel,
            local_sender_worker_kernels,
            remote_sender_worker_kernels);

        run_test(
            test_resources,
            local_kernel,
            remote_kernel,
            local_sender_worker_kernels,
            remote_sender_worker_kernels,
            config);

        success = true;
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Test failed with exception: {}", e.what());
        test_fixture.TearDown();
        return -1;
    }

    log_info(tt::LogAlways, "Test completed successfully");
    return success ? 0 : -1;
}
