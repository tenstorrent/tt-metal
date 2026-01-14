// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cstdint>
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
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <chrono>
#include <cstring>
#include <iostream>

#include <tt_stl/assert.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/tt_align.hpp>
#include "common/tt_backend_api_types.hpp"
#include <llrt/tt_cluster.hpp>

#include <array>
#include <bit>

using namespace tt;
using namespace tt::test_utils;

// Timing stats structure matching kernel side
struct TimingStats {
    uint64_t total_acquire_cycles = 0;
    uint64_t total_release_cycles = 0;
    uint64_t total_test_cycles = 0;
    uint64_t total_misc_cycles = 0;
    uint32_t acquire_count = 0;
    uint32_t release_count = 0;
    uint32_t misc_count = 0;
};

// Worker timing stats structure matching worker kernel side
struct WorkerTimingStats {
    uint64_t total_idle_cycles = 0;
    uint32_t idle_count = 0;
};

class N300TestDevice {
public:
    N300TestDevice() : num_devices_(tt::tt_metal::GetNumAvailableDevices()) {
        tt_fabric::SetFabricConfig(tt_fabric::FabricConfig::DISABLED);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 2 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<ChipId> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);
            devices_ = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(ids);
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
        for (auto& [id, device] : devices_) {
            device->close();
        }
        devices_.clear();
    }

    std::map<ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

private:
    bool device_open{false};
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

/**
 * Parse a simple JSON string into TestConfig
 * Expected format: {"n_chunks": 1, "chunk_n_pkts": 2, "rx_chunk_n_pkts": 10, "packet_size": 4352, "bidirectional_mode":
 * true, "message_size": 16, "total_messages": 10000, "n_workers": 1, "fabric_mcast_factor": 1}
 */
TestConfig parse_json_config(const std::string& json_str) {
    TestConfig config = {};

    // Simple JSON parser - look for key-value pairs
    auto find_value = [&json_str](const std::string& key) -> std::string {
        std::string search_key = "\"" + key + "\"";
        auto pos = json_str.find(search_key);
        if (pos == std::string::npos) {
            throw std::runtime_error("Key '" + key + "' not found in JSON");
        }

        // Find the colon after the key
        pos = json_str.find(':', pos);
        if (pos == std::string::npos) {
            throw std::runtime_error("Malformed JSON - no colon after key '" + key + "'");
        }

        // Skip whitespace after colon
        pos++;
        while (pos < json_str.length() && std::isspace(json_str[pos])) {
            pos++;
        }

        // Extract the value until comma or closing brace
        size_t end_pos = pos;
        while (end_pos < json_str.length() && json_str[end_pos] != ',' && json_str[end_pos] != '}') {
            end_pos++;
        }

        std::string value = json_str.substr(pos, end_pos - pos);

        // Trim whitespace
        auto trim_start = value.find_first_not_of(" \t\n\r");
        if (trim_start != std::string::npos) {
            value = value.substr(trim_start);
        }
        auto trim_end = value.find_last_not_of(" \t\n\r");
        if (trim_end != std::string::npos) {
            value = value.substr(0, trim_end + 1);
        }

        return value;
    };

    try {
        config.n_chunks = std::stoul(find_value("n_chunks"));
        config.chunk_n_pkts = std::stoul(find_value("chunk_n_pkts"));
        config.rx_chunk_n_pkts = std::stoul(find_value("rx_chunk_n_pkts"));
        config.packet_size = std::stoul(find_value("packet_size"));

        std::string bidir_str = find_value("bidirectional_mode");
        config.bidirectional_mode = (bidir_str == "true" || bidir_str == "1");

        config.message_size = std::stoul(find_value("message_size"));
        config.total_messages = std::stoul(find_value("total_messages"));
        config.n_workers = std::stoul(find_value("n_workers"));
        config.fabric_mcast_factor = std::stoul(find_value("fabric_mcast_factor"));

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse JSON config: " + std::string(e.what()));
    }

    return config;
}

/**
 * Parse TestConfig from command line arguments (traditional mode)
 */
TestConfig parse_cli_config(int argc, char** argv) {
    if (argc != 10) {
        log_error(
            tt::LogTest,
            "Usage: {} <n_chunks> <chunk_n_pkts> <rx_chunk_n_pkts> <packet_size> <bidirectional> <message_size> "
            "<total_messages> <n_workers> <fabric_mcast_factor>",
            argv[0]);
        throw std::runtime_error("Invalid number of command line arguments");
    }

    TestConfig config{};
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

    return config;
}

struct DeviceTestResources {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device = nullptr;
    CoreRangeSet worker_cores;
    std::vector<CoreCoord> worker_cores_vec;
    CoreCoord eth_core;
    tt_metal::Program program;
    uint32_t worker_ack_semaphore_id = std::numeric_limits<uint32_t>::max();
    uint32_t worker_new_chunk_semaphore_id = std::numeric_limits<uint32_t>::max();
    uint32_t worker_src_buffer_address = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> worker_timing_stats_addresses;
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
    std::vector<uint32_t> ct_args = {config.n_chunks, config.chunk_n_pkts};

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
    uint32_t rx_n_pkts = config.rx_chunk_n_pkts;
    std::vector<uint32_t> erisc_kernel_compile_time_args = {
        config.n_chunks,            // N_CHUNKS
        rx_n_pkts,                  // RX_N_PKTS
        config.chunk_n_pkts,        // CHUNK_N_PKTS
        config.packet_size,         // PACKET_SIZE
        config.bidirectional_mode,  // BIDIRECTIONAL_MODE
        config.n_workers,           // N_SRC_CHANS
        config.fabric_mcast_factor  // FABRIC_MCAST_FACTOR
    };

    const auto* erisc_kernel_name =
        "tests/tt_metal/tt_fabric/feature_bringup/kernels/fabric_elastic_channels_erisc_forward_worker_traffic.cpp";
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
}

void set_worker_runtime_args(
    DeviceTestResources& device_resources,
    std::vector<tt_metal::KernelHandle>& worker_kernels,
    const TestConfig& config) {
    auto eth_core_virtual = device_resources.device->ethernet_core_from_logical_core(device_resources.eth_core);

    for (size_t i = 0; i < device_resources.worker_cores_vec.size(); i++) {
        auto worker_core = device_resources.worker_cores_vec[i];

        TT_FATAL(
            device_resources.worker_ack_semaphore_id != std::numeric_limits<uint32_t>::max(),
            "worker_ack_semaphore_id is not set");
        TT_FATAL(
            device_resources.worker_new_chunk_semaphore_id != std::numeric_limits<uint32_t>::max(),
            "worker_new_chunk_semaphore_id is not set");
        TT_FATAL(
            device_resources.worker_src_buffer_address != std::numeric_limits<uint32_t>::max(),
            "worker_src_buffer_address is not set");
        TT_FATAL(
            i < device_resources.worker_timing_stats_addresses.size(),
            "worker_timing_stats_addresses not allocated for worker {}",
            i);

        uint32_t worker_timing_stats_addr = device_resources.worker_timing_stats_addresses[i];

        std::vector<uint32_t> rt_args = {
            config.total_messages,
            device_resources.worker_src_buffer_address,
            eth_core_virtual.x,
            eth_core_virtual.y,
            config.message_size,
            device_resources.worker_new_chunk_semaphore_id,
            device_resources.worker_ack_semaphore_id,
            i,
            worker_timing_stats_addr};

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
    TimingStats stats{};

    // Read timing stats from L1 memory - use same address as calculated in host
    uint32_t timing_stats_addr = handshake_addr + 0x800;

    std::vector<uint32_t> timing_data{};
    timing_data.resize(sizeof(TimingStats) / sizeof(uint32_t));
    tt_metal::detail::ReadFromDeviceL1(
        device, core, timing_stats_addr, sizeof(TimingStats), timing_data, CoreType::ETH);

    constexpr size_t num_words = sizeof(TimingStats) / sizeof(uint32_t);
    std::array<uint32_t, num_words> arr{};
    std::memcpy(arr.data(), timing_data.data(), sizeof(arr));
    stats = std::bit_cast<TimingStats>(arr);

    return stats;
}

WorkerTimingStats read_worker_timing_stats(tt_metal::IDevice* device, CoreCoord core, uint32_t timing_stats_addr) {
    WorkerTimingStats stats{};

    std::vector<uint32_t> timing_data{};
    timing_data.resize(sizeof(WorkerTimingStats) / sizeof(uint32_t));
    tt_metal::detail::ReadFromDeviceL1(
        device, core, timing_stats_addr, sizeof(WorkerTimingStats), timing_data, CoreType::WORKER);

    constexpr size_t num_words = sizeof(WorkerTimingStats) / sizeof(uint32_t);
    std::array<uint32_t, num_words> arr{};
    std::memcpy(arr.data(), timing_data.data(), sizeof(arr));
    stats = std::bit_cast<WorkerTimingStats>(arr);

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

        // eth transfers must be 16B aligned
        uint32_t recv_buffer_base = tt::align(send_buffer_base + send_buffer_size, 16);

        uint32_t timing_stats_addr =
            base_addr + 0x800;  // allocated definitely more than needed for timing stats... it's a test, relax

        auto handshake_addr = base_addr;
        std::vector<uint32_t> rt_args = {
            handshake_addr,
            config.total_messages,
            config.message_size,
            static_cast<uint32_t>(send_channels_at_offset_0),
            send_buffer_base,
            recv_buffer_base,
            timing_stats_addr,
            device_resources.worker_src_buffer_address};

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

    tt_metal::SetRuntimeArgs(
        test_resources.local_device.program,
        local_erisc_kernel,
        test_resources.local_device.eth_core,
        rt_args(true, test_resources.local_device));
    tt_metal::SetRuntimeArgs(
        test_resources.remote_device.program,
        remote_erisc_kernel,
        test_resources.remote_device.eth_core,
        rt_args(false, test_resources.remote_device));

    if (config.n_workers > 0) {
        log_info(tt::LogAlways, "Setting worker runtime args");
        set_worker_runtime_args(test_resources, local_sender_worker_kernels, remote_sender_worker_kernels, config);
    }

    log_info(tt::LogAlways, "Launching programs");

    tt_metal::distributed::MeshWorkload local_workload;
    tt_metal::distributed::MeshWorkload remote_workload;
    local_workload.add_program(
        tt_metal::distributed::MeshCoordinateRange({0, 0}, {0, 0}), std::move(test_resources.local_device.program));
    remote_workload.add_program(
        tt_metal::distributed::MeshCoordinateRange({0, 0}, {0, 0}), std::move(test_resources.remote_device.program));

    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread th2 = std::thread([&] {
            tt_metal::distributed::EnqueueMeshWorkload(
                test_resources.local_device.device->mesh_command_queue(), local_workload, true);
        });
        std::thread th1 = std::thread([&] {
            tt_metal::distributed::EnqueueMeshWorkload(
                test_resources.remote_device.device->mesh_command_queue(), remote_workload, true);
        });

        th2.join();
        th1.join();
    } else {
        tt_metal::distributed::EnqueueMeshWorkload(
            test_resources.local_device.device->mesh_command_queue(), local_workload, false);
        tt_metal::distributed::EnqueueMeshWorkload(
            test_resources.remote_device.device->mesh_command_queue(), remote_workload, false);

        tt_metal::distributed::Finish(test_resources.local_device.device->mesh_command_queue());
        tt_metal::distributed::Finish(test_resources.remote_device.device->mesh_command_queue());
    }

    //
    // Timing Stats Readback
    //
    uint32_t handshake_addr = static_cast<uint32_t>(tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED));

    //
    // Results Reporting
    //
    TimingStats stats0 = read_timing_stats(
        test_resources.local_device.device->get_devices()[0], test_resources.local_device.eth_core, handshake_addr);
    TimingStats stats1 = read_timing_stats(
        test_resources.remote_device.device->get_devices()[0], test_resources.remote_device.eth_core, handshake_addr);
    auto report_erisc_timing_stats = [](TimingStats& stats, const std::string& label) {
        log_info(tt::LogAlways, "{} Timing Stats:", label);
        log_info(tt::LogAlways, "  Acquire operations: {}", stats.acquire_count);
        log_info(tt::LogAlways, "  Total acquire cycles: {}", stats.total_acquire_cycles);
        log_info(
            tt::LogTest,
            "  Average acquire cycles: {:.2f}",
            stats.acquire_count > 0 ? (double)stats.total_acquire_cycles / stats.acquire_count : 0.0);
        log_info(tt::LogAlways, "  Release operations: {}", stats.release_count);
        log_info(tt::LogAlways, "  Total release cycles: {}", stats.total_release_cycles);
        log_info(
            tt::LogTest,
            "  Average release cycles: {:.2f}",
            stats.release_count > 0 ? (double)stats.total_release_cycles / stats.release_count : 0.0);
        log_info(tt::LogAlways, "  Misc operations: {}", stats.misc_count);
        log_info(tt::LogAlways, "  Total misc cycles: {}", stats.total_misc_cycles);
        log_info(
            tt::LogTest,
            "  Average misc cycles: {:.2f}",
            stats.misc_count > 0 ? (double)stats.total_misc_cycles / stats.misc_count : 0.0);
    };
    report_erisc_timing_stats(stats0, "Local Device");
    report_erisc_timing_stats(stats1, "Remote Device");

    //
    // Worker Timing Stats Readback + Reporting
    //
    if (config.n_workers > 0) {
        log_info(tt::LogAlways, "Worker Timing Stats:");

        for (const auto& [resources, label] :
             {std::make_pair(&test_resources.local_device, "Local Device"),
              std::make_pair(&test_resources.remote_device, "Remote Device")}) {
            for (size_t i = 0; i < config.n_workers; i++) {
                auto worker_core = resources->worker_cores_vec[i];
                uint32_t worker_timing_stats_addr = resources->worker_timing_stats_addresses[i];
                WorkerTimingStats worker_stats = read_worker_timing_stats(
                    resources->device->get_devices()[0], worker_core, worker_timing_stats_addr);

                log_info(tt::LogAlways, "  {} Device Worker {}: ", label, i);
                log_info(tt::LogAlways, "    Idle operations: {}", worker_stats.idle_count);
                log_info(tt::LogAlways, "    Total idle cycles: {}", worker_stats.total_idle_cycles);
                log_info(
                    tt::LogTest,
                    "    Average idle cycles: {:.2f}",
                    worker_stats.idle_count > 0 ? (double)worker_stats.total_idle_cycles / worker_stats.idle_count
                                                : 0.0);
            }
        }
    }

    //
    // Calculate Throughput
    //
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
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_0,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_1,
    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    const TestConfig& config) {
    TestResources resources;
    resources.local_device.device = std::move(device_0);
    resources.remote_device.device = std::move(device_1);
    resources.local_device.eth_core = eth_sender_core;
    resources.remote_device.eth_core = eth_receiver_core;

    for (const auto& device_resource_reference :
         {std::ref(resources.local_device), std::ref(resources.remote_device)}) {
        auto& device_resource = device_resource_reference.get();
        auto& worker_cores = device_resource.worker_cores;
        auto& worker_cores_vec = device_resource.worker_cores_vec;
        worker_cores_vec.reserve(config.n_workers);
        worker_cores = CoreRange(CoreCoord(0, 0), CoreCoord(0, config.n_workers - 1));
        worker_cores_vec = corerange_to_cores(worker_cores);

        device_resource.worker_ack_semaphore_id =
            tt_metal::CreateSemaphore(device_resource.program, device_resource.worker_cores, 0, CoreType::WORKER);
        device_resource.worker_new_chunk_semaphore_id =
            tt_metal::CreateSemaphore(device_resource.program, device_resource.worker_cores, 0, CoreType::WORKER);

        // Create MeshBuffer for worker source data
        tt::tt_metal::distributed::ReplicatedBufferConfig replicated_config{
            .size = config.n_workers * config.message_size};
        tt::tt_metal::distributed::DeviceLocalBufferConfig local_config{
            .page_size = config.n_workers * config.message_size,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .sharding_args = tt::tt_metal::BufferShardingArgs()};
        auto worker_src_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            replicated_config, local_config, device_resource.device.get());
        device_resource.worker_src_buffer_address = worker_src_buffer->address();

        // Allocate timing stats buffers for each worker core
        device_resource.worker_timing_stats_addresses.reserve(config.n_workers);
        tt::tt_metal::distributed::ReplicatedBufferConfig timing_replicated_config{
            .size = config.n_workers * sizeof(WorkerTimingStats) * 2};
        tt::tt_metal::distributed::DeviceLocalBufferConfig timing_local_config{
            .page_size = config.n_workers * sizeof(WorkerTimingStats) * 2,
            .buffer_type = tt::tt_metal::BufferType::L1,
            .sharding_args = tt::tt_metal::BufferShardingArgs()};
        auto worker_timing_stats_buffer = tt::tt_metal::distributed::MeshBuffer::create(
            timing_replicated_config, timing_local_config, device_resource.device.get());
        for (uint32_t i = 0; i < config.n_workers; i++) {
            device_resource.worker_timing_stats_addresses.push_back(worker_timing_stats_buffer->address());
        }
    }

    return resources;
}

void validate_test_config(const TestConfig& config) {
    if (config.n_workers == 0) {
        log_error(tt::LogTest, "Number of workers must be greater than 0");
        exit(-1);
    }
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

    if (config.chunk_n_pkts == 0 || config.n_chunks == 0 || config.packet_size == 0 || config.message_size == 0 ||
        config.total_messages == 0 || config.n_workers == 0) {
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

/**
 * Run a single test case with the given configuration
 */
int run_test_case(const TestConfig& config, N300TestDevice& test_fixture) {
    try {
        const auto& device_0 = test_fixture.devices_.at(0);
        const auto& active_eth_cores = device_0->get_devices()[0]->get_active_ethernet_cores(true);
        auto eth_sender_core_iter = active_eth_cores.begin();
        TT_FATAL(eth_sender_core_iter != active_eth_cores.end(), "No active ethernet cores found");
        while (eth_sender_core_iter != active_eth_cores.end() and
               not tt::tt_metal::MetalContext::instance().get_cluster().is_ethernet_link_up(
                   device_0->get_devices()[0]->id(), *eth_sender_core_iter)) {
            eth_sender_core_iter++;
        }
        TT_FATAL(eth_sender_core_iter != active_eth_cores.end(), "No active ethernet cores found");
        auto eth_sender_core = *eth_sender_core_iter;
        TT_FATAL(
            device_0->get_devices()[0]->is_active_ethernet_core(eth_sender_core),
            "Not an active ethernet core {}",
            eth_sender_core);

        auto [device_id, eth_receiver_core] = device_0->get_devices()[0]->get_connected_ethernet_core(eth_sender_core);
        const auto& device_1 = test_fixture.devices_.at(device_id);

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

        log_info(tt::LogAlways, "Test completed successfully");
        return 0;
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Test failed with exception: {}", e.what());
        return -1;
    }
}

int main(int argc, char** argv) {
    // Check for pipe mode flag
    bool pipe_mode = false;
    if (argc >= 2 && std::string(argv[1]) == "--pipe-mode") {
        pipe_mode = true;
        log_info(tt::LogAlways, "Running in pipe mode - reading test configurations from stdin");
    }

    // Check hardware prerequisites
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

    if (pipe_mode) {
        // Pipe mode: read JSON configurations from stdin
        std::string line;
        int test_count = 0;
        int successful_tests = 0;

        log_info(
            tt::LogAlways, "Ready to receive test configurations. Send JSON configs via stdin, empty line to finish.");

        while (std::getline(std::cin, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
            line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

            // Empty line signals end of configurations
            if (line.empty()) {
                log_info(tt::LogAlways, "Received empty line - finishing test session");
                break;
            }

            // Skip comment lines
            if (line[0] == '#') {
                continue;
            }

            test_count++;
            log_info(tt::LogAlways, "Processing test case #{}", test_count);

            try {
                TestConfig config = parse_json_config(line);

                // Align packet size if needed
                if (config.packet_size % 16 != 0) {
                    log_warning(tt::LogTest, "Packet size is not aligned to 16 bytes. Aligning to 16 bytes.");
                    config.packet_size = tt::align(config.packet_size, 16);
                }

                validate_test_config(config);

                int result = run_test_case(config, test_fixture);
                if (result == 0) {
                    successful_tests++;
                    log_info(tt::LogAlways, "Test case #{} completed successfully", test_count);
                } else {
                    log_error(tt::LogTest, "Test case #{} failed", test_count);
                }

                // Flush output to ensure immediate visibility
                std::cout.flush();
                std::cerr.flush();

            } catch (const std::exception& e) {
                log_error(tt::LogTest, "Failed to parse or run test case #{}: {}", test_count, e.what());
            }
        }

        log_info(tt::LogAlways, "Pipe mode finished: {}/{} test cases successful", successful_tests, test_count);
        return (successful_tests == test_count && test_count > 0) ? 0 : -1;

    }  // Traditional CLI mode: single test from command line arguments
    TestConfig config{};
    try {
        config = parse_cli_config(argc, argv);

        // Align packet size if needed
        if (config.packet_size % 16 != 0) {
            log_warning(tt::LogTest, "Packet size is not aligned to 16 bytes. Aligning to 16 bytes.");
            config.packet_size = tt::align(config.packet_size, 16);
        }

        validate_test_config(config);

    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Configuration error: {}", e.what());
        log_info(tt::LogTest, "For pipe mode, use: {} --pipe-mode", argv[0]);
        return -1;
    }

    // Run single test case
    int result = run_test_case(config, test_fixture);
    return result;
}
