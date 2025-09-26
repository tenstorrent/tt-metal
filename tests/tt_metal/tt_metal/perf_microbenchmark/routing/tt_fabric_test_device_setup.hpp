// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <string_view>
#include <memory>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt_metal/fabric/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>

#include "impl/context/metal_context.hpp"

#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

const std::string default_sender_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_sender.cpp";
const std::string default_receiver_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_receiver.cpp";
const std::string default_sync_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_sync.cpp";

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

namespace tt::tt_fabric {
namespace fabric_tests {

// forward declaration
struct TestDevice;

enum class TestWorkerType : uint8_t { SENDER, RECEIVER };

struct TestWorker {
public:
    TestWorker(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void set_kernel_src(const std::string_view& kernel_src);
    void create_kernel(
        const MeshCoordinate& device_coord,
        const std::vector<uint32_t>& ct_args,
        const std::vector<uint32_t>& rt_args,
        const std::vector<uint32_t>& local_args,
        uint32_t local_args_address,
        const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear) const;
    void collect_results();
    virtual bool validate_results(std::vector<uint32_t>& data) const = 0;
    void dump_results();

protected:
    CoreCoord logical_core_;
    uint32_t worker_id_{};
    std::string kernel_src_;
    TestDevice* test_device_ptr_;
};

struct TestSender : TestWorker {
public:
    TestSender(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void add_config(TestTrafficSenderConfig config);
    void add_sync_config(TestTrafficSenderConfig sync_config);
    void connect_to_fabric_router();
    bool validate_results(std::vector<uint32_t>& data) const override;

    // Method to access traffic configurations for traffic analysis
    const std::vector<std::pair<TestTrafficSenderConfig, uint32_t>>& get_configs() const { return configs_; }

    // global line sync configs - stores sync traffic configs with their fabric connection indices
    std::vector<std::pair<TestTrafficSenderConfig, uint32_t>> global_sync_configs_;

    // stores traffic config and the correspoding fabric_connection idx to use
    std::vector<std::pair<TestTrafficSenderConfig, uint32_t>> configs_;

    // book-keeping for all the fabric connections needed for this sender
    // [RoutingDirection][link_idx]
    std::vector<std::pair<RoutingDirection, uint32_t>> fabric_connections_;

    // book-keeping for sync-specific fabric connections
    // [RoutingDirection][link_idx]
    std::vector<std::pair<RoutingDirection, uint32_t>> sync_fabric_connections_;
};

struct TestReceiver : TestWorker {
public:
    TestReceiver(
        CoreCoord logical_core,
        TestDevice* test_device_ptr,
        bool is_shared,
        std::optional<std::string_view> kernel_src);
    void add_config(TestTrafficReceiverConfig config);
    bool is_shared_receiver();
    bool validate_results(std::vector<uint32_t>& data) const override;

    bool is_shared_;
    std::vector<TestTrafficReceiverConfig> configs_;
};

struct TestDevice {
public:
    TestDevice(
        const MeshCoordinate& coord,
        std::shared_ptr<IDeviceInfoProvider> device_info_provider,
        std::shared_ptr<IRouteManager> route_manager,
        const SenderMemoryMap* sender_memory_map = nullptr,
        const ReceiverMemoryMap* receiver_memory_map = nullptr);
    tt::tt_metal::Program& get_program_handle();
    const FabricNodeId& get_node_id() const;
    uint32_t add_fabric_connection(
        RoutingDirection direction, const std::vector<uint32_t>& link_indices, bool is_sync_fabric);
    void add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config);
    void add_sender_sync_config(CoreCoord logical_core, TestTrafficSenderConfig sync_config);
    void add_receiver_traffic_config(CoreCoord logical_core, const TestTrafficReceiverConfig& config);
    void create_kernels();
    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }
    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }
    void set_global_sync_val(uint32_t global_sync_val) { global_sync_val_ = global_sync_val; }
    void set_enable_mux(bool enable_mux) { enable_mux_ = enable_mux; }
    bool is_mux_enabled() const { return enable_mux_; }
    void set_mux_core(tt::tt_fabric::RoutingDirection direction, CoreCoord core) {
        if (core.x == 0 && core.y == 0) {
            log_warning(
                tt::LogTest,
                "Setting potentially invalid mux core (0,0) for direction {} on device {}",
                static_cast<int>(direction),
                fabric_node_id_);
        }
        mux_kernels_[direction].core = core;
        log_debug(
            tt::LogTest,
            "Set mux core {} for direction {} on device {}",
            core,
            static_cast<int>(direction),
            fabric_node_id_);
    }
    RoutingDirection get_forwarding_direction(const std::unordered_map<RoutingDirection, uint32_t>& hops) const;
    RoutingDirection get_forwarding_direction(const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const;
    std::vector<uint32_t> get_forwarding_link_indices_in_direction(const RoutingDirection& direction) const;
    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id, const RoutingDirection& direction) const;
    void validate_results() const;
    void set_sync_core(CoreCoord coord) { sync_core_coord_ = coord; };
    void set_local_runtime_args_for_core(
        const MeshCoordinate& device_coord,
        CoreCoord logical_core,
        uint32_t local_args_address,
        const std::vector<uint32_t>& args) const;

    // Method to access sender configurations for traffic analysis
    const std::unordered_map<CoreCoord, TestSender>& get_senders() const { return senders_; }

    const std::unordered_map<RoutingDirection, std::set<uint32_t>>& get_used_fabric_connections() const {
        return used_fabric_connections_;
    }

private:
    void add_worker(TestWorkerType worker_type, CoreCoord logical_core);
    std::vector<uint32_t> get_fabric_connection_args(CoreCoord core, RoutingDirection direction, uint32_t link_idx);
    std::vector<uint32_t> generate_fabric_connection_args(
        CoreCoord core, const std::vector<std::pair<RoutingDirection, uint32_t>>& fabric_connections);
    uint32_t calculate_initial_credits(uint32_t buffer_size_bytes, uint32_t packet_size_bytes);
    std::vector<uint32_t> generate_sender_traffic_config_args(CoreCoord core, const TestTrafficSenderConfig& config);
    void create_sender_kernels();
    void create_receiver_kernels();
    void validate_sender_results() const;
    void validate_receiver_results() const;
    void create_sync_kernel();

    // Mux support methods
    void setup_mux_channel_assignments();
    void create_mux_kernels();
    uint32_t assign_sender_channel(tt::tt_fabric::RoutingDirection direction, CoreCoord sender_core);
    uint32_t assign_receiver_channel(tt::tt_fabric::RoutingDirection direction, CoreCoord receiver_core);
    uint32_t get_sender_channel_id(tt::tt_fabric::RoutingDirection direction, CoreCoord sender_core) const;
    uint32_t get_receiver_channel_id(tt::tt_fabric::RoutingDirection direction, CoreCoord receiver_core) const;
    std::optional<FabricNodeId> get_dst_fabric_node_for_direction(tt::tt_fabric::RoutingDirection direction) const;

    MeshCoordinate coord_;
    std::shared_ptr<IDeviceInfoProvider> device_info_provider_;
    std::shared_ptr<IRouteManager> route_manager_;
    const SenderMemoryMap* sender_memory_map_;
    const ReceiverMemoryMap* receiver_memory_map_;

    FabricNodeId fabric_node_id_ = FabricNodeId(MeshId{0}, 0);

    tt_metal::Program program_handle_;

    std::unordered_map<CoreCoord, TestSender> senders_;
    std::unordered_map<CoreCoord, TestReceiver> receivers_;
    std::unordered_map<CoreCoord, TestSender> sync_senders_;  // Separate sync cores

    std::unordered_map<RoutingDirection, std::set<uint32_t>> used_fabric_connections_{};
    std::unordered_map<RoutingDirection, std::set<uint32_t>> used_sync_fabric_connections_{};

    bool benchmark_mode_ = false;
    bool global_sync_ = false;
    uint32_t global_sync_val_ = 0;
    CoreCoord sync_core_coord_;

    // Mux support
    bool enable_mux_ = false;
    struct MuxKernelInfo {
        CoreCoord core = {0, 0};  // Initialize to (0,0) for safety checks
        tt::tt_fabric::RoutingDirection direction;
        bool is_active = false;
        uint32_t full_size_channels = 0;
        uint32_t header_only_channels = 0;
        std::unique_ptr<tt::tt_fabric::FabricMuxConfig> config;
    };
    std::unordered_map<tt::tt_fabric::RoutingDirection, MuxKernelInfo> mux_kernels_;

    struct ChannelAssignment {
        uint32_t channel_id;
        CoreCoord assigned_core;
    };

    // Channel tracking per direction
    std::unordered_map<tt::tt_fabric::RoutingDirection, std::vector<ChannelAssignment>>
        sender_channel_assignments_;  // Full-size channels
    std::unordered_map<tt::tt_fabric::RoutingDirection, std::vector<ChannelAssignment>>
        receiver_channel_assignments_;  // Header-only channels

    // controller?
};

// Mux method implementations
inline void TestDevice::setup_mux_channel_assignments() {
    log_info(
        tt::LogTest,
        "SETTING UP mux channel assignments for device {} with {} senders, {} receivers",
        fabric_node_id_,
        senders_.size(),
        receivers_.size());

    // Initialize mux kernels for all directions
    constexpr tt::tt_fabric::RoutingDirection directions[] = {
        tt::tt_fabric::RoutingDirection::N,  // N = 0
        tt::tt_fabric::RoutingDirection::E,  // E = 2
        tt::tt_fabric::RoutingDirection::S,  // S = 4
        tt::tt_fabric::RoutingDirection::W   // W = 8
    };

    for (auto direction : directions) {
        mux_kernels_[direction].direction = direction;
        mux_kernels_[direction].is_active = false;
        mux_kernels_[direction].full_size_channels = 0;
        mux_kernels_[direction].header_only_channels = 0;
    }

    // Assign channels to senders (full-size channels for data)
    uint32_t total_sender_connections = 0;
    for (const auto& [core, sender] : senders_) {
        log_debug(
            tt::LogTest,
            "Processing sender on core {} with {} fabric connections",
            core,
            sender.fabric_connections_.size());
        for (const auto& [direction, link_idx] : sender.fabric_connections_) {
            assign_sender_channel(direction, core);
            total_sender_connections++;
        }
    }

    // Assign channels to receivers (header-only channels for credits)
    uint32_t total_receiver_connections = 0;
    for (const auto& [core, receiver] : receivers_) {
        log_debug(tt::LogTest, "Processing receiver on core {}", core);
        // For each receiver, determine which direction it should send credits back
        // This is based on where traffic is coming FROM (reverse direction)

        // For receivers, we need to assign credit return channels to activate mux kernels
        // For now, simplified logic: if we have receiver configs, assign one credit return direction
        // TODO: Make this more sophisticated based on actual traffic source topology
        if (!receiver.configs_.empty()) {
            log_info(
                tt::LogTest,
                "DEVICE {} HAS {} receiver configs - WILL ASSIGN MUX CREDIT CHANNEL (✅ Receivers need mux for "
                "credits!)",
                fabric_node_id_,
                receiver.configs_.size());

            // In a simple point-to-point test, receivers typically send credits back in the reverse direction
            // For now, just activate mux for credit return in a reasonable direction (e.g., W for eastbound traffic)
            tt::tt_fabric::RoutingDirection credit_return_direction = tt::tt_fabric::RoutingDirection::W;

            // Assign receiver channel for credit return (this will activate the mux)
            assign_receiver_channel(credit_return_direction, core);
            total_receiver_connections++;

            log_info(
                tt::LogTest,
                "✅ ASSIGNED receiver channel for credit return in direction {} for receiver on core {} - Device {} "
                "needs MUX for credits!",
                static_cast<int>(credit_return_direction),
                core,
                fabric_node_id_);
        } else {
            log_info(
                tt::LogTest,
                "Device {} core {} receiver has NO configs - skipping mux assignment",
                fabric_node_id_,
                core);
        }
    }

    // Log final channel counts
    for (const auto& [direction, mux_kernel] : mux_kernels_) {
        if (mux_kernel.is_active) {
            log_info(
                tt::LogTest,
                "Mux direction {} active: {} full-size channels, {} header-only channels",
                static_cast<int>(direction),
                mux_kernel.full_size_channels,
                mux_kernel.header_only_channels);
        }
    }

    log_debug(
        tt::LogTest,
        "Mux channel assignment complete for device {} - {} sender connections, {} receiver connections",
        fabric_node_id_,
        total_sender_connections,
        total_receiver_connections);
}

inline void TestDevice::create_mux_kernels() {
    log_info(tt::LogTest, "CREATING mux kernels for device {}", fabric_node_id_);

    // First, show which directions have active mux kernels
    for (const auto& [direction, mux_kernel] : mux_kernels_) {
        if (mux_kernel.is_active) {
            log_info(
                tt::LogTest,
                "  Device {} direction {} is ACTIVE: {} full-size, {} header-only channels",
                fabric_node_id_,
                static_cast<int>(direction),
                mux_kernel.full_size_channels,
                mux_kernel.header_only_channels);
        } else {
            log_info(tt::LogTest, "  Device {} direction {} is INACTIVE", fabric_node_id_, static_cast<int>(direction));
        }
    }

    const std::string mux_kernel_src = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
    // Mux configuration defaults - sensible values for typical use cases
    const uint32_t mux_base_l1_address = 0x1000;  // Default base address (TODO: Get from memory allocator)
    const uint32_t buffer_size_bytes = 2048;      // Default: 2KB buffer per channel
    const uint8_t buffers_per_channel = 4;        // Default: 4 buffers per channel

    uint32_t active_mux_kernels = 0;
    for (auto& [direction, mux_kernel] : mux_kernels_) {
        if (!mux_kernel.is_active) {
            continue;
        }

        // Safety check: FabricMuxConfig requires at least one channel
        if (mux_kernel.full_size_channels == 0 && mux_kernel.header_only_channels == 0) {
            log_warning(
                tt::LogTest,
                "Mux kernel for direction {} is marked active but has no channels - skipping creation",
                static_cast<int>(mux_kernel.direction));
            mux_kernel.is_active = false;
            continue;
        }

        log_debug(
            tt::LogTest,
            "Creating mux kernel for direction {} with {} full-size, {} header-only channels",
            static_cast<int>(mux_kernel.direction),
            mux_kernel.full_size_channels,
            mux_kernel.header_only_channels);

        // Safety check: validate mux core coordinates
        if (mux_kernel.core.x == 0 && mux_kernel.core.y == 0) {
            log_warning(
                tt::LogTest,
                "Mux kernel for direction {} has uninitialized core coordinates (0,0) - skipping creation",
                static_cast<int>(mux_kernel.direction));
            mux_kernel.is_active = false;
            continue;
        }

        log_debug(
            tt::LogTest,
            "Using mux core {} for direction {} kernel",
            mux_kernel.core,
            static_cast<int>(mux_kernel.direction));

        // 1. Create FabricMuxConfig with the channel counts
        log_info(
            tt::LogTest,
            "CREATING FabricMuxConfig: full_size={}, header_only={}, buffers={}, size={}, base=0x{:x}",
            mux_kernel.full_size_channels,
            mux_kernel.header_only_channels,
            buffers_per_channel,
            buffer_size_bytes,
            mux_base_l1_address);

        mux_kernel.config = std::make_unique<tt::tt_fabric::FabricMuxConfig>(
            mux_kernel.full_size_channels,    // num_full_size_channels
            mux_kernel.header_only_channels,  // num_header_only_channels
            buffers_per_channel,              // num_buffers_full_size_channel
            buffers_per_channel,              // num_buffers_header_only_channel
            buffer_size_bytes,                // buffer_size_bytes_full_size_channel
            mux_base_l1_address               // base_l1_address
        );

        // Debug: Check if the config was created successfully
        auto status_addr = mux_kernel.config->get_status_address();
        auto termination_addr = mux_kernel.config->get_termination_signal_address();
        log_info(
            tt::LogTest, "CREATED FabricMuxConfig: status_addr={}, termination_addr={}", status_addr, termination_addr);

        if (status_addr == 0) {
            log_error(tt::LogTest, "ERROR: FabricMuxConfig returned status_address=0! Config creation failed.");
        }
        if (termination_addr == 0) {
            log_error(tt::LogTest, "ERROR: FabricMuxConfig returned termination_address=0! Config creation failed.");
        }

        active_mux_kernels++;

        // 2. Get the destination device for fabric connection
        // For now, use a simple approach - connect to a neighboring device in the mux direction
        auto dst_fabric_node_id = get_dst_fabric_node_for_direction(mux_kernel.direction);
        if (!dst_fabric_node_id.has_value()) {
            log_debug(
                tt::LogTest,
                "No destination device found for direction {}, skipping mux kernel creation",
                static_cast<int>(mux_kernel.direction));
            continue;
        }

        // 3. Get available links for the connection
        auto available_links = tt::tt_fabric::get_forwarding_link_indices(fabric_node_id_, dst_fabric_node_id.value());
        if (available_links.empty()) {
            log_warning(
                tt::LogTest,
                "No forwarding links available from {} to {} for direction {}",
                fabric_node_id_,
                dst_fabric_node_id.value(),
                static_cast<int>(mux_kernel.direction));
            continue;
        }

        // 4. Get compile-time args from the config and configure for our test environment
        // Force wait_for_fabric_endpoint_ready to false to avoid hanging on fabric router readiness
        mux_kernel.config->set_wait_for_fabric_endpoint_ready(false);
        std::vector<uint32_t> mux_ct_args = mux_kernel.config->get_fabric_mux_compile_time_args();

        // 5. Get run-time args from the config
        std::vector<uint32_t> mux_rt_args = mux_kernel.config->get_fabric_mux_run_time_args(
            fabric_node_id_, dst_fabric_node_id.value(), available_links[0], program_handle_, mux_kernel.core);

        // 6. Create the mux kernel
        auto kernel_handle = tt::tt_metal::CreateKernel(
            program_handle_,
            mux_kernel_src,
            {mux_kernel.core},
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_ct_args,
                .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

        // 7. Set runtime args
        tt::tt_metal::SetRuntimeArgs(program_handle_, kernel_handle, mux_kernel.core, mux_rt_args);

        log_debug(
            tt::LogTest,
            "Successfully created mux kernel for direction {} on core {}",
            static_cast<int>(mux_kernel.direction),
            mux_kernel.core);
    }

    log_info(
        tt::LogTest,
        "Mux kernel creation complete for device {} - {} active mux kernels created",
        fabric_node_id_,
        active_mux_kernels);
}

inline uint32_t TestDevice::assign_sender_channel(tt::tt_fabric::RoutingDirection direction, CoreCoord sender_core) {
    // Check if this sender core already has a channel assigned for this direction
    for (const auto& assignment : sender_channel_assignments_[direction]) {
        if (assignment.assigned_core == sender_core) {
            log_debug(
                tt::LogTest,
                "Reusing existing sender channel {} for core {} in direction {}",
                assignment.channel_id,
                sender_core,
                static_cast<int>(direction));
            return assignment.channel_id;
        }
    }

    // Assign new channel
    uint32_t channel_id = sender_channel_assignments_[direction].size();
    sender_channel_assignments_[direction].push_back({channel_id, sender_core});

    // Update mux kernel info
    mux_kernels_[direction].is_active = true;
    mux_kernels_[direction].full_size_channels++;

    log_debug(
        tt::LogTest,
        "Assigned sender channel {} to core {} for direction {}",
        channel_id,
        sender_core,
        static_cast<int>(direction));

    return channel_id;
}

inline uint32_t TestDevice::assign_receiver_channel(
    tt::tt_fabric::RoutingDirection direction, CoreCoord receiver_core) {
    // Check if this receiver core already has a channel assigned for this direction
    for (const auto& assignment : receiver_channel_assignments_[direction]) {
        if (assignment.assigned_core == receiver_core) {
            log_debug(
                tt::LogTest,
                "Reusing existing receiver channel {} for core {} in direction {}",
                assignment.channel_id,
                receiver_core,
                static_cast<int>(direction));
            return assignment.channel_id;
        }
    }

    // Assign new channel
    uint32_t channel_id = receiver_channel_assignments_[direction].size();
    receiver_channel_assignments_[direction].push_back({channel_id, receiver_core});

    // Update mux kernel info
    mux_kernels_[direction].is_active = true;
    mux_kernels_[direction].header_only_channels++;

    log_debug(
        tt::LogTest,
        "Assigned receiver channel {} to core {} for direction {}",
        channel_id,
        receiver_core,
        static_cast<int>(direction));

    return channel_id;
}

inline uint32_t TestDevice::get_sender_channel_id(
    tt::tt_fabric::RoutingDirection direction, CoreCoord sender_core) const {
    auto it = sender_channel_assignments_.find(direction);
    if (it == sender_channel_assignments_.end()) {
        TT_THROW("No channel assignments found for direction {}", static_cast<int>(direction));
    }

    for (const auto& assignment : it->second) {
        if (assignment.assigned_core == sender_core) {
            return assignment.channel_id;
        }
    }

    TT_THROW("No channel assigned for sender core {} in direction {}", sender_core, static_cast<int>(direction));
}

inline uint32_t TestDevice::get_receiver_channel_id(
    tt::tt_fabric::RoutingDirection direction, CoreCoord receiver_core) const {
    auto it = receiver_channel_assignments_.find(direction);
    if (it == receiver_channel_assignments_.end()) {
        TT_THROW("No channel assignments found for direction {}", static_cast<int>(direction));
    }

    for (const auto& assignment : it->second) {
        if (assignment.assigned_core == receiver_core) {
            return assignment.channel_id;
        }
    }

    TT_THROW("No channel assigned for receiver core {} in direction {}", receiver_core, static_cast<int>(direction));
}

inline std::optional<FabricNodeId> TestDevice::get_dst_fabric_node_for_direction(
    tt::tt_fabric::RoutingDirection direction) const {
    // This is a simplified approach - in a real implementation, we'd need to:
    // 1. Look at the actual fabric topology
    // 2. Find the neighboring device in the specified direction
    // 3. Return its fabric node ID
    //
    // For now, we'll use the route manager to find a destination that uses this direction

    // Check if we have any senders using this direction
    for (const auto& [core, sender] : senders_) {
        for (const auto& [sender_direction, link_idx] : sender.fabric_connections_) {
            if (sender_direction == direction) {
                // Find a config that uses this direction and return its destination
                for (const auto& [config, fabric_connection_idx] : sender.configs_) {
                    if (!config.dst_node_ids.empty()) {
                        return config.dst_node_ids[0];
                    }
                }
            }
        }
    }

    // If no direct destination found, return nullopt - mux kernel creation will skip this direction
    return std::nullopt;
}

/* ********************
 * TestWorker Methods *
 **********************/
inline TestWorker::TestWorker(
    CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    logical_core_(logical_core), test_device_ptr_(test_device_ptr) {
    if (kernel_src.has_value()) {
        this->kernel_src_ = std::string(kernel_src.value());
    }

    // populate worker id
}

inline void TestWorker::set_kernel_src(const std::string_view& kernel_src) {
    this->kernel_src_ = std::string(kernel_src);
}

inline void TestWorker::create_kernel(
    const MeshCoordinate& device_coord,
    const std::vector<uint32_t>& ct_args,
    const std::vector<uint32_t>& rt_args,
    const std::vector<uint32_t>& local_args,
    uint32_t local_args_address,
    const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear) const {
    auto kernel_handle = tt::tt_metal::CreateKernel(
        this->test_device_ptr_->get_program_handle(),
        this->kernel_src_,
        {this->logical_core_},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    // Set fabric connection runtime args (for WorkerToFabricEdmSender::build_from_args)
    tt::tt_metal::SetRuntimeArgs(
        this->test_device_ptr_->get_program_handle(), kernel_handle, this->logical_core_, rt_args);

    // Set local args to memory buffer
    if (!local_args.empty()) {
        this->test_device_ptr_->set_local_runtime_args_for_core(
            device_coord, this->logical_core_, local_args_address, local_args);
    }

    // TODO: move this to mesh buffer?
    /*
    for (const auto& [address, num_bytes] : addresses_and_size_to_clear) {
        std::vector<uint32_t> zero_vec((num_bytes / sizeof(uint32_t)), 0);
        tt::tt_metal::detail::WriteToDeviceL1(device_handle, this->logical_core_, address, zero_vec);
    }
    */
}

/* ********************
 * TestSender Methods *
 **********************/
inline TestSender::TestSender(
    CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src) {
    // TODO: init mem map?
}

inline void TestSender::add_config(TestTrafficSenderConfig config) {
    std::optional<RoutingDirection> outgoing_direction;
    std::vector<uint32_t> outgoing_link_indices;
    // either we will have hops specified or the dest node id
    // With 2d unicast, we have bugs where we try to follow the input hop count but the routing tables
    // cause the packets to fail to reach the destination properly in some cases, due to torus links
    bool is_torus_2d_unicast = (config.parameters.topology == tt::tt_fabric::Topology::Torus) &&
                               (config.parameters.is_2D_routing_enabled) &&
                               (config.parameters.chip_send_type == ChipSendType::CHIP_UNICAST);
    if (config.hops.has_value() && !is_torus_2d_unicast) {
        outgoing_direction = this->test_device_ptr_->get_forwarding_direction(config.hops.value());
        outgoing_link_indices =
            this->test_device_ptr_->get_forwarding_link_indices_in_direction(outgoing_direction.value());
    } else {
        const auto dst_node_id = config.dst_node_ids[0];
        const auto src_node_id = this->test_device_ptr_->get_node_id();
        outgoing_direction = this->test_device_ptr_->get_forwarding_direction(src_node_id, dst_node_id);
        TT_FATAL(
            outgoing_direction.has_value(), "No forwarding direction found for {} from {}", dst_node_id, src_node_id);
        outgoing_link_indices = this->test_device_ptr_->get_forwarding_link_indices_in_direction(
            src_node_id, dst_node_id, outgoing_direction.value());
        TT_FATAL(
            !outgoing_link_indices.empty(),
            "No forwarding link indices found for {} from {}",
            dst_node_id,
            src_node_id);
    }

    std::optional<uint32_t> fabric_connection_idx;
    // first try to re-use an existing fabric connection as much as possible
    for (const auto& idx : outgoing_link_indices) {
        auto it = std::find(
            this->fabric_connections_.begin(),
            this->fabric_connections_.end(),
            std::make_pair(outgoing_direction.value(), idx));
        if (it != this->fabric_connections_.end()) {
            fabric_connection_idx = std::distance(this->fabric_connections_.begin(), it);
            break;
        }
    }

    if (!fabric_connection_idx.has_value()) {
        // insert a new fabric connection by first checking the device for unused links
        auto new_link_idx =
            this->test_device_ptr_->add_fabric_connection(outgoing_direction.value(), outgoing_link_indices, false);
        this->fabric_connections_.emplace_back(outgoing_direction.value(), new_link_idx);
        fabric_connection_idx = this->fabric_connections_.size() - 1;
    }

    this->configs_.emplace_back(std::move(config), fabric_connection_idx.value());
}

inline void TestSender::add_sync_config(TestTrafficSenderConfig sync_config) {
    // Similar to add_config but for sync patterns - uses separate sync fabric connections
    std::optional<RoutingDirection> outgoing_direction;
    std::vector<uint32_t> outgoing_link_indices;

    // Sync configs should always have hops specified (multicast pattern)
    outgoing_direction = this->test_device_ptr_->get_forwarding_direction(sync_config.hops.value());
    outgoing_link_indices =
        this->test_device_ptr_->get_forwarding_link_indices_in_direction(outgoing_direction.value());

    std::optional<uint32_t> sync_fabric_connection_idx;
    // Try to re-use existing sync fabric connection first
    for (const auto& idx : outgoing_link_indices) {
        auto it = std::find(
            this->sync_fabric_connections_.begin(),
            this->sync_fabric_connections_.end(),
            std::make_pair(outgoing_direction.value(), idx));
        if (it != this->sync_fabric_connections_.end()) {
            sync_fabric_connection_idx = std::distance(this->sync_fabric_connections_.begin(), it);
            break;
        }
    }

    if (!sync_fabric_connection_idx.has_value()) {
        // Add new sync fabric connection
        auto new_link_idx =
            this->test_device_ptr_->add_fabric_connection(outgoing_direction.value(), outgoing_link_indices, true);
        this->sync_fabric_connections_.emplace_back(outgoing_direction.value(), new_link_idx);
        sync_fabric_connection_idx = this->sync_fabric_connections_.size() - 1;
    }

    this->global_sync_configs_.emplace_back(std::move(sync_config), sync_fabric_connection_idx.value());
}

inline void TestSender::connect_to_fabric_router() {}

inline bool TestSender::validate_results(std::vector<uint32_t>& data) const {
    bool pass = data[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS;
    if (!pass) {
        const auto status = tt_fabric_status_to_string(data[TT_FABRIC_STATUS_INDEX]);
        log_error(tt::LogTest, "Sender on core {} failed with status: {}", this->logical_core_, status);
        return false;
    }

    uint32_t num_expected_packets = 0;
    for (const auto& [config, _] : this->configs_) {
        num_expected_packets += config.parameters.num_packets;
    }
    pass &= data[TT_FABRIC_WORD_CNT_INDEX] == num_expected_packets;
    if (!pass) {
        log_error(
            tt::LogTest,
            "Sender on core {} expected to send {} packets, sent {}",
            this->logical_core_,
            num_expected_packets,
            data[TT_FABRIC_WORD_CNT_INDEX]);
        return false;
    }

    return pass;
}

/* **********************
 * TestReceiver Methods *
 ************************/
inline TestReceiver::TestReceiver(
    CoreCoord logical_core, TestDevice* test_device_ptr, bool is_shared, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src), is_shared_(is_shared) {
    // TODO: init mem map?
}

inline void TestReceiver::add_config(TestTrafficReceiverConfig config) {
    this->configs_.emplace_back(std::move(config));
}

inline bool TestReceiver::is_shared_receiver() { return this->is_shared_; }

inline bool TestReceiver::validate_results(std::vector<uint32_t>& data) const {
    bool pass = data[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS;
    if (!pass) {
        const auto status = tt_fabric_status_to_string(data[TT_FABRIC_STATUS_INDEX]);
        log_error(tt::LogTest, "Receiver on core {} failed with status: {}", this->logical_core_, status);
        return false;
    }

    uint32_t num_expected_packets = 0;
    for (const auto& config : this->configs_) {
        num_expected_packets += config.parameters.num_packets;
    }
    pass &= data[TT_FABRIC_WORD_CNT_INDEX] == num_expected_packets;
    if (!pass) {
        log_error(
            tt::LogTest,
            "Receiver on core {} expected to receive {} packets, received {}",
            this->logical_core_,
            num_expected_packets,
            data[TT_FABRIC_WORD_CNT_INDEX]);
        return false;
    }

    return pass;
}

/* ********************
 * TestDevice Methods *
 **********************/
inline TestDevice::TestDevice(
    const MeshCoordinate& coord,
    std::shared_ptr<IDeviceInfoProvider> device_info_provider,
    std::shared_ptr<IRouteManager> route_manager,
    const SenderMemoryMap* sender_memory_map,
    const ReceiverMemoryMap* receiver_memory_map) :
    coord_(coord),
    device_info_provider_(std::move(device_info_provider)),
    route_manager_(std::move(route_manager)),
    sender_memory_map_(sender_memory_map),
    receiver_memory_map_(receiver_memory_map) {
    program_handle_ = tt::tt_metal::CreateProgram();
    fabric_node_id_ = device_info_provider_->get_fabric_node_id(coord);

    // TODO: init routers
}

inline tt::tt_metal::Program& TestDevice::get_program_handle() { return this->program_handle_; }

inline const FabricNodeId& TestDevice::get_node_id() const { return this->fabric_node_id_; }

inline uint32_t TestDevice::add_fabric_connection(
    RoutingDirection direction, const std::vector<uint32_t>& link_indices, bool is_sync_fabric) {
    auto& used_fabric_connections =
        is_sync_fabric ? this->used_sync_fabric_connections_ : this->used_fabric_connections_;
    // if all the connections have already been used by another worker, then its an error
    // else try to add whichever is not used
    if (used_fabric_connections.count(direction) == 0) {
        used_fabric_connections[direction] = {};
    }

    const auto& used_link_indices = used_fabric_connections.at(direction);
    std::optional<uint32_t> unused_link_idx;
    for (const auto& link_idx : link_indices) {
        if (used_link_indices.count(link_idx)) {
            continue;
        }
        unused_link_idx = link_idx;
        break;
    }
    TT_FATAL(
        unused_link_idx.has_value(),
        "On node: {}, in direction: {}, all the link indices are already used. Either update the allocation policy to "
        "use more sender configs per worker or add mux",
        this->fabric_node_id_,
        direction);

    used_fabric_connections[direction].insert(unused_link_idx.value());
    return unused_link_idx.value();
}

inline void TestDevice::add_worker(TestWorkerType worker_type, CoreCoord logical_core) {
    if (worker_type == TestWorkerType::SENDER) {
        if (this->senders_.count(logical_core)) {
            log_fatal(
                tt::LogTest,
                "On node: {}, trying to add a sender worker to an already occupied core: {}",
                this->fabric_node_id_,
                logical_core);
            throw std::runtime_error("Core already has a sender worker");
        }
        this->senders_.emplace(logical_core, TestSender(logical_core, this, default_sender_kernel_src));
    } else if (worker_type == TestWorkerType::RECEIVER) {
        if (this->receivers_.count(logical_core)) {
            log_fatal(
                tt::LogTest,
                "On node: {}, trying to add a receiver worker to an already occupied core: {}",
                this->fabric_node_id_,
                logical_core);
            throw std::runtime_error("Core already has a receiver worker");
        }
        bool is_shared_receiver = false;
        this->receivers_.emplace(
            logical_core, TestReceiver(logical_core, this, is_shared_receiver, default_receiver_kernel_src));
    } else {
        log_fatal(tt::LogTest, "Unknown worker type for core reservation: {}", worker_type);
        throw std::runtime_error("Unknown worker type");
    }
}

inline std::vector<uint32_t> TestDevice::get_fabric_connection_args(
    CoreCoord core, RoutingDirection direction, uint32_t link_idx) {
    std::vector<uint32_t> fabric_connection_args;
    const auto neighbor_node_id = this->route_manager_->get_neighbor_node_id(this->fabric_node_id_, direction);
    append_fabric_connection_rt_args(
        this->fabric_node_id_, neighbor_node_id, link_idx, this->program_handle_, core, fabric_connection_args);
    return fabric_connection_args;
}

inline std::vector<uint32_t> TestDevice::generate_fabric_connection_args(
    CoreCoord core, const std::vector<std::pair<RoutingDirection, uint32_t>>& fabric_connections) {
    std::vector<uint32_t> fabric_connection_args;
    for (const auto& [direction, link_idx] : fabric_connections) {
        const auto& args = get_fabric_connection_args(core, direction, link_idx);
        fabric_connection_args.insert(fabric_connection_args.end(), args.begin(), args.end());
    }
    return fabric_connection_args;
}

inline uint32_t TestDevice::calculate_initial_credits(uint32_t buffer_size_bytes, uint32_t packet_size_bytes) {
    // Calculate how many packets can fit in the buffer
    uint32_t max_packets_in_buffer = buffer_size_bytes / packet_size_bytes;

    // Use a percentage of the buffer as initial credits to avoid completely filling it
    const float credit_ratio = 0.8f;  // Sensible default: 80% of buffer capacity
    uint32_t initial_credits = static_cast<uint32_t>(max_packets_in_buffer * credit_ratio);

    // Ensure at least 1 credit
    return std::max(1u, initial_credits);
}

inline void TestDevice::create_sync_kernel() {
    log_debug(tt::LogTest, "creating sync kernel on node: {}", fabric_node_id_);

    // TODO: fetch these dynamically
    const bool is_2D_routing_enabled = this->device_info_provider_->is_2D_routing_enabled();
    const bool is_dynamic_routing_enabled = this->device_info_provider_->is_dynamic_routing_enabled();

    // Assuming single sync core per device for now
    TT_FATAL(
        sync_senders_.size() == 1,
        "Currently expecting exactly one sync core per device, got {}",
        sync_senders_.size());

    auto& [sync_core, sync_sender] = *sync_senders_.begin();

    // Compile-time args
    std::vector<uint32_t> ct_args = {
        is_2D_routing_enabled,
        is_dynamic_routing_enabled,
        sync_sender.sync_fabric_connections_.size(),        /* num sync fabric connections */
        static_cast<uint32_t>(senders_.size() + 1),         /* num local sync cores (all senders + sync core) */
        sender_memory_map_->common.get_kernel_config_size() /* kernel config buffer size */
    };

    // Runtime args: memory map args, then sync fabric connection args
    std::vector<uint32_t> rt_args = sender_memory_map_->get_memory_map_args();

    // Add sync fabric connection args (for WorkerToFabricEdmSender::build_from_args)
    std::vector<uint32_t> sync_fabric_connection_args;
    if (!sync_sender.sync_fabric_connections_.empty()) {
        sync_fabric_connection_args = generate_fabric_connection_args(sync_core, sync_sender.sync_fabric_connections_);
        rt_args.insert(rt_args.end(), sync_fabric_connection_args.begin(), sync_fabric_connection_args.end());
    }

    // Local args (all the rest go to local args buffer)
    std::vector<uint32_t> local_args;

    // Expected sync value for global sync
    local_args.push_back(this->global_sync_val_);

    // Add sync routing args for each sync config
    for (size_t i = 0; i < sync_sender.global_sync_configs_.size(); ++i) {
        const auto& [sync_config, fabric_conn_idx] = sync_sender.global_sync_configs_[i];

        // Add sync routing args (chip send type + routing info)
        auto sync_traffic_args = sync_config.get_args(true);
        log_debug(
            tt::LogTest,
            "fabric connection {} has sync config src_node_id: {} dst_node_ids {} hops {} mcast_start_hops {} ",
            fabric_conn_idx,
            sync_config.src_node_id,
            sync_config.dst_node_ids,
            sync_config.hops,
            sync_config.parameters.mcast_start_hops);
        local_args.insert(local_args.end(), sync_traffic_args.begin(), sync_traffic_args.end());
    }

    // Local sync args
    uint32_t local_sync_val = static_cast<uint32_t>(senders_.size() + 1);  // Expected sync value
    local_args.push_back(sender_memory_map_->get_local_sync_address());
    local_args.push_back(local_sync_val);

    // Add sync core's own NOC encoding first
    uint32_t sync_core_noc_encoding = this->device_info_provider_->get_worker_noc_encoding(sync_core);
    local_args.push_back(sync_core_noc_encoding);

    // Add other sender core coordinates for local sync
    for (const auto& [sender_core, _] : this->senders_) {
        uint32_t sender_noc_encoding = this->device_info_provider_->get_worker_noc_encoding(sender_core);
        local_args.push_back(sender_noc_encoding);
    }

    // create sync kernel with local args
    sync_sender.create_kernel(coord_, ct_args, rt_args, local_args, sender_memory_map_->get_local_args_address(), {});
    log_debug(tt::LogTest, "created sync kernel on core: {}", sync_core);
}

inline void TestDevice::create_sender_kernels() {
    // Unified sender kernel creation - handles both fabric and mux connections based on per-pattern flow control
    const bool is_2D_routing_enabled = this->device_info_provider_->is_2D_routing_enabled();
    const bool is_dynamic_routing_enabled = this->device_info_provider_->is_dynamic_routing_enabled();
    uint32_t num_local_sync_cores = static_cast<uint32_t>(this->senders_.size()) + 1;

    for (const auto& [core, sender] : this->senders_) {
        TT_FATAL(sender_memory_map_ != nullptr, "Sender memory map is required for creating sender kernels");
        TT_FATAL(sender_memory_map_->is_valid(), "Sender memory map is invalid");

        // NEW: Determine if any traffic configs need flow control (instead of global mux flag)
        bool any_traffic_needs_flow_control = false;
        for (const auto& config : sender.configs_) {
            if (config.parameters.enable_flow_control) {
                any_traffic_needs_flow_control = true;
                break;
            }
        }

        // Compile-time args
        std::vector<uint32_t> ct_args = {
            is_2D_routing_enabled,
            is_dynamic_routing_enabled,
            sender.fabric_connections_.size(),                   /* num connections */
            sender.configs_.size(),                              /* num traffic configs */
            (uint32_t)benchmark_mode_,                           /* benchmark mode */
            (uint32_t)global_sync_,                              /* line sync enabled */
            num_local_sync_cores,                                /* num local sync cores */
            sender_memory_map_->common.get_kernel_config_size(), /* kernel config buffer size */
            any_traffic_needs_flow_control ? 1u : 0u             /* USE_MUX (informational only) */
        };

        // Runtime args with connection type information
        std::vector<uint32_t> rt_args = sender_memory_map_->get_memory_map_args();

        // NEW: Per-connection type determination based on traffic configs that use each connection
        std::vector<bool> connection_needs_mux(sender.fabric_connections_.size(), false);

        // Determine which connections need mux based on traffic configs
        for (size_t i = 0; i < sender.configs_.size(); ++i) {
            const auto& config = sender.configs_[i];
            if (config.parameters.enable_flow_control) {
                // This traffic config needs flow control, so its connection needs mux
                uint8_t connection_idx = config.connection_idx;
                if (connection_idx < connection_needs_mux.size()) {
                    connection_needs_mux[connection_idx] = true;
                }
            }
        }

        // Add connection args with per-connection type flags
        size_t connection_idx = 0;
        for (const auto& [direction, link_idx] : sender.fabric_connections_) {
            bool this_connection_needs_mux = connection_needs_mux[connection_idx];

            // Add connection type flag first
            rt_args.push_back(this_connection_needs_mux ? 1u : 0u);  // is_mux flag

            if (this_connection_needs_mux) {
                // TODO: Add proper mux connection args (placeholder for now)
                log_warning(tt::LogTest, "MUX connection args not yet implemented for connection {}", connection_idx);
                // For now, add dummy mux connection args to avoid kernel crashes
                for (int i = 0; i < 12; i++) {
                    rt_args.push_back(0u);  // Placeholder mux args
                }
            } else {
                // Add fabric connection args (existing logic)
                auto fabric_conn_args = this->generate_fabric_connection_args(core, {{direction, link_idx}});
                rt_args.insert(rt_args.end(), fabric_conn_args.begin(), fabric_conn_args.end());
            }
            connection_idx++;
        }

        // Local args for traffic configs (existing logic)
        std::vector<uint32_t> local_args;

        // Add local sync args FIRST (parsed first by kernel)
        if (global_sync_) {
            // Add local sync configuration args (same as sync core, but no global sync)
            uint32_t local_sync_val =
                static_cast<uint32_t>(senders_.size() + 1);  // Expected sync value (all senders + sync core)
            local_args.push_back(sender_memory_map_->get_local_sync_address());
            local_args.push_back(local_sync_val);

            // Add sync core's NOC encoding (the master for local sync)
            uint32_t sync_core_noc_encoding = this->device_info_provider_->get_worker_noc_encoding(sync_core_coord_);
            local_args.push_back(sync_core_noc_encoding);

            // Add other sender core coordinates for local sync
            for (const auto& [sender_core, _] : this->senders_) {
                uint32_t sender_noc_encoding = this->device_info_provider_->get_worker_noc_encoding(sender_core);
                local_args.push_back(sender_noc_encoding);
            }
        }

        // Add traffic config connection mapping AFTER sync args
        for (size_t i = 0; i < sender.configs_.size(); ++i) {
            auto fabric_connection_idx = sender.configs_[i].connection_idx;
            local_args.push_back(fabric_connection_idx);
        }

        // Add sender traffic config args (including credit management info)
        for (const auto& config : sender.configs_) {
            auto traffic_config_args = this->generate_sender_traffic_config_args(core, config);
            local_args.insert(local_args.end(), traffic_config_args.begin(), traffic_config_args.end());

            // Credit management info is now included in traffic_config_args via TestTrafficSenderConfig::get_args()
        }

        // Create kernel using helper (consistent with sync and receiver kernel creation)
        sender.create_kernel(coord_, ct_args, rt_args, local_args, sender_memory_map_->get_local_args_address(), {});

        log_info(
            tt::LogTest,
            "Created sender kernel on core {} (flow_control_patterns={})",
            core,
            any_traffic_needs_flow_control);
    }
}

inline void TestDevice::create_receiver_kernels() {
    // Unified receiver kernel creation - handles both fabric and mux connections based on per-pattern flow control
    for (const auto& [core, receiver] : this->receivers_) {
        TT_FATAL(receiver_memory_map_ != nullptr, "Receiver memory map is required for creating receiver kernels");
        TT_FATAL(receiver_memory_map_->is_valid(), "Receiver memory map is invalid");

        // NEW: Determine if any traffic configs need flow control (receivers need mux connections for credit return)
        bool any_traffic_needs_flow_control = false;
        for (const auto& config : receiver.configs_) {
            if (config.parameters.enable_flow_control) {
                any_traffic_needs_flow_control = true;
                break;
            }
        }

        // Compile-time args
        std::vector<uint32_t> ct_args = {
            receiver.configs_.size(),                              /* num traffic configs */
            benchmark_mode_ ? 1u : 0u,                             /* benchmark mode */
            receiver_memory_map_->common.get_kernel_config_size(), /* kernel config buffer size */
            any_traffic_needs_flow_control ? 1u : 0u               /* USE_MUX (flow control enabled) */
        };

        // Runtime args: memory map args + credit connection args (if flow control enabled)
        std::vector<uint32_t> rt_args = receiver_memory_map_->get_memory_map_args();

        // NEW: Add credit connection args if flow control is enabled
        if (any_traffic_needs_flow_control) {
            // For each traffic config that needs flow control, add mux connection args for credit return
            // TODO: Implement proper credit connection args generation
            // For now, add placeholder connection args
            for (const auto& config : receiver.configs_) {
                if (config.parameters.enable_flow_control) {
                    // Placeholder: Add mux connection args (TODO: implement proper generation)
                    rt_args.push_back(1u);  // is_mux = true
                    // TODO: Add proper mux connection arguments here
                }
            }
        }

        // Local args for traffic configs
        std::vector<uint32_t> local_args;
        if (!receiver.configs_.empty()) {
            const auto first_traffic_args = receiver.configs_[0].get_args();
            local_args.reserve(local_args.size() + receiver.configs_.size() * first_traffic_args.size());
            local_args.insert(local_args.end(), first_traffic_args.begin(), first_traffic_args.end());

            for (size_t i = 1; i < receiver.configs_.size(); ++i) {
                const auto traffic_args = receiver.configs_[i].get_args();
                local_args.insert(local_args.end(), traffic_args.begin(), traffic_args.end());
            }
        }

        receiver.create_kernel(
            coord_, ct_args, rt_args, local_args, receiver_memory_map_->get_local_args_address(), {});
        log_info(
            tt::LogTest,
            "Created receiver kernel on core {} (flow_control_patterns={})",
            core,
            any_traffic_needs_flow_control);
    }
}

inline void TestDevice::create_kernels() {
    log_debug(tt::LogTest, "creating kernels on node: {}", fabric_node_id_);

    // Setup mux channel assignments if mux is enabled
    if (enable_mux_ && !benchmark_mode_) {
        this->setup_mux_channel_assignments();
        this->create_mux_kernels();
    }

    // create sync kernels
    if (global_sync_) {
        this->create_sync_kernel();
    }
    // create sender kernels
    this->create_sender_kernels();

    // create receiver kernels
    this->create_receiver_kernels();
}

inline void TestDevice::add_receiver_traffic_config(CoreCoord logical_core, const TestTrafficReceiverConfig& config) {
    if (this->receivers_.find(logical_core) == this->receivers_.end()) {
        this->add_worker(TestWorkerType::RECEIVER, logical_core);
    }
    this->receivers_.at(logical_core).add_config(config);
}

inline void TestDevice::add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config) {
    if (this->senders_.find(logical_core) == this->senders_.end()) {
        this->add_worker(TestWorkerType::SENDER, logical_core);
    }
    this->senders_.at(logical_core).add_config(std::move(config));
}

inline void TestDevice::add_sender_sync_config(CoreCoord logical_core, TestTrafficSenderConfig sync_config) {
    // Create sync sender if it doesn't exist
    if (this->sync_senders_.find(logical_core) == this->sync_senders_.end()) {
        this->sync_senders_.emplace(logical_core, TestSender(logical_core, this, default_sync_kernel_src));
    }
    this->sync_senders_.at(logical_core).add_sync_config(std::move(sync_config));
}

inline RoutingDirection TestDevice::get_forwarding_direction(
    const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
    return this->route_manager_->get_forwarding_direction(hops);
}

inline RoutingDirection TestDevice::get_forwarding_direction(
    const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const {
    return this->route_manager_->get_forwarding_direction(src_node_id, dst_node_id);
}

inline std::vector<uint32_t> TestDevice::get_forwarding_link_indices_in_direction(
    const RoutingDirection& direction) const {
    const auto link_indices =
        this->route_manager_->get_forwarding_link_indices_in_direction(this->fabric_node_id_, direction);
    TT_FATAL(
        !link_indices.empty(),
        "No forwarding link indices found in direction: {} from {}",
        direction,
        this->fabric_node_id_);
    return link_indices;
}

inline std::vector<uint32_t> TestDevice::get_forwarding_link_indices_in_direction(
    const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id, const RoutingDirection& direction) const {
    const auto link_indices =
        this->route_manager_->get_forwarding_link_indices_in_direction(src_node_id, dst_node_id, direction);
    TT_FATAL(
        !link_indices.empty(),
        "No forwarding link indices found in direction: {} from {}",
        direction,
        this->fabric_node_id_);
    return link_indices;
}

inline std::vector<uint32_t> TestDevice::generate_sender_traffic_config_args(
    CoreCoord core, const TestTrafficSenderConfig& config) {
    // Simple wrapper around TestTrafficSenderConfig::get_args() to match the interface
    // The credit management info is now included automatically via TestTrafficSenderConfig::get_args()
    return config.get_args(false);  // false = not a sync config
}

inline void TestDevice::validate_sender_results() const {
    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(this->senders_.size());
    for (const auto& [core, _] : this->senders_) {
        sender_cores.push_back(core);
    }

    if (sender_cores.empty()) {
        return;
    }

    // capture data from all the cores and then indidividually validate
    auto data = this->device_info_provider_->read_buffer_from_cores(
        this->coord_,
        sender_cores,
        sender_memory_map_->get_result_buffer_address(),
        sender_memory_map_->get_result_buffer_size());

    // validate data
    for (const auto& [core, sender] : this->senders_) {
        bool pass = sender.validate_results(data.at(core));
        TT_FATAL(pass, "Sender on device: {} core: {} failed", this->fabric_node_id_, core);
    }
}

inline void TestDevice::validate_receiver_results() const {
    std::vector<CoreCoord> receiver_cores;
    receiver_cores.reserve(this->receivers_.size());
    for (const auto& [core, _] : this->receivers_) {
        receiver_cores.push_back(core);
    }

    if (receiver_cores.empty()) {
        return;
    }

    // capture data from all the cores and then indidividually validate
    auto data = this->device_info_provider_->read_buffer_from_cores(
        this->coord_,
        receiver_cores,
        receiver_memory_map_->get_result_buffer_address(),
        receiver_memory_map_->get_result_buffer_size());

    // validate data
    for (const auto& [core, receiver] : this->receivers_) {
        bool pass = receiver.validate_results(data.at(core));
        TT_FATAL(pass, "Receiver on device: {} core: {} failed", this->fabric_node_id_, core);
    }
}

inline void TestDevice::validate_results() const {
    this->validate_sender_results();
    this->validate_receiver_results();
}

inline void TestDevice::set_local_runtime_args_for_core(
    const MeshCoordinate& device_coord,
    CoreCoord logical_core,
    uint32_t local_args_address,
    const std::vector<uint32_t>& args) const {
    device_info_provider_->write_data_to_core(device_coord, logical_core, local_args_address, args);
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
