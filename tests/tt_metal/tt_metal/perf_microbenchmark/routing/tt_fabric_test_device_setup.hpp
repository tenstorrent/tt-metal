// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

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
const std::string default_mux_kernel_src = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using FabricMuxConfig = tt::tt_fabric::FabricMuxConfig;

namespace tt::tt_fabric::fabric_tests {

struct ConnectionKey {
    RoutingDirection direction;
    uint32_t link_idx;

    bool operator==(const ConnectionKey& other) const {
        return direction == other.direction && link_idx == other.link_idx;
    }

    bool operator<(const ConnectionKey& other) const {
        return std::tie(direction, link_idx) < std::tie(other.direction, other.link_idx);
    }
};

// Hash function for ConnectionKey to enable unordered_map
struct ConnectionKeyHash {
    std::size_t operator()(const ConnectionKey& key) const {
        return std::hash<int>()(static_cast<int>(key.direction)) ^ (std::hash<uint32_t>()(key.link_idx) << 1);
    }
};

// Worker type enum - defines different types of cores that can use fabric connections
enum class TestWorkerType : uint8_t { SENDER, RECEIVER, SYNC, MUX };

struct Connection {
    std::set<CoreCoord> sender_cores;           // Data senders (full-size channels)
    std::set<CoreCoord> receiver_cores;         // Credit senders (header-only channels)
    std::set<CoreCoord> sync_cores;             // Sync senders (header-only channels)
    std::map<CoreCoord, uint32_t> channel_map;  // Core -> channel assignment
    std::map<CoreCoord, TestWorkerType> core_worker_types;  // Core -> worker type mapping
    bool needs_mux = false;
};

// forward declarations
struct TestDevice;

// LocalDeviceCoreAllocator: Simple utility to manage pristine cores for local allocation
// Takes ownership of pristine cores and pops them on-demand
class LocalDeviceCoreAllocator {
public:
    explicit LocalDeviceCoreAllocator(std::vector<CoreCoord>&& available_cores) :
        available_cores_(std::move(available_cores)) {}

    std::optional<CoreCoord> allocate_core() {
        if (available_cores_.empty()) {
            return std::nullopt;
        }
        CoreCoord allocated = available_cores_.back();
        available_cores_.pop_back();
        return allocated;
    }

private:
    std::vector<CoreCoord> available_cores_;
};

// FabricConnectionManager: Centralized connection tracking and management
// Handles connection registration, mux detection, channel assignment, and argument generation
// This is a standalone utility class that can be used independently of TestDevice
class FabricConnectionManager {
public:
    FabricConnectionManager() = default;

    // Helper: Determine required channel type for a worker type
    static FabricMuxChannelType get_required_channel_type(TestWorkerType worker_type) {
        switch (worker_type) {
            case TestWorkerType::SENDER: return FabricMuxChannelType::FULL_SIZE_CHANNEL;  // Full payload packets
            case TestWorkerType::RECEIVER:                                                // Credit return packets
            case TestWorkerType::SYNC:                                                    // Atomic inc sync packets
                return FabricMuxChannelType::HEADER_ONLY_CHANNEL;
            default:
                TT_FATAL(
                    false,
                    "Invalid TestWorkerType for channel type determination: {}. "
                    "MUX channel type must be determined from its clients, not directly.",
                    worker_type);
        }
    }

    // Register a connection from a core in a specific direction and link
    void register_client(
        const CoreCoord& core, RoutingDirection direction, uint32_t link_idx, TestWorkerType worker_type);

    // Processing: Call once at start of create_kernels()
    // local_alloc: allocator for on-demand mux core allocation
    // Creates mux configs for all connections that need mux
    void process(
        LocalDeviceCoreAllocator& local_alloc,
        TestDevice* test_device_ptr,
        const std::shared_ptr<IDeviceInfoProvider>& device_info_provider);

    // Get all used fabric links (for telemetry/result reading)
    // Returns map of direction -> set of link indices that have been registered
    std::unordered_map<RoutingDirection, std::set<uint32_t>> get_used_fabric_links() const;

    // Get all connection keys for a specific core (fast lookup via reverse map)
    std::vector<ConnectionKey> get_connection_keys_for_core(const CoreCoord& core, TestWorkerType worker_type) const;

    // Get number of fabric connections for a specific core
    size_t get_connection_count_for_core(const CoreCoord& core, TestWorkerType worker_type) const;

    // Get the array index for a connection key in the core's connection list
    uint32_t get_connection_array_index_for_key(
        const CoreCoord& core, TestWorkerType worker_type, const ConnectionKey& key) const;

    // Check if a core is a mux client
    bool is_mux_client(const CoreCoord& core) const;

    // Get the number of muxes to terminate (for compile-time arg)
    uint32_t get_num_muxes_to_terminate() const { return static_cast<uint32_t>(mux_configs_.size()); }

    // Generate mux termination local args for a core
    // Returns empty vector if core is not a mux client
    std::vector<uint32_t> generate_mux_termination_local_args_for_core(
        const CoreCoord& core, const std::shared_ptr<IDeviceInfoProvider>& device_info_provider) const;

    // Generate all fabric connection args for a specific core
    // Returns rt_args to append (includes is_mux flag + connection args for each connection)
    // Parameters are passed in from TestDevice since it has all the context
    std::vector<uint32_t> generate_connection_args_for_core(
        const CoreCoord& core,
        TestWorkerType worker_type,
        const std::shared_ptr<IDeviceInfoProvider>& device_info_provider,
        const std::shared_ptr<IRouteManager>& route_manager,
        const FabricNodeId& fabric_node_id,
        tt::tt_metal::Program& program_handle) const;

private:
    std::unordered_map<ConnectionKey, Connection, ConnectionKeyHash> connections_;
    std::unordered_map<CoreCoord, std::set<ConnectionKey>> sender_core_to_keys_;
    std::unordered_map<CoreCoord, std::set<ConnectionKey>> receiver_core_to_keys_;
    std::unordered_map<CoreCoord, std::set<ConnectionKey>> sync_core_to_keys_;
    std::unordered_map<CoreCoord, ConnectionKey> mux_core_to_key_;

    // Mux state (populated during process())
    // One mux per connection key (each fabric link has its own mux)
    std::unordered_map<ConnectionKey, CoreCoord, ConnectionKeyHash> mux_cores_;  // connection key -> mux core location
    std::unordered_map<CoreCoord, std::unique_ptr<FabricMuxConfig>>
        mux_configs_;  // mux core -> mux config (1:1 with mux_cores_)

    // Mux termination state (populated during process())
    std::set<CoreCoord> all_mux_client_cores_;  // All cores that use mux connections
    CoreCoord global_termination_master_;       // First mux client (in deterministic order)

    static constexpr uint32_t MAX_FULL_SIZE_CHANNELS = 8;
    static constexpr uint32_t MAX_HEADER_ONLY_CHANNELS = 64;
    static constexpr uint32_t BUFFERS_PER_CHANNEL = 8;

    // Helper method to assign and validate mux channels for a connection
    void assign_and_validate_channels(Connection& conn, const ConnectionKey& key);
};

struct TestWorker {
public:
    virtual ~TestWorker() = default;
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
    ~TestSender() override = default;
    TestSender(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void add_config(TestTrafficSenderConfig config);
    bool validate_results(std::vector<uint32_t>& data) const override;

    const std::vector<std::pair<TestTrafficSenderConfig, ConnectionKey>>& get_configs() const { return configs_; }

    // Accessors for progress monitoring
    CoreCoord get_core() const { return logical_core_; }
    uint64_t get_total_packets() const;  // Defined out-of-line

    // stores traffic config and the corresponding fabric connection key
    // Managed by TestDevice::connection_manager_
    std::vector<std::pair<TestTrafficSenderConfig, ConnectionKey>> configs_;
};

struct TestReceiver : TestWorker {
public:
    TestReceiver(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void add_config(TestTrafficReceiverConfig config);
    bool validate_results(std::vector<uint32_t>& data) const override;

    // stores traffic config and the optional credit connection key (if flow control enabled)
    // Managed by TestDevice::connection_manager_
    std::vector<std::pair<TestTrafficReceiverConfig, std::optional<ConnectionKey>>> configs_;
};

struct TestSync : TestWorker {
public:
    TestSync(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void add_config(TestTrafficSyncConfig config);
    bool validate_results(std::vector<uint32_t>& data) const override;

    // stores traffic config and the corresponding fabric connection key
    // Managed by TestDevice::sync_connection_manager_
    std::vector<std::pair<TestTrafficSyncConfig, ConnectionKey>> configs_;
};

struct TestMux : TestWorker {
public:
    TestMux(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void set_config(FabricMuxConfig* config, ConnectionKey connection_key);
    bool validate_results(std::vector<uint32_t>& data) const override { return true; }  // Mux doesn't validate

    FabricMuxConfig* config_ = nullptr;
    ConnectionKey connection_key_{};
};

struct TestDevice {
    static constexpr uint32_t MAX_LATENCY_SAMPLES = 1024;

    // Friend declarations for tight coupling with worker structs
    friend struct TestSender;
    friend struct TestReceiver;
    friend struct TestSync;
    friend struct TestMux;

public:
    TestDevice(
        const MeshCoordinate& coord,
        std::shared_ptr<IDeviceInfoProvider> device_info_provider,
        std::shared_ptr<IRouteManager> route_manager,
        const SenderMemoryMap* sender_memory_map = nullptr,
        const ReceiverMemoryMap* receiver_memory_map = nullptr);
    tt::tt_metal::Program& get_program_handle();
    const FabricNodeId& get_node_id() const;
    void add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config);
    void add_sender_sync_config(CoreCoord logical_core, TestTrafficSyncConfig sync_config);
    void add_receiver_traffic_config(CoreCoord logical_core, const TestTrafficReceiverConfig& config);
    void add_mux_worker_config(CoreCoord logical_core, FabricMuxConfig* config, ConnectionKey connection_key);
    void create_kernels();

    // Latency test kernel creation (called directly by TestContext)
    // Uses static memory map addresses for semaphores (same as bandwidth tests)
    void create_latency_sender_kernel(
        CoreCoord core,
        FabricNodeId dest_node,
        uint32_t payload_size,
        uint32_t num_samples,
        NocSendType noc_send_type,
        CoreCoord responder_virtual_core);

    void create_latency_responder_kernel(
        CoreCoord core,
        FabricNodeId sender_node,
        uint32_t payload_size,
        uint32_t num_samples,
        NocSendType noc_send_type,
        uint32_t sender_send_buffer_address,
        uint32_t sender_receive_buffer_address,
        CoreCoord sender_virtual_core);

    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }
    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }
    void set_progress_monitoring_enabled(bool enabled) { progress_monitoring_enabled_ = enabled; }
    void set_pristine_cores(std::vector<CoreCoord>&& cores) { pristine_cores_ = std::move(cores); }

    // Set kernel source for specific workers (used by latency tests to override default kernels)
    void set_sender_kernel_src(CoreCoord core, const std::string& kernel_src);
    void set_receiver_kernel_src(CoreCoord core, const std::string& kernel_src);

    RoutingDirection get_forwarding_direction(const std::unordered_map<RoutingDirection, uint32_t>& hops) const;
    RoutingDirection get_forwarding_direction(const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const;
    std::vector<uint32_t> get_forwarding_link_indices_in_direction(const RoutingDirection& direction) const;
    std::vector<uint32_t> get_forwarding_link_indices_in_direction(
        const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id, const RoutingDirection& direction) const;
    // Validation read operations struct for non-blocking reads
    struct ValidationReadOps {
        TestFixture::ReadBufferOperation sender_op;
        TestFixture::ReadBufferOperation receiver_op;
        bool has_senders = false;
        bool has_receivers = false;
    };

    // New split validation methods for non-blocking operation
    ValidationReadOps initiate_results_readback() const;
    void validate_results_after_readback(const ValidationReadOps& ops) const;

    // Original validation method for backward compatibility
    void validate_results() const;
    void set_sync_core(CoreCoord coord) { sync_core_coord_ = coord; };
    void set_local_runtime_args_for_core(
        const MeshCoordinate& device_coord,
        CoreCoord logical_core,
        uint32_t local_args_address,
        const std::vector<uint32_t>& args) const;

    void set_use_unified_connection_manager(bool use_unified_connection_manager) {
        use_unified_connection_manager_ = use_unified_connection_manager;
    }

    // Method to access sender and receiver configurations for traffic analysis
    const std::unordered_map<CoreCoord, TestSender>& get_senders() const { return senders_; }
    const std::unordered_map<CoreCoord, TestReceiver>& get_receivers() const { return receivers_; }

    std::unordered_map<RoutingDirection, std::set<uint32_t>> get_used_fabric_connections() const {
        return connection_manager_.get_used_fabric_links();
    }

    const FabricConnectionManager& get_sync_connection_manager() const {
        if (use_unified_connection_manager_) {
            return connection_manager_;
        }
        return sync_connection_manager_;
    }

    FabricConnectionManager& get_sync_connection_manager() {
        if (use_unified_connection_manager_) {
            return connection_manager_;
        }
        return sync_connection_manager_;
    }

    // Connection managers: separate instances for regular and sync connections
    // Regular: persistent connections (open for duration of test)
    FabricConnectionManager connection_manager_;

    // Sync: ephemeral connections (open/close per sync, can reuse same physical links as regular)
    FabricConnectionManager sync_connection_manager_;

    std::shared_ptr<IDeviceInfoProvider> get_device_info_provider() const { return device_info_provider_; }

    // Latency buffer address getters (public so TestContext can query them)
    size_t get_latency_send_buffer_address() const;
    size_t get_latency_receive_buffer_address(uint32_t payload_size) const;

private:
    void add_worker(TestWorkerType worker_type, CoreCoord logical_core);
    void create_sender_kernels();
    void create_receiver_kernels();
    void validate_sender_results() const;
    void validate_receiver_results() const;
    void create_sync_kernel();
    void create_mux_kernels();

    // Helper: Common connection registration logic for senders and receivers
    // Registers a fabric connection for the specified direction and link
    ConnectionKey register_fabric_connection(
        CoreCoord logical_core,
        TestWorkerType worker_type,
        FabricConnectionManager& connection_mgr,
        RoutingDirection outgoing_direction,
        uint32_t link_idx);

    MeshCoordinate coord_;
    std::shared_ptr<IDeviceInfoProvider> device_info_provider_;
    std::shared_ptr<IRouteManager> route_manager_;
    const SenderMemoryMap* sender_memory_map_;
    const ReceiverMemoryMap* receiver_memory_map_;

    FabricNodeId fabric_node_id_ = FabricNodeId(MeshId{0}, 0);

    tt_metal::Program program_handle_;

    std::unordered_map<CoreCoord, TestSender> senders_;
    std::unordered_map<CoreCoord, TestReceiver> receivers_;
    std::unordered_map<CoreCoord, TestSync> sync_workers_;    // Separate sync cores
    std::unordered_map<CoreCoord, TestMux> muxes_;            // Mux workers

    bool benchmark_mode_ = false;
    bool global_sync_ = false;
    CoreCoord sync_core_coord_;
    bool progress_monitoring_enabled_ = false;

    // Pristine cores for mux allocation (transferred from allocator)
    std::vector<CoreCoord> pristine_cores_;

    bool use_unified_connection_manager_ = false;
};

}  // namespace tt::tt_fabric::fabric_tests
