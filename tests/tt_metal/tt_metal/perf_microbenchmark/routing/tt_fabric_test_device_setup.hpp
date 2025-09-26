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

    // Forward declare ConnectionKey from FabricConnectionManager
    using ConnectionKey = TestDevice::FabricConnectionManager::ConnectionKey;

    // Method to access traffic configurations for traffic analysis
    const std::vector<std::pair<TestTrafficSenderConfig, ConnectionKey>>& get_configs() const { return configs_; }

    // global line sync configs - stores sync traffic configs with their fabric connection keys
    // Managed by TestDevice::sync_connection_manager_ (separate instance from regular connections)
    std::vector<std::pair<TestTrafficSenderConfig, ConnectionKey>> global_sync_configs_;

    // stores traffic config and the corresponding fabric connection key
    // Managed by TestDevice::connection_manager_
    std::vector<std::pair<TestTrafficSenderConfig, ConnectionKey>> configs_;
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

struct TestMux : TestWorker {
public:
    TestMux(
        CoreCoord logical_core,
        TestDevice* test_device_ptr,
        RoutingDirection direction,
        uint32_t link_idx,
        uint32_t full_size_channels,
        uint32_t header_only_channels);
    void create_kernel();
    bool validate_results(std::vector<uint32_t>& data) const override { return true; }  // Mux doesn't validate

    RoutingDirection direction_;
    uint32_t link_idx_;
    uint32_t full_size_channels_;
    uint32_t header_only_channels_;
    std::unique_ptr<tt::tt_fabric::FabricMuxConfig> config_;
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
    void add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config);
    void add_sender_sync_config(CoreCoord logical_core, TestTrafficSenderConfig sync_config);
    void add_receiver_traffic_config(CoreCoord logical_core, const TestTrafficReceiverConfig& config);
    void create_kernels();
    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }
    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }
    void set_global_sync_val(uint32_t global_sync_val) { global_sync_val_ = global_sync_val; }
    // NOTE: Mux enablement is now handled per-pattern via enable_flow_control and FabricConnectionManager
    void set_mux_cores(const std::unordered_map<tt::tt_fabric::RoutingDirection, CoreCoord>& mux_cores) {
        mux_cores_ = mux_cores;
        log_debug(tt::LogTest, "Set {} mux cores for device {}", mux_cores_.size(), fabric_node_id_);
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

    std::unordered_map<RoutingDirection, std::set<uint32_t>> get_used_fabric_connections() const {
        return connection_manager_.get_used_fabric_links();
    }

private:
    void add_worker(TestWorkerType worker_type, CoreCoord logical_core);
    std::vector<uint32_t> get_fabric_connection_args(CoreCoord core, RoutingDirection direction, uint32_t link_idx);
    std::vector<uint32_t> generate_fabric_connection_args(
        CoreCoord core, const std::vector<std::pair<RoutingDirection, uint32_t>>& fabric_connections);
    uint32_t calculate_initial_credits(uint32_t buffer_size_bytes, uint32_t packet_size_bytes);
    void create_sender_kernels();
    void create_receiver_kernels();
    void validate_sender_results() const;
    void validate_receiver_results() const;
    void create_sync_kernel();
    void create_mux_kernels();

    // Mux support is now handled by FabricConnectionManager

    // Fabric Connection Management - simplified and streamlined
    class FabricConnectionManager {
    public:
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

        // Registration: Call from add_sender/receiver_traffic_config if enable_flow_control
        void register_client(const CoreCoord& core, RoutingDirection direction, uint32_t link_idx, bool is_sender) {
            ConnectionKey key = {direction, link_idx};
            auto& conn = connections_[key];  // Auto-creates if new

            // Add to appropriate set (dedup happens automatically with std::set)
            if (is_sender) {
                conn.sender_cores.insert(core);
                sender_core_to_keys_[core].insert(key);
            } else {
                conn.receiver_cores.insert(core);
                receiver_core_to_keys_[core].insert(key);
            }
        }

        // Processing: Call once at start of create_kernels()
        void process() {
            uint32_t connection_idx = 0;

            for (auto& [key, conn] : connections_) {
                conn.connection_idx = connection_idx++;
                conn.needs_mux = (conn.sender_cores.size() + conn.receiver_cores.size()) > 1;

                if (conn.needs_mux) {
                    assign_and_validate_channels(conn, key);
                }
            }
        }

        // Access to all connections (for mux kernel creation)
        const std::unordered_map<ConnectionKey, Connection, ConnectionKeyHash>& get_all_connections() const {
            return connections_;
        }

        // Get all used fabric links (for telemetry/result reading)
        // Returns map of direction -> set of link indices that have been registered
        std::unordered_map<RoutingDirection, std::set<uint32_t>> get_used_fabric_links() const {
            std::unordered_map<RoutingDirection, std::set<uint32_t>> result;
            for (const auto& [key, conn] : connections_) {
                result[key.direction].insert(key.link_idx);
            }
            return result;
        }

        // Get all connection keys for a specific core (fast lookup via reverse map)
        std::vector<ConnectionKey> get_connection_keys_for_core(const CoreCoord& core, bool is_sender) const {
            const auto& map = is_sender ? sender_core_to_keys_ : receiver_core_to_keys_;
            auto it = map.find(core);
            if (it != map.end()) {
                return std::vector<ConnectionKey>(it->second.begin(), it->second.end());
            }
            return {};
        }

        // Select an unused link from candidates (used for connection allocation)
        // Returns the first link_idx that is not currently registered
        std::optional<uint32_t> select_unused_link(
            RoutingDirection direction, const std::vector<uint32_t>& candidate_link_indices) const {
            auto used_links = get_used_fabric_links();
            const auto& used_link_indices =
                used_links.count(direction) ? used_links.at(direction) : std::set<uint32_t>{};

            for (const auto& link_idx : candidate_link_indices) {
                if (used_link_indices.count(link_idx) == 0) {
                    return link_idx;
                }
            }

            return std::nullopt;  // All candidates are used
        }

        // Get number of fabric connections for a specific core
        size_t get_connection_count_for_core(const CoreCoord& core, bool is_sender) const {
            const auto& map = is_sender ? sender_core_to_keys_ : receiver_core_to_keys_;
            auto it = map.find(core);
            return (it != map.end()) ? it->second.size() : 0;
        }

        // Get the array index for a connection key in the core's connection list
        uint32_t get_connection_array_index_for_key(
            const CoreCoord& core, bool is_sender, const ConnectionKey& key) const {
            auto keys = get_connection_keys_for_core(core, is_sender);

            for (size_t i = 0; i < keys.size(); i++) {
                if (keys[i] == key) {
                    return i;
                }
            }

            return UINT32_MAX;  // Not found
        }

        // Generate all fabric connection args for a specific core
        // Returns rt_args to append (includes is_mux flag + connection args for each connection)
        std::vector<uint32_t> generate_connection_args_for_core(
            const CoreCoord& core, bool is_sender, TestDevice* device_ptr) const {
            std::vector<uint32_t> rt_args;

            auto keys = get_connection_keys_for_core(core, is_sender);

            for (const auto& key : keys) {
                auto conn_it = connections_.find(key);
                if (conn_it == connections_.end()) {
                    continue;
                }

                const auto& conn = conn_it->second;

                // Add connection type flag first
                rt_args.push_back(conn.needs_mux ? 1u : 0u);

                if (conn.needs_mux) {
                    // TODO: Add proper mux connection args
                    log_warning(
                        tt::LogTest,
                        "MUX connection args not yet fully implemented for core {} dir={} link={}",
                        core,
                        static_cast<int>(key.direction),
                        key.link_idx);
                    // Placeholder mux args
                    for (int i = 0; i < 12; i++) {
                        rt_args.push_back(0u);
                    }
                } else {
                    // Generate fabric connection args
                    auto fabric_conn_args =
                        device_ptr->generate_fabric_connection_args(core, {{key.direction, key.link_idx}});
                    rt_args.insert(rt_args.end(), fabric_conn_args.begin(), fabric_conn_args.end());
                }
            }

            return rt_args;
        }

    private:
        struct Connection {
            std::set<CoreCoord> sender_cores;           // Dedup built-in
            std::set<CoreCoord> receiver_cores;         // Dedup built-in
            std::map<CoreCoord, uint32_t> channel_map;  // Core -> channel assignment
            uint32_t connection_idx = 0;
            bool needs_mux = false;
        };

        // Main connection storage (O(1) lookup)
        std::unordered_map<ConnectionKey, Connection, ConnectionKeyHash> connections_;

        // Reverse lookup maps: core -> set of ConnectionKeys (O(1) lookup to get all keys for a core)
        std::unordered_map<CoreCoord, std::set<ConnectionKey>> sender_core_to_keys_;
        std::unordered_map<CoreCoord, std::set<ConnectionKey>> receiver_core_to_keys_;

        static constexpr uint32_t MAX_FULL_SIZE_CHANNELS = 16;    // Based on mux L1 limits
        static constexpr uint32_t MAX_HEADER_ONLY_CHANNELS = 16;  // Based on mux L1 limits

        void assign_and_validate_channels(Connection& conn, const ConnectionKey& key) {
            uint32_t next_full_size = 0;
            uint32_t next_header_only = 0;

            // Assign full-size channels to senders
            for (const auto& core : conn.sender_cores) {
                conn.channel_map[core] = next_full_size++;
            }

            // Assign header-only channels to receivers
            for (const auto& core : conn.receiver_cores) {
                conn.channel_map[core] = next_header_only++;
            }

            // Validate inline
            TT_FATAL(
                next_full_size <= MAX_FULL_SIZE_CHANNELS,
                "Exceeded full-size channel limit: {} senders for connection direction={} link={} (max={})",
                next_full_size,
                static_cast<int>(key.direction),
                key.link_idx,
                MAX_FULL_SIZE_CHANNELS);

            TT_FATAL(
                next_header_only <= MAX_HEADER_ONLY_CHANNELS,
                "Exceeded header-only channel limit: {} receivers for connection direction={} link={} (max={})",
                next_header_only,
                static_cast<int>(key.direction),
                key.link_idx,
                MAX_HEADER_ONLY_CHANNELS);
        }
    };

    // Connection managers: separate instances for regular and sync connections
    // Regular: persistent connections (open for duration of test)
    FabricConnectionManager connection_manager_;

    // Sync: ephemeral connections (open/close per sync, can reuse same physical links as regular)
    FabricConnectionManager sync_connection_manager_;

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
    std::unordered_map<CoreCoord, TestMux> muxes_;            // Mux workers

    // Fabric connection tracking is now fully handled by FabricConnectionManager
    // (used for link selection, mux detection, and connection arg generation)

    bool benchmark_mode_ = false;
    bool global_sync_ = false;
    uint32_t global_sync_val_ = 0;
    CoreCoord sync_core_coord_;

    // NOTE: Mux support is now handled per-pattern via enable_flow_control and FabricConnectionManager
    // Map from direction to mux core coordinate
    std::unordered_map<tt::tt_fabric::RoutingDirection, CoreCoord> mux_cores_;

    // controller?
};

inline void TestDevice::create_mux_kernels() {
    log_info(tt::LogTest, "Creating mux kernels for device {} based on FabricConnectionManager", fabric_node_id_);

    // Get all connections that need mux from the connection manager
    const auto& all_connections = connection_manager_.get_all_connections();

    if (all_connections.empty()) {
        log_debug(tt::LogTest, "No fabric connections registered, skipping mux kernel creation");
        return;
    }

    uint32_t mux_kernels_created = 0;

    // Iterate through all connections and create mux kernels for those that need it
    for (const auto& [key, conn_info] : all_connections) {
        if (!conn_info.needs_mux) {
            log_debug(
                tt::LogTest,
                "Connection (direction={}, link_idx={}) doesn't need mux - skipping",
                static_cast<int>(key.direction),
                key.link_idx);
            continue;
        }

        // Get mux core for this direction (set via set_mux_cores() from allocator)
        auto mux_core_it = mux_cores_.find(key.direction);
        if (mux_core_it == mux_cores_.end()) {
            log_warning(
                tt::LogTest,
                "No mux core assigned for direction {} on device {} - skipping mux kernel creation",
                static_cast<int>(key.direction),
                fabric_node_id_);
            continue;
        }
        CoreCoord mux_core = mux_core_it->second;

        // Validate mux core coordinates
        if (mux_core.x == 0 && mux_core.y == 0) {
            log_warning(
                tt::LogTest,
                "Mux core (0,0) may be invalid for direction {} on device {} - skipping",
                static_cast<int>(key.direction),
                fabric_node_id_);
            continue;
        }

        log_info(
            tt::LogTest,
            "Creating mux at core {} for direction {} link {} on device {} - {} senders, {} receivers",
            mux_core,
            static_cast<int>(key.direction),
            key.link_idx,
            fabric_node_id_,
            conn_info.sender_cores.size(),
            conn_info.receiver_cores.size());

        // Count channel types: senders use full-size channels, receivers use header-only channels
        uint32_t full_size_channels = conn_info.sender_cores.size();
        uint32_t header_only_channels = conn_info.receiver_cores.size();

        // Create TestMux worker and add to muxes_ map (consistent with senders/receivers)
        muxes_.emplace(
            mux_core, TestMux(mux_core, this, key.direction, key.link_idx, full_size_channels, header_only_channels));

        // Create the actual kernel
        muxes_.at(mux_core).create_kernel();

        mux_kernels_created++;
    }

    log_info(
        tt::LogTest,
        "Mux kernel creation complete for device {} - created {} mux kernels",
        fabric_node_id_,
        mux_kernels_created);
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

    // Two-level reuse strategy:
    // 1. First, try to reuse an existing connection that THIS core already has
    // 2. If not found, allocate a new link from the device (which finds an unused link)

    std::optional<ConnectionKey> fabric_connection_key;

    // Check if this core already has a connection for any of the candidate links
    auto registered_keys = this->test_device_ptr_->connection_manager_.get_connection_keys_for_core(
        this->logical_core_, true);  // is_sender=true

    for (const auto& link_idx : outgoing_link_indices) {
        ConnectionKey candidate_key{outgoing_direction.value(), link_idx};

        // Check if this core already registered this connection
        if (std::find(registered_keys.begin(), registered_keys.end(), candidate_key) != registered_keys.end()) {
            fabric_connection_key = candidate_key;
            break;
        }
    }

    if (!fabric_connection_key.has_value()) {
        // No existing connection found - select an unused link from candidates
        auto new_link_idx = this->test_device_ptr_->connection_manager_.select_unused_link(
            outgoing_direction.value(), outgoing_link_indices);

        TT_FATAL(
            new_link_idx.has_value(),
            "On node {}, in direction {}, all link indices are already used. Either update allocation policy "
            "or enable mux (flow control)",
            this->test_device_ptr_->get_node_id(),
            static_cast<int>(outgoing_direction.value()));

        fabric_connection_key = ConnectionKey{outgoing_direction.value(), new_link_idx.value()};

        // Register the new connection with the connection manager
        // (Only new connections need registration; existing ones are already registered)
        this->test_device_ptr_->connection_manager_.register_client(
            this->logical_core_, outgoing_direction.value(), new_link_idx.value(), true);  // is_sender=true
    }

    this->configs_.emplace_back(std::move(config), fabric_connection_key.value());
}

inline void TestSender::add_sync_config(TestTrafficSenderConfig sync_config) {
    // Sync configs use SEPARATE FabricConnectionManager instance (sync_connection_manager_)
    // - Sync runs on a separate core and has different connection lifecycle
    // - Sync connections: OPEN → SEND → CLOSE (ephemeral, within global_sync())
    // - Regular connections: OPEN → (stay open) → SEND many packets → CLOSE
    // - No temporal overlap → can safely reuse same physical fabric links

    std::optional<RoutingDirection> outgoing_direction;
    std::vector<uint32_t> outgoing_link_indices;

    // Sync configs should always have hops specified (multicast pattern)
    outgoing_direction = this->test_device_ptr_->get_forwarding_direction(sync_config.hops.value());
    outgoing_link_indices =
        this->test_device_ptr_->get_forwarding_link_indices_in_direction(outgoing_direction.value());

    // Use same two-level reuse strategy as regular configs, but with sync_connection_manager_
    std::optional<ConnectionKey> fabric_connection_key;

    // Check if this core already has a sync connection for any of the candidate links
    auto registered_keys = this->test_device_ptr_->sync_connection_manager_.get_connection_keys_for_core(
        this->logical_core_, true);  // is_sender=true

    for (const auto& link_idx : outgoing_link_indices) {
        ConnectionKey candidate_key{outgoing_direction.value(), link_idx};

        if (std::find(registered_keys.begin(), registered_keys.end(), candidate_key) != registered_keys.end()) {
            fabric_connection_key = candidate_key;
            break;
        }
    }

    if (!fabric_connection_key.has_value()) {
        // No existing sync connection found - select an unused link from sync_connection_manager_
        // Note: sync_connection_manager_ is independent from connection_manager_
        // So sync can reuse links that regular connections are using
        auto new_link_idx = this->test_device_ptr_->sync_connection_manager_.select_unused_link(
            outgoing_direction.value(), outgoing_link_indices);

        TT_FATAL(
            new_link_idx.has_value(),
            "On node {}, in direction {}, all link indices are already used for sync. "
            "This should not happen as sync uses a separate connection manager.",
            this->test_device_ptr_->get_node_id(),
            static_cast<int>(outgoing_direction.value()));

        fabric_connection_key = ConnectionKey{outgoing_direction.value(), new_link_idx.value()};

        // Register the new sync connection with sync_connection_manager_
        this->test_device_ptr_->sync_connection_manager_.register_client(
            this->logical_core_, outgoing_direction.value(), new_link_idx.value(), true);  // is_sender=true
    }

    this->global_sync_configs_.emplace_back(std::move(sync_config), fabric_connection_key.value());
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

/* *****************
 * TestMux Methods *
 *******************/
inline TestMux::TestMux(
    CoreCoord logical_core,
    TestDevice* test_device_ptr,
    RoutingDirection direction,
    uint32_t link_idx,
    uint32_t full_size_channels,
    uint32_t header_only_channels) :
    TestWorker(logical_core, test_device_ptr, "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp"),
    direction_(direction),
    link_idx_(link_idx),
    full_size_channels_(full_size_channels),
    header_only_channels_(header_only_channels) {
    // Config will be created in create_kernel()
}

inline void TestMux::create_kernel() {
    // Mux configuration - use device info provider for proper values
    const uint32_t mux_base_l1_address = test_device_ptr_->device_info_provider_->get_l1_unreserved_base();
    const uint32_t buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint8_t buffers_per_channel = 4;  // 4 buffers per channel

    log_debug(
        tt::LogTest,
        "TestMux creating kernel at core {} for direction {} link {} with {} full-size, {} header-only channels",
        logical_core_,
        static_cast<int>(direction_),
        link_idx_,
        full_size_channels_,
        header_only_channels_);

    // Create FabricMuxConfig
    config_ = std::make_unique<tt::tt_fabric::FabricMuxConfig>(
        full_size_channels_,    // num_full_size_channels
        header_only_channels_,  // num_header_only_channels
        buffers_per_channel,    // num_buffers_full_size_channel
        buffers_per_channel,    // num_buffers_header_only_channel
        buffer_size_bytes,      // buffer_size_bytes_full_size_channel
        mux_base_l1_address     // base_l1_address
    );

    // Get destination fabric node using route manager - cleaner than searching through configs
    const auto dst_fabric_node_id =
        test_device_ptr_->route_manager_->get_neighbor_node_id(test_device_ptr_->fabric_node_id_, direction_);

    // Configure mux to not wait for fabric endpoint ready (avoid hangs during testing)
    config_->set_wait_for_fabric_endpoint_ready(false);

    // Get compile-time and runtime args
    // Use link_idx directly from the connection key instead of querying again
    std::vector<uint32_t> mux_ct_args = config_->get_fabric_mux_compile_time_args();
    std::vector<uint32_t> mux_rt_args = config_->get_fabric_mux_run_time_args(
        test_device_ptr_->fabric_node_id_,
        dst_fabric_node_id,
        link_idx_,  // Use link_idx from connection key
        test_device_ptr_->program_handle_,
        logical_core_);

    // Create the mux kernel using base class method to avoid duplication
    // Pass empty local_args and addresses_to_clear since mux doesn't need them
    this->create_kernel(
        test_device_ptr_->coord_,
        mux_ct_args,
        mux_rt_args,
        {},   // local_args (empty for mux)
        0,    // local_args_address (not used for mux)
        {});  // addresses_and_size_to_clear (empty for mux)

    log_info(
        tt::LogTest,
        "TestMux successfully created kernel at core {} for direction {} link {} on device {}",
        logical_core_,
        static_cast<int>(direction_),
        link_idx_,
        test_device_ptr_->fabric_node_id_);
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

    // Get sync connection count from sync_connection_manager_
    size_t num_sync_connections =
        sync_connection_manager_.get_connection_count_for_core(sync_core, true);  // is_sender=true

    // Compile-time args
    std::vector<uint32_t> ct_args = {
        is_2D_routing_enabled,
        is_dynamic_routing_enabled,
        (uint32_t)num_sync_connections,                     /* num sync fabric connections */
        static_cast<uint32_t>(senders_.size() + 1),         /* num local sync cores (all senders + sync core) */
        sender_memory_map_->common.get_kernel_config_size() /* kernel config buffer size */
    };

    // Runtime args: memory map args, then sync fabric connection args
    std::vector<uint32_t> rt_args = sender_memory_map_->get_memory_map_args();

    // Add sync fabric connection args via sync_connection_manager_
    auto sync_connection_args =
        sync_connection_manager_.generate_connection_args_for_core(sync_core, true, this);  // is_sender=true
    rt_args.insert(rt_args.end(), sync_connection_args.begin(), sync_connection_args.end());

    // Local args (all the rest go to local args buffer)
    std::vector<uint32_t> local_args;

    // Expected sync value for global sync
    local_args.push_back(this->global_sync_val_);

    // Add sync routing args for each sync config
    for (size_t i = 0; i < sync_sender.global_sync_configs_.size(); ++i) {
        const auto& [sync_config, connection_key] = sync_sender.global_sync_configs_[i];

        // Get array index for this connection key from sync_connection_manager_
        uint32_t fabric_conn_idx = sync_connection_manager_.get_connection_array_index_for_key(
            sync_core, true, connection_key);  // is_sender=true

        // Add sync routing args (chip send type + routing info)
        auto sync_traffic_args = sync_config.get_args(true);
        log_debug(
            tt::LogTest,
            "fabric connection {} (dir={} link={}) has sync config src_node_id: {} dst_node_ids {} hops {} "
            "mcast_start_hops {} ",
            fabric_conn_idx,
            static_cast<int>(connection_key.direction),
            connection_key.link_idx,
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

        // Get connection count and generate all connection args via FabricConnectionManager
        size_t num_connections = connection_manager_.get_connection_count_for_core(core, true);  // is_sender=true

        // Compile-time args
        std::vector<uint32_t> ct_args = {
            is_2D_routing_enabled,
            is_dynamic_routing_enabled,
            (uint32_t)num_connections,                           /* num connections (from FabricConnectionManager) */
            sender.configs_.size(),                              /* num traffic configs */
            (uint32_t)benchmark_mode_,                           /* benchmark mode */
            (uint32_t)global_sync_,                              /* line sync enabled */
            num_local_sync_cores,                                /* num local sync cores */
            sender_memory_map_->common.get_kernel_config_size(), /* kernel config buffer size */
            any_traffic_needs_flow_control ? 1u : 0u             /* USE_MUX (informational only) */
        };

        // Runtime args with connection type information
        std::vector<uint32_t> rt_args = sender_memory_map_->get_memory_map_args();

        // Add all connection args via FabricConnectionManager
        auto connection_args =
            connection_manager_.generate_connection_args_for_core(core, true, this);  // is_sender=true
        rt_args.insert(rt_args.end(), connection_args.begin(), connection_args.end());

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
        // Query the array index for each traffic config's connection key
        for (const auto& [config, connection_key] : sender.configs_) {
            uint32_t array_idx =
                connection_manager_.get_connection_array_index_for_key(core, true, connection_key);  // is_sender=true
            TT_FATAL(
                array_idx != UINT32_MAX, "Failed to find connection array index for traffic config on core {}", core);
            local_args.push_back(array_idx);
        }

        // Add sender traffic config args (including credit management info)
        for (const auto& [config, _] : sender.configs_) {
            auto traffic_config_args = config.get_args(false);  // false = not a sync config
            local_args.insert(local_args.end(), traffic_config_args.begin(), traffic_config_args.end());
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

        // Get connection count and generate all connection args via FabricConnectionManager (for credit return)
        size_t num_connections = connection_manager_.get_connection_count_for_core(core, false);  // is_sender=false

        // Compile-time args
        std::vector<uint32_t> ct_args = {
            (uint32_t)num_connections,                             /* num fabric connections (for credit return) */
            receiver.configs_.size(),                              /* num traffic configs */
            benchmark_mode_ ? 1u : 0u,                             /* benchmark mode */
            receiver_memory_map_->common.get_kernel_config_size(), /* kernel config buffer size */
            any_traffic_needs_flow_control ? 1u : 0u               /* USE_MUX (flow control enabled) */
        };

        // Runtime args: memory map args + credit connection args (if flow control enabled)
        std::vector<uint32_t> rt_args = receiver_memory_map_->get_memory_map_args();

        // Add all connection args via FabricConnectionManager (for credit return)
        auto connection_args =
            connection_manager_.generate_connection_args_for_core(core, false, this);  // is_sender=false
        rt_args.insert(rt_args.end(), connection_args.begin(), connection_args.end());

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

    // Process fabric connections to determine mux requirements and assign channels
    connection_manager_.process();
    sync_connection_manager_.process();  // Process sync connections separately

    // Create mux kernels for connections that need them
    this->create_mux_kernels();

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

    // Register with connection manager if flow control enabled
    if (config.parameters.enable_flow_control) {
        // TODO: For now, register receivers with a default direction for credit return
        // This should be refined to determine the actual direction based on sender locations
        // For the initial implementation, we'll register with a placeholder direction
        // The connection manager will handle multiple clients on the same connection

        // Simple heuristic: register for all possible directions for now
        // This will be optimized once we have full sender-receiver coordination
        std::vector<RoutingDirection> possible_directions = {
            RoutingDirection::North, RoutingDirection::South, RoutingDirection::East, RoutingDirection::West};

        // For now, register with direction North, link_idx 0 as a placeholder
        // TODO: Implement proper direction determination based on traffic patterns
        connection_manager_.register_client(logical_core, RoutingDirection::North, 0, false);  // is_sender=false
    }

    this->receivers_.at(logical_core).add_config(config);
}

inline void TestDevice::add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config) {
    if (this->senders_.find(logical_core) == this->senders_.end()) {
        this->add_worker(TestWorkerType::SENDER, logical_core);
    }

    // Connection registration is now handled inside TestSender::add_config()
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
