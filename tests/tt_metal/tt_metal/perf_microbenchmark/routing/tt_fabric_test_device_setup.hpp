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
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>

#include "impl/context/metal_context.hpp"

#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common.hpp"

const std::string default_sender_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_sender.cpp";
const std::string default_receiver_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_receiver.cpp";

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

namespace tt::tt_fabric {
namespace fabric_tests {

// forward declaration
struct TestDevice;

struct TestFabricBuilderConfigs {
private:
    // collection of all the builder configs we may need
    // TODO: maybe supply a hash function to avoid nesting
    std::unordered_map<
        Topology,
        std::unordered_map<
            tt::tt_metal::FabricConfig,
            std::unordered_map<bool, std::unique_ptr<FabricEriscDatamoverConfig>>>>
        builder_configs_{};

    TestFabricBuilderConfigs() {}

public:
    TestFabricBuilderConfigs(const TestFabricBuilderConfigs&) = delete;
    TestFabricBuilderConfigs& operator=(const TestFabricBuilderConfigs&) = delete;

    static TestFabricBuilderConfigs& get_instance() {
        static TestFabricBuilderConfigs instance;
        return instance;
    }

    size_t get_packet_header_size_bytes(Topology topology, tt::tt_metal::FabricConfig config) const;
    size_t get_max_payload_size_bytes(Topology topology) const;
    FabricEriscDatamoverConfig& get_builder_config(
        Topology topology, tt::tt_metal::FabricConfig config, bool is_dateline);
};

struct TestFabricBuilder {
private:
    // hack to cache the builder
    // TODO: find a better way?
    std::vector<FabricEriscDatamoverBuilder> builder_;

public:
    void build();
    void connect_to_downstream_builder();
};

// per router config
struct TestFabricRouter {
private:
    chan_id_t my_chan_;
    chip_id_t remote_chip_id_;
    std::vector<chan_id_t> downstream_router_chans_;
    bool is_on_dateline_connection_;
    bool is_loopback_router_;

    TestFabricBuilder fabric_builder_;

    // my_direction_ / eth_direction_;
    // my_link_idx_;
    TestDevice* test_device_ptr_;

public:
    TestFabricRouter(chan_id_t chan, chip_id_t remote_chip_id);
    void add_downstream_router(chan_id_t downstream_router_chan);
    void set_dateline_flag();
    void set_loopback_flag();
    void set_remote_chip();
    void connect_to_downstream_routers();
};

struct TestDeviceFabricRouters {
private:
    std::unordered_map<RoutingDirection, std::vector<chan_id_t>> direction_and_chans_{};
    // chip_neighbors_in_directions_;
    std::unordered_map<chan_id_t, TestFabricRouter> routers_{};
    chan_id_t master_router_chan_{};
    std::optional<Topology> topology_;
    TestDevice* test_device_ptr_;

public:
    TestDeviceFabricRouters();
    void set_topology(Topology topology);
    void add_active_router();
    std::vector<RoutingDirection> get_downstream_directions(RoutingDirection direction, Topology topology);
    void connect_routing_planes();
    void setup_topology();
    void build_fabric();
    void wait_for_router_sync();
    void notify_routers();
    void terminate_routers();
};

// for now keep the memory map same for both senders and receivers
struct TestWorkerMemoryMap {
    uint32_t worker_usable_address;
};

enum class TestWorkerType : uint8_t { SENDER, RECEIVER };

struct TestWorker {
public:
    TestWorker(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void set_kernel_src(const std::string_view& kernel_src);
    void create_kernel(
        const MeshCoordinate& device_coord,
        const std::vector<uint32_t>& ct_args,
        const std::vector<uint32_t>& rt_args,
        const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear) const;
    void collect_results();
    void validate_results();
    void dump_results();

protected:
    CoreCoord logical_core_;
    uint32_t worker_id_;
    std::string_view kernel_src_;
    TestDevice* test_device_ptr_;
};

struct TestSender : TestWorker {
public:
    TestSender(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void add_config(const TestTrafficSenderConfig& config);
    void connect_to_fabric_router();

    TestWorkerMemoryMap memory_map_;

    // stores traffic config and the correspoding fabric_connection idx to use
    std::vector<std::pair<TestTrafficSenderConfig, uint32_t>> configs_;

    // book-keeping for all the fabric connections needed for this sender
    // [RoutingDirection][link_idx]
    std::vector<std::pair<RoutingDirection, uint32_t>> fabric_connections_;
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

    TestWorkerMemoryMap memory_map_;
    bool is_shared_;
    std::vector<TestTrafficReceiverConfig> configs_;
};

struct TestDevice {
public:
    TestDevice(
        const MeshCoordinate& coord,
        std::shared_ptr<IDeviceInfoProvider> device_info_provider,
        std::shared_ptr<IRouteManager> route_manager);
    tt::tt_metal::Program& get_program_handle();
    const FabricNodeId& get_node_id();
    uint32_t add_fabric_connection(RoutingDirection direction, const std::vector<uint32_t>& link_indices);
    void add_sender_traffic_config(CoreCoord logical_core, const TestTrafficSenderConfig& config);
    void add_receiver_traffic_config(CoreCoord logical_core, const TestTrafficReceiverConfig& config);
    void create_kernels();
    RoutingDirection get_forwarding_direction(const std::unordered_map<RoutingDirection, uint32_t>& hops) const;
    std::vector<uint32_t> get_forwarding_link_indices_in_direction(const RoutingDirection& direction) const;

private:
    void add_worker(TestWorkerType worker_type, CoreCoord logical_core);
    std::vector<uint32_t> get_fabric_connection_args(CoreCoord core, RoutingDirection direction, uint32_t link_idx);
    void create_sender_kernels();
    void create_receiver_kernels();

    MeshCoordinate coord_;
    std::shared_ptr<IDeviceInfoProvider> device_info_provider_;
    std::shared_ptr<IRouteManager> route_manager_;

    FabricNodeId fabric_node_id_ = FabricNodeId(MeshId{0}, 0);

    tt_metal::Program program_handle_;

    std::unordered_map<CoreCoord, TestSender> senders_;
    std::unordered_map<CoreCoord, TestReceiver> receivers_;

    std::unordered_map<RoutingDirection, std::set<uint32_t>> used_fabric_connections_{};

    // controller?

    TestDeviceFabricRouters fabric_routers_;
};

/* **********************************
 * TestFabricBuilderConfigs Methods *
 ************************************/
inline size_t TestFabricBuilderConfigs::get_packet_header_size_bytes(
    Topology topology, tt::tt_metal::FabricConfig config) const {
    if (topology == Topology::Mesh) {
        return (config == tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC) ? sizeof(MeshPacketHeader)
                                                                         : sizeof(LowLatencyMeshPacketHeader);
    } else {
        return sizeof(PacketHeader);
    }
}

inline size_t TestFabricBuilderConfigs::get_max_payload_size_bytes(Topology topology) const {
    if (topology == Topology::Mesh) {
        return FabricEriscDatamoverBuilder::default_mesh_packet_payload_size_bytes;
    } else {
        return FabricEriscDatamoverBuilder::default_packet_payload_size_bytes;
    }
}

inline FabricEriscDatamoverConfig& TestFabricBuilderConfigs::get_builder_config(
    Topology topology, tt::tt_metal::FabricConfig config, bool is_dateline) {
    bool found = true;
    const auto topology_it = this->builder_configs_.find(topology);
    if (topology_it == this->builder_configs_.end()) {
        found = false;
    }

    if (found) {
        const auto config_it = this->builder_configs_.at(topology).find(config);
        if (config_it == this->builder_configs_.at(topology).end()) {
            found = false;
        }

        if (found) {
            const auto dateline_it = this->builder_configs_.at(topology).at(config).find(is_dateline);
            if (dateline_it == this->builder_configs_.at(topology).at(config).end()) {
                found = false;
            }
        }
    }

    if (!found) {
        const auto packet_header_size_bytes = this->get_packet_header_size_bytes(topology, config);
        const auto max_payload_size_bytes = this->get_max_payload_size_bytes(topology);
        const auto channel_buffer_size_bytes = packet_header_size_bytes + max_payload_size_bytes;
        /*
        this->builder_configs_[topology][config][is_dateline] =
            std::make_unique<FabricEriscDatamoverConfig>(channel_buffer_size_bytes, topology, is_dateline);
    */
    }

    return *this->builder_configs_.at(topology).at(config).at(is_dateline).get();
}

/* **************************
 * TestFabricRouter Methods *
 ****************************/
inline TestFabricRouter::TestFabricRouter(chan_id_t chan, chip_id_t remote_chip_id) {
    this->my_chan_ = chan;
    this->remote_chip_id_ = remote_chip_id;
}

inline void TestFabricRouter::add_downstream_router(chan_id_t downstream_router_chan) {
    this->downstream_router_chans_.push_back(downstream_router_chan);
}

inline void TestFabricRouter::set_dateline_flag() { this->is_on_dateline_connection_ = true; }

inline void TestFabricRouter::set_loopback_flag() {
    this->is_loopback_router_ = true;

    // for loopback mode, set the downstream router chan to be itself
    if (!this->downstream_router_chans_.empty()) {
        log_fatal(
            tt::LogTest,
            "For node: {}, chan: {}, downstream router is already set to: {}, but tried to set as loopback",
            this->test_device_ptr_->get_node_id(),
            this->my_chan_,
            this->downstream_router_chans_);
        throw std::runtime_error("Tried to set loopback when downstream router chan is already set");
    }
    this->downstream_router_chans_ = {this->my_chan_};
}

inline void TestFabricRouter::set_remote_chip() { this->remote_chip_id_ = true; }

/*
    void build(tt::tt_metal::IDevice* device_handle, tt_metal::Program& program_handle) {
        this->builder_ = FabricEriscDatamoverBuilder::build(
            device_handle,
            program_handle,
            fabric_eth_chan_to_logical_core.at(eth_chan),
            physical_chip_id,
            remote_chip_id,
            edm_config,
            true,
            false,
            is_dateline);
        this->builder_.set_wait_for_host_signal(true);
    }
*/
inline void TestFabricRouter::connect_to_downstream_routers() {
    // this->builder_.connect_to_downstream_edm(downstream_router.builder_);
}

/* *********************************
 * TestDeviceFabricRouters Methods *
 ***********************************/
inline TestDeviceFabricRouters::TestDeviceFabricRouters() {
    // TODO: take in the control plane ptr as a part of setup
    // since we may be setting up a custom control plane

    // TODO update chip neighbors
}

inline void TestDeviceFabricRouters::set_topology(Topology topology) { this->topology_ = topology; }

inline void TestDeviceFabricRouters::add_active_router() {}

inline std::vector<RoutingDirection> TestDeviceFabricRouters::get_downstream_directions(
    RoutingDirection direction, Topology topology) {
    if (topology == Topology::Linear) {
        switch (direction) {
            case RoutingDirection::N: return {RoutingDirection::S};
            case RoutingDirection::S: return {RoutingDirection::N};
            case RoutingDirection::E: return {RoutingDirection::W};
            case RoutingDirection::W: return {RoutingDirection::E};
            default: throw std::runtime_error("Invalid direction for quering forwarding directions");
        }
    } else if (topology == Topology::Mesh) {
        // return all the directions except itself
        std::vector<RoutingDirection> directions = FabricContext::routing_directions;
        directions.erase(std::remove(directions.begin(), directions.end(), direction), directions.end());
        return directions;
    } else {
        // throw error?
    }

    return {};
}

// blanket method/preset mode method
inline void TestDeviceFabricRouters::connect_routing_planes() {
    for (const auto& [direction, router_chans] : this->direction_and_chans_) {
        for (auto routing_plane_id = 0; routing_plane_id < router_chans.size(); routing_plane_id++) {
            const auto channel = router_chans[routing_plane_id];
            if (this->routers_.find(channel) == this->routers_.end()) {
                // no router is active for this channel
                continue;
            }

            const auto& downstream_directions = get_downstream_directions(direction, this->topology_.value());
            for (const auto& downstream_dir : downstream_directions) {
                auto it = this->direction_and_chans_.find(downstream_dir);
                if (it == this->direction_and_chans_.end()) {
                    continue;
                }

                const auto& downstream_channels = it->second;
                if (routing_plane_id >= downstream_channels.size()) {
                    // no downstream channel for this routing plane
                    continue;
                }

                this->routers_.at(channel).add_downstream_router(downstream_channels[routing_plane_id]);
            }
        }
    }
}

inline void TestDeviceFabricRouters::setup_topology() {
    // TODO: add checks for some of the router configs that may have already been setup

    if (!this->topology_.has_value()) {
        return;
    }

    this->connect_routing_planes();
}

inline void TestDeviceFabricRouters::build_fabric() {
    setup_topology();

    // by now we assume that each individual router has been setup

    // first invoke individual builders
    for (const auto& [_, router] : this->routers_) {
        // router.build()
    }

    // connect downstream routers
    for (const auto& [_, router] : this->routers_) {
        // router.connect_to_downstream_routers();
    }

    // create and compile kernels
}

inline void TestDeviceFabricRouters::wait_for_router_sync() {}

inline void TestDeviceFabricRouters::notify_routers() {}

inline void TestDeviceFabricRouters::terminate_routers() {}

/* ********************
 * TestWorker Methods *
 **********************/
inline TestWorker::TestWorker(
    CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    logical_core_(logical_core), test_device_ptr_(test_device_ptr) {
    if (kernel_src.has_value()) {
        this->kernel_src_ = kernel_src.value();
    }

    // populate worker id
}

inline void TestWorker::set_kernel_src(const std::string_view& kernel_src) { this->kernel_src_ = kernel_src; }

inline void TestWorker::create_kernel(
    const MeshCoordinate& device_coord,
    const std::vector<uint32_t>& ct_args,
    const std::vector<uint32_t>& rt_args,
    const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear) const {
    auto kernel_handle = tt::tt_metal::CreateKernel(
        this->test_device_ptr_->get_program_handle(),
        std::string(this->kernel_src_),
        {this->logical_core_},
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = ct_args,
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
    tt::tt_metal::SetRuntimeArgs(
        this->test_device_ptr_->get_program_handle(), kernel_handle, this->logical_core_, rt_args);

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

inline void TestSender::add_config(const TestTrafficSenderConfig& config) {
    std::optional<RoutingDirection> outgoing_direction;
    std::vector<uint32_t> outgoing_link_indices;

    // either we will have hops specified or the dest node id
    if (true /* config.hops.has_value() */) {
        outgoing_direction = this->test_device_ptr_->get_forwarding_direction(config.hops);
        outgoing_link_indices =
            this->test_device_ptr_->get_forwarding_link_indices_in_direction(outgoing_direction.value());
    } else {
        // TODO: figure out if we need this
        /*
        const auto dst_node_id = config.dst_node_ids[0];
        outgoing_direction =
            this->test_device_ptr_->route_manager_->get_forwarding_direction(my_node_id, dst_node_id);
        TT_FATAL(outgoing_direction.has_value(), "No forwarding direction found for {} from {}", dst_node_id,
        my_node_id); outgoing_link_indices =
            this->test_device_ptr_->route_manager_->get_forwarding_link_indices_in_direction(
                my_node_id, dst_node_id, outgoing_direction.value());
        TT_FATAL(!outgoing_link_indices.empty(), "No forwarding link indices found for {} from {}", dst_node_id,
        my_node_id);
        */
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
            this->test_device_ptr_->add_fabric_connection(outgoing_direction.value(), outgoing_link_indices);
        this->fabric_connections_.push_back({outgoing_direction.value(), new_link_idx});
        fabric_connection_idx = this->fabric_connections_.size() - 1;
    }

    this->configs_.push_back(std::make_pair(std::move(config), fabric_connection_idx.value()));
}

inline void TestSender::connect_to_fabric_router() {}

/* **********************
 * TestReceiver Methods *
 ************************/
inline TestReceiver::TestReceiver(
    CoreCoord logical_core, TestDevice* test_device_ptr, bool is_shared, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src), is_shared_(is_shared) {
    // TODO: init mem map?
}

inline void TestReceiver::add_config(TestTrafficReceiverConfig config) { this->configs_.push_back(config); }

inline bool TestReceiver::is_shared_receiver() { return this->is_shared_; }

/* ********************
 * TestDevice Methods *
 **********************/
inline TestDevice::TestDevice(
    const MeshCoordinate& coord,
    std::shared_ptr<IDeviceInfoProvider> device_info_provider,
    std::shared_ptr<IRouteManager> route_manager) :
    coord_(coord), device_info_provider_(std::move(device_info_provider)), route_manager_(std::move(route_manager)) {
    program_handle_ = tt::tt_metal::CreateProgram();
    fabric_node_id_ = device_info_provider_->get_fabric_node_id(coord);

    // TODO: init routers
}

inline tt::tt_metal::Program& TestDevice::get_program_handle() { return this->program_handle_; }

inline const FabricNodeId& TestDevice::get_node_id() { return this->fabric_node_id_; }

inline uint32_t TestDevice::add_fabric_connection(
    RoutingDirection direction, const std::vector<uint32_t>& link_indices) {
    // if all the connections have already been used by another worker, then its an error
    // else try to add whichever is not used
    if (this->used_fabric_connections_.count(direction) == 0) {
        this->used_fabric_connections_[direction] = {};
    }

    const auto& used_link_indices = this->used_fabric_connections_.at(direction);
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

    this->used_fabric_connections_[direction].insert(unused_link_idx.value());
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
        this->senders_.insert(std::make_pair(logical_core, TestSender(logical_core, this, default_sender_kernel_src)));
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
        this->receivers_.insert(std::make_pair(
            logical_core, TestReceiver(logical_core, this, is_shared_receiver, default_receiver_kernel_src)));
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

inline void TestDevice::create_sender_kernels() {
    // TODO: fetch these dynamically
    const bool is_2d_fabric = false;
    const bool use_dynamic_routing = false;

    for (const auto& [core, sender] : this->senders_) {
        // get ct args
        // TODO: fix these- number of fabric connections, mappings etc
        std::vector<uint32_t> ct_args = {
            is_2d_fabric,
            use_dynamic_routing,
            1, /* num fabric connections */
            sender.configs_.size(),
            0 /* benchmark mode */};

        // memory map args
        // TODO: move to the right place
        uint32_t packet_header_region_base = 0x30000;
        uint32_t payload_buffer_region_base = 0x40000;
        uint32_t highest_usable_address = 0x100000;
        std::vector<uint32_t> memory_allocator_args = {
            packet_header_region_base, payload_buffer_region_base, highest_usable_address};

        std::vector<uint32_t> fabric_connection_args;
        for (const auto& [direction, link_idx] : sender.fabric_connections_) {
            const auto& args = get_fabric_connection_args(core, direction, link_idx);
            fabric_connection_args.insert(fabric_connection_args.end(), args.begin(), args.end());
        }

        // TODO: handle this properly when adding configs for the sender
        std::vector<uint32_t> traffic_config_to_fabric_connection_args;
        traffic_config_to_fabric_connection_args.reserve(sender.configs_.size());
        for (const auto& [_, fabric_connection_idx] : sender.configs_) {
            traffic_config_to_fabric_connection_args.push_back(fabric_connection_idx);
        }

        std::vector<uint32_t> traffic_config_args;
        for (const auto& [config, _] : sender.configs_) {
            const auto traffic_args = config.get_args();
            traffic_config_args.insert(traffic_config_args.end(), traffic_args.begin(), traffic_args.end());
        }

        std::vector<uint32_t> rt_args;
        rt_args.insert(rt_args.end(), memory_allocator_args.begin(), memory_allocator_args.end());
        rt_args.insert(rt_args.end(), fabric_connection_args.begin(), fabric_connection_args.end());
        rt_args.insert(
            rt_args.end(),
            traffic_config_to_fabric_connection_args.begin(),
            traffic_config_to_fabric_connection_args.end());
        rt_args.insert(rt_args.end(), traffic_config_args.begin(), traffic_config_args.end());

        // create kernel
        sender.create_kernel(coord_, ct_args, rt_args, {});
        log_info(tt::LogTest, "created sender kernel on core: {}", core);
    }
}

inline void TestDevice::create_receiver_kernels() {
    for (const auto& [core, receiver] : this->receivers_) {
        // get ct args
        // TODO: fix these
        std::vector<uint32_t> ct_args = {receiver.configs_.size(), 0 /* benchmark mode */};

        std::vector<uint32_t> traffic_config_args;
        for (const auto& config : receiver.configs_) {
            const auto traffic_args = config.get_args();
            traffic_config_args.insert(traffic_config_args.end(), traffic_args.begin(), traffic_args.end());
        }

        std::vector<uint32_t> rt_args;
        rt_args.insert(rt_args.end(), traffic_config_args.begin(), traffic_config_args.end());

        receiver.create_kernel(coord_, ct_args, rt_args, {});
        log_info(tt::LogTest, "created receiver kernel on core: {}", core);
    }
}

inline void TestDevice::create_kernels() {
    log_info(tt::LogTest, "creating kernels on node: {}", fabric_node_id_);
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

inline void TestDevice::add_sender_traffic_config(CoreCoord logical_core, const TestTrafficSenderConfig& config) {
    if (this->senders_.find(logical_core) == this->senders_.end()) {
        this->add_worker(TestWorkerType::SENDER, logical_core);
    }
    this->senders_.at(logical_core).add_config(config);
}

inline RoutingDirection TestDevice::get_forwarding_direction(
    const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
    return this->route_manager_->get_forwarding_direction(hops);
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

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
