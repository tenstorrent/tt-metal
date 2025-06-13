// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <string_view>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>

#include "impl/context/metal_context.hpp"

#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_interfaces.hpp"
#include "tt_fabric_test_common.hpp"

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

private:
    CoreCoord logical_core_;
    uint32_t worker_id_;
    std::string_view kernel_src_;
    TestDevice* test_device_ptr_;
};

struct TestSender : TestWorker {
public:
    TestSender(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void add_config(TestTrafficSenderConfig config);
    void connect_to_fabric_router();

    TestWorkerMemoryMap memory_map_;
    // for now assume that a worker can handle multiple test configs in the same direction
    std::vector<TestTrafficSenderConfig> configs_;
};

struct TestReceiverAllocator {
public:
    void init(uint32_t start_address, uint32_t end_address, uint32_t chunk_size);
    uint32_t allocate();

private:
    uint32_t start_address_;
    uint32_t end_address_;
    uint32_t total_size_;
    uint32_t chunk_size_;
    std::vector<uint32_t> available_addresses_;
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
    uint32_t allocate_address_for_sender();

    TestWorkerMemoryMap memory_map_;
    TestReceiverAllocator allocator_;
    bool is_shared_;
    std::vector<TestTrafficReceiverConfig> configs_;
};

struct TestDevice {
public:
    TestDevice(const MeshCoordinate& coord, IDeviceInfoProvider& device_info_provider, IRouteManager& route_manager);
    tt::tt_metal::Program& get_program_handle();
    const FabricNodeId& get_node_id();
    void reserve_core_for_worker(
        TestWorkerType worker_type, CoreCoord logical_core, std::optional<std::string_view> kernel_src);
    CoreCoord allocate_worker_core() const;
    std::vector<CoreCoord> get_available_worker_cores() const;
    uint32_t allocate_address_for_sender(CoreCoord receiver_core);
    void add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config);
    void add_receiver_traffic_config(CoreCoord logical_core, TestTrafficReceiverConfig config);
    void create_kernels();

private:
    void reserve_worker_core(CoreCoord logical_core);
    void create_sender_kernels();
    void create_receiver_kernels();

    MeshCoordinate coord_;
    IDeviceInfoProvider& device_info_provider_;
    IRouteManager& route_manager_;

    FabricNodeId fabric_node_id_ = FabricNodeId(MeshId{0}, 0);

    tt_metal::Program program_handle_;

    std::vector<CoreCoord> avaialble_worker_logical_cores_;

    // For now instead of moving to a new worker for the receiver every time, just pick one and exhaust it
    // if can no longer allocate, fetch a new core (as per policy)
    std::optional<CoreCoord> current_receiver_logical_core_;

    std::unordered_map<CoreCoord, TestSender> senders_;
    std::unordered_map<CoreCoord, TestReceiver> receivers_;

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

inline void TestSender::add_config(TestTrafficSenderConfig config) { this->configs_.push_back(config); }

inline void TestSender::connect_to_fabric_router() {}

/* *******************************
 * TestReceiverAllocator Methods *
 *********************************/
inline void TestReceiverAllocator::init(uint32_t start_address, uint32_t end_address, uint32_t chunk_size) {
    // TODO: sanity check on start and end addresses

    this->start_address_ = start_address;
    this->end_address_ = end_address;
    this->total_size_ = end_address - start_address;
    this->chunk_size_ = chunk_size;

    // for now initialize the addresses in a contiguous fashion
    uint32_t next_avaialable_address = this->start_address_;
    while (next_avaialable_address + this->chunk_size_ <= this->end_address_) {
        this->available_addresses_.push_back(next_avaialable_address);
        next_avaialable_address += this->chunk_size_;
    }
}

inline uint32_t TestReceiverAllocator::allocate() {
    // TODO: can have a policy for allocator as well. For now, just allocate the next available
    if (this->available_addresses_.empty()) {
        return 0;
    }

    uint32_t avaialble_address = this->available_addresses_.back();
    this->available_addresses_.pop_back();
    return avaialble_address;
}

/* **********************
 * TestReceiver Methods *
 ************************/
inline TestReceiver::TestReceiver(
    CoreCoord logical_core, TestDevice* test_device_ptr, bool is_shared, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src), is_shared_(is_shared) {
    // TODO: init mem map?
    // TODO: get these from the mem map
    uint32_t start_address = 0x30000;
    uint32_t end_address = 0x100000;
    uint32_t chunk_size = 0x10000;  // TODO: get this from the config/settings
    this->allocator_.init(start_address, end_address, chunk_size);
}

inline void TestReceiver::add_config(TestTrafficReceiverConfig config) { this->configs_.push_back(config); }

inline bool TestReceiver::is_shared_receiver() { return this->is_shared_; }

inline uint32_t TestReceiver::allocate_address_for_sender() { return this->allocator_.allocate(); }

/* ********************
 * TestDevice Methods *
 **********************/
inline void TestDevice::reserve_worker_core(CoreCoord logical_core) {
    if (this->senders_.find(logical_core) != this->senders_.end() ||
        this->receivers_.find(logical_core) != this->receivers_.end()) {
        log_fatal(tt::LogTest, "On chip coord: {}, requested core: {} is already reserved", this->coord_, logical_core);
        throw std::runtime_error("Failed to reserve worker core");
    }

    auto it = std::find(
        this->avaialble_worker_logical_cores_.begin(), this->avaialble_worker_logical_cores_.end(), logical_core);
    if (it == this->avaialble_worker_logical_cores_.end()) {
        log_fatal(tt::LogTest, "On chip coord: {}, requested core: {} not found", this->coord_, logical_core);
        throw std::runtime_error("Failed to reserve worker core");
    }
    this->avaialble_worker_logical_cores_.erase(it);
}

inline TestDevice::TestDevice(
    const MeshCoordinate& coord, IDeviceInfoProvider& device_info_provider, IRouteManager& route_manager) :
    coord_(coord), device_info_provider_(device_info_provider), route_manager_(route_manager) {
    program_handle_ = tt::tt_metal::CreateProgram();
    fabric_node_id_ = device_info_provider_.get_fabric_node_id(coord);
    const auto grid_size = this->device_info_provider_.get_worker_grid_size();
    for (auto i = 0; i < grid_size.x; i++) {
        for (auto j = 0; j < grid_size.y; j++) {
            this->avaialble_worker_logical_cores_.push_back(CoreCoord({i, j}));
        }
    }

    this->current_receiver_logical_core_ = std::nullopt;

    // TODO: init routers
}

inline tt::tt_metal::Program& TestDevice::get_program_handle() { return this->program_handle_; }

inline const FabricNodeId& TestDevice::get_node_id() { return this->fabric_node_id_; }

inline void TestDevice::reserve_core_for_worker(
    TestWorkerType worker_type, CoreCoord logical_core, std::optional<std::string_view> kernel_src) {
    this->reserve_worker_core(logical_core);

    if (worker_type == TestWorkerType::SENDER) {
        this->senders_.insert(std::make_pair(logical_core, TestSender(logical_core, this, kernel_src)));
    } else if (worker_type == TestWorkerType::RECEIVER) {
        bool is_shared_receiver = false;
        this->receivers_.insert(
            std::make_pair(logical_core, TestReceiver(logical_core, this, is_shared_receiver, kernel_src)));
    } else {
        log_fatal(tt::LogTest, "Unknown worker type for core reservation: {}", worker_type);
        throw std::runtime_error("Unknown worker type");
    }
}

inline CoreCoord TestDevice::allocate_worker_core() const {
    // currently pick from last -> can be configured as a policy (random, optimized etc)
    if (this->avaialble_worker_logical_cores_.empty()) {
        log_fatal(tt::LogTest, "On node: {}, no more worker cores avaialble", fabric_node_id_);
        throw std::runtime_error("Failed to allocate worker core");
    }

    CoreCoord worker_core = this->avaialble_worker_logical_cores_.back();
    return worker_core;
}

inline std::vector<CoreCoord> TestDevice::get_available_worker_cores() const {
    return this->avaialble_worker_logical_cores_;
}

inline uint32_t TestDevice::allocate_address_for_sender(CoreCoord receiver_core) {
    auto it = this->receivers_.find(receiver_core);
    if (it == this->receivers_.end()) {
        log_fatal(tt::LogTest, "On node: {}, for logical core: {}, no receiver found", fabric_node_id_, receiver_core);
        throw std::runtime_error("Failed to allocate address for sender");
    }

    uint32_t address = it->second.allocate_address_for_sender();
    if (address == 0) {
        log_fatal(
            tt::LogTest,
            "On node: {}, for logical core: {}, no space available for sender",
            fabric_node_id_,
            receiver_core);
        throw std::runtime_error("Failed to allocate address for sender");
    }

    return address;
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

        // get fabric connection args
        // for now just assume that there is only 1 recv per sender
        // TODO: move this elsewhere and fix the logic
        const auto& temp_config = sender.configs_[0];
        const auto dst_node_id = temp_config.dst_node_ids[0];

        const auto available_links = get_forwarding_link_indices(fabric_node_id_, dst_node_id);
        const auto link_idx = available_links[0];
        std::vector<uint32_t> fabric_connection_args;
        append_fabric_connection_rt_args(
            fabric_node_id_, dst_node_id, link_idx, this->program_handle_, core, fabric_connection_args);

        // TODO: handle this properly when adding configs for the sender
        std::vector<uint32_t> traffic_config_to_fabric_connection_args(sender.configs_.size(), 0);

        std::vector<uint32_t> traffic_config_args;
        for (const auto& config : sender.configs_) {
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

inline void TestDevice::add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config) {
    auto it = this->senders_.find(logical_core);
    if (it == this->senders_.end()) {
        log_fatal(tt::LogTest, "On node: {}, for logical core: {}, no sender found", fabric_node_id_, logical_core);
        throw std::runtime_error("Failed to add traffic config for sender");
    }
    it->second.add_config(config);
}

inline void TestDevice::add_receiver_traffic_config(CoreCoord logical_core, TestTrafficReceiverConfig config) {
    auto it = this->receivers_.find(logical_core);
    if (it == this->receivers_.end()) {
        log_fatal(tt::LogTest, "On node: {}, for logical core: {}, no receiver found", fabric_node_id_, logical_core);
        throw std::runtime_error("Failed to add traffic config for receiver");
    }
    it->second.add_config(config);
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
