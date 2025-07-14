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
#include "tt_fabric_test_memory_map.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

const std::string default_sender_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_sender.cpp";
const std::string default_receiver_kernel_src =
    "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_test_receiver.cpp";

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
        const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear) const;
    void collect_results();
    virtual bool validate_results(std::vector<uint32_t>& data) const = 0;
    void dump_results();

protected:
    CoreCoord logical_core_;
    uint32_t worker_id_;
    std::string kernel_src_;
    TestDevice* test_device_ptr_;
};

struct TestSender : TestWorker {
public:
    TestSender(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src);
    void add_config(TestTrafficSenderConfig config);
    void connect_to_fabric_router();
    bool validate_results(std::vector<uint32_t>& data) const override;

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
    const FabricNodeId& get_node_id();
    uint32_t add_fabric_connection(RoutingDirection direction, const std::vector<uint32_t>& link_indices);
    void add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config);
    void add_receiver_traffic_config(CoreCoord logical_core, const TestTrafficReceiverConfig& config);
    void create_kernels();
    RoutingDirection get_forwarding_direction(const std::unordered_map<RoutingDirection, uint32_t>& hops) const;
    std::vector<uint32_t> get_forwarding_link_indices_in_direction(const RoutingDirection& direction) const;
    void validate_results() const;

private:
    void add_worker(TestWorkerType worker_type, CoreCoord logical_core);
    std::vector<uint32_t> get_fabric_connection_args(CoreCoord core, RoutingDirection direction, uint32_t link_idx);
    void create_sender_kernels();
    void create_receiver_kernels();
    void validate_sender_results() const;
    void validate_receiver_results() const;

    MeshCoordinate coord_;
    std::shared_ptr<IDeviceInfoProvider> device_info_provider_;
    std::shared_ptr<IRouteManager> route_manager_;
    const SenderMemoryMap* sender_memory_map_;
    const ReceiverMemoryMap* receiver_memory_map_;

    FabricNodeId fabric_node_id_ = FabricNodeId(MeshId{0}, 0);

    tt_metal::Program program_handle_;

    std::unordered_map<CoreCoord, TestSender> senders_;
    std::unordered_map<CoreCoord, TestReceiver> receivers_;

    std::unordered_map<RoutingDirection, std::set<uint32_t>> used_fabric_connections_{};

    // controller?
};

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

inline void TestSender::add_config(TestTrafficSenderConfig config) {
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
        this->fabric_connections_.emplace_back(outgoing_direction.value(), new_link_idx);
        fabric_connection_idx = this->fabric_connections_.size() - 1;
    }

    this->configs_.emplace_back(std::move(config), fabric_connection_idx.value());
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

inline void TestDevice::create_sender_kernels() {
    // TODO: fetch these dynamically
    const bool is_2d_fabric = this->device_info_provider_->is_2d_fabric();
    const bool use_dynamic_routing = this->device_info_provider_->use_dynamic_routing();

    for (const auto& [core, sender] : this->senders_) {
        // get ct args
        // TODO: fix these- number of fabric connections, mappings etc
        std::vector<uint32_t> ct_args = {
            is_2d_fabric,
            use_dynamic_routing,
            sender.fabric_connections_.size(), /* num fabric connections */
            sender.configs_.size(),
            0 /* benchmark mode */};

        // Get memory layout from sender memory map
        TT_FATAL(sender_memory_map_ != nullptr, "Sender memory map is required for creating sender kernels");
        TT_FATAL(sender_memory_map_->is_valid(), "Sender memory map is invalid");

        // Get all memory map arguments in one call
        std::vector<uint32_t> memory_map_args = sender_memory_map_->get_memory_map_args();

        std::vector<uint32_t> fabric_connection_args;
        if (!sender.fabric_connections_.empty()) {
            const auto& first_args = get_fabric_connection_args(
                core, sender.fabric_connections_[0].first, sender.fabric_connections_[0].second);
            fabric_connection_args.reserve(sender.fabric_connections_.size() * first_args.size());
            fabric_connection_args.insert(fabric_connection_args.end(), first_args.begin(), first_args.end());

            for (size_t i = 1; i < sender.fabric_connections_.size(); ++i) {
                const auto& args = get_fabric_connection_args(
                    core, sender.fabric_connections_[i].first, sender.fabric_connections_[i].second);
                fabric_connection_args.insert(fabric_connection_args.end(), args.begin(), args.end());
            }
        }

        // TODO: handle this properly when adding configs for the sender
        std::vector<uint32_t> traffic_config_to_fabric_connection_args;
        traffic_config_to_fabric_connection_args.reserve(sender.configs_.size());
        for (const auto& [_, fabric_connection_idx] : sender.configs_) {
            traffic_config_to_fabric_connection_args.push_back(fabric_connection_idx);
        }

        std::vector<uint32_t> traffic_config_args;
        if (!sender.configs_.empty()) {
            // Estimate total size based on first config to reduce reallocations
            const auto first_traffic_args = sender.configs_[0].first.get_args();
            traffic_config_args.reserve(sender.configs_.size() * first_traffic_args.size());
            traffic_config_args.insert(traffic_config_args.end(), first_traffic_args.begin(), first_traffic_args.end());

            for (size_t i = 1; i < sender.configs_.size(); ++i) {
                const auto traffic_args = sender.configs_[i].first.get_args();
                traffic_config_args.insert(traffic_config_args.end(), traffic_args.begin(), traffic_args.end());
            }
        }

        // Pre-calculate total rt_args size to avoid reallocations
        const size_t total_rt_args_size = memory_map_args.size() + fabric_connection_args.size() +
                                          traffic_config_to_fabric_connection_args.size() + traffic_config_args.size();

        std::vector<uint32_t> rt_args;
        rt_args.reserve(total_rt_args_size);
        rt_args.insert(rt_args.end(), memory_map_args.begin(), memory_map_args.end());
        rt_args.insert(rt_args.end(), fabric_connection_args.begin(), fabric_connection_args.end());
        rt_args.insert(
            rt_args.end(),
            traffic_config_to_fabric_connection_args.begin(),
            traffic_config_to_fabric_connection_args.end());
        rt_args.insert(rt_args.end(), traffic_config_args.begin(), traffic_config_args.end());

        // create kernel
        sender.create_kernel(coord_, std::move(ct_args), std::move(rt_args), {});
        log_info(tt::LogTest, "created sender kernel on core: {}", core);
    }
}

inline void TestDevice::create_receiver_kernels() {
    for (const auto& [core, receiver] : this->receivers_) {
        // get ct args
        // TODO: fix these
        std::vector<uint32_t> ct_args = {receiver.configs_.size(), 0 /* benchmark mode */};

        // Get memory layout from receiver memory map
        TT_FATAL(receiver_memory_map_ != nullptr, "Receiver memory map is required for creating receiver kernels");
        TT_FATAL(receiver_memory_map_->is_valid(), "Receiver memory map is invalid");

        // Get all memory map arguments in one call
        std::vector<uint32_t> memory_map_args = receiver_memory_map_->get_memory_map_args();

        std::vector<uint32_t> traffic_config_args;
        if (!receiver.configs_.empty()) {
            // Estimate total size based on first config to reduce reallocations
            const auto first_traffic_args = receiver.configs_[0].get_args();
            traffic_config_args.reserve(receiver.configs_.size() * first_traffic_args.size());
            traffic_config_args.insert(traffic_config_args.end(), first_traffic_args.begin(), first_traffic_args.end());

            for (size_t i = 1; i < receiver.configs_.size(); ++i) {
                const auto traffic_args = receiver.configs_[i].get_args();
                traffic_config_args.insert(traffic_config_args.end(), traffic_args.begin(), traffic_args.end());
            }
        }

        // Pre-calculate total rt_args size to avoid reallocations
        const size_t total_rt_args_size = memory_map_args.size() + traffic_config_args.size();

        std::vector<uint32_t> rt_args;
        rt_args.reserve(total_rt_args_size);
        rt_args.insert(rt_args.end(), memory_map_args.begin(), memory_map_args.end());
        rt_args.insert(rt_args.end(), traffic_config_args.begin(), traffic_config_args.end());

        receiver.create_kernel(coord_, std::move(ct_args), std::move(rt_args), {});
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

inline void TestDevice::add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config) {
    if (this->senders_.find(logical_core) == this->senders_.end()) {
        this->add_worker(TestWorkerType::SENDER, logical_core);
    }
    this->senders_.at(logical_core).add_config(std::move(config));
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

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
