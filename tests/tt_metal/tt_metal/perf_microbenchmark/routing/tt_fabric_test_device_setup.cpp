// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_fabric_test_device_setup.hpp"

namespace tt::tt_fabric::fabric_tests {

// ====================================
// FabricConnectionManager Implementation
// ====================================

void FabricConnectionManager::register_client(
    const CoreCoord& core, RoutingDirection direction, uint32_t link_idx, TestWorkerType worker_type) {
    ConnectionKey key = {direction, link_idx};
    auto& conn = connections_[key];

    // Store worker type for this core (for channel assignment later)
    conn.core_worker_types[core] = worker_type;

    // Add to appropriate set based on worker type
    // Each worker type has its own set for proper tracking
    switch (worker_type) {
        case TestWorkerType::SENDER:
            conn.sender_cores.insert(core);
            sender_core_to_keys_[core].insert(key);
            break;
        case TestWorkerType::RECEIVER:
            conn.receiver_cores.insert(core);
            receiver_core_to_keys_[core].insert(key);
            break;
        case TestWorkerType::SYNC:
            conn.sync_cores.insert(core);
            sync_core_to_keys_[core].insert(key);
            break;
        case TestWorkerType::MUX:
            TT_FATAL(false, "MUX should not be registered as a client via register_client()");
            break;
    }
}

void FabricConnectionManager::process(
    LocalDeviceCoreAllocator& local_alloc,
    TestDevice* test_device_ptr,
    const std::shared_ptr<IDeviceInfoProvider>& device_info_provider) {
    for (auto& [key, conn] : connections_) {
        // Mux is needed if more than 1 client (any type) uses this link
        size_t total_clients = conn.sender_cores.size() + conn.receiver_cores.size() + conn.sync_cores.size();
        conn.needs_mux = total_clients > 1;

        if (conn.needs_mux) {
            assign_and_validate_channels(conn, key);

            // Allocate mux core on-demand from local allocator
            auto mux_core = local_alloc.allocate_core();
            TT_FATAL(
                mux_core.has_value(),
                "No pristine cores available for mux on device {} for connection dir={} link={}. "
                "Consider optimizing core placement or reducing worker cores.",
                test_device_ptr->get_node_id(),
                key.direction,
                key.link_idx);

            // Store the connection key -> mux core mapping
            mux_cores_[key] = mux_core.value();
            mux_core_to_key_[mux_core.value()] = key;

            // mux config shouldnt exist already (one config per connection/mux)
            TT_FATAL(
                !mux_configs_.contains(mux_core.value()),
                "Mux config already exists for mux core {}",
                mux_core.value());

            // Create and store mux config for this connection key
            // Count channels by type using worker types
            uint8_t num_full_size_channels = 0;
            uint8_t num_header_only_channels = 0;
            for (const auto& [core, worker_type] : conn.core_worker_types) {
                if (get_required_channel_type(worker_type) == FabricMuxChannelType::FULL_SIZE_CHANNEL) {
                    num_full_size_channels++;
                } else {
                    num_header_only_channels++;
                }
            }

            const uint8_t num_buffers_full_size_channel = BUFFERS_PER_CHANNEL;
            const uint8_t num_buffers_header_only_channel = BUFFERS_PER_CHANNEL;
            const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
            const size_t mux_base_l1_address = device_info_provider->get_l1_unreserved_base();

            // Create the mux config
            mux_configs_.emplace(
                mux_core.value(),
                std::make_unique<FabricMuxConfig>(
                    num_full_size_channels,
                    num_header_only_channels,
                    num_buffers_full_size_channel,
                    num_buffers_header_only_channel,
                    buffer_size_bytes_full_size_channel,
                    mux_base_l1_address));

            // Track all mux client cores
            for (const auto& [client_core, _] : conn.channel_map) {
                all_mux_client_cores_.insert(client_core);
            }

            test_device_ptr->add_mux_worker_config(mux_core.value(), mux_configs_.at(mux_core.value()).get(), key);
        }
    }

    // Designate first mux client (in deterministic order) as global termination master
    if (!all_mux_client_cores_.empty()) {
        global_termination_master_ = *all_mux_client_cores_.begin();
    }
}

std::unordered_map<RoutingDirection, std::set<uint32_t>> FabricConnectionManager::get_used_fabric_links() const {
    std::unordered_map<RoutingDirection, std::set<uint32_t>> result;
    for (const auto& [key, conn] : connections_) {
        result[key.direction].insert(key.link_idx);
    }
    return result;
}

std::vector<ConnectionKey> FabricConnectionManager::get_connection_keys_for_core(
    const CoreCoord& core, TestWorkerType worker_type) const {
    const auto* map = [&]() -> const std::unordered_map<CoreCoord, std::set<ConnectionKey>>* {
        switch (worker_type) {
            case TestWorkerType::SENDER: return &sender_core_to_keys_;
            case TestWorkerType::RECEIVER: return &receiver_core_to_keys_;
            case TestWorkerType::SYNC: return &sync_core_to_keys_;
            case TestWorkerType::MUX:
                TT_FATAL(false, "MUX should not be registered as a client via get_connection_keys_for_core()");
            default: TT_FATAL(false, "Invalid worker type: {}", static_cast<int>(worker_type));
        }
        return nullptr;
    }();

    auto it = map->find(core);
    if (it != map->end()) {
        return std::vector<ConnectionKey>(it->second.begin(), it->second.end());
    }
    return {};
}

size_t FabricConnectionManager::get_connection_count_for_core(const CoreCoord& core, TestWorkerType worker_type) const {
    auto keys = get_connection_keys_for_core(core, worker_type);
    return keys.size();
}

uint32_t FabricConnectionManager::get_connection_array_index_for_key(
    const CoreCoord& core, TestWorkerType worker_type, const ConnectionKey& key) const {
    auto keys = get_connection_keys_for_core(core, worker_type);

    for (size_t i = 0; i < keys.size(); i++) {
        if (keys[i] == key) {
            return i;
        }
    }

    return UINT32_MAX;  // Not found
}

bool FabricConnectionManager::is_mux_client(const CoreCoord& core) const {
    return all_mux_client_cores_.contains(core);
}

std::vector<uint32_t> FabricConnectionManager::generate_mux_termination_local_args_for_core(
    const CoreCoord& core, const std::shared_ptr<IDeviceInfoProvider>& device_info_provider) const {
    // Not a mux client? Return empty vector
    if (!is_mux_client(core)) {
        return {};
    }

    const bool is_master = (core == global_termination_master_);

    std::vector<uint32_t> args = {
        is_master,                                                                 // is_master
        static_cast<uint32_t>(all_mux_client_cores_.size()),                       // num_mux_clients
        device_info_provider->get_worker_noc_encoding(global_termination_master_)  // master_noc_encoding
    };

    // If master, add mux termination info
    if (is_master) {
        // Arg 3: number of muxes to terminate
        args.push_back(static_cast<uint32_t>(mux_configs_.size()));

        // For each mux: x, y, signal_addr (triples, not NOC encoding)
        for (const auto& [mux_core, mux_config] : mux_configs_) {
            auto mux_virtual_core = device_info_provider->get_virtual_core_from_logical_core(mux_core);
            uint32_t signal_addr = mux_config->get_termination_signal_address();

            args.push_back(static_cast<uint32_t>(mux_virtual_core.x));
            args.push_back(static_cast<uint32_t>(mux_virtual_core.y));
            args.push_back(signal_addr);
        }
    }

    return args;
}

std::vector<uint32_t> FabricConnectionManager::generate_connection_args_for_core(
    const CoreCoord& core,
    TestWorkerType worker_type,
    const std::shared_ptr<IDeviceInfoProvider>& device_info_provider,
    const std::shared_ptr<IRouteManager>& route_manager,
    const FabricNodeId& fabric_node_id,
    tt::tt_metal::Program& program_handle) const {
    std::vector<uint32_t> rt_args;

    auto keys = get_connection_keys_for_core(core, worker_type);

    for (const auto& key : keys) {
        auto conn_it = connections_.find(key);
        if (conn_it == connections_.end()) {
            continue;
        }

        const auto& conn = conn_it->second;

        // Add connection type flag first
        rt_args.push_back(conn.needs_mux ? 1u : 0u);

        if (conn.needs_mux) {
            // Get the channel assignment for this core
            auto channel_it = conn.channel_map.find(core);
            TT_FATAL(
                channel_it != conn.channel_map.end(),
                "Core {} not found in channel map for connection dir={} link={}",
                core,
                static_cast<int>(key.direction),
                key.link_idx);
            uint8_t channel_id = static_cast<uint8_t>(channel_it->second);

            // Get the stored mux core location for this connection
            auto mux_core_it = mux_cores_.find(key);
            TT_FATAL(
                mux_core_it != mux_cores_.end(),
                "Mux core not found for connection dir={} link={}",
                static_cast<int>(key.direction),
                key.link_idx);
            const CoreCoord& mux_core = mux_core_it->second;

            // Get the stored mux config using the mux core as key
            auto mux_config_it = mux_configs_.find(mux_core);
            TT_FATAL(
                mux_config_it != mux_configs_.end(),
                "Mux config not found for mux core {} (connection dir={} link={}, worker core={}). "
                "Did you call process() before creating kernels?",
                mux_core,
                static_cast<int>(key.direction),
                key.link_idx,
                core);

            const auto mux_virtual_core = device_info_provider->get_virtual_core_from_logical_core(mux_core);
            const auto& mux_config = mux_config_it->second;
            const auto channel_type = get_required_channel_type(worker_type);

            // kernel will allocate local semaphores (including status buffer for wait)
            std::vector<uint32_t> mux_rt_args = {
                mux_virtual_core.x,
                mux_virtual_core.y,
                mux_config->get_channel_credits_stream_id(channel_type, channel_id),
                mux_config->get_num_buffers(channel_type),
                static_cast<uint32_t>(mux_config->get_buffer_size_bytes(channel_type)),
                static_cast<uint32_t>(mux_config->get_channel_base_address(channel_type, channel_id)),
                static_cast<uint32_t>(mux_config->get_connection_info_address(channel_type, channel_id)),
                static_cast<uint32_t>(mux_config->get_connection_handshake_address(channel_type, channel_id)),
                static_cast<uint32_t>(mux_config->get_flow_control_address(channel_type, channel_id)),
                static_cast<uint32_t>(mux_config->get_buffer_index_address(channel_type, channel_id)),
                static_cast<uint32_t>(mux_config->get_status_address())};
            rt_args.insert(rt_args.end(), mux_rt_args.begin(), mux_rt_args.end());
        } else {
            // Generate fabric connection args directly using passed parameters
            const auto neighbor_node_id = route_manager->get_neighbor_node_id(fabric_node_id, key.direction);
            append_fabric_connection_rt_args(
                fabric_node_id, neighbor_node_id, key.link_idx, program_handle, core, rt_args);
        }
    }

    return rt_args;
}

void FabricConnectionManager::assign_and_validate_channels(Connection& conn, const ConnectionKey& key) {
    uint32_t next_full_size = 0;
    uint32_t next_header_only = 0;

    for (const auto& [core, worker_type] : conn.core_worker_types) {
        FabricMuxChannelType channel_type = get_required_channel_type(worker_type);

        if (channel_type == FabricMuxChannelType::FULL_SIZE_CHANNEL) {
            conn.channel_map[core] = next_full_size++;
        } else {  // HEADER_ONLY_CHANNEL
            conn.channel_map[core] = next_header_only++;
        }
    }

    // Validate channel limits
    TT_FATAL(
        next_full_size <= MAX_FULL_SIZE_CHANNELS,
        "Exceeded full-size channel limit: {} for connection direction={} link={} (max={})",
        next_full_size,
        static_cast<int>(key.direction),
        key.link_idx,
        MAX_FULL_SIZE_CHANNELS);

    TT_FATAL(
        next_header_only <= MAX_HEADER_ONLY_CHANNELS,
        "Exceeded header-only channel limit: {} for connection direction={} link={} (max={})",
        next_header_only,
        static_cast<int>(key.direction),
        key.link_idx,
        MAX_HEADER_ONLY_CHANNELS);
}

// ====================================
// TestWorker Implementation
// ====================================

TestWorker::TestWorker(
    CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    logical_core_(logical_core), test_device_ptr_(test_device_ptr) {
    if (kernel_src.has_value()) {
        this->kernel_src_ = std::string(kernel_src.value());
    }

    // populate worker id
}

void TestWorker::set_kernel_src(const std::string_view& kernel_src) { this->kernel_src_ = std::string(kernel_src); }

void TestWorker::create_kernel(
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

    const auto device_info_provider_ptr = this->test_device_ptr_->get_device_info_provider();
    for (const auto& [address, num_bytes] : addresses_and_size_to_clear) {
        if (address == 0 || num_bytes == 0) {
            continue;
        }
        device_info_provider_ptr->zero_out_buffer_on_cores(device_coord, {this->logical_core_}, address, num_bytes);
    }
}

// ====================================
// TestSender Implementation
// ====================================

TestSender::TestSender(
    CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src) {
    // TODO: init mem map?
}

void TestSender::add_config(TestTrafficSenderConfig config) {
    // Determine direction for fabric connection
    const auto dst_node_id = config.dst_node_ids[0];

    // Special handling: For torus 2D unicast, we have bugs where we try to follow the input hop count
    // but the routing tables cause packets to fail to reach the destination properly in some cases,
    // due to torus links. In this case, we use node IDs instead of hops.
    RoutingDirection outgoing_direction;
    bool is_torus_2d_unicast = (config.parameters.topology == tt::tt_fabric::Topology::Torus) &&
                               (config.parameters.is_2D_routing_enabled) &&
                               (config.parameters.chip_send_type == ChipSendType::CHIP_UNICAST);

    if (config.hops.has_value() && !is_torus_2d_unicast) {
        // Use hops to determine direction (for static routing with explicit hops)
        // However, NeighborExchange topology does not support multi-hop.
        outgoing_direction = this->test_device_ptr_->get_forwarding_direction(config.hops.value());
    } else {
        // Derive direction from src->dst node IDs
        outgoing_direction =
            this->test_device_ptr_->get_forwarding_direction(this->test_device_ptr_->get_node_id(), dst_node_id);
    }

    // Use common helper to register fabric connection
    auto fabric_connection_key = this->test_device_ptr_->register_fabric_connection(
        this->logical_core_,
        TestWorkerType::SENDER,
        this->test_device_ptr_->connection_manager_,
        outgoing_direction,
        config.link_id);

    this->configs_.emplace_back(std::move(config), fabric_connection_key);
}

bool TestSender::validate_results(std::vector<uint32_t>& data) const {
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

// ====================================
// TestReceiver Implementation
// ====================================

TestReceiver::TestReceiver(
    CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src) {
    // TODO: init mem map?
}

void TestReceiver::add_config(TestTrafficReceiverConfig config) {
    std::optional<ConnectionKey> credit_connection_key;

    // Register with connection manager if flow control is enabled
    // Receivers need fabric connections to send credits back to senders
    if (config.parameters.enable_flow_control && config.receiver_credit_info.has_value()) {
        const auto& credit_info = config.receiver_credit_info.value();

        // Determine direction for credit return connection (back to sender)
        const auto dst_node_id = credit_info.sender_node_id;
        const auto src_node_id = this->test_device_ptr_->get_node_id();
        auto outgoing_direction = this->test_device_ptr_->get_forwarding_direction(src_node_id, dst_node_id);

        // Use common helper to register fabric connection for credit return
        credit_connection_key = this->test_device_ptr_->register_fabric_connection(
            this->logical_core_,
            TestWorkerType::RECEIVER,
            this->test_device_ptr_->connection_manager_,
            outgoing_direction,
            config.link_id);
    }

    this->configs_.emplace_back(std::move(config), credit_connection_key);
}

bool TestReceiver::validate_results(std::vector<uint32_t>& data) const {
    bool pass = data[TT_FABRIC_STATUS_INDEX] == TT_FABRIC_STATUS_PASS;
    if (!pass) {
        const auto status = tt_fabric_status_to_string(data[TT_FABRIC_STATUS_INDEX]);
        log_error(tt::LogTest, "Receiver on core {} failed with status: {}", this->logical_core_, status);
        return false;
    }

    uint32_t num_expected_packets = 0;
    for (const auto& [config, credit_connection_key] : this->configs_) {
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

// ====================================
// TestSync Implementation
// ====================================

TestSync::TestSync(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src) {
    // TODO: init mem map?
}

void TestSync::add_config(TestTrafficSyncConfig sync_config) {
    const auto& sender_config = sync_config.sender_config;

    // Determine outgoing direction for sync message
    RoutingDirection outgoing_direction;
    // Multicast sync configs should always have hops specified (multicast pattern)
    TT_FATAL(sender_config.hops.has_value(), "Sync config on core {} should have hops specified", this->logical_core_);
    outgoing_direction = this->test_device_ptr_->get_forwarding_direction(sender_config.hops.value());

    // Use common helper to register sync fabric connection
    auto fabric_connection_key = this->test_device_ptr_->register_fabric_connection(
        this->logical_core_,
        TestWorkerType::SYNC,
        this->test_device_ptr_->get_sync_connection_manager(),
        outgoing_direction,
        sender_config.link_id);

    this->configs_.emplace_back(std::move(sync_config), fabric_connection_key);
}

bool TestSync::validate_results(std::vector<uint32_t>& /*data*/) const {
    // no-op for now
    return true;
}

// ====================================
// TestMux Implementation
// ====================================

TestMux::TestMux(CoreCoord logical_core, TestDevice* test_device_ptr, std::optional<std::string_view> kernel_src) :
    TestWorker(logical_core, test_device_ptr, kernel_src) {}

void TestMux::set_config(FabricMuxConfig* mux_config, ConnectionKey connection_key) {
    TT_FATAL(config_ == nullptr, "Mux config already set for core {}", logical_core_);
    config_ = mux_config;
    connection_key_ = connection_key;
}

// ====================================
// TestDevice Implementation
// ====================================

TestDevice::TestDevice(
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
}

tt::tt_metal::Program& TestDevice::get_program_handle() { return this->program_handle_; }

const FabricNodeId& TestDevice::get_node_id() const { return this->fabric_node_id_; }

void TestDevice::add_worker(TestWorkerType worker_type, CoreCoord logical_core) {
    auto core_already_occupied = [this](TestWorkerType worker_type, CoreCoord logical_core) {
        switch (worker_type) {
            case TestWorkerType::SENDER: return senders_.contains(logical_core);
            case TestWorkerType::RECEIVER: return receivers_.contains(logical_core);
            case TestWorkerType::SYNC: return sync_workers_.contains(logical_core);
            case TestWorkerType::MUX: return muxes_.contains(logical_core);
            default: TT_FATAL(false, "Invalid worker type: {}", static_cast<int>(worker_type));
        }
    };

    TT_FATAL(
        !core_already_occupied(worker_type, logical_core),
        "On node: {}, trying to add a worker type {} to an already occupied core: {}",
        this->fabric_node_id_,
        static_cast<int>(worker_type),
        logical_core);

    switch (worker_type) {
        case TestWorkerType::SENDER:
            senders_.emplace(logical_core, TestSender(logical_core, this, default_sender_kernel_src));
            break;
        case TestWorkerType::RECEIVER:
            receivers_.emplace(logical_core, TestReceiver(logical_core, this, default_receiver_kernel_src));
            break;
        case TestWorkerType::SYNC:
            sync_workers_.emplace(logical_core, TestSync(logical_core, this, default_sync_kernel_src));
            break;
        case TestWorkerType::MUX:
            muxes_.emplace(logical_core, TestMux(logical_core, this, default_mux_kernel_src));
            break;
        default: TT_FATAL(false, "Invalid worker type: {}", static_cast<int>(worker_type));
    }
}

ConnectionKey TestDevice::register_fabric_connection(
    CoreCoord logical_core,
    TestWorkerType worker_type,
    FabricConnectionManager& connection_mgr,
    RoutingDirection outgoing_direction,
    uint32_t link_idx) {
    // Get available link indices for this direction (to validate link_idx)
    std::vector<uint32_t> available_link_indices = get_forwarding_link_indices_in_direction(outgoing_direction);

    TT_FATAL(
        !available_link_indices.empty(),
        "No forwarding link indices found for direction {} from node {}",
        static_cast<int>(outgoing_direction),
        this->fabric_node_id_);

    TT_FATAL(
        std::find(available_link_indices.begin(), available_link_indices.end(), link_idx) !=
            available_link_indices.end(),
        "On node {}, link_idx={} is not valid for direction {}",
        this->fabric_node_id_,
        link_idx,
        static_cast<int>(outgoing_direction));

    // Check if this core already registered this connection
    ConnectionKey connection_key{outgoing_direction, link_idx};
    auto registered_keys = connection_mgr.get_connection_keys_for_core(logical_core, worker_type);

    if (std::find(registered_keys.begin(), registered_keys.end(), connection_key) != registered_keys.end()) {
        // Connection already registered - reuse it
        return connection_key;
    }

    // Register the new connection with the connection manager
    connection_mgr.register_client(logical_core, outgoing_direction, link_idx, worker_type);

    log_debug(
        tt::LogTest,
        "Worker type {} core {} registered with connection_manager: direction={}, link_idx={}",
        static_cast<int>(worker_type),
        logical_core,
        static_cast<int>(outgoing_direction),
        link_idx);

    return connection_key;
}

void TestDevice::add_sender_traffic_config(CoreCoord logical_core, TestTrafficSenderConfig config) {
    if (!this->senders_.contains(logical_core)) {
        this->add_worker(TestWorkerType::SENDER, logical_core);
    }

    this->senders_.at(logical_core).add_config(std::move(config));
}

void TestDevice::add_sender_sync_config(CoreCoord logical_core, TestTrafficSyncConfig sync_config) {
    if (!this->sync_workers_.contains(logical_core)) {
        this->add_worker(TestWorkerType::SYNC, logical_core);
    }

    this->sync_workers_.at(logical_core).add_config(std::move(sync_config));
}

void TestDevice::add_receiver_traffic_config(CoreCoord logical_core, const TestTrafficReceiverConfig& config) {
    if (!this->receivers_.contains(logical_core)) {
        this->add_worker(TestWorkerType::RECEIVER, logical_core);
    }

    this->receivers_.at(logical_core).add_config(config);
}

void TestDevice::add_mux_worker_config(
    CoreCoord logical_core, FabricMuxConfig* mux_config, ConnectionKey connection_key) {
    if (!this->muxes_.contains(logical_core)) {
        this->add_worker(TestWorkerType::MUX, logical_core);
    }

    this->muxes_.at(logical_core).set_config(mux_config, connection_key);
}

void TestDevice::create_kernels() {
    log_debug(tt::LogTest, "creating kernels on node: {}", fabric_node_id_);

    // Create local allocator for on-demand mux core allocation
    LocalDeviceCoreAllocator local_alloc(std::move(pristine_cores_));

    // Process fabric connections to determine mux requirements and assign channels
    connection_manager_.process(local_alloc, this, device_info_provider_);

    // Process sync connections separately if not using unified connection manager
    if (!use_unified_connection_manager_) {
        sync_connection_manager_.process(local_alloc, this, device_info_provider_);
    }

    this->create_mux_kernels();

    if (global_sync_) {
        this->create_sync_kernel();
    }

    // Normal flow (benchmark_mode affects kernel behavior via compile-time args)
    // Note: Latency tests call create_latency_sender_kernel() and create_latency_responder_kernel() directly
    this->create_sender_kernels();
    this->create_receiver_kernels();
}

void TestDevice::create_mux_kernels() {
    for (const auto& [mux_core, mux_worker] : muxes_) {
        auto* mux_config = mux_worker.config_;
        const auto& connection_key = mux_worker.connection_key_;

        const auto dst_node_id = route_manager_->get_neighbor_node_id(fabric_node_id_, connection_key.direction);

        auto mux_ct_args = mux_config->get_fabric_mux_compile_time_args();
        auto mux_rt_args = mux_config->get_fabric_mux_run_time_args(
            fabric_node_id_, dst_node_id, connection_key.link_idx, program_handle_, mux_core);

        mux_worker.create_kernel(
            coord_,
            mux_ct_args,
            mux_rt_args,
            {},  // no local args
            {},  // no local args address
            {}   // no addresses and size to clear
        );

        log_debug(tt::LogTest, "created mux kernel on core: {}", mux_core);
    }
}

void TestDevice::create_sync_kernel() {
    log_debug(tt::LogTest, "creating sync kernel on node: {}", fabric_node_id_);

    // TODO: fetch these dynamically
    const bool is_2D_routing_enabled = this->device_info_provider_->is_2D_routing_enabled();

    // Assuming single sync core per device for now
    TT_FATAL(
        sync_workers_.size() == 1,
        "Currently expecting exactly one sync core per device, got {}",
        sync_workers_.size());

    auto& [sync_core, sync_worker] = *sync_workers_.begin();

    const auto& sync_connection_manager = get_sync_connection_manager();

    size_t num_sync_connections =
        sync_connection_manager.get_connection_count_for_core(sync_core, TestWorkerType::SYNC);

    // Check if sync core has mux connections
    bool has_mux_connections = sync_connection_manager.is_mux_client(sync_core);
    uint32_t num_muxes_to_terminate = sync_connection_manager.get_num_muxes_to_terminate();

    // If the test is using the NeighborExchange topology, synchronization must use unicast packets
    const auto topology = tt::tt_fabric::get_fabric_topology();
    bool use_unicast_sync_packets = (topology == tt::tt_fabric::Topology::NeighborExchange);

    // Compile-time args
    std::vector<uint32_t> ct_args = {
        is_2D_routing_enabled,
        (uint32_t)num_sync_connections,                      /* num sync fabric connections */
        static_cast<uint32_t>(senders_.size() + 1),          /* num local sync cores (all senders + sync core) */
        sender_memory_map_->common.get_kernel_config_size(), /* kernel config buffer size */
        has_mux_connections ? 1u : 0u,                       /* HAS_MUX_CONNECTIONS */
        num_muxes_to_terminate,                              /* NUM_MUXES_TO_TERMINATE */
        use_unicast_sync_packets                             /* USE_UNICAST_SYNC_PACKETS */
    };

    // Runtime args: memory map args, then sync fabric connection args
    std::vector<uint32_t> rt_args = sender_memory_map_->get_memory_map_args();

    auto sync_connection_args = sync_connection_manager.generate_connection_args_for_core(
        sync_core, TestWorkerType::SYNC, device_info_provider_, route_manager_, fabric_node_id_, program_handle_);
    rt_args.insert(rt_args.end(), sync_connection_args.begin(), sync_connection_args.end());

    // Local args (all the rest go to local args buffer)
    std::vector<uint32_t> local_args;

    // Push in sync val first before pushing in rest of sync args
    // All sync configs for a device have been assigned the same sync val in
    // tt_fabric_test_context.hpp:process_traffic_config So we can just use the first sync config to get the sync val
    TT_FATAL(!sync_worker.configs_.empty(), "No sync configs found for core {}", sync_core.str());
    const auto& sync_val = sync_worker.configs_.front().first.sync_val;
    local_args.push_back(sync_val);

    // Add sync config to fabric connection mapping (same pattern as sender traffic configs)
    // This mapping tells each LineSyncConfig which fabric connection index to use
    for (const auto& [sync_config, connection_key] : sync_worker.configs_) {
        uint32_t array_idx =
            sync_connection_manager.get_connection_array_index_for_key(sync_core, TestWorkerType::SYNC, connection_key);
        TT_FATAL(
            array_idx != UINT32_MAX, "Failed to find connection array index for sync config on core {}", sync_core);
        local_args.push_back(array_idx);
    }

    // Add sync routing args for each sync config
    for (const auto& [sync_config, _] : sync_worker.configs_) {
        const auto& sender_config = sync_config.sender_config;
        auto sync_traffic_args = sender_config.get_args(true /* is_sync_config */);
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

    // Add mux termination local args (empty vector if not a mux client)
    auto mux_termination_local_args =
        sync_connection_manager.generate_mux_termination_local_args_for_core(sync_core, device_info_provider_);
    local_args.insert(local_args.end(), mux_termination_local_args.begin(), mux_termination_local_args.end());

    std::vector<std::pair<size_t, size_t>> addresses_and_size_to_clear;

    // Only clear result buffer from host if progress monitoring is enabled
    if (progress_monitoring_enabled_) {
        addresses_and_size_to_clear.push_back(
            {sender_memory_map_->get_result_buffer_address(), sender_memory_map_->get_result_buffer_size()});
    }

    addresses_and_size_to_clear.push_back(
        {sender_memory_map_->get_global_sync_address(), sender_memory_map_->get_global_sync_region_size()});
    addresses_and_size_to_clear.push_back(
        {sender_memory_map_->get_local_sync_address(), sender_memory_map_->get_local_sync_region_size()});

    // clear out mux termination sync address (if mux connections are present)
    if (!mux_termination_local_args.empty()) {
        addresses_and_size_to_clear.push_back(
            {sender_memory_map_->get_mux_termination_sync_address(),
             sender_memory_map_->get_mux_termination_sync_size()});
    }

    // create sync kernel with local args
    sync_worker.create_kernel(
        coord_,
        ct_args,
        rt_args,
        local_args,
        sender_memory_map_->get_local_args_address(),
        addresses_and_size_to_clear);
    log_debug(tt::LogTest, "created sync kernel on core: {}", sync_core);
}

void TestDevice::create_sender_kernels() {
    // Unified sender kernel creation - handles both fabric and mux connections based on per-pattern flow control
    const bool is_2D_routing_enabled = this->device_info_provider_->is_2D_routing_enabled();
    uint32_t num_local_sync_cores = static_cast<uint32_t>(this->senders_.size()) + 1;

    TT_FATAL(sender_memory_map_ != nullptr, "Sender memory map is required for creating sender kernels");
    TT_FATAL(sender_memory_map_->is_valid(), "Sender memory map is invalid");

    for (const auto& [core, sender] : this->senders_) {
        // Get connection count and generate all connection args via FabricConnectionManager
        size_t num_connections = connection_manager_.get_connection_count_for_core(core, TestWorkerType::SENDER);

        // Check if this core has mux connections
        bool has_mux_connections = connection_manager_.is_mux_client(core);
        uint32_t num_muxes_to_terminate = connection_manager_.get_num_muxes_to_terminate();

        // Compile-time args (FLOW_CONTROL_ENABLED removed - now handled per-traffic-config)
        std::vector<uint32_t> ct_args = {
            is_2D_routing_enabled,
            (uint32_t)num_connections,                           /* num connections (from FabricConnectionManager) */
            sender.configs_.size(),                              /* num traffic configs */
            (uint32_t)benchmark_mode_,                           /* benchmark mode */
            (uint32_t)global_sync_,                              /* line sync enabled */
            num_local_sync_cores,                                /* num local sync cores */
            sender_memory_map_->common.get_kernel_config_size(), /* kernel config buffer size */
            has_mux_connections ? 1u : 0u,                       /* HAS_MUX_CONNECTIONS */
            num_muxes_to_terminate                               /* NUM_MUXES_TO_TERMINATE */
        };

        // Runtime args with connection type information
        std::vector<uint32_t> rt_args = sender_memory_map_->get_memory_map_args();

        // Add all connection args via FabricConnectionManager
        auto connection_args = connection_manager_.generate_connection_args_for_core(
            core, TestWorkerType::SENDER, device_info_provider_, route_manager_, fabric_node_id_, program_handle_);
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
                connection_manager_.get_connection_array_index_for_key(core, TestWorkerType::SENDER, connection_key);
            TT_FATAL(
                array_idx < num_connections,
                "Connection array idx should be < num_connections. Got idx {}, num_connections {}",
                array_idx,
                num_connections);
            local_args.push_back(array_idx);
        }

        // Add sender traffic config args (including credit management info)
        for (const auto& [config, _] : sender.configs_) {
            auto traffic_config_args = config.get_args(false);  // false = not a sync config
            local_args.insert(local_args.end(), traffic_config_args.begin(), traffic_config_args.end());
        }

        // Add mux termination local args (empty vector if not a mux client)
        auto mux_termination_local_args =
            connection_manager_.generate_mux_termination_local_args_for_core(core, device_info_provider_);
        local_args.insert(local_args.end(), mux_termination_local_args.begin(), mux_termination_local_args.end());

        std::vector<std::pair<size_t, size_t>> addresses_and_size_to_clear;

        // Only clear result buffer from host if progress monitoring is enabled
        if (progress_monitoring_enabled_) {
            addresses_and_size_to_clear.push_back(
                {sender_memory_map_->get_result_buffer_address(), sender_memory_map_->get_result_buffer_size()});
        }

        // clear out local sync address (if line sync is enabled)
        if (global_sync_) {
            addresses_and_size_to_clear.push_back(
                {sender_memory_map_->get_local_sync_address(), sender_memory_map_->get_local_sync_region_size()});
        }

        // clear out mux termination sync address (if mux connections are present)
        if (!mux_termination_local_args.empty()) {
            addresses_and_size_to_clear.push_back(
                {sender_memory_map_->get_mux_termination_sync_address(),
                 sender_memory_map_->get_mux_termination_sync_size()});
        }

        sender.create_kernel(
            coord_,
            ct_args,
            rt_args,
            local_args,
            sender_memory_map_->get_local_args_address(),
            addresses_and_size_to_clear);

        log_debug(tt::LogTest, "Created sender kernel on core {}", core);
    }
}

void TestDevice::create_receiver_kernels() {
    const bool is_2D_routing_enabled = this->device_info_provider_->is_2D_routing_enabled();

    TT_FATAL(receiver_memory_map_ != nullptr, "Receiver memory map is required for creating receiver kernels");
    TT_FATAL(receiver_memory_map_->is_valid(), "Receiver memory map is invalid");

    for (const auto& [core, receiver] : this->receivers_) {
        // Get connection count and generate all connection args via FabricConnectionManager (for credit return)
        size_t num_connections = connection_manager_.get_connection_count_for_core(core, TestWorkerType::RECEIVER);

        // Check if this core has mux connections
        bool has_mux_connections = connection_manager_.is_mux_client(core);
        uint32_t num_muxes_to_terminate = connection_manager_.get_num_muxes_to_terminate();

        // Compile-time args (order must match receiver kernel .cpp file)
        std::vector<uint32_t> ct_args = {
            is_2D_routing_enabled ? 1u : 0u,                       /* IS_2D_FABRIC */
            receiver.configs_.size(),                              /* NUM_TRAFFIC_CONFIGS */
            benchmark_mode_ ? 1u : 0u,                             /* BENCHMARK_MODE */
            receiver_memory_map_->common.get_kernel_config_size(), /* KERNEL_CONFIG_BUFFER_SIZE */
            (uint32_t)num_connections,                             /* NUM_CREDIT_CONNECTIONS */
            has_mux_connections ? 1u : 0u,                         /* HAS_MUX_CONNECTIONS */
            num_muxes_to_terminate                                 /* NUM_MUXES_TO_TERMINATE */
        };

        // Runtime args: memory map args + credit connection args + traffic-to-connection mapping
        std::vector<uint32_t> rt_args = receiver_memory_map_->get_memory_map_args();

        // Add all connection args via FabricConnectionManager (for credit return)
        auto connection_args = connection_manager_.generate_connection_args_for_core(
            core, TestWorkerType::RECEIVER, device_info_provider_, route_manager_, fabric_node_id_, program_handle_);
        rt_args.insert(rt_args.end(), connection_args.begin(), connection_args.end());

        // Build traffic config to credit connection mapping (same as sender side)
        // Query the array index for each traffic config's connection key
        for (const auto& [config, credit_connection_key] : receiver.configs_) {
            uint8_t connection_idx = 0xFF;  // Invalid index by default

            if (credit_connection_key.has_value()) {
                uint32_t array_idx = connection_manager_.get_connection_array_index_for_key(
                    core, TestWorkerType::RECEIVER, credit_connection_key.value());

                TT_FATAL(
                    array_idx < num_connections,
                    "Connection array idx should be < num_connections. Got idx {}, num_connections {}",
                    array_idx,
                    num_connections);
                connection_idx = static_cast<uint8_t>(array_idx);
            }

            rt_args.push_back(connection_idx);
        }

        std::vector<std::pair<size_t, size_t>> addresses_and_size_to_clear;

        // Only clear result buffer from host if progress monitoring is enabled
        if (progress_monitoring_enabled_) {
            addresses_and_size_to_clear.push_back(
                {receiver_memory_map_->get_result_buffer_address(), receiver_memory_map_->get_result_buffer_size()});
        }

        // Local args for traffic configs
        std::vector<uint32_t> local_args;
        if (!receiver.configs_.empty()) {
            const auto first_traffic_args = receiver.configs_[0].first.get_args();
            local_args.reserve(local_args.size() + (receiver.configs_.size() * first_traffic_args.size()));
            local_args.insert(local_args.end(), first_traffic_args.begin(), first_traffic_args.end());

            // clear out the atomic inc address if used
            if (receiver.configs_[0].first.atomic_inc_address.has_value()) {
                addresses_and_size_to_clear.push_back(
                    {receiver.configs_[0].first.atomic_inc_address.value(), sizeof(uint32_t)});
            }

            for (size_t i = 1; i < receiver.configs_.size(); ++i) {
                const auto& traffic_config = receiver.configs_[i].first;
                const auto traffic_args = traffic_config.get_args();
                local_args.insert(local_args.end(), traffic_args.begin(), traffic_args.end());

                // clear out the atomic inc address if used
                if (traffic_config.atomic_inc_address.has_value()) {
                    addresses_and_size_to_clear.push_back(
                        {traffic_config.atomic_inc_address.value(), sizeof(uint32_t)});
                }
            }
        }

        // Add mux termination local args (empty vector if not a mux client)
        auto mux_termination_local_args =
            connection_manager_.generate_mux_termination_local_args_for_core(core, device_info_provider_);
        local_args.insert(local_args.end(), mux_termination_local_args.begin(), mux_termination_local_args.end());

        if (!mux_termination_local_args.empty()) {
            addresses_and_size_to_clear.push_back(
                {receiver_memory_map_->get_mux_termination_sync_address(),
                 receiver_memory_map_->get_mux_termination_sync_size()});
        }

        receiver.create_kernel(
            coord_,
            ct_args,
            rt_args,
            local_args,
            receiver_memory_map_->get_local_args_address(),
            addresses_and_size_to_clear);

        log_debug(tt::LogTest, "Created receiver kernel on core {}", core);
    }
}

RoutingDirection TestDevice::get_forwarding_direction(
    const std::unordered_map<RoutingDirection, uint32_t>& hops) const {
    return this->route_manager_->get_forwarding_direction(hops);
}

RoutingDirection TestDevice::get_forwarding_direction(
    const FabricNodeId& src_node_id, const FabricNodeId& dst_node_id) const {
    return this->route_manager_->get_forwarding_direction(src_node_id, dst_node_id);
}

std::vector<uint32_t> TestDevice::get_forwarding_link_indices_in_direction(const RoutingDirection& direction) const {
    const auto link_indices =
        this->route_manager_->get_forwarding_link_indices_in_direction(this->fabric_node_id_, direction);
    TT_FATAL(
        !link_indices.empty(),
        "No forwarding link indices found in direction: {} from {}",
        direction,
        this->fabric_node_id_);
    return link_indices;
}

std::vector<uint32_t> TestDevice::get_forwarding_link_indices_in_direction(
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

void TestDevice::validate_results() const {
    TT_FATAL(
        false,
        "validate_results is deprecated. Use initiate_results_readback and validate_results_after_readback instead");
    this->validate_sender_results();
    this->validate_receiver_results();
}

TestDevice::ValidationReadOps TestDevice::initiate_results_readback() const {
    ValidationReadOps ops;

    // Get sender cores
    std::vector<CoreCoord> sender_cores;
    sender_cores.reserve(this->senders_.size());
    for (const auto& [core, _] : this->senders_) {
        sender_cores.push_back(core);
    }

    if (!sender_cores.empty()) {
        ops.has_senders = true;
        // Cast to TestFixture to access the new methods
        const auto* fixture = dynamic_cast<const TestFixture*>(this->device_info_provider_.get());
        TT_FATAL(fixture != nullptr, "Failed to cast device_info_provider to TestFixture");
        ops.sender_op = fixture->initiate_read_buffer_from_cores(
            this->coord_,
            sender_cores,
            this->sender_memory_map_->get_result_buffer_address(),
            this->sender_memory_map_->get_result_buffer_size());
    }

    // Get receiver cores
    std::vector<CoreCoord> receiver_cores;
    receiver_cores.reserve(this->receivers_.size());
    for (const auto& [core, _] : this->receivers_) {
        receiver_cores.push_back(core);
    }

    if (!receiver_cores.empty()) {
        ops.has_receivers = true;
        const auto* fixture = dynamic_cast<const TestFixture*>(this->device_info_provider_.get());
        TT_FATAL(fixture != nullptr, "Failed to cast device_info_provider to TestFixture");
        ops.receiver_op = fixture->initiate_read_buffer_from_cores(
            this->coord_,
            receiver_cores,
            this->receiver_memory_map_->get_result_buffer_address(),
            this->receiver_memory_map_->get_result_buffer_size());
    }

    return ops;
}

void TestDevice::validate_results_after_readback(const ValidationReadOps& ops) const {
    const auto* fixture = dynamic_cast<const TestFixture*>(this->device_info_provider_.get());
    TT_FATAL(fixture != nullptr, "Failed to cast device_info_provider to TestFixture");

    // Validate senders
    if (ops.has_senders) {
        auto data = fixture->complete_read_buffer_from_cores(ops.sender_op);
        for (const auto& [core, sender] : this->senders_) {
            bool pass = sender.validate_results(data.at(core));
            TT_FATAL(pass, "Sender on device: {} core: {} failed", this->fabric_node_id_, core);
        }
    }

    // Validate receivers
    if (ops.has_receivers) {
        auto data = fixture->complete_read_buffer_from_cores(ops.receiver_op);
        for (const auto& [core, receiver] : this->receivers_) {
            bool pass = receiver.validate_results(data.at(core));
            TT_FATAL(pass, "Receiver on device: {} core: {} failed", this->fabric_node_id_, core);
        }
    }
}

void TestDevice::validate_sender_results() const {
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

void TestDevice::validate_receiver_results() const {
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

void TestDevice::set_local_runtime_args_for_core(
    const MeshCoordinate& device_coord,
    CoreCoord logical_core,
    uint32_t local_args_address,
    const std::vector<uint32_t>& args) const {
    device_info_provider_->write_data_to_core(device_coord, logical_core, local_args_address, args);
}

size_t TestDevice::get_latency_send_buffer_address() const {
    // Send buffer starts after the latency result buffer
    // Results are stored as uint32_t elapsed times (1 per sample)
    return sender_memory_map_->get_result_buffer_address() + (TestDevice::MAX_LATENCY_SAMPLES * sizeof(uint32_t));
}

size_t TestDevice::get_latency_receive_buffer_address(uint32_t payload_size) const {
    // Receive buffer starts after the send buffer
    // Send buffer size must accommodate the full message payload
    TT_FATAL(payload_size > 0, "Latency payload size must be greater than 0");
    return get_latency_send_buffer_address() + payload_size;
}

void TestDevice::create_latency_sender_kernel(
    CoreCoord core,
    FabricNodeId dest_node,
    uint32_t payload_size,
    uint32_t num_samples,
    NocSendType noc_send_type,
    CoreCoord responder_virtual_core) {
    log_debug(tt::LogTest, "Creating latency sender kernel on node: {}", fabric_node_id_);

    // Use static memory map address for semaphore (same as bandwidth tests)
    // Both sender and responder use the same memory layout, so they share the sync address
    uint32_t semaphore_address = sender_memory_map_->get_local_sync_address();

    // Get topology information
    const bool is_2d_fabric = device_info_provider_->is_2D_routing_enabled();

    // Compute routing information
    uint32_t num_hops = 0;
    if (!is_2d_fabric) {
        // For 1D topology, compute hop distance
        auto hops_map = route_manager_->get_hops_to_chip(fabric_node_id_, dest_node);
        // For 1D (linear/ring), sum all hops (there should only be one non-zero direction)
        for (const auto& [dir, hop_count] : hops_map) {
            num_hops += hop_count;
        }
    }

    // Register fabric connection to mark ethernet link as "used"
    // This is required for telemetry and code profiling to know which cores to read from
    RoutingDirection outgoing_direction = get_forwarding_direction(fabric_node_id_, dest_node);
    auto available_links = get_forwarding_link_indices_in_direction(outgoing_direction);
    TT_FATAL(
        !available_links.empty(),
        "No forwarding links available in direction {} from node {} to node {}",
        static_cast<int>(outgoing_direction),
        fabric_node_id_,
        dest_node);
    uint32_t link_idx = available_links[0];  // Use first available link

    // Compile-time args: fused_sync, sem_inc_only, is_2d_fabric
    bool enable_fused_payload_with_sync = (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC);
    bool sem_inc_only = (payload_size == 0);
    std::vector<uint32_t> ct_args = {
        enable_fused_payload_with_sync ? 1u : 0u, sem_inc_only ? 1u : 0u, is_2d_fabric ? 1u : 0u};

    // Runtime args
    // Calculate send and receive buffer addresses after timestamp storage
    TT_FATAL(
        num_samples <= MAX_LATENCY_SAMPLES,
        "Latency test num_samples ({}) exceeds maximum supported samples ({}). "
        "Increase RESULT_BUFFER_SIZE or reduce num_samples.",
        num_samples,
        MAX_LATENCY_SAMPLES);

    uint32_t send_buffer_address = get_latency_send_buffer_address();
    uint32_t receive_buffer_address = get_latency_receive_buffer_address(payload_size);

    // responder_virtual_core is passed as parameter from TestContext
    // Build runtime args - routing parameters differ between 1D and 2D
    std::vector<uint32_t> rt_args = {
        sender_memory_map_->get_result_buffer_address(),  // result buffer for latency samples
        semaphore_address,                                // shared semaphore address (same offset on all devices)
        payload_size,                                     // payload size
        num_samples,                                      // number of latency samples to collect
        send_buffer_address,                              // sender's send buffer (to write before sending)
        receive_buffer_address,                           // sender's receive buffer (to wait on)
        static_cast<uint32_t>(responder_virtual_core.x),  // responder's virtual NOC X coordinate
        static_cast<uint32_t>(responder_virtual_core.y),  // responder's virtual NOC Y coordinate
    };

    // Add topology-specific routing information
    if (!is_2d_fabric) {
        // 1D: add hop count
        rt_args.push_back(num_hops);
    } else {
        // 2D: add device and mesh IDs (for Hybrid Mesh routing)
        rt_args.push_back(dest_node.chip_id);
        rt_args.push_back(dest_node.mesh_id.get());
    }

    // Add fabric connection args
    tt::tt_fabric::append_fabric_connection_rt_args(
        fabric_node_id_, dest_node, link_idx, program_handle_, {core}, rt_args);

    const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear = {
        {semaphore_address, sender_memory_map_->get_local_sync_region_size()}};

    senders_.at(core).create_kernel(coord_, ct_args, rt_args, {}, {}, addresses_and_size_to_clear);

    log_debug(
        tt::LogTest,
        "Created latency sender kernel on core {} with shared semaphore address 0x{:x}",
        core,
        semaphore_address);
}

void TestDevice::create_latency_responder_kernel(
    CoreCoord core,
    FabricNodeId sender_node,
    uint32_t payload_size,
    uint32_t num_samples,
    NocSendType noc_send_type,
    uint32_t sender_send_buffer_address,
    uint32_t sender_receive_buffer_address,
    CoreCoord sender_virtual_core) {
    log_debug(tt::LogTest, "Creating latency responder kernel on node: {}", fabric_node_id_);

    // Use static memory map address for semaphore (same as bandwidth tests)
    // Both sender and responder use the same memory layout, so they share the sync address
    uint32_t semaphore_address = sender_memory_map_->get_local_sync_address();

    // Get topology information
    const bool is_2d_fabric = device_info_provider_->is_2D_routing_enabled();

    // Compute routing information
    uint32_t num_hops_back = 0;
    if (!is_2d_fabric) {
        // For 1D topology, compute hop distance back to sender
        auto hops_map = route_manager_->get_hops_to_chip(fabric_node_id_, sender_node);
        // For 1D (linear/ring), sum all hops (there should only be one non-zero direction)
        for (const auto& [dir, hop_count] : hops_map) {
            num_hops_back += hop_count;
        }
    }

    // Register fabric connection to mark ethernet link as "used"
    // This is required for telemetry and code profiling to know which cores to read from
    // Note: Responder sends back to sender, so use RECEIVER worker type (similar to flow control credits)
    RoutingDirection outgoing_direction = get_forwarding_direction(fabric_node_id_, sender_node);
    auto available_links = get_forwarding_link_indices_in_direction(outgoing_direction);
    TT_FATAL(
        !available_links.empty(),
        "No forwarding links available in direction {} from node {} to node {}",
        static_cast<int>(outgoing_direction),
        fabric_node_id_,
        sender_node);
    uint32_t link_idx = available_links[0];  // Use first available link

    // Compile-time args: fused_sync, sem_inc_only, is_2d_fabric, use_dynamic_routing
    bool enable_fused_payload_with_sync = (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC);
    bool sem_inc_only = (payload_size == 0);
    std::vector<uint32_t> ct_args = {
        enable_fused_payload_with_sync ? 1u : 0u, sem_inc_only ? 1u : 0u, is_2d_fabric ? 1u : 0u};

    // Runtime args
    // Calculate responder's receive buffer and sender's receive buffer addresses
    constexpr uint32_t MAX_LATENCY_SAMPLES = 1024;
    TT_FATAL(
        num_samples <= MAX_LATENCY_SAMPLES,
        "Latency test num_samples ({}) exceeds maximum supported samples ({}). "
        "Increase RESULT_BUFFER_SIZE or reduce num_samples.",
        num_samples,
        MAX_LATENCY_SAMPLES);

    // Use sender's actual buffer addresses and coordinates passed from caller (no recomputation)
    // Responder receives from sender's send buffer
    uint32_t responder_receive_buffer_address = sender_send_buffer_address;
    // sender_virtual_core is passed as parameter from TestContext

    // Build runtime args - routing parameters differ between 1D and 2D
    std::vector<uint32_t> rt_args = {
        sender_memory_map_->get_result_buffer_address(),  // local buffer for timestamp storage
        semaphore_address,                                // shared semaphore address (same offset on all devices)
        payload_size,                                     // payload size
        num_samples,                                      // number of latency samples to collect
        responder_receive_buffer_address,                 // responder's receive buffer (receives from sender)
        sender_receive_buffer_address,                    // sender's receive buffer (responder writes here)
        static_cast<uint32_t>(sender_virtual_core.x),     // sender's virtual NOC X coordinate
        static_cast<uint32_t>(sender_virtual_core.y),     // sender's virtual NOC Y coordinate
    };

    // Add topology-specific routing information (for sending back to sender)
    if (!is_2d_fabric) {
        // 1D: add hop count back to sender
        rt_args.push_back(num_hops_back);
    } else {
        // 2D: add device and mesh IDs for sender (for Hybrid Mesh routing)
        rt_args.push_back(sender_node.chip_id);
        rt_args.push_back(sender_node.mesh_id.get());
    }

    // Add fabric connection args (back to sender)
    tt::tt_fabric::append_fabric_connection_rt_args(
        fabric_node_id_, sender_node, link_idx, program_handle_, {core}, rt_args);

    const std::vector<std::pair<size_t, size_t>>& addresses_and_size_to_clear = {
        {semaphore_address, sender_memory_map_->get_local_sync_region_size()}};

    receivers_.at(core).create_kernel(coord_, ct_args, rt_args, {}, {}, addresses_and_size_to_clear);

    log_debug(
        tt::LogTest,
        "Created latency responder kernel on core {} with shared semaphore address 0x{:x}",
        core,
        semaphore_address);
}

// Set kernel source for specific workers (used by latency tests to override default kernels)
void TestDevice::set_sender_kernel_src(CoreCoord core, const std::string& kernel_src) {
    auto it = senders_.find(core);
    if (it != senders_.end()) {
        it->second.set_kernel_src(kernel_src);
    }
}

void TestDevice::set_receiver_kernel_src(CoreCoord core, const std::string& kernel_src) {
    auto it = receivers_.find(core);
    if (it != receivers_.end()) {
        it->second.set_kernel_src(kernel_src);
    }
}

// TestSender accessor implementations (need complete TestDevice)
uint64_t TestSender::get_total_packets() const {
    uint64_t total = 0;
    for (const auto& [config, _] : configs_) {
        total += config.parameters.num_packets;
    }

    return total;
}

}  // namespace tt::tt_fabric::fabric_tests
