// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>

#include <umd/device/types/cluster_descriptor_types.h>
#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/edm_fabric_counters.hpp>
#include <tt-metalium/routing_table_generator.hpp>  // for FabricNodeId
#include <unordered_map>
#include <optional>
#include <cstdint>
#include <vector>
#include <array>
#include <cstddef>

namespace tt::tt_fabric {

struct FabricRiscConfig;

/**
 * Specify the EDM types—Default, Dateline, DatelineUpstream, and DatelineUpstreamAdjacentDevice—used to configure
 * different EDM sender/receiver buffer slots. We don't need the S2, R0 channels for the dateline EDM, and no need for
 * S1, R1 channels for the Upstream dateline EDM.
 *
 *      ┌────────────────────────────────────────────────┐    ┌─────────────────────────────────────────────────────┐
 *      │                                                │    │                                                     │
 *      │                                                │    │                                                     │
 *      │         Dateline           Dateline Upstream   │    │   Dateline Upstream Device Adjacent                 │
 *      │  ┌──────────────────┐     ┌─────────────────┐  │    │         ┌─────────────────┐                         │
 *      │  │ ┌──────────────┐ │     │ ┌─────────────┐ │  │    │         │ ┌─────────────┐ │                         │
 *      │  │ │      S0      │ │     │ │     S0     ─┼─┼──┼────┼─────┐   │ │     S0      │ │                         │
 *      │  │ └──────────────┘ │     │ └─────────────┘ │  │    │     │   │ └─────────────┘ │                         │
 *      │  │ ┌──────────────┐ │     │                 │  │    │     │   │ ┌─────────────┐ │                         │
 *      │  │ │      S1      ◄─┼──┐  │                 │  │    │     │   │ │     S1      │ │                         │
 *      │  │ └──────────────┘ │  │  │                 │  │    │     │   │ └─────────────┘ │                         │
 *      │  │                  │  │  │ ┌─────────────┐ │  │    │     │   │ ┌─────────────┐ │                         │
 *      │  │        ┌─────────┼──┼──┼─►     S2      ┼─┼──┼────┼─┐   │   │ │     S2      │ │                         │
 *      │  │        │         │  │  │ └─────────────┘ │  │    │ │   │   │ └─────────────┘ │                         │
 *      │  │        │         │  │  │ ┌─────────────┐ │  │    │ │   │   │ ┌─────────────┐ │                         │
 *      │  │        │         │  └──┼─┼─    R0      │ │  │    │ │   └───┼─┼►    R0      │ │                         │
 *      │  │        │         │     │ └─────────────┘ │  │    │ │       │ └─────────────┘ │                         │
 *      │  │ ┌──────┼───────┐ │     │                 │  │    │ │       │ ┌─────────────┐ │                         │
 *──────┼──┼─►      R1      │ │     │                 │  │    │ └───────┼─┼►    R1      │ │                         │
 *      │  │ └──────────────┘ │     │                 │  │    │         │ └─────────────┘ │                         │
 *      │  └──────────────────┘     └─────────────────┘  │    │         └─────────────────┘                         │
 *      │                                                │    │                                                     │
 *      │                                                │    │                                                     │
 *      │                                                │    │                                                     │
 *      └────────────────────────────────────────────────┘    └─────────────────────────────────────────────────────┘
 */
enum class FabricEriscDatamoverType {
    Default = 0,
    Dateline = 1,
    DatelineUpstream = 2,
    DatelineUpstreamAdjacentDevice = 3,
    DatelineUpstreamAdjacentDeviceUpstream = 4,
    Invalid = 5,
};

enum class FabricEriscDatamoverAxis : std::size_t {
    Short = 0,
    Long = 1,
    Invalid = 2,
};

enum class FabricEriscDatamoverContextSwitchType : uint8_t {
    // Context switch at the interval only if idle for a certain number of cycles
    WAIT_FOR_IDLE = 0,
    // Context switch every interval
    INTERVAL = 1,
};

struct FabricRouterBufferConfig {
    bool enable_dateline_sender_extra_buffer_slots = false;
    bool enable_dateline_receiver_extra_buffer_slots = false;
    bool enable_dateline_upstream_sender_extra_buffer_slots = false;
    bool enable_dateline_upstream_receiver_extra_buffer_slots = false;
    bool enable_dateline_upstream_adjacent_sender_extra_buffer_slots = false;
};

// enable extra buffer slots configuration based on sender/receiver channel and EDM type.
struct FabricEriscDatamoverOptions {
    FabricEriscDatamoverType edm_type = FabricEriscDatamoverType::Default;
    FabricEriscDatamoverAxis edm_axis = FabricEriscDatamoverAxis::Short;
    FabricRouterBufferConfig edm_buffer_config = FabricRouterBufferConfig{};
};

struct FabricEriscDatamoverConfig {
    static constexpr uint32_t WR_CMD_BUF = 0;      // for large writes
    static constexpr uint32_t RD_CMD_BUF = 1;      // for all reads
    static constexpr uint32_t WR_REG_CMD_BUF = 2;  // for small writes (e.g., registers, semaphores)
    static constexpr uint32_t AT_CMD_BUF = 3;      // for atomics
    static constexpr uint32_t DEFAULT_NOC_VC = 2;
    static constexpr uint32_t NUM_EDM_NOC_VCS = 2;

    static constexpr uint32_t DEFAULT_RECEIVER_FORWARDING_NOC = 1;
    static constexpr uint32_t DEFAULT_RECEIVER_LOCAL_WRITE_NOC = 1;
    static constexpr uint32_t DEFAULT_SENDER_ACK_NOC = 0;

    // If a mesh axis spans eight or more devices, use more buffer slot configuration.
    // Threshold (8 devices) was determined empirically.
    static constexpr std::size_t MESH_LONG_AXIS_OPTIMIZATION_THRESHOLD = 8;

    static constexpr std::size_t dateline_sender_channel_skip_idx = 2;
    static constexpr std::size_t dateline_receiver_channel_skip_idx = 0;
    static constexpr std::size_t dateline_upstream_sender_channel_skip_idx = 1;
    static constexpr std::size_t dateline_upstream_receiver_channel_skip_idx = 1;
    static constexpr std::size_t dateline_upstream_adjcent_sender_channel_skip_idx = 2;

    static constexpr std::size_t num_sender_channels_1d = 3;
    static constexpr std::size_t num_sender_channels_2d = 5;
    static constexpr std::size_t num_sender_channels = std::max(num_sender_channels_1d, num_sender_channels_2d);
    static constexpr std::size_t num_downstream_sender_channels = num_sender_channels - 1;

    static constexpr std::size_t num_receiver_channels = 2;
    static constexpr std::size_t num_downstream_edms_vc0 = 1;
    static constexpr std::size_t num_downstream_edms_2d_vc0 = 4;
    static constexpr std::size_t num_downstream_edms_vc1 = 1;
    static constexpr std::size_t num_downstream_edms = num_downstream_edms_vc0 + num_downstream_edms_vc1;
    static constexpr std::size_t num_downstream_edms_2d = num_downstream_edms_2d_vc0 + num_downstream_edms_vc1;
    static constexpr std::size_t max_downstream_edms = std::max(num_downstream_edms, num_downstream_edms_2d);
    static constexpr uint32_t num_virtual_channels = 2;

    static constexpr std::size_t field_size = 16;
    static constexpr std::size_t buffer_alignment = 32;
    static constexpr std::size_t eth_word_l1_alignment = 16;
    static constexpr uint32_t default_iterations_between_ctx_switch_and_teardown_checks = 32;
    static_assert(((buffer_alignment - 1) & buffer_alignment) == 0);
    static constexpr bool enable_fabric_counters = false;
    static constexpr bool enable_fabric_pkt_header_recording = false;

    // Global
    static constexpr std::size_t eth_channel_sync_size = 16;
    std::size_t handshake_addr = 0;
    std::size_t edm_channel_ack_addr = 0;
    std::size_t termination_signal_address = 0;  // pad extra bytes to match old EDM so handshake logic will still work
    std::size_t edm_local_sync_address = 0;
    std::size_t edm_status_address = 0;

    // Debug and Counters
    static constexpr std::size_t receiver_channel_counters_size_bytes =
        (((tt::tt_fabric::receiver_channel_counters_l1_size - 1) / field_size) + 1) * field_size;
    static constexpr std::size_t sender_channel_counters_size_bytes =
        (((tt::tt_fabric::sender_channel_counters_l1_size - 1) / field_size) + 1) * field_size;

    std::array<std::size_t, num_receiver_channels> receiver_channels_counters_address = {};
    std::array<std::size_t, num_sender_channels> sender_channels_counters_address = {};

    // Packet header history buffer(s)
    static constexpr std::size_t receiver_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t receiver_completed_packet_header_cb_size_bytes =
        sizeof(tt::tt_fabric::PacketHeader) * receiver_completed_packet_header_cb_size_headers;
    static constexpr std::size_t sender_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t sender_completed_packet_header_cb_size_bytes =
        sizeof(tt::tt_fabric::PacketHeader) * sender_completed_packet_header_cb_size_headers;
    std::array<std::size_t, num_receiver_channels> receivers_completed_packet_header_cb_address = {};
    std::array<std::size_t, num_sender_channels> senders_completed_packet_header_cb_address = {};

    std::vector<FabricRiscConfig> risc_configs;
    // ----------- Sender Channels
    std::array<std::size_t, num_sender_channels> sender_channels_buffer_index_address = {};
    // Connection info layout:
    // 0: buffer_index_rdptr -> Tells EDM the address in worker L1 to update EDM's copy of channel rdptr
    // 1: worker_teardown_semaphore_address -> Tells EDM where to signal connection teardown completion in worker's L1
    // 2: WorkerXY (as uint32_t)
    // 3: Hold's EDM's rdptr for the buffer index in the channel
    std::array<std::size_t, num_sender_channels> sender_channels_worker_conn_info_base_address = {};
    std::array<std::size_t, num_sender_channels> sender_channels_local_flow_control_semaphore_address = {};
    std::array<std::size_t, num_sender_channels> sender_channels_producer_terminate_connection_address = {};
    // persistent mode field
    std::array<std::size_t, num_sender_channels> sender_channels_connection_semaphore_address = {};
    // persistent mode field
    std::array<std::size_t, num_sender_channels> sender_channels_buffer_index_semaphore_address = {};

    static_assert(sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo) % field_size == 0);

    // ----------- Receiver Channels
    std::array<std::size_t, max_downstream_edms> receiver_channels_local_buffer_index_address = {};
    // persistent mode field
    std::array<std::size_t, max_downstream_edms> receiver_channels_downstream_flow_control_semaphore_address = {};
    std::array<std::size_t, max_downstream_edms> receiver_channels_downstream_teardown_semaphore_address = {};

    // Conditionally used fields. BlackHole with 2-erisc uses these fields for sending credits back to sender.
    // We use/have these fields because we can't send reg-writes over Ethernet on both TXQs. Therefore,
    // use use a different crediting scheme.
    std::array<std::size_t, num_sender_channels> to_sender_channel_remote_ack_counter_addrs = {};
    std::array<std::size_t, num_sender_channels> to_sender_channel_remote_completion_counter_addrs = {};
    std::array<std::size_t, num_receiver_channels> receiver_channel_remote_ack_counter_addrs = {};
    std::array<std::size_t, num_receiver_channels> receiver_channel_remote_completion_counter_addrs = {};

    // Channel Allocations
    std::size_t max_l1_loading_size = 0;
    std::size_t buffer_region_start = 0;
    std::size_t available_channel_buffering_space = 0;

    FabricEriscDatamoverConfig(
        std::size_t channel_buffer_size_bytes,
        Topology topology = Topology::Linear,
        FabricEriscDatamoverOptions options = {});

    std::size_t channel_buffer_size_bytes = 0;

    std::array<std::size_t, num_sender_channels> sender_channels_size_bytes = {};
    std::array<std::size_t, num_receiver_channels> receiver_channels_size_bytes = {};
    std::array<std::size_t, num_sender_channels> sender_channels_num_buffers = {};
    std::array<std::size_t, num_receiver_channels> receiver_channels_num_buffers = {};

    // Remote channels sizes, used to calculate the remote buffer addresses.
    std::array<std::size_t, num_sender_channels> remote_sender_channels_size_bytes = {};
    std::array<std::size_t, num_receiver_channels> remote_receiver_channels_size_bytes = {};
    // Remote recv channels number of buffers, use by the local sender channel to check free slots.
    std::array<std::size_t, num_sender_channels> remote_sender_channels_num_buffers = {};
    std::array<std::size_t, num_receiver_channels> remote_receiver_channels_num_buffers = {};
    // Downstream sender channels number of buffers, used by the local receiver channel to check free slots.
    std::array<std::size_t, num_downstream_sender_channels> downstream_sender_channels_num_buffers = {};

    std::array<std::size_t, num_sender_channels> sender_channels_base_address = {};
    std::array<std::size_t, num_receiver_channels> receiver_channels_base_address = {};
    // the base addr per remote channel, used by local channels.
    std::array<std::size_t, num_sender_channels> remote_sender_channels_base_address = {};
    std::array<std::size_t, num_receiver_channels> remote_receiver_channels_base_address = {};

    std::size_t num_used_sender_channels = 0;
    std::size_t num_used_receiver_channels = 0;
    std::size_t num_fwd_paths = 0;
    std::size_t sender_txq_id = 0;
    std::size_t receiver_txq_id = 0;
    std::size_t num_riscv_cores = 0;

    Topology topology = Topology::Linear;

    // add the noc-usage and cmd_buf-usage here
    std::array<std::size_t, num_receiver_channels> receiver_channel_forwarding_noc_ids = {};
    std::array<std::size_t, num_receiver_channels> receiver_channel_forwarding_data_cmd_buf_ids = {};
    std::array<std::size_t, num_receiver_channels> receiver_channel_forwarding_sync_cmd_buf_ids = {};
    std::array<std::size_t, num_receiver_channels> receiver_channel_local_write_noc_ids = {};
    std::array<std::size_t, num_receiver_channels> receiver_channel_local_write_cmd_buf_ids = {};

    std::array<std::size_t, num_sender_channels> sender_channel_ack_noc_ids = {};
    std::array<std::size_t, num_sender_channels> sender_channel_ack_cmd_buf_ids = {};

    // Dateline Upstream EDM skip connection flag
    bool skip_sender_channel_1_connection = false;
    bool skip_receiver_channel_1_connection = false;
    bool skip_sender_vc1_channel_connection = false;

    // emd vcs
    std::size_t edm_noc_vc = 0;

private:
    void configure_buffer_slots_helper(
        Topology topology,
        const FabricEriscDatamoverOptions& options,
        std::array<size_t, num_sender_channels>& num_sender_buffer_slots,
        std::array<size_t, num_sender_channels>& num_remote_sender_buffer_slots,
        std::array<size_t, num_receiver_channels>& num_receiver_buffer_slots,
        std::array<size_t, num_receiver_channels>& num_remote_receiver_buffer_slots,
        std::array<size_t, num_downstream_sender_channels>& num_downstream_sender_buffer_slots);

    FabricEriscDatamoverConfig(Topology topology = Topology::Linear);
};

struct FabricRiscConfig {
    FabricRiscConfig(uint32_t risc_id);
    bool enable_handshake() const { return enable_handshake_; };
    bool enable_context_switch() const { return enable_context_switch_; };
    bool enable_interrupts() const { return enable_interrupts_; };
    size_t iterations_between_ctx_switch_and_teardown_checks() const {
        return iterations_between_ctx_switch_and_teardown_checks_;
    };
    bool is_sender_channel_serviced(int id) const { return is_sender_channel_serviced_[id]; };
    bool is_receiver_channel_serviced(int id) const { return is_receiver_channel_serviced_[id]; };

private:
    bool enable_handshake_ = false;
    bool enable_context_switch_ = false;
    bool enable_interrupts_ = false;
    size_t iterations_between_ctx_switch_and_teardown_checks_ = 0;
    std::array<bool, FabricEriscDatamoverConfig::num_sender_channels> is_sender_channel_serviced_;
    std::array<bool, FabricEriscDatamoverConfig::num_receiver_channels> is_receiver_channel_serviced_;
};

struct SenderWorkerAdapterSpec {
    size_t edm_noc_x = 0;
    size_t edm_noc_y = 0;
    size_t edm_buffer_base_addr = 0;
    size_t num_buffers_per_channel = 0;
    size_t edm_l1_sem_addr = 0;
    size_t edm_connection_handshake_addr = 0;
    size_t edm_worker_location_info_addr = 0;  // The EDM's location for `EDMChannelWorkerLocationInfo`
    size_t buffer_size_bytes = 0;
    size_t buffer_index_semaphore_id = 0;  // the semaphore ID on the EDM, not the worker
    eth_chan_directions edm_direction = eth_chan_directions::EAST;
};

struct edm_termination_info_t {
    uint32_t distance = 0;
    uint32_t edm_noc_x = 0;
    uint32_t edm_noc_y = 0;
    uint32_t termination_addr = 0;
};

void get_runtime_args_for_edm_termination_infos(
    const std::vector<edm_termination_info_t>& edm_termination_infos, std::vector<uint32_t>& args_out);
void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    size_t sender_worker_flow_control_semaphore_id,
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);

void append_worker_to_fabric_edm_sender_rt_args(
    tt::tt_fabric::chan_id_t eth_channel,
    size_t sender_worker_flow_control_semaphore_id,
    size_t sender_worker_teardown_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);

// TODO: will be deprecated
void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    chip_id_t chip_id,
    const CoreRangeSet& worker_cores,
    size_t sender_worker_flow_control_semaphore_id,
    size_t sender_worker_teardown_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);
size_t log_worker_to_fabric_edm_sender_rt_args(const std::vector<uint32_t>& args, size_t starting_arg_idx = 0);

class FabricEriscDatamoverBuilder {
public:
    static constexpr size_t default_firmware_context_switch_interval = 10000;
    static constexpr auto default_firmware_context_switch_type = FabricEriscDatamoverContextSwitchType::WAIT_FOR_IDLE;
    // payload only, no header
    static constexpr size_t default_packet_payload_size_bytes = tt::tile_size(tt::DataFormat::Bfp8_b) * 4;
    static constexpr size_t default_mesh_packet_payload_size_bytes = tt::tile_size(tt::DataFormat::Bfp8_b) * 2;

    FabricEriscDatamoverBuilder(
        const CoreCoord& my_eth_core_logical,
        size_t my_noc_x,
        size_t my_noc_y,
        const FabricNodeId& local_fabric_node_id,
        const FabricNodeId& peer_fabric_node_id,

        const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>&
            receiver_channels_downstream_flow_control_semaphore_id,
        const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>&
            receiver_channels_downstream_teardown_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_flow_control_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_connection_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_buffer_index_semaphore_id,

        const FabricEriscDatamoverConfig& config,
        eth_chan_directions direction,
        bool build_in_worker_connection_mode = false,
        bool dateline_connection = false);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const CoreCoord& ethernet_core,
        const FabricNodeId& local_fabric_node_id,
        const FabricNodeId& peer_fabric_node_id,
        const FabricEriscDatamoverConfig& config,
        bool build_in_worker_connection_mode = false,
        bool dateline_connection = false,
        eth_chan_directions direction = eth_chan_directions::EAST);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const CoreCoord& ethernet_core,
        chip_id_t local_physical_chip_id,
        chip_id_t peer_physical_chip_id,
        const FabricEriscDatamoverConfig& config,
        bool build_in_worker_connection_mode = false,
        bool dateline_connection = false,
        eth_chan_directions direction = eth_chan_directions::EAST);

    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_worker_channel() const;
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc);

    [[nodiscard]] std::vector<uint32_t> get_compile_time_args(uint32_t risc_id) const;

    [[nodiscard]] std::vector<uint32_t> get_runtime_args() const;

    void connect_to_downstream_edm(FabricEriscDatamoverBuilder& downstream_edm);

    eth_chan_directions get_direction() const;
    size_t get_configured_risc_count() const;
    size_t get_noc_x() const;
    size_t get_noc_y() const;

    void dump_to_log() const {
        // TODO
    }

    void teardown_from_host(
        tt::tt_metal::IDevice* d,
        tt::tt_fabric::TerminationSignal termination_signal =
            tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE) const;

    void set_firmware_context_switch_interval(size_t interval);
    void set_firmware_context_switch_type(FabricEriscDatamoverContextSwitchType type);
    void set_wait_for_host_signal(bool wait_for_host_signal);

    //    protected:
    friend class EdmLineFabricOpInterface;
    CoreCoord my_eth_core_logical;
    size_t my_noc_x = 0;
    size_t my_noc_y = 0;

    FabricEriscDatamoverConfig config;

    FabricNodeId local_fabric_node_id = FabricNodeId(MeshId{0}, 0);
    FabricNodeId peer_fabric_node_id = FabricNodeId(MeshId{0}, 0);
    size_t handshake_address = 0;
    size_t channel_buffer_size = 0;

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_num_buffers = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> receiver_channels_num_buffers = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> remote_receiver_channels_num_buffers = {};

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> local_sender_channels_buffer_address = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> local_receiver_channels_buffer_address = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> remote_sender_channels_base_address = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> remote_receiver_channels_base_address = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_downstream_sender_channels>
        downstream_sender_channels_num_buffers = {};

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> local_sender_channels_connection_info_addr = {};

    size_t termination_signal_ptr = 0;
    size_t edm_local_sync_ptr = 0;
    size_t edm_status_ptr = 0;
    eth_chan_directions direction = eth_chan_directions::EAST;
    size_t downstream_edms_connected = 0;

    // Semaphore IDs
    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        receiver_channels_downstream_flow_control_semaphore_id = {};
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        receiver_channels_downstream_teardown_semaphore_id = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_flow_control_semaphore_id = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_connection_semaphore_id = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_buffer_index_semaphore_id = {};
    std::array<size_t, FabricEriscDatamoverConfig::max_downstream_edms> receiver_channels_local_buffer_index_address =
        {};

    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms> downstream_edm_vcs_noc_x = {};
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms> downstream_edm_vcs_noc_y = {};
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        downstream_edm_vcs_buffer_base_address = {};
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        downstream_edm_vcs_semaphore_address = {};
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        downstream_edm_vcs_worker_registration_address = {};
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
        downstream_edm_vcs_worker_location_info_address = {};
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>
        downstream_vcs_sender_channel_buffer_index_semaphore_id = {};

    std::array<bool, FabricEriscDatamoverConfig::num_sender_channels>
        sender_channel_connection_liveness_check_disable_array = {};

    bool build_in_worker_connection_mode = false;
    size_t firmware_context_switch_interval = default_firmware_context_switch_interval;
    FabricEriscDatamoverContextSwitchType firmware_context_switch_type = default_firmware_context_switch_type;
    bool enable_first_level_ack = false;
    bool fuse_receiver_flush_and_completion_ptr = true;
    bool dateline_connection = false;
    bool wait_for_host_signal = false;
};

}  // namespace tt::tt_fabric
