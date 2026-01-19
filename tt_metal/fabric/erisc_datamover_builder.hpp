// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/assert.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>

#include <umd/device/types/cluster_descriptor_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/experimental/fabric/edm_fabric_counters.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>  // for FabricNodeId
#include <hostdevcommon/fabric_common.h>
#include <optional>
#include <cstdint>
#include <vector>
#include <array>
#include <cstddef>
#include <memory>
#include "builder/fabric_channel_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"
#include "tt_metal/fabric/fabric_datamover_builder_base.hpp"

namespace tt::tt_fabric {

struct FabricRiscConfig;
class FabricRouterBuilder;
class ComputeMeshRouterBuilder;
class MultiPoolChannelAllocator;
class ChannelToPoolMapping;
class FabricRemoteChannelsAllocator;

class FabricEriscDatamoverBuilder;
class FabricTensixDatamoverBuilder;

enum class FabricEriscDatamoverContextSwitchType : uint8_t {
    // Context switch at the interval only if idle for a certain number of cycles
    WAIT_FOR_IDLE = 0,
    // Context switch every interval
    INTERVAL = 1,
};

/*
Receiver channel side registers are defined here to receive free-slot credits from downstream sender channels.

                                North Router
                        ┌───────────────────────────────────┐
                        │                                   │
                        │  ┌────┐ ┌────┐ ┌────┐ ┌────┐      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  └────┘ └────┘ └────┘ └────┘      │
                        │  ┌────┐ ┌────┐ ┌────┐ ┌────┐      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  │    │ │    │ │    │ │    │      │
                        │  └────┘ └─┬──┘ └────┘ └────┘      │
    West Router         └───────────┼───────────────────────┘        East Router
 ┌─────────────────────┐            │                             ┌────────────────────────────┐
 │                     │            │                             │                            │
 │                     │            │                             │                            │
 │               ┌────┐│ (increment)│    Acks From East           │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    ◄┼────────────┼───────────────────┐         ││              │ │    │ E   │
 │     East      │    ││            │                   │         ││              │ │    │     │
 │               └────┘│            │                   │         │└──────────────┘ └────┘     │
 │                 12  │            │                   │         │                            │
 │               ┌────┐│            │                   │         │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    ││            │                   │         ││              │ │    │ W   │
 │     West      │    ││            │                   └─────────┼┼              │ │    │     │
 │               └────┘│            │                             │└──────────────┘ └────┘     │
 │                 13  │            │                             │                            │
 │               ┌────┐│ (increment)│                             │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    │◄────────────┘                             ││              │ │    │ N   │
 │     North     │    ││  Acks From North                         ││              │ │    │     │
 │               └────┘│                                          │└──────────────┘ └────┘     │
 │                 14  │                                          │                            │
 │               ┌────┐│  Acks From South                         │┌──────────────┐ ┌────┐     │
 │   Free Slots  │    │◄────────────────┐                         ││              │ │    │ S   │
 │     South     │    ││ (increment)    │                         ││              │ │    │     │
 │               └────┘│                │                         │└──────────────┘ └────┘     │
 │                 15  │                │                         │                            │
 │                     │                │                         │                            │
 │                     │                │                         │                            │
 └─────────────────────┘  ┌─────────────┼───────────────────┐     └────────────────────────────┘
                          │   ┌────┐ ┌──┼─┐ ┌────┐ ┌────┐   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   └────┘ └────┘ └────┘ └────┘   │
                          │   ┌────┐ ┌────┐ ┌────┐ ┌────┐   │
                          │   │    │ │    │ │    │ │    │   │
                          │   │    │ │    │ │    │ │    │   │
                          │   └────┘ └────┘ └────┘ └────┘   │
                          │                                 │
                          └─────────────────────────────────┘
                                   South Router
*/
struct StreamRegAssignments {
    // Packet send/ack/complete stream IDs
    static constexpr uint32_t to_receiver_0_pkts_sent_id = 0;      // VC0 Ethernet Rx
    static constexpr uint32_t to_receiver_1_pkts_sent_id = 1;      // VC1 Ethernet Rx
    static constexpr uint32_t to_sender_0_pkts_acked_id = 2;       // VC0 Ethernet Sender Channel 0
    static constexpr uint32_t to_sender_1_pkts_acked_id = 3;       // VC0 Ethernet Sender Channel 1
    static constexpr uint32_t to_sender_2_pkts_acked_id = 4;       // VC0 Ethernet Sender Channel 2
    static constexpr uint32_t to_sender_3_pkts_acked_id = 5;       // VC0 Ethernet Sender Channel 3
    static constexpr uint32_t to_sender_0_pkts_completed_id = 6;   // VC0 Tensix Worker on upstream device
    static constexpr uint32_t to_sender_1_pkts_completed_id = 7;   // VC0 Passthrough from upstream device X/Y edge
    static constexpr uint32_t to_sender_2_pkts_completed_id = 8;   // VC0 Passthrough from upstream device X/Y edge
    static constexpr uint32_t to_sender_3_pkts_completed_id = 9;   // VC0 Passthrough from upstream device X/Y edge
    static constexpr uint32_t to_sender_4_pkts_completed_id = 10;  // VC1 Passthrough from upstream device Z edge
    static constexpr uint32_t to_sender_5_pkts_completed_id = 11;  // VC1 Passthrough from upstream device X/Y edge
    static constexpr uint32_t to_sender_6_pkts_completed_id = 12;  // VC1 Passthrough from upstream device X/Y edge
    static constexpr uint32_t to_sender_7_pkts_completed_id = 13;  // VC1 Passthrough from upstream device X/Y edge
    // Receiver channel free slots stream IDs
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_1 =
        14;  // for downstream E/W/N/S edge on: 2D X/Y Router->VC0, E edge on: 2D Z Router->VC0
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_2 =
        15;  // for downstream E/W/N/S edge on: 2D X/Y Router->VC0, W edge on: 2D Z Router->VC0
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_3 =
        16;  // for downstream E/W/N/S edge on: 2D X/Y Router->VC0, N edge on: 2D Z Router->VC0
    static constexpr uint32_t vc_0_free_slots_from_downstream_edge_4 =
        17;  // for downstream Z edge on: 2D+Z X/Y Router->VC0, S edge on: 2D Z Router->VC0
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_1 =
        18;  // for downstream E/W/N/S edge on: 2D X/Y Router->VC1
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_2 =
        19;  // for downstream E/W/N/S edge on: 2D X/Y Router->VC1
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_3 =
        20;  // for downstream E/W/N/S edge on: 2D X/Y Router->VC1
    static constexpr uint32_t vc_1_free_slots_from_downstream_edge_4 =
        21;  // for downstream Z edge on: 2D+Z X/Y Router->VC1, S edge on: 2D Z Router->VC1
    // Sender channel free slots stream IDs.
    // Decremented by respective upstream senders.
    static constexpr uint32_t sender_channel_0_free_slots_stream_id = 22;  // for upstream tensix worker
    static constexpr uint32_t sender_channel_1_free_slots_stream_id =
        23;  // for upstream edge on: 1D->VC0, E/W/N/S edge on: 2D X/Y Router->VC0, E edge on: 2D Z Router->VC0
    static constexpr uint32_t sender_channel_2_free_slots_stream_id =
        24;  // for upstream E/W/N/S edge on: 2D X/Y Router->VC0, W edge on: 2D Z Router->VC0
    static constexpr uint32_t sender_channel_3_free_slots_stream_id =
        25;  // for upstream E/W/N/S edge on: 2D X/Y Router->VC0, N edge on: 2D Z Router->VC0
    static constexpr uint32_t sender_channel_4_free_slots_stream_id =
        26;  // for upstream E/W/N/S edge on: 2D X/Y Router->VC1, S edge on: 2D Z Router->VC0
    static constexpr uint32_t sender_channel_5_free_slots_stream_id =
        27;  // for upstream E/W/N/S edge on: 2D X/Y Router->VC1
    static constexpr uint32_t sender_channel_6_free_slots_stream_id =
        28;  // for upstream E/W/N/S edge on: 2D X/Y Router->VC1
    static constexpr uint32_t sender_channel_7_free_slots_stream_id = 29;  // for upstream Z edge on: 2D+Z->VC1

    // Local tensix relay free slots stream ID (UDM mode only)
    static constexpr uint32_t tensix_relay_local_free_slots_stream_id = 30;
    // Multi-RISC teardown synchronization stream ID
    // overlay scratch register
    static constexpr uint32_t multi_risc_teardown_sync_stream_id = 31;
    // Eth retrain synchronization stream ID
    // overlay scratch register
    static constexpr uint32_t eth_retrain_link_sync_stream_id = 30;

    static const auto& get_all_stream_ids() {
        static constexpr std::array stream_ids = {
            to_receiver_0_pkts_sent_id,
            to_receiver_1_pkts_sent_id,
            to_sender_0_pkts_acked_id,
            to_sender_1_pkts_acked_id,
            to_sender_2_pkts_acked_id,
            to_sender_3_pkts_acked_id,
            to_sender_0_pkts_completed_id,
            to_sender_1_pkts_completed_id,
            to_sender_2_pkts_completed_id,
            to_sender_3_pkts_completed_id,
            to_sender_4_pkts_completed_id,
            to_sender_5_pkts_completed_id,
            to_sender_6_pkts_completed_id,
            to_sender_7_pkts_completed_id,
            vc_0_free_slots_from_downstream_edge_1,
            vc_0_free_slots_from_downstream_edge_2,
            vc_0_free_slots_from_downstream_edge_3,
            vc_0_free_slots_from_downstream_edge_4,
            vc_1_free_slots_from_downstream_edge_1,
            vc_1_free_slots_from_downstream_edge_2,
            vc_1_free_slots_from_downstream_edge_3,
            vc_1_free_slots_from_downstream_edge_4,
            sender_channel_0_free_slots_stream_id,
            sender_channel_1_free_slots_stream_id,
            sender_channel_2_free_slots_stream_id,
            sender_channel_3_free_slots_stream_id,
            sender_channel_4_free_slots_stream_id,
            sender_channel_5_free_slots_stream_id,
            sender_channel_6_free_slots_stream_id,
            sender_channel_7_free_slots_stream_id,
            tensix_relay_local_free_slots_stream_id,
            multi_risc_teardown_sync_stream_id,
            eth_retrain_link_sync_stream_id};
        return stream_ids;
    }
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
    static constexpr uint32_t BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_FORWARDING_NOC = 1;
    static constexpr uint32_t BLACKHOLE_SINGLE_ERISC_MODE_RECEIVER_LOCAL_WRITE_NOC = 1;
    static constexpr uint32_t BLACKHOLE_SINGLE_ERISC_MODE_SENDER_ACK_NOC = 1;

    static constexpr std::size_t field_size = 16;
    static constexpr std::size_t buffer_alignment = 32;
    static constexpr std::size_t eth_word_l1_alignment = 16;
    static constexpr uint32_t default_iterations_between_ctx_switch_and_teardown_checks = 32;
    static_assert(((buffer_alignment - 1) & buffer_alignment) == 0);

    // Global
    static constexpr std::size_t eth_channel_sync_size = 16;
    std::size_t handshake_addr = 0;
    std::size_t edm_channel_ack_addr = 0;
    std::size_t termination_signal_address = 0;  // pad extra bytes to match old EDM so handshake logic will still work
    std::size_t edm_local_sync_address = 0;
    std::size_t edm_local_tensix_sync_address = 0;
    std::size_t edm_status_address = 0;
    std::size_t notify_worker_of_read_counter_update_src_address = 0;

    // Performance telemetry buffer address (16B aligned)
    std::size_t perf_telemetry_buffer_address = 0;

    // Code profiling buffer address (16B aligned)
    std::size_t code_profiling_buffer_address = 0;

    std::vector<FabricRiscConfig> risc_configs;
    // ----------- Sender Channels
    std::array<std::size_t, builder_config::num_max_sender_channels> sender_channels_buffer_index_address = {};
    // Connection info layout:
    // 0: buffer_index_rdptr -> Tells EDM the address in worker L1 to update EDM's copy of channel rdptr
    // 1: worker_teardown_semaphore_address -> Tells EDM where to signal connection teardown completion in worker's L1
    // 2: WorkerXY (as uint32_t)
    // 3: Hold's EDM's rdptr for the buffer index in the channel
    std::array<std::size_t, builder_config::num_max_sender_channels> sender_channels_worker_conn_info_base_address = {};
    std::array<std::size_t, builder_config::num_max_sender_channels>
        sender_channels_local_flow_control_semaphore_address = {};
    std::array<std::size_t, builder_config::num_max_sender_channels>
        sender_channels_producer_terminate_connection_address = {};
    // persistent mode field
    std::array<std::size_t, builder_config::num_max_sender_channels> sender_channels_connection_semaphore_address = {};
    // persistent mode field
    std::array<std::size_t, builder_config::num_max_sender_channels> sender_channels_buffer_index_semaphore_address =
        {};

    static_assert(sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo) % field_size == 0);

    // ----------- Receiver Channels
    // persistent mode field
    std::array<std::size_t, builder_config::max_downstream_edms>
        receiver_channels_downstream_flow_control_semaphore_address = {};
    std::array<std::size_t, builder_config::max_downstream_edms>
        receiver_channels_downstream_teardown_semaphore_address = {};

    // Conditionally used fields. BlackHole with 2-erisc uses these fields for sending credits back to sender.
    // We use/have these fields because we can't send reg-writes over Ethernet on both TXQs. Therefore,
    // use use a different crediting scheme.
    size_t to_sender_channel_remote_ack_counters_base_addr = 0;
    size_t to_sender_channel_remote_completion_counters_base_addr = 0;
    size_t receiver_channel_remote_ack_counters_base_addr = 0;
    size_t receiver_channel_remote_completion_counters_base_addr = 0;
    size_t router_buffer_clear_size_words = 1;

    // ----------- Local Tensix Relay Connection (UDM mode only)
    // Connection buffer index for the local tensix relay interface
    size_t tensix_relay_connection_buffer_index_id = 0;

    // Channel Allocations
    std::size_t max_l1_loading_size = 0;
    std::vector<MemoryRegion> available_buffer_memory_regions;

    FabricEriscDatamoverConfig(
        std::size_t channel_buffer_size_bytes,
        Topology topology,
        FabricEriscDatamoverOptions options,
        const std::array<std::size_t, builder_config::MAX_NUM_VCS>& sender_channels_per_vc,
        const std::array<std::size_t, builder_config::MAX_NUM_VCS>& receiver_channels_per_vc);

    std::size_t channel_buffer_size_bytes = 0;

    std::size_t num_used_sender_channels = 0;    // Total across all VCs (duplicate in allocator... don't modify)
    std::size_t num_used_receiver_channels = 0;  // Total across all VCs (duplicate in allocator... don't modify)
    std::array<std::size_t, builder_config::MAX_NUM_VCS> num_used_sender_channels_per_vc = {0, 0};    // Per-VC sender channel counts
    std::array<std::size_t, builder_config::MAX_NUM_VCS> num_used_receiver_channels_per_vc = {0, 0};  // Per-VC receiver channel counts
    std::size_t num_fwd_paths = 0;
    std::size_t sender_txq_id = 0;
    std::size_t receiver_txq_id = 0;
    std::size_t num_riscv_cores = 0;

    Topology topology = Topology::Linear;

    // add the noc-usage and cmd_buf-usage here
    std::array<std::size_t, builder_config::num_max_receiver_channels> receiver_channel_forwarding_noc_ids = {};
    std::array<std::size_t, builder_config::num_max_receiver_channels> receiver_channel_forwarding_data_cmd_buf_ids =
        {};
    std::array<std::size_t, builder_config::num_max_receiver_channels> receiver_channel_forwarding_sync_cmd_buf_ids =
        {};
    std::array<std::size_t, builder_config::num_max_receiver_channels> receiver_channel_local_write_noc_ids = {};
    std::array<std::size_t, builder_config::num_max_receiver_channels> receiver_channel_local_write_cmd_buf_ids = {};

    std::array<std::size_t, builder_config::num_max_sender_channels> sender_channel_ack_noc_ids = {};
    std::array<std::size_t, builder_config::num_max_sender_channels> sender_channel_ack_cmd_buf_ids = {};

    // emd vcs
    std::size_t edm_noc_vc = 0;

    // Fabric channel allocator for L1 memory management
    // Points to the primary allocator (typically static allocator for single-pool configs)
    std::shared_ptr<FabricChannelAllocator> channel_allocator;

    // Multi-pool allocator coordinator - manages all pool allocators
    // Emits pool metadata and delegates to individual pools for CT args
    std::shared_ptr<MultiPoolChannelAllocator> multi_pool_allocator;

    // Channel-to-pool mapping for multi-pool support
    std::shared_ptr<ChannelToPoolMapping> channel_to_pool_mapping;
    // Channel-to-pool mapping for remote (over eth) channels multi-pool support
    std::shared_ptr<ChannelToPoolMapping> remote_channel_to_pool_mapping;

    // Remote channels allocator - tracks remote receiver channel info for the remote ethernet core
    std::shared_ptr<FabricRemoteChannelsAllocator> remote_channels_allocator;

private:
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
    tt::tt_metal::NOC get_configured_noc() const { return noc_; };

    void set_configured_noc(tt::tt_metal::NOC noc) { noc_ = noc; };
    bool telemetry_enabled() const { return telemetry_enabled_; }
    void set_telemetry_enabled(bool enabled) { telemetry_enabled_ = enabled; }
    uint8_t telemetry_stats_mask() const { return telemetry_stats_mask_; }
    void set_telemetry_stats_mask(uint8_t mask) { telemetry_stats_mask_ = mask; }

private:
    tt::tt_metal::NOC noc_ = tt::tt_metal::NOC::NOC_0;
    size_t iterations_between_ctx_switch_and_teardown_checks_ = 0;
    bool enable_handshake_ = false;
    bool enable_context_switch_ = false;
    bool enable_interrupts_ = false;
    bool telemetry_enabled_ = true;
    uint8_t telemetry_stats_mask_ = 0xFF;
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
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);

// TODO: will be deprecated
void append_worker_to_fabric_edm_sender_rt_args(
    const SenderWorkerAdapterSpec& connection,
    ChipId chip_id,
    const CoreRangeSet& worker_cores,
    size_t sender_worker_terminate_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);
size_t log_worker_to_fabric_edm_sender_rt_args(const std::vector<uint32_t>& args, size_t starting_arg_idx = 0);

/*
 * The `FabricEriscDatamoverBuilder` is a general class that is used to build fabric router erisc kernels.
 * It is instantiated per fabric (erisc) router. It works closely with the `FabricEriscDatamoverConfig` class.
 *
 * Note on 2-ERISC enablement (Blackhole):
 *   Builder logic may enable an effective "fabric 2-ERISC" mode by default when the platform exposes
 *   two ERISCs on the Ethernet core and Fabric Tensix MUX is enabled. Decisions such as ERISC count and
 *   TXQ selection are derived from this effective mode. A presence-based disable env exists to force-disable.
 */
class FabricEriscDatamoverBuilder : public FabricDatamoverBuilderBase {
    friend class FabricRouterBuilder;
    friend class ComputeMeshRouterBuilder;

public:
    static constexpr size_t default_firmware_context_switch_interval = 10000;
    static constexpr auto default_firmware_context_switch_type = FabricEriscDatamoverContextSwitchType::WAIT_FOR_IDLE;

    // Default packet payload sizes (optimized for 4 tiles of Bfp8_b)
    // Users can configure larger sizes up to architecture-specific maximums
    // via FabricRouterConfig in SetFabricConfig()
    // payload only, no header
    static constexpr size_t default_packet_payload_size_bytes = tt::tile_size(tt::DataFormat::Bfp8_b) * 4;
    static constexpr size_t default_mesh_packet_payload_size_bytes = tt::tile_size(tt::DataFormat::Bfp8_b) * 4;

    // Architecture-specific maximum packet payload size limits
    //
    // Calculated from NoC constraints:
    //   max_payload = floor((max_noc_packet_size - max_packet_header_size) / tile_size) * tile_size
    //
    // Where:
    //   - Max NoC packet size: Wormhole = 8192 bytes, Blackhole = 16384 bytes
    //   - Max packet header size: 96 bytes (HybridMeshPacketHeaderT<35> for 2D mesh routing)
    //   - Tile size (Bfp8_b): 1088 bytes
    //
    // Payload is rounded down to tile boundaries for efficient tile-aligned transfers.
    //
    // Wormhole:  (8192 - 96) / 1088 = 7.44 tiles → 7 tiles = 7616 bytes
    // Blackhole: (16384 - 96) / 1088 = 14.97 tiles → 14 tiles = 15232 bytes
    static constexpr size_t max_packet_payload_size_bytes_wormhole =
        tt::tile_size(tt::DataFormat::Bfp8_b) * 7;  // 7616 bytes
    static constexpr size_t max_packet_payload_size_bytes_blackhole =
        tt::tile_size(tt::DataFormat::Bfp8_b) * 14;  // 15232 bytes

    static_assert(default_packet_payload_size_bytes == 4352, "Packet size must be 4352 bytes");
    static_assert(default_mesh_packet_payload_size_bytes == 4352, "Mesh packet size must be 4352 bytes");

    // Get architecture-specific maximum packet payload size
    static size_t get_max_packet_payload_size_for_arch(tt::ARCH arch);

    FabricEriscDatamoverBuilder(
        const CoreCoord& my_eth_core_logical,
        size_t my_noc_x,
        size_t my_noc_y,
        const FabricNodeId& local_fabric_node_id,
        const FabricNodeId& peer_fabric_node_id,

        const std::array<std::optional<size_t>, builder_config::max_downstream_edms>&
            receiver_channels_downstream_flow_control_semaphore_id,
        const std::array<std::optional<size_t>, builder_config::max_downstream_edms>&
            receiver_channels_downstream_teardown_semaphore_id,
        const std::array<size_t, builder_config::num_max_sender_channels>& sender_channels_flow_control_semaphore_id,
        const std::array<size_t, builder_config::num_max_sender_channels>& sender_channels_connection_semaphore_id,
        const std::array<size_t, builder_config::num_max_sender_channels>& sender_channels_buffer_index_semaphore_id,

        const FabricEriscDatamoverConfig& config,
        eth_chan_directions direction,
        std::vector<bool>&& sender_channel_injection_flags,
        bool build_in_worker_connection_mode = false,
        bool has_tensix_extension = false,
        std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_sender_channels_per_vc = std::nullopt,
        std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_receiver_channels_per_vc = std::nullopt);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const CoreCoord& ethernet_core,
        const FabricNodeId& local_fabric_node_id,
        const FabricNodeId& peer_fabric_node_id,
        const FabricEriscDatamoverConfig& config,
        std::vector<bool>&& sender_channel_injection_flags,
        bool build_in_worker_connection_mode = false,
        eth_chan_directions direction = eth_chan_directions::EAST,
        bool has_tensix_extension = false,
        std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_sender_channels_per_vc = std::nullopt,
        std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_receiver_channels_per_vc = std::nullopt);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const CoreCoord& ethernet_core,
        ChipId local_physical_chip_id,
        ChipId peer_physical_chip_id,
        const FabricEriscDatamoverConfig& config,
        std::vector<bool>&& sender_channel_injection_flags,
        bool build_in_worker_connection_mode = false,
        eth_chan_directions direction = eth_chan_directions::EAST,
        bool has_tensix_extension = false,
        std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_sender_channels_per_vc = std::nullopt,
        std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_receiver_channels_per_vc = std::nullopt);

    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_worker_channel() const;
    // Overload that accepts VC, absolute channel ID, and VC-relative channel ID
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel(
        uint32_t vc, uint32_t absolute_channel_id, uint32_t vc_relative_channel_id) const;
    // Base class override (for backward compatibility, treats channel_id as VC0-relative)
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const override;
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc, uint32_t ds_edm) const;

    [[nodiscard]] std::vector<uint32_t> get_compile_time_args(uint32_t risc_id) const;

    // Helper for `get_compile_time_args`
    void get_telemetry_compile_time_args(uint32_t risc_id, std::vector<uint32_t>& ct_args) const;

    [[nodiscard]] std::vector<uint32_t> get_runtime_args() const;

    void connect_to_downstream_edm(FabricDatamoverBuilderBase* downstream_builder);

    size_t get_configured_risc_count() const;

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

    bool is_first_level_ack_enabled() const { return this->enable_first_level_ack; }

    //    protected:
    CoreCoord my_eth_core_logical;
    chan_id_t my_eth_channel;

    FabricEriscDatamoverConfig config;

    FabricNodeId local_fabric_node_id = FabricNodeId(MeshId{0}, 0);
    FabricNodeId peer_fabric_node_id = FabricNodeId(MeshId{0}, 0);
    bool is_inter_mesh = false;  // True if this data mover connects to a different mesh (inter-mesh router)
    size_t handshake_address = 0;
    size_t channel_buffer_size = 0;

    std::shared_ptr<tt::tt_fabric::ChannelConnectionWriterAdapter> receiver_channel_to_downstream_adapter;
    std::array<std::shared_ptr<tt::tt_fabric::FabricChannelAllocator>, builder_config::max_downstream_edms>
        downstream_allocators = {};

    std::array<size_t, builder_config::num_max_receiver_channels> receiver_channels_num_buffers = {};
    std::array<size_t, builder_config::num_max_receiver_channels> remote_receiver_channels_num_buffers = {};
    std::array<size_t, builder_config::num_max_receiver_channels> local_receiver_channels_buffer_address = {};
    std::array<size_t, builder_config::num_max_receiver_channels> remote_receiver_channels_base_address = {};

    std::array<size_t, builder_config::num_max_sender_channels> local_sender_channels_connection_info_addr = {};

    size_t termination_signal_ptr = 0;
    size_t edm_local_sync_ptr = 0;
    size_t edm_local_tensix_sync_ptr = 0;
    size_t edm_status_ptr = 0;
    size_t downstream_edms_connected = 0;

    // Semaphore IDs
    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    std::array<std::optional<size_t>, builder_config::max_downstream_edms>
        receiver_channels_downstream_flow_control_semaphore_id = {};
    std::array<std::optional<size_t>, builder_config::max_downstream_edms>
        receiver_channels_downstream_teardown_semaphore_id = {};
    std::array<size_t, builder_config::num_max_sender_channels> sender_channels_flow_control_semaphore_id = {};
    std::array<size_t, builder_config::num_max_sender_channels> sender_channels_connection_semaphore_id = {};
    std::array<size_t, builder_config::num_max_sender_channels> sender_channels_buffer_index_semaphore_id = {};

    std::array<size_t, builder_config::num_max_sender_channels>
        downstream_vcs_sender_channel_buffer_index_semaphore_id = {};

    mutable std::array<bool, builder_config::num_max_sender_channels>
        sender_channel_connection_liveness_check_disable_array = {};

    mutable std::vector<bool> sender_channel_is_traffic_injection_channel_array;

    // Actual channel counts per VC for this specific router (may differ from config max)
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_sender_channels_per_vc_ = std::nullopt;
    std::optional<std::array<std::size_t, builder_config::MAX_NUM_VCS>> actual_receiver_channels_per_vc_ = std::nullopt;

    bool build_in_worker_connection_mode = false;
    size_t firmware_context_switch_interval = default_firmware_context_switch_interval;
    FabricEriscDatamoverContextSwitchType firmware_context_switch_type = default_firmware_context_switch_type;
    bool fuse_receiver_flush_and_completion_ptr = true;
    bool wait_for_host_signal = false;
    bool has_tensix_extension = false;
    uint32_t num_downstream_tensix_connections = 0;
    bool udm_mode = false;                        // UDM mode: router connects to local tensix relay
    uint32_t local_tensix_relay_num_buffers = 0;  // Number of buffers in the local relay channel

private:
    // Per-RISC channel servicing flags [risc_id][channel_id]
    std::array<std::array<bool, builder_config::num_max_sender_channels>, builder_config::MAX_NUM_VCS> is_sender_channel_serviced_{};
    std::array<std::array<bool, builder_config::num_max_receiver_channels>, builder_config::MAX_NUM_VCS> is_receiver_channel_serviced_{};

    // first level acks are acknowledgement credits sent from receiver to sender channels on receipt of packets
    // and can be used to know when the sender is able to recover a buffer slot in the channel, for new data from
    // its producer(s).
    // First level acks are required for any topologies that are using bubble flow control (e.g. ring/torus topologies)
    // it is optional for other topologies and usually hurts performance in those other cases due to added CPU overheads
    bool enable_first_level_ack = false;

    // Shared helper for setting up VC connections
    // upstream_vc_idx: VC of this router's receiver channel
    // downstream_vc_idx: VC of downstream router's sender channel
    // absolute_channel_id: flattened channel index across all VCs (for flat arrays)
    // vc_relative_channel_id: 0-based index within the VC (for allocator calls)
    // For normal connections: upstream_vc_idx == downstream_vc_idx
    // For crossover (inter-mesh to intra-mesh): upstream_vc_idx=0, downstream_vc_idx=1
    void setup_downstream_vc_connection(
        FabricDatamoverBuilderBase* downstream_builder,
        uint32_t upstream_vc_idx,
        uint32_t downstream_vc_idx,
        uint32_t absolute_channel_id,
        uint32_t vc_relative_channel_id);

    // Internal implementation for connect_to_downstream_edm
    void connect_to_downstream_edm_impl(FabricDatamoverBuilderBase* downstream_builder);

    // Configure telemetry settings for all RISC cores
    void configure_telemetry_settings();
};

}  // namespace tt::tt_fabric
