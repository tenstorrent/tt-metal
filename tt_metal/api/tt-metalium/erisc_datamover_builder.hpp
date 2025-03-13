// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program_impl.hpp>
#include <tt-metalium/hal_exp.hpp>

#include "umd/device/types/cluster_descriptor_types.h"
#include "fabric_edm_types.hpp"
#include "fabric_edm_packet_header.hpp"
#include "edm_fabric_counters.hpp"

#include <unordered_map>
#include <optional>
#include <cstdint>
#include <vector>

namespace tt::tt_fabric {

struct FabricEriscDatamoverConfig {
    static constexpr std::size_t num_sender_channels = 3;
    static constexpr std::size_t num_receiver_channels = 2;
    static constexpr bool constrain_to_power_of_2_buffer_slot_counts = true;

    static constexpr std::size_t field_size = 16;
    static constexpr std::size_t buffer_alignment = 32;
    static constexpr std::size_t eth_word_l1_alignment = 16;
    static_assert(((buffer_alignment - 1) & buffer_alignment) == 0);
    static constexpr bool enable_fabric_counters = false;
    static constexpr bool enable_fabric_pkt_header_recording = false;

    // Global
    static constexpr std::size_t eth_channel_sync_size = 16;
    std::size_t handshake_addr;
    std::size_t edm_channel_ack_addr;
    std::size_t termination_signal_address;  // pad extra bytes to match old EDM so handshake logic will still work
    std::size_t edm_local_sync_address;
    std::size_t edm_status_address;

    // Debug and Counters
    static constexpr std::size_t receiver_channel_counters_size_bytes =
        (((tt::tt_fabric::receiver_channel_counters_l1_size - 1) / field_size) + 1) * field_size;
    static constexpr std::size_t sender_channel_counters_size_bytes =
        (((tt::tt_fabric::sender_channel_counters_l1_size - 1) / field_size) + 1) * field_size;

    std::array<std::size_t, num_receiver_channels> receiver_channels_counters_address;
    std::array<std::size_t, num_sender_channels> sender_channels_counters_address;

    // Packet header history buffer(s)
    static constexpr std::size_t receiver_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t receiver_completed_packet_header_cb_size_bytes =
        sizeof(tt::tt_fabric::PacketHeader) * receiver_completed_packet_header_cb_size_headers;
    static constexpr std::size_t sender_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t sender_completed_packet_header_cb_size_bytes =
        sizeof(tt::tt_fabric::PacketHeader) * sender_completed_packet_header_cb_size_headers;
    std::array<std::size_t, num_receiver_channels> receivers_completed_packet_header_cb_address;
    std::array<std::size_t, num_sender_channels> senders_completed_packet_header_cb_address;

    // ----------- Sender Channels
    std::array<std::size_t, num_sender_channels> sender_channels_buffer_index_address;
    // Connection info layout:
    // 0: buffer_index_rdptr -> Tells EDM the address in worker L1 to update EDM's copy of channel rdptr
    // 1: worker_teardown_semaphore_address -> Tells EDM where to signal connection teardown completion in worker's L1
    // 2: WorkerXY (as uint32_t)
    // 3: Hold's EDM's rdptr for the buffer index in the channel
    std::array<std::size_t, num_sender_channels> sender_channels_worker_conn_info_base_address;
    std::array<std::size_t, num_sender_channels> sender_channels_local_flow_control_semaphore_address;
    std::array<std::size_t, num_sender_channels> sender_channels_producer_terminate_connection_address;
    // persistent mode field
    std::array<std::size_t, num_sender_channels> sender_channels_connection_semaphore_address;
    // persistent mode field
    std::array<std::size_t, num_sender_channels> sender_channels_buffer_index_semaphore_address;

    static_assert(sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo) % field_size == 0);

    // ----------- Receiver Channels
    std::array<std::size_t, num_receiver_channels> receiver_channels_local_buffer_index_address;
    // persistent mode field
    std::array<std::size_t, num_receiver_channels> receiver_channels_downstream_flow_control_semaphore_address;

    // Channel Allocations
    std::size_t max_l1_loading_size;
    ;
    std::size_t buffer_region_start;
    std::size_t available_channel_buffering_space;

    FabricEriscDatamoverConfig(
        std::size_t channel_buffer_size_bytes,
        std::size_t sender_ratio_size,
        std::size_t receiver_ratio_size,
        Topology topology = Topology::Linear);

    std::size_t channel_buffer_size_bytes = 0;
    std::size_t channel_buffer_size_bytes_with_channel_sync = 0;

    std::array<std::size_t, num_sender_channels> sender_channels_size_bytes;
    std::array<std::size_t, num_receiver_channels> receiver_channels_size_bytes;
    std::array<std::size_t, num_sender_channels> sender_channels_num_buffers;
    std::array<std::size_t, num_receiver_channels> receiver_channels_num_buffers;

    std::array<std::size_t, num_sender_channels> sender_channels_base_address;
    std::array<std::size_t, num_receiver_channels> receiver_channels_base_address;

    std::size_t num_used_sender_channels = 0;
    std::size_t num_used_receiver_channels = 0;

    Topology topology = Topology::Linear;

private:
    FabricEriscDatamoverConfig();
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
    bool persistent_fabric = false;
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
    size_t sender_worker_teardown_semaphore_id,
    size_t sender_worker_buffer_index_semaphore_id,
    std::vector<uint32_t>& args_out);
size_t log_worker_to_fabric_edm_sender_rt_args(const std::vector<uint32_t>& args, size_t starting_arg_idx = 0);

class FabricEriscDatamoverBuilder {
public:
    static constexpr size_t default_firmware_context_switch_interval = 10000;
    // payload only, no header
    static constexpr size_t default_packet_payload_size_bytes = tt::tile_size(tt::DataFormat::Bfp8_b) * 4;

    static constexpr uint32_t num_virtual_channels = 2;

    FabricEriscDatamoverBuilder(
        const CoreCoord& my_eth_core_logical,
        size_t my_noc_x,
        size_t my_noc_y,
        size_t my_chip_id,
        size_t peer_chip_id,

        const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>&
            receiver_channels_downstream_flow_control_semaphore_id,
        const std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>&
            receiver_channels_downstream_teardown_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_flow_control_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_connection_semaphore_id,
        const std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels>&
            sender_channels_buffer_index_semaphore_id,

        const FabricEriscDatamoverConfig& config,
        bool enable_persistent_mode,
        bool build_in_worker_connection_mode = false,
        bool dateline_connection = false);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const CoreCoord& ethernet_core,
        chip_id_t local_chip_id,
        chip_id_t peer_chip_id,
        const FabricEriscDatamoverConfig& config,
        bool enable_persistent_mode,
        bool build_in_worker_connection_mode = false,
        bool dateline_connection = false);

    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_worker_channel() const;
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t vc) const;

    [[nodiscard]] std::vector<uint32_t> get_compile_time_args() const;

    [[nodiscard]] std::vector<uint32_t> get_runtime_args() const;

    void connect_to_downstream_edm(const FabricEriscDatamoverBuilder& downstream_edm);

    void dump_to_log() const {
        // TODO
    }

    void teardown_from_host(
        tt::tt_metal::IDevice* d,
        tt::tt_fabric::TerminationSignal termination_signal =
            tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE) const;

    void set_firmware_context_switch_interval(size_t interval);

    //    protected:
    friend class EdmLineFabricOpInterface;
    CoreCoord my_eth_core_logical;
    size_t my_noc_x = 0;
    size_t my_noc_y = 0;

    FabricEriscDatamoverConfig config;

    size_t my_chip_id = 0;
    size_t peer_chip_id = 0;
    size_t handshake_address = 0;
    size_t channel_buffer_size = 0;

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_num_buffers;
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> receiver_channels_num_buffers;

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> local_sender_channels_buffer_address;
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> local_receiver_channels_buffer_address;

    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> local_sender_channels_connection_info_addr;

    size_t termination_signal_ptr = 0;
    size_t edm_local_sync_ptr = 0;
    size_t edm_status_ptr = 0;

    // Semaphore IDs
    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>
        receiver_channels_downstream_flow_control_semaphore_id;
    std::array<std::optional<size_t>, FabricEriscDatamoverConfig::num_receiver_channels>
        receiver_channels_downstream_teardown_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_flow_control_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_connection_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_sender_channels> sender_channels_buffer_index_semaphore_id;
    std::array<size_t, FabricEriscDatamoverConfig::num_receiver_channels> receiver_channels_local_buffer_index_address;

    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_noc_x;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_noc_y;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_buffer_base_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_semaphore_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_worker_registration_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_edm_vcs_worker_location_info_address;
    std::array<std::optional<size_t>, num_virtual_channels> downstream_vcs_sender_channel_buffer_index_semaphore_id;

    bool enable_persistent_mode = false;
    bool build_in_worker_connection_mode = false;
    size_t firmware_context_switch_interval = default_firmware_context_switch_interval;
    bool enable_first_level_ack = false;
    bool fuse_receiver_flush_and_completion_ptr = true;
    bool dateline_connection = false;
};

}  // namespace tt::tt_fabric
