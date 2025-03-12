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
    static constexpr bool constrain_to_power_of_2_buffer_slot_counts = true;

    static constexpr std::size_t field_size = 16;
    static constexpr std::size_t buffer_alignment = 32;
    static constexpr std::size_t eth_word_l1_alignment = 16;
    static_assert(((buffer_alignment - 1) & buffer_alignment) == 0);
    static constexpr bool enable_fabric_counters = false;
    static constexpr bool enable_fabric_pkt_header_recording = false;

    // Global
    static constexpr std::size_t eth_channel_sync_size = 16;
    std::size_t handshake_addr = tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base() /* + 1024*/;
    std::size_t edm_channel_ack_addr = handshake_addr + eth_channel_sync_size;
    std::size_t termination_signal_address =
        edm_channel_ack_addr +
        (4 * eth_channel_sync_size);  // pad extra bytes to match old EDM so handshake logic will still work
    std::size_t edm_status_address = termination_signal_address + field_size;

    // Debug and Counters
    static constexpr std::size_t receiver_channel_counters_size_bytes =
        (((tt::tt_fabric::receiver_channel_counters_l1_size - 1) / field_size) + 1) * field_size;
    static constexpr std::size_t sender_channel_counters_size_bytes =
        (((tt::tt_fabric::sender_channel_counters_l1_size - 1) / field_size) + 1) * field_size;

    std::size_t receiver_channel_counters_address = edm_status_address + field_size;
    std::size_t sender_channel_0_counters_address =
        receiver_channel_counters_address + receiver_channel_counters_size_bytes;
    std::size_t sender_channel_1_counters_address =
        sender_channel_0_counters_address + sender_channel_counters_size_bytes;

    // Packet header history buffer(s)
    static constexpr std::size_t receiver_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t receiver_completed_packet_header_cb_size_bytes =
        sizeof(tt::tt_fabric::PacketHeader) * receiver_completed_packet_header_cb_size_headers;
    static constexpr std::size_t sender_completed_packet_header_cb_size_headers = 32;
    static constexpr std::size_t sender_completed_packet_header_cb_size_bytes =
        sizeof(tt::tt_fabric::PacketHeader) * sender_completed_packet_header_cb_size_headers;
    std::size_t receiver_completed_packet_header_cb_address =
        sender_channel_1_counters_address + sender_channel_counters_size_bytes;
    std::size_t sender_0_completed_packet_header_cb_address =
        receiver_completed_packet_header_cb_address + receiver_completed_packet_header_cb_size_bytes;
    std::size_t sender_1_completed_packet_header_cb_address =
        sender_0_completed_packet_header_cb_address + sender_completed_packet_header_cb_size_bytes;

    // ----------- Sender Channel 0
    std::size_t sender_channel_0_buffer_index_address =
        sender_1_completed_packet_header_cb_address + sender_completed_packet_header_cb_size_bytes;
    // Connection info layout:
    // 0: buffer_index_rdptr -> Tells EDM the address in worker L1 to update EDM's copy of channel rdptr
    // 1: worker_teardown_semaphore_address -> Tells EDM where to signal connection teardown completion in worker's L1
    // 2: WorkerXY (as uint32_t)
    // 3: Hold's EDM's rdptr for the buffer index in the channel
    std::size_t sender_channel_0_worker_conn_info_base_address = sender_channel_0_buffer_index_address + field_size;
    std::size_t sender_channel_0_local_flow_control_semaphore_address =
        sender_channel_0_worker_conn_info_base_address + sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo);
    // sender_channel_0_conn_info_edm_rdptr_address_address + field_size;
    std::size_t sender_channel_0_producer_terminate_connection_address =
        sender_channel_0_local_flow_control_semaphore_address + field_size;
    // persistent mode field
    std::size_t sender_channel_0_connection_semaphore_address =
        sender_channel_0_producer_terminate_connection_address + field_size;
    // persistent mode field
    std::size_t sender_channel_0_buffer_index_semaphore_address =
        sender_channel_0_connection_semaphore_address + field_size;

    static_assert(sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo) % field_size == 0);

    // ----------- Sender Channel 1
    std::size_t sender_channel_1_buffer_index_address = sender_channel_0_buffer_index_semaphore_address + field_size;
    // Connection info layout:
    // 0: buffer_index_rdptr -> Tells EDM the address in worker L1 to update EDM's copy of channel rdptr
    // 1: worker_teardown_semaphore_address -> Tells EDM where to signal connection teardown completion in worker's L1
    // 2: WorkerXY (as uint32_t)
    // 3: Hold's EDM's rdptr for the buffer index in the channel
    std::size_t sender_channel_1_worker_conn_info_base_address = sender_channel_1_buffer_index_address + field_size;
    std::size_t sender_channel_1_local_flow_control_semaphore_address =
        sender_channel_1_worker_conn_info_base_address + sizeof(tt::tt_fabric::EDMChannelWorkerLocationInfo);
    // sender_channel_1_conn_info_edm_rdptr_address_address + field_size;
    std::size_t sender_channel_1_producer_terminate_connection_address =
        sender_channel_1_local_flow_control_semaphore_address + field_size;

    // persistent mode field
    std::size_t sender_channel_1_connection_semaphore_address =
        sender_channel_1_producer_terminate_connection_address + field_size;
    // persistent mode field
    std::size_t sender_channel_1_buffer_index_semaphore_address =
        sender_channel_1_connection_semaphore_address + field_size;

    // ----------- Receiver Channel
    std::size_t receiver_channel_local_buffer_index_address =
        sender_channel_1_buffer_index_semaphore_address + field_size;
    // persistent mode field
    std::size_t receiver_channel_downstream_flow_control_semaphore_address =
        receiver_channel_local_buffer_index_address + field_size;

    // Channel Allocations
    std::size_t max_l1_loading_size = tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_size() +
                                      tt::tt_metal::experimental::hal::get_erisc_l1_unreserved_base();
    std::size_t buffer_region_start =
        (receiver_channel_downstream_flow_control_semaphore_address + field_size + buffer_alignment) &
        ~(buffer_alignment - 1);  // Align
    std::size_t available_channel_buffering_space = max_l1_loading_size - buffer_region_start;

    FabricEriscDatamoverConfig(
        std::size_t channel_buffer_size_bytes, std::size_t sender_ratio_size, std::size_t receiver_ratio_size);

    std::size_t channel_buffer_size_bytes = 0;
    std::size_t channel_buffer_size_bytes_with_channel_sync = 0;
    std::size_t sender_0_channel_size_bytes = 0;
    std::size_t sender_0_num_buffers = 0;
    std::size_t sender_1_channel_size_bytes = 0;
    std::size_t sender_1_num_buffers = 0;
    std::size_t receiver_channel_size_bytes = 0;
    std::size_t receiver_num_buffers = 0;

    std::size_t sender_0_channel_base_address = 0;
    std::size_t sender_1_channel_base_address = 0;
    std::size_t receiver_channel_base_address = 0;
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

    FabricEriscDatamoverBuilder(
        const CoreCoord& my_eth_core_logical,
        size_t my_noc_x,
        size_t my_noc_y,
        size_t my_chip_id,
        size_t peer_chip_id,

        std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id,
        std::optional<size_t> receiver_channel_downstream_teardown_semaphore_id,
        size_t sender_channel_0_flow_control_semaphore_id,
        size_t sender_channel_1_flow_control_semaphore_id,
        size_t sender_channel_0_connection_semaphore_id,
        size_t sender_channel_1_connection_semaphore_id,
        size_t sender_channel_0_buffer_index_semaphore_id,
        size_t sender_channel_1_buffer_index_semaphore_id,

        const FabricEriscDatamoverConfig& config,
        bool enable_persistent_mode,
        bool build_in_worker_connection_mode = false);

    static FabricEriscDatamoverBuilder build(
        tt::tt_metal::IDevice* device,
        tt::tt_metal::Program& program,
        const CoreCoord& ethernet_core,
        chip_id_t local_chip_id,
        chip_id_t peer_chip_id,
        const FabricEriscDatamoverConfig& config,
        bool enable_persistent_mode,
        bool build_in_worker_connection_mode = false);

    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_worker_channel() const;
    [[nodiscard]] SenderWorkerAdapterSpec build_connection_to_fabric_channel() const;

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

    size_t sender_0_num_buffers = 0;
    size_t sender_1_num_buffers = 0;
    size_t receiver_num_buffers = 0;

    size_t local_sender_channel_0_buffer_address = 0;
    size_t local_sender_channel_0_connection_info_addr = 0;
    size_t local_sender_channel_1_buffer_address = 0;
    size_t local_sender_channel_1_connection_info_addr = 0;
    size_t local_receiver_channel_buffer_address = 0;

    size_t termination_signal_ptr = 0;
    size_t edm_status_ptr = 0;

    // Semaphore IDs
    // this is the receiver channel's local sem for flow controlling with downstream fabric sender
    std::optional<size_t> receiver_channel_downstream_flow_control_semaphore_id;
    std::optional<size_t> receiver_channel_downstream_teardown_semaphore_id;
    size_t sender_channel_0_flow_control_semaphore_id = 0;
    size_t sender_channel_1_flow_control_semaphore_id = 0;
    size_t sender_channel_0_connection_semaphore_id = 0;
    size_t sender_channel_1_connection_semaphore_id = 0;
    size_t sender_channel_0_buffer_index_semaphore_id = 0;
    size_t sender_channel_1_buffer_index_semaphore_id = 0;
    size_t receiver_channel_local_buffer_index_address = 0;

    std::optional<size_t> downstream_edm_noc_x;
    std::optional<size_t> downstream_edm_noc_y;
    std::optional<size_t> downstream_edm_buffer_base_address;
    std::optional<size_t> downstream_edm_semaphore_address;
    std::optional<size_t> downstream_edm_worker_registration_address;
    std::optional<size_t> downstream_edm_worker_location_info_address;
    std::optional<size_t> downstream_sender_channel_buffer_index_semaphore_id;
    bool enable_persistent_mode = false;
    bool build_in_worker_connection_mode = false;
    size_t firmware_context_switch_interval = default_firmware_context_switch_interval;
};

}  // namespace tt::tt_fabric
