// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "core_coord.hpp"
#include "tt_metal/fabric/builder/fabric_static_sized_channels_allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h"
#include <vector>

namespace tt::tt_fabric {

// Local tensix (relay) connection info for UDM mode
struct LocalTensixRelayConnectionInfo {
    CoreCoord noc_xy = {0, 0};
    size_t buffer_base_address = 0;
    size_t worker_registration_address = 0;
    size_t worker_location_info_address = 0;
    size_t free_slots_stream_id = 0;
    bool is_connected = false;
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

/*
 * Base class for channel connection writer adapters.
 * These adapters are used during the fabric build phase and hold information about the connection between
 * a producer (i.e. receiver/inbound channel) and its consumer. The adapter is from the producer's
 * perspective; the adapter itself does not perform and write functions.
 *
 * General metadata about the connection is held in the adapter:
 *  - how many downstream routers there are and what their Noc coordinates are
 *  - connection address information
 *    - where to store our noc x/y on the consumer core
 *    - where to signal the consumer core to establish a connection
 *    - etc
 *
 * There is an abstract base class because depending on how the channels are instantiated
 * (static-sized vs elastic), the adapter will have different implementations and relevant
 * information. For example, credit schemes and addresses are different for each type of channel.
 */
class ChannelConnectionWriterAdapter {
public:
    // Adds downstream noc x/y
    virtual void add_downstream_connection(
        const SenderWorkerAdapterSpec& adapter_spec,
        uint32_t inbound_vc_idx,
        uint32_t sender_channel_idx,
        eth_chan_directions downstream_direction,
        CoreCoord downstream_noc_xy,
        bool is_2D_routing) = 0;

    virtual void pack_inbound_channel_rt_args(uint32_t vc_idx, std::vector<uint32_t>& args_out) const = 0;
    virtual void pack_adaptor_to_relay_rt_args(std::vector<uint32_t>& args_out) const = 0;
    virtual void emit_ct_args(std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const = 0;

    // Add connection to local tensix (relay in UDM mode)
    virtual void add_local_tensix_connection(const SenderWorkerAdapterSpec&, eth_chan_directions, CoreCoord) = 0;

    // Get the number of downstream EDMs connected for a specific VC
    virtual uint32_t get_downstream_edm_count_for_vc(uint32_t vc_idx) const = 0;

    // Get the connection mask for a specific VC
    virtual uint32_t get_downstream_edm_mask_for_vc(uint32_t vc_idx) const = 0;

protected:
    ~ChannelConnectionWriterAdapter() = default;

private:
    std::array<std::optional<size_t>, builder_config::num_max_receiver_channels> downstream_edm_vcs_noc_x = {};
    std::array<std::optional<size_t>, builder_config::num_max_receiver_channels> downstream_edm_vcs_noc_y = {};
    std::array<std::optional<size_t>, builder_config::num_max_receiver_channels>
        downstream_edm_vcs_worker_registration_address = {};
    std::array<std::optional<size_t>, builder_config::num_max_receiver_channels>
        downstream_edm_vcs_worker_location_info_address = {};
};

/*
 * Static-sized channel connection writer adapter to represent a connection to a static-sized
 * downstream sender(outbound) channel.
 */
class StaticSizedChannelConnectionWriterAdapter final : public ChannelConnectionWriterAdapter {
public:
    StaticSizedChannelConnectionWriterAdapter(
        FabricStaticSizedChannelsAllocator& allocator,
        tt::tt_fabric::Topology topology,
        eth_chan_directions my_direction);

    void add_downstream_connection(
        const SenderWorkerAdapterSpec& adapter_spec,
        uint32_t inbound_vc_idx,
        uint32_t sender_channel_idx,
        eth_chan_directions downstream_direction,
        CoreCoord downstream_noc_xy,
        bool is_2D_routing) override;

    void add_local_tensix_connection(
        const SenderWorkerAdapterSpec& adapter_spec,
        eth_chan_directions tensix_direction,
        CoreCoord tensix_noc_xy) override;

    void pack_inbound_channel_rt_args(uint32_t vc_idx, std::vector<uint32_t>& args_out) const override;
    void pack_adaptor_to_relay_rt_args(std::vector<uint32_t>& args_out) const override;

    uint32_t get_downstream_edms_connected() const;

    // Get the number of downstream EDMs connected for a specific VC
    uint32_t get_downstream_edm_count_for_vc(uint32_t vc_idx) const override {
        return downstream_edms_connected_by_vc.at(vc_idx).size();
    }

    // Get the connection mask for a specific VC
    uint32_t get_downstream_edm_mask_for_vc(uint32_t vc_idx) const override {
        return downstream_edms_connected_by_vc_mask.at(vc_idx);
    }

    // Get buffer index semaphore address for a specific VC and compact index
    std::optional<size_t> get_buffer_index_semaphore_address(uint32_t vc_idx, size_t compact_idx) const {
        return downstream_edm_buffer_index_semaphore_addresses.at(vc_idx).at(compact_idx);
    }

private:
    uint32_t pack_downstream_noc_y_rt_arg(uint32_t vc_idx) const;
    uint32_t pack_downstream_noc_x_rt_arg(uint32_t vc_idx) const;
    uint32_t encode_noc_ord_for_2d(
        const std::array<
            std::vector<std::pair<eth_chan_directions, CoreCoord>>,
            builder_config::num_max_receiver_channels>& downstream_edms_connected_by_vc,
        uint32_t vc_idx,
        const std::function<uint32_t(CoreCoord)>& get_noc_ord) const;

    void emit_ct_args(std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const override;

    std::unordered_set<uint32_t> downstream_edms_connected_by_vc_set;

    // holds which downstream cores a given receiver/inbound channel VC can feed into
    std::array<std::vector<std::pair<eth_chan_directions, CoreCoord>>, builder_config::num_max_receiver_channels>
        downstream_edms_connected_by_vc = {};

    // holds the number of buffer slots per downstream sender channel
    std::array<std::optional<size_t>, builder_config::num_max_sender_channels> sender_channels_num_buffers = {};

    // holds the number of buffer slots per downstream VC. i.e. if forwarding to VC 0, use index 0 in the
    // array, if forwarding to VC 1, use index 1 in the array
    std::array<size_t, builder_config::num_max_receiver_channels> downstream_sender_channels_num_buffers = {};

    // For VC0: holds base addresses for up to 3 downstream EDMs (indexed by compact index)
    std::array<
        std::array<std::optional<size_t>, builder_config::max_downstream_edms>,
        builder_config::num_max_receiver_channels>
        downstream_edm_buffer_base_addresses = {};

    // Per-VC connection mask: bitmask indicating which downstream EDMs are connected for each VC
    std::array<uint32_t, builder_config::num_max_receiver_channels> downstream_edms_connected_by_vc_mask = {};

    std::array<
        std::array<std::optional<size_t>, builder_config::max_downstream_edms>,
        builder_config::num_max_receiver_channels>
        downstream_edm_worker_registration_addresses = {};
    std::array<
        std::array<std::optional<size_t>, builder_config::max_downstream_edms>,
        builder_config::num_max_receiver_channels>
        downstream_edm_worker_location_info_addresses = {};
    std::array<
        std::array<std::optional<size_t>, builder_config::max_downstream_edms>,
        builder_config::num_max_receiver_channels>
        downstream_edm_buffer_index_semaphore_addresses = {};

    bool is_2D_routing = false;
    eth_chan_directions my_direction = eth_chan_directions::EAST;

    // Local tensix (relay) connection info for UDM mode
    LocalTensixRelayConnectionInfo relay_connection_info;
};


}  // namespace tt::tt_fabric
