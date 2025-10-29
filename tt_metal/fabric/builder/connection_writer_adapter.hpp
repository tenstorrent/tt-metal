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
        SenderWorkerAdapterSpec const& adapter_spec,
        uint32_t inbound_vc_idx,
        eth_chan_directions downstream_direction,
        CoreCoord downstream_noc_xy,
        bool is_2D_routing,
        bool is_vc1) = 0;

    virtual void pack_inbound_channel_rt_args(uint32_t vc_idx, std::vector<uint32_t>& args_out) const = 0;
    virtual void emit_ct_args(std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const = 0;

protected:
    ~ChannelConnectionWriterAdapter() = default;

private:
    std::array<std::optional<size_t>, builder_config::num_receiver_channels> downstream_edm_vcs_noc_x = {};
    std::array<std::optional<size_t>, builder_config::num_receiver_channels> downstream_edm_vcs_noc_y = {};
    std::array<std::optional<size_t>, builder_config::num_receiver_channels> downstream_edm_vcs_worker_registration_address = {};
    std::array<std::optional<size_t>, builder_config::num_receiver_channels> downstream_edm_vcs_worker_location_info_address = {};
};

/*
 * Static-sized channel connection writer adapter to represent a connection to a static-sized
 * downstream sender(outbound) channel.
 */
class StaticSizedChannelConnectionWriterAdapter final : public ChannelConnectionWriterAdapter {
public:
    StaticSizedChannelConnectionWriterAdapter(
        FabricStaticSizedChannelsAllocator& allocator, tt::tt_fabric::Topology topology);

     void add_downstream_connection(
        SenderWorkerAdapterSpec const& adapter_spec,
        uint32_t inbound_vc_idx,
        eth_chan_directions downstream_direction,
        CoreCoord downstream_noc_xy,
        bool is_2D_routing,
        bool is_vc1) override;

    void pack_inbound_channel_rt_args(uint32_t vc_idx, std::vector<uint32_t>& args_out) const override;

    uint32_t get_downstream_edms_connected(bool is_2d_routing, bool is_vc1) const;

private:
    uint32_t pack_downstream_noc_y_rt_arg(uint32_t vc_idx) const;
    uint32_t pack_downstream_noc_x_rt_arg(uint32_t vc_idx) const;
    uint32_t encode_noc_ord_for_2d(
        const std::array<std::vector<std::pair<eth_chan_directions, CoreCoord>>, builder_config::num_receiver_channels>& downstream_edms_connected_by_vc,
        uint32_t vc_idx,
        const std::function<uint32_t(CoreCoord)>& get_noc_ord) const;

    void emit_ct_args(std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const override;

    std::unordered_set<uint32_t> downstream_edms_connected_by_vc_set;

    // holds which downstream cores a given receiver/inbound channel VC can feed into
    std::array<std::vector<std::pair<eth_chan_directions, CoreCoord>>, builder_config::num_receiver_channels> downstream_edms_connected_by_vc = {};

    // holds the number of buffer slots per downstream sender channel
    std::array<std::optional<size_t>, builder_config::num_sender_channels> sender_channels_num_buffers = {};

    // holds the number of buffer slots per downstream VC. i.e. if forwarding to VC 0, use index 0 in the
    // array, if forwarding to VC 1, use index 1 in the array
    std::array<size_t, builder_config::num_receiver_channels> downstream_sender_channels_num_buffers = {};

    // holds the base address of the downstream sender channel buffer, by downstream sender/outbound VC index
    std::array<std::optional<size_t>, builder_config::num_receiver_channels> downstream_edm_vcs_buffer_base_address = {};

    uint32_t downstream_edms_connected = 0;

    std::array<std::optional<size_t>, builder_config::num_receiver_channels>
        downstream_edm_vcs_worker_registration_address = {};
    std::array<std::optional<size_t>, builder_config::num_receiver_channels>
        downstream_edm_vcs_worker_location_info_address = {};

    bool is_2D_routing = false;
};


}  // namespace tt::tt_fabric
