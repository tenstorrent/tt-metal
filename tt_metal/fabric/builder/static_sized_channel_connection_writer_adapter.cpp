// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"
#include <tt-metalium/control_plane.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"

namespace tt::tt_fabric {

StaticSizedChannelConnectionWriterAdapter::StaticSizedChannelConnectionWriterAdapter(
    FabricStaticSizedChannelsAllocator& /*allocator*/, tt::tt_fabric::Topology topology) :
    is_2D_routing(topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Torus) {}

void StaticSizedChannelConnectionWriterAdapter::add_downstream_connection(
    SenderWorkerAdapterSpec const& adapter_spec,
    uint32_t inbound_vc_idx,
    eth_chan_directions downstream_direction,
    CoreCoord downstream_noc_xy,
    bool is_2D_routing,
    bool is_vc1) {
    downstream_edms_connected_by_vc.at(inbound_vc_idx).push_back(
        {downstream_direction, CoreCoord(downstream_noc_xy.x, downstream_noc_xy.y)});

    if (is_2D_routing) {
        if (!is_vc1) {
            this->downstream_edms_connected |= (1 << downstream_direction);
        }
    } else {
        this->downstream_edms_connected = 1;
    }

    this->downstream_edm_vcs_buffer_base_address.at(inbound_vc_idx) = adapter_spec.edm_buffer_base_addr;
    this->downstream_edm_vcs_worker_registration_address.at(inbound_vc_idx) = adapter_spec.edm_connection_handshake_addr;
    this->downstream_edm_vcs_worker_location_info_address.at(inbound_vc_idx) = adapter_spec.edm_worker_location_info_addr;
    this->downstream_sender_channels_num_buffers.at(inbound_vc_idx) = adapter_spec.num_buffers_per_channel;
    this->downstream_edms_connected_by_vc_set.insert(inbound_vc_idx);
}

void StaticSizedChannelConnectionWriterAdapter::add_local_tensix_connection(
    const SenderWorkerAdapterSpec& adapter_spec, eth_chan_directions /*tensix_direction*/, CoreCoord tensix_noc_xy) {
    this->local_tensix_noc_xy = tensix_noc_xy;
    this->local_tensix_buffer_base_address = adapter_spec.edm_buffer_base_addr;
    this->local_tensix_worker_registration_address = adapter_spec.edm_connection_handshake_addr;
    this->local_tensix_worker_location_info_address = adapter_spec.edm_worker_location_info_addr;

    // Get relay-specific info from fabric context
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_tensix_config();

    // Store free slots stream ID (no teardown semaphore needed - router doesn't call close)
    constexpr uint32_t relay_channel_id = static_cast<uint32_t>(UdmRelayChannelId::ROUTER_CHANNEL);
    this->local_tensix_free_slots_stream_id =
        tensix_config.get_channel_credits_stream_id(relay_channel_id, FabricTensixCoreType::RELAY);

    this->local_tensix_connected_set = true;
}

void StaticSizedChannelConnectionWriterAdapter::pack_inbound_channel_rt_args(uint32_t vc_idx, std::vector<uint32_t>& args_out) const {

    TT_FATAL(downstream_edm_vcs_buffer_base_address.size() > vc_idx, "VC index is out of bounds for downstream_edm_vcs_buffer_base_address");
    TT_FATAL(downstream_edm_vcs_worker_registration_address.size() > vc_idx, "VC index is out of bounds for downstream_edm_vcs_worker_registration_address");
    TT_FATAL(downstream_edm_vcs_worker_location_info_address.size() > vc_idx, "VC index is out of bounds for downstream_edm_vcs_worker_location_info_address");

    auto rt_args = std::initializer_list<uint32_t>{
        vc_idx == 0 ? this->downstream_edms_connected : this->downstream_edm_vcs_buffer_base_address[vc_idx] != std::nullopt,
        this->downstream_edm_vcs_buffer_base_address[vc_idx].value_or(0),
        this->pack_downstream_noc_x_rt_arg(vc_idx),
        this->pack_downstream_noc_y_rt_arg(vc_idx),
        this->downstream_edm_vcs_worker_registration_address[vc_idx].value_or(0),
        this->downstream_edm_vcs_worker_location_info_address[vc_idx].value_or(0),
    };

    args_out.reserve(args_out.size() + rt_args.size());
    std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(args_out));
}

void StaticSizedChannelConnectionWriterAdapter::pack_adaptor_to_relay_rt_args(std::vector<uint32_t>& args_out) const {
    // Pack local tensix (relay) connection info at the end of runtime args
    // If no relay connection, just pack the flag (0)
    if (!this->local_tensix_connected_set) {
        args_out.push_back(0u);  // has_local_tensix_relay_connection = false
    } else {
        // Pack full relay connection info
        auto relay_rt_args = std::initializer_list<uint32_t>{
            1u,                                               // has_local_tensix_relay_connection = true
            this->local_tensix_buffer_base_address,           // relay_buffer_base_addr
            this->local_tensix_noc_xy.x,                      // relay_noc_x
            this->local_tensix_noc_xy.y,                      // relay_noc_y
            this->local_tensix_worker_registration_address,   // relay_connection_handshake_addr
            this->local_tensix_worker_location_info_address,  // relay_worker_location_info_addr
            this->local_tensix_free_slots_stream_id,          // relay_free_slots_stream_id
        };

        args_out.reserve(args_out.size() + relay_rt_args.size());
        std::copy(relay_rt_args.begin(), relay_rt_args.end(), std::back_inserter(args_out));
    }
}

uint32_t StaticSizedChannelConnectionWriterAdapter::get_downstream_edms_connected(
    bool /*is_2d_routing*/, bool /*is_vc1*/) const {
    return this->downstream_edms_connected;
}

uint32_t StaticSizedChannelConnectionWriterAdapter::pack_downstream_noc_y_rt_arg(uint32_t vc_idx) const {
    return encode_noc_ord_for_2d(
        this->downstream_edms_connected_by_vc, vc_idx, [](CoreCoord noc_xy) { return noc_xy.y; });
}
uint32_t StaticSizedChannelConnectionWriterAdapter::pack_downstream_noc_x_rt_arg(uint32_t vc_idx) const {
    return encode_noc_ord_for_2d(
        this->downstream_edms_connected_by_vc, vc_idx, [](CoreCoord noc_xy) { return noc_xy.x; });
}
/*
 * For 2D fabric, downstream noc x/y coords are packed into uint32_t, one per byte
 * X and Y have separate uint32s
 */
uint32_t StaticSizedChannelConnectionWriterAdapter::encode_noc_ord_for_2d(
    const std::array<std::vector<std::pair<eth_chan_directions, CoreCoord>>, builder_config::num_receiver_channels>& downstream_edms_connected_by_vc,
    uint32_t vc_idx,
    const std::function<uint32_t(CoreCoord)>& get_noc_ord) const {
    if (vc_idx == 1 || !is_2D_routing) {
        if (downstream_edms_connected_by_vc[vc_idx].empty()) {
            return 0; // no connection here
        }
        TT_FATAL(downstream_edms_connected_by_vc[vc_idx].size() == 1, "Downstream edms connected by vc should be 1 for vc1 or non-2D routing. vc_idx: {}, size: {}", vc_idx, downstream_edms_connected_by_vc[vc_idx].size());
        auto ord = get_noc_ord(downstream_edms_connected_by_vc[vc_idx].front().second);
        return ord;
    } else {
        uint32_t ord = 0;
        for (const auto& [direction, noc_xy] : downstream_edms_connected_by_vc[vc_idx]) {
            ord |= (get_noc_ord(noc_xy) << (direction * 8));
        }
        return ord;
    }
}

void StaticSizedChannelConnectionWriterAdapter::emit_ct_args(std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const {
    ct_args_out.insert(
        ct_args_out.end(),
        this->downstream_sender_channels_num_buffers.begin(),
        this->downstream_sender_channels_num_buffers.begin() + num_fwd_paths);

    for (size_t i = 0; i < num_fwd_paths; i++) {
        if (this->downstream_edms_connected_by_vc_set.find(i) != this->downstream_edms_connected_by_vc_set.end()) {
            TT_FATAL(this->downstream_sender_channels_num_buffers[i] != 0, "Downstream sender channels num buffers must be greater than 0 for vc_idx: {}", i);
        }
    }
}

}  // namespace tt::tt_fabric
