// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"

namespace tt::tt_fabric {

StaticSizedChannelConnectionWriterAdapter::StaticSizedChannelConnectionWriterAdapter(
    FabricStaticSizedChannelsAllocator& /*allocator*/, tt::tt_fabric::Topology topology) :
    ChannelConnectionWriterAdapter(topology) {}

void StaticSizedChannelConnectionWriterAdapter::add_downstream_connection_impl(
    const SenderWorkerAdapterSpec& adapter_spec,
    uint32_t inbound_vc_idx,
    eth_chan_directions /*downstream_direction*/,
    CoreCoord /*downstream_noc_xy*/,
    bool /*is_2D_routing*/,
    bool is_vc1) {
    if (is_vc1) {
        // TT_FATAL(adapter_spec.edm_buffer_base_addr != 122016, "VC1 base address must not be 122016");
    }
    this->downstream_edm_vcs_buffer_base_address.at(inbound_vc_idx) = adapter_spec.edm_buffer_base_addr;
    this->downstream_edm_vcs_worker_registration_address.at(inbound_vc_idx) = adapter_spec.edm_connection_handshake_addr;
    this->downstream_edm_vcs_worker_location_info_address.at(inbound_vc_idx) = adapter_spec.edm_worker_location_info_addr;
    this->downstream_sender_channels_num_buffers.at(inbound_vc_idx) = adapter_spec.num_buffers_per_channel;
    // TT_FATAL(
    //     this->downstream_sender_channels_num_buffers.at(inbound_vc_idx) != 0,
    //     "A Downstream sender channels num buffers must be greater than 0 for vc_idx: {}",
    //     inbound_vc_idx);
    // TT_FATAL(
    //     this->downstream_edm_vcs_buffer_base_address.at(inbound_vc_idx) != 0,
    //     "Downstream edm vcs buffer base address must be non-zero for vc_idx: {}",
    //     inbound_vc_idx);
    // TT_FATAL(
    //     this->downstream_edm_vcs_worker_registration_address.at(inbound_vc_idx) != 0,
    //     "Downstream edm vcs worker registration address must be non-zero for vc_idx: {}",
    //     inbound_vc_idx);
    // TT_FATAL(
    //     this->downstream_edm_vcs_worker_location_info_address.at(inbound_vc_idx) != 0,
    //     "Downstream edm vcs worker location info address must be non-zero for vc_idx: {}",
    //     inbound_vc_idx);
}

void StaticSizedChannelConnectionWriterAdapter::pack_inbound_channel_rt_args_impl(
    uint32_t vc_idx, std::vector<uint32_t>& args_out) const {
    TT_FATAL(
        this->downstream_edm_vcs_buffer_base_address.size() > vc_idx,
        "VC index is out of bounds for downstream_edm_vcs_buffer_base_address");
    TT_FATAL(
        this->downstream_edm_vcs_worker_registration_address.size() > vc_idx,
        "VC index is out of bounds for downstream_edm_vcs_worker_registration_address");
    TT_FATAL(
        this->downstream_edm_vcs_worker_location_info_address.size() > vc_idx,
        "VC index is out of bounds for downstream_edm_vcs_worker_location_info_address");

    // log_info(tt::LogFabric, "VC1 base address: {}",
    // this->downstream_edm_vcs_buffer_base_address[vc_idx].value_or(0)); for (size_t i = 0; i <
    // this->downstream_edm_vcs_buffer_base_address.size(); i++) {
    //     log_info(
    //         tt::LogFabric, "\tVC{} base address: {}", i,
    //         this->downstream_edm_vcs_buffer_base_address[i].value_or(0));
    // }

    args_out.push_back(this->downstream_edm_vcs_buffer_base_address[vc_idx].value_or(0));
}

void StaticSizedChannelConnectionWriterAdapter::emit_ct_args_impl(
    std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const {
    ct_args_out.push_back(downstream_sender_channels_num_buffers.at(num_fwd_paths));
    // ct_args_out.insert(
    //     ct_args_out.end(),
    //     this->downstream_sender_channels_num_buffers.begin(),
    //     this->downstream_sender_channels_num_buffers.begin() + num_fwd_paths);

    for (size_t i = 0; i < num_fwd_paths; i++) {
        if (this->downstream_edms_connected_by_vc_set.find(i) != this->downstream_edms_connected_by_vc_set.end()) {
            TT_FATAL(this->downstream_sender_channels_num_buffers[i] != 0, "Downstream sender channels num buffers must be greater than 0 for vc_idx: {}", i);
        }
    }
}

}  // namespace tt::tt_fabric
