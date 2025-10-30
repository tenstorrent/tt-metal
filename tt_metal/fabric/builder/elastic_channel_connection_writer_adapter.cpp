// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"

namespace tt::tt_fabric {

ChannelConnectionWriterAdapter::ChannelConnectionWriterAdapter(tt::tt_fabric::Topology topology) :
    is_2D_routing(topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Torus) {}

ElasticChannelConnectionWriterAdapter::ElasticChannelConnectionWriterAdapter(
    ElasticChannelsAllocator& allocator, tt::tt_fabric::Topology topology) :
    ChannelConnectionWriterAdapter(topology) {}

void ChannelConnectionWriterAdapter::add_downstream_connection(
    const SenderWorkerAdapterSpec& adapter_spec,
    uint32_t inbound_vc_idx,
    eth_chan_directions downstream_direction,
    CoreCoord downstream_noc_xy,
    bool is_2D_routing,
    bool is_vc1) {
    downstream_edms_connected_by_vc.at(inbound_vc_idx)
        .push_back({downstream_direction, CoreCoord(downstream_noc_xy.x, downstream_noc_xy.y)});

    if (is_2D_routing) {
        if (!is_vc1) {
            this->downstream_edms_connected |= (1 << downstream_direction);
        }
    } else {
        this->downstream_edms_connected = 1;
    }

    this->downstream_edms_connected_by_vc_set.insert(inbound_vc_idx);

    this->add_downstream_connection_impl(
        adapter_spec, inbound_vc_idx, downstream_direction, downstream_noc_xy, is_2D_routing, is_vc1);
}

void ElasticChannelConnectionWriterAdapter::pack_inbound_channel_rt_args(
    uint32_t vc_idx, std::vector<uint32_t>& args_out) const {
    TT_FATAL(false, "ElasticChannelConnectionWriterAdapter::pack_inbound_channel_rt_args not implemented");
}

void ElasticChannelConnectionWriterAdapter::emit_ct_args(
    std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const {
    TT_FATAL(false, "ElasticChannelConnectionWriterAdapter::emit_ct_args not implemented");
}

void ElasticChannelConnectionWriterAdapter::add_downstream_connection_impl(
    const SenderWorkerAdapterSpec& adapter_spec,
    uint32_t inbound_vc_idx,
    eth_chan_directions downstream_direction,
    CoreCoord downstream_noc_xy,
    bool is_2D_routing,
    bool is_vc1) {
    // Nothing to do unique to ElasticChannelConnectionWriterAdapter
}

}  // namespace tt::tt_fabric
