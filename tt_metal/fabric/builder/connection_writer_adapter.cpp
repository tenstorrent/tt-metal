// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"

namespace tt::tt_fabric {

// Adds downstream noc x/y
void ChannelConnectionWriterAdapter::add_downstream_connection(
    const SenderWorkerAdapterSpec& adapter_spec,
    uint32_t inbound_vc_idx,
    eth_chan_directions downstream_direction,
    CoreCoord downstream_noc_xy,
    bool is_2D_routing,
    bool is_vc1) {}

void ChannelConnectionWriterAdapter::pack_inbound_channel_rt_args(
    uint32_t vc_idx, std::vector<uint32_t>& args_out) const {
    bool vc_1_enabled = topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus;
    auto rt_args = std::initializer_list<uint32_t>{
        vc_idx == 0 ? this->downstream_edms_connected : vc_1_enabled,
        this->pack_downstream_noc_x_rt_arg(vc_idx),
        this->pack_downstream_noc_y_rt_arg(vc_idx),
        this->downstream_edm_vcs_worker_registration_address[vc_idx].value_or(0),
        this->downstream_edm_vcs_worker_location_info_address[vc_idx].value_or(0),
    };

    args_out.reserve(args_out.size() + rt_args.size());
    std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(args_out));

    this->pack_inbound_channel_rt_args_impl(vc_idx, args_out);
}
void ChannelConnectionWriterAdapter::emit_ct_args(std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const {
    this->emit_ct_args_impl(ct_args_out, num_fwd_paths);
}

}  // namespace tt::tt_fabric
