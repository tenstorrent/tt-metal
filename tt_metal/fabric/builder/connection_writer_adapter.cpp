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
    this->downstream_edm_vcs_worker_registration_address.at(inbound_vc_idx) =
        adapter_spec.edm_connection_handshake_addr;
    this->downstream_edm_vcs_worker_location_info_address.at(inbound_vc_idx) =
        adapter_spec.edm_worker_location_info_addr;

    this->add_downstream_connection_impl(
        adapter_spec, inbound_vc_idx, downstream_direction, downstream_noc_xy, is_2D_routing, is_vc1);
}

void ChannelConnectionWriterAdapter::pack_inbound_channel_rt_args(
    uint32_t vc_idx, std::vector<uint32_t>& args_out) const {
    bool vc_1_enabled = (topology == tt::tt_fabric::Topology::Ring || topology == tt::tt_fabric::Topology::Torus) &&
                        (this->downstream_edms_connected_by_vc_set.contains(vc_idx));
    bool is_active = vc_idx == 0 ? this->downstream_edms_connected : vc_1_enabled;
    auto rt_args = std::initializer_list<uint32_t>{
        is_active,
        this->pack_downstream_noc_x_rt_arg(vc_idx),
        this->pack_downstream_noc_y_rt_arg(vc_idx),
        this->downstream_edm_vcs_worker_registration_address[vc_idx].value_or(0),
        this->downstream_edm_vcs_worker_location_info_address[vc_idx].value_or(0),
    };
    if (is_active && vc_idx == 1) {
        log_info(
            tt::LogFabric,
            "Downstream edm vcs buffer base address[0]: {}",
            dynamic_cast<const StaticSizedChannelConnectionWriterAdapter*>(this)
                ->downstream_edm_vcs_buffer_base_address.at(0)
                .value_or(0));
        log_info(
            tt::LogFabric,
            "Downstream edm vcs buffer base address[1]: {}",
            dynamic_cast<const StaticSizedChannelConnectionWriterAdapter*>(this)
                ->downstream_edm_vcs_buffer_base_address.at(1)
                .value_or(0));
        TT_FATAL(
            dynamic_cast<const StaticSizedChannelConnectionWriterAdapter*>(this)
                    ->downstream_edm_vcs_buffer_base_address[vc_idx]
                    .value_or(0) != 0,
            "Downstream edm vcs buffer base address must be non-zero for vc_idx: {}",
            vc_idx);
    }

    args_out.reserve(args_out.size() + rt_args.size());
    std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(args_out));

    this->pack_inbound_channel_rt_args_impl(vc_idx, args_out);
}
void ChannelConnectionWriterAdapter::emit_ct_args(std::vector<uint32_t>& ct_args_out, size_t num_fwd_paths) const {
    this->emit_ct_args_impl(ct_args_out, num_fwd_paths);
}

}  // namespace tt::tt_fabric
