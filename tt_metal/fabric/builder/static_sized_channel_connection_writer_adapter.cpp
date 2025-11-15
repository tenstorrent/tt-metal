// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"

namespace tt::tt_fabric {

StaticSizedChannelConnectionWriterAdapter::StaticSizedChannelConnectionWriterAdapter(
    FabricStaticSizedChannelsAllocator& /*allocator*/,
    tt::tt_fabric::Topology topology,
    eth_chan_directions my_direction) :
    is_2D_routing(topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Torus),
    my_direction(my_direction) {}

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
            // Calculate compact index based on downstream_direction relative to my_direction
            // The compact index excludes the router's own direction
            // For EAST router (my_direction=0): WEST(1)→0, NORTH(2)→1, SOUTH(3)→2
            // For WEST router (my_direction=1): EAST(0)→0, NORTH(2)→1, SOUTH(3)→2
            // For NORTH router (my_direction=2): EAST(0)→0, WEST(1)→1, SOUTH(3)→2
            // For SOUTH router (my_direction=3): EAST(0)→0, WEST(1)→1, NORTH(2)→2
            size_t compact_index;
            if (my_direction == 0) {
                // EAST router: skip index 0, map 1→0, 2→1, 3→2
                compact_index = downstream_direction - 1;
            } else {
                // For other directions: if downstream < my_direction, use as-is; else subtract 1
                compact_index =
                    (downstream_direction < my_direction) ? downstream_direction : (downstream_direction - 1);
            }
            this->downstream_edms_connected |= (1 << compact_index);

            // Store addresses indexed by [vc_idx][compact_index]
            this->downstream_edm_buffer_base_addresses.at(inbound_vc_idx).at(compact_index) =
                adapter_spec.edm_buffer_base_addr;
            this->downstream_edm_worker_registration_addresses.at(inbound_vc_idx).at(compact_index) =
                adapter_spec.edm_connection_handshake_addr;
            this->downstream_edm_worker_location_info_addresses.at(inbound_vc_idx).at(compact_index) =
                adapter_spec.edm_worker_location_info_addr;
        }
    } else {
        this->downstream_edms_connected = 1;

        // For 1D, store at compact index 0
        this->downstream_edm_buffer_base_addresses.at(inbound_vc_idx).at(0) = adapter_spec.edm_buffer_base_addr;
        this->downstream_edm_worker_registration_addresses.at(inbound_vc_idx).at(0) =
            adapter_spec.edm_connection_handshake_addr;
        this->downstream_edm_worker_location_info_addresses.at(inbound_vc_idx).at(0) =
            adapter_spec.edm_worker_location_info_addr;
    }

    this->downstream_sender_channels_num_buffers.at(inbound_vc_idx) = adapter_spec.num_buffers_per_channel;
    this->downstream_edms_connected_by_vc_set.insert(inbound_vc_idx);
}

void StaticSizedChannelConnectionWriterAdapter::pack_inbound_channel_rt_args(uint32_t vc_idx, std::vector<uint32_t>& args_out) const {
    if (vc_idx == 0 && is_2D_routing) {
        // For VC0 in 2D: pack connection mask and data for 3 downstream EDMs
        args_out.push_back(this->downstream_edms_connected);  // 3-bit mask

        // Pack 3 buffer base addresses (one per compact index 0-2)
        for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc0; compact_idx++) {
            uint32_t buffer_addr = this->downstream_edm_buffer_base_addresses[vc_idx][compact_idx].value_or(0);
            // log_info(tt::LogOp, "VC0 2D: my_direction: {}, downstream_edm_buffer_base_addresses[{}][{}] = 0x{:x}",
            // my_direction, vc_idx, compact_idx, buffer_addr);
            args_out.push_back(buffer_addr);
        }

        // Pack NOC X and Y (already compacted properly)
        args_out.push_back(this->pack_downstream_noc_x_rt_arg(vc_idx));
        args_out.push_back(this->pack_downstream_noc_y_rt_arg(vc_idx));

        // Pack 3 worker registration addresses
        for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc0; compact_idx++) {
            args_out.push_back(this->downstream_edm_worker_registration_addresses[vc_idx][compact_idx].value_or(0));
        }

        // Pack 3 worker location info addresses
        for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc0; compact_idx++) {
            args_out.push_back(this->downstream_edm_worker_location_info_addresses[vc_idx][compact_idx].value_or(0));
        }
    } else {
        // For VC1 or 1D: single downstream connection (backward compatible)
        bool has_connection = vc_idx == 0 ? (this->downstream_edms_connected != 0)
                                          : this->downstream_edm_buffer_base_addresses[vc_idx][0].has_value();

        uint32_t buffer_addr = this->downstream_edm_buffer_base_addresses[vc_idx][0].value_or(0);
        // log_info(tt::LogOp, "VC{} 1D: my_direction: {}, downstream_edm_buffer_base_addresses[{}][0] = 0x{:x}",
        // vc_idx, my_direction, vc_idx, buffer_addr);

        auto rt_args = std::initializer_list<uint32_t>{
            has_connection,
            buffer_addr,
            this->pack_downstream_noc_x_rt_arg(vc_idx),
            this->pack_downstream_noc_y_rt_arg(vc_idx),
            this->downstream_edm_worker_registration_addresses[vc_idx][0].value_or(0),
            this->downstream_edm_worker_location_info_addresses[vc_idx][0].value_or(0),
        };

        args_out.reserve(args_out.size() + rt_args.size());
        std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(args_out));
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
            // Calculate compact index based on direction relative to my_direction
            size_t compact_index;
            if (my_direction == 0) {
                compact_index = direction - 1;
            } else {
                compact_index = (direction < my_direction) ? direction : (direction - 1);
            }
            ord |= (get_noc_ord(noc_xy) << (compact_index * 8));
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
