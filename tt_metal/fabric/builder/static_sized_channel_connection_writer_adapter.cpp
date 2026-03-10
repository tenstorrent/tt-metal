// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"

namespace tt::tt_fabric {

StaticSizedChannelConnectionWriterAdapter::StaticSizedChannelConnectionWriterAdapter(
    FabricStaticSizedChannelsAllocator& /*allocator*/,
    tt::tt_fabric::Topology topology,
    eth_chan_directions my_direction) :
    is_2D_routing(topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Torus),
    my_direction(my_direction) {}

void StaticSizedChannelConnectionWriterAdapter::add_downstream_connection(
    const SenderWorkerAdapterSpec& adapter_spec,
    uint32_t inbound_vc_idx,
    uint32_t /*sender_channel_idx*/,
    eth_chan_directions downstream_direction,
    CoreCoord downstream_noc_xy,
    bool is_2D_routing) {
    // Track connections per VC for packing
    downstream_edms_connected_by_vc.at(inbound_vc_idx)
        .push_back({downstream_direction, CoreCoord(downstream_noc_xy.x, downstream_noc_xy.y)});

    if (is_2D_routing) {
        // Calculate compact index based on downstream_direction relative to my_direction
        // The compact index excludes the router's own direction
        // For EAST router  (my_direction=0): WEST(1)→0, NORTH(2)→1, SOUTH(3)→2, Z(4)->3
        // For WEST router  (my_direction=1): EAST(0)→0, NORTH(2)→1, SOUTH(3)→2, Z(4)->3
        // For NORTH router (my_direction=2): EAST(0)→0, WEST(1)→1,  SOUTH(3)→2, Z(4)->3
        // For SOUTH router (my_direction=3): EAST(0)→0, WEST(1)→1,  NORTH(2)→2, Z(4)->3
        // For Z router     (my_direction=4): EAST(0)→0, WEST(1)→1,  NORTH(2)→2, SOUTH(3)→3
        size_t compact_index = get_receiver_channel_compact_index(my_direction, downstream_direction);
        this->downstream_edms_connected_by_vc_mask.at(inbound_vc_idx) |= (1 << compact_index);

        // Store addresses indexed by [vc_idx][compact_index]
        // NOTE: For INTRA_MESH connections, this works fine (one connection per compact_index)
        // For Z router multi-target, we'll use vc_to_downstreams_ instead
        this->downstream_edm_buffer_base_addresses.at(inbound_vc_idx).at(compact_index) =
            adapter_spec.edm_buffer_base_addr;
        this->downstream_edm_worker_registration_addresses.at(inbound_vc_idx).at(compact_index) =
            adapter_spec.edm_connection_handshake_addr;
        this->downstream_edm_worker_location_info_addresses.at(inbound_vc_idx).at(compact_index) =
            adapter_spec.edm_worker_location_info_addr;
        this->downstream_edm_buffer_index_semaphore_addresses.at(inbound_vc_idx).at(compact_index) =
            adapter_spec.buffer_index_semaphore_id;
    } else {
        this->downstream_edms_connected_by_vc_mask.at(inbound_vc_idx) = 1;

        // For 1D, store at compact index 0
        this->downstream_edm_buffer_base_addresses.at(inbound_vc_idx).at(0) = adapter_spec.edm_buffer_base_addr;
        this->downstream_edm_worker_registration_addresses.at(inbound_vc_idx).at(0) =
            adapter_spec.edm_connection_handshake_addr;
        this->downstream_edm_worker_location_info_addresses.at(inbound_vc_idx).at(0) =
            adapter_spec.edm_worker_location_info_addr;
        this->downstream_edm_buffer_index_semaphore_addresses.at(inbound_vc_idx).at(0) =
            adapter_spec.buffer_index_semaphore_id;
    }

    this->downstream_sender_channels_num_buffers.at(inbound_vc_idx) = adapter_spec.num_buffers_per_channel;
    this->downstream_edms_connected_by_vc_set.insert(inbound_vc_idx);
}

void StaticSizedChannelConnectionWriterAdapter::add_local_tensix_connection(
    const SenderWorkerAdapterSpec& adapter_spec, eth_chan_directions /*tensix_direction*/, CoreCoord tensix_noc_xy) {
    this->relay_connection_info.noc_xy = tensix_noc_xy;
    this->relay_connection_info.buffer_base_address = adapter_spec.edm_buffer_base_addr;
    this->relay_connection_info.worker_registration_address = adapter_spec.edm_connection_handshake_addr;
    this->relay_connection_info.worker_location_info_address = adapter_spec.edm_worker_location_info_addr;

    // Get relay-specific info from fabric context
    const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
    const auto& tensix_config = fabric_context.get_builder_context().get_tensix_config();

    // Store free slots stream ID
    constexpr uint32_t relay_channel_id = static_cast<uint32_t>(UdmRelayChannelId::ROUTER_CHANNEL);
    this->relay_connection_info.free_slots_stream_id =
        tensix_config.get_channel_credits_stream_id(relay_channel_id, FabricTensixCoreType::RELAY);

    this->relay_connection_info.is_connected = true;
}

void StaticSizedChannelConnectionWriterAdapter::pack_inbound_channel_rt_args(
    uint32_t vc_idx, std::vector<uint32_t>& args_out) const {
    // Standard packing for all connection types
    //
    // IMPORTANT: All connections on the same VC share the same downstream buffer address.
    // This is a fundamental constraint of the fabric architecture:
    // - For INTRA_MESH: Each sender channel connects to one downstream router
    // - For Z_TO_MESH: Multiple sender channels (one per direction) each connect to one downstream router
    // - All connections on a VC write to the same receiver channel buffer on their respective targets
    //
    // Because of this constraint, the standard packing path (which stores one buffer address per VC)
    // is sufficient for all connection types, including multi-target scenarios.
    if (is_2D_routing) {
        // For 2D: Use fixed slot count based on VC (kernel expects fixed-size arrays)
        // VC0: 3 slots (mesh directions)
        // Get the connection mask to determine which compact indices are valid
        uint32_t mask = this->downstream_edms_connected_by_vc_mask.at(vc_idx);

        // Pack connection mask (bit mask indicating which slots are valid)
        args_out.push_back(mask);

        // Dense pack: iterate through mask bits and pack only valid compact indices
        // For example, if mask=0x5 (binary 101), we pack compact indices 0 and 2 into args[0] and args[1]

        // Pack buffer base addresses (dense packed based on mask)
        for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc1_with_z; compact_idx++) {
            if (mask & (1 << compact_idx)) {
                uint32_t buffer_addr = this->downstream_edm_buffer_base_addresses[vc_idx][compact_idx].value_or(0);
                args_out.push_back(buffer_addr);
            }
        }

        // Dense pack NOC X and Y (methods get mask internally)
        args_out.push_back(this->pack_downstream_noc_x_rt_arg(vc_idx));
        args_out.push_back(this->pack_downstream_noc_y_rt_arg(vc_idx));

        // Pack worker registration addresses (dense packed based on mask)
        for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc1_with_z; compact_idx++) {
            if (mask & (1 << compact_idx)) {
                args_out.push_back(this->downstream_edm_worker_registration_addresses[vc_idx][compact_idx].value_or(0));
            }
        }

        // Pack worker location info addresses (dense packed based on mask)
        for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc1_with_z; compact_idx++) {
            if (mask & (1 << compact_idx)) {
                args_out.push_back(
                    this->downstream_edm_worker_location_info_addresses[vc_idx][compact_idx].value_or(0));
            }
        }

        // Pack buffer index semaphore addresses (dense packed based on mask)
        for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc1_with_z; compact_idx++) {
            if (mask & (1 << compact_idx)) {
                args_out.push_back(
                    this->downstream_edm_buffer_index_semaphore_addresses[vc_idx][compact_idx].value_or(0));
            }
        }
    } else {
        // For 1D: single downstream connection (only VC0 supported)
        TT_FATAL(vc_idx == 0, "VC1 is not supported for 1D routing");
        uint32_t mask = this->downstream_edms_connected_by_vc_mask.at(vc_idx);
        bool has_connection = mask != 0;

        uint32_t buffer_addr = this->downstream_edm_buffer_base_addresses[vc_idx][0].value_or(0);

        auto rt_args = std::initializer_list<uint32_t>{
            has_connection,
            buffer_addr,
            this->pack_downstream_noc_x_rt_arg(vc_idx),
            this->pack_downstream_noc_y_rt_arg(vc_idx),
            static_cast<uint32_t>(this->downstream_edm_worker_registration_addresses[vc_idx][0].value_or(0)),
            static_cast<uint32_t>(this->downstream_edm_worker_location_info_addresses[vc_idx][0].value_or(0)),
            static_cast<uint32_t>(this->downstream_edm_buffer_index_semaphore_addresses[vc_idx][0].value_or(0)),
        };

        args_out.reserve(args_out.size() + rt_args.size());
        std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(args_out));
    }
}

void StaticSizedChannelConnectionWriterAdapter::pack_adaptor_to_relay_rt_args(std::vector<uint32_t>& args_out) const {
    // Pack local tensix (relay) connection info at the end of runtime args
    // If no relay connection, just pack the flag (0)
    if (!this->relay_connection_info.is_connected) {
        args_out.push_back(0u);  // has_local_tensix_relay_connection = false
    } else {
        // Query the fabric router config from fabric context
        const auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        const auto& fabric_router_config = fabric_context.get_builder_context().get_fabric_router_config();

        // Pack full relay connection info
        // Query connection_buffer_index_id from fabric router config (consistent with other adapter connections)
        auto relay_rt_args = std::initializer_list<uint32_t>{
            1u,  // has_local_tensix_relay_connection = true
            static_cast<uint32_t>(this->relay_connection_info.buffer_base_address),  // relay_buffer_base_addr
            static_cast<uint32_t>(this->relay_connection_info.noc_xy.x),             // relay_noc_x
            static_cast<uint32_t>(this->relay_connection_info.noc_xy.y),             // relay_noc_y
            static_cast<uint32_t>(
                this->relay_connection_info.worker_registration_address),  // relay_connection_handshake_addr
            static_cast<uint32_t>(
                this->relay_connection_info.worker_location_info_address),            // relay_worker_location_info_addr
            static_cast<uint32_t>(this->relay_connection_info.free_slots_stream_id),  // relay_free_slots_stream_id
            static_cast<uint32_t>(
                fabric_router_config.tensix_relay_connection_buffer_index_id),  // relay_connection_buffer_index_id
                                                                                // (queried from fabric context)
        };

        args_out.reserve(args_out.size() + relay_rt_args.size());
        std::copy(relay_rt_args.begin(), relay_rt_args.end(), std::back_inserter(args_out));
    }
}

uint32_t StaticSizedChannelConnectionWriterAdapter::pack_downstream_noc_y_rt_arg(uint32_t vc_idx) const {
    if (!is_2D_routing) {
        // 1D routing: single downstream connection
        if (downstream_edms_connected_by_vc[vc_idx].empty()) {
            return 0;
        }
        return downstream_edms_connected_by_vc[vc_idx].front().second.y;
    }

    // 2D routing: dense pack based on mask (get mask internally)
    // CRITICAL: Must iterate through compact_idx in order, not push_back order,
    // to match the order used when packing addresses!
    uint32_t mask = this->downstream_edms_connected_by_vc_mask.at(vc_idx);
    uint32_t noc_y_packed = 0;
    uint32_t dense_idx = 0;

    // Iterate through compact indices in order to match address packing order
    for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc1_with_z; compact_idx++) {
        if (mask & (1 << compact_idx)) {
            // Find the connection with this compact_idx
            for (const auto& [direction, noc_xy] : downstream_edms_connected_by_vc[vc_idx]) {
                size_t conn_compact_idx = get_receiver_channel_compact_index(my_direction, direction);
                if (conn_compact_idx == compact_idx) {
                    noc_y_packed |= (noc_xy.y << (dense_idx * 8));
                    dense_idx++;
                    break;
                }
            }
        }
    }
    return noc_y_packed;
}

uint32_t StaticSizedChannelConnectionWriterAdapter::pack_downstream_noc_x_rt_arg(uint32_t vc_idx) const {
    if (!is_2D_routing) {
        // 1D routing: single downstream connection
        if (downstream_edms_connected_by_vc[vc_idx].empty()) {
            return 0;
        }
        return downstream_edms_connected_by_vc[vc_idx].front().second.x;
    }

    // 2D routing: dense pack based on mask (get mask internally)
    // CRITICAL: Must iterate through compact_idx in order, not push_back order,
    // to match the order used when packing addresses!
    uint32_t mask = this->downstream_edms_connected_by_vc_mask.at(vc_idx);
    uint32_t noc_x_packed = 0;
    uint32_t dense_idx = 0;

    // Iterate through compact indices in order to match address packing order
    for (size_t compact_idx = 0; compact_idx < builder_config::num_downstream_edms_2d_vc1_with_z; compact_idx++) {
        if (mask & (1 << compact_idx)) {
            // Find the connection with this compact_idx
            for (const auto& [direction, noc_xy] : downstream_edms_connected_by_vc[vc_idx]) {
                size_t conn_compact_idx = get_receiver_channel_compact_index(my_direction, direction);
                if (conn_compact_idx == compact_idx) {
                    noc_x_packed |= (noc_xy.x << (dense_idx * 8));
                    dense_idx++;
                    break;
                }
            }
        }
    }
    return noc_x_packed;
}
/*
 * For 2D fabric, downstream noc x/y coords are packed into uint32_t, one per byte
 * X and Y have separate uint32s
 */
uint32_t StaticSizedChannelConnectionWriterAdapter::encode_noc_ord_for_2d(
    const std::array<std::vector<std::pair<eth_chan_directions, CoreCoord>>, builder_config::num_max_receiver_channels>&
        downstream_edms_connected_by_vc,
    uint32_t vc_idx,
    const std::function<uint32_t(CoreCoord)>& get_noc_ord) const {
    if (!is_2D_routing) {
        // 1D routing: single downstream connection
        if (downstream_edms_connected_by_vc[vc_idx].empty()) {
            return 0;  // no connection here
        }
        TT_FATAL(
            downstream_edms_connected_by_vc[vc_idx].size() == 1,
            "Downstream edms connected by vc should be 1 for non-2D routing. vc_idx: {}, size: {}",
            vc_idx,
            downstream_edms_connected_by_vc[vc_idx].size());
        auto ord = get_noc_ord(downstream_edms_connected_by_vc[vc_idx].front().second);
        return ord;
    }  // 2D routing: encode NOC coordinates using compact index (works for both VC0 and VC1)
    uint32_t ord = 0;
    for (const auto& [direction, noc_xy] : downstream_edms_connected_by_vc[vc_idx]) {
        // Calculate compact index based on direction relative to my_direction
        size_t compact_index = get_receiver_channel_compact_index(my_direction, direction);
        ord |= (get_noc_ord(noc_xy) << (compact_index * 8));
    }
    return ord;
}

void StaticSizedChannelConnectionWriterAdapter::emit_ct_args(
    std::vector<uint32_t>& ct_args_out, size_t /*num_fwd_paths*/) const {
    // Always emit MAX_NUM_RECEIVER_CHANNELS elements (one per VC) to match device side array size
    ct_args_out.insert(
        ct_args_out.end(),
        this->downstream_sender_channels_num_buffers.begin(),
        this->downstream_sender_channels_num_buffers.begin() + builder_config::num_max_receiver_channels);

    // Validate that all connected VCs have non-zero buffer counts
    // downstream_sender_channels_num_buffers is indexed by VC (receiver channel), not by sender channel
    for (size_t vc_idx = 0; vc_idx < builder_config::num_max_receiver_channels; vc_idx++) {
        if (this->downstream_edms_connected_by_vc_set.contains(vc_idx)) {
            TT_FATAL(
                this->downstream_sender_channels_num_buffers[vc_idx] != 0,
                "Downstream sender channels num buffers must be greater than 0 for vc_idx: {}",
                vc_idx);
        }
    }
}

}  // namespace tt::tt_fabric
