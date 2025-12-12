// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/connection_writer_adapter.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_tensix_builder.hpp"
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_fabric {

namespace {
const char* direction_to_string(eth_chan_directions direction) {
    switch (direction) {
        case eth_chan_directions::EAST: return "EAST";
        case eth_chan_directions::WEST: return "WEST";
        case eth_chan_directions::NORTH: return "NORTH";
        case eth_chan_directions::SOUTH: return "SOUTH";
        default: return "UNKNOWN";
    }
}
}  // namespace

StaticSizedChannelConnectionWriterAdapter::StaticSizedChannelConnectionWriterAdapter(
    FabricStaticSizedChannelsAllocator& /*allocator*/,
    tt::tt_fabric::Topology topology,
    eth_chan_directions my_direction) :
    is_2D_routing(topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Torus),
    my_direction(my_direction) {}

void StaticSizedChannelConnectionWriterAdapter::add_downstream_connection(
    const SenderWorkerAdapterSpec& adapter_spec,
    uint32_t inbound_vc_idx,
    eth_chan_directions downstream_direction,
    CoreCoord downstream_noc_xy,
    bool is_2D_routing) {
    downstream_edms_connected_by_vc.at(inbound_vc_idx).push_back(
        {downstream_direction, CoreCoord(downstream_noc_xy.x, downstream_noc_xy.y)});

    if (is_2D_routing) {
        // Calculate compact index based on downstream_direction relative to my_direction
        // The compact index excludes the router's own direction
        // For EAST router (my_direction=0): WEST(1)→0, NORTH(2)→1, SOUTH(3)→2
        // For WEST router (my_direction=1): EAST(0)→0, NORTH(2)→1, SOUTH(3)→2
        // For NORTH router (my_direction=2): EAST(0)→0, WEST(1)→1, SOUTH(3)→2
        // For SOUTH router (my_direction=3): EAST(0)→0, WEST(1)→1, NORTH(2)→2
        size_t compact_index = get_receiver_channel_compact_index(my_direction, downstream_direction);
        this->downstream_edms_connected_by_vc_mask.at(inbound_vc_idx) |= (1 << compact_index);

        // Store addresses indexed by [vc_idx][compact_index]
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
    // Record the starting size to track what we're adding
    size_t args_start_size = args_out.size();

    if (is_2D_routing) {
        // Get the appropriate downstream EDM count based on VC index
        uint32_t num_downstream_edms = (vc_idx == 0) ? builder_config::get_vc0_downstream_edm_count(is_2D_routing)
                                                     : builder_config::get_vc1_downstream_edm_count(is_2D_routing);

        // Pack connection mask for this VC (3-bit mask for VC0, 3-bit mask for VC1)
        args_out.push_back(this->downstream_edms_connected_by_vc_mask.at(vc_idx));

        // Pack buffer base addresses (one per compact index)
        for (size_t compact_idx = 0; compact_idx < num_downstream_edms; compact_idx++) {
            uint32_t buffer_addr = this->downstream_edm_buffer_base_addresses[vc_idx][compact_idx].value_or(0);
            args_out.push_back(buffer_addr);
        }

        // Pack NOC X and Y (already compacted properly)
        args_out.push_back(this->pack_downstream_noc_x_rt_arg(vc_idx));
        args_out.push_back(this->pack_downstream_noc_y_rt_arg(vc_idx));

        // Pack worker registration addresses (connection handshake addresses)
        for (size_t compact_idx = 0; compact_idx < num_downstream_edms; compact_idx++) {
            args_out.push_back(this->downstream_edm_worker_registration_addresses[vc_idx][compact_idx].value_or(0));
        }

        // Pack worker location info addresses
        for (size_t compact_idx = 0; compact_idx < num_downstream_edms; compact_idx++) {
            args_out.push_back(this->downstream_edm_worker_location_info_addresses[vc_idx][compact_idx].value_or(0));
        }

        // Pack buffer index semaphore addresses
        for (size_t compact_idx = 0; compact_idx < num_downstream_edms; compact_idx++) {
            args_out.push_back(this->downstream_edm_buffer_index_semaphore_addresses[vc_idx][compact_idx].value_or(0));
        }
    } else {
        // For 1D: single downstream connection (only VC0 supported)
        TT_FATAL(vc_idx == 0, "VC1 is not supported for 1D routing");
        bool has_connection = this->downstream_edms_connected_by_vc_mask.at(vc_idx) != 0;

        uint32_t buffer_addr = this->downstream_edm_buffer_base_addresses[vc_idx][0].value_or(0);

        auto rt_args = std::initializer_list<uint32_t>{
            has_connection,
            buffer_addr,
            this->pack_downstream_noc_x_rt_arg(vc_idx),
            this->pack_downstream_noc_y_rt_arg(vc_idx),
            this->downstream_edm_worker_registration_addresses[vc_idx][0].value_or(0),
            this->downstream_edm_worker_location_info_addresses[vc_idx][0].value_or(0),
            this->downstream_edm_buffer_index_semaphore_addresses[vc_idx][0].value_or(0),
        };

        args_out.reserve(args_out.size() + rt_args.size());
        std::copy(rt_args.begin(), rt_args.end(), std::back_inserter(args_out));
    }

    // Log the packed runtime arguments
    size_t num_packed_args = args_out.size() - args_start_size;
    std::string packed_values_str;

    // Calculate indices for NOC X and Y based on routing type
    size_t noc_x_idx = 0;
    size_t noc_y_idx = 0;
    uint32_t num_downstream_edms = 0;

    if (is_2D_routing) {
        num_downstream_edms = (vc_idx == 0) ? builder_config::get_vc0_downstream_edm_count(is_2D_routing)
                                            : builder_config::get_vc1_downstream_edm_count(is_2D_routing);
        // Structure: [mask(1), buffer_addrs(num_downstream_edms), noc_x(1), noc_y(1), ...]
        noc_x_idx = args_start_size + 1 + num_downstream_edms;
        noc_y_idx = args_start_size + 2 + num_downstream_edms;
    } else {
        // Structure: [has_connection(1), buffer_addr(1), noc_x(1), noc_y(1), ...]
        noc_x_idx = args_start_size + 2;
        noc_y_idx = args_start_size + 3;
        num_downstream_edms = 1;  // 1D routing has single downstream EDM
    }

    for (size_t i = args_start_size; i < args_out.size(); i++) {
        if (i > args_start_size) {
            packed_values_str += ", ";
        }

        // Special formatting for NOC X and Y coordinates
        if (i == noc_x_idx) {
            // Extract 8-bit values from packed NOC X (one byte per downstream EDM)
            uint32_t packed_x = args_out[i];
            std::string x_coords;
            for (uint32_t byte_idx = 0; byte_idx < num_downstream_edms && byte_idx < 4; byte_idx++) {
                uint8_t x_val = (packed_x >> (byte_idx * 8)) & 0xFF;
                if (!x_coords.empty()) {
                    x_coords += ", ";
                }
                x_coords += std::to_string(x_val);
            }
            packed_values_str += "NOC_X[" + x_coords + "]";
        } else if (i == noc_y_idx) {
            // Extract 8-bit values from packed NOC Y (one byte per downstream EDM)
            uint32_t packed_y = args_out[i];
            std::string y_coords;
            for (uint32_t byte_idx = 0; byte_idx < num_downstream_edms && byte_idx < 4; byte_idx++) {
                uint8_t y_val = (packed_y >> (byte_idx * 8)) & 0xFF;
                if (!y_coords.empty()) {
                    y_coords += ", ";
                }
                y_coords += std::to_string(y_val);
            }
            packed_values_str += "NOC_Y[" + y_coords + "]";
        } else {
            packed_values_str += std::to_string(args_out[i]);
        }
    }

    // Extract and format NOC coordinates as (x, y) pairs
    std::string noc_pairs_str;
    if (noc_x_idx < args_out.size() && noc_y_idx < args_out.size()) {
        uint32_t packed_x = args_out[noc_x_idx];
        uint32_t packed_y = args_out[noc_y_idx];

        for (uint32_t byte_idx = 0; byte_idx < num_downstream_edms && byte_idx < 4; byte_idx++) {
            uint8_t x_val = (packed_x >> (byte_idx * 8)) & 0xFF;
            uint8_t y_val = (packed_y >> (byte_idx * 8)) & 0xFF;

            if (!noc_pairs_str.empty()) {
                noc_pairs_str += ", ";
            }
            noc_pairs_str += "(" + std::to_string(x_val) + "," + std::to_string(y_val) + ")";
        }
    }

    log_info(
        tt::LogFabric,
        "pack_inbound_channel_rt_args: VC={}, EDM_direction={}, num_packed_args={}, packed_values=[{}], "
        "NOC_coords=[{}]",
        vc_idx,
        direction_to_string(this->my_direction),
        num_packed_args,
        packed_values_str,
        noc_pairs_str);
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
            1u,                                                            // has_local_tensix_relay_connection = true
            this->relay_connection_info.buffer_base_address,               // relay_buffer_base_addr
            this->relay_connection_info.noc_xy.x,                          // relay_noc_x
            this->relay_connection_info.noc_xy.y,                          // relay_noc_y
            this->relay_connection_info.worker_registration_address,       // relay_connection_handshake_addr
            this->relay_connection_info.worker_location_info_address,      // relay_worker_location_info_addr
            this->relay_connection_info.free_slots_stream_id,              // relay_free_slots_stream_id
            fabric_router_config.tensix_relay_connection_buffer_index_id,  // relay_connection_buffer_index_id (queried
                                                                           // from fabric context)
        };

        args_out.reserve(args_out.size() + relay_rt_args.size());
        std::copy(relay_rt_args.begin(), relay_rt_args.end(), std::back_inserter(args_out));
    }
}

// uint32_t StaticSizedChannelConnectionWriterAdapter::get_downstream_edms_connected() const {
//     // Return combined mask for backward compatibility (sum of all VC masks)
//     // Note: This may not be accurate if multiple VCs have overlapping compact indices
//     uint32_t combined = 0;
//     for (size_t vc_idx = 0; vc_idx < builder_config::num_max_receiver_channels; vc_idx++) {
//         combined |= this->downstream_edms_connected_by_vc_mask.at(vc_idx);
//     }
//     return combined;
// }

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
    const std::array<std::vector<std::pair<eth_chan_directions, CoreCoord>>, builder_config::num_max_receiver_channels>&
        downstream_edms_connected_by_vc,
    uint32_t vc_idx,
    const std::function<uint32_t(CoreCoord)>& get_noc_ord) const {
    if (!is_2D_routing) {
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
            size_t compact_index = get_receiver_channel_compact_index(my_direction, direction);
            ord |= (get_noc_ord(noc_xy) << (compact_index * 8));
        }
        return ord;
    }
}

void StaticSizedChannelConnectionWriterAdapter::emit_ct_args(
    std::vector<uint32_t>& ct_args_out, size_t /*num_fwd_paths*/) const {
    // Always emit MAX_NUM_RECEIVER_CHANNELS elements (one per VC) to match device side array size
    ct_args_out.insert(
        ct_args_out.end(),
        this->downstream_sender_channels_num_buffers.begin(),
        this->downstream_sender_channels_num_buffers.begin() + builder_config::num_max_receiver_channels);
    log_debug(LogFabric, "downstream_sender_channels_num_buffers:");
    for (size_t i = 0; i < this->downstream_sender_channels_num_buffers.size(); ++i) {
        log_debug(LogFabric, "  [{}]: {}", i, this->downstream_sender_channels_num_buffers[i]);
    }

    // Validate that all connected VCs have non-zero buffer counts
    // downstream_sender_channels_num_buffers is indexed by VC (receiver channel), not by sender channel
    for (size_t vc_idx = 0; vc_idx < builder_config::num_max_receiver_channels; vc_idx++) {
        if (this->downstream_edms_connected_by_vc_set.find(vc_idx) != this->downstream_edms_connected_by_vc_set.end()) {
            TT_FATAL(
                this->downstream_sender_channels_num_buffers[vc_idx] != 0,
                "Downstream sender channels num buffers must be greater than 0 for vc_idx: {}",
                vc_idx);
        }
    }
}

}  // namespace tt::tt_fabric
