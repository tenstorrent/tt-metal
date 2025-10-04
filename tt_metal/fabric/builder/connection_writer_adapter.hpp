// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

class ChannelConnectionWriterAdapter {
public:
    // Adds downstream noc x/y
    void add_downstream_connection(uint32_t vc_idx, const SenderWorkerAdapterSpec& adapter_spec);

private:
    std::vector<std::optional<size_t>> downstream_edm_vcs_noc_x = {};
    std::vector<std::optional<size_t>> downstream_edm_vcs_noc_y = {};
    std::vector<std::optional<size_t>> downstream_edm_vcs_worker_registration_address = {};
    std::vector<std::optional<size_t>> downstream_edm_vcs_worker_location_info_address = {};

    // std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms> downstream_edm_vcs_noc_x = {};
    // std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms> downstream_edm_vcs_noc_y = {};
};

// TODO: add transient vs persistent variants

class StaticSizedChannelConnectionWriterAdapter : public ChannelConnectionWriterAdapter {
public:
    StaticSizedChannelConnectionWriterAdapter(
        FabricStaticSizedChannelsAllocator& allocator, tt::tt_fabric::Topology topology) :
        is_2D_routing(topology == tt::tt_fabric::Topology::Mesh || topology == tt::tt_fabric::Topology::Torus) {}

    void add_downstream_connection(
        uint32_t sender_vc_idx,
        eth_chan_directions downstream_direction,
        CoreCoord downstream_noc_xy,
        bool is_2D_routing,
        bool is_vc1) {
        downstream_edms_connected_by_vc.resize(sender_vc_idx + 1);
        downstream_edms_connected_by_vc[sender_vc_idx].push_back(
            {downstream_direction, CoreCoord(downstream_noc_xy.x, downstream_noc_xy.y)});

        if (is_2D_routing) {
            if (!is_vc1) {
                this->downstream_edms_connected |= (1 << downstream_direction);
            }
        } else {
            this->downstream_edms_connected = 1;
        }
    }

    uint32_t get_downstream_edms_connected(bool is_2d_routing, bool is_vc1) const {
        return this->downstream_edms_connected;
    }

    uint32_t get_downstream_edm_vcs_noc_y_rt_arg(uint32_t vc_idx) const {
        return encode_noc_ord_for_2d(
            this->downstream_edms_connected_by_vc[vc_idx], [](CoreCoord noc_xy) { return noc_xy.y; });
    }
    uint32_t get_downstream_edm_vcs_noc_x_rt_arg(uint32_t vc_idx) const {
        return encode_noc_ord_for_2d(
            this->downstream_edms_connected_by_vc[vc_idx], [](CoreCoord noc_xy) { return noc_xy.x; });
    }

private:
    uint32_t encode_noc_ord_for_2d(
        const std::vector<std::pair<eth_chan_directions, CoreCoord>>& downstream_edms_connected_by_vc,
        std::function<uint32_t(CoreCoord)> get_noc_ord) const {
        uint32_t ord = 0;
        for (const auto& [direction, noc_xy] : downstream_edms_connected_by_vc) {
            ord |= (get_noc_ord(noc_xy) << (direction * 8));
        }
        return ord;
    }

    std::vector<std::vector<std::pair<eth_chan_directions, CoreCoord>>> downstream_edms_connected_by_vc = {};
    std::array<size_t, builder_config::num_sender_channels> sender_channels_num_buffers = {};
    std::array<size_t, builder_config::num_sender_channels> local_sender_channels_buffer_address = {};
    std::array<size_t, builder_config::num_sender_channels> remote_sender_channels_base_address = {};
    std::array<size_t, builder_config::num_downstream_sender_channels> downstream_sender_channels_num_buffers = {};
    std::vector<std::optional<size_t>> downstream_edm_vcs_buffer_base_address = {};
    uint32_t downstream_edms_connected = 0;

    bool is_2D_routing = false;

    // std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
    //     downstream_edm_vcs_buffer_base_address = {};
    // std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
    //     downstream_edm_vcs_worker_registration_address = {};
    // std::array<std::optional<size_t>, FabricEriscDatamoverConfig::max_downstream_edms>
    //     downstream_edm_vcs_worker_location_info_address = {};
};

// class ElasticChannelConnectionWriterAdapter : public ChannelConnectionWriterAdapter {
// public:
//     ElasticChannelConnectionWriterAdapter(FabricElasticChannelAllocator& allocator);
// };

}  // namespace tt::tt_fabric
