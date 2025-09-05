// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/builder/fabric_core_placement.hpp"
#include <tt-logger/tt-logger.hpp>

#include <vector>
#include <functional>

namespace tt::tt_fabric::core_placement {

static void run_default_galaxy_optimizer(
    const CorePlacementContext& ctx,
    tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder1,
    tt::tt_fabric::FabricEriscDatamoverBuilder& edm_builder2,
    size_t l) {

    if (!ctx.is_galaxy) return;

    constexpr uint32_t ring_noc_selection_link_threshold = 3;
    constexpr uint32_t line_noc_selection_link_threshold = 2;
    bool enable_noc_selection_opt = false;
    if (ctx.topology == Topology::Ring) {
        enable_noc_selection_opt =
            (ctx.num_links > ring_noc_selection_link_threshold) && (edm_builder1.my_noc_y != edm_builder2.my_noc_y);
    } else {
        enable_noc_selection_opt =
            (ctx.num_links > line_noc_selection_link_threshold) && (edm_builder1.my_noc_y != edm_builder2.my_noc_y);
    }
    log_debug(
        tt::LogTest,
        "Fabric MeshId {} ChipId {} edm_builder1 {} {} is connecting to edm_builder2 {} {} num links {}",
        *(edm_builder1.local_fabric_node_id.mesh_id),
        edm_builder1.local_fabric_node_id.chip_id,
        edm_builder1.my_noc_x,
        edm_builder1.my_noc_y,
        edm_builder2.my_noc_x,
        edm_builder2.my_noc_y,
        ctx.num_links);

    if (enable_noc_selection_opt) {
        if (edm_builder1.my_noc_x < edm_builder2.my_noc_x) {
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_forwarding_noc_ids[i] = 0;
                edm_builder2.config.receiver_channel_forwarding_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_local_write_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_local_write_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_sender_channels; i++) {
                edm_builder1.config.sender_channel_ack_noc_ids[i] = 1;
                edm_builder2.config.sender_channel_ack_noc_ids[i] = 0;
            }
        } else if (edm_builder1.my_noc_x > edm_builder2.my_noc_x) {
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_forwarding_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_forwarding_noc_ids[i] = 0;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_receiver_channels; i++) {
                edm_builder1.config.receiver_channel_local_write_noc_ids[i] = 1;
                edm_builder2.config.receiver_channel_local_write_noc_ids[i] = 1;
            }
            for (uint32_t i = 0; i < edm_builder1.config.num_sender_channels; i++) {
                edm_builder1.config.sender_channel_ack_noc_ids[i] = 0;
                edm_builder2.config.sender_channel_ack_noc_ids[i] = 1;
            }
        }
    }
}


void apply_core_placement_optimizations(
    const CorePlacementContext& ctx,
    FabricEriscDatamoverBuilder& edm_fwd,
    FabricEriscDatamoverBuilder& edm_bwd,
    size_t link_index) {
    bool enable_core_placement_opt = false;
    // currently is_galaxy is only being passed in through the fabric unit test, once we switch to fabric
    // device init, will use proper cluster type to decide which machine it is. For the optimzation on noc
    // selection, we empirically optimize on 3/4 links for linear, and 4 links on ring, as less links caused
    // perf degradation, potentially caused by sw overhead of checking two nocs.
    if (ctx.is_galaxy) {
        if (ctx.topology == Topology::Ring) {
            enable_core_placement_opt = (ctx.num_links > 3) && (edm_fwd.my_noc_y != edm_bwd.my_noc_y);
        } else {
            enable_core_placement_opt = (ctx.num_links > 2) && (edm_fwd.my_noc_y != edm_bwd.my_noc_y);
        }
    }

    if (enable_core_placement_opt) {
        run_default_galaxy_optimizer(ctx, edm_fwd, edm_bwd, link_index);
    }

}



} // namespace tt::tt_fabric::core_placement
