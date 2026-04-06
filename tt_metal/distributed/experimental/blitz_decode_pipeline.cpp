// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/experimental/blitz_decode_pipeline.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/device_types.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental::blitz {

namespace {

struct PhysicalPipelineStageConfig {
    uint32_t entry_node_tray_id;
    uint32_t exit_node_tray_id;
    uint32_t entry_node_asic_location;
    uint32_t exit_node_asic_location;
};

PhysicalSystemDescriptor create_physical_system_descriptor() {
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto& driver_ref = const_cast<tt::umd::Cluster&>(*cluster.get_driver());

    return tt::tt_metal::run_physical_system_discovery(driver_ref, distributed_context, rtoptions.get_target_device());
}

std::vector<PhysicalPipelineStageConfig> generate_physical_pipeline_config(bool rev_c) {
    auto num_procs = *(tt::tt_metal::MetalContext::instance().get_distributed_context_ptr()->size());
    std::uint32_t tray_id_top_right = rev_c ? 2 : 3;
    std::uint32_t tray_id_bottom_left = rev_c ? 3 : 2;

    switch (num_procs) {
        case 4:
            return {
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3}};
        case 16:
            return {
                // First Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Second Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Third Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                // Fourth Tray
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                // Wrap-around
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
            };

        case 32:
            return {
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                // jump to pod 3
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 4}};
        case 64:
            return {
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                // jump to pod 4
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                // jump to pod 3
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 7},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 5,
                 .exit_node_asic_location = 6},
                {.entry_node_tray_id = 1,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 7,
                 .exit_node_asic_location = 8},
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 8,
                 .exit_node_asic_location = 4},
                // jump to pod 2
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 2},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = tray_id_top_right,
                 .entry_node_asic_location = 3,
                 .exit_node_asic_location = 4},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 4,
                 .exit_node_asic_location = 3},
                {.entry_node_tray_id = 4,
                 .exit_node_tray_id = 4,
                 .entry_node_asic_location = 2,
                 .exit_node_asic_location = 1},
                {.entry_node_tray_id = tray_id_top_right,
                 .exit_node_tray_id = 1,
                 .entry_node_asic_location = 1,
                 .exit_node_asic_location = 5},
                // wrap-around
                {.entry_node_tray_id = tray_id_bottom_left,
                 .exit_node_tray_id = tray_id_bottom_left,
                 .entry_node_asic_location = 6,
                 .exit_node_asic_location = 5}};
        default: TT_THROW("Unsupported number of processes for creating Multi-Host Pipeline: {}", num_procs);
    }
}

std::vector<BlitzDecodePipelineStage> build_pipeline(
    const tt::tt_metal::PhysicalSystemDescriptor& physical_system_descriptor) {
    bool rev_c = physical_system_descriptor.is_bh_galaxy_rev_c();
    std::vector<PhysicalPipelineStageConfig> physical_pipeline_stage_configs = generate_physical_pipeline_config(rev_c);

    auto& metal_context = tt::tt_metal::MetalContext::instance();
    const auto& control_plane = metal_context.get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& mesh_ids = mesh_graph.get_all_mesh_ids();
    const std::size_t num_procs = static_cast<std::size_t>(*metal_context.get_distributed_context_ptr()->size());

    std::vector<BlitzDecodePipelineStage> logical_pipeline_stage_configs;
    logical_pipeline_stage_configs.reserve(physical_pipeline_stage_configs.size());

    std::unordered_map<tt::tt_fabric::MeshId, std::unordered_set<tt::tt_fabric::FabricNodeId>>
        used_intermesh_chips_by_mesh;

    for (std::size_t stage_index = 0; stage_index < physical_pipeline_stage_configs.size(); stage_index++) {
        const auto& stage_config = physical_pipeline_stage_configs[stage_index];
        (void)stage_config;
        const tt::tt_fabric::MeshId mesh_id = mesh_ids[stage_index % num_procs];
        const tt::tt_fabric::MeshId mesh_prev = mesh_ids[(stage_index + num_procs - 1) % num_procs];
        const tt::tt_fabric::MeshId mesh_next = mesh_ids[(stage_index + 1) % num_procs];

        std::size_t stage_repeat_on_mesh = 0;
        for (std::size_t k = 0; k < stage_index; k++) {
            if (mesh_ids[k % num_procs] == mesh_id) {
                stage_repeat_on_mesh++;
            }
        }

        const auto toward_prev_mesh = control_plane.get_exit_fabric_node_ids_between_meshes(mesh_id, mesh_prev);
        const auto toward_next_mesh = control_plane.get_exit_fabric_node_ids_between_meshes(mesh_id, mesh_next);

        TT_FATAL(
            !toward_prev_mesh.empty() && !toward_next_mesh.empty(),
            "Blitz pipeline stage {}: expected inter-mesh exit FabricNodeIds from mesh {} (toward prev {}: {} nodes, "
            "toward next {}: {} nodes)",
            stage_index,
            *mesh_id,
            *mesh_prev,
            toward_prev_mesh.size(),
            *mesh_next,
            toward_next_mesh.size());

        // Entry and exit must be distinct FabricNodeIds (legacy table used different tray/loc per role). Toward-prev
        // and toward-next lists can share the same chip; pick a distinct pair. Prefer chips not yet used on this mesh
        // in earlier pipeline stages when possible.
        auto& used_on_mesh = used_intermesh_chips_by_mesh[mesh_id];
        std::pair<tt::tt_fabric::FabricNodeId, tt::tt_fabric::FabricNodeId> entry_exit_pair{
            toward_prev_mesh.front(), toward_next_mesh.front()};
        bool found_pair = false;

        auto search_pairs = [&](bool avoid_already_used) {
            if (found_pair) {
                return;
            }
            for (std::size_t entry_offset = 0; entry_offset < toward_prev_mesh.size(); ++entry_offset) {
                const std::size_t entry_idx = (stage_repeat_on_mesh + entry_offset) % toward_prev_mesh.size();
                const tt::tt_fabric::FabricNodeId candidate_entry = toward_prev_mesh[entry_idx];
                for (std::size_t exit_offset = 0; exit_offset < toward_next_mesh.size(); ++exit_offset) {
                    const std::size_t exit_idx = (stage_repeat_on_mesh + exit_offset) % toward_next_mesh.size();
                    const tt::tt_fabric::FabricNodeId candidate_exit = toward_next_mesh[exit_idx];
                    if (candidate_entry == candidate_exit) {
                        continue;
                    }
                    if (avoid_already_used &&
                        (used_on_mesh.contains(candidate_entry) || used_on_mesh.contains(candidate_exit))) {
                        continue;
                    }
                    entry_exit_pair = {candidate_entry, candidate_exit};
                    found_pair = true;
                    return;
                }
            }
        };

        search_pairs(true);
        search_pairs(false);

        TT_FATAL(
            found_pair,
            "Blitz pipeline stage {} on mesh {}: no distinct inter-mesh exit chips toward {} vs {} (lists can be "
            "identical or single overlapping chip)",
            stage_index,
            *mesh_id,
            *mesh_prev,
            *mesh_next);

        used_on_mesh.insert(entry_exit_pair.first);
        used_on_mesh.insert(entry_exit_pair.second);

        const tt::tt_fabric::FabricNodeId& entry_fabric_node = entry_exit_pair.first;
        const tt::tt_fabric::FabricNodeId& exit_fabric_node = entry_exit_pair.second;

        const distributed::MeshCoordinate entry_node_coord =
            mesh_graph.chip_to_coordinate(entry_fabric_node.mesh_id, static_cast<ChipId>(entry_fabric_node.chip_id));
        const distributed::MeshCoordinate exit_node_coord =
            mesh_graph.chip_to_coordinate(exit_fabric_node.mesh_id, static_cast<ChipId>(exit_fabric_node.chip_id));

        logical_pipeline_stage_configs.emplace_back(BlitzDecodePipelineStage{
            .stage_index = stage_index, .entry_node_coord = entry_node_coord, .exit_node_coord = exit_node_coord});
    }

    return logical_pipeline_stage_configs;
}

}  // namespace

std::vector<BlitzDecodePipelineStage> generate_blitz_decode_pipeline(const distributed::MeshDevice& mesh_device) {
    (void)mesh_device;
    auto physical_system_descriptor = create_physical_system_descriptor();
    return build_pipeline(physical_system_descriptor);
}

}  // namespace tt::tt_metal::experimental::blitz
