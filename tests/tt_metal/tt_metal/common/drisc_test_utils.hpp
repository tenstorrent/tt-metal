// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

// The lowest logical DRAM core of `bank` that runs DRISC firmware, for a test that needs one DRISC.
// get_metal_dram_cores leaves out the endpoints the syseng firmware may own: the NOC0 endpoint
// (logical y 0), and on a DRAM-harvested Blackhole the NOC1 endpoint (logical y 1) too.
inline CoreCoord first_drisc_core(const distributed::MeshDevice& mesh_device, uint32_t bank = 0) {
    const auto& soc_desc =
        MetalContext::instance(mesh_device.impl().get_context_id()).get_cluster().get_soc_desc(mesh_device.build_id());
    std::optional<CoreCoord> first;
    for (const CoreCoord& core : soc_desc.get_metal_dram_cores(CoordSystem::LOGICAL)) {
        if (core.x == bank && (!first.has_value() || core.y < first->y)) {
            first = core;
        }
    }
    TT_FATAL(first.has_value(), "DRAM bank {} has no core that runs DRISC firmware", bank);
    return *first;
}

}  // namespace tt::tt_metal
