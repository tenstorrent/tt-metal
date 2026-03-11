// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "tt_metal/llrt/hal.hpp"

namespace tt::scaleout_tools {

using tt::ChipId;
using tt::CoordSystem;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::FWMailboxMsg;
using tt::tt_metal::PhysicalSystemDescriptor;

struct ResetLink {
    ChipId chip_id;
    uint32_t channel;
    std::string log_message;
};

// ============================================================================
// Consolidated helpers (should be arch agnostic)
// ============================================================================

void send_reset_msg_to_links(const std::vector<ResetLink>& links_to_reset);

}  // namespace tt::scaleout_tools
