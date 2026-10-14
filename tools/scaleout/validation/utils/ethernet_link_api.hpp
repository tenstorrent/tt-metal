// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

// Stops any Metal kernel left on the links' ERISC0s (e.g. fabric routers from a killed workload) and returns the
// cores to base firmware. Blackhole only.
void return_links_to_base_firmware(const std::vector<ResetLink>& links);

// Returns the links whose ethernet firmware did not process the port-down message in time.
std::vector<ResetLink> send_port_down_msg_to_links(const std::vector<ResetLink>& links_to_reset);

// Returns false if the ethernet firmware on any link did not process a reset message in time.
bool send_reset_msg_to_links(const std::vector<ResetLink>& links_to_reset);

}  // namespace tt::scaleout_tools
