// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/fabric/hw/inc/addrgen_api_common.h"

namespace tt::tt_fabric {

namespace mesh {

// Expose all addrgen functions from addrgen_api_common.h
using tt::tt_fabric::to_noc_fused_unicast_write_atomic_inc;
using tt::tt_fabric::to_noc_unicast_scatter_write;
using tt::tt_fabric::to_noc_unicast_write;

// Expose constants and helper functions
using tt::tt_fabric::max_fabric_addrgen_payload_size;
using tt::tt_fabric::validate_max_payload_size;

}  // namespace mesh

};  // namespace tt::tt_fabric
