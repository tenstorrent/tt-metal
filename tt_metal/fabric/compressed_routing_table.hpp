// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

extern template struct compressed_routing_table_t<MAX_MESH_SIZE>;
extern template struct compressed_routing_table_t<MAX_NUM_MESHES>;

}  // namespace tt::tt_fabric
