// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

extern template class compressed_routing_table_t<MAX_MESH_SIZE>;
#if MAX_MESH_SIZE != MAX_NUM_MESHES
extern template class compressed_routing_table_t<MAX_NUM_MESHES>;
#endif

}  // namespace tt::tt_fabric
