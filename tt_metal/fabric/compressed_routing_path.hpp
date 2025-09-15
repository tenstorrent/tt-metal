// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hostdevcommon/fabric_common.h"

namespace tt::tt_fabric {

// Template specializations are defined in compressed_routing_path.cpp
template struct routing_path_t<1, false>;
template struct routing_path_t<1, true>;
template struct routing_path_t<2, true>;

}  // namespace tt::tt_fabric
