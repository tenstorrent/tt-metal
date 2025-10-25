// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hal/lite_fabric_hal.hpp"
#include "tt_cluster.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/control_plane.hpp>

namespace lite_fabric {

void LaunchLiteFabric(tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const SystemDescriptor& desc);

}  // namespace lite_fabric
