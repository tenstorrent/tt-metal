// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hal/lite_fabric_hal.hpp"
#include <umd/device/types/xy_pair.hpp>
#include <tt-metalium/control_plane.hpp>
#include <memory>

namespace lite_fabric {

void InitializeLiteFabric(std::shared_ptr<lite_fabric::LiteFabricHal>& lite_fabric_hal);

}  // namespace lite_fabric
