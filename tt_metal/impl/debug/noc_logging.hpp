// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <host_api.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "impl/context/metal_env_impl.hpp"

namespace tt {
void ClearNocData(tt_metal::MetalEnvImpl& env, ChipId device_id);
void DumpNocData(
    tt_metal::MetalEnvImpl& env,
    const std::vector<ChipId>& devices,
    uint8_t num_hw_cqs,
    const tt_metal::DispatchCoreConfig& dispatch_core_config);
}  // namespace tt
