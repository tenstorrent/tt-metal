// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "context/metal_env_impl.hpp"

namespace tt::tt_fabric {

// Entry point: checks all guards internally (runtime option, Mock device, buffer enabled).
// Safe to call unconditionally — returns immediately if capture is not active.
void export_channel_trimming_capture(tt::tt_metal::MetalEnvImpl& env);

}  // namespace tt::tt_fabric
