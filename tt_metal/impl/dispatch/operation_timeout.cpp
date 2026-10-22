// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>

#include <tt-metalium/experimental/operation_timeout.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::experimental {

float GetOperationTimeoutSeconds() {
    return MetalContext::instance().rtoptions().get_timeout_duration_for_operations().count();
}

void SetOperationTimeoutSeconds(float seconds) {
    MetalContext::instance().rtoptions().set_timeout_duration_for_operations(std::chrono::duration<float>(seconds));
}

}  // namespace tt::tt_metal::experimental
