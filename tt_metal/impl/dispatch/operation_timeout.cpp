// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/operation_timeout.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::experimental {

std::chrono::duration<float> GetOperationTimeout() {
    return MetalContext::instance().rtoptions().get_timeout_duration_for_operations();
}

void SetOperationTimeout(std::chrono::duration<float> timeout) {
    MetalContext::instance().rtoptions().set_timeout_duration_for_operations(timeout);
}

}  // namespace tt::tt_metal::experimental
