// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_core_manager.hpp"
#include "dispatch_core_common.hpp"
#include "get_platform_architecture.hpp"

namespace tt::tt_metal {

DispatchCoreAxis DispatchCoreConfig::get_default_axis() {
    return (tt::tt_metal::get_platform_architecture() == tt::ARCH::BLACKHOLE) ? DispatchCoreAxis::COL
                                                                              : DispatchCoreAxis::ROW;
}

}  // namespace tt::tt_metal
