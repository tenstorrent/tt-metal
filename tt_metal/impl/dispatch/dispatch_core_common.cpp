// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_core_common.hpp"
#include "dispatch_core_manager.hpp"
#include "get_platform_architecture.hpp"
#include <umd/device/types/arch.h>

enum class CoreType;

namespace tt::tt_metal {

DispatchCoreAxis DispatchCoreConfig::get_default_axis() {
    return (tt::tt_metal::get_platform_architecture() == tt::ARCH::BLACKHOLE) ? DispatchCoreAxis::COL
                                                                              : DispatchCoreAxis::ROW;
}

DispatchCoreConfig get_dispatch_core_config() { return dispatch_core_manager::instance().get_dispatch_core_config(); }

CoreType get_dispatch_core_type() { return dispatch_core_manager::instance().get_dispatch_core_type(); };

}  // namespace tt::tt_metal
