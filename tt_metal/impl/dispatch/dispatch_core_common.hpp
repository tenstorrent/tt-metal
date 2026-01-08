// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/dispatch_core_common.hpp>

namespace tt::tt_metal {

enum DispatchWorkerType : uint32_t {
    PREFETCH = 0,
    PREFETCH_HD = 1,
    PREFETCH_H = 2,
    PREFETCH_D = 3,
    DISPATCH = 4,
    DISPATCH_HD = 5,
    DISPATCH_H = 6,
    DISPATCH_D = 7,
    DISPATCH_S = 8,
    FABRIC_MUX = 17,         // Downstream from MMIO to remote mux. Tunnel index is required.
    RETURN_FABRIC_MUX = 18,  // Upstream from remote to MMIO mux. Tunnel index will be determined from the device id.
    COUNT,
};

// Helper functions to get the dispatch core config/type
DispatchCoreConfig get_dispatch_core_config();

}  // namespace tt::tt_metal
