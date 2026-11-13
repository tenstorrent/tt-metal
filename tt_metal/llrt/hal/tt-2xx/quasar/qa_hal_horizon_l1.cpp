// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads Horizon's L1 size from the register header its device kernels compile against. Each variant's header
// shares include guards and macro names with the others, so each needs its own translation unit.

#include <cstdint>

#include "internal/tt-2xx/tt-2.0.1/meta/registers/memory_port_cacheable_reg.h"
#include "qa_hal.hpp"

namespace tt::tt_metal::quasar {

uint32_t get_horizon_l1_size() { return MEMORY_PORT_CACHEABLE_MEM_PORT_MEM_SIZE; }

}  // namespace tt::tt_metal::quasar
