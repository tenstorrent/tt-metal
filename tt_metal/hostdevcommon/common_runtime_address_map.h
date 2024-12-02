// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common_values.hpp"
#include "dev_mem_map.h"
#include "noc/noc_parameters.h"

/*
 * This file contains addresses that are visible to both host and device compiled code.
 */

// TODO: move this to the memory manager, make configurable through the API
constexpr static std::uint32_t L1_KERNEL_CONFIG_BASE = MEM_MAP_END;
constexpr static std::uint32_t L1_KERNEL_CONFIG_SIZE = 69 * 1024;

static_assert(L1_KERNEL_CONFIG_BASE % L1_ALIGNMENT == 0);
