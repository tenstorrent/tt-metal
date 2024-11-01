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

constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;

// Helper functions to convert NoC coordinates to NoC-0 coordinates, used in metal as "physical" coordinates.
#define NOC_0_X(noc_index, noc_size_x, x) (noc_index == 0 ? (x) : (noc_size_x-1-(x)))
#define NOC_0_Y(noc_index, noc_size_y, y) (noc_index == 0 ? (y) : (noc_size_y-1-(y)))

static_assert(L1_KERNEL_CONFIG_BASE % L1_ALIGNMENT == 0);
