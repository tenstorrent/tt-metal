// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common_values.hpp"
#include "dev_mem_map.h"
#include "noc/noc_parameters.h"
#include "hostdevcommon/profiler_common.h"

/*
* This file contains addresses that are visible to both host and device compiled code.
*/

// Reserved DRAM addresses
// Host writes (4B value) to and reads from DRAM_BARRIER_BASE across all channels to ensure previous writes have been committed
constexpr static std::uint32_t DRAM_BARRIER_BASE = 0;
constexpr static std::uint32_t DRAM_ALIGNMENT = NOC_DRAM_READ_ALIGNMENT_BYTES >= NOC_DRAM_WRITE_ALIGNMENT_BYTES ? NOC_DRAM_READ_ALIGNMENT_BYTES : NOC_DRAM_WRITE_ALIGNMENT_BYTES;
constexpr static std::uint32_t DRAM_BARRIER_SIZE = ((sizeof(uint32_t) + DRAM_ALIGNMENT - 1) / DRAM_ALIGNMENT) * DRAM_ALIGNMENT;
constexpr static std::uint32_t DRAM_UNRESERVED_BASE = DRAM_BARRIER_BASE + DRAM_BARRIER_SIZE; // Start of unreserved space

// Take max alignment to satisfy NoC rd/wr constraints
// Tensix/Eth -> PCIe/DRAM src and dst addrs must be L1_ALIGNMENT aligned
// PCIe/DRAM -> Tensix/Eth src and dst addrs must be DRAM_ALIGNMENT aligned
// Tensix/Eth <-> Tensix/Eth src and dst addrs must be L1_ALIGNMENT aligned
constexpr static std::uint32_t ALLOCATOR_ALIGNMENT = DRAM_ALIGNMENT >= L1_ALIGNMENT ? DRAM_ALIGNMENT : L1_ALIGNMENT;

// TODO: move these out of the memory map into profiler code
constexpr static std::uint32_t PROFILER_OP_SUPPORT_COUNT = 1000;
constexpr static std::uint32_t PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC = kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE * (kernel_profiler::PROFILER_L1_PROGRAM_ID_COUNT +  kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT + kernel_profiler::PROFILER_L1_OP_MIN_OPTIONAL_MARKER_COUNT) * PROFILER_OP_SUPPORT_COUNT;
constexpr static std::uint32_t PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC = PROFILER_FULL_HOST_VECTOR_SIZE_PER_RISC * sizeof(uint32_t);

static_assert (PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC > kernel_profiler::PROFILER_L1_BUFFER_SIZE);

// Kernel config buffer is WIP
// Size is presently based on the old sizes of the RTAs + CB config + Sems
// plus some extra space freed up in the mem map
constexpr static std::uint32_t L1_KERNEL_CONFIG_BASE = MEM_MAP_END;
constexpr static std::uint32_t L1_KERNEL_CONFIG_SIZE = 4 * 1024 + 256 + 128 + 512;

constexpr static std::uint32_t IDLE_ERISC_L1_KERNEL_CONFIG_BASE = 32 * 1024;

constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;
constexpr static std::uint32_t UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG = 4;

constexpr static std::uint32_t L1_UNRESERVED_BASE = ((L1_KERNEL_CONFIG_BASE + L1_KERNEL_CONFIG_SIZE - 1) | (DRAM_ALIGNMENT - 1)) + 1;

constexpr static std::uint32_t ERISC_L1_UNRESERVED_BASE = L1_UNRESERVED_BASE; // Start of unreserved space

// Helper functions to convert NoC coordinates to NoC-0 coordinates, used in metal as "physical" coordinates.
#define NOC_0_X(noc_index, noc_size_x, x) (noc_index == 0 ? (x) : (noc_size_x-1-(x)))
#define NOC_0_Y(noc_index, noc_size_y, y) (noc_index == 0 ? (y) : (noc_size_y-1-(y)))

static_assert(L1_KERNEL_CONFIG_BASE % L1_ALIGNMENT == 0);
