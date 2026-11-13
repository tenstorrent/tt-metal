// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "dev_mem_map.h"
#include "hostdev/drisc_service/abi.hpp"

// The kernel's L1 outside its ELF: below MEM_DRISC_FIRMWARE_BASE after the command block, at the top of the firmware
// window, and past tt-metal's DRISC map. Namespaced because metal headers own bare names like L1_SIZE.
namespace l1 {

// One 64B line staging values published to the control page.
constexpr uint32_t SCRATCH = tt::tt_metal::drisc_service::abi::L1CommandBlockLayout::end_addr;
constexpr uint32_t SCRATCH_SIZE = 64;

// One window per client slot for its CMD_BATCH lists: a ring of two halves, each pulled from the host batch slot with
// one NOC read while the other is carved. A half is a whole number of 64 B lines, so its pulls stay 64B-aligned like
// the host slot. The windows sit at the top of the firmware window, above the kernel's text (drisc_service_guard.ld).
constexpr uint32_t WINDOW_HALF_COMMANDS = 8;
constexpr uint32_t WINDOW_HALF_SIZE = WINDOW_HALF_COMMANDS * sizeof(tt::tt_metal::drisc_service::abi::KernelCommand);
constexpr uint32_t WINDOW_SIZE = 2 * WINDOW_HALF_SIZE;
constexpr uint32_t BATCH_WINDOWS_SIZE = tt::tt_metal::drisc_service::abi::MAX_CLIENTS * WINDOW_SIZE;
constexpr uint32_t BATCH_WINDOWS_START = (MEM_DRISC_MAP_END - BATCH_WINDOWS_SIZE) & ~63u;

// Bounce arena for the DRAM <-> host streaming pipe: a byte ring of variably
// sized extents (one per in-flight transaction), each capped at CHUNK_SIZE.
constexpr uint32_t BOUNCE_BUFFER_START = (MEM_DRISC_BANK_TO_NOC_SCRATCH + 63) & ~63u;
constexpr uint32_t CHUNK_SIZE = 24 * 1024;
constexpr uint32_t BOUNCE_BUFFER_SIZE = 4 * CHUNK_SIZE;

static_assert(SCRATCH % 64 == 0);
static_assert(SCRATCH + SCRATCH_SIZE <= MEM_DRISC_FIRMWARE_BASE);
static_assert(WINDOW_HALF_SIZE % 64 == 0);
static_assert(BATCH_WINDOWS_START >= MEM_DRISC_FIRMWARE_BASE);
static_assert(BOUNCE_BUFFER_START + BOUNCE_BUFFER_SIZE <= MEM_DRISC_L1_SIZE);

}  // namespace l1
