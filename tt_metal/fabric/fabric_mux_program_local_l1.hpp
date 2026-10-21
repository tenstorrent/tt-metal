// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {
class Program;
struct ProgramDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_fabric::detail {

// Reserves program-local L1 for a fabric mux's memory map on mux_cores: a circular buffer of map_size_bytes at
// cb_index, and a semaphore with initial value 0 and the same id on every mux core, in which the mux publishes the
// buffer's address. Returns the semaphore id.
uint32_t reserve_fabric_mux_program_local_l1(
    tt::tt_metal::ProgramDescriptor& desc,
    const tt::tt_metal::CoreRangeSet& mux_cores,
    uint8_t cb_index,
    uint32_t map_size_bytes);
uint32_t reserve_fabric_mux_program_local_l1(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& mux_cores,
    uint8_t cb_index,
    uint32_t map_size_bytes);

}  // namespace tt::tt_fabric::detail
