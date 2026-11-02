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

struct FabricMuxProgramLocalL1 {
    uint8_t cb_index = 0;
    uint32_t l1_base_semaphore_id = 0;
};

// Reserves program-local L1 for a fabric mux's memory map on mux_cores: a circular buffer of map_size_bytes at the
// lowest circular buffer index free on every mux core, and a semaphore with initial value 0 and the same id on every
// mux core, in which the mux publishes the buffer's address. Dispatch writes a core's circular buffer configs up to
// its highest index at every launch, so the lowest free index keeps that write small.
FabricMuxProgramLocalL1 reserve_fabric_mux_program_local_l1(
    tt::tt_metal::ProgramDescriptor& desc, const tt::tt_metal::CoreRangeSet& mux_cores, uint32_t map_size_bytes);
FabricMuxProgramLocalL1 reserve_fabric_mux_program_local_l1(
    tt::tt_metal::Program& program, const tt::tt_metal::CoreRangeSet& mux_cores, uint32_t map_size_bytes);

}  // namespace tt::tt_fabric::detail
