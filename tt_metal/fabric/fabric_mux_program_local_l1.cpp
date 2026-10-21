// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_mux_program_local_l1.hpp"

#include <bitset>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>
#include "impl/buffers/semaphore.hpp"

namespace tt::tt_fabric::detail {

uint32_t reserve_fabric_mux_program_local_l1(
    tt::tt_metal::ProgramDescriptor& desc,
    const tt::tt_metal::CoreRangeSet& mux_cores,
    uint8_t cb_index,
    uint32_t map_size_bytes) {
    TT_FATAL(!mux_cores.empty(), "Fabric mux program-local L1 placement needs at least one mux core");
    for (const auto& cb : desc.cbs) {
        if (!cb.core_ranges.intersects(mux_cores)) {
            continue;
        }
        for (const auto& format : cb.format_descriptors) {
            TT_FATAL(
                format.buffer_index != cb_index,
                "Fabric mux circular buffer index {} is already used on cores {}, which overlap mux cores {}",
                cb_index,
                cb.core_ranges.str(),
                mux_cores.str());
        }
    }

    // One id for every mux core: the mux kernel's compile-time args, and its clients', are shared across them.
    std::bitset<tt::tt_metal::NUM_SEMAPHORES> used_semaphore_ids;
    for (const auto& semaphore : desc.semaphores) {
        if (semaphore.core_type == CoreType::WORKER && semaphore.core_ranges.intersects(mux_cores)) {
            used_semaphore_ids.set(semaphore.id);
        }
    }
    uint32_t semaphore_id = 0;
    while (semaphore_id < tt::tt_metal::NUM_SEMAPHORES && used_semaphore_ids.test(semaphore_id)) {
        semaphore_id++;
    }
    TT_FATAL(
        semaphore_id < tt::tt_metal::NUM_SEMAPHORES,
        "No semaphore id is free on every fabric mux core {}; all {} are in use on at least one of them",
        mux_cores.str(),
        tt::tt_metal::NUM_SEMAPHORES);
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = semaphore_id, .core_type = CoreType::WORKER, .core_ranges = mux_cores, .initial_value = 0});

    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = map_size_bytes,
        .core_ranges = mux_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = cb_index,
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = map_size_bytes,
        }}},
    });
    return semaphore_id;
}

uint32_t reserve_fabric_mux_program_local_l1(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& mux_cores,
    uint8_t cb_index,
    uint32_t map_size_bytes) {
    TT_FATAL(!mux_cores.empty(), "Fabric mux program-local L1 placement needs at least one mux core");
    // CreateSemaphore picks an id that is free on every core in mux_cores.
    const uint32_t semaphore_id = tt::tt_metal::CreateSemaphore(program, mux_cores, 0);
    tt::tt_metal::CreateCircularBuffer(
        program,
        mux_cores,
        tt::tt_metal::CircularBufferConfig(map_size_bytes, {{cb_index, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_index, map_size_bytes));
    return semaphore_id;
}

}  // namespace tt::tt_fabric::detail
