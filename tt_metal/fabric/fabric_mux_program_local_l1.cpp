// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/fabric_mux_program_local_l1.hpp"

#include <bitset>
#include <cstddef>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>
#include "impl/buffers/circular_buffer.hpp"
#include "impl/buffers/semaphore.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_fabric::detail {

namespace {

template <std::size_t N>
uint32_t lowest_free_id(const std::bitset<N>& used_ids) {
    uint32_t id = 0;
    while (id < N && used_ids.test(id)) {
        id++;
    }
    return id;
}

uint8_t lowest_free_cb_index(
    const std::bitset<NUM_CIRCULAR_BUFFERS>& used_cb_indices, const tt::tt_metal::CoreRangeSet& mux_cores) {
    const uint32_t cb_index = lowest_free_id(used_cb_indices);
    TT_FATAL(
        cb_index < NUM_CIRCULAR_BUFFERS,
        "No circular buffer index is free on every fabric mux core {}; all {} are in use on at least one of them",
        mux_cores.str(),
        NUM_CIRCULAR_BUFFERS);
    return static_cast<uint8_t>(cb_index);
}

}  // namespace

FabricMuxProgramLocalL1 reserve_fabric_mux_program_local_l1(
    tt::tt_metal::ProgramDescriptor& desc, const tt::tt_metal::CoreRangeSet& mux_cores, uint32_t map_size_bytes) {
    TT_FATAL(!mux_cores.empty(), "Fabric mux program-local L1 placement needs at least one mux core");
    std::bitset<NUM_CIRCULAR_BUFFERS> used_cb_indices;
    for (const auto& cb : desc.cbs) {
        if (!cb.core_ranges.intersects(mux_cores)) {
            continue;
        }
        for (const auto& format : cb.format_descriptors) {
            used_cb_indices.set(format.buffer_index);
        }
        for (const auto& format : cb.remote_format_descriptors) {
            used_cb_indices.set(format.buffer_index);
        }
    }
    const uint8_t cb_index = lowest_free_cb_index(used_cb_indices, mux_cores);

    // One id for every mux core: the mux kernel's compile-time args, and its clients', are shared across them.
    std::bitset<tt::tt_metal::NUM_SEMAPHORES> used_semaphore_ids;
    for (const auto& semaphore : desc.semaphores) {
        if (semaphore.core_type == CoreType::WORKER && semaphore.core_ranges.intersects(mux_cores)) {
            used_semaphore_ids.set(semaphore.id);
        }
    }
    const uint32_t semaphore_id = lowest_free_id(used_semaphore_ids);
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
    return {.cb_index = cb_index, .l1_base_semaphore_id = semaphore_id};
}

FabricMuxProgramLocalL1 reserve_fabric_mux_program_local_l1(
    tt::tt_metal::Program& program, const tt::tt_metal::CoreRangeSet& mux_cores, uint32_t map_size_bytes) {
    TT_FATAL(!mux_cores.empty(), "Fabric mux program-local L1 placement needs at least one mux core");
    std::bitset<NUM_CIRCULAR_BUFFERS> used_cb_indices;
    for (const auto& mux_range : mux_cores.ranges()) {
        for (const auto& cb : program.impl().circular_buffers_on_corerange(mux_range)) {
            for (const uint8_t buffer_index : cb->buffer_indices()) {
                used_cb_indices.set(buffer_index);
            }
        }
    }
    const uint8_t cb_index = lowest_free_cb_index(used_cb_indices, mux_cores);

    // CreateSemaphore picks an id that is free on every core in mux_cores.
    const uint32_t semaphore_id = tt::tt_metal::CreateSemaphore(program, mux_cores, 0);
    tt::tt_metal::CreateCircularBuffer(
        program,
        mux_cores,
        tt::tt_metal::CircularBufferConfig(map_size_bytes, {{cb_index, tt::DataFormat::RawUInt32}})
            .set_page_size(cb_index, map_size_bytes));
    return {.cb_index = cb_index, .l1_base_semaphore_id = semaphore_id};
}

}  // namespace tt::tt_fabric::detail
