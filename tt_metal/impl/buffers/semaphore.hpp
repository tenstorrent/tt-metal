// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {

// NOTE: the DM_LOCAL_CACHED pool must have a slot for every possible id (its slot is
// MEM_DM_CACHED_SEM_BASE + id * L1_ALIGNMENT). That invariant is enforced device-side, in
// tt_metal/hw/inc/api/dataflow/noc_semaphore.h, where both this bound and MEM_DM_CACHED_SEM_SIZE are
// visible -- a static_assert here would be silently vacuous, since this host header never sees the
// device memory map.
constexpr std::uint32_t NUM_SEMAPHORES = 16;

class Semaphore {
public:
    Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value);

    Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value, CoreType core_type);

    Semaphore(const Semaphore& other);

    Semaphore& operator=(const Semaphore& other);

    Semaphore(Semaphore&& other) noexcept;

    Semaphore& operator=(Semaphore&& other) noexcept;

    uint32_t id() const { return id_; }

    uint32_t offset() const;

    CoreRangeSet core_range_set() const { return core_range_set_; }

    CoreType core_type() const { return core_type_; }

    uint32_t initial_value() const { return initial_value_; }

    bool initialized_on_logical_core(const CoreCoord& logical_core) const;

private:
    CoreRangeSet core_range_set_;  // Ranges of cores where this semaphore is initialized
    uint32_t id_;
    uint32_t initial_value_;  // Initial value of semaphore
    CoreType core_type_;
};

class Program;

uint32_t CreateSemaphore(
    Program& program,
    const std::variant<CoreRange, CoreRangeSet>& core_spec,
    uint32_t initial_value,
    CoreType core_type);

}  // namespace tt::tt_metal
