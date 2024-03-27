// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

// Semaphores are statically allocated withing range [SEMAPHORE_BASE, SEMAPHORE_BASE + SEMAPHORE_SIZE]
class Semaphore {
   public:
    Semaphore(
        const CoreRangeSet &core_range_set,
        uint32_t address,
        uint32_t initial_value) : core_range_set_(core_range_set), address_(address), initial_value_(initial_value), core_type_(CoreType::WORKER) {}

    Semaphore(
        const CoreRangeSet &core_range_set,
        uint32_t address,
        uint32_t initial_value,
        CoreType core_type) : core_range_set_(core_range_set), address_(address), initial_value_(initial_value), core_type_(core_type) {}

    Semaphore(const Semaphore &other);

    Semaphore& operator=(const Semaphore &other);

    Semaphore(Semaphore &&other);

    Semaphore& operator=(Semaphore &&other);

    constexpr uint32_t size() const { return SEMAPHORE_SIZE / NUM_SEMAPHORES; }

    uint32_t id() const { return (address_ - SEMAPHORE_BASE) / L1_ALIGNMENT; }

    uint32_t address() const { return address_; }

    CoreRangeSet core_range_set() const { return core_range_set_; }

    CoreType core_type() const { return core_type_; }

    uint32_t initial_value() const { return initial_value_; }

    bool initialized_on_logical_core(const CoreCoord &logical_core) const;

   private:
    CoreRangeSet core_range_set_;             // Ranges of cores where this semaphore is initialized
    uint32_t address_;
    uint32_t initial_value_;              // Initial value of semaphore
    CoreType core_type_;                       // Type of core. ETH, WORKER.
};

}  // namespace tt_metal

}  // namespace tt
