// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llrt/hal.hpp"
#include "common/core_coord.hpp"
#include "tt_metal/third_party/umd/device/tt_soc_descriptor.h"

namespace tt {

namespace tt_metal {

constexpr std::uint32_t NUM_SEMAPHORES = 8;

class Semaphore {
   public:
    Semaphore(const CoreRangeSet &core_range_set, uint32_t id, uint32_t initial_value);

    Semaphore(const CoreRangeSet &core_range_set, uint32_t id, uint32_t initial_value, CoreType core_type);

    Semaphore(const Semaphore &other);

    Semaphore& operator=(const Semaphore &other);

    Semaphore(Semaphore &&other);

    Semaphore& operator=(Semaphore &&other);

    uint32_t id() const { return id_; }

    uint32_t offset() const;

    CoreRangeSet core_range_set() const { return core_range_set_; }

    CoreType core_type() const { return core_type_; }

    uint32_t initial_value() const { return initial_value_; }

    bool initialized_on_logical_core(const CoreCoord &logical_core) const;

   private:
    CoreRangeSet core_range_set_;             // Ranges of cores where this semaphore is initialized
    uint32_t id_;
    uint32_t initial_value_;              // Initial value of semaphore
    CoreType core_type_;
};

}  // namespace tt_metal

}  // namespace tt
