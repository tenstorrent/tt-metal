// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <semaphore.hpp>
#include <stdint.h>

#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/types/core_coordinates.hpp>

namespace tt {

namespace tt_metal {

Semaphore::Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value) :
    core_range_set_(core_range_set), id_(id), initial_value_(initial_value), core_type_(CoreType::WORKER) {}

Semaphore::Semaphore(const CoreRangeSet& core_range_set, uint32_t id, uint32_t initial_value, CoreType core_type) :
    core_range_set_(core_range_set), id_(id), initial_value_(initial_value), core_type_(core_type) {}

Semaphore::Semaphore(const Semaphore& other) = default;

Semaphore& Semaphore::operator=(const Semaphore& other) = default;

Semaphore::Semaphore(Semaphore&& other) noexcept = default;

Semaphore& Semaphore::operator=(Semaphore&& other) noexcept = default;

bool Semaphore::initialized_on_logical_core(const CoreCoord& logical_core) const {
    return this->core_range_set_.contains(logical_core);
}

uint32_t Semaphore::offset() const {
    uint32_t offset = MetalContext::instance().hal().get_alignment(HalMemType::L1) * id_;
    return offset;
}

}  // namespace tt_metal

}  // namespace tt
