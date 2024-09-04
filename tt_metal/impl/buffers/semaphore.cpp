// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/semaphore.hpp"

namespace tt {

namespace tt_metal {

Semaphore::Semaphore(const CoreRangeSet &core_range_set, uint32_t id, uint32_t initial_value) :
    core_range_set_(core_range_set), id_(id), initial_value_(initial_value), core_type_(CoreType::WORKER) {}

Semaphore::Semaphore(const CoreRangeSet &core_range_set, uint32_t id, uint32_t initial_value, CoreType core_type) :
    core_range_set_(core_range_set), id_(id), initial_value_(initial_value), core_type_(core_type) {}

Semaphore::Semaphore(const Semaphore &other) :
    core_range_set_(other.core_range_set_),
    id_(other.id_),
    initial_value_(other.initial_value_),
    core_type_(other.core_type_) {}

Semaphore &Semaphore::operator=(const Semaphore &other) {
    if (this != &other) {
        this->core_range_set_ = other.core_range_set_;
        this->id_ = other.id_;
        this->initial_value_ = other.initial_value_;
        this->core_type_ = other.core_type_;
    }
    return *this;
}

Semaphore::Semaphore(Semaphore &&other) :
    core_range_set_(other.core_range_set_),
    id_(other.id_),
    initial_value_(other.initial_value_),
    core_type_(other.core_type_) {}

Semaphore &Semaphore::operator=(Semaphore &&other) {
    if (this != &other) {
        this->core_range_set_ = other.core_range_set_;
        this->id_ = other.id_;
        this->initial_value_ = other.initial_value_;
        this->core_type_ = other.core_type_;
    }
    return *this;
}

bool Semaphore::initialized_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}

uint32_t Semaphore::offset() const {
    uint32_t offset = L1_ALIGNMENT * id_;
    return offset;
}

}  // namespace tt_metal

}  // namespace tt
