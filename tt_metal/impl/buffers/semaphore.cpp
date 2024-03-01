// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/semaphore.hpp"

namespace tt {

namespace tt_metal {

Semaphore::Semaphore(const Semaphore &other) : core_range_set_(other.core_range_set_), address_(other.address_), initial_value_(other.initial_value_), core_type_(other.core_type_) {}

Semaphore &Semaphore::operator=(const Semaphore &other) {
    if (this != &other) {
        this->core_range_set_ = other.core_range_set_;
        this->address_ = other.address_;
        this->initial_value_ = other.initial_value_;
        this->core_type_ = other.core_type_;
    }
    return *this;
}

Semaphore::Semaphore(Semaphore &&other) : core_range_set_(other.core_range_set_), address_(other.address_), initial_value_(other.initial_value_), core_type_(other.core_type_) {}

Semaphore &Semaphore::operator=(Semaphore &&other) {
    if (this != &other) {
        this->core_range_set_ = other.core_range_set_;
        this->address_ = other.address_;
        this->initial_value_ = other.initial_value_;
        this->core_type_ = other.core_type_;
    }
    return *this;
}

bool Semaphore::initialized_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}

}  // namespace tt_metal

}  // namespace tt
