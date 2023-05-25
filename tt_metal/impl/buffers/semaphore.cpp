#include "tt_metal/impl/buffers/semaphore.hpp"

namespace tt {

namespace tt_metal {

Semaphore::Semaphore(const Semaphore &other) : device_(other.device_), core_range_set_(other.core_range_set_), address_(other.address_), initial_value_(other.initial_value_) {}

Semaphore &Semaphore::operator=(const Semaphore &other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->core_range_set_ = other.core_range_set_;
        this->address_ = other.address_;
        this->initial_value_ = other.initial_value_;
    }
    return *this;
}

Semaphore::Semaphore(Semaphore &&other) : device_(other.device_), core_range_set_(other.core_range_set_), address_(other.address_), initial_value_(other.initial_value_) {}

Semaphore &Semaphore::operator=(Semaphore &&other) {
    if (this != &other) {
        this->device_ = other.device_;
        this->core_range_set_ = other.core_range_set_;
        this->address_ = other.address_;
        this->initial_value_ = other.initial_value_;
    }
    return *this;
}

bool Semaphore::initialized_on_logical_core(const CoreCoord &logical_core) const {
    return this->core_range_set_.core_coord_in_core_ranges(logical_core);
}

}  // namespace tt_metal

}  // namespace tt
