#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/core_coord.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

// Semaphores are statically allocated withing range [SEMAPHORE_BASE, SEMAPHORE_BASE + SEMAPHORE_SIZE]
class Semaphore {
   public:
    Semaphore(
        Device *device,
        const CoreCoord &logical_core,
        uint32_t address,
        uint32_t initial_value) : device_(device), logical_core_(logical_core), address_(address), initial_value_(initial_value) {}

    Semaphore(const Semaphore &other) : device_(other.device_), logical_core_(other.logical_core_), address_(other.address_), initial_value_(other.initial_value_) {}

    Semaphore& operator=(const Semaphore &other) {
        if (this != &other) {
            this->device_ = other.device_;
            this->logical_core_ = other.logical_core_;
            this->address_ = other.address_;
            this->initial_value_ = other.initial_value_;
        }
        return *this;
    }

    Semaphore(Semaphore &&other) : device_(other.device_), logical_core_(other.logical_core_), address_(other.address_), initial_value_(other.initial_value_) {}

    Semaphore& operator=(Semaphore &&other) {
        if (this != &other) {
            this->device_ = other.device_;
            this->logical_core_ = other.logical_core_;
            this->address_ = other.address_;
            this->initial_value_ = other.initial_value_;
        }
        return *this;
    }

    constexpr uint32_t size() const { return SEMAPHORE_SIZE / NUM_SEMAPHORES; }

    Device *device() const { return device_; }

    uint32_t address() const { return address_; }

    CoreCoord logical_core() const { return logical_core_; }

    uint32_t initial_value() const { return initial_value_; }

    CoreCoord noc_coordinates() const { return this->device_->worker_core_from_logical_core(this->logical_core_); }

   private:
    Device *device_;
    CoreCoord logical_core_;             // Logical core
    uint32_t address_;
    uint32_t initial_value_;              // Initial value of semaphore
};




}  // namespace tt_metal

}  // namespace tt
