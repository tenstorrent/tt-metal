#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

// Semaphores are statically allocated withing range [SEMAPHORE_BASE, SEMAPHORE_BASE + SEMAPHORE_SIZE]
class Semaphore {
   public:
    Semaphore(
        Device *device,
        const tt_xy_pair &logical_core,
        uint32_t address,
        uint32_t initial_value) : device_(device), logical_core_(logical_core), address_(address), initial_value_(initial_value) {}

    constexpr uint32_t size() const { return SEMAPHORE_SIZE / NUM_SEMAPHORES; }

    Device *device() const { return device_; }

    uint32_t address() const { return address_; }

    tt_xy_pair logical_core() const { return logical_core_; }

    uint32_t initial_value() const { return initial_value_; }

    tt_xy_pair noc_coordinates() const { return this->device_->worker_core_from_logical_core(this->logical_core_); }

   private:
    Device *device_;
    tt_xy_pair logical_core_;             // Logical core
    uint32_t address_;
    uint32_t initial_value_;              // Initial value of semaphore
};




}  // namespace tt_metal

}  // namespace tt
