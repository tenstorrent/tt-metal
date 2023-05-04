#pragma once

#include "common/tt_backend_api_types.hpp"
#include "common/tt_xy_pair.h"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt {

namespace tt_metal {

// Semaphores are statically allocated withing range [SEMAPHORE_BASE, SEMAPHORE_BASE + SEMAPHORE_SIZE]
class Semaphore : public Buffer {
   public:
    Semaphore(
        Device *device,
        const tt_xy_pair &logical_core,
        uint32_t address,
        uint32_t initial_value) : logical_core_(logical_core), initial_value_(initial_value), Buffer(device, SEMAPHORE_SIZE / NUM_SEMAPHORES, address, false) {}

    ~Semaphore() {}

    Buffer *clone() { return new Semaphore(this->device_, this->logical_core_, this->address_, this->initial_value_); }

    tt_xy_pair logical_core() const { return logical_core_; }

    uint32_t initial_value() const { return initial_value_; }

    tt_xy_pair noc_coordinates() const { return this->device_->worker_core_from_logical_core(this->logical_core_); }

   private:
    void free() {}

    tt_xy_pair logical_core_;             // Logical core
    uint32_t initial_value_;              // Initial value of semaphore
};




}  // namespace tt_metal

}  // namespace tt
