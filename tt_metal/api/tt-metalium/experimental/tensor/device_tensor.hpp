// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/shape.hpp>

namespace tt::tt_metal /*::tensor*/ {

class IDevice;

/**
 * DeviceTensor is a device memory object. The user’s mental model of DeviceTensor is an owning handle to
 * device-allocated memory.
 *
 * DeviceTensor should have RAII semantics with unique ownership:
 * - Device memory resource lifetime == object lifetime
 *   - Device memory is allocated on construction, and released on destruction.
 *   - The programmer explicitly manages the device-allocated memory lifetime.
 *   - This can be tricky in an asynchronous runtime environment. For now, the onus is on the programmer to correctly
 *     manage DeviceTensor lifetime around queue synchronization events.
 * - Movable (RAII transfer of ownership)
 * - Non-copyable
 * - No equality/inequality operator. (If we did add this, equality would mean the same underlying allocation – no value
 *   semantics)
 *
 */
class DeviceTensor {
public:
    // Special Member functions

    /**
     * Construct a tensor that does not own any device memory.
     */
    DeviceTensor() = default;
    /**
     * Deallocates any owning device memory.
     */
    ~DeviceTensor() = default;

    // TODO: what if clients actually want to copy?
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    // Transfers ownership of other's memory
    DeviceTensor(DeviceTensor&& other) = default;
    DeviceTensor& operator=(DeviceTensor&& other) = default;

    // End speical member functions

    /**
     * Deallocate and release owned device memory.
     */
    void deallocate();

    // reshape transformation, mutating version
    // TODO: figure out what we will be doing for reshape
    void reshape(/* */);

    // Getters

    IDevice* get_device() const;

    // TODO(River): understand what is sharding better
    bool is_sharded() const;
    // TODO: what is the return type here?
    Shape element_size() const;
};

}  // namespace tt::tt_metal
