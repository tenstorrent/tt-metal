// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental::MeshDevice {

/**
 * @brief Stores arbitrary user data on a MeshDevice.
 *
 * Associates a shared pointer to user-defined data with a key on the given MeshDevice.
 * If the key already exists, the previous value is replaced. This function is thread-safe.
 * Whatever data is pointed to cannot reference the MeshDevice in its destructor, as all MeshDevice shared pointers will
 * be invalid at that point.
 *
 * @param mesh_device Reference to the MeshDevice on which to store the data.
 * @param key Unique identifier for the user data (typically derived from a pointer or type hash).
 * @param value Shared pointer to the data to store. Ownership is shared with the MeshDevice.
 */
void SetUserData(MeshDevice& mesh_device, uintptr_t key, std::shared_ptr<void> value);

/**
 * @brief Retrieves user data previously stored on a MeshDevice.
 *
 * Looks up the user data associated with the given key. This function is thread-safe.
 *
 * @param mesh_device Reference to the MeshDevice from which to retrieve the data.
 * @param key Unique identifier for the user data.
 * @return Shared pointer to the stored data, or nullptr if the key does not exist.
 */
std::shared_ptr<void> GetUserData(MeshDevice& mesh_device, uintptr_t key);

/**
 * @brief Removes user data from a MeshDevice.
 *
 * Erases the user data associated with the given key. If the key does not exist,
 * this function has no effect. This function is thread-safe.
 *
 * @param mesh_device Reference to the MeshDevice from which to remove the data.
 * @param key Unique identifier for the user data to remove.
 */
void RemoveUserData(MeshDevice& mesh_device, uintptr_t key);

}  // namespace tt::tt_metal::experimental::MeshDevice
