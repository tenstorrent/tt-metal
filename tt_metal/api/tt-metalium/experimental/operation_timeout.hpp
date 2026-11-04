// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {

/**
 * Get the operation timeout (0 = off), initialized from TT_METAL_OPERATION_TIMEOUT_SECONDS.
 *
 * A command queue wait that sees no dispatch progress for this long runs
 * TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE (if set) and throws.
 * The value is per MetalEnv; mesh_device only selects which one.
 *
 * Return value: std::chrono::duration<float>
 */
std::chrono::duration<float> GetOperationTimeout(const distributed::MeshDevice& mesh_device);

/**
 * Set the operation timeout (0 = off), overriding TT_METAL_OPERATION_TIMEOUT_SECONDS.
 *
 * The value is per MetalEnv; mesh_device only selects which one.
 * Each wait reads the value when it starts, so a wait already in progress keeps its old value.
 *
 * Return value: void
 */
void SetOperationTimeout(distributed::MeshDevice& mesh_device, std::chrono::duration<float> timeout);

}  // namespace tt::tt_metal::experimental
