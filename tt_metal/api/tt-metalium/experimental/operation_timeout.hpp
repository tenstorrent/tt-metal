// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental {

/**
 * Get the operation timeout in seconds (0 = off), initialized from TT_METAL_OPERATION_TIMEOUT_SECONDS.
 *
 * A command queue wait that sees no dispatch progress for this long runs
 * TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE (if set) and throws.
 *
 * Return value: float
 */
float GetOperationTimeoutSeconds();

/**
 * Set the operation timeout in seconds (0 = off), overriding TT_METAL_OPERATION_TIMEOUT_SECONDS.
 *
 * Each wait reads the value when it starts, so a wait already in progress keeps its old value.
 *
 * Return value: void
 */
void SetOperationTimeoutSeconds(float seconds);

}  // namespace tt::tt_metal::experimental
