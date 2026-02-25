// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>

namespace tt::tt_fabric {

/**
 * @brief Exception thrown when ControlPlane initialization fails due to transient conditions.
 *
 * This exception indicates that the fabric control plane could not be initialized due to
 * incomplete or invalid fabric node to physical chip mapping. Common causes include:
 *
 * - Hardware not yet ready (ASICs still initializing)
 * - Network links not fully established
 * - Incomplete mesh graph descriptor for the current system state
 * - Transient race conditions during multi-process initialization
 *
 * **Retry Semantics:**
 * This exception is designed to be retryable. Callers should catch this exception and
 * implement an appropriate retry strategy with exponential backoff. The MetalContext
 * automatically retries ControlPlane initialization up to 5 times with exponential
 * backoff (1s, 2s, 4s, 8s, 16s) before giving up.
 *
 * **Example Usage:**
 * @code
 * constexpr int kMaxRetries = 5;
 * for (int attempt = 1; attempt <= kMaxRetries; ++attempt) {
 *     try {
 *         control_plane = std::make_unique<ControlPlane>(...);
 *         break;  // Success
 *     } catch (const ControlPlaneInitFailure& e) {
 *         if (attempt == kMaxRetries) {
 *             throw;  // Final attempt failed
 *         }
 *         std::this_thread::sleep_for(std::chrono::seconds(1 << (attempt - 1)));
 *     }
 * }
 * @endcode
 *
 * **Non-Retryable Failures:**
 * If initialization fails due to a non-transient error (e.g., invalid configuration,
 * missing mesh graph descriptor file), other exception types will be thrown instead.
 *
 * @see tt::tt_metal::MetalContext::initialize_control_plane_impl() for the retry implementation
 */
class ControlPlaneInitFailure : public std::runtime_error {
public:
    explicit ControlPlaneInitFailure(const std::string& what_arg) : std::runtime_error(what_arg) {}
    explicit ControlPlaneInitFailure(const char* what_arg) : std::runtime_error(what_arg) {}
};

}  // namespace tt::tt_fabric
