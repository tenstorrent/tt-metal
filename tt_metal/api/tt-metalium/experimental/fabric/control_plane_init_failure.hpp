// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>

namespace tt::tt_fabric {

// Thrown when ControlPlane initialization fails due to incomplete or invalid
// fabric node to physical chip mapping (e.g. hardware/networking not ready).
// Callers may retry initialization.
class ControlPlaneInitFailure : public std::runtime_error {
public:
    explicit ControlPlaneInitFailure(const std::string& what_arg) : std::runtime_error(what_arg) {}
    explicit ControlPlaneInitFailure(const char* what_arg) : std::runtime_error(what_arg) {}
};

}  // namespace tt::tt_fabric
