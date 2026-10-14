// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal::experimental {

struct RAIIResourceKind {
    constexpr RAIIResourceKind() = default;

    // Disable copy
    RAIIResourceKind(const RAIIResourceKind&) = delete;
    RAIIResourceKind& operator=(const RAIIResourceKind&) = delete;

    // Allow move
    constexpr RAIIResourceKind(RAIIResourceKind&&) = default;
    constexpr RAIIResourceKind& operator=(RAIIResourceKind&&) = default;
};

};  // namespace tt::tt_metal::experimental
