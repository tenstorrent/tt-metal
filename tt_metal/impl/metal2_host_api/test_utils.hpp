// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal test utilities for metal2_host_api.
// This is intended for use by unit tests only.

#include <optional>
#include <umd/device/types/arch.hpp>  // tt::ARCH

namespace tt::tt_metal::experimental::metal2_host_api {

// ============================================================================
// Architecture Override for Testing
// ============================================================================
//
// These utilities allow tests to override the detected architecture without
// requiring actual hardware. The override is thread-local for parallel test safety.
//
// Example usage:
//   TEST(MyTest, WorksOnQuasar) {
//       ArchOverrideGuard guard(tt::ARCH::QUASAR);
//       // All code in this scope will see QUASAR as the architecture
//       auto program = MakeProgramFromSpec(spec);
//   }

// Thread-local architecture override storage.
// Prefer using ArchOverrideGuard rather than calling this directly.
std::optional<tt::ARCH>& arch_override();

// RAII guard for setting architecture override in tests.
// Sets the override on construction, clears it on destruction.
class ArchOverrideGuard {
public:
    explicit ArchOverrideGuard(tt::ARCH arch);
    ~ArchOverrideGuard();

    // Non-copyable, non-movable
    ArchOverrideGuard(const ArchOverrideGuard&) = delete;
    ArchOverrideGuard& operator=(const ArchOverrideGuard&) = delete;
    ArchOverrideGuard(ArchOverrideGuard&&) = delete;
    ArchOverrideGuard& operator=(ArchOverrideGuard&&) = delete;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
