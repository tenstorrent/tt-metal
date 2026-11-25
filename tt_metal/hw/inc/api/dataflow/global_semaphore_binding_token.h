// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

class GlobalSemaphore;

// A codegen-emitted handle for a Metal 2.0 kernel's GlobalSemaphore binding. The user never constructs
// this type; they construct a GlobalSemaphore from the generated token:
//   GlobalSemaphore ready(global_sem::ready);
class GlobalSemaphoreBindingToken {
public:
    explicit constexpr GlobalSemaphoreBindingToken(uint32_t crta_offset) noexcept : crta_offset_(crta_offset) {}

private:
    friend class GlobalSemaphore;

    uint32_t crta_offset_;  // Index of the common runtime arg.
};
