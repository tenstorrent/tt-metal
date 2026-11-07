// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

class GlobalSemaphore;

// Names the common runtime arg that holds a GlobalSemaphore's address. Because the address is a runtime
// arg, switching to a different semaphore does not recompile the kernel.
//
// Example:
//   // Host code writes the semaphore's address() into common runtime arg word 0.
//   constexpr GlobalSemaphoreBindingToken ready_token{0};
//   GlobalSemaphore ready(ready_token);
class GlobalSemaphoreBindingToken {
public:
    explicit constexpr GlobalSemaphoreBindingToken(uint32_t crta_offset) noexcept : crta_offset_(crta_offset) {}

private:
    friend class GlobalSemaphore;

    uint32_t crta_offset_;  // Index of the common runtime arg.
};
