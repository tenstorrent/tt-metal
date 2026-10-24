// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

class GlobalSemaphore;

// Handle a kernel constructs a GlobalSemaphore from. It holds the index of the common runtime
// arg word that carries the semaphore's L1 address, so the address is supplied at run time and a
// different semaphore never needs a recompile.
//
// Usage example:
//   // Host code writes the semaphore's address() into common runtime arg word 0.
//   constexpr GlobalSemaphoreBindingToken ready_token{0};
//   GlobalSemaphore ready(ready_token);
class GlobalSemaphoreBindingToken {
public:
    explicit constexpr GlobalSemaphoreBindingToken(uint32_t crta_offset) noexcept : crta_offset_(crta_offset) {}

private:
    friend class GlobalSemaphore;

    uint32_t crta_offset_;  // word index of the address slot in the CRTA buffer
};
