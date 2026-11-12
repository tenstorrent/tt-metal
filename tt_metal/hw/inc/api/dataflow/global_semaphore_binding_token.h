// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

class GlobalSemaphore;

// Names the common runtime arg that holds a GlobalSemaphore's address. Because the address is a runtime
// arg, switching to a different semaphore does not recompile the kernel.
//
// With Metal 2.0, bind the semaphore in the KernelSpec and use the generated token:
//   GlobalSemaphore ready(global_sem::ready);
// Otherwise, write the semaphore's address() into a common runtime arg and name its index:
//   GlobalSemaphore ready(GlobalSemaphoreBindingToken{0});
class GlobalSemaphoreBindingToken {
public:
    explicit constexpr GlobalSemaphoreBindingToken(uint32_t crta_offset) noexcept : crta_offset_(crta_offset) {}

private:
    friend class GlobalSemaphore;

    uint32_t crta_offset_;  // Index of the common runtime arg.
};
