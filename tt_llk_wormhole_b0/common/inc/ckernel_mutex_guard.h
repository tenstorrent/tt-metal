// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace ckernel
{

class [[nodiscard]] T6MutexLockGuard final
{
public:
    explicit T6MutexLockGuard(const uint8_t index) noexcept : mutex_index(index)
    {
        t6_mutex_acquire(mutex_index);
    }

    ~T6MutexLockGuard()
    {
        t6_mutex_release(mutex_index);
    }

    // Non-copyable
    T6MutexLockGuard(const T6MutexLockGuard&)            = delete;
    T6MutexLockGuard& operator=(const T6MutexLockGuard&) = delete;

    // Non-movable
    T6MutexLockGuard(T6MutexLockGuard&&)            = delete;
    T6MutexLockGuard& operator=(T6MutexLockGuard&&) = delete;

private:
    const uint8_t mutex_index;
};

} // namespace ckernel
