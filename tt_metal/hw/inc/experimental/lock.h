// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace experimental {

/**
 * @brief RAII style wrapper for a scoped lock
 *
 * @tparam ReleaseFunc The function to call when this instance goes out of scope.
 */
template <typename ReleaseFunc>
class Lock {
public:
    inline __attribute__((always_inline)) Lock(ReleaseFunc release_func) : release_func_(release_func) {}
    inline __attribute__((always_inline)) ~Lock() { release_func_(); }

    Lock(const Lock&) = delete;
    Lock(Lock&&) = delete;
    Lock& operator=(const Lock&) = delete;
    Lock& operator=(Lock&&) = delete;

private:
    ReleaseFunc release_func_;
};

}  // namespace experimental
