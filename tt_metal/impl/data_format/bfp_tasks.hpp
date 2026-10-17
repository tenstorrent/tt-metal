// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <future>
#include <vector>

namespace tt::tt_metal::detail {

// Packing tasks borrow the caller's source and output buffers. Pool futures do
// not join on destruction, so drain all submitted work before propagating an
// error from submission or execution. Allocate every future slot before launch.
template <typename Submit>
void run_bfp_tasks(size_t count, const Submit& submit) {
    std::vector<std::shared_future<void>> pending(count);
    try {
        for (size_t i = 0; i < count; ++i) {
            pending[i] = submit(i);
        }
        for (auto& future : pending) {
            future.get();
        }
    } catch (...) {
        for (auto& future : pending) {
            if (future.valid()) {
                future.wait();
            }
        }
        throw;
    }
}

}  // namespace tt::tt_metal::detail
