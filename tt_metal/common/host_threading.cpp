// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_threading.hpp"

namespace tt::tt_metal::detail {

size_t get_host_worker_threads() {
    // Host concurrency is a process startup setting. Keeping the cache in one
    // compiled definition ensures that tt-metal and its consumers share the
    // same value. C++ guarantees thread-safe initialization of this static.
    static const size_t host_worker_threads =
        parse_host_worker_threads(std::getenv("TT_METAL_HOST_WORKER_THREADS"), hardware_concurrency_or_one());
    return host_worker_threads;
}

}  // namespace tt::tt_metal::detail
