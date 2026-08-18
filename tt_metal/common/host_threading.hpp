// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <thread>

namespace tt::tt_metal::detail {

inline size_t hardware_concurrency_or_one() {
    return std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1;
}

inline size_t get_host_worker_threads() {
    const char* value = std::getenv("TT_METAL_HOST_WORKER_THREADS");
    if (value == nullptr || value[0] == '\0') {
        return hardware_concurrency_or_one();
    }

    // Reject anything not starting with a decimal digit. std::strtoul silently
    // accepts a leading '-' (negating modulo ULONG_MAX+1 with no ERANGE), a
    // leading '+', and leading whitespace, so "-1" would otherwise parse to a
    // huge value and drive an attempt to spawn ~2^64 worker threads.
    if (value[0] < '0' || value[0] > '9') {
        return hardware_concurrency_or_one();
    }

    char* end = nullptr;
    errno = 0;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
        return hardware_concurrency_or_one();
    }

    // Clamp to a sane upper bound. strtoul accepts any digit-only string up to
    // ULONG_MAX without ERANGE (ERANGE only fires for values *exceeding* the
    // range), so a typo like "10000000000" or an explicit ULONG_MAX would
    // otherwise be returned verbatim and drive an attempt to spawn that many
    // worker threads, exhausting the process. Oversubscribing host worker
    // threads beyond the detected hardware concurrency yields no benefit, so
    // that is the natural cap.
    return std::min(static_cast<size_t>(parsed), hardware_concurrency_or_one());
}

}  // namespace tt::tt_metal::detail
