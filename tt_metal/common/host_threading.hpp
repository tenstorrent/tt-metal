// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
    return static_cast<size_t>(parsed);
}

}  // namespace tt::tt_metal::detail
