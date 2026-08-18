// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdlib>
#include <string>

#include "tt_metal/common/host_threading.hpp"

namespace {

using tt::tt_metal::detail::get_host_worker_threads;
using tt::tt_metal::detail::hardware_concurrency_or_one;

// RAII helper that sets TT_METAL_HOST_WORKER_THREADS for the duration of a test
// case and restores the previous value (or unsets it) on destruction, so tests
// don't leak environment state into one another.
class ScopedWorkerThreadsEnv {
public:
    explicit ScopedWorkerThreadsEnv(const char* value) {
        const char* previous = std::getenv(kName);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value != nullptr) {
            setenv(kName, value, /*overwrite=*/1);
        } else {
            unsetenv(kName);
        }
    }

    ~ScopedWorkerThreadsEnv() {
        if (had_previous_) {
            setenv(kName, previous_.c_str(), /*overwrite=*/1);
        } else {
            unsetenv(kName);
        }
    }

    ScopedWorkerThreadsEnv(const ScopedWorkerThreadsEnv&) = delete;
    ScopedWorkerThreadsEnv& operator=(const ScopedWorkerThreadsEnv&) = delete;

private:
    static constexpr const char* kName = "TT_METAL_HOST_WORKER_THREADS";
    bool had_previous_ = false;
    std::string previous_;
};

}  // namespace

// A negative value must not be accepted. std::strtoul would parse "-1" to
// ULONG_MAX without error; the guard must reject it and fall back.
TEST(HostThreading, RejectsNegativeOne) {
    ScopedWorkerThreadsEnv env("-1");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

TEST(HostThreading, RejectsNegativeFive) {
    ScopedWorkerThreadsEnv env("-5");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

// Leading whitespace is silently skipped by strtoul; the guard must reject it.
TEST(HostThreading, RejectsLeadingWhitespace) {
    ScopedWorkerThreadsEnv env("  4");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

// A leading '+' is accepted by strtoul but should be rejected here.
TEST(HostThreading, RejectsLeadingPlus) {
    ScopedWorkerThreadsEnv env("+4");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

TEST(HostThreading, RejectsNonNumeric) {
    ScopedWorkerThreadsEnv env("abc");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

TEST(HostThreading, RejectsEmpty) {
    ScopedWorkerThreadsEnv env("");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

TEST(HostThreading, RejectsZero) {
    ScopedWorkerThreadsEnv env("0");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

TEST(HostThreading, RejectsTrailingGarbage) {
    ScopedWorkerThreadsEnv env("3x");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

TEST(HostThreading, UnsetFallsBackToHardwareConcurrency) {
    ScopedWorkerThreadsEnv env(nullptr);
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

// A valid positive value must still parse exactly.
TEST(HostThreading, AcceptsValidPositive) {
    ScopedWorkerThreadsEnv env("3");
    EXPECT_EQ(get_host_worker_threads(), static_cast<size_t>(3));
}

// ULONG_MAX is digit-only and in-range for strtoul (no ERANGE), so it clears
// the digit/parse guards. It must be clamped to the hardware-concurrency cap
// rather than returned verbatim, which would drive spawning ~2^64 threads.
TEST(HostThreading, ClampsUlongMax) {
    ScopedWorkerThreadsEnv env("18446744073709551615");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}

// A plausible typo (10 billion) is well under ULONG_MAX, so it also clears the
// parse guards and must be clamped rather than returned verbatim.
TEST(HostThreading, ClampsPlausibleTypo) {
    ScopedWorkerThreadsEnv env("10000000000");
    EXPECT_EQ(get_host_worker_threads(), hardware_concurrency_or_one());
}
