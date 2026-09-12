// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <thread>

#include "tt_metal/common/host_threading.hpp"

namespace {

using tt::tt_metal::detail::get_host_worker_threads;
using tt::tt_metal::detail::hardware_concurrency_or_one;
using tt::tt_metal::detail::parse_host_worker_threads;

constexpr size_t kHardwareConcurrency = 8;

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
    EXPECT_EQ(parse_host_worker_threads("-1", kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, RejectsNegativeFive) {
    EXPECT_EQ(parse_host_worker_threads("-5", kHardwareConcurrency), kHardwareConcurrency);
}

// Leading whitespace is silently skipped by strtoul; the guard must reject it.
TEST(HostThreading, RejectsLeadingWhitespace) {
    EXPECT_EQ(parse_host_worker_threads("  4", kHardwareConcurrency), kHardwareConcurrency);
}

// A leading '+' is accepted by strtoul but should be rejected here.
TEST(HostThreading, RejectsLeadingPlus) {
    EXPECT_EQ(parse_host_worker_threads("+4", kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, RejectsNonNumeric) {
    EXPECT_EQ(parse_host_worker_threads("abc", kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, RejectsEmpty) {
    EXPECT_EQ(parse_host_worker_threads("", kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, RejectsZero) {
    EXPECT_EQ(parse_host_worker_threads("0", kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, RejectsTrailingGarbage) {
    EXPECT_EQ(parse_host_worker_threads("3x", kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, UnsetFallsBackToHardwareConcurrency) {
    EXPECT_EQ(parse_host_worker_threads(nullptr, kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, AcceptsValidPositive) {
    EXPECT_EQ(parse_host_worker_threads("3", kHardwareConcurrency), size_t{3});
}

// ULONG_MAX is digit-only and in-range for strtoul (no ERANGE), so it clears
// the digit/parse guards. It must be clamped to the hardware-concurrency cap
// rather than returned verbatim, which would drive spawning ~2^64 threads.
TEST(HostThreading, ClampsUlongMax) {
    EXPECT_EQ(parse_host_worker_threads("18446744073709551615", kHardwareConcurrency), kHardwareConcurrency);
}

// A plausible typo (10 billion) is well under ULONG_MAX, so it also clears the
// parse guards and must be clamped rather than returned verbatim.
TEST(HostThreading, ClampsPlausibleTypo) {
    EXPECT_EQ(parse_host_worker_threads("10000000000", kHardwareConcurrency), kHardwareConcurrency);
}

TEST(HostThreading, ZeroHardwareConcurrencyStillUsesOneWorker) {
    EXPECT_EQ(parse_host_worker_threads(nullptr, 0), size_t{1});
    EXPECT_EQ(parse_host_worker_threads("4", 0), size_t{1});
}

TEST(HostThreading, WorkerCountIsThreadSafeAndStableAfterEnvironmentChanges) {
    const size_t hardware_concurrency = hardware_concurrency_or_one();
    const size_t expected_worker_threads =
        parse_host_worker_threads(std::getenv("TT_METAL_HOST_WORKER_THREADS"), hardware_concurrency);

    constexpr size_t caller_count = 8;
    std::array<size_t, caller_count> observed_worker_threads{};
    std::array<std::thread, caller_count> callers;
    for (size_t i = 0; i < callers.size(); ++i) {
        callers[i] =
            std::thread([i, &observed_worker_threads] { observed_worker_threads[i] = get_host_worker_threads(); });
    }
    for (auto& caller : callers) {
        caller.join();
    }

    const size_t initial_worker_threads = observed_worker_threads.front();
    EXPECT_EQ(initial_worker_threads, expected_worker_threads);
    for (size_t worker_threads : observed_worker_threads) {
        EXPECT_EQ(worker_threads, initial_worker_threads);
    }

    if (hardware_concurrency == 1) {
        GTEST_SKIP() << "The hardware cap makes all valid settings equivalent";
    }

    const char* different_worker_threads = initial_worker_threads == 1 ? "2" : "1";
    ScopedWorkerThreadsEnv env(different_worker_threads);
    ASSERT_NE(parse_host_worker_threads(different_worker_threads, hardware_concurrency), initial_worker_threads);
    EXPECT_EQ(get_host_worker_threads(), initial_worker_threads);
}
