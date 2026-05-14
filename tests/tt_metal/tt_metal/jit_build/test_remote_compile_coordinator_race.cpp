// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "impl/jit_server/remote_compile_coordinator.hpp"

namespace tt::tt_metal {

TEST(RemoteCompileCoordinatorRaceTest, SameHashRunsDescriptorFactoryOnce) {
    constexpr std::size_t kHash = 0x5eed;
    constexpr int kNumThreads = 100;

    std::atomic<int> ready_to_submit{0};
    std::atomic<int> factory_invocations{0};
    std::atomic<int> stopped_before_rpc{0};
    std::mutex mutex;
    std::condition_variable cv;

    std::vector<std::future<void>> workers;
    workers.reserve(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i) {
        workers.push_back(std::async(std::launch::async, [&] {
            RemoteCompileCoordinator coordinator(
                std::vector<std::string>{"unused-endpoint"}, /*device_build_id=*/0, /*build_key=*/0);

            {
                std::lock_guard lock(mutex);
                ++ready_to_submit;
            }
            cv.notify_all();

            try {
                coordinator.submit(kHash, [&] {
                    ++factory_invocations;

                    std::unique_lock lock(mutex);
                    cv.wait(lock, [&] { return ready_to_submit.load() == kNumThreads; });
                    lock.unlock();

                    // Keep the owner in-flight long enough for the other threads to observe the cache entry.
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    throw std::runtime_error("stop before RPC");
                    return KernelCompileDescriptor{};
                });
            } catch (const std::runtime_error& ex) {
                EXPECT_STREQ(ex.what(), "stop before RPC");
                ++stopped_before_rpc;
            }
        }));
    }

    for (auto& worker : workers) {
        worker.get();
    }

    EXPECT_EQ(factory_invocations.load(), 1);
    EXPECT_EQ(stopped_before_rpc.load(), 1);
}

TEST(RemoteCompileCoordinatorRaceTest, SameHashRunsDescriptorFactoryOnceInBrokerMode) {
    constexpr std::size_t kHash = 0x5eed;
    constexpr int kNumThreads = 100;

    std::atomic<int> ready_to_submit{0};
    std::atomic<int> factory_invocations{0};
    std::atomic<int> stopped_before_rpc{0};
    std::mutex mutex;
    std::condition_variable cv;

    std::vector<std::future<void>> workers;
    workers.reserve(kNumThreads);

    for (int i = 0; i < kNumThreads; ++i) {
        workers.push_back(std::async(std::launch::async, [&] {
            RemoteCompileCoordinator coordinator("unused-broker", /*device_build_id=*/0, /*build_key=*/0);

            {
                std::lock_guard lock(mutex);
                ++ready_to_submit;
            }
            cv.notify_all();

            try {
                coordinator.submit(kHash, [&] {
                    ++factory_invocations;

                    std::unique_lock lock(mutex);
                    cv.wait(lock, [&] { return ready_to_submit.load() == kNumThreads; });
                    lock.unlock();

                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    throw std::runtime_error("stop before RPC");
                    return KernelCompileDescriptor{};
                });
            } catch (const std::runtime_error& ex) {
                EXPECT_STREQ(ex.what(), "stop before RPC");
                ++stopped_before_rpc;
            }
        }));
    }

    for (auto& worker : workers) {
        worker.get();
    }

    EXPECT_EQ(factory_invocations.load(), 1);
    EXPECT_EQ(stopped_before_rpc.load(), 1);
}

TEST(RemoteCompileCoordinatorRaceTest, BrokerDispatchFailureCleansDedupCacheForQueuedItem) {
    constexpr std::size_t kHash = 0xBADC0DE;
    std::atomic<int> factory_invocations{0};

    RemoteCompileCoordinator first_coordinator("localhost:1", /*device_build_id=*/0, /*build_key=*/0);
    first_coordinator.submit(kHash, [&] {
        ++factory_invocations;
        KernelCompileDescriptor descriptor;
        descriptor.kernel_hash = kHash;
        descriptor.request.build_key = 0;
        descriptor.request.kernel_name = "kernel/broker-fail";
        return descriptor;
    });
    EXPECT_ANY_THROW(first_coordinator.finish());

    RemoteCompileCoordinator second_coordinator("localhost:1", /*device_build_id=*/0, /*build_key=*/0);
    EXPECT_THROW(
        second_coordinator.submit(
            kHash,
            [&] {
                ++factory_invocations;
                throw std::runtime_error("second submit should run descriptor");
                return KernelCompileDescriptor{};
            }),
        std::runtime_error);

    EXPECT_EQ(factory_invocations.load(), 2);
}

TEST(RemoteCompileCoordinatorRaceTest, StaticDispatchFailureCleansDedupCacheForQueuedItem) {
    constexpr std::size_t kHash = 0xC0FFEE;
    std::atomic<int> factory_invocations{0};

    RemoteCompileCoordinator first_coordinator(
        std::vector<std::string>{"localhost:1"}, /*device_build_id=*/0, /*build_key=*/0);
    first_coordinator.submit(kHash, [&] {
        ++factory_invocations;
        KernelCompileDescriptor descriptor;
        descriptor.kernel_hash = kHash;
        descriptor.request.build_key = 0;
        descriptor.request.kernel_name = "kernel/static-fail";
        return descriptor;
    });
    EXPECT_ANY_THROW(first_coordinator.finish());

    RemoteCompileCoordinator second_coordinator(
        std::vector<std::string>{"localhost:1"}, /*device_build_id=*/0, /*build_key=*/0);
    EXPECT_THROW(
        second_coordinator.submit(
            kHash,
            [&] {
                ++factory_invocations;
                throw std::runtime_error("second submit should run descriptor");
                return KernelCompileDescriptor{};
            }),
        std::runtime_error);

    EXPECT_EQ(factory_invocations.load(), 2);
}

}  // namespace tt::tt_metal
