// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <unordered_set>
#include <vector>

#include <layer_completion_message.hpp>
#include <layer_completion_queue.hpp>

namespace tt::tt_metal::distributed::test {

// 4 producer threads each push a disjoint block of seqs; a single
// consumer drains until it has seen them all. Assert: no loss, no dup.
TEST(LayerCompletionQueueMpsc, FourProducersOneConsumerNoLossNoDup) {
    const std::string name = "/tt_lcq_test_mpsc";
    std::remove(("/dev/shm" + name).c_str());
    auto queue = LayerCompletionQueue::create(name);

    constexpr int kProducers = 4;
    constexpr uint64_t kPerProducer = 50'000;
    const uint64_t total = static_cast<uint64_t>(kProducers) * kPerProducer;

    std::atomic<bool> start{false};
    std::vector<std::thread> producers;
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p] {
            while (!start.load(std::memory_order_acquire)) {
            }
            for (uint64_t i = 0; i < kPerProducer; ++i) {
                LayerCompletionMessage m{};
                m.seq = static_cast<uint64_t>(p) * kPerProducer + i;
                while (!queue->try_push(m)) {
                    std::this_thread::yield();  // ring full → back off
                }
            }
        });
    }

    std::unordered_set<uint64_t> seen;
    seen.reserve(total);
    std::thread consumer([&] {
        LayerCompletionMessage out{};
        while (seen.size() < total) {
            if (queue->try_pop(out)) {
                auto [_, inserted] = seen.insert(out.seq);
                ASSERT_TRUE(inserted) << "duplicate seq " << out.seq;
            } else {
                std::this_thread::yield();
            }
        }
    });

    start.store(true, std::memory_order_release);
    for (auto& t : producers) {
        t.join();
    }
    consumer.join();

    EXPECT_EQ(seen.size(), total);  // no loss
    queue->shutdown();
}

}  // namespace tt::tt_metal::distributed::test
