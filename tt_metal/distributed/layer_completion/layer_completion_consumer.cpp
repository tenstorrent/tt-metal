// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <internal/disaggregation/layer_completion_consumer.hpp>

#include <chrono>
#include <cstdio>

#include <internal/service/inter_process_counter_channel.hpp>

namespace tt::tests::prefill_test {

using tt::tt_metal::distributed::InterProcessCounterChannel;

LayerCompletionConsumer::LayerCompletionConsumer(
    const std::string& channel_shm_name, uint64_t expected, uint32_t connect_timeout_ms, uint64_t log_step) :
    channel_(InterProcessCounterChannel::connect(channel_shm_name, connect_timeout_ms)),
    expected_(expected),
    log_step_(log_step ? log_step : 1) {
    thread_ = std::thread([this] { run(); });
}

LayerCompletionConsumer::~LayerCompletionConsumer() { stop(); }

void LayerCompletionConsumer::stop() {
    if (stopped_.exchange(true)) {
        return;
    }
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
    if (channel_) {
        total_.fetch_add(channel_->try_consume_all(), std::memory_order_relaxed);  // final drain
        channel_->shutdown();
    }
}

void LayerCompletionConsumer::run() {
    // Native thread: try_consume_all() is a plain SHM read — no Python objects touched — so this
    // drains regardless of what the rank's Python main thread is doing. Only this thread writes total_.
    uint64_t logged = 0;
    while (!stop_.load(std::memory_order_acquire)) {
        const uint32_t n = channel_->try_consume_all();
        // Only this thread writes total_, so a plain accumulate suffices (fetch_add(0) is a no-op).
        const uint64_t cur = total_.fetch_add(n, std::memory_order_relaxed) + n;
        if (cur - logged >= log_step_) {
            std::fprintf(
                stderr,
                "[completion-check] C++ consumer drained %llu/%llu completions\n",
                static_cast<unsigned long long>(cur),
                static_cast<unsigned long long>(expected_));
            logged = cur;
        }
        if (cur >= expected_) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const uint64_t fin = total_.load(std::memory_order_relaxed);
    if (fin >= expected_) {
        std::fprintf(
            stderr,
            "[completion-check] PASS: C++ consumer drained %llu == %llu (expected)\n",
            static_cast<unsigned long long>(fin),
            static_cast<unsigned long long>(expected_));
    }
}

}  // namespace tt::tests::prefill_test
