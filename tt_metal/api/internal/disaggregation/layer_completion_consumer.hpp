// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Test / scheduler-stand-in consumer for the layer-completion aggregation feature. Test-only: it is
// built into the standalone `_layer_completion` nanobind extension (NOT the ttnn module) and is NOT
// production router infra. Drains the master router's scheduler counter
// channel on a NATIVE C++ thread so it is immune to the Python GIL: a Python drain thread on the
// master rank gets starved the moment that rank's main thread blocks in a GIL-holding call (e.g. the
// request loop waiting on H2D), which made the old Python shim stall at a partial count while the
// router had already injected every completion. This is also how a real scheduler would consume the
// channel. Self-terminates once `expected` completions are drained and logs a PASS line — it does
// NOT depend on any Python teardown running.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace tt::tt_metal::distributed {
class InterProcessCounterChannel;
}

namespace tt::tests::prefill_test {

class LayerCompletionConsumer {
public:
    LayerCompletionConsumer(
        const std::string& channel_shm_name,
        uint64_t expected,
        uint32_t connect_timeout_ms = 30'000,
        uint64_t log_step = 61);
    ~LayerCompletionConsumer();

    LayerCompletionConsumer(const LayerCompletionConsumer&) = delete;
    LayerCompletionConsumer& operator=(const LayerCompletionConsumer&) = delete;

    void stop();  // idempotent: signal + join + final drain + shutdown channel
    uint64_t total() const noexcept { return total_.load(std::memory_order_relaxed); }
    bool reached_expected() const noexcept { return total_.load(std::memory_order_relaxed) >= expected_; }

private:
    void run();

    std::unique_ptr<tt::tt_metal::distributed::InterProcessCounterChannel> channel_;
    uint64_t expected_;
    uint64_t log_step_;
    std::atomic<uint64_t> total_{0};
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> stopped_{false};
};

}  // namespace tt::tests::prefill_test
