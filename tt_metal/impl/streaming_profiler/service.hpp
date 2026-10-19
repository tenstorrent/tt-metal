// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt_stl/tt_pause.hpp>

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal::streaming_profiler {

class Receiver;
class SteadyClock;

void set_thread_name(const std::string& name);

// How a polling thread waits for work: `spins` empty polls, then sleeps growing to `cap_us`.
inline constexpr uint32_t kEmptyPollsBeforeSleep = 1000;
struct IdleBackoff {
    uint32_t cap_us;
    uint32_t spins;
    uint32_t empty_polls = 0;
    uint32_t sleep_us = 1;
    explicit IdleBackoff(uint32_t cap, uint32_t spins = kEmptyPollsBeforeSleep) : cap_us(cap), spins(spins) {}
    void idle() {
        if (++empty_polls < spins) {
            ttsl::pause();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            sleep_us = std::min(sleep_us + sleep_us / 4 + 1, cap_us);
        }
    }
    // The spin phase alone: true while the caller should poll again, false once it should park.
    bool spin() {
        if (++empty_polls < spins) {
            ttsl::pause();
            return true;
        }
        return false;
    }
    void reset() {
        empty_polls = 0;
        sleep_us = 1;
    }
};

using BatchCallback = std::function<void(const experimental::streaming_profiler::detail::BatchData&)>;

class Service {
public:
    Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;

    // Adds a consumer that receives the records of the types `kinds` through `callback`, and returns its id.
    experimental::streaming_profiler::detail::CallbackId add_consumer(
        std::string name, std::vector<experimental::streaming_profiler::detail::Kind> kinds, BatchCallback callback);
    // Removes the consumer. It returns once the callback can no longer run, or at once if the callback removes itself.
    void remove_consumer(experimental::streaming_profiler::detail::CallbackId id);

    // Attaches the receiver's streams to every consumer, and returns once each one reads them, so nothing published
    // afterwards is missed.
    void attach_receiver(Receiver& receiver);
    // Detaches the receiver from every consumer, and returns once each one has drained its streams and delivered the
    // batches it held back. The receiver must finish its clock map first, because those batches are placed through it.
    // Detaching the last receiver writes the file sinks.
    void detach_receiver(Receiver& receiver);
    bool is_active() const;

    // The Tracy sink and the CSV writers rtoptions select; subsequent calls do nothing.
    void register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions);
    SteadyClock& steady();

    // Wakes every consumer.
    void wake_consumers() {
        wake_gen_.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_consumers_.load(std::memory_order_seq_cst) != 0) {
            wake_gen_.notify_all();
        }
    }
    // The wake generation, which every wake_consumers() call advances. Take it before looking for work, so a wake that
    // lands before wait_wake() isn't missed.
    uint32_t wake_token() const { return wake_gen_.load(std::memory_order_acquire); }
    void wait_wake(uint32_t seen) const {
        sleeping_consumers_.fetch_add(1, std::memory_order_seq_cst);
        wake_gen_.wait(seen, std::memory_order_seq_cst);
        sleeping_consumers_.fetch_sub(1, std::memory_order_acq_rel);
    }

private:
    struct Consumer;
    class ConsumerLoop;
    void consumer_thread(Consumer& consumer);
    void post_control(Consumer& c, Receiver* receiver, bool attach);
    void wait_acks(std::unique_lock<std::mutex>& lk);

    // Serializes add/remove/attach/detach against each other; never taken by a consumer thread.
    std::mutex topology_mu_;
    // Guards the lists below and the ack handshake; taken briefly by consumer threads.
    mutable std::mutex mu_;
    std::condition_variable ack_cv_;
    uint64_t pending_acks_ = 0;
    alignas(64) std::atomic<uint32_t> wake_gen_{0};
    alignas(64) mutable std::atomic<uint32_t> sleeping_consumers_{0};
    std::vector<std::unique_ptr<Consumer>> consumers_;
    std::vector<Receiver*> receivers_;
    std::vector<std::function<void()>> file_sinks_;
    std::unique_ptr<SteadyClock> steady_;
    std::vector<experimental::streaming_profiler::Callback> builtin_callbacks_;
    uint32_t next_id_ = 1;
    std::once_flag builtins_once_;
};

// The process's Service. Subscriptions outlive every device and context, so it is never destroyed.
Service& service();

}  // namespace tt::tt_metal::streaming_profiler
