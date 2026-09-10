// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The streaming profiler's consumer side, one per process. Each capture's receiver attaches as a producer; each
// subscriber is a consumer on its own thread with a reader on every producer's frame queue, decoding the frames it
// finds into records for its callback. A consumer that falls behind loses its own oldest frames and nothing else
// backs up. Producers come and go with MeshDevices; consumers persist until removed.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <tt_stl/tt_pause.hpp>

#include "impl/streaming_profiler/streaming_profiler_consumer.hpp"

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal::streaming_profiler {

class TracySink;

void set_os_thread_name(const std::string& name);

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

// The device's 64-bit stream position from its 32-bit bytes_sent word and a position the ingest thread has already
// observed: the device is never more than one FIFO (2 GB at most) past what it has been credited, so the difference
// fits 32 bits.
inline uint64_t widen_head(uint64_t observed, uint32_t bytes_sent) {
    return observed + static_cast<uint32_t>(bytes_sent - static_cast<uint32_t>(observed));
}

// Everything the device has landed lies below head + SPSC_NOTIFY_CAP_BYTES (the relay writes bytes_sent before it
// pushes more than that), and a frame at `offset` is overwritten only by writes from offset + fifo_bytes on.
inline bool frame_intact(uint64_t offset, uint64_t fifo_bytes, uint64_t head) {
    return head + kernel_profiler::SPSC_NOTIFY_CAP_BYTES <= offset + fifo_bytes;
}

// The ingest records the first frame boundary in every kMarkBytes of a stream, so a consumer the device has lapped
// resumes at the oldest boundary past the overwrite horizon instead of at the ingest's position.
inline constexpr uint64_t kMarkBytes = 64 * 1024;
static_assert(kMarkBytes >= profiler::kSpscMaxFrameWords * 4, "every block holds at least one frame boundary");

// The first recorded boundary in [from, end], or end.
inline uint64_t resume_mark(std::span<const std::atomic<uint64_t>> marks, uint64_t from, uint64_t end) {
    for (uint64_t b = from / kMarkBytes; !marks.empty() && b * kMarkBytes < end; b++) {
        const uint64_t v = marks[b % marks.size()].load(std::memory_order_acquire);
        if (v >= from && v <= end) {
            return v;
        }
    }
    return end;
}

// One pass of a consumer's walk over a stream.
struct Walked {
    uint32_t frames = 0;   // copied into `out`, back to back
    size_t bytes = 0;      // their total length
    uint64_t dropped = 0;  // bytes the consumer skipped because the device had reached them
    uint64_t cursor = 0;   // where the next pass starts
};

// Copies whole frames from `cursor` up to `end` (the ingest's walk position, always a frame boundary) into `out`, at
// most frame_words.size() of them, recording each one's length in words. The FIFO (a power-of-two size) is read in
// place while the device keeps writing it, so a frame the device has reached is never trusted: a cursor the device
// has passed resumes at the first recorded boundary an eighth of the FIFO past the overwrite horizon (the device keeps
// writing during the copy) and counts what it skipped; if the device reached the pass's first frame during the copy
// the lengths read from those pages are void, so the pass is discarded and the next one resumes further on.
template <typename LiveHead>
Walked walk_frames(
    std::span<const std::byte> fifo,
    uint64_t cursor,
    uint64_t end,
    std::span<const std::atomic<uint64_t>> marks,
    std::span<std::byte> out,
    std::span<uint32_t> frame_words,
    LiveHead live_head) {
    Walked w{.cursor = cursor};
    if (cursor >= end) {
        return w;
    }
    uint64_t start = cursor;
    if (!frame_intact(start, fifo.size(), live_head())) {
        const uint64_t written = live_head() + kernel_profiler::SPSC_NOTIFY_CAP_BYTES;
        const uint64_t horizon = written > fifo.size() ? written - fifo.size() : 0;
        start = resume_mark(marks, horizon + fifo.size() / 8, end);
        w.dropped = start - cursor;
        w.cursor = start;
        if (start >= end) {
            return w;
        }
    }
    const size_t mask = fifo.size() - 1;
    while (w.cursor < end && w.frames < frame_words.size()) {
        uint32_t w1;
        std::memcpy(&w1, fifo.data() + ((w.cursor + 4) & mask), 4);
        const uint32_t fw = kernel_profiler::spsc_span_frame_words(w1);
        const size_t bytes = size_t{fw} * 4;
        if (w1 > profiler::kSpscMaxPayloadWords || bytes > end - w.cursor || w.bytes + bytes > out.size()) {
            break;
        }
        const size_t at = w.cursor & mask;
        const size_t first = std::min(bytes, fifo.size() - at);
        std::memcpy(out.data() + w.bytes, fifo.data() + at, first);
        std::memcpy(out.data() + w.bytes + first, fifo.data(), bytes - first);
        frame_words[w.frames++] = fw;
        w.bytes += bytes;
        w.cursor += bytes;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (!frame_intact(start, fifo.size(), live_head())) {
        w = Walked{.dropped = start - cursor, .cursor = start};
    }
    return w;
}

// One stream of a capture: its FIFO and how far the ingest has walked it. Every frame below `walked` has a valid
// header, so a consumer reads frames in place from its own cursor up to there.
struct ProducerStream {
    std::span<const std::byte> fifo;
    const std::atomic<uint64_t>* walked = nullptr;  // bytes, absolute
    uint32_t dev = 0;                               // index into capture_context().devices
    std::span<const std::atomic<uint64_t>> marks;   // frame boundaries, one per kMarkBytes of `fifo`
};

// One capture's streams and what a consumer needs to decode them. Valid from attach_producer() until
// detach_producer() returns.
class Producer {
public:
    virtual ~Producer() = default;
    virtual std::span<const ProducerStream> streams() const = 0;
    virtual const CaptureContext& capture_context() const = 0;
    virtual const DeviceClock& clock(uint32_t dev) const = 0;
    // The device's write position in a stream, in bytes; callable from any consumer thread.
    virtual uint64_t live_head(uint32_t stream) const = 0;
    // A consumer's totals for one of streams(): the frame bytes it never read and its decoder's counters, delivered
    // on its thread as it releases the stream.
    virtual void finish_stream(uint32_t stream, uint64_t dropped_bytes, const StreamStats& stats) = 0;
};

// A consumer's delivery; `capture` numbers the producer the batch came from, per consumer, so a sink can tell a new
// capture from the last.
using BatchCallback = std::function<void(
    const experimental::streaming_profiler::Batch<experimental::streaming_profiler::RecordType::All>&,
    uint64_t capture)>;

struct ClockSample;
// Optional hooks a consumer carries besides its batch callback, both run on the consumer's own thread.
// clock_sink: every PP_CLOCK sample this consumer's decoders route (idle-eth trackers, link stamps); the other
// consumers' decoders drop them. on_capture_end: once a producer's last frame is decoded, before its streams
// are finished -- where a fit over the whole capture belongs.
struct ConsumerHooks {
    std::function<void(const ClockSample&)> clock_sink;
    std::function<void(const CaptureContext&)> on_capture_end;
};

class Service {
public:
    Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;

    // The callback runs on the consumer's own thread, one call at a time, for every attached producer. Not from
    // inside a consumer callback.
    ConsumerHandle add_consumer(std::string name, BatchCallback cb);
    ConsumerHandle add_consumer(std::string name, BatchCallback cb, ConsumerHooks hooks);
    // Returns once the callback can no longer run.
    void remove_consumer(ConsumerHandle handle);

    // Returns once every consumer reads the producer's queues, so nothing published afterwards is missed.
    void attach_producer(Producer& producer);
    // Returns once every consumer has drained the producer's queues and released its readers. Detaching the last
    // producer writes the file sinks.
    void detach_producer(Producer& producer);
    bool is_active() const;

    // The Tracy sink and the CSV writers rtoptions select; subsequent calls do nothing.
    void register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions);

    // A producer calls this once after a pass that published. A reader takes wake_token() before checking the queues
    // and, finding nothing, wait_wake()s on it, so a bump between the two returns at once: wait() returns without
    // sleeping when the token has moved, so the notify is only needed, and only issued, while a reader is parked.
    void wake_consumers() {
        wake_gen_.fetch_add(1, std::memory_order_release);
        if (parked_.load(std::memory_order_acquire) != 0) {
            wake_gen_.notify_all();
        }
    }
    uint32_t wake_token() const { return wake_gen_.load(std::memory_order_acquire); }
    void wait_wake(uint32_t seen) const {
        parked_.fetch_add(1, std::memory_order_acq_rel);
        wake_gen_.wait(seen, std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);
    }

private:
    struct Consumer;
    struct AttachedStream;
    struct Attached;
    void consumer_thread(Consumer& c);
    void post_control(Consumer& c, Producer* producer, bool attach);
    void wait_acks(std::unique_lock<std::mutex>& lk);
    static void warn_missed(const Consumer& c);

    // Serializes add/remove/attach/detach against each other; never taken by a consumer thread.
    std::mutex topology_mu_;
    // Guards the lists below and the ack handshake; taken briefly by consumer threads.
    mutable std::mutex mu_;
    std::condition_variable ack_cv_;
    uint64_t pending_acks_ = 0;
    alignas(64) std::atomic<uint32_t> wake_gen_{0};
    alignas(64) mutable std::atomic<uint32_t> parked_{0};
    std::vector<std::unique_ptr<Consumer>> consumers_;
    std::vector<Producer*> producers_;
    std::vector<std::function<void()>> file_sinks_;
    std::unique_ptr<TracySink> tracy_;
    ConsumerHandle next_handle_ = 1;
    std::once_flag builtins_once_;
};

// The process's Service. Subscriptions outlive every device and context, so it is never destroyed.
Service& service();

}  // namespace tt::tt_metal::streaming_profiler
