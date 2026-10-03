// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/sync/tile_sync.hpp"

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal::streaming_profiler {

class Receiver;

void set_thread_name(const std::string& name);
void init_site_registry();

// The device is never more than one FIFO (2 GB at most) past what it has been credited, so the difference
// fits 32 bits.
inline uint64_t widen_head(uint64_t observed, uint32_t bytes_sent) {
    return observed + static_cast<uint32_t>(bytes_sent - static_cast<uint32_t>(observed));
}

// Everything the device has landed lies below head + SPSC_NOTIFY_CAP_BYTES (the relay writes bytes_sent before it
// pushes more than that), and a frame at `offset` is overwritten only by writes from offset + fifo_bytes on.
inline bool frame_intact(uint64_t offset, uint64_t fifo_bytes, uint64_t head) {
    return head + kernel_profiler::SPSC_NOTIFY_CAP_BYTES <= offset + fifo_bytes;
}

inline constexpr uint64_t kMarkBytes = 64 * 1024;
static_assert(kMarkBytes >= profiler::kSpscMaxFrameWords * 4, "every block holds at least one frame boundary");

// The first recorded frame boundary in [from, end], or end.
inline uint64_t resume_mark(std::span<const std::atomic<uint64_t>> marks, uint64_t from, uint64_t end) {
    for (uint64_t b = from / kMarkBytes; !marks.empty() && b * kMarkBytes < end; b++) {
        const uint64_t v = marks[b % marks.size()].load(std::memory_order_acquire);
        if (v >= from && v <= end) {
            return v;
        }
    }
    return end;
}

struct Walked {
    uint32_t frames = 0;   // copied into `out`, back to back
    size_t bytes = 0;      // their total length
    uint64_t dropped = 0;  // bytes skipped because the device had reached them
    uint64_t cursor = 0;   // where the next pass starts
};

// Copies whole frames from `cursor` up to `end` into `out`, at most frame_words.size() of them, and records each one's
// length in words; `out` holds frame_words.size() frames of kSpscMaxFrameWords. A cursor the device has passed resumes
// an eighth of the FIFO past the overwrite horizon (the device keeps writing during the copy); if the device reached
// the pass's first frame during the copy the lengths read from those pages are void, so the pass is discarded.
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
    if (const uint64_t head = live_head(); !frame_intact(start, fifo.size(), head)) {
        const uint64_t written = head + kernel_profiler::SPSC_NOTIFY_CAP_BYTES;
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
        uint32_t payload_words;
        std::memcpy(
            &payload_words,
            fifo.data() + ((w.cursor + kernel_profiler::SPSC_PREFIX_PAYLOAD_WORDS * sizeof(uint32_t)) & mask),
            sizeof(uint32_t));
        const uint32_t fw = kernel_profiler::spsc_span_frame_words(payload_words);
        const size_t bytes = size_t{fw} * 4;
        if (payload_words > profiler::kSpscMaxPayloadWords || bytes > end - w.cursor) {
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

// Every frame below `walked` has a valid header, so a consumer reads frames in place from its own cursor up to there.
struct ReceiverStream {
    std::span<const std::byte> fifo;
    const std::atomic<uint64_t>* walked = nullptr;
    uint32_t dev = 0;
    std::span<const std::atomic<uint64_t>> marks;  // frame boundaries, one per kMarkBytes of `fifo`
    bool sync = false;
};

class SteadyClock;

using BatchCallback = std::function<void(const experimental::streaming_profiler::detail::BatchData&)>;

class Service {
public:
    Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;

    experimental::streaming_profiler::detail::CallbackId add_consumer(std::string name, BatchCallback callback);
    // Returns once the callback can no longer run. A callback removing itself returns at once.
    void remove_consumer(experimental::streaming_profiler::detail::CallbackId id);

    // Returns once the sync thread, then every consumer, reads the receiver's streams, so the covers exist before any
    // consumer parks a batch on them and nothing the receiver publishes afterwards is missed.
    void attach_receiver(Receiver& receiver);
    // Returns once the sync thread, then every consumer, has drained the receiver's streams, so the covers are final
    // before the consumers flush what they parked. Detaching the last receiver writes the CSV file sinks.
    void detach_receiver(Receiver& receiver);
    bool is_active() const;

    void register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions);
    void set_tile_clocks(ContextId context_id, uint32_t chip, TileClocks clocks);
    const TileClocks* tile_clocks(ContextId context_id, uint32_t chip) const;
    SteadyClock& steady();
    // Tracy keys a plot by its name's address and reads the name after the capture ends.
    const char* plot_name(const std::string& name);

    // Wakes every StreamWalker (the sync thread and each consumer). A walker takes wake_token() before checking the
    // queues and, finding nothing, wait_wake()s on it, so a wake between the two returns at once.
    void wake_walkers() {
        wake_gen_.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_readers_.load(std::memory_order_seq_cst) != 0) {
            wake_gen_.notify_all();
        }
    }
    uint32_t wake_token() const { return wake_gen_.load(std::memory_order_acquire); }
    void wait_wake(uint32_t seen) const {
        sleeping_readers_.fetch_add(1, std::memory_order_seq_cst);
        wake_gen_.wait(seen, std::memory_order_seq_cst);
        sleeping_readers_.fetch_sub(1, std::memory_order_acq_rel);
    }

private:
    enum class Control { Attach, Detach };
    struct ControlItem {
        Receiver* receiver = nullptr;
        Control action = Control::Attach;
    };
    struct ControlQueue {
        std::atomic<bool> pending{false};
        std::mutex mu;
        std::vector<ControlItem> items;
    };
    enum class Streams { Sync, Profiler };
    struct Consumer;
    struct AttachedStream;
    struct Attached;
    struct Parked;
    class StreamWalker;
    class ConsumerLoop;
    class SyncLoop;
    void consumer_thread(Consumer& consumer);
    void sync_thread();
    void post_control(ControlQueue& queue, Receiver* receiver, Control action);
    void wait_acks(std::unique_lock<std::mutex>& lk);

    // Never taken by a consumer thread.
    std::mutex topology_mu_;
    mutable std::mutex mu_;
    std::condition_variable ack_cv_;
    uint64_t pending_acks_ = 0;
    alignas(64) std::atomic<uint32_t> wake_gen_{0};
    alignas(64) mutable std::atomic<uint32_t> sleeping_readers_{0};
    std::vector<std::unique_ptr<Consumer>> consumers_;
    std::vector<Receiver*> receivers_;
    std::vector<std::function<void()>> file_sinks_;
    std::unique_ptr<SteadyClock> steady_;
    ControlQueue sync_control_;
    std::thread sync_thread_;
    std::vector<experimental::streaming_profiler::Callback> builtin_callbacks_;
    uint64_t next_id_ = 1;
    std::once_flag builtins_once_;
    mutable std::mutex tile_clocks_mu_;
    std::map<std::pair<ContextId, uint32_t>, TileClocks> tile_clocks_;
    std::mutex plot_names_mu_;
    std::unordered_set<std::string> plot_names_;
};

// Subscriptions outlive every device and context, so it is never destroyed.
Service& service();

}  // namespace tt::tt_metal::streaming_profiler
