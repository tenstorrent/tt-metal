// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/clock_solver.hpp"

namespace tt {
class Cluster;
namespace umd {
class IoWindow;
}
}  // namespace tt

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace streaming_profiler {

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

struct Walked {
    uint32_t frames = 0;   // copied into `out`, back to back
    size_t bytes = 0;      // their total length
    uint64_t dropped = 0;  // bytes skipped because the device had reached them
    uint64_t cursor = 0;   // where the next pass starts
};

// Copies whole frames from `cursor` up to `end`, the ingest's walk position, which is always a frame boundary. It
// copies at most frame_words.size() frames into `out`, back to back, and records each one's length in words in
// `frame_words`, so `out` needs room for that many frames of kSpscMaxFrameWords words. The FIFO's size is a power of
// two, and the device keeps writing it during the copy. If the device may already have overwritten the frame at
// `cursor`, the walk skips ahead to the first recorded frame boundary an eighth of the FIFO past the device's overwrite
// point, which leaves room for the device's writes during the copy, and counts the skipped bytes as dropped. If the
// device reaches the first copied frame during the copy, the lengths read from the FIFO may be wrong, so the pass
// returns no frames and the next pass skips ahead.
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
        uint32_t w1;
        std::memcpy(&w1, fifo.data() + ((w.cursor + 4) & mask), 4);
        const uint32_t fw = kernel_profiler::spsc_span_frame_words(w1);
        const size_t bytes = size_t{fw} * 4;
        if (w1 > profiler::kSpscMaxPayloadWords || bytes > end - w.cursor) {
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
struct ReceiverStream {
    std::span<const std::byte> fifo;
    const std::atomic<uint64_t>* walked = nullptr;  // bytes, absolute
    uint32_t dev = 0;                               // index into capture_context().devices
    std::span<const std::atomic<uint64_t>> marks;   // frame boundaries, one per kMarkBytes of `fifo`
};

// Reads the root chip's refclk from the host through the PCIe tile's count-from-reset timer. That timer counts the same
// distributed clock as the eth tiles, sits one NoC hop from the PCIe entry, and nothing else reads it, so its high word
// holds between the two halves of a read. It must be destroyed before its device is torn down.
class RootRefclkWindow {
public:
    RootRefclkWindow(tt::Cluster& cluster, uint32_t chip_id);
    ~RootRefclkWindow();

    // The refclk and the host TSC when the window opened.
    const ClockBases& bases() const { return bases_; }
    // The NUMA node of the root chip's PCIe link, or -1 if it has none.
    int numa_node() const { return numa_node_; }
    // Reads the refclk kRefclkBurstReads times back to back.
    RefclkBurst read_burst();

private:
    const int numa_node_;
    std::unique_ptr<tt::umd::IoWindow> window_;
    // The timer's low and high words in the window's mapping.
    const volatile uint32_t* lo_ = nullptr;
    const volatile uint32_t* hi_ = nullptr;
    ClockBases bases_;
};

class Receiver {
public:
    static std::unique_ptr<Receiver> create(const std::shared_ptr<distributed::MeshDevice>& mesh_device);
    ~Receiver();

    Receiver(const Receiver&) = delete;
    Receiver& operator=(const Receiver&) = delete;

    // The streams that carry profiler records.
    std::span<const ReceiverStream> streams() const { return streams_view_; }
    const CaptureContext& capture_context() const { return programs_->capture_context(); }
    // The absolute byte position up to which the device has written stream `stream`'s FIFO. Callable from any thread.
    uint64_t live_head(uint32_t stream) const;
    const ClockMap& clock_map() const { return clock_solver_.map(); }

private:
    Receiver(tt::Cluster& cluster, std::unique_ptr<DevicePrograms> programs);

    struct Stream {
        CapturedSocket captured;
        std::span<const std::byte> fifo;  // the socket's host FIFO: `capacity` pages of SPSC_SPAN_PAGE_WORDS
        uint64_t capacity = 0;            // a power of two
        std::atomic<RelayState> relay{RelayState::Running};

        uint64_t arrived = 0, consumed = 0, acked = 0;  // absolute pages: landed, walked, credited back
        uint64_t fullest = 0;                    // most pages ever awaiting credit; capacity means the device waited
        std::atomic<uint64_t> arrived_bytes{0};  // `arrived`, for consumers widening the device's head
        std::atomic<uint64_t> walked_bytes{0};   // `consumed`: the consumers' walk limit and resync point
        uint64_t frames = 0;
        std::vector<std::atomic<uint64_t>> marks;  // first frame boundary per kMarkBytes; UINT64_MAX until written
        uint64_t mark_block = UINT64_MAX;

        const std::byte* page(uint64_t p) const {
            return fifo.data() + (p & (capacity - 1)) * kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
        }
    };

    void ingest_thread(std::vector<Stream*> streams);
    // Feeds the clock solver the clock sync records that the eth relays send on their sync sockets, plus a burst of
    // host reads of the root chip's refclk every kRefclkBurstPeriod. Once stopped, drains the sync streams and finishes
    // the clock map.
    void sync_thread(const std::stop_token& stop);
    // Wakes the sync thread, which sleeps until a sync stream publishes or it is stopped.
    void wake_sync_thread();
    bool poll(Stream& s);
    bool walk_frame(Stream& s);
    // Publishes the stream's walk position, credits walked pages back once the relay has stopped, and returns whether
    // the relay is done.
    bool settle(Stream& s);
    Stream& stream(uint32_t device_index, uint32_t socket_index);
    void log_report() const;

    std::unique_ptr<DevicePrograms> programs_;
    RootRefclkWindow root_refclk_;
    // The window onto the root chip's wall clock when that chip has no Ethernet cores (p100a), and null otherwise.
    std::unique_ptr<tt::umd::IoWindow> fallback_wall_clock_;
    ClockSolver clock_solver_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<ReceiverStream> streams_view_;
    std::vector<std::thread> ingest_threads_;
    std::atomic<uint32_t> sync_wake_{0};
    std::jthread sync_thread_;
};

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
