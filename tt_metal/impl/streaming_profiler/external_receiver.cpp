// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/external_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#include <tt-logger/tt-logger.hpp>
#include <tt-umd/chip_helpers/silicon_sysmem_manager.hpp>
#include <tt-umd/chip_helpers/sysmem_buffer.hpp>
#include <tt-umd/tt_device/tt_device.hpp>
#include <tt_stl/assert.hpp>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

namespace kp = kernel_profiler;
using profiler::kSpscRingCap;
using profiler::kSpscRingMask;
using profiler::spsc_packet_words;

constexpr uint32_t kPageWords = kp::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kLanes = kp::PROFILER_SPSC_TENSIX_RISC;
// Each relay's ring, which the relays overwrite and never wait on, so it only needs to hold what lands between two
// reads. Its buffer holds the relay's bytes_sent in its last page, and is a whole number of 2 MiB hugepages, the only
// size UMD backs with them.
constexpr uint32_t kRingBufferBytes = 4u << 20;
constexpr uint32_t kRingBytes = kRingBufferBytes - kPageWords * 4;
// How far past the bytes_sent this process sees a relay may have landed copies: it sends bytes_sent once every write
// before it is acked, and pushes at most a notify cap and a frame past the bytes_sent it sent last.
constexpr uint32_t kLandedPastBytesSent = 2 * (kp::SPSC_NOTIFY_CAP_BYTES + profiler::kSpscMaxFrameWords * 4);
// Each device's stream of repacked frames, which the consumers read like a device's FIFO.
constexpr uint64_t kStreamBytes = 16u << 20;
constexpr auto kSteadySamplePeriod = std::chrono::milliseconds(100);

// A copy's word 0 holds its pages [first, end) of the frame it was copied from, in words here.
uint32_t span_words(uint32_t w0, uint32_t shift) {
    return ((w0 >> shift) & kp::SPSC_SPAN_ROUTED_PAGE_MASK) * kPageWords;
}
// A copy is the frame's words up to its second page or its routed lanes' first page, then the pages [first, end).
uint32_t copy_words(const uint32_t* copy) {
    const uint32_t first = span_words(copy[0], kp::SPSC_SPAN_ROUTED_FIRST_SHIFT);
    const uint32_t end = span_words(copy[0], kp::SPSC_SPAN_ROUTED_END_SHIFT);
    return std::min(first, 2 * kPageWords) + end - first;
}

}  // namespace

struct ExternalReceiver::Ring {
    std::unique_ptr<tt::umd::SysmemBuffer> buffer;
    const volatile uint32_t* data = nullptr;
    const volatile uint32_t* bytes_sent = nullptr;
    uint32_t read = 0;    // bytes_sent at the last read
    uint32_t offset = 0;  // where `read` is in the ring
    uint32_t dev = 0;
};

struct ExternalReceiver::Stream {
    std::unique_ptr<std::byte[]> fifo;  // kStreamBytes
    std::atomic<uint64_t> head{0};      // where the frame being written ends
    std::atomic<uint64_t> walked{0};    // where the last frame written ends
    std::vector<std::atomic<uint64_t>> marks;
    uint64_t mark_block = UINT64_MAX;
    // Where each lane's last run ends, in positions of the lane's own that start each run on a ring boundary, so no
    // run is a wrap image. By core, then RISC.
    std::vector<uint32_t> lane_tails;
};

ExternalReceiver::ExternalReceiver(std::unique_ptr<external::ExternalEnd> link) : link_(std::move(link)) {
    const external::Hello& hello = *link_->hello();
    capture_.devices = hello.devices;
    clock_map_ = std::make_unique<ClockMap>(hello.devices.size(), hello.bases);
    for (uint32_t dev = 0; dev < hello.devices.size(); dev++) {
        auto stream = std::make_unique<Stream>();
        stream->fifo = std::make_unique_for_overwrite<std::byte[]>(kStreamBytes);
        stream->marks = std::vector<std::atomic<uint64_t>>(kStreamBytes / kMarkBytes);
        for (auto& mark : stream->marks) {
            mark.store(UINT64_MAX, std::memory_order_relaxed);
        }
        stream->lane_tails.assign(hello.devices[dev].lanes.size(), 0);
        streams_view_.push_back(ReceiverStream{
            .fifo = std::span<const std::byte>(stream->fifo.get(), kStreamBytes),
            .walked = &stream->walked,
            .dev = dev,
            .marks = stream->marks});
        streams_.push_back(std::move(stream));
    }
    link_->start([this](std::span<const ClockUpdate> updates) {
        for (const ClockUpdate& update : updates) {
            clock_map_->apply(update);
        }
        service().wake_consumers();
    });
    std::vector<external::RelayBuffer> buffers;
    rings_.resize(hello.relays.size());
    for (uint32_t r = 0; r < rings_.size(); r++) {
        const external::Hello::Relay& relay = hello.relays[r];
        auto& [tt_device, sysmem] = pcie_devices_[relay.pcie_device];
        if (!tt_device) {
            tt_device = tt::umd::TTDevice::create(static_cast<int>(relay.pcie_device));
            sysmem = std::make_unique<tt::umd::SiliconSysmemManager>(tt_device.get(), 0);
        }
        Ring& ring = rings_[r];
        ring.dev = relay.dev;
        ring.buffer = sysmem->allocate_sysmem_buffer(kRingBufferBytes, /*map_to_noc=*/true);
        ring.data = static_cast<const volatile uint32_t*>(ring.buffer->get_va());
        ring.bytes_sent = ring.data + kRingBytes / 4;
        const_cast<volatile uint32_t*>(ring.bytes_sent)[0] = 0;
        buffers.push_back({.relay = r, .noc_addr = ring.buffer->get_noc_address().value(), .bytes = kRingBytes});
    }
    // A consumer reading a record's time() needs the steady clock's map, so it has a sample before any record arrives.
    service().steady().sample();
    service().attach_receiver(*this);
    if (!link_->attach(buffers)) {
        log_warning(tt::LogMetal, "[streaming profiler] the profiled process couldn't point every relay at this process");
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] receiving external from {} relay(s) on {} device(s)",
        rings_.size(),
        capture_.devices.size());
}

ExternalReceiver::~ExternalReceiver() {
    // A profiled process that ended its capture has finished the clock map already; one that died hasn't.
    for (uint32_t dev = 0; dev < capture_.devices.size(); dev++) {
        clock_map_->finish(dev);
    }
    service().detach_receiver(*this);
    link_->detach();
    if (lost_bytes_ != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] the relays wrote over {} bytes of copies before this process read them",
            lost_bytes_);
    }
}

uint64_t ExternalReceiver::live_head(uint32_t stream) const {
    return streams_[stream]->head.load(std::memory_order_acquire);
}

void ExternalReceiver::run() {
    auto next_sample = std::chrono::steady_clock::now();
    while (true) {
        // A pass that starts after the link closed reads every copy that landed before, since the profiled process
        // stops its relays' copies before it closes the link.
        const bool closed = link_->closed();
        bool any = false;
        for (Ring& ring : rings_) {
            any |= read_ring(ring);
        }
        if (any) {
            service().wake_consumers();
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_sample) {
            next_sample = now + kSteadySamplePeriod;
            service().steady().sample();
        }
        if (closed && !any) {
            return;
        }
        if (!any) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

// Reads the copies the relay has sent since the last read. Like a BroadcastRing's writer, the relay overwrites the ring
// without waiting for this reader, which drops what the relay may have written over before or while it read. bytes_sent
// only stops where a copy ends, so a reader the relay laps picks up there.
bool ExternalReceiver::read_ring(Ring& ring) {
    const uint32_t sent = *ring.bytes_sent;
    if (sent == ring.read) {
        return false;
    }
    // Whether the relay may have written over the oldest unread byte, by the bytes_sent it shows.
    const auto lapped = [&](uint32_t shown) { return shown - ring.read > kRingBytes - kLandedPastBytesSent; };
    uint32_t read = sent;
    if (lapped(sent)) {
        lost_bytes_ += sent - ring.read;
    } else {
        words_.resize((sent - ring.read) / 4);
        const auto* data = const_cast<const uint32_t*>(ring.data);
        const size_t from = ring.offset / 4;
        const size_t first = std::min(words_.size(), kRingBytes / 4 - from);
        std::memcpy(words_.data(), data + from, first * 4);
        std::memcpy(words_.data() + first, data, (words_.size() - first) * 4);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (lapped(*ring.bytes_sent)) {
            lost_bytes_ += sent - ring.read;
        } else {
            size_t at = 0;
            while (at < words_.size()) {
                const uint32_t n = copy_words(&words_[at]);
                // A copy that runs past bytes_sent is left for a later read.
                if (n == 0 || at + n > words_.size()) {
                    break;
                }
                repack(&words_[at], ring.dev);
                at += n;
            }
            read = ring.read + static_cast<uint32_t>(at * sizeof(uint32_t));
        }
    }
    const bool moved = read != ring.read;
    ring.offset = (ring.offset + (read - ring.read)) % kRingBytes;
    ring.read = read;
    return moved;
}

// Rebuilds the copied frame, then writes a frame of its records for this process: the records with PP_TO_EXTERNAL
// from each routed lane's run, whole, with this process's ids for their call sites. A run cut short ends at its last
// whole record.
void ExternalReceiver::repack(const uint32_t* copy, uint32_t dev) {
    const uint32_t first = span_words(copy[0], kp::SPSC_SPAN_ROUTED_FIRST_SHIFT);
    const uint32_t end = span_words(copy[0], kp::SPSC_SPAN_ROUTED_END_SHIFT);
    const uint32_t head = std::min(first, 2 * kPageWords);
    frame_.assign(end, 0);
    std::copy(copy, copy + head, frame_.begin());
    std::copy(copy + head, copy + head + end - first, frame_.begin() + first);
    const uint32_t core = capture_.devices[dev].core_of_xy.find(frame_[kp::SPSC_PREFIX_XY]);
    TT_FATAL(
        core != CoreTable::kNone,
        "streaming profiler: a copy from NoC core {:#x}, which the profiled process's capture didn't seed",
        frame_[kp::SPSC_PREFIX_XY]);
    const uint32_t* ctrl = frame_.data() + kp::SPSC_SPAN_PREFIX_WORDS;
    Stream& stream = *streams_[dev];
    uint32_t* const tails = stream.lane_tails.data() + size_t{core} * kLanes;
    packed_.assign(frame_.begin(), frame_.begin() + kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS);
    uint32_t off = kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_SPAN_WIRE_CTRL_WORDS;
    bool kept_any = false;
    for (uint32_t risc = 0; risc < kLanes; risc++) {
        const uint32_t start = frame_[kp::SPSC_PREFIX_HEAD_0 + risc];
        const uint32_t take = ctrl[kp::SPSC_WIRE_TAIL_0 + risc] - start;
        uint32_t lane_start = tails[risc];
        size_t kept = 0;
        if (take != 0) {
            const bool image = kp::spsc_span_wrap_image(start, take, kSpscRingCap);
            off += kp::spsc_span_pack_pad(image ? 0u : start, off);
            const uint32_t words = image ? kSpscRingCap : take;
            // The relay copies a ring image whole or not at all, and a run in ring order up to the copy's end.
            const uint32_t copied =
                image ? (off + words <= end ? take : 0) : (off < end ? std::min(take, end - off) : 0);
            if (off >= first && copied != 0) {
                lane_start = (tails[risc] + kSpscRingCap - 1) & ~(kSpscRingCap - 1);
                const size_t pad_at = packed_.size();
                packed_.resize(pad_at + kp::spsc_span_pack_pad(lane_start, static_cast<uint32_t>(pad_at)), 0);
                const size_t run_at = packed_.size();
                const auto word = [&](uint32_t k) { return frame_[off + (image ? (start + k) & kSpscRingMask : k)]; };
                for (uint32_t k = 0; k < copied;) {
                    const uint32_t w0 = word(k);
                    const uint32_t n = spsc_packet_words(w0);
                    if (n == 0 || k + n > copied) {
                        break;
                    }
                    if (pp_type(w0) == PP_RECORD && (w0 & PP_TO_EXTERNAL) != 0) {
                        if (const std::optional<uint32_t> id = local_id(w0 & PP_ID_MASK)) {
                            packed_.push_back((w0 & ~PP_ID_MASK) | *id);
                            for (uint32_t j = 1; j < n; j++) {
                                packed_.push_back(word(k + j));
                            }
                        }
                    }
                    k += n;
                }
                kept = packed_.size() - run_at;
                if (kept == 0) {
                    packed_.resize(pad_at);
                    lane_start = tails[risc];
                }
            }
            off += words;
        }
        tails[risc] = lane_start + static_cast<uint32_t>(kept);
        packed_[kp::SPSC_PREFIX_HEAD_0 + risc] = lane_start;
        packed_[kp::SPSC_SPAN_PREFIX_WORDS + kp::SPSC_WIRE_TAIL_0 + risc] = tails[risc];
        kept_any |= kept != 0;
    }
    if (!kept_any) {
        return;
    }
    packed_[0] = kp::spsc_span_w0();
    packed_[kp::SPSC_PREFIX_PAYLOAD_WORDS] = static_cast<uint32_t>(packed_.size()) - kp::SPSC_SPAN_PREFIX_WORDS;
    packed_.resize(kp::spsc_span_frame_words(packed_[kp::SPSC_PREFIX_PAYLOAD_WORDS]), 0);
    write(stream);
}

// Appends packed_ to the stream. The head moves past the frame before its bytes are written, so a consumer this write
// laps sees so when it checks the head after its copy.
void ExternalReceiver::write(Stream& stream) {
    const uint64_t at = stream.walked.load(std::memory_order_relaxed);
    const size_t bytes = packed_.size() * sizeof(uint32_t);
    stream.head.store(at + bytes, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const size_t pos = at & (kStreamBytes - 1);
    const size_t first = std::min<size_t>(bytes, kStreamBytes - pos);
    std::memcpy(stream.fifo.get() + pos, packed_.data(), first);
    std::memcpy(stream.fifo.get(), reinterpret_cast<const std::byte*>(packed_.data()) + first, bytes - first);
    if (at / kMarkBytes != stream.mark_block) {
        stream.mark_block = at / kMarkBytes;
        stream.marks[stream.mark_block % stream.marks.size()].store(at, std::memory_order_release);
    }
    stream.walked.store(at + bytes, std::memory_order_release);
}

std::optional<uint32_t> ExternalReceiver::local_id(uint32_t id) {
    auto [it, inserted] = sites_.try_emplace(id);
    if (inserted) {
        it->second = link_->add_site(id);
    }
    return it->second;
}

namespace {

// This process's socket, which it listens on until it creates a MetalContext of its own.
struct Listening {
    std::mutex mu;
    std::condition_variable cv;
    bool stopped = false;
    external::ExternalListener listener;
};
Listening& listening() {
    static auto* instance = new Listening;
    return *instance;
}
std::atomic<bool> serve_started{false};

// Listens once the socket name is free, and is false when this process stopped serving first. Another process holds the
// name while an external process runs, and so does a Metal process that registered callbacks before opening its
// devices, until it opens them.
bool listen_when_free(Listening& l) {
    std::unique_lock lock(l.mu);
    while (!l.stopped && !l.listener.listen()) {
        l.cv.wait_for(lock, std::chrono::seconds(1), [&] { return l.stopped; });
    }
    return !l.stopped;
}

}  // namespace

void serve_profiled_processes() {
    static std::once_flag once;
    std::call_once(once, [] {
        serve_started.store(true, std::memory_order_release);
        // The first try is made here, so a process that finds the name free listens before its RegisterCallback
        // returns.
        bool listening_now = false;
        {
            Listening& l = listening();
            std::lock_guard lock(l.mu);
            listening_now = !l.stopped && l.listener.listen();
        }
        std::thread([listening_now] {
            set_thread_name("sp-external");
            // The first call sleeps while it measures the rate, so it is made before any capture.
            static_cast<void>(ns_per_tsc_tick());
            Listening& l = listening();
            if (!listening_now && !listen_when_free(l)) {
                return;
            }
            for (int fd = l.listener.accept(); fd >= 0; fd = l.listener.accept()) {
                auto link = std::make_unique<external::ExternalEnd>(fd);
                if (!link->hello()) {
                    continue;
                }
                ExternalReceiver receiver(std::move(link));
                receiver.run();
            }
        }).detach();
    });
}

void stop_serving_profiled_processes() {
    if (!serve_started.load(std::memory_order_acquire)) {
        return;
    }
    Listening& l = listening();
    {
        std::lock_guard lock(l.mu);
        l.stopped = true;
        l.listener.stop();
    }
    l.cv.notify_all();
}

}  // namespace tt::tt_metal::streaming_profiler
