// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/external_receiver.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#include <tt-logger/tt-logger.hpp>
#include <tt-umd/chip_helpers/silicon_sysmem_manager.hpp>
#include <tt-umd/chip_helpers/sysmem_buffer.hpp>
#include <tt-umd/tt_device/tt_device.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/types/io_window_config.hpp>
#include <unistd.h>
#include <tt_stl/assert.hpp>

#include "distributed/mesh_socket_utils.hpp"
#include "hostdev/socket.h"
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
// The reach of the window onto a relay's L1. A DRISC's L1 sits high in its NoC address space, so the window starts at
// the aligned block holding the relay's control block.
constexpr uint64_t kRelayL1Bytes = uint64_t{1} << 20;
uint64_t relay_l1_base(const external::Hello::Relay& relay) { return relay.ctrl & ~(kRelayL1Bytes - 1); }

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

// This process's own clock solver, which builds the map from the sync records the eth relays copy here and from its own
// reads of the root chip's refclk. It joins the capture when this process attaches.
struct ExternalReceiver::Clock {
    RootRefclkWindow refclk;
    ClockSolver solver;
    std::chrono::steady_clock::time_point next_burst{};

    Clock(tt::umd::TTDevice& root, tt_xy_pair pcie, const CaptureContext& ctx, const ClockBases& bases) :
        refclk(root, pcie), solver(ctx, bases) {
        solver.resume(refclk.bases().root_refclk);
    }
};

ExternalReceiver::ExternalReceiver(std::unique_ptr<external::ExternalEnd> link) : link_(std::move(link)) {
    const external::Hello& hello = *link_->hello();
    capture_.devices = hello.devices;
    capture_.links = hello.links;
    bases_ = hello.bases;
    root_pcie_ = {hello.root_pcie_x, hello.root_pcie_y};
    relays_ = hello.relays;
    l1_alignment_ = hello.l1_alignment;
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
    link_->start();
    rings_.resize(hello.relays.size());
    for (uint32_t r = 0; r < rings_.size(); r++) {
        const external::Hello::Relay& relay = hello.relays[r];
        auto& [tt_device, sysmem] = pcie_devices_[relay.pcie_device];
        if (!tt_device) {
            tt_device = tt::umd::TTDevice::create(static_cast<int>(relay.pcie_device));
            sysmem = std::make_unique<tt::umd::SiliconSysmemManager>(tt_device.get(), 0);
        }
        if (relay.dev == CaptureContext::kRootDevice) {
            root_pcie_device_ = relay.pcie_device;
        }
        Ring& ring = rings_[r];
        ring.dev = relay.dev;
        ring.buffer = sysmem->allocate_sysmem_buffer(kRingBufferBytes, /*map_to_noc=*/true);
        ring.data = static_cast<const volatile uint32_t*>(ring.buffer->get_va());
        ring.bytes_sent = ring.data + kRingBytes / 4;
        const_cast<volatile uint32_t*>(ring.bytes_sent)[0] = 0;
        // Its offset 0 is NoC address relay_l1_base(relay).
        relay_l1_.push_back(tt_device->create_io_window(
            tt::umd::TargetIoWindowConfig{
                .core_start = {relay.core_x, relay.core_y}, .addr = relay_l1_base(relay), .noc = tt::umd::NocId::NOC0},
            tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::UC, .size = kRelayL1Bytes}));
    }
    clock_ = std::make_unique<Clock>(*pcie_devices_.at(root_pcie_device_).first, root_pcie_, capture_, bases_);
    // A consumer reading a record's time() needs the steady clock's map, so it has a sample before any record arrives.
    service().steady().sample();
    service().attach_receiver(*this);
    // A generation no earlier external process gave a relay, which follows only a change.
    static std::atomic<uint32_t> attaches{0};
    point_relays(static_cast<uint32_t>(::getpid()) << 8 | (++attaches & 0xFFu));
    log_info(
        tt::LogMetal,
        "[streaming profiler] receiving external from {} relay(s) on {} device(s)",
        rings_.size(),
        capture_.devices.size());
}

ExternalReceiver::~ExternalReceiver() {
    // run() has finished the map of a profiled process that ended its capture, but not of one that died.
    try {
        end_capture();
    } catch (const std::exception& e) {
        log_error(tt::LogMetal, "[streaming profiler] Metal process {}'s clock map: {}", link_->pid(), e.what());
    }
    service().detach_receiver(*this);
    // Relays that finished the capture copy nothing more, and their cores are the next profiled process's.
    if (!relays_done_) {
        point_relays(0);
    }
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

const ClockMap& ExternalReceiver::clock_map() const { return clock_->solver.map(); }

void ExternalReceiver::drop(const std::exception& failure) {
    log_error(
        tt::LogMetal, "[streaming profiler] dropping the link to Metal process {}: {}", link_->pid(), failure.what());
    link_->drop();
}

void ExternalReceiver::run(const std::stop_token& stop) {
    auto next_sample = std::chrono::steady_clock::now();
    bool linked = true;
    while (!stop.stop_requested()) {
        // A profiled process that ends its capture stops the relays, whose last copies have landed by then, before it
        // closes the link; one that dies leaves them running. The link is checked before the rings are read, so that
        // read covers every copy made before the close.
        const bool closing = linked && link_->closed();
        bool any = false;
        for (Ring& ring : rings_) {
            any |= read_ring(ring);
        }
        if (closing) {
            linked = false;
            if (relays_finished()) {
                relays_done_ = true;
                end_capture();
                service().wake_consumers();
                return;
            }
            log_info(
                tt::LogMetal, "[streaming profiler] the profiled process is gone; its relays and this clock map go on");
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= clock_->next_burst) {
            clock_->next_burst = now + kRefclkBurstPeriod;
            clock_->solver.on_refclk_burst(clock_->refclk.read_burst());
            solver_pending_ = true;
        }
        if (solver_pending_) {
            solver_pending_ = false;
            any |= clock_->solver.on_batch_end();
        }
        if (any) {
            service().wake_consumers();
        }
        if (now >= next_sample) {
            next_sample = now + kSteadySamplePeriod;
            service().steady().sample();
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
                if (words_[at] & kp::SPSC_SPAN_SYNC) {
                    on_sync(&words_[at], ring.dev);
                } else {
                    repack(&words_[at], ring.dev);
                }
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

void ExternalReceiver::point_relays(uint32_t attach) {
    constexpr uint64_t kAttach = offsetof(kp::ResidentCtrl, external.attach);
    constexpr uint64_t kAttached = offsetof(kp::ResidentCtrl, external.attached);
    for (uint32_t r = 0; r < relays_.size(); r++) {
        const external::Hello::Relay& relay = relays_[r];
        tt::umd::IoWindow& l1 = *relay_l1_[r];
        if (attach != 0) {
            // A D2H socket's sender config (distributed/d2h_socket.cpp), aimed at the ring, with the relay's
            // bytes_sent in the word after it.
            const uint64_t noc_addr = rings_[r].buffer->get_noc_address().value();
            const uint64_t bytes_sent_addr = noc_addr + kRingBytes;
            const distributed::SocketSenderSize size(l1_alignment_);
            std::vector<uint32_t> config((size.md_size_bytes + size.ack_size_bytes + size.enc_size_bytes) / 4, 0);
            config[offsetof(sender_socket_md, num_downstreams) / 4] = 1;
            config[offsetof(sender_socket_md, downstream_bytes_sent_addr) / 4] = static_cast<uint32_t>(bytes_sent_addr);
            config[offsetof(sender_socket_md, downstream_fifo_addr) / 4] = static_cast<uint32_t>(noc_addr);
            config[offsetof(sender_socket_md, downstream_fifo_total_size) / 4] = kRingBytes;
            config[offsetof(sender_socket_md, is_d2h) / 4] = 1;
            uint32_t* encoding = config.data() + (size.md_size_bytes + size.ack_size_bytes) / 4;
            encoding[0] = static_cast<uint32_t>(bytes_sent_addr >> 32);
            encoding[1] = static_cast<uint32_t>(noc_addr >> 32);
            encoding[2] = relay.pcie_encoding;
            l1.write_block(relay.config - relay_l1_base(relay), config.data(), config.size() * sizeof(uint32_t));
        }
        l1.write32(relay.ctrl - relay_l1_base(relay) + kAttach, attach);
    }
    // A relay echoes the generation once it follows; turning copies off, it first waits for its last one to land.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    uint32_t followed = 0;
    for (uint32_t r = 0; r < relays_.size(); r++) {
        const external::Hello::Relay& relay = relays_[r];
        tt::umd::IoWindow& l1 = *relay_l1_[r];
        const uint64_t attached = relay.ctrl - relay_l1_base(relay) + kAttached;
        while (l1.read32(attached) != attach && std::chrono::steady_clock::now() < deadline) {
        }
        followed += l1.read32(attached) == attach;
    }
    if (attach != 0 && followed != relays_.size()) {
        log_warning(tt::LogMetal, "[streaming profiler] {} of {} relays took this process's rings", followed, relays_.size());
    }
}

// Feeds a frame of clock sync records an eth relay copied here to this process's solver.
void ExternalReceiver::on_sync(const uint32_t* frame, uint32_t dev) {
    const uint32_t core = capture_.devices[dev].core_of_xy.find(frame[kp::SPSC_PREFIX_XY]);
    const uint32_t records = frame[kp::SPSC_PREFIX_PAYLOAD_WORDS] / kp::kSyncRecordWords;
    const auto* record = reinterpret_cast<const kp::SyncRecord*>(frame + kp::SPSC_SPAN_PREFIX_WORDS);
    for (uint32_t r = 0; r < records; r++) {
        clock_->solver.on_record(dev, core, record[r]);
    }
    solver_pending_ = true;
}

bool ExternalReceiver::relays_finished() {
    for (uint32_t r = 0; r < relays_.size(); r++) {
        const uint64_t done = relays_[r].ctrl - relay_l1_base(relays_[r]) + offsetof(kp::ResidentCtrl, done);
        if (relay_l1_[r]->read32(done) != kp::kResidentDoneWord) {
            return false;
        }
    }
    return true;
}

void ExternalReceiver::end_capture() {
    if (std::exchange(ended_, true)) {
        return;
    }
    if (solver_pending_) {
        solver_pending_ = false;
        clock_->solver.on_batch_end();
    }
    // Bursts after the capture's last records extend the refclk-to-TSC map past them, until enough exist to fit it.
    while (!clock_->solver.on_refclk_burst(clock_->refclk.read_burst())) {
    }
    clock_->solver.on_capture_end();
}

std::optional<uint32_t> ExternalReceiver::local_id(uint32_t id) {
    auto [it, inserted] = sites_.try_emplace(id);
    if (inserted) {
        it->second = link_->local_id(id);
        if (!it->second) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] Metal process {}'s link ended without its call site {}, whose records are "
                "skipped",
                link_->pid(),
                id);
        }
    }
    return it->second;
}

namespace {

// The receiver of the profiled process this process serves now, and the thread reading its rings. Its relays copy into
// the rings until it is destroyed, so the process destroys it before it exits.
struct Serving {
    std::mutex mu;
    std::unique_ptr<ExternalReceiver> receiver;
    std::jthread reading;

    void replace(std::unique_ptr<ExternalReceiver> next) {
        std::lock_guard lock(mu);
        reading = {};
        receiver = std::move(next);
        if (receiver) {
            reading = std::jthread([r = receiver.get()](const std::stop_token& stop) {
                try {
                    r->run(stop);
                } catch (const std::exception& e) {
                    r->drop(e);
                }
            });
        }
    }
};
Serving& serving() {
    static auto* instance = new Serving;
    return *instance;
}

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
        std::atexit([] { serving().replace(nullptr); });
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
            // A link that fails is dropped, and this process goes on to serve the next profiled process.
            for (int fd = l.listener.accept(); fd >= 0; fd = l.listener.accept()) {
                try {
                    auto link = std::make_unique<external::ExternalEnd>(fd);
                    if (!link->hello()) {
                        continue;
                    }
                    // A profiled process that connects has booted the devices' relays anew, so the last one's receiver
                    // goes.
                    serving().replace(nullptr);
                    serving().replace(std::make_unique<ExternalReceiver>(std::move(link)));
                } catch (const std::exception& e) {
                    log_error(tt::LogMetal, "[streaming profiler] couldn't serve a Metal process: {}", e.what());
                }
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
    serving().replace(nullptr);
}

}  // namespace tt::tt_metal::streaming_profiler
