// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_receiver.hpp"

#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <thread>
#include <utility>
#include <vector>
#include <sys/prctl.h>
#include <pthread.h>
#include <numa.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "context/metal_context.hpp"
#include "llrt/zone_meta.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

// Credits go back about once per relay push rather than per poll.
constexpr uint32_t kAckBatchPages = 8 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4;
// Idle probe period: credits are returned as soon as pages are walked, so a sleep only delays the frames that land
// during it, never a credit the device is short of (the FIFO holds over a millisecond of egress).
constexpr uint32_t kProbeSleepCapUs = 100;

}  // namespace

void set_os_thread_name(const std::string& n) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", n.c_str());
    pthread_setname_np(pthread_self(), buf);
}

Receiver::Receiver(std::unique_ptr<Devices> relays, std::vector<CapturedDevice> devices) :
    relays_(std::move(relays)), devices_(std::move(devices)) {
    for (uint32_t d = 0; d < devices_.size(); d++) {
        auto& dev = devices_[d];
        for (uint32_t sk = 0; sk < dev.sockets.size(); sk++) {
            auto s = std::make_unique<Stream>();
            s->sock = dev.sockets[sk].get();
            s->dev = d;
            s->sock_idx = sk;
            s->fifo = s->sock->host_fifo();
            TT_FATAL(
                s->sock->get_fifo_curr_size() == s->fifo.size() && s->fifo.size() % kPageBytes == 0 &&
                    std::has_single_bit(s->fifo.size()),
                "streaming profiler: the host FIFO must be a power-of-two number of pages");
            s->capacity = s->fifo.size() / kPageBytes;
            s->marks = std::vector<std::atomic<uint64_t>>(s->fifo.size() / kMarkBytes);
            for (auto& m : s->marks) {
                m.store(UINT64_MAX, std::memory_order_relaxed);
            }
            streams_.push_back(std::move(s));
        }
        ctx_.devices.push_back(dev.ctx);
    }
    for (const auto& st : streams_) {
        streams_view_.push_back(
            {st->fifo, &st->walked_bytes, st->dev, std::span<const std::atomic<uint64_t>>(st->marks)});
    }
}

std::unique_ptr<Receiver> Receiver::create(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto relays = std::make_unique<Devices>();
    std::vector<CapturedDevice> devices;
    try {
        devices = relays->boot(mesh_device);
    } catch (const std::exception& e) {
        log_warning(tt::LogMetal, "[streaming profiler] init failed ({}); disabled for this session.", e.what());
        relays->quiesce({});
        return nullptr;
    }
    if (devices.empty()) {
        return nullptr;
    }
    std::unique_ptr<Receiver> receiver(new Receiver(std::move(relays), std::move(devices)));
    service().register_builtin_consumers(MetalContext::instance(mesh_device->impl().get_context_id()).rtoptions());
    service().attach_producer(*receiver);
    for (uint32_t d = 0; d < receiver->devices_.size(); d++) {
        std::vector<Stream*> owned;
        for (auto& s : receiver->streams_) {
            if (s->dev == d) {
                owned.push_back(s.get());
            }
        }
        if (!owned.empty()) {
            receiver->ingest_threads_.emplace_back(&Receiver::ingest_thread, receiver.get(), std::move(owned));
        }
    }
    return receiver;
}

Receiver::~Receiver() {
    relays_->quiesce([this](uint32_t device_index, uint32_t socket_index, RelayState state) {
        stream(device_index, socket_index).relay.store(state, std::memory_order_release);
    });
    for (auto& t : ingest_threads_) {
        t.join();
    }
    service().detach_producer(*this);
    for (uint32_t d = 0; d < devices_.size(); d++) {
        relays_->verify_completeness(d);
    }
    log_report();
    const uint64_t foreign = llrt::ZoneMetaRegistry::instance().foreign_sections();
    const uint64_t collisions = llrt::ZoneMetaRegistry::instance().collisions();
    if (collisions != 0 || foreign != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] zone names: {} id collisions, {} foreign metadata sections ignored (the JIT "
            "cache holds ELFs from a different .tt_zone_meta layout)",
            collisions,
            foreign);
    }
}

bool Receiver::poll(Stream& s) {
    // pages_available counts from the last ack, so it includes the pages already seen
    const uint64_t arrived = s.acked + s.sock->pages_available();
    TT_FATAL(
        arrived >= s.arrived,
        "streaming profiler: device {} socket {} bytes_sent went backwards ({} to {} pages)",
        s.dev,
        s.sock_idx,
        s.arrived,
        arrived);
    if (arrived == s.arrived) {
        return false;
    }
    s.arrived = arrived;
    s.arrived_bytes.store(arrived * kPageBytes, std::memory_order_release);
    s.fullest = std::max(s.fullest, arrived - s.acked);
    if (s.consumed < arrived) {
        __builtin_prefetch(s.page(s.consumed));
    }
    return true;
}

// Frames are walked one per stream per round so the header loads of a device's streams are in flight together: a
// frame's header is a dependent DRAM miss, and walked back to back they would serialize at that latency. Every
// landed page below `arrived` is a frame header or inside the frame before it: the relay notifies only bytes the
// PCIe tile has acknowledged.
bool Receiver::walk_frame(Stream& s) {
    namespace kp = kernel_profiler;
    if (s.consumed >= s.arrived) {
        return false;
    }
    const uint32_t* page = reinterpret_cast<const uint32_t*>(s.page(s.consumed));
    const uint32_t w0 = page[0];
    const uint32_t w1 = page[1];
    TT_FATAL(
        pp_is_bulkspan(w0) && w1 >= kp::SPSC_SPAN_WIRE_CTRL_WORDS && w1 <= profiler::kSpscMaxPayloadWords,
        "streaming profiler: device {} socket {} page {} is not a frame header ({:#010x} {:#010x}); {} of {} pages "
        "landed",
        s.dev,
        s.sock_idx,
        s.consumed,
        w0,
        w1,
        s.arrived - s.acked,
        s.capacity);
    const uint32_t fw = kp::spsc_span_frame_words(w1);
    const uint64_t frame_pages = fw / kPageWords;
    if (s.arrived - s.consumed < frame_pages) {
        return false;
    }
    s.frames++;
    const uint64_t start = s.consumed * kPageBytes;
    if (start / kMarkBytes != s.mark_block) {
        s.mark_block = start / kMarkBytes;
        s.marks[s.mark_block % s.marks.size()].store(start, std::memory_order_release);
    }
    s.consumed += frame_pages;
    // The next header is known; the ones after are guessed at the same stride, which a relay shipping a core per
    // sweep hits every time, so several of the walk's dependent misses are in flight instead of one. Never past
    // `arrived`: the device rewrites those pages before the walk reaches them.
    for (uint64_t p = s.consumed; p < s.arrived && p <= s.consumed + 3 * frame_pages; p += frame_pages) {
        __builtin_prefetch(s.page(p));
    }
    // Credits go back as the walk earns them: a pass over eight sockets can run long once any of them is behind, and
    // credits held until its end would let the others fill meanwhile.
    if (s.consumed - s.acked >= kAckBatchPages) {
        s.sock->pop(static_cast<uint32_t>(s.consumed - s.acked), true);
        s.acked = s.consumed;
    }
    return true;
}

bool Receiver::publish(Stream& s) {
    const uint64_t walked = s.consumed * kPageBytes;
    if (walked == s.walked_bytes.load(std::memory_order_relaxed)) {
        return false;
    }
    s.walked_bytes.store(walked, std::memory_order_release);
    return true;
}

// Credits only ever follow the walk, and a drained relay reports done only when every byte it sent is credited, so
// done means the walk has consumed every landed page.
bool Receiver::settle(Stream& s) {
    const bool published = publish(s);
    const RelayState relay = s.relay.load(std::memory_order_acquire);
    if (relay == RelayState::Running) {
        return published;
    }
    if (s.acked < s.consumed) {
        s.sock->pop(static_cast<uint32_t>(s.consumed - s.acked), true);
        s.acked = s.consumed;
    }
    s.retired = relay == RelayState::Done;
    return published;
}

uint64_t Receiver::live_head(uint32_t stream) const {
    const Stream& s = *streams_[stream];
    return widen_head(s.arrived_bytes.load(std::memory_order_acquire), s.sock->bytes_sent());
}

void Receiver::ingest_thread(std::vector<Stream*> streams) {
    std::string name = "sp-ingest:";
    for (Stream* s : streams) {
        name += std::to_string(s->dev) + "." + std::to_string(s->sock_idx) + ",";
    }
    name.pop_back();
    tracy::SetThreadName(name.c_str());
    set_os_thread_name(name);
    prctl(PR_SET_TIMERSLACK, 1000);  // default 50 us slack would round every probe sleep up to it
    // The sockets bind their FIFOs to the device's node; walked from the other node, the headers' dependent misses
    // run at half the rate and the FIFOs fill.
    if (const int node = devices_[streams.front()->dev].numa_node; node >= 0 && numa_available() != -1) {
        numa_run_on_node(node);
    }
    IdleBackoff backoff(kProbeSleepCapUs, 0);
    for (;;) {
        bool any = false;
        bool all_retired = true;
        for (Stream* s : streams) {
            if (s->retired) {
                continue;
            }
            all_retired = false;
            any |= poll(*s);
        }
        if (all_retired) {
            break;
        }
        for (bool progress = true; progress;) {
            progress = false;
            for (Stream* s : streams) {
                if (!s->retired) {
                    progress |= walk_frame(*s);
                }
            }
        }
        bool published = false;
        for (Stream* s : streams) {
            if (!s->retired) {
                published |= settle(*s);
            }
        }
        if (published) {
            service().wake_consumers();
        }
        if (any) {
            backoff.reset();
            continue;
        }
        backoff.idle();
    }
}

Receiver::Stream& Receiver::stream(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->dev == device_index && s->sock_idx == socket_index) {
            return *s;
        }
    }
    TT_THROW("streaming profiler: no stream for device {} socket {}", device_index, socket_index);
}

void Receiver::finish_stream(uint32_t stream, uint64_t dropped_bytes, const StreamStats& st) {
    std::lock_guard<std::mutex> lk(stats_mu_);
    Stream& s = *streams_[stream];
    s.consumer_dropped = std::max(s.consumer_dropped, dropped_bytes);
    StreamStats& c = s.consumer_stats;
    c.records = std::max(c.records, st.records);
    c.zones = std::max(c.zones, st.zones);
    c.order_regressions = std::max(c.order_regressions, st.order_regressions);
    c.epoch_fixes = std::max(c.epoch_fixes, st.epoch_fixes);
    c.clock_samples = std::max(c.clock_samples, st.clock_samples);
}

void Receiver::log_report() const {
    uint64_t pages = 0, frames = 0, consumer_dropped = 0;
    std::string fill;
    StreamStats t;
    for (const auto& s : streams_) {
        fill += fmt::format("{}{}%", fill.empty() ? "" : " ", s->fullest * 100 / s->capacity);
        pages += s->arrived;
        frames += s->frames;
        consumer_dropped += s->consumer_dropped;
        const StreamStats& c = s->consumer_stats;
        t.records += c.records;
        t.zones += c.zones;
        t.order_regressions += c.order_regressions;
        t.epoch_fixes += c.epoch_fixes;
        t.clock_samples += c.clock_samples;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] capture: {} frames, {:.1f} MB from {} device(s); the fullest consumer decoded {} zones, "
        "{} records; FIFO high-water marks {}",
        frames,
        pages * static_cast<double>(kPageBytes) / 1e6,
        devices_.size(),
        t.zones,
        t.records,
        fill);
    if (t.epoch_fixes != 0) {
        log_info(
            tt::LogMetal, "[streaming profiler] {} timestamps repaired for the wall-clock latch race", t.epoch_fixes);
    }
    if (t.clock_samples != 0) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] {} PP_CLOCK samples decoded from the idle-eth clock trackers",
            t.clock_samples);
    }
    if (consumer_dropped != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] {:.1f} MB of frames missed by the slowest consumer",
            consumer_dropped / 1e6);
    }
    if (t.order_regressions != 0) {
        log_warning(tt::LogMetal, "[streaming profiler] {} order regressions", t.order_regressions);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
