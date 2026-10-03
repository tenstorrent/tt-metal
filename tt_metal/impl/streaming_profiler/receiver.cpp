// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/receiver.hpp"

#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>
#include <sys/prctl.h>
#include <numa.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "context/metal_context.hpp"
#include "llrt/zone_meta.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

// Credits go back about once per relay push rather than per poll.
constexpr uint32_t kAckBatchPages = 8 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4;
// Idle probe period: credits are returned as soon as pages are walked, so a sleep only delays the frames that land
// during it, never a credit the device is short of (the FIFO holds over a millisecond of egress).
constexpr uint32_t kProbeSleepCapUs = 100;

}  // namespace

Receiver::Receiver(tt::Cluster& cluster, std::unique_ptr<DevicePrograms> programs) :
    programs_(std::move(programs)),
    host_sync_(cluster, programs_->capture_context().devices[CaptureContext::kRootDevice].chip_id, service().steady()),
    map_(programs_->capture_context().devices.size(), ClockMap::kSeriesNodes, host_sync_.bases()),
    sync_(programs_->capture_context(), map_) {
    for (const CapturedSocket& captured : programs_->sockets()) {
        auto stream = std::make_unique<Stream>();
        stream->captured = captured;
        stream->fifo = captured.socket->host_fifo();
        stream->capacity_pages = stream->fifo.size() / kPageBytes;
        stream->marks = std::vector<std::atomic<uint64_t>>(stream->fifo.size() / kMarkBytes);
        for (auto& mark : stream->marks) {
            mark.store(UINT64_MAX, std::memory_order_relaxed);
        }
        streams_view_.push_back(ReceiverStream{
            .fifo = stream->fifo,
            .walked = &stream->walked_bytes,
            .dev = captured.dev,
            .marks = stream->marks,
            .sync = captured.sync});
        streams_.push_back(std::move(stream));
    }
}

std::unique_ptr<Receiver> Receiver::create(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto& mc = MetalContext::instance(mesh_device->impl().get_context_id());
    auto programs = std::make_unique<DevicePrograms>();
    if (!programs->boot(mesh_device)) {
        return nullptr;
    }
    service().register_builtin_consumers(mc.rtoptions());
    std::unique_ptr<Receiver> receiver(new Receiver(mc.get_cluster(), std::move(programs)));
    service().attach_receiver(*receiver);
    for (uint32_t d = 0; d < receiver->capture_context().devices.size(); d++) {
        std::vector<Stream*> owned;
        for (auto& s : receiver->streams_) {
            if (s->captured.dev == d) {
                owned.push_back(s.get());
            }
        }
        if (!owned.empty()) {
            receiver->ingest_threads_.emplace_back(&Receiver::ingest_thread, receiver.get(), std::move(owned));
        }
    }
    receiver->programs_->start();
    return receiver;
}

Receiver::~Receiver() {
    programs_->quiesce([this](uint32_t device_index, uint32_t socket_index, RelayState state) {
        stream(device_index, socket_index).relay.store(state, std::memory_order_release);
    });
    for (auto& t : ingest_threads_) {
        t.join();
    }
    service().detach_receiver(*this);
    programs_->verify_completeness();
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
    const uint64_t arrived = s.acked_pages + s.captured.socket->pages_available();
    if (arrived == s.arrived_pages) {
        return false;
    }
    s.arrived_pages = arrived;
    s.arrived_bytes.store(arrived * kPageBytes, std::memory_order_release);
    s.fullest_pages = std::max(s.fullest_pages, arrived - s.acked_pages);
    if (s.walked_pages < arrived) {
        __builtin_prefetch(s.page(s.walked_pages));
    }
    return true;
}

// Frames are walked one per stream per round so the header loads of a device's streams are in flight together: a
// frame's header is a dependent DRAM miss, and walked back to back they would serialize at that latency. Every
// landed page below `arrived_pages` is a frame header or inside the frame before it: the relay notifies only bytes the
// PCIe tile has acknowledged.
bool Receiver::walk_frame(Stream& s) {
    namespace kp = kernel_profiler;
    if (s.walked_pages >= s.arrived_pages) {
        return false;
    }
    const uint32_t* page = reinterpret_cast<const uint32_t*>(s.page(s.walked_pages));
    const uint32_t w0 = page[0];
    const uint32_t payload_words = page[kp::SPSC_PREFIX_PAYLOAD_WORDS];
    TT_FATAL(
        pp_is_bulkspan(w0) && payload_words >= kp::SPSC_SPAN_WIRE_CTRL_WORDS &&
            payload_words <= profiler::kSpscMaxPayloadWords,
        "streaming profiler: device {} socket {} page {} is not a frame header ({:#010x} {:#010x}); {} of {} pages "
        "landed",
        s.captured.dev,
        s.captured.index,
        s.walked_pages,
        w0,
        payload_words,
        s.arrived_pages - s.acked_pages,
        s.capacity_pages);
    const uint32_t fw = kp::spsc_span_frame_words(payload_words);
    const uint64_t frame_pages = fw / kPageWords;
    if (s.arrived_pages - s.walked_pages < frame_pages) {
        return false;
    }
    s.frames++;
    const uint64_t start = s.walked_pages * kPageBytes;
    if (start / kMarkBytes != s.mark_block) {
        s.mark_block = start / kMarkBytes;
        s.marks[s.mark_block % s.marks.size()].store(start, std::memory_order_release);
    }
    s.walked_pages += frame_pages;
    // The next header is known; the ones after are guessed at the same stride, which a relay shipping a core per
    // sweep hits every time, so several of the walk's dependent misses are in flight instead of one. Never past
    // `arrived_pages`: the device rewrites those pages before the walk reaches them.
    for (uint64_t next = s.walked_pages; next < s.arrived_pages && next <= s.walked_pages + 3 * frame_pages;
         next += frame_pages) {
        __builtin_prefetch(s.page(next));
    }
    // Credits go back as the walk earns them: a pass over eight sockets can run long once any of them is behind, and
    // credits held until its end would let the others fill meanwhile.
    if (s.walked_pages - s.acked_pages >= kAckBatchPages) {
        s.captured.socket->pop(static_cast<uint32_t>(s.walked_pages - s.acked_pages), true);
        s.acked_pages = s.walked_pages;
    }
    return true;
}

// Credits only ever follow the walk, and a drained relay reports done only when every byte it sent is credited, so
// done means the walk has consumed every landed page.
bool Receiver::settle(Stream& s) {
    const uint64_t walked = s.walked_pages * kPageBytes;
    const bool published = walked != s.walked_bytes.load(std::memory_order_relaxed);
    if (published) {
        s.walked_bytes.store(walked, std::memory_order_release);
    }
    const RelayState relay = s.relay.load(std::memory_order_acquire);
    if (relay == RelayState::Running) {
        return published;
    }
    if (s.acked_pages < s.walked_pages) {
        s.captured.socket->pop(static_cast<uint32_t>(s.walked_pages - s.acked_pages), true);
        s.acked_pages = s.walked_pages;
    }
    s.retired = relay == RelayState::Done;
    return published;
}

uint64_t Receiver::live_head(uint32_t stream) const {
    const Stream& s = *streams_[stream];
    return widen_head(s.arrived_bytes.load(std::memory_order_acquire), s.captured.socket->bytes_sent());
}

void Receiver::ingest_thread(std::vector<Stream*> streams) {
    std::string name = "sp-ingest:";
    for (Stream* s : streams) {
        name += std::to_string(s->captured.dev) + "." + std::to_string(s->captured.index) + ",";
    }
    name.pop_back();
    set_thread_name(name);
    prctl(PR_SET_TIMERSLACK, 1000);  // default 50 us slack would round every probe sleep up to it
    // The sockets bind their FIFOs to the device's node; walked from the other node, the headers' dependent misses
    // run at half the rate and the FIFOs fill.
    if (const int node = streams.front()->captured.numa_node; node >= 0 && numa_available() != -1) {
        numa_run_on_node(node);
    }
    uint32_t sleep_us = 1;
    while (true) {
        bool any = false;
        for (Stream* s : streams) {
            any |= poll(*s);
        }
        for (bool progress = true; progress;) {
            progress = false;
            for (Stream* s : streams) {
                progress |= walk_frame(*s);
            }
        }
        bool published = false;
        for (Stream* s : streams) {
            published |= settle(*s);
        }
        if (published) {
            service().wake_walkers();
        }
        std::erase_if(streams, [](const Stream* stream) { return stream->retired; });
        if (streams.empty()) {
            break;
        }
        if (any) {
            sleep_us = 1;
            continue;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        sleep_us = std::min(sleep_us + sleep_us / 4 + 1, kProbeSleepCapUs);
    }
}

Receiver::Stream& Receiver::stream(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->captured.dev == device_index && s->captured.index == socket_index) {
            return *s;
        }
    }
    TT_THROW("streaming profiler: no stream for device {} socket {}", device_index, socket_index);
}

void Receiver::log_report() const {
    uint64_t pages = 0, frames = 0;
    std::string fill;
    for (const auto& s : streams_) {
        fill += fmt::format("{}{}%", fill.empty() ? "" : " ", s->fullest_pages * 100 / s->capacity_pages);
        pages += s->arrived_pages;
        frames += s->frames;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] capture: {} frames, {:.1f} MB from {} device(s); FIFO high-water marks {}",
        frames,
        pages * static_cast<double>(kPageBytes) / 1e6,
        capture_context().devices.size(),
        fill);
}

}  // namespace tt::tt_metal::streaming_profiler
