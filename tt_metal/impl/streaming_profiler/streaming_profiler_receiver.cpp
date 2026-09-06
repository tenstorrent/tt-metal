// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_receiver.hpp"

#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>
#include <sys/prctl.h>
#include <pthread.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>
#include <umd/device/driver_atomics.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "tt_metal/common/broadcast_ring.hpp"
#include "context/metal_context.hpp"
#include "llrt/zone_meta.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

// Credits go back about once per relay push rather than per poll.
constexpr uint32_t kAckBatchPages = 8 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
// 64 MiB is ~1.6 ms of device egress at 40 GB/s: the ingest stall the device rides out before it backpressures.
constexpr uint64_t kRunwayPages = (64ull << 20) / kPageBytes;
// Idle probe period. Under ~50 us sleep_for rounds up unless the timer slack is shrunk; raising it to 200 us
// doubled the relays' worst credit wait, since no credit returns during the sleep.
constexpr uint32_t kProbeSleepCapUs = 5;

}  // namespace

void set_os_thread_name(const std::string& n) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", n.c_str());
    pthread_setname_np(pthread_self(), buf);
}

Receiver::Receiver(std::unique_ptr<Devices> relays, std::vector<CapturedDevice> devices) :
    relays_(std::move(relays)), devices_(std::move(devices)) {
    TT_FATAL(
        devices_.size() <= kStreamingProfilerMaxDevices,
        "record dev field holds {} devices",
        kStreamingProfilerMaxDevices);
    // The scalar decode packs meta through the bit-field; the vector paths pack it by hand, so pin the layout.
    static_assert(static_cast<uint32_t>(RecType::Zone) == profiler::kSpscRecTypeZone);
    static_assert(static_cast<uint32_t>(RecType::Data) == profiler::kSpscRecTypeData);
    static_assert(static_cast<uint32_t>(RecType::Event) == profiler::kSpscRecTypeEvent);
    static_assert(static_cast<uint32_t>(RecType::Ext) == profiler::kSpscRecTypeExt);
    static_assert(static_cast<uint32_t>(RecType::Cont) == profiler::kSpscRecTypeCont);
    const RecMeta meta_probe{0, 5, 2, RecType::Data};
    TT_FATAL(
        std::bit_cast<uint32_t>(meta_probe) == ((5u << 16) | (2u << 26) | (2u << 29)),
        "RecMeta bit-field layout does not match the vectorized packer");
    for (uint32_t d = 0; d < devices_.size(); d++) {
        auto& dev = devices_[d];
        TT_FATAL(
            dev.ctx.lanes.size() <= kStreamingProfilerMaxLanes,
            "record lane field holds {} lanes, device has {}",
            kStreamingProfilerMaxLanes,
            dev.ctx.lanes.size());
        ctx_.devices.push_back(dev.ctx);
        clocks_.push_back(dev.clock);
        for (uint32_t sk = 0; sk < dev.sockets.size(); sk++) {
            auto s = std::make_unique<Stream>();
            s->sock = dev.sockets[sk].get();
            s->dev = d;
            s->sock_idx = sk;
            const std::span<std::byte> fifo = s->sock->host_fifo();
            TT_FATAL(
                s->sock->get_fifo_curr_size() == fifo.size() && fifo.size() % kPageBytes == 0,
                "streaming profiler: the host FIFO must be a whole number of pages");
            s->capacity = fifo.size() / kPageBytes;
            TT_FATAL(
                std::has_single_bit(s->capacity),
                "TT_METAL_STREAMING_PROFILER_FIFO_MB must be a power of two: the FIFO is the frame ring");
            s->keep = s->capacity - std::min<uint64_t>(kRunwayPages, s->capacity / 4);
            s->claim = s->capacity;
            s->ring =
                std::make_unique<BroadcastRing<RingLine>>(s->capacity, fifo, BroadcastRing<RingLine>::AdoptStorage{});
            streams_.push_back(std::move(s));
        }
    }
    for (const auto& st : streams_) {
        streams_view_.push_back({st->ring.get(), st->dev});
    }
}

std::unique_ptr<Receiver> Receiver::create(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto relays = std::make_unique<Devices>();
    std::vector<CapturedDevice> devices;
    try {
        devices = relays->boot(mesh_device);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] init failed at step [{}] ({}); disabled for this session.",
            bringup_step(),
            e.what());
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
        Stream& s = stream(device_index, socket_index);
        (state == RelayState::Done ? s.producers_done : s.drained).store(true, std::memory_order_release);
    });
    stop_.store(true, std::memory_order_release);
    for (auto& t : ingest_threads_) {
        t.join();
    }
    service().detach_producer(*this);
    for (uint32_t d = 0; d < devices_.size(); d++) {
        relays_->verify_completeness(d, final_lane_heads(d));
    }
    log_report();
    const auto zm = llrt::ZoneMetaRegistry::instance().stats();
    const uint64_t collisions = llrt::ZoneMetaRegistry::instance().collisions();
    if (collisions != 0 || zm.foreign_sections != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] zone names: {} id collisions, {} foreign metadata sections ignored (the JIT "
            "cache holds ELFs from a different .tt_zone_meta layout)",
            collisions,
            zm.foreign_sections);
    }
}

// The device is the ring's writer. Its progress is the socket's bytes_sent; the pages it may still write or
// overwrite are bounded by the credits the host returned, so the ring's horizon is acked + capacity and credits are
// returned only up to arrived - keep. That keeps `keep` pages behind the device intact for lagging readers while
// the device always has the runway ahead. Once the relay has published drained it writes nothing more, so everything
// is acked for its socket barrier and the horizon stays where it is.
bool Receiver::ingest_pass(Stream& s) {
    // pages_available counts from the last ack, so it includes the pages already published
    const uint64_t arrived = s.acked + s.sock->pages_available();
    const bool drained = s.drained.load(std::memory_order_acquire);
    if (arrived == s.arrived) {
        if (drained && s.acked < arrived) {
            s.sock->pop(static_cast<uint32_t>(arrived - s.acked), true);
            s.acked = arrived;
        }
        if (s.producers_done.load(std::memory_order_acquire)) {
            s.retired = true;
        }
        return false;
    }
    s.arrived = arrived;
    auto& w = s.ring->writer();
    if (drained) {
        w.publish_external(arrived, s.claim);
        s.sock->pop(static_cast<uint32_t>(arrived - s.acked), true);
        s.acked = arrived;
        return true;
    }
    // The credit write may go through a write-combining PCIe window, which x86 does not order behind the cached
    // claim store; the sfence keeps the horizon visible before the device can act on the credit.
    const uint64_t ack_to = arrived > s.keep ? arrived - s.keep : 0;
    if (ack_to - s.acked >= kAckBatchPages) {
        s.claim = ack_to + s.capacity;
        w.publish_external(arrived, s.claim);
        tt_driver_atomics::sfence();
        s.sock->pop(static_cast<uint32_t>(ack_to - s.acked), true);
        s.acked = ack_to;
    } else {
        w.publish_external(arrived, s.claim);
    }
    return true;
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
    IdleBackoff backoff(kProbeSleepCapUs);
    std::chrono::steady_clock::time_point stop_deadline{};
    for (;;) {
        bool any = false;
        bool all_retired = true;
        for (Stream* s : streams) {
            if (s->retired) {
                continue;
            }
            all_retired = false;
            if (ingest_pass(*s)) {
                any = true;
            }
        }
        if (all_retired) {
            break;
        }
        if (any) {
            service().wake_consumers();
            backoff.reset();
            continue;
        }
        if (stop_.load(std::memory_order_acquire)) {
            if (stop_deadline == std::chrono::steady_clock::time_point{}) {
                stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            }
            if (std::chrono::steady_clock::now() >= stop_deadline) {
                break;
            }
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

void Receiver::finish_stream(uint32_t stream, const StreamStats& stats) {
    std::lock_guard<std::mutex> lk(stats_mu_);
    Stream& s = *streams_[stream];
    if (stats.records >= s.stats.records) {
        s.stats = stats;
    }
}

std::vector<uint32_t> Receiver::final_lane_heads(uint32_t device_index) const {
    const size_t nl = ctx_.devices[device_index].lanes.size();
    std::vector<uint32_t> heads;
    for (const auto& s : streams_) {
        if (s->dev != device_index || s->stats.heads.empty()) {
            continue;
        }
        heads.resize(nl, 0);
        for (size_t l = 0; l < nl && l < s->stats.heads.size(); l++) {
            heads[l] = std::max(heads[l], s->stats.heads[l]);
        }
    }
    return heads;  // empty when no subscriber decoded the device: nothing to check completeness against
}

void Receiver::log_report() const {
    uint64_t pages = 0;
    bool decoded = false;
    StreamStats t;
    for (const auto& s : streams_) {
        pages += s->arrived;
        decoded |= !s->stats.heads.empty();
        t.records += s->stats.records;
        t.zones += s->stats.zones;
        t.order_regressions += s->stats.order_regressions;
        t.bad_frames += s->stats.bad_frames;
        t.epoch_fixes += s->stats.epoch_fixes;
        t.resync_words += s->stats.resync_words;
        t.anomalies += s->stats.anomalies;
        t.unknown_core_frames += s->stats.unknown_core_frames;
        t.dropped += s->stats.dropped;
    }
    if (!decoded) {
        log_info(
            tt::LogMetal,
            "[streaming profiler] capture: {:.1f} MB from {} device(s); no subscriber decoded it",
            pages * static_cast<double>(kPageBytes) / 1e6,
            devices_.size());
        return;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] capture: {} zones, {} records, {:.1f} MB from {} device(s)",
        t.zones,
        t.records,
        pages * static_cast<double>(kPageBytes) / 1e6,
        devices_.size());
    if (t.epoch_fixes != 0) {
        log_info(
            tt::LogMetal, "[streaming profiler] {} timestamps repaired for the wall-clock latch race", t.epoch_fixes);
    }
    if (t.dropped != 0 || t.resync_words != 0 || t.order_regressions != 0 || t.bad_frames != 0 || t.anomalies != 0 ||
        t.unknown_core_frames != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] capture loss: {} lines dropped by the reporting subscriber, {} resync words, {} "
            "order "
            "regressions, {} bad frames, {} anomalies, {} unknown-core frames",
            t.dropped,
            t.resync_words,
            t.order_regressions,
            t.bad_frames,
            t.anomalies,
            t.unknown_core_frames);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
