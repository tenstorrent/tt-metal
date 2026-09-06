// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_receiver.hpp"

#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_device.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>
#include <sys/prctl.h>
#include <pthread.h>
#include <x86intrin.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

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

double tsc_ns_per_tick() {
    static const double v = [] {
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t c0 = __rdtsc();
        while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(20)) {
        }
        const uint64_t c1 = __rdtsc();
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        return c1 > c0 ? ns / static_cast<double>(c1 - c0) : 1.0;
    }();
    return v;
}

Receiver::Receiver(std::vector<ReceiverDeviceConfig> devices) : devices_(std::move(devices)) {
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
        const uint32_t nl = dev.num_cores * profiler::kSpscNRiscDecode;
        TT_FATAL(
            nl <= kStreamingProfilerMaxLanes,
            "record lane field holds {} lanes, device has {}",
            kStreamingProfilerMaxLanes,
            nl);
        TT_FATAL(dev.lane_table.size() == nl, "lane table size mismatch");
        auto& cd = ctx_.devices.emplace_back();
        cd.chip_id = dev.chip_id;
        clocks_.push_back(dev.clock);
        cd.lanes = dev.lane_table;
        cd.core_of_xy = dev.core_of_xy;
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
            s->decode.reset(dev.num_cores);
            s->decode.core_of_xy.load(dev.core_of_xy);
            s->last_zone_ts.assign(nl, 0);
            streams_.push_back(std::move(s));
        }
    }
    for (const auto& st : streams_) {
        streams_view_.push_back({st->ring.get(), st->dev});
    }
}

std::unique_ptr<Receiver> Receiver::create(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto relays = std::make_unique<Devices>();
    std::vector<ReceiverDeviceConfig> configs;
    try {
        configs = relays->boot(mesh_device);
    } catch (const std::exception& e) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] init failed at step [{}] ({}); disabled for this session.",
            bringup_step(),
            e.what());
        relays->quiesce(nullptr);
        return nullptr;
    }
    if (configs.empty()) {
        return nullptr;
    }
    std::unique_ptr<Receiver> receiver(new Receiver(std::move(configs)));
    receiver->relays_ = std::move(relays);
    service().register_builtin_consumers(MetalContext::instance(mesh_device->impl().get_context_id()).rtoptions());
    service().attach_producer(*receiver);
    receiver->attached_ = true;
    receiver->start();
    return receiver;
}

Receiver::~Receiver() { stop(); }

void Receiver::stop() {
    if (relays_ != nullptr) {
        relays_->quiesce(this);
    }
    shutdown();
    if (attached_) {
        service().detach_producer(*this);
        attached_ = false;
    }
    if (relays_ != nullptr) {
        relays_->verify(*this);
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
        relays_.reset();
    }
}

void Receiver::start() {
    // The audit attaches before ingest starts so its readers see the ring from line 0.
    audit_thread_ = std::thread(&Receiver::audit_thread, this);
    for (uint32_t d = 0; d < devices_.size(); d++) {
        std::vector<Stream*> owned;
        for (auto& s : streams_) {
            if (s->dev == d) {
                owned.push_back(s.get());
            }
        }
        if (!owned.empty()) {
            decode_threads_.emplace_back(&Receiver::decode_thread, this, std::move(owned));
        }
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
    s.pages = arrived;
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

void Receiver::decode_thread(std::vector<Stream*> streams) {
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

std::vector<experimental::streaming_profiler::Clock> Receiver::clocks() const {
    std::lock_guard<std::mutex> lk(clocks_mu_);
    return clocks_;
}

void Receiver::audit_thread() {
    tracy::SetThreadName("sp-audit");
    set_os_thread_name("sp-audit");
    std::vector<BroadcastRing<RingLine>::Reader> readers;
    readers.reserve(streams_.size());
    for (auto& s : streams_) {
        readers.push_back(s->ring->make_reader());
    }
    std::vector<FrameWalker> walkers(streams_.size());
    std::vector<StreamDecoder<profiler::SpscNullRecSink>> decs(streams_.size());
    for (size_t i = 0; i < streams_.size(); i++) {
        decs[i].st = &streams_[i]->decode;
        decs[i].last_ts = streams_[i]->last_zone_ts.data();
        decs[i].dev = streams_[i]->dev;
    }
    auto publish = [&](size_t i) {
        Stream& s = *streams_[i];
        const auto& dec = decs[i];
        s.records = dec.recs;
        s.zones = dec.zones;
        s.order_regressions = dec.order_regressions;
        s.epoch_fixes = dec.epoch_fixes;
    };
    auto pass_all = [&] {
        bool any = false;
        for (size_t i = 0; i < readers.size(); i++) {
            any |= walkers[i].pass(readers[i], decs[i], [&](uint64_t) { publish(i); });
        }
        return any;
    };
    IdleBackoff backoff(0);
    for (;;) {
        const uint32_t seen = service().wake_token();
        if (pass_all()) {
            backoff.reset();
            continue;
        }
        if (audit_drain_.load(std::memory_order_acquire)) {
            break;
        }
        if (backoff.spin()) {
            continue;
        }
        service().wait_wake(seen);
    }
    for (const auto& r : readers) {
        audit_dropped_ += r.dropped();
    }
}

void Receiver::notify_producers_drained(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->dev == device_index && s->sock_idx == socket_index) {
            s->drained.store(true, std::memory_order_release);
        }
    }
}

void Receiver::notify_producers_done(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->dev == device_index && s->sock_idx == socket_index) {
            s->producers_done.store(true, std::memory_order_release);
        }
    }
}

void Receiver::shutdown() {
    if (shutdown_done_.exchange(true)) {
        return;
    }
    stop_.store(true, std::memory_order_release);
    for (auto& t : decode_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    audit_drain_.store(true, std::memory_order_release);
    service().wake_consumers();
    if (audit_thread_.joinable()) {
        audit_thread_.join();
    }
}

std::vector<uint32_t> Receiver::final_lane_heads(uint32_t device_index) const {
    const uint32_t nl = devices_[device_index].num_cores * profiler::kSpscNRiscDecode;
    std::vector<uint32_t> heads(nl, 0);
    for (const auto& s : streams_) {
        if (s->dev != device_index) {
            continue;
        }
        for (uint32_t l = 0; l < nl; l++) {
            if (s->decode.seeded[l] != 0) {
                heads[l] = std::max(heads[l], s->decode.head[l]);
            }
        }
    }
    return heads;
}

void Receiver::log_report() const {
    uint64_t pages = 0, zones = 0, records = 0;
    uint64_t resync_words = 0, order_regressions = 0, bad_frames = 0, anomalies = 0, unknown_core_frames = 0;
    uint64_t epoch_fixes = 0;
    for (size_t i = 0; i < streams_.size(); i++) {
        const Stream& s = *streams_[i];
        pages += s.pages;
        zones += s.zones;
        records += s.records;
        resync_words += s.decode.resync_words;
        order_regressions += s.order_regressions;
        epoch_fixes += s.epoch_fixes;
        bad_frames += s.bad_frames;
        anomalies += s.decode.anomalies;
        unknown_core_frames += s.decode.unknown_core_frames;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] capture: {} zones, {} records, {:.1f} MB from {} device(s)",
        zones,
        records,
        pages * static_cast<double>(kPageBytes) / 1e6,
        devices_.size());
    if (epoch_fixes != 0) {
        log_info(
            tt::LogMetal, "[streaming profiler] {} timestamps repaired for the wall-clock latch race", epoch_fixes);
    }
    if (audit_dropped_ != 0 || resync_words != 0 || order_regressions != 0 || bad_frames != 0 || anomalies != 0 ||
        unknown_core_frames != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] capture loss: {} lines dropped by the audit, {} resync words, {} order regressions, "
            "{} bad frames, {} anomalies, {} unknown-core frames",
            audit_dropped_,
            resync_words,
            order_regressions,
            bad_frames,
            anomalies,
            unknown_core_frames);
    }
}

}  // namespace tt::tt_metal::streaming_profiler
