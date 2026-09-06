// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/streaming_profiler_receiver.hpp"

#include "tools/profiler/streaming_profiler_decode.hpp"
#include "tools/profiler/streaming_profiler_device.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <pthread.h>
#include <x86intrin.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "tt_metal/common/broadcast_ring.hpp"
#include "impl/threading/thread_pool.hpp"
#include "context/metal_context.hpp"
#include "llrt/zone_meta.hpp"
#include "tools/profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

void StallIdMirror::refresh() {
    std::vector<llrt::ZoneMetaEntry> delta;
    cursor = llrt::ZoneMetaRegistry::instance().additions_since(cursor, delta);
    bool grew = false;
    for (const auto& e : delta) {
        if (e.name == "PROFILER-STALL") {
            ids.push_back(e.zone_id);
            grew = true;
        }
    }
    if (grew) {
        // Rebuild at <= 25% load so the miss path is one probe.
        uint32_t cap = 64;
        while (cap < ids.size() * 4) {
            cap *= 2;
        }
        mask = cap - 1;
        table.assign(cap, 0xFFFFFFFFu);
        for (uint32_t id : ids) {
            uint32_t slot = (id * 0x9E3779B9u) & mask;
            while (table[slot] != 0xFFFFFFFFu && table[slot] != id) {
                slot = (slot + 1) & mask;
            }
            table[slot] = id;
        }
    }
}

namespace {

// pop+ack every 8 decoded frames (about one relay push) so the device sees credit at decode pace.
constexpr uint32_t kAckBatchPages = 8 * profiler::kSpscMaxFramePages;
// Pages peeked but not consumed are clflushed again by the next peek, so the only re-flush waste is a partial
// tail frame.
constexpr uint32_t kMaxPagesPerPass = 64 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
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
    const RecMeta meta_probe{0, 5, 2, RecType::Data};
    TT_FATAL(
        std::bit_cast<uint32_t>(meta_probe) == ((5u << 16) | (2u << 26) | (2u << 29)),
        "RecMeta bit-field layout does not match the vectorized packer");
    const auto& rtoptions = MetalContext::instance().rtoptions();
    const uint64_t ring_mb = rtoptions.get_streaming_profiler_ring_mb();
    const uint64_t ring_lines = std::bit_ceil(ring_mb << 14);
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
            s->ring_node = dev.numa_node;
            s->decode.reset(dev.num_cores);
            s->decode.core_of_xy = dev.core_of_xy;
            s->last_zone_ts.assign(nl, 0);
            streams_.push_back(std::move(s));
        }
    }
    create_rings(ring_lines);
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

Receiver::MappedRegion::~MappedRegion() {
    if (base != nullptr) {
        ::munmap(base, bytes);
    }
}

// Each ring is a private anonymous mapping bound to its device's NUMA node before the ring constructor faults
// it, so placement does not depend on which thread touches it first; the rings are constructed in parallel
// because faulting tens of GB is the slow part. Huge pages need a 2 MiB-aligned start, hence the over-map.
void Receiver::create_rings(uint64_t ring_lines) {
    using Ring = BroadcastRing<RingLine>;
    constexpr size_t kHugePage = size_t{2} << 20;
    const size_t bytes = Ring::storage_bytes(ring_lines);
    std::vector<std::thread> workers;
    workers.reserve(streams_.size());
    for (auto& s : streams_) {
        workers.emplace_back([st = s.get(), bytes, ring_lines]() {
            const size_t map_bytes = bytes + kHugePage;
            void* base = ::mmap(nullptr, map_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            TT_FATAL(base != MAP_FAILED, "streaming profiler: mmap of a {} MB ring failed", bytes >> 20);
            st->ring_storage.base = base;
            st->ring_storage.bytes = map_bytes;
            auto* aligned =
                reinterpret_cast<std::byte*>((reinterpret_cast<uintptr_t>(base) + kHugePage - 1) & ~(kHugePage - 1));
            ::madvise(aligned, bytes, MADV_HUGEPAGE);
            bind_memory_to_numa_node(aligned, bytes, st->ring_node);
            st->ring = std::make_unique<Ring>(ring_lines, std::span<std::byte>(aligned, bytes));
        });
    }
    for (auto& w : workers) {
        w.join();
    }
}

void Receiver::start() {
    const uint32_t nthreads = std::clamp<uint32_t>(
        MetalContext::instance().rtoptions().get_streaming_profiler_decode_threads(), 1, streams_.size());
    nthreads_ = nthreads;
    // The audit attaches before ingest starts so its readers see the ring from line 0.
    audit_thread_ = std::thread(&Receiver::audit_thread, this);
    for (uint32_t t = 0; t < nthreads; t++) {
        std::vector<Stream*> owned;
        for (uint32_t i = t; i < streams_.size(); i += nthreads) {
            owned.push_back(streams_[i].get());
        }
        decode_threads_.emplace_back(&Receiver::decode_thread, this, std::move(owned));
    }
}

namespace {
// Whole aligned lines, one NT store each: the ring is written at wire rate and read back cold, so cached
// stores would pollute both sides (memcpy measured 28% slower per thread). The distance prefetch runs past the
// run's end into the pages behind it in the FIFO.
__attribute__((target("avx512f"))) void ingest_copy_lines_512(std::byte* dst, const uint32_t* src, uint32_t nlines) {
    for (uint32_t k = 0; k < nlines; k++) {
        _mm_prefetch(reinterpret_cast<const char*>(src + 16ull * k) + 4096, _MM_HINT_T0);
        _mm512_stream_si512(reinterpret_cast<__m512i*>(dst + 64ull * k), _mm512_loadu_si512(src + 16ull * k));
    }
}
void ingest_copy_lines_256(std::byte* dst, const uint32_t* src, uint32_t nlines) {
    for (uint32_t k = 0; k < nlines; k++) {
        _mm_prefetch(reinterpret_cast<const char*>(src + 16ull * k) + 4096, _MM_HINT_T0);
        const uint8_t* line = reinterpret_cast<const uint8_t*>(src) + 64ull * k;
        std::byte* out = dst + 64ull * k;
        _mm256_stream_si256(
            reinterpret_cast<__m256i*>(out), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(line)));
        _mm256_stream_si256(
            reinterpret_cast<__m256i*>(out + 32), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(line + 32)));
    }
}
// Lines [first, first + count) of the peeked pages, which may span the FIFO's own wrap, into contiguous dst.
template <typename Segments>
void ingest_copy_lines(std::byte* dst, const Segments& segments, size_t first, size_t count, bool use512) {
    size_t seg_first = 0;
    for (const std::span<const uint32_t> segment : segments) {
        const size_t seg_lines = segment.size() / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
        const size_t lo = std::max(first, seg_first);
        const size_t hi = std::min(first + count, seg_first + seg_lines);
        if (lo < hi) {
            const uint32_t* src = segment.data() + (lo - seg_first) * kernel_profiler::SPSC_SPAN_PAGE_WORDS;
            std::byte* out = dst + (lo - first) * 64;
            if (use512) {
                ingest_copy_lines_512(out, src, static_cast<uint32_t>(hi - lo));
            } else {
                ingest_copy_lines_256(out, src, static_cast<uint32_t>(hi - lo));
            }
        }
        seg_first += seg_lines;
    }
}
}  // namespace

bool Receiver::ingest_pass(Stream& s) {
    const uint32_t avail = s.sock->pages_available();
    if (avail == 0) {
        if (s.producers_done.load(std::memory_order_acquire)) {
            s.retired = true;
        }
        return false;
    }

    const uint32_t np = std::min(avail, kMaxPagesPerPass);
    const uint64_t t0 = tsc_now();
    auto& w = s.ring->writer();
    // A FIFO page is a ring line, so the pages stream straight into the ring, a chunk at a time. Frames start on a
    // line and never straddle the FIFO wrap, so each header is read from the FIFO source right after its lines
    // are copied, while they are still cached; the ring copy is never read back. Credit goes back per chunk.
    const bool use512 = profiler::spsc_host_avx512();
    const uint64_t end = s.wpos + np;
    uint64_t lpos = s.frame_end;  // next frame boundary, possibly beyond the lines written so far
    uint32_t frames = 0;
    for (uint64_t pos = s.wpos; pos < end;) {
        const uint32_t pages = std::min<uint32_t>(end - pos, kAckBatchPages);
        const auto segments = s.sock->peek(pages).base();
        w.publish_direct(pages, [&](std::span<std::byte> run, size_t first) {
            ingest_copy_lines(run.data(), segments, first, run.size() / sizeof(RingLine), use512);
        });
        for (const std::span<const uint32_t> segment : segments) {
            const uint32_t nlines = segment.size() / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
            const uint64_t seg_end = pos + nlines;
            while (lpos < seg_end) {
                const uint32_t* hdr = segment.data() + (lpos - pos) * kernel_profiler::SPSC_SPAN_PAGE_WORDS;
                const uint32_t w1 = hdr[1];
                if (!pp_is_bulkspan(hdr[0]) || w1 < kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS ||
                    w1 > profiler::kSpscMaxPayloadWords) {
                    // Framing is lost; the line is in the ring anyway and the consumers' resync skips it.
                    s.bad_frames++;
                    lpos++;
                    continue;
                }
                lpos += kernel_profiler::spsc_span_frame_words(w1) / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
                frames++;
            }
            pos = seg_end;
        }
        s.sock->pop(pages, true);
    }
    s.wpos = end;
    s.frame_end = lpos;
    s.decode_ticks += tsc_now() - t0;
    s.pages += np;
    s.frames += frames;
    if (frames == 0 && s.producers_done.load(std::memory_order_acquire)) {
        if (!s.desync_warned && lpos > end) {
            s.desync_warned = true;
            log_warning(
                tt::LogMetal,
                "[streaming profiler receiver] d{}/s{}: the last frame is {} lines short after the relays "
                "finished -- flow control desynchronized",
                s.dev,
                s.sock_idx,
                lpos - end);
        }
        s.retired = true;
        return false;
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

// Baseline code: no AVX-512 target attribute here, or the compiler could emit it into paths that run on any
// host. The 512-bit kernels are separate attributed functions gated on spsc_host_avx512().

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
    std::vector<RingLine> lines(kConsumerLineBatch);
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
            any |= walkers[i].pass(readers[i], lines, decs[i], [&](uint64_t) { publish(i); });
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
    uint64_t pages = 0, wire_words = 0, zones = 0, records = 0, frames = 0;
    uint64_t resync_words = 0, order_regressions = 0, bad_frames = 0, anomalies = 0, unknown_core_frames = 0;
    uint64_t epoch_fixes = 0;
    // Busy is the busiest thread, so ticks group by owning thread (stream i -> thread i % nthreads_); with fewer
    // threads than sockets a per-stream max understates busy.
    std::vector<uint64_t> thread_ticks(std::max<uint32_t>(nthreads_, 1), 0);
    for (size_t i = 0; i < streams_.size(); i++) {
        const Stream& s = *streams_[i];
        pages += s.pages;
        wire_words += s.decode.live_words;
        zones += s.zones;
        records += s.records;
        frames += s.frames;
        resync_words += s.decode.resync_words;
        order_regressions += s.order_regressions;
        epoch_fixes += s.epoch_fixes;
        bad_frames += s.bad_frames;
        anomalies += s.decode.anomalies;
        unknown_core_frames += s.decode.unknown_core_frames;
        thread_ticks[i % thread_ticks.size()] += s.decode_ticks;
    }
    const double busy_ms = ticks_to_ms(*std::max_element(thread_ticks.begin(), thread_ticks.end()));
    auto rate = [busy_ms](double num) { return busy_ms > 0.0 ? num / (busy_ms / 1e3) : 0.0; };
    log_info(
        tt::LogMetal,
        "[streaming profiler] capture: {} zones, {} records, {:.1f} MB from {} device(s)",
        zones,
        records,
        pages * static_cast<double>(kPageBytes) / 1e6,
        devices_.size());
    log_debug(
        tt::LogMetal,
        "[streaming profiler] ingest: {} frames, busy {:.1f} ms -> {:.2f} GB/s D2H, {:.2f} GB/s wire, {:.2f} Mzones/s",
        frames,
        busy_ms,
        rate(pages * static_cast<double>(kPageBytes) / 1e9),
        rate(wire_words * 4.0 / 1e9),
        rate(zones / 1e6));
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
