// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/streaming_profiler_receiver.hpp"

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
#include "impl/threading/thread_pool.hpp"
#include "llrt/zone_meta.hpp"
#include "tools/profiler/streaming_profiler_env.hpp"
#include "tools/profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

void StallIdMirror::refresh() {
    std::vector<llrt::ZoneMetaEntry> delta;
    cursor = llrt::ZoneMetaRegistry::instance().additions_since(cursor, delta);
    bool grew = false;
    for (const auto& e : delta) {
        if (e.name == "PRODUCER-STALL") {
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

// Credit-return quantum: pop+ack every 8 decoded frames (about one relay push) rather than once per
// pass, so the device sees credit at decode pace.
constexpr uint32_t kAckBatchFrames = 8;
// Per-pass peek window (~680 KB). Pages peeked but not consumed are clflushed again by the next peek,
// so the only re-flush waste is a partial tail frame.
constexpr uint32_t kMaxPagesPerPass = 64 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr size_t kConsumerScratchRecs = 1 << 16;
// Ring lines a consumer pulls per read (256 KB scratch), and the records one frame can decode to at
// most (payload <= 2640 words, 2 words per record, plus DATA head/ext expansion slack).
constexpr size_t kConsumerLineBatch = 1 << 12;
constexpr size_t kMaxFrameRecs = 2048;
constexpr uint32_t kEmptyPollsBeforeSleep = 1000;
// Decode threads probe the FIFO at least this often when idle; consumers are latency-tolerant and may
// sleep longer. Anything under ~50 us needs the timer slack shrunk or sleep_for quietly rounds up to it.
// Deliberately tiny despite the wakeup cost: the sleep is a window in which no credit is returned at all,
// and raising it to 200 us to reclaim cores doubled the relays' worst credit wait and multiplied producer
// stalls several-fold.
constexpr uint32_t kProbeSleepCapUs = 5;

namespace {
void set_os_thread_name(const std::string& n) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", n.c_str());
    pthread_setname_np(pthread_self(), buf);
}
}  // namespace

constexpr uint32_t kConsumerSleepCapUs = 100;

inline uint64_t tsc_now() { return __rdtsc(); }

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

double ticks_to_ms(uint64_t ticks) { return ticks * tsc_ns_per_tick() / 1e6; }

thread_local bool t_in_consumer = false;

struct IdleBackoff {
    uint32_t cap_us;
    uint32_t empty_polls = 0;
    uint32_t sleep_us = 1;
    explicit IdleBackoff(uint32_t cap) : cap_us(cap) {}
    void idle() {
        if (++empty_polls < kEmptyPollsBeforeSleep) {
            ttsl::pause();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            sleep_us = std::min(sleep_us + sleep_us / 4 + 1, cap_us);
        }
    }
    void reset() {
        empty_polls = 0;
        sleep_us = 1;
    }
};

}  // namespace

StreamingProfilerReceiver::StreamingProfilerReceiver(std::vector<ReceiverDeviceConfig> devices) :
    devices_(std::move(devices)) {
    TT_FATAL(
        devices_.size() <= kStreamingProfilerMaxDevices,
        "record dev field holds {} devices",
        kStreamingProfilerMaxDevices);
    // The scalar decode packs meta through the bit-field; the AVX2 path packs it by hand, so pin the layout.
    const StreamingProfilerRawRecMeta meta_probe{0, 5, 2, StreamingProfilerRawRecType::ZoneEnd};
    TT_FATAL(
        std::bit_cast<uint32_t>(meta_probe) == ((5u << 16) | (2u << 26) | (2u << 29)),
        "StreamingProfilerRawRecMeta bit-field layout does not match the vectorized packer");
    no_decode_ = env_flag("TT_METAL_STREAMING_PROFILER_NO_DECODE");
    read_only_ = env_flag("TT_METAL_STREAMING_PROFILER_READ_ONLY");
    die_after_ = env_u32("TT_METAL_STREAMING_PROFILER_WRITER_DIE_AFTER", 0);
    watchdog_ = std::chrono::seconds(env_u32("TT_METAL_STREAMING_PROFILER_WRITER_TIMEOUT_S", 120));
    // The ring is the capture's elastic buffer (the FIFO only lands frames), so its size bounds how far
    // a capture can outrun its consumers before per-consumer drops start: at ~9.8 wire bytes per zone,
    // the 512 MiB default holds ~55 M zones per stream. Rounded up to a power of two lines.
    const uint64_t ring_mb = env_u64("TT_METAL_STREAMING_PROFILER_RING_MB", 512);
    const uint64_t ring_lines = std::bit_ceil(std::max<uint64_t>(ring_mb, 1) << 14);
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
        cd.clock_synced = dev.clock_synced;
        cd.frequency_ghz = dev.frequency_ghz;
        cd.lanes = dev.lane_table;
        for (uint32_t sk = 0; sk < dev.sockets.size(); sk++) {
            auto s = std::make_unique<Stream>();
            s->sock = dev.sockets[sk].get();
            s->dev = d;
            s->sock_idx = sk;
            // Slots are faulted by prefault_rings() below, not by the decode thread.
            s->ring = std::make_unique<BroadcastRing<StreamingProfilerRingLine>>(
                ring_lines, BroadcastRing<StreamingProfilerRingLine>::DeferSlotInit{});
            s->ring_node = dev.numa_node;
            s->decode.reset(dev.num_cores);
            s->decode.core_of_xy = dev.core_of_xy;
            s->last_zone_ts.assign(nl, 0);
            streams_.push_back(std::move(s));
        }
    }
    prefault_rings();
}

StreamingProfilerReceiver::~StreamingProfilerReceiver() { shutdown(); }

// Pin each ring's pages to its device's node and fault them here, before the workload can produce: the
// ~400 MB per stream would otherwise be faulted by the decode thread with the producers already running.
// numa_tonode_memory makes placement independent of the touching thread, so the prefault can run
// anywhere -- one thread per stream, bound to that node so the zeroing is local.
void StreamingProfilerReceiver::prefault_rings() {
    std::vector<std::thread> pf;
    pf.reserve(streams_.size());
    for (auto& s : streams_) {
        pf.emplace_back([st = s.get()]() {
            const auto [base, bytes] = st->ring->raw_mapping();
            bind_memory_to_numa_node(base, bytes, st->ring_node);
            bind_current_thread_to_numa_node(st->ring_node);
            st->ring->construct_slots();
        });
    }
    for (auto& t : pf) {
        t.join();
    }
}

void StreamingProfilerReceiver::start() {
    TT_FATAL(!started_, "receiver already started");
    started_ = true;
    // Two decode threads per device is the design point, one per relay; sockets round-robin across them
    // via the strided partition below.
    const uint32_t nthreads = std::clamp<uint32_t>(
        env_u32("TT_METAL_STREAMING_PROFILER_DECODE_THREADS", 2), 1, streams_.size());
    nthreads_ = nthreads;
    // The audit attaches before ingest starts so its readers see the ring from line 0.
    if (!no_decode_ && !read_only_ && env_u32("TT_METAL_STREAMING_PROFILER_AUDIT", 1) != 0) {
        std::lock_guard<std::mutex> lk(consumers_mu_);
        auto c = std::make_unique<Consumer>();
        c->name = "audit";
        c->audit = true;
        c->handle = next_handle_++;
        c->thread = std::thread(&StreamingProfilerReceiver::consumer_thread, this, std::ref(*c));
        consumers_.push_back(std::move(c));
    }
    for (uint32_t t = 0; t < nthreads; t++) {
        std::vector<Stream*> owned;
        for (uint32_t i = t; i < streams_.size(); i += nthreads) {
            owned.push_back(streams_[i].get());
        }
        decode_threads_.emplace_back(&StreamingProfilerReceiver::decode_thread, this, std::move(owned));
    }
}

namespace {
// Whole aligned lines, each written once by one NT store: the ring is written at wire rate and read back
// cold by consumers, so cached stores would only pollute both sides' caches. The source is cold DRAM, so
// the distance prefetch keeps lines in flight -- it runs into the next frame's bytes, which sit
// contiguously behind this one in the FIFO.
__attribute__((target("avx512f"))) void ingest_copy_lines_512(
    BroadcastRing<StreamingProfilerRingLine>::Writer& w, uint64_t lpos, const uint32_t* frame, uint32_t nlines) {
    for (uint32_t k = 0; k < nlines; k++) {
        _mm_prefetch(reinterpret_cast<const char*>(frame + 16ull * k) + 4096, _MM_HINT_T0);
        _mm512_stream_si512(
            reinterpret_cast<__m512i*>(w.emit_slot_ptr(lpos + k)), _mm512_loadu_si512(frame + 16ull * k));
    }
}
void ingest_copy_lines_256(
    BroadcastRing<StreamingProfilerRingLine>::Writer& w, uint64_t lpos, const uint32_t* frame, uint32_t nlines) {
    for (uint32_t k = 0; k < nlines; k++) {
        _mm_prefetch(reinterpret_cast<const char*>(frame + 16ull * k) + 4096, _MM_HINT_T0);
        uint8_t* dst = reinterpret_cast<uint8_t*>(w.emit_slot_ptr(lpos + k));
        const uint8_t* src = reinterpret_cast<const uint8_t*>(frame) + 64ull * k;
        _mm256_stream_si256(
            reinterpret_cast<__m256i*>(dst), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src)));
        _mm256_stream_si256(
            reinterpret_cast<__m256i*>(dst + 32), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 32)));
    }
}
}  // namespace

bool StreamingProfilerReceiver::ingest_pass(Stream& s) {
    const uint32_t avail = s.sock->pages_available();
    if (avail == 0) {
        if (s.producers_done.load(std::memory_order_acquire)) {
            s.retired = true;
        }
        return false;
    }

    const uint32_t np = std::min(avail, kMaxPagesPerPass);
    if (s.first_data_tsc == 0) {
        s.first_data_tsc = tsc_now();
    }
    if (no_decode_) {
        s.sock->pop(np, true);
        s.last_commit_tsc = tsc_now();
        s.pages += np;
        s.passes++;
        return true;
    }
    if (read_only_) {
        const auto v = s.sock->peek(np);
        const uint64_t t0 = tsc_now();
        uint64_t acc = 0;
        for (size_t i = 0; i < v.first_bytes / 4; i += 16) {
            acc += v.first[i];
        }
        for (size_t i = 0; i < v.second_bytes / 4; i += 16) {
            acc += v.second[i];
        }
        s.checksum ^= acc;
        s.decode_ticks += tsc_now() - t0;
        s.sock->pop(np, true);
        s.last_commit_tsc = tsc_now();
        s.pages += np;
        s.passes++;
        return true;
    }

    const auto view = s.sock->peek(np);
    const uint64_t t0 = tsc_now();
    const bool use512 = profiler::spsc_host_avx512();
    const size_t first_words = view.first_bytes / 4;
    const size_t total_words = first_words + view.second_bytes / 4;
    alignas(64) uint32_t bounce[profiler::kSpscMaxFrameWords];
    // Headers never split across the spans: frames start on page boundaries and the FIFO wraps on one.
    auto word_at = [&](size_t o) { return o < first_words ? view.first[o] : view.second[o - first_words]; };
    auto& w = s.ring->writer();
    uint64_t lpos = s.wpos;
    size_t o = 0;
    uint32_t frames = 0, pages_done = 0, acked_pages = 0;
    while (o + kernel_profiler::SPSC_SPAN_PREFIX_WORDS <= total_words) {
        const uint32_t w1 = word_at(o + 1);
        if (!pp_is_bulkspan(word_at(o)) || w1 < kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS ||
            w1 > profiler::kSpscMaxPayloadWords) {
            // Framing is lost; step one page and rescan for the next header.
            s.bad_frames++;
            o += kernel_profiler::SPSC_SPAN_PAGE_WORDS;
            pages_done++;
            continue;
        }
        const uint32_t fw = kernel_profiler::spsc_span_frame_words(w1);
        if (o + fw > total_words) {
            break;  // the device pushes whole frames, so the rest of this one is in flight; leave it unpopped
        }
        const uint32_t* frame;
        if (o + fw <= first_words) {
            frame = view.first + o;
        } else if (o >= first_words) {
            frame = view.second + (o - first_words);
        } else {
            const size_t head_words = first_words - o;
            std::copy_n(view.first + o, head_words, bounce);
            std::copy_n(view.second, fw - head_words, bounce + head_words);
            frame = bounce;
        }
        const uint32_t nlines = fw / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
        w.emit_reserve(lpos + nlines);
        const uint64_t tf0 = tsc_now();
        if (use512) {
            ingest_copy_lines_512(w, lpos, frame, nlines);
        } else {
            ingest_copy_lines_256(w, lpos, frame, nlines);
        }
        s.frame_ticks += tsc_now() - tf0;
        lpos += nlines;
        o += fw;
        pages_done += nlines;
        frames++;
        if (frames % kAckBatchFrames == 0) {
            const uint64_t ta0 = tsc_now();
            s.sock->pop(pages_done - acked_pages, true);
            s.ack_ticks += tsc_now() - ta0;
            acked_pages = pages_done;
        }
    }
    if (lpos != s.wpos) {
        s.wpos = lpos;
        w.emit_commit(lpos);  // its sfence orders the NT line stores before the head advance
    }
    s.decode_ticks += tsc_now() - t0;
    if (pages_done > acked_pages) {
        const uint64_t ta0 = tsc_now();
        s.sock->pop(pages_done - acked_pages, true);
        s.ack_ticks += tsc_now() - ta0;
    }
    if (pages_done == 0) {
        if (s.producers_done.load(std::memory_order_acquire)) {
            if (!s.desync_warned) {
                s.desync_warned = true;
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler receiver] d{}/s{}: {} pages (a partial frame) remain after the relays "
                    "finished -- flow control desynchronized; dropping them",
                    s.dev,
                    s.sock_idx,
                    avail);
                s.sock->pop(avail, true);
            }
            s.retired = true;
        }
        return false;
    }
    s.last_commit_tsc = tsc_now();
    s.pages += pages_done;
    s.frames += frames;
    s.passes++;
    return true;
}

void StreamingProfilerReceiver::decode_thread(std::vector<Stream*> streams) {
    // Run on the node the ring was bound to: this thread NT-stores into the ring at tens of GB/s while
    // reading the FIFO at tens more, so a cross-node ring puts every store on the interconnect -- one
    // stream took 172 ms of frame time against a peer's 111 ms for the same work, and the slowest stream
    // gates the pipeline: its relay blocks on credit, its ring fills, its producers stall.
    if (!streams.empty() && streams.front()->ring_node >= 0) {
        bind_current_thread_to_numa_node(streams.front()->ring_node);
    }
    std::string name = "pd-dec:";
    for (Stream* s : streams) {
        name += std::to_string(s->dev) + "." + std::to_string(s->sock_idx) + ",";
    }
    name.pop_back();
    tracy::SetThreadName(name.c_str());
    set_os_thread_name(name);
    prctl(PR_SET_TIMERSLACK, 1000);  // default 50 us slack would round every probe sleep up to it
    IdleBackoff backoff(kProbeSleepCapUs);
    uint64_t data_passes = 0;
    for (Stream* s : streams) {
        s->last_progress = std::chrono::steady_clock::now();
    }
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
                data_passes++;
                s->last_progress = std::chrono::steady_clock::now();
            } else if (
                !s->producers_done.load(std::memory_order_relaxed) && !s->watchdog_fired &&
                std::chrono::steady_clock::now() - s->last_progress > watchdog_) {
                s->watchdog_fired = true;
                log_warning(
                    tt::LogMetal,
                    "[streaming profiler receiver] d{}/s{}: no data for {} s while producers are live",
                    s->dev,
                    s->sock_idx,
                    watchdog_.count());
            }
        }
        if (all_retired) {
            break;
        }
        if (die_after_ != 0 && data_passes >= die_after_) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler receiver] {}: TEST HOOK exiting after {} passes; acks stop here",
                name,
                data_passes);
            break;
        }
        if (any) {
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

StreamingProfilerConsumerHandle StreamingProfilerReceiver::add_consumer(
    std::string name, StreamingProfilerRecordCallback cb) {
    TT_FATAL(!t_in_consumer, "add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> lk(consumers_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->cb = std::move(cb);
    c->handle = next_handle_++;
    c->thread = std::thread(&StreamingProfilerReceiver::consumer_thread, this, std::ref(*c));
    consumers_.push_back(std::move(c));
    return consumers_.back()->handle;
}

StreamingProfilerConsumerHandle StreamingProfilerReceiver::add_raw_consumer(
    std::string name, StreamingProfilerRawRecordCallback cb) {
    TT_FATAL(!t_in_consumer, "add_raw_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> lk(consumers_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->raw_cb = std::move(cb);
    c->handle = next_handle_++;
    c->thread = std::thread(&StreamingProfilerReceiver::consumer_thread, this, std::ref(*c));
    consumers_.push_back(std::move(c));
    return consumers_.back()->handle;
}

void StreamingProfilerReceiver::remove_consumer(StreamingProfilerConsumerHandle handle) {
    TT_FATAL(!t_in_consumer, "remove_consumer must not be called from a consumer callback");
    std::unique_ptr<Consumer> victim;
    {
        std::lock_guard<std::mutex> lk(consumers_mu_);
        auto it =
            std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& c) { return c->handle == handle; });
        TT_FATAL(it != consumers_.end(), "unknown consumer handle {}", handle);
        victim = std::move(*it);
        consumers_.erase(it);
    }
    victim->mode.store(2, std::memory_order_release);
    victim->thread.join();
    if (victim->dropped != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler receiver] consumer \"{}\" removed having dropped {} records",
            victim->name,
            victim->dropped);
    }
}

// Baseline code: nothing here may carry an AVX-512 target attribute or the compiler could emit AVX-512
// into paths that run on any host. The 512-bit kernels are separate attributed functions, called once
// per block and gated on spsc_host_avx512().
namespace {

// One stream's decode, owned by one consumer thread. The Sink type decides what a decoded record
// becomes: SpscCachedRecSink composes 24 B records into the consumer's scratch (cached stores -- the
// scratch is re-read immediately by the same thread), while SpscNullRecSink composes nothing and the
// vector blocks skip their compose entirely -- that is the audit, which only wants the wire-integrity
// accounting. `st`/`last_ts` are externally owned so the audit can decode straight into the Stream's
// report fields while user consumers keep private copies.
template <typename Sink>
struct StreamDecoder {
    using SinkT = Sink;
    tt::tt_metal::profiler::SpanDecodeState* st = nullptr;
    uint64_t* last_ts = nullptr;
    tt::tt_metal::streaming_profiler::StallIdMirror stall_ids;
    uint32_t dev = 0;
    Sink sink{};
    uint64_t recs = 0, zone_markers = 0, stall_zones = 0, order_regressions = 0, bad_frames = 0;
    uint64_t stall_mark = 0;  // stall_zones at the last delivery; the batch delta is the difference
    uint64_t min_ts = 0, max_ts = 0;

    uint32_t decode_frame(const uint32_t* frame, uint32_t fw);
};

template <typename Sink>
uint32_t StreamDecoder<Sink>::decode_frame(const uint32_t* frame, uint32_t fw) {
    namespace profiler = tt::tt_metal::profiler;
    using tt::tt_metal::streaming_profiler::StreamingProfilerRawRecType;
    stall_ids.refresh();
    // Raw locals for the per-marker probe and tallies: the emitters store through casted pointers, so
    // anything reached via `this` would be reloaded after every store. Locals whose address never
    // escapes stay in registers.
    const uint32_t* const stall_tab = stall_ids.table.empty() ? nullptr : stall_ids.table.data();
    const uint32_t stall_mask = stall_ids.mask;
    auto is_stall = [stall_tab, stall_mask](uint32_t id) -> bool {
        if (stall_tab == nullptr) {
            return false;
        }
        uint32_t slot = (id * 0x9E3779B9u) & stall_mask;
        while (true) {
            const uint32_t v = stall_tab[slot];
            if (v == id) {
                return true;
            }
            if (v == 0xFFFFFFFFu) {
                return false;
            }
            slot = (slot + 1) & stall_mask;
        }
    };
    const uint32_t d = dev;
    const bool k512 = profiler::spsc_host_avx512();
    Sink& sk = sink;
    uint64_t zm = 0, sz = 0, oreg = 0, rc = 0;
    uint64_t mn = min_ts, mx = max_ts;
    uint64_t* const lts = last_ts;
    const auto meta_hi = [d](uint32_t lane, StreamingProfilerRawRecType rt) {
        return static_cast<uint64_t>((lane << 16) | (d << 26) | (static_cast<uint32_t>(rt) << 29)) << 32;
    };

    auto emit = [&](uint32_t lane, uint32_t type, uint32_t zone_id, uint64_t ts, uint32_t prog, uint32_t duration) {
        StreamingProfilerRawRecType rt = StreamingProfilerRawRecType::ZoneTotal;
        if (type == PP_ZONE_ATOMIC) {
            rt = StreamingProfilerRawRecType::Zone;
        } else if (type != PP_ZONE_TOTAL) {
            rt = type == PP_ZONE_START ? StreamingProfilerRawRecType::ZoneStart : StreamingProfilerRawRecType::ZoneEnd;
        }
        if (type != PP_ZONE_TOTAL) {
            // Counted in halves: a ZONE_ATOMIC record is a whole zone, a START/END is half of one, and
            // every reader of this counter divides by two.
            zm += type == PP_ZONE_ATOMIC ? 2 : 1;
            // PRODUCER-STALL, matched by ELF-resolved name via the id table: a producer RISC blocked on
            // a full ring. Stall zones ship as single ZONE_ATOMIC packets, one per stall event; the START
            // probe covers the non-atomic path.
            sz += ((type == PP_ZONE_ATOMIC || type == PP_ZONE_START) && is_stall(zone_id)) ? 1 : 0;
            if (ts < lts[lane]) {
                oreg++;
            } else {
                lts[lane] = ts;
            }
            if (mn == 0) {
                mn = ts;
            }
            mx = ts;
        }
        rc++;
        if constexpr (Sink::kStores) {
            sk.put3(ts, meta_hi(lane, rt) | zone_id, (static_cast<uint64_t>(duration) << 32) | prog);
        }
    };
    auto emit_data = [&](uint32_t lane,
                         uint32_t type,
                         uint32_t id,
                         uint64_t ts,
                         uint32_t prog,
                         const uint32_t* payload,
                         uint32_t n) {
        const uint64_t pg = prog;
        // PP_EVENT is payload-less by wire shape: one record is the whole packet, and no Ext or Cont
        // ever follows an Event.
        if (type != PP_DATA) {
            rc++;
            if constexpr (Sink::kStores) {
                sk.put3(ts, meta_hi(lane, StreamingProfilerRawRecType::Event) | id, pg);
            }
            return;
        }
        rc += 2 + (n > 2 ? (n - 1) / 2 : 0);
        if constexpr (Sink::kStores) {
            // Ext carries the payload count in its id field and payload words 1-2 in its ts field, so a
            // short DATA is two records.
            const uint64_t hi0 = n >= 1 ? payload[0] : 0;
            const uint64_t lo0 = n >= 2 ? payload[1] : 0;
            sk.put3(ts, meta_hi(lane, StreamingProfilerRawRecType::Data) | id, pg);
            sk.put3((hi0 << 32) | lo0, meta_hi(lane, StreamingProfilerRawRecType::Ext) | n, pg);
            for (uint32_t k = 2; k < n; k += 2) {
                const uint64_t hi = payload[k];
                const uint64_t lo = (k + 1 < n) ? payload[k + 1] : 0;
                sk.put3((hi << 32) | lo, meta_hi(lane, StreamingProfilerRawRecType::Cont), pg);
            }
        } else {
            (void)payload;
        }
    };

#if defined(__AVX2__)
    auto emit_zones8 = [&](uint32_t lane, uint32_t th, uint32_t prog, __m256i w0s, __m256i w1s) {
        zm += 8;
        // Order invariant at block endpoints only: the producer guarantees monotonicity inside a run, so
        // the boundary compare still catches every head-mirror/resync error class. th is block-constant,
        // so comparing on it is exact.
        const uint64_t ts_first =
            (static_cast<uint64_t>(th) << 32) | static_cast<uint32_t>(_mm256_extract_epi32(w1s, 0));
        const uint64_t ts_last =
            (static_cast<uint64_t>(th) << 32) | static_cast<uint32_t>(_mm256_extract_epi32(w1s, 7));
        oreg += ts_first < lts[lane] ? 1 : 0;
        lts[lane] = ts_last;
        if (mn == 0) {
            mn = ts_first;
        }
        mx = ts_last;
        rc += 8;
        const __m256i ids = _mm256_and_si256(w0s, _mm256_set1_epi32(0x07FFFFFF));
        alignas(32) uint32_t id_arr[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(id_arr), ids);
        const uint32_t end_mask = static_cast<uint32_t>(
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(w0s, 4))));  // bit27 -> sign bit: 1 = ZONE_END
        for (int k = 0; k < 8; k++) {
            sz += (((end_mask >> k) & 1u) == 0 && is_stall(id_arr[k])) ? 1 : 0;
        }
        if constexpr (Sink::kStores) {
            // A 24 B record is three quadwords: q0 = ts (th<<32 | w1), q1 = meta<<32 | id27, q2 = prog.
            // meta = lane | dev | type, with type = ZoneStart + the wire's END bit (an ADD, not an OR --
            // the type field is a small integer, not a bit set).
            const uint32_t meta_base = (lane << 16) | (d << 26) | (1u << 29);  // type = ZoneStart
            const __m256i meta = _mm256_add_epi32(
                _mm256_slli_epi32(_mm256_and_si256(w0s, _mm256_set1_epi32(0x08000000)), 2),  // END: +1 in type
                _mm256_set1_epi32(static_cast<int>(meta_base)));
            alignas(32) uint32_t w1_arr[8];
            alignas(32) uint32_t meta_arr[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(w1_arr), w1s);
            _mm256_store_si256(reinterpret_cast<__m256i*>(meta_arr), meta);
            const uint64_t th_hi = static_cast<uint64_t>(th) << 32;
            const uint64_t prog64 = prog;
            for (uint32_t k = 0; k < 8; k++) {
                sk.put3(th_hi | w1_arr[k], (static_cast<uint64_t>(meta_arr[k]) << 32) | id_arr[k], prog64);
            }
        } else {
            (void)prog;
        }
    };
    auto emit_atomic16 = [&](uint32_t lane, uint32_t th, uint32_t prog, const uint32_t* src, uint32_t avail,
                             uint32_t max_recs) -> uint32_t {
        if (!k512) {
            return 0;  // atomics are rare (launch/rewind anchors); the scalar arm carries them
        }
        const auto a = profiler::spsc_atomic16_avx512(src, avail, max_recs, th, prog, lane, d, sk);
        if (a.n == 0) {
            return 0;
        }
        zm += a.n * 2;  // halves: every record in an atomic block is a whole zone
        // Stall zones ride the atomic wire, so the block pays the id probe: one L1 load per record on
        // the miss path.
        if (stall_tab != nullptr) {
            for (uint32_t k = 0; k < a.n; k++) {
                sz += is_stall(src[3u * k] & 0x07FFFFFFu) ? 1 : 0;
            }
        }
        oreg += a.ts_first < lts[lane] ? 1 : 0;
        lts[lane] = a.ts_last;
        if (mn == 0) {
            mn = a.ts_first;
        }
        mx = a.ts_last;
        rc += a.n;
        return a.n;
    };
    auto emit_zone_s16 = [&](uint32_t lane, uint64_t cursor, uint32_t prog, const uint32_t* src, uint32_t avail,
                             uint32_t max_recs) -> profiler::SpscZoneS16Result {
        const auto z = k512 ? profiler::spsc_zone_s16_avx512(src, avail, max_recs, cursor, prog, lane, d, sk)
                            : profiler::spsc_zone_s8_avx2(src, avail, max_recs, cursor, prog, lane, d, sk);
        if (z.n == 0) {
            return z;
        }
        zm += z.n * 2;
        // Exact here, not sampled: in-block ends are cursor + positive deltas, monotonic by
        // construction.
        oreg += z.ts_first < lts[lane] ? 1 : 0;
        lts[lane] = z.ts_last;
        if (mn == 0) {
            mn = z.ts_first;
        }
        mx = z.ts_last;
        rc += z.n;
        return z;
    };
#endif

#if defined(__AVX2__)
    const uint32_t payload = profiler::spsc_decode_frame(
        *st,
        frame,
        emit,
        emit_data,
        profiler::SpscIgnoreProg{},
        emit_zones8,
        profiler::SpscNoAtomic8{},
        emit_atomic16,
        emit_zone_s16,
        fw);
#else
    const uint32_t payload = profiler::spsc_decode_frame(*st, frame, emit, emit_data);
#endif
    zone_markers += zm;
    stall_zones += sz;
    order_regressions += oreg;
    recs += rc;
    min_ts = mn;
    max_ts = mx;
    return payload;
}

}  // namespace

void StreamingProfilerReceiver::consumer_thread(Consumer& c) {
    const std::string name = "pd-con:" + c.name;
    tracy::SetThreadName(name.c_str());
    set_os_thread_name(name);
    t_in_consumer = true;
    const bool audit = c.audit;
    std::vector<BroadcastRing<StreamingProfilerRingLine>::Reader> readers;
    readers.reserve(streams_.size());
    for (auto& s : streams_) {
        readers.push_back(s->ring->make_reader());
    }
    std::vector<StreamingProfilerRingLine> lines(kConsumerLineBatch);
    std::vector<uint64_t> last_dropped(readers.size(), 0);
    // Frame re-assembly, per ring: ring lines accumulate here until a whole frame is present. After a
    // drop the stream position is arbitrary, so scan line by line for the next plausible frame head --
    // the decoder's head-adoption path then counts the gap as a resync, as for a device-side loss.
    struct Assembly {
        std::vector<uint32_t> pending;
        bool scanning = false;
    };
    std::vector<Assembly> assy(streams_.size());
    std::vector<StreamDecoder<profiler::SpscNullRecSink>> adecs;
    std::vector<StreamDecoder<profiler::SpscCachedRecSink>> udecs;
    std::vector<profiler::SpanDecodeState> ustates;
    std::vector<std::vector<uint64_t>> ults;
    std::vector<StreamingProfilerRawRec> scratch;
    if (audit) {
        adecs.resize(streams_.size());
        for (size_t i = 0; i < streams_.size(); i++) {
            adecs[i].st = &streams_[i]->decode;
            adecs[i].last_ts = streams_[i]->last_zone_ts.data();
            adecs[i].dev = streams_[i]->dev;
        }
    } else {
        scratch.resize(kConsumerScratchRecs + 3);  // sink puts over-store up to one vector; slack, not capacity
        udecs.resize(streams_.size());
        ustates.resize(streams_.size());
        ults.resize(streams_.size());
        for (size_t i = 0; i < streams_.size(); i++) {
            const auto& dev = devices_[streams_[i]->dev];
            ustates[i].reset(dev.num_cores);
            ustates[i].core_of_xy = dev.core_of_xy;
            ults[i].assign(static_cast<size_t>(dev.num_cores) * profiler::kSpscNRiscDecode, 0);
            udecs[i].st = &ustates[i];
            udecs[i].last_ts = ults[i].data();
            udecs[i].dev = streams_[i]->dev;
            udecs[i].sink.buf = reinterpret_cast<uint8_t*>(scratch.data());
        }
    }
    // The pairing stage (public consumers only). Zones are RAII scopes on the device, so per lane the
    // raw stream obeys strict stack discipline: push on ZoneStart, pop on ZoneEnd, and the pop's mate is
    // the matching open. One stack per (dev, lane), owned by this thread -- every consumer thread reads
    // the whole ring independently, so there is no sharing and no lock. A Zone record is emitted at END
    // time; everything else converts 1:1.
    struct OpenZone {
        uint64_t ts;
        uint32_t id;
        uint32_t prog;
    };
    std::vector<std::vector<OpenZone>> stacks;
    if (!audit && c.raw_cb == nullptr) {
        stacks.resize(ctx_.devices.size() * kStreamingProfilerMaxLanes);
    }
    std::vector<StreamingProfilerRec> out;
    out.reserve(kConsumerScratchRecs);
    uint64_t unmatched_ends = 0;  // ZoneEnd with an empty stack (only possible after ring drops)
    uint64_t id_mismatches = 0;   // ZoneEnd whose id differs from the matching open: trust neither, drop
    auto pair_batch = [&](std::span<const StreamingProfilerRawRec> got) {
        out.clear();
        for (const StreamingProfilerRawRec& r : got) {
            const uint32_t si = r.meta.dev * kStreamingProfilerMaxLanes + r.meta.lane;
            switch (r.meta.type) {
                case StreamingProfilerRawRecType::ZoneStart: stacks[si].push_back({r.ts, r.id, r.prog}); break;
                case StreamingProfilerRawRecType::ZoneEnd: {
                    if (stacks[si].empty()) {
                        unmatched_ends++;
                        break;
                    }
                    const OpenZone open = stacks[si].back();
                    stacks[si].pop_back();
                    if (open.id != r.id) {
                        id_mismatches++;  // corrupt pair: emitting under either id would mislabel it
                        break;
                    }
                    StreamingProfilerRec& o = out.emplace_back();
                    o.data.zone = {open.ts, r.ts - open.ts};
                    o.id = open.id;
                    o.meta = {0, r.meta.lane, r.meta.dev, StreamingProfilerRecType::Zone};
                    o.prog = open.prog;
                    break;
                }
                case StreamingProfilerRawRecType::Zone: {
                    // The device's atomic-zone path: ts is the END and duration is set, so no stack.
                    StreamingProfilerRec& o = out.emplace_back();
                    o.data.zone = {r.ts - r.duration, r.duration};
                    o.id = r.id;
                    o.meta = {0, r.meta.lane, r.meta.dev, StreamingProfilerRecType::Zone};
                    o.prog = r.prog;
                    break;
                }
                case StreamingProfilerRawRecType::ZoneTotal: {
                    StreamingProfilerRec& o = out.emplace_back();
                    o.data.sum = r.ts;
                    o.id = r.id;
                    o.meta = {0, r.meta.lane, r.meta.dev, StreamingProfilerRecType::ZoneTotal};
                    o.prog = r.prog;
                    break;
                }
                default: {  // Data / Event / Ext / Cont: 1:1
                    StreamingProfilerRec& o = out.emplace_back();
                    o.data.ts = r.ts;
                    o.id = r.id;
                    StreamingProfilerRecType t = StreamingProfilerRecType::Data;
                    if (r.meta.type == StreamingProfilerRawRecType::Event) {
                        t = StreamingProfilerRecType::Event;
                    } else if (r.meta.type == StreamingProfilerRawRecType::Ext) {
                        t = StreamingProfilerRecType::Ext;
                    } else if (r.meta.type == StreamingProfilerRawRecType::Cont) {
                        t = StreamingProfilerRecType::Cont;
                    }
                    o.meta = {0, r.meta.lane, r.meta.dev, t};
                    o.prog = r.prog;
                    break;
                }
            }
        }
    };
    // Hand the scratch records (plus the ring-drop and stall deltas) to the callback, then reset the
    // scratch. The audit has no callback: it publishes its tallies into the Stream's report fields
    // instead, and is their only writer.
    auto deliver = [&](auto& dec, size_t r, uint64_t& dd) {
        using DT = std::decay_t<decltype(dec)>;
        if constexpr (std::is_same_v<typename DT::SinkT, profiler::SpscCachedRecSink>) {
            const size_t nrec = dec.sink.off / sizeof(StreamingProfilerRawRec);
            const uint64_t sd = dec.stall_zones - dec.stall_mark;
            dec.stall_mark = dec.stall_zones;
            dec.sink.off = 0;
            if (nrec == 0 && dd == 0 && sd == 0) {
                return;
            }
            const std::span<const StreamingProfilerRawRec> got(scratch.data(), nrec);
            try {
                if (c.raw_cb != nullptr) {
                    c.delivered += nrec;
                    c.raw_cb(StreamingProfilerRawRecordBatch{got, dd, &ctx_, sd});
                } else {
                    pair_batch(got);
                    if (!out.empty() || dd != 0 || sd != 0) {
                        c.delivered += out.size();
                        c.cb(StreamingProfilerRecordBatch{std::span<const StreamingProfilerRec>(out), dd, &ctx_, sd});
                    }
                }
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "[streaming profiler receiver] consumer \"{}\" threw: {}", c.name, e.what());
            }
            dd = 0;
        } else {
            Stream& s = *streams_[r];
            s.records = dec.recs;
            s.zone_markers = dec.zone_markers;
            s.stall_zones = dec.stall_zones;
            s.order_regressions = dec.order_regressions;
            s.min_zone_ts = dec.min_ts;
            s.max_zone_ts = dec.max_ts;
            c.delivered = 0;
            for (const auto& ad : adecs) {
                c.delivered += ad.recs;
            }
            (void)dd;
        }
    };
    auto pass_ring = [&](auto& dec, size_t r) -> bool {
        const auto got = readers[r].read_batch(std::span<StreamingProfilerRingLine>(lines));
        const uint64_t dropped_total = readers[r].dropped();
        uint64_t dd = dropped_total - last_dropped[r];
        last_dropped[r] = dropped_total;
        if (got.empty() && dd == 0) {
            return false;
        }
        Assembly& a = assy[r];
        if (dd != 0) {
            a.pending.clear();
            a.scanning = true;
        }
        const size_t oldw = a.pending.size();
        a.pending.resize(oldw + got.size() * 16);
        if (!got.empty()) {
            std::memcpy(a.pending.data() + oldw, got.data(), got.size() * sizeof(StreamingProfilerRingLine));
        }
        size_t off = 0;
        while (a.pending.size() - off >= kernel_profiler::SPSC_SPAN_PREFIX_WORDS) {
            const uint32_t w0 = a.pending[off];
            const uint32_t w1 = a.pending[off + 1];
            if (!pp_is_bulkspan(w0) || w1 < kernel_profiler::SPSC_SPAN_WIRE_CTRL_WORDS ||
                w1 > profiler::kSpscMaxPayloadWords) {
                if (!a.scanning) {
                    dec.bad_frames++;
                }
                off += kernel_profiler::SPSC_SPAN_PAGE_WORDS;
                continue;
            }
            a.scanning = false;
            const uint32_t fw = kernel_profiler::spsc_span_frame_words(w1);
            if (a.pending.size() - off < fw) {
                break;  // frames arrive whole; the tail lines are still in the ring or in flight
            }
            using DT = std::decay_t<decltype(dec)>;
            if constexpr (std::is_same_v<typename DT::SinkT, profiler::SpscCachedRecSink>) {
                if (dec.sink.off / sizeof(StreamingProfilerRawRec) + kMaxFrameRecs > kConsumerScratchRecs) {
                    deliver(dec, r, dd);
                }
            }
            const uint32_t payload = dec.decode_frame(&a.pending[off], fw);
            if (payload != 0 && payload != w1) {
                dec.st->anomalies++;
            }
            off += fw;
        }
        a.pending.erase(a.pending.begin(), a.pending.begin() + static_cast<ptrdiff_t>(std::min(off, a.pending.size())));
        deliver(dec, r, dd);
        return true;
    };
    IdleBackoff backoff(kConsumerSleepCapUs);
    for (;;) {
        bool any = false;
        for (size_t r = 0; r < readers.size(); r++) {
            const uint64_t t0 = tsc_now();
            if (audit ? pass_ring(adecs[r], r) : pass_ring(udecs[r], r)) {
                any = true;
                c.busy_ticks += tsc_now() - t0;
            }
        }
        if (any) {
            backoff.reset();
            continue;
        }
        if (c.mode.load(std::memory_order_acquire) != 0) {
            break;
        }
        backoff.idle();
    }
    c.dropped = 0;
    for (auto& r : readers) {
        c.dropped += r.dropped();
    }
    // A lossless capture ends with every stack empty: the producers close every scope they open and the
    // quiesce path drains to the last marker. Leftover opens mean records were lost (ring drops for this
    // consumer) or a start/end pair was corrupted.
    uint64_t leftover_opens = 0;
    for (const auto& st : stacks) {
        leftover_opens += st.size();
    }
    if (leftover_opens != 0 || unmatched_ends != 0 || id_mismatches != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler receiver] consumer \"{}\" pairing: {} zones left OPEN at shutdown, {} unmatched ends, "
            "{} start/end id mismatches [all MUST be 0 on a lossless capture]",
            c.name,
            leftover_opens,
            unmatched_ends,
            id_mismatches);
    }
}

void StreamingProfilerReceiver::notify_producers_done(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->dev == device_index && s->sock_idx == socket_index) {
            s->producers_done.store(true, std::memory_order_release);
        }
    }
}

void StreamingProfilerReceiver::shutdown() {
    if (shutdown_done_.exchange(true)) {
        return;
    }
    stop_.store(true, std::memory_order_release);
    for (auto& t : decode_threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    std::vector<std::unique_ptr<Consumer>> consumers;
    {
        std::lock_guard<std::mutex> lk(consumers_mu_);
        consumers.swap(consumers_);
    }
    for (auto& c : consumers) {
        c->mode.store(1, std::memory_order_release);
    }
    for (auto& c : consumers) {
        if (c->thread.joinable()) {
            c->thread.join();
        }
    }
    consumers_report_ = std::move(consumers);
}

std::vector<uint32_t> StreamingProfilerReceiver::final_lane_heads(uint32_t device_index) const {
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

void StreamingProfilerReceiver::log_report() const {
    uint64_t total_pages = 0, total_wire_words = 0, total_zone_markers = 0, total_resync_words = 0;
    uint64_t busy_ticks = 0, order_regressions = 0;
    uint64_t first_tsc = 0, last_tsc = 0;
    // Busy is the busiest thread, so ticks group by owning thread (stream i -> thread i % nthreads_) and
    // not by stream: with fewer threads than sockets a per-stream max understates busy several-fold.
    std::vector<uint64_t> thread_ticks(std::max<uint32_t>(nthreads_, 1), 0);
    for (size_t i = 0; i < streams_.size(); i++) {
        const Stream& s = *streams_[i];
        total_pages += s.pages;
        total_wire_words += s.decode.live_words;
        total_zone_markers += s.zone_markers;
        total_resync_words += s.decode.resync_words;
        order_regressions += s.order_regressions;
        thread_ticks[i % thread_ticks.size()] += s.decode_ticks;
        busy_ticks = std::max(busy_ticks, thread_ticks[i % thread_ticks.size()]);
        if (s.first_data_tsc != 0 && (first_tsc == 0 || s.first_data_tsc < first_tsc)) {
            first_tsc = s.first_data_tsc;
        }
        last_tsc = std::max(last_tsc, s.last_commit_tsc);
        log_info(
            tt::LogMetal,
            "[streaming profiler receiver] d{}/s{}: {} frames ({:.1f} MB) in {} passes | decode {:.1f} ms "
            "(frame {:.1f} + "
            "ack {:.1f} + other {:.1f}) | {} records "
            "({} zones, {} stall-zones) | resyncs {} ({} words) | head-lag {} | anomalies {} | bad frames {} | "
            "unknown-core frames {} | order regressions {} [MUST be 0]",
            s.dev,
            s.sock_idx,
            s.frames,
            s.pages * static_cast<double>(kPageBytes) / 1e6,
            s.passes,
            ticks_to_ms(s.decode_ticks),
            ticks_to_ms(s.frame_ticks),
            ticks_to_ms(s.ack_ticks),
            ticks_to_ms(s.decode_ticks - std::min(s.decode_ticks, s.frame_ticks + s.ack_ticks)),
            s.records,
            s.zone_markers / 2,
            s.stall_zones,
            s.decode.resync_events,
            s.decode.resync_words,
            s.decode.head_lag,
            s.decode.anomalies,
            s.bad_frames,
            s.decode.unknown_core_frames,
            s.order_regressions);
        const uint64_t vrec = s.decode.vec_zone_recs + s.decode.vec_atomic_recs + s.decode.vec_zone_s_recs;
        const uint64_t allrec = vrec + s.decode.scalar_recs;
        log_info(
            tt::LogMetal,
            "[streaming profiler receiver] d{}/s{} decode paths: {:.1f}% vectorized "
            "({} zoneS16 + {} zone8 + {} atomic16), "
            "{} scalar, {} vec-block rejects",
            s.dev,
            s.sock_idx,
            allrec ? 100.0 * static_cast<double>(vrec) / static_cast<double>(allrec) : 0.0,
            s.decode.vec_zone_s_recs,
            s.decode.vec_zone_recs,
            s.decode.vec_atomic_recs,
            s.decode.scalar_recs,
            s.decode.vec_block_rejects);
        {
            const auto& d = s.decode;
            std::string bt;
            for (uint32_t t = 0; t < 16; t++) {
                if (d.scalar_by_type[t] != 0) {
                    bt += fmt::format("t{}={} ", t, d.scalar_by_type[t]);
                }
            }
            log_info(
                tt::LogMetal,
                "[streaming profiler receiver] d{}/s{} zoneS16 blocks: {} calls, {} recs, {:.2f} recs/call "
                "| atomic blocks: "
                "{} calls, {} recs, {:.2f} recs/call (max 16) | scalar by type: {}",
                s.dev, s.sock_idx, d.vec_zone_s_calls, d.vec_zone_s_recs,
                d.vec_zone_s_calls ? double(d.vec_zone_s_recs) / double(d.vec_zone_s_calls) : 0.0,
                d.vec_atomic_calls, d.vec_atomic_recs,
                d.vec_atomic_calls ? double(d.vec_atomic_recs) / double(d.vec_atomic_calls) : 0.0, bt);
        }
    }
    uint64_t consumer_drops = 0;
    for (const auto& c : consumers_report_) {
        consumer_drops += c->dropped;
        log_info(
            tt::LogMetal,
            "[streaming profiler receiver] consumer \"{}\": {} delivered, {} dropped, busy {:.1f} ms",
            c->name,
            c->delivered,
            c->dropped,
            ticks_to_ms(c->busy_ticks));
    }
    const double busy_ms = ticks_to_ms(busy_ticks);
    const double wall_ms = (first_tsc != 0 && last_tsc > first_tsc) ? ticks_to_ms(last_tsc - first_tsc) : 0.0;
    const double d2h_gb = total_pages * static_cast<double>(kPageBytes) / 1e9;
    const double wire_gb = total_wire_words * 4.0 / 1e9;
    const double mzones = total_zone_markers / 2.0 / 1e6;
    auto rate = [](double num, double ms) { return ms > 0.0 ? num / (ms / 1e3) : 0.0; };
    log_info(
        tt::LogMetal,
        "[streaming profiler receiver] SUSTAINED THROUGHPUT: busy {:.1f} ms (max ingest thread) -> {:.2f} GB/s D2H | "
        "{:.2f} GB/s marker-wire | {:.2f} Mzones/s || wall {:.1f} ms (first data -> last commit) -> {:.2f} GB/s "
        "D2H | {:.2f} Mzones/s",
        busy_ms,
        rate(d2h_gb, busy_ms),
        rate(wire_gb, busy_ms),
        rate(mzones, busy_ms),
        wall_ms,
        rate(d2h_gb, wall_ms),
        rate(mzones, wall_ms));
    for (uint32_t d = 0; d < devices_.size(); d++) {
        uint64_t lo = 0, hi = 0;
        for (const auto& sp : streams_) {
            if (sp->dev != d || sp->min_zone_ts == 0) {
                continue;
            }
            lo = lo == 0 ? sp->min_zone_ts : std::min(lo, sp->min_zone_ts);
            hi = std::max(hi, sp->max_zone_ts);
        }
        const double freq = devices_[d].frequency_ghz;
        if (lo != 0 && hi > lo && freq > 0.0) {
            const double dev_ms = (hi - lo) / freq / 1e6;
            log_info(
                tt::LogMetal,
                "[streaming profiler receiver] device {} zone window {:.1f} ms (first->last zone @ {:.6f} GHz): {:.2f} "
                "GB/s D2H | {:.2f} Mzones/s",
                devices_[d].chip_id,
                dev_ms,
                freq,
                rate(d2h_gb, dev_ms),
                rate(mzones, dev_ms));
        }
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler receiver] loss: decode resyncs {} words | consumer ring drops {} | order regressions {} "
        "[device credit-timeout drops reported by the profiler above; all zero = lossless capture]",
        total_resync_words,
        consumer_drops,
        order_regressions);
}

}  // namespace tt::tt_metal::streaming_profiler
