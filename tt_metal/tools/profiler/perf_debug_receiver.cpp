// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_receiver.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <sys/prctl.h>
#include <x86intrin.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/tt_pause.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>

#include "tt_metal/common/broadcast_ring.hpp"
#include "tools/profiler/perf_debug_env.hpp"
#include "tools/profiler/spsc_packet.h"

namespace tt::tt_metal::perf_debug {

namespace {

// Credit-return quantum: pop+ack every 8 decoded frames (about one mover push) instead of once per pass,
// so the device sees credit at decode pace rather than in whole-pass steps.
constexpr uint32_t kAckBatchFrames = 8;
// Per-pass peek window (~680 KB). Every complete frame inside it is consumed -- pages peeked but not
// consumed are clflushed again by the next peek, so the only re-flush waste is a partial tail frame.
constexpr uint32_t kMaxPagesPerPass = 64 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr size_t kConsumerScratchRecs = 1 << 16;
constexpr uint32_t kEmptyPollsBeforeSleep = 1000;
// Decode threads probe the FIFO at least this often when idle; consumers are latency-tolerant and may
// sleep longer. Anything under ~50 us needs the timer slack shrunk or sleep_for quietly rounds up to it.
constexpr uint32_t kProbeSleepCapUs = 5;
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

PerfDebugReceiver::PerfDebugReceiver(ReceiverConfig config, std::vector<ReceiverDeviceConfig> devices) :
    cfg_(std::move(config)), devices_(std::move(devices)) {
    TT_FATAL(devices_.size() <= kPerfDebugMaxDevices, "record dev field holds {} devices", kPerfDebugMaxDevices);
    // The scalar decode packs meta through the bit-field; the AVX2 path packs it by hand, so pin the layout.
    const PerfDebugRecMeta meta_probe{0x1234, 5, 2, PerfDebugRecType::ZoneEnd};
    TT_FATAL(
        std::bit_cast<uint32_t>(meta_probe) == (0x1234u | (5u << 16) | (2u << 26) | (2u << 29)),
        "PerfDebugRecMeta bit-field layout does not match the vectorized packer");
    no_decode_ = env_flag("TT_METAL_PERF_DEBUG_NO_DECODE");
    read_only_ = env_flag("TT_METAL_PERF_DEBUG_READ_ONLY");
    stall_only_ = env_flag("TT_METAL_PERF_DEBUG_STALL_ONLY");
    die_after_ = env_u32("TT_METAL_PERF_DEBUG_WRITER_DIE_AFTER", 0);
    watchdog_ = std::chrono::seconds(env_u32("TT_METAL_PERF_DEBUG_WRITER_TIMEOUT_S", 120));
    // 2 GiB per stream ring: big enough that a whole capture's records fit, so the consumer side only
    // drops on truly pathological lag instead of on every heavy run.
    const uint64_t ring_recs = env_u64("TT_METAL_PERF_DEBUG_RING_RECS", 128ull << 20);
    for (uint32_t d = 0; d < devices_.size(); d++) {
        auto& dev = devices_[d];
        const uint32_t nl = dev.num_cores * profiler::kSpscNRiscDecode;
        TT_FATAL(nl <= kPerfDebugMaxLanes, "record lane field holds {} lanes, device has {}", kPerfDebugMaxLanes, nl);
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
            s->ring = std::make_unique<BroadcastRing<PerfDebugRec>>(ring_recs);
            s->decode.reset(dev.num_cores);
            s->decode.core_of_xy = dev.core_of_xy;
            s->last_zone_ts.assign(nl, 0);
            streams_.push_back(std::move(s));
        }
    }
}

PerfDebugReceiver::~PerfDebugReceiver() { shutdown(); }

void PerfDebugReceiver::start() {
    TT_FATAL(!started_, "receiver already started");
    started_ = true;
    const uint32_t nthreads =
        std::clamp<uint32_t>(env_u32("TT_METAL_PERF_DEBUG_DECODE_THREADS", streams_.size()), 1, streams_.size());
    for (uint32_t t = 0; t < nthreads; t++) {
        std::vector<Stream*> owned;
        for (uint32_t i = t; i < streams_.size(); i += nthreads) {
            owned.push_back(streams_[i].get());
        }
        decode_threads_.emplace_back(&PerfDebugReceiver::decode_thread, this, std::move(owned));
    }
}

bool PerfDebugReceiver::decode_pass(Stream& s) {
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
    const size_t first_words = view.first_bytes / 4;
    auto& w = s.ring->writer();
    uint64_t pos = w.position();
    const uint64_t pos0 = pos;
    const uint32_t dev = s.dev;
    const bool sink = !stall_only_;
    // Pass-local stats, folded into the stream once per pass: the NT ring stores go through casted
    // pointers, so per-record updates against Stream fields cannot stay in registers (the compiler must
    // assume aliasing) -- measured at ~20% of the decode profile as a load-add-store per marker.
    uint64_t zone_markers = 0, stall_zones = 0, order_regressions = 0;
    uint64_t min_ts = s.min_zone_ts, max_ts = s.max_zone_ts;
    uint64_t* const last_ts = s.last_zone_ts.data();

    auto emit = [&](uint32_t lane, uint32_t type, uint32_t hash, uint64_t ts, uint32_t prog) {
        PerfDebugRecType rt = PerfDebugRecType::ZoneTotal;
        if (type != PP_ZONE_TOTAL) {
            rt = type == PP_ZONE_START ? PerfDebugRecType::ZoneStart : PerfDebugRecType::ZoneEnd;
            zone_markers++;
            stall_zones += (hash == 0x7FFFu && type == PP_ZONE_START) ? 1 : 0;
            if (ts < last_ts[lane]) {
                order_regressions++;
            } else {
                last_ts[lane] = ts;
            }
            if (min_ts == 0) {
                min_ts = ts;
            }
            max_ts = ts;
        }
        if (sink) {
            w.emit_store(pos++, PerfDebugRec{ts, {hash, lane, dev, rt}, prog});
        }
    };
    auto emit_data = [&](uint32_t lane,
                         uint32_t type,
                         uint32_t id,
                         uint64_t ts,
                         uint32_t prog,
                         const uint32_t* payload,
                         uint32_t n) {
        if (!sink) {
            return;
        }
        const PerfDebugRecType rt = type == PP_DATA ? PerfDebugRecType::Data : PerfDebugRecType::Event;
        w.emit_store(pos++, PerfDebugRec{ts, {id & 0xFFFFu, lane, dev, rt}, prog});
        w.emit_store(
            pos++, PerfDebugRec{(static_cast<uint64_t>(id) << 32) | n, {0, lane, dev, PerfDebugRecType::Ext}, prog});
        for (uint32_t k = 0; k < (n + 1) / 2; k++) {
            const uint64_t hi = payload[2 * k];
            const uint64_t lo = (2 * k + 1 < n) ? payload[2 * k + 1] : 0;
            w.emit_store(pos++, PerfDebugRec{(hi << 32) | lo, {0, lane, dev, PerfDebugRecType::Cont}, prog});
        }
    };

#if defined(__AVX2__)
    auto emit_zones8 = [&](uint32_t lane, uint32_t th, uint32_t prog, __m256i w0s, __m256i w1s) {
        zone_markers += 8;
        stall_zones += static_cast<uint32_t>(__builtin_popcount(static_cast<uint32_t>(
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(w0s, _mm256_set1_epi32(0x7FFF)))))));
        // Order invariant at block endpoints only: the producer guarantees monotonicity inside a run, so
        // in-block inversions would need a producer bug, while the boundary compare still catches every
        // head-mirror/resync error class. th is block-constant, so comparing on it is exact.
        const uint64_t ts_first =
            (static_cast<uint64_t>(th) << 32) | static_cast<uint32_t>(_mm256_extract_epi32(w1s, 0));
        const uint64_t ts_last =
            (static_cast<uint64_t>(th) << 32) | static_cast<uint32_t>(_mm256_extract_epi32(w1s, 7));
        order_regressions += ts_first < last_ts[lane] ? 1 : 0;
        last_ts[lane] = ts_last;
        if (min_ts == 0) {
            min_ts = ts_first;
        }
        max_ts = ts_last;
        if (!sink) {
            return;
        }
        // meta = hash | type bit (wire bit 27 -> record bit 29) | lane | dev | ZoneStart base. A record's
        // two quadwords are (th<<32|w1) and (prog<<32|meta), so interleaving w1s/metas against the two
        // splatted constants at 32-bit then pairing at 64-bit yields finished records with no widening.
        const uint32_t meta_base = (lane << 16) | (dev << 26) | (1u << 29);
        const __m256i meta = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_and_si256(w0s, _mm256_set1_epi32(0xFFFF)),
                _mm256_slli_epi32(_mm256_and_si256(w0s, _mm256_set1_epi32(0x08000000)), 2)),
            _mm256_set1_epi32(static_cast<int>(meta_base)));
        const __m256i th32 = _mm256_set1_epi32(static_cast<int>(th));
        const __m256i prog32 = _mm256_set1_epi32(static_cast<int>(prog));
        const __m256i a_lo = _mm256_unpacklo_epi32(w1s, th32);  // q0 of records 0,1 | 4,5
        const __m256i a_hi = _mm256_unpackhi_epi32(w1s, th32);  // q0 of records 2,3 | 6,7
        const __m256i b_lo = _mm256_unpacklo_epi32(meta, prog32);
        const __m256i b_hi = _mm256_unpackhi_epi32(meta, prog32);
        const __m256i r04 = _mm256_unpacklo_epi64(a_lo, b_lo);
        const __m256i r15 = _mm256_unpackhi_epi64(a_lo, b_lo);
        const __m256i r26 = _mm256_unpacklo_epi64(a_hi, b_hi);
        const __m256i r37 = _mm256_unpackhi_epi64(a_hi, b_hi);
        auto slot = [&](uint64_t p) { return reinterpret_cast<__m128i*>(w.emit_slot_ptr(p)); };
        _mm_stream_si128(slot(pos + 0), _mm256_castsi256_si128(r04));
        _mm_stream_si128(slot(pos + 1), _mm256_castsi256_si128(r15));
        _mm_stream_si128(slot(pos + 2), _mm256_castsi256_si128(r26));
        _mm_stream_si128(slot(pos + 3), _mm256_castsi256_si128(r37));
        _mm_stream_si128(slot(pos + 4), _mm256_extracti128_si256(r04, 1));
        _mm_stream_si128(slot(pos + 5), _mm256_extracti128_si256(r15, 1));
        _mm_stream_si128(slot(pos + 6), _mm256_extracti128_si256(r26, 1));
        _mm_stream_si128(slot(pos + 7), _mm256_extracti128_si256(r37, 1));
        pos += 8;
    };
#endif

    const uint64_t t0 = tsc_now();
    const size_t total_words = first_words + view.second_bytes / 4;
    alignas(64) uint32_t bounce[profiler::kSpscMaxFrameWords];
    // Headers never split across the spans: frames start on page boundaries and the FIFO wraps on one.
    auto word_at = [&](size_t o) { return o < first_words ? view.first[o] : view.second[o - first_words]; };
    size_t o = 0;
    uint32_t frames = 0, pages_done = 0, acked_pages = 0;
    while (o + kernel_profiler::SPSC_SPAN_PREFIX_WORDS <= total_words) {
        const uint32_t w1 = word_at(o + 1);
        if (!pp_is_bulkspan(word_at(o)) || w1 < kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE ||
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
        if (frame != bounce) {
            for (uint32_t line = 0; line < 6; line++) {
                profiler::spsc_prefetch(frame + fw + 16 * line);
            }
        }
        if (sink) {
            w.emit_reserve(pos + fw);
        }
#if defined(__AVX2__)
        const uint32_t payload =
            profiler::spsc_decode_frame(s.decode, frame, emit, emit_data, profiler::SpscIgnoreProg{}, emit_zones8);
#else
        const uint32_t payload = profiler::spsc_decode_frame(s.decode, frame, emit, emit_data);
#endif
        if (payload != 0 && payload != w1) {
            s.bad_frames++;
        }
        o += fw;
        pages_done += fw / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
        frames++;
        if (frames % kAckBatchFrames == 0) {
            s.sock->pop(pages_done - acked_pages, true);
            acked_pages = pages_done;
        }
    }
    if (pos != pos0) {
        w.emit_commit(pos);
        s.records += pos - pos0;
    }
    s.zone_markers += zone_markers;
    s.stall_zones += stall_zones;
    s.order_regressions += order_regressions;
    s.min_zone_ts = min_ts;
    s.max_zone_ts = max_ts;
    s.decode_ticks += tsc_now() - t0;
    if (pages_done > acked_pages) {
        s.sock->pop(pages_done - acked_pages, true);
    }
    if (pages_done == 0) {
        if (s.producers_done.load(std::memory_order_acquire)) {
            if (!s.desync_warned) {
                s.desync_warned = true;
                log_warning(
                    tt::LogMetal,
                    "[perf-debug receiver] d{}/s{}: {} pages (a partial frame) remain after the drainers "
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

void PerfDebugReceiver::decode_thread(std::vector<Stream*> streams) {
    std::string name = "pd-dec:";
    for (Stream* s : streams) {
        name += std::to_string(s->dev) + "." + std::to_string(s->sock_idx) + ",";
    }
    name.pop_back();
    tracy::SetThreadName(name.c_str());
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
            if (decode_pass(*s)) {
                any = true;
                data_passes++;
                s->last_progress = std::chrono::steady_clock::now();
            } else if (
                !s->producers_done.load(std::memory_order_relaxed) && !s->watchdog_fired &&
                std::chrono::steady_clock::now() - s->last_progress > watchdog_) {
                s->watchdog_fired = true;
                log_warning(
                    tt::LogMetal,
                    "[perf-debug receiver] d{}/s{}: no data for {} s while producers are live",
                    s->dev,
                    s->sock_idx,
                    watchdog_.count());
                if (cfg_.starvation_diagnostic) {
                    cfg_.starvation_diagnostic(s->dev, s->sock_idx);
                }
            }
        }
        if (all_retired) {
            break;
        }
        if (die_after_ != 0 && data_passes >= die_after_) {
            log_warning(
                tt::LogMetal,
                "[perf-debug receiver] {}: TEST HOOK exiting after {} passes; acks stop here",
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

PerfDebugConsumerHandle PerfDebugReceiver::add_consumer(std::string name, PerfDebugRecordCallback cb) {
    TT_FATAL(!t_in_consumer, "add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> lk(consumers_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->cb = std::move(cb);
    c->handle = next_handle_++;
    c->thread = std::thread(&PerfDebugReceiver::consumer_thread, this, std::ref(*c));
    consumers_.push_back(std::move(c));
    return consumers_.back()->handle;
}

void PerfDebugReceiver::remove_consumer(PerfDebugConsumerHandle handle) {
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
            "[perf-debug receiver] consumer \"{}\" removed having dropped {} records",
            victim->name,
            victim->dropped);
    }
}

void PerfDebugReceiver::consumer_thread(Consumer& c) {
    const std::string name = "pd-con:" + c.name;
    tracy::SetThreadName(name.c_str());
    t_in_consumer = true;
    std::vector<BroadcastRing<PerfDebugRec>::Reader> readers;
    readers.reserve(streams_.size());
    for (auto& s : streams_) {
        readers.push_back(s->ring->make_reader());
    }
    std::vector<PerfDebugRec> scratch(kConsumerScratchRecs);
    std::vector<uint64_t> last_dropped(readers.size(), 0);
    IdleBackoff backoff(kConsumerSleepCapUs);
    for (;;) {
        bool any = false;
        for (size_t r = 0; r < readers.size(); r++) {
            auto got = readers[r].read_batch(std::span<PerfDebugRec>(scratch));
            if (got.empty()) {
                continue;
            }
            any = true;
            std::call_once(names_once_, [this] {
                if (cfg_.load_zone_names) {
                    cfg_.load_zone_names(ctx_.zone_names);
                }
            });
            c.delivered += got.size();
            const uint64_t dropped = readers[r].dropped();
            const PerfDebugRecordBatch batch{got, dropped - last_dropped[r], &ctx_};
            last_dropped[r] = dropped;
            try {
                c.cb(batch);
            } catch (const std::exception& e) {
                log_warning(tt::LogMetal, "[perf-debug receiver] consumer \"{}\" threw: {}", c.name, e.what());
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
}

void PerfDebugReceiver::notify_producers_done(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->dev == device_index && s->sock_idx == socket_index) {
            s->producers_done.store(true, std::memory_order_release);
        }
    }
}

void PerfDebugReceiver::shutdown() {
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

std::vector<uint32_t> PerfDebugReceiver::final_lane_heads(uint32_t device_index) const {
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

void PerfDebugReceiver::log_report() const {
    uint64_t total_pages = 0, total_wire_words = 0, total_zone_markers = 0, total_resync_words = 0;
    uint64_t busy_ticks = 0, order_regressions = 0;
    uint64_t first_tsc = 0, last_tsc = 0;
    for (const auto& sp : streams_) {
        const Stream& s = *sp;
        total_pages += s.pages;
        total_wire_words += s.decode.live_words;
        total_zone_markers += s.zone_markers;
        total_resync_words += s.decode.resync_words;
        order_regressions += s.order_regressions;
        busy_ticks = std::max(busy_ticks, s.decode_ticks);
        if (s.first_data_tsc != 0 && (first_tsc == 0 || s.first_data_tsc < first_tsc)) {
            first_tsc = s.first_data_tsc;
        }
        last_tsc = std::max(last_tsc, s.last_commit_tsc);
        log_info(
            tt::LogMetal,
            "[perf-debug receiver] d{}/s{}: {} frames ({:.1f} MB) in {} passes | decode {:.1f} ms | {} records "
            "({} zones, {} stall-zones) | resyncs {} ({} words) | head-lag {} | anomalies {} | bad frames {} | "
            "unknown-core frames {} | order regressions {} [MUST be 0]",
            s.dev,
            s.sock_idx,
            s.frames,
            s.pages * static_cast<double>(kPageBytes) / 1e6,
            s.passes,
            ticks_to_ms(s.decode_ticks),
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
    }
    uint64_t consumer_drops = 0;
    for (const auto& c : consumers_report_) {
        consumer_drops += c->dropped;
        log_info(
            tt::LogMetal,
            "[perf-debug receiver] consumer \"{}\": {} delivered, {} dropped",
            c->name,
            c->delivered,
            c->dropped);
    }
    const double busy_ms = ticks_to_ms(busy_ticks);
    const double wall_ms = (first_tsc != 0 && last_tsc > first_tsc) ? ticks_to_ms(last_tsc - first_tsc) : 0.0;
    const double d2h_gb = total_pages * static_cast<double>(kPageBytes) / 1e9;
    const double wire_gb = total_wire_words * 4.0 / 1e9;
    const double mzones = total_zone_markers / 2.0 / 1e6;
    auto rate = [](double num, double ms) { return ms > 0.0 ? num / (ms / 1e3) : 0.0; };
    log_info(
        tt::LogMetal,
        "[perf-debug receiver] SUSTAINED THROUGHPUT: busy {:.1f} ms (max decode thread) -> {:.2f} GB/s D2H | "
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
                "[perf-debug receiver] device {} zone window {:.1f} ms (first->last zone @ {:.6f} GHz): {:.2f} "
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
        "[perf-debug receiver] loss: decode resyncs {} words | consumer ring drops {} | order regressions {} "
        "[device credit-timeout drops reported by the profiler above; all zero = lossless capture]",
        total_resync_words,
        consumer_drops,
        order_regressions);
}

}  // namespace tt::tt_metal::perf_debug
