// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_receiver.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
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
#include "tools/profiler/perf_debug_env.hpp"
#include "tools/profiler/spsc_packet.h"

namespace tt::tt_metal::perf_debug {

void PerfDebugReceiver::StallIdMirror::refresh() {
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
        // Rebuild at <= 25% load so the miss path is one probe. Sized generously: a few dozen ids.
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

// Credit-return quantum: pop+ack every 8 decoded frames (about one mover push) instead of once per pass,
// so the device sees credit at decode pace rather than in whole-pass steps.
constexpr uint32_t kAckBatchFrames = 8;
// Commit quantum, in ring records. MUST STAY 1. Batching commits leaves the carry holding a partial 64 B
// line across passes, but decode_pass re-seeds ntc.line_off on entry, so the carry's line stops starting
// where it believes it does and its stores land misplaced: consumers read misaligned records -- zone
// durations rendered in seconds -- while every decode-side counter (records, zones, anomalies, order
// regressions) stays clean, because none of them read the ring back. Batching is not even faster:
// 1352-1360 ms at 1 against 1365 ms at 32768.
constexpr uint32_t kCommitBatchRecs = 1;
// Per-pass peek window (~680 KB). Every complete frame inside it is consumed -- pages peeked but not
// consumed are clflushed again by the next peek, so the only re-flush waste is a partial tail frame.
constexpr uint32_t kMaxPagesPerPass = 64 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageBytes = kernel_profiler::SPSC_SPAN_PAGE_WORDS * 4;
constexpr size_t kConsumerScratchRecs = 1 << 16;
constexpr uint32_t kEmptyPollsBeforeSleep = 1000;
// Decode threads probe the FIFO at least this often when idle; consumers are latency-tolerant and may
// sleep longer. Anything under ~50 us needs the timer slack shrunk or sleep_for quietly rounds up to it.
// Load-bearing, and the wakeup cost is the point: at 5 us each thread churns ~80 K context switches/s
// (1.3 M/s across 16 streams, ~6 cores) but credit keeps flowing through the micro-gaps mid-burst where
// avail briefly hits 0. Raising it to 200 us to reclaim those cores doubled the movers' worst credit wait
// (1.3 -> 2.7 ms), took producer stalls from 600-1257 to 1178-4732 and filled five DRAM rings instead of
// two -- the sleep becomes a window in which no credit is returned at all.
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

PerfDebugReceiver::PerfDebugReceiver(ReceiverConfig config, std::vector<ReceiverDeviceConfig> devices) :
    cfg_(std::move(config)), devices_(std::move(devices)) {
    TT_FATAL(devices_.size() <= kPerfDebugMaxDevices, "record dev field holds {} devices", kPerfDebugMaxDevices);
    // The scalar decode packs meta through the bit-field; the AVX2 path packs it by hand, so pin the layout.
    const PerfDebugRawRecMeta meta_probe{0, 5, 2, PerfDebugRawRecType::ZoneEnd};
    TT_FATAL(
        std::bit_cast<uint32_t>(meta_probe) == ((5u << 16) | (2u << 26) | (2u << 29)),
        "PerfDebugRawRecMeta bit-field layout does not match the vectorized packer");
    no_decode_ = env_flag("TT_METAL_PERF_DEBUG_NO_DECODE");
    read_only_ = env_flag("TT_METAL_PERF_DEBUG_READ_ONLY");
    stall_only_ = env_flag("TT_METAL_PERF_DEBUG_STALL_ONLY");
    die_after_ = env_u32("TT_METAL_PERF_DEBUG_WRITER_DIE_AFTER", 0);
    watchdog_ = std::chrono::seconds(env_u32("TT_METAL_PERF_DEBUG_WRITER_TIMEOUT_S", 120));
    // 2 GiB per stream ring: big enough that a whole capture's records fit, so the consumer side only
    // drops on truly pathological lag instead of on every heavy run.
    // 16 Mi records x 24 B = ~403 MB per stream, so sixteen streams (eight devices, two sockets each) is
    // ~6.4 GB; 32 Mi would be ~12.9 GB. At the measured ~78 Mzones/s per stream that is ~215 ms of consumer
    // lag tolerance against a ~240 ms burst, and a reader that falls further behind is overwritten and
    // counted, never blocking the decode thread.
    const uint64_t ring_recs = env_u64("TT_METAL_PERF_DEBUG_RING_RECS", 16ull << 21);
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
            // Slots are faulted by prefault_rings() below, not by the decode thread.
            s->ring = std::make_unique<BroadcastRing<PerfDebugRawRec>>(
                ring_recs, BroadcastRing<PerfDebugRawRec>::DeferSlotInit{});
            s->ring_node = dev.numa_node;
            s->decode.reset(dev.num_cores);
            s->decode.core_of_xy = dev.core_of_xy;
            s->last_zone_ts.assign(nl, 0);
            streams_.push_back(std::move(s));
        }
    }
    prefault_rings();
}

PerfDebugReceiver::~PerfDebugReceiver() { shutdown(); }

// Pin each ring's pages to its device's node and fault them HERE, before the workload can produce. The
// fault is ~400 MB per stream: left on the decode thread it lands after the producers are already running
// (invisible on a model, where setup hides it; a burst of first-iteration producer stalls on a synthetic
// that starts immediately). numa_tonode_memory makes placement independent of the touching thread, so the
// prefault can run anywhere -- one thread per stream, bound to that node so the zeroing is local.
void PerfDebugReceiver::prefault_rings() {
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

// Attributed as one region with the decode block it calls: the emitters below hand every record to
// SpscNtCarry::put3, and an attribute mismatch there would stop put3 inlining and spill the carry's
// 512-bit accumulator to memory on the scalar path.
#pragma clang attribute push(__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))), apply_to = function)
bool PerfDebugReceiver::decode_pass(Stream& s) {
    const uint32_t avail = s.sock->pages_available();
    if (avail == 0) {
        if (auto& w0 = s.ring->writer(); s.wpos != w0.position()) {  // input dry: deliver what is pending
            s.ntc.flush_tail();
            w0.emit_commit(s.wpos);
        }
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
    const size_t first_words = view.first_bytes / 4;
    auto& w = s.ring->writer();
    uint64_t pos = s.wpos;  // the written position; the writer's own only tracks the batched commits
    const uint64_t pos0 = pos;
    const uint32_t dev = s.dev;
    const bool sink = !stall_only_;
    const bool count_stalls = stall_only_;
    // The PRODUCER-STALL ids, by ELF-resolved NAME; cheap (an empty delta) except when an ELF just loaded.
    s.stall_ids.refresh();
    // Raw locals for the per-marker probe: the emit loops store through casted pointers, so anything
    // reached via an object or vector would be reloaded from memory after every NT store (the compiler
    // must assume aliasing). Locals whose address never escapes stay in registers.
    const uint32_t* const stall_tab = s.stall_ids.table.empty() ? nullptr : s.stall_ids.table.data();
    const uint32_t stall_mask = s.stall_ids.mask;
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
    // Ring geometry as locals, same aliasing argument: one address computation per BLOCK in the
    // vectorized sink instead of a slot_at() member-pointer chase per record.
    const uint64_t ring_mask = s.ring->capacity() - 1;
    char* const ring_base = reinterpret_cast<char*>(w.emit_slot_ptr(0));
    // EVERY emitter below writes through this. A 24 B record means whole-line alignment depends on pos,
    // which output arity scrambles; the carry holds the sub-line remainder in a register so each ring line
    // is streamed exactly once, aligned, whatever the record mix. Mixing a cached store into those lines is
    // what cost 4x, so no emitter may write the ring directly.
    // One carry per stream, persisting across passes (see the Stream field). Bound to the ring once.
    profiler::SpscNtCarry& ntc = s.ntc;
    if (ntc.ring == nullptr) {
        ntc.ring = reinterpret_cast<uint8_t*>(ring_base);
        ntc.cap_bytes = (ring_mask + 1ull) * sizeof(PerfDebugRawRec);
        ntc.line_off = pos * sizeof(PerfDebugRawRec) % ntc.cap_bytes;  // pos is 0 on the first pass, so this is aligned
    }
    const auto meta_hi = [dev](uint32_t lane, PerfDebugRawRecType rt) {
        return static_cast<uint64_t>((lane << 16) | (dev << 26) | (static_cast<uint32_t>(rt) << 29)) << 32;
    };
    // Pass-local stats, folded into the stream once per pass: the NT ring stores go through casted
    // pointers, so per-record updates against Stream fields cannot stay in registers (the compiler must
    // assume aliasing) -- measured at ~20% of the decode profile as a load-add-store per marker.
    uint64_t zone_markers = 0, stall_zones = 0, order_regressions = 0;
    uint64_t min_ts = s.min_zone_ts, max_ts = s.max_zone_ts;
    uint64_t* const last_ts = s.last_zone_ts.data();

    auto emit = [&](uint32_t lane, uint32_t type, uint32_t zone_id, uint64_t ts, uint32_t prog, uint32_t duration) {
        PerfDebugRawRecType rt = PerfDebugRawRecType::ZoneTotal;
        if (type == PP_ZONE_ATOMIC) {
            rt = PerfDebugRawRecType::Zone;
        } else if (type != PP_ZONE_TOTAL) {
            rt = type == PP_ZONE_START ? PerfDebugRawRecType::ZoneStart : PerfDebugRawRecType::ZoneEnd;
        }
        if (type != PP_ZONE_TOTAL) {
            // In HALVES: a ZONE_ATOMIC record is a whole zone, a START/END is half of one, and every
            // consumer of this counter divides by two. Counting records instead halved every atomic-path
            // zone tally (measured: 27.6 M reported against 55.0 M ZONE_ATOMIC records captured).
            zone_markers += type == PP_ZONE_ATOMIC ? 2 : 1;
            // PRODUCER-STALL, matched by ELF-resolved NAME via the id table: a producer RISC blocked on
            // a FULL ring. STALL_ONLY mode only -- on a normal run this per-marker probe is redundant
            // (the control plane reads the workers' own L1 stall counters at teardown) and measures ~10%
            // of the decode wall, so the hot path does not pay for a diagnostic the device already keeps.
            stall_zones += (count_stalls && type == PP_ZONE_START && is_stall(zone_id)) ? 1 : 0;
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
            ntc.put3(ts, meta_hi(lane, rt) | zone_id, (static_cast<uint64_t>(duration) << 32) | prog);
            pos++;
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
        const uint64_t pg = prog;
        // PP_EVENT is payload-less by wire shape, so its Ext record carried nothing the Event record
        // does not: one record IS the whole packet, and no Ext/Cont ever follows an Event.
        if (type != PP_DATA) {
            ntc.put3(ts, meta_hi(lane, PerfDebugRawRecType::Event) | id, pg);
            pos += 1;
            return;
        }
        const uint64_t q1 = meta_hi(lane, PerfDebugRawRecType::Data) | id;
        // Ext carries the payload COUNT in its id field and payload words 1-2 in its ts field: the id
        // repetition it used to carry is already in the head record, so the common short DATA is two
        // records, not three -- and the pair goes out as ONE carry put (a put3 is a full carry rotation).
        const uint64_t q4 = meta_hi(lane, PerfDebugRawRecType::Ext) | n;
        const uint64_t hi0 = n >= 1 ? payload[0] : 0;
        const uint64_t lo0 = n >= 2 ? payload[1] : 0;
        __m512i v[2];
        v[0] = _mm512_set_epi64(
            0, 0, static_cast<long long>(pg), static_cast<long long>(q4),
            static_cast<long long>((hi0 << 32) | lo0), static_cast<long long>(pg), static_cast<long long>(q1),
            static_cast<long long>(ts));
        v[1] = v[0];
        ntc.put(v, 6);
        pos += 2;
        for (uint32_t k = 2; k < n; k += 2) {
            const uint64_t hi = payload[k];
            const uint64_t lo = (k + 1 < n) ? payload[k + 1] : 0;
            ntc.put3((hi << 32) | lo, meta_hi(lane, PerfDebugRawRecType::Cont), pg);
            pos++;
        }
    };

#if defined(__AVX2__)
    auto emit_zones8 = [&](uint32_t lane, uint32_t th, uint32_t prog, __m256i w0s, __m256i w1s) {
        zone_markers += 8;
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
        // A 24 B record is three quadwords: q0 = ts (th<<32 | w1), q1 = meta<<32 | id27, q2 = prog.
        // meta = lane | dev | type, with type = ZoneStart + the wire's END bit (an ADD, not an OR -- the
        // type field is a small integer, not a bit set). 24 B slots alternate 16-byte alignment, so the
        // stores are 8-byte movnti rather than 16-byte movntdq; the id/meta math stays vectorized.
        // PRODUCER-STALL is checked per START in the same scalar pass -- one L1 probe on the miss path.
        const __m256i ids = _mm256_and_si256(w0s, _mm256_set1_epi32(0x07FFFFFF));
        const uint32_t meta_base = (lane << 16) | (dev << 26) | (1u << 29);  // type = ZoneStart
        const __m256i meta = _mm256_add_epi32(
            _mm256_slli_epi32(_mm256_and_si256(w0s, _mm256_set1_epi32(0x08000000)), 2),  // END: +1 in type
            _mm256_set1_epi32(static_cast<int>(meta_base)));
        alignas(32) uint32_t w1_arr[8];
        alignas(32) uint32_t id_arr[8];
        alignas(32) uint32_t meta_arr[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(w1_arr), w1s);
        _mm256_store_si256(reinterpret_cast<__m256i*>(id_arr), ids);
        _mm256_store_si256(reinterpret_cast<__m256i*>(meta_arr), meta);
        if (count_stalls) {  // STALL_ONLY mode only -- see the scalar emit path
            const uint32_t end_mask = static_cast<uint32_t>(
                _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(w0s, 4))));  // bit27 -> sign bit: 1 = ZONE_END
            for (int k = 0; k < 8; k++) {
                stall_zones += (((end_mask >> k) & 1u) == 0 && is_stall(id_arr[k])) ? 1 : 0;
            }
        }
        if (!sink) {
            return;
        }
        const uint64_t th_hi = static_cast<uint64_t>(th) << 32;
        const uint64_t prog64 = prog;
        for (uint32_t k = 0; k < 8; k++) {
            ntc.put3(th_hi | w1_arr[k], (static_cast<uint64_t>(meta_arr[k]) << 32) | id_arr[k], prog64);
        }
        pos += 8;
    };
    // Forwards into the 512-bit block and keeps every piece of ring/bookkeeping state
    // here. Returns records consumed so the decoder can advance; 0 means "use your own path", which is how
    // the ring-wrap case is handed back without the block needing to know about wrapping.
    // Forwards into the block and keeps ring/bookkeeping state here. No wrap bailout and no alignment
    // gate: the carry writer handles both, and the block emits any n, so there is nothing to fall back to.
    auto emit_atomic16 = [&](uint32_t lane, uint32_t th, uint32_t prog, const uint32_t* src, uint32_t avail,
                             uint32_t max_recs) -> uint32_t {
        const auto a = profiler::spsc_atomic16_avx512(src, avail, max_recs, th, prog, lane, dev, ntc);
        if (a.n == 0) {
            return 0;
        }
        zone_markers += a.n * 2;  // halves: every record in an atomic block is a whole zone
        order_regressions += a.ts_first < last_ts[lane] ? 1 : 0;
        last_ts[lane] = a.ts_last;
        if (min_ts == 0) {
            min_ts = a.ts_first;
        }
        max_ts = a.ts_last;
        pos += a.n;
        return a.n;
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
        if (frame != bounce) {
            for (uint32_t line = 0; line < 6; line++) {
                profiler::spsc_prefetch(frame + fw + 16 * line);
            }
        }
        if (sink) {
            w.emit_reserve(pos + fw);
        }
        const uint64_t tf0 = tsc_now();
#if defined(__AVX2__)
        const uint32_t payload = profiler::spsc_decode_frame(
            s.decode,
            frame,
            emit,
            emit_data,
            profiler::SpscIgnoreProg{},
            emit_zones8,
            profiler::SpscNoAtomic8{},
            emit_atomic16,
            fw);
#else
        const uint32_t payload = profiler::spsc_decode_frame(s.decode, frame, emit, emit_data);
#endif
        s.frame_ticks += tsc_now() - tf0;
        if (payload != 0 && payload != w1) {
            s.bad_frames++;
        }
        o += fw;
        pages_done += fw / kernel_profiler::SPSC_SPAN_PAGE_WORDS;
        frames++;
        if (frames % kAckBatchFrames == 0) {
            const uint64_t ta0 = tsc_now();
            s.sock->pop(pages_done - acked_pages, true);
            s.ack_ticks += tsc_now() - ta0;
            acked_pages = pages_done;
        }
    }
    if (pos != pos0) {
        s.records += pos - pos0;
        s.wpos = pos;
    }
    if (s.wpos != w.position() && (pages_done == 0 || s.wpos - w.position() >= kCommitBatchRecs)) {
        ntc.flush_tail();  // the sub-line remainder must be visible before the commit advertises it
        w.emit_commit(s.wpos);
    }
    s.zone_markers += zone_markers;
    s.stall_zones += stall_zones;
    s.order_regressions += order_regressions;
    s.min_zone_ts = min_ts;
    s.max_zone_ts = max_ts;
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
#pragma clang attribute pop

void PerfDebugReceiver::decode_thread(std::vector<Stream*> streams) {
    // FIRST-TOUCH OUR OWN RING. First touch fixes a page's NUMA node for the life of the mapping, and this
    // thread NT-stores into the ring at tens of GB/s while reading the FIFO at tens more. Constructed on
    // the main thread instead, every stream's ring landed on whichever node built the receiver, so the
    // decode threads the scheduler placed on the other node crossed the interconnect on every store --
    // measured as one stream taking 172 ms of frame time against a peer's 111 ms for the same work, and
    // that slowest stream is what gates the pipeline: its mover blocks on credit, its ring fills, its
    // producers stall. Faulting here makes the memory follow the thread, with no affinity policy at all.
    // Bound to the node FIRST, or first touch pins the ring wherever the scheduler happened to start us and
    // a later migration turns every ring store into an interconnect crossing.
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

PerfDebugConsumerHandle PerfDebugReceiver::add_raw_consumer(std::string name, PerfDebugRawRecordCallback cb) {
    TT_FATAL(!t_in_consumer, "add_raw_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> lk(consumers_mu_);
    auto c = std::make_unique<Consumer>();
    c->name = std::move(name);
    c->raw_cb = std::move(cb);
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
    set_os_thread_name(name);
    t_in_consumer = true;
    std::vector<BroadcastRing<PerfDebugRawRec>::Reader> readers;
    readers.reserve(streams_.size());
    for (auto& s : streams_) {
        readers.push_back(s->ring->make_reader());
    }
    std::vector<PerfDebugRawRec> scratch(kConsumerScratchRecs);
    std::vector<uint64_t> last_dropped(readers.size(), 0);
    // ---- The pairing stage (public consumers only) ------------------------------------------------
    // Zones are RAII scopes on the device, so per lane the raw stream obeys strict stack discipline:
    // push on ZoneStart, pop on ZoneEnd, and the pop's mate is the matching open. One stack per
    // (dev, lane), owned by THIS thread -- every consumer thread reads the whole ring independently,
    // so there is no sharing and no lock. A Zone record is emitted at END time; everything else
    // converts 1:1. All pairing cost lands here, on the consumer's own thread, never on decode.
    struct OpenZone {
        uint64_t ts;
        uint32_t id;
        uint32_t prog;
    };
    std::vector<std::vector<OpenZone>> stacks;
    if (c.raw_cb == nullptr) {
        stacks.resize(ctx_.devices.size() * kPerfDebugMaxLanes);
    }
    std::vector<PerfDebugRec> out;
    out.reserve(kConsumerScratchRecs);
    uint64_t unmatched_ends = 0;  // ZoneEnd with an empty stack (only possible after ring drops)
    uint64_t id_mismatches = 0;   // ZoneEnd whose id differs from the matching open: trust neither, drop
    auto pair_batch = [&](std::span<const PerfDebugRawRec> got) {
        out.clear();
        for (const PerfDebugRawRec& r : got) {
            const uint32_t si = r.meta.dev * kPerfDebugMaxLanes + r.meta.lane;
            switch (r.meta.type) {
                case PerfDebugRawRecType::ZoneStart: stacks[si].push_back({r.ts, r.id, r.prog}); break;
                case PerfDebugRawRecType::ZoneEnd: {
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
                    PerfDebugRec& o = out.emplace_back();
                    o.data.zone = {open.ts, r.ts - open.ts};
                    o.id = open.id;
                    o.meta = {0, r.meta.lane, r.meta.dev, PerfDebugRecType::Zone};
                    o.prog = open.prog;
                    break;
                }
                case PerfDebugRawRecType::Zone: {
                    // Already a complete zone (the device's atomic-zone path): ts is the END and duration
                    // is set, so it converts 1:1 with no stack. Without this case it fell into the default
                    // arm and every public consumer received it as a Data record with the duration dropped.
                    PerfDebugRec& o = out.emplace_back();
                    o.data.zone = {r.ts - r.duration, r.duration};
                    o.id = r.id;
                    o.meta = {0, r.meta.lane, r.meta.dev, PerfDebugRecType::Zone};
                    o.prog = r.prog;
                    break;
                }
                case PerfDebugRawRecType::ZoneTotal: {
                    PerfDebugRec& o = out.emplace_back();
                    o.data.sum = r.ts;
                    o.id = r.id;
                    o.meta = {0, r.meta.lane, r.meta.dev, PerfDebugRecType::ZoneTotal};
                    o.prog = r.prog;
                    break;
                }
                default: {  // Data / Event / Ext / Cont: 1:1
                    PerfDebugRec& o = out.emplace_back();
                    o.data.ts = r.ts;
                    o.id = r.id;
                    PerfDebugRecType t = PerfDebugRecType::Data;
                    if (r.meta.type == PerfDebugRawRecType::Event) {
                        t = PerfDebugRecType::Event;
                    } else if (r.meta.type == PerfDebugRawRecType::Ext) {
                        t = PerfDebugRecType::Ext;
                    } else if (r.meta.type == PerfDebugRawRecType::Cont) {
                        t = PerfDebugRecType::Cont;
                    }
                    o.meta = {0, r.meta.lane, r.meta.dev, t};
                    o.prog = r.prog;
                    break;
                }
            }
        }
    };
    IdleBackoff backoff(kConsumerSleepCapUs);
    for (;;) {
        bool any = false;
        for (size_t r = 0; r < readers.size(); r++) {
            auto got = readers[r].read_batch(std::span<PerfDebugRawRec>(scratch));
            if (got.empty()) {
                continue;
            }
            any = true;
            const uint64_t dropped = readers[r].dropped();
            const uint64_t dropped_delta = dropped - last_dropped[r];
            last_dropped[r] = dropped;
            try {
                if (c.raw_cb != nullptr) {
                    c.delivered += got.size();
                    c.raw_cb(PerfDebugRawRecordBatch{got, dropped_delta, &ctx_});
                } else {
                    pair_batch(got);
                    if (out.empty() && dropped_delta == 0) {
                        continue;  // a batch of nothing but ZoneStarts; the Zones come with their ends
                    }
                    c.delivered += out.size();
                    c.cb(PerfDebugRecordBatch{std::span<const PerfDebugRec>(out), dropped_delta, &ctx_});
                }
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
    // A lossless capture ends with every stack empty: the producers close every scope they open, and
    // the quiesce path drains to the last marker. Leftover opens mean records were lost (ring drops
    // for THIS consumer) or a start/end pair was corrupted -- say so rather than ending silently.
    uint64_t leftover_opens = 0;
    for (const auto& st : stacks) {
        leftover_opens += st.size();
    }
    if (leftover_opens != 0 || unmatched_ends != 0 || id_mismatches != 0) {
        log_warning(
            tt::LogMetal,
            "[perf-debug receiver] consumer \"{}\" pairing: {} zones left OPEN at shutdown, {} unmatched ends, "
            "{} start/end id mismatches [all MUST be 0 on a lossless capture]",
            c.name,
            leftover_opens,
            unmatched_ends,
            id_mismatches);
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
            "[perf-debug receiver] d{}/s{}: {} frames ({:.1f} MB) in {} passes | decode {:.1f} ms (frame {:.1f} + "
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
        const uint64_t vrec = s.decode.vec_zone_recs + s.decode.vec_atomic_recs;
        const uint64_t allrec = vrec + s.decode.scalar_recs;
        log_info(
            tt::LogMetal,
            "[perf-debug receiver] d{}/s{} decode paths: {:.1f}% vectorized ({} zone8 + {} atomic8), {} scalar, "
            "{} vec-block rejects",
            s.dev,
            s.sock_idx,
            allrec ? 100.0 * static_cast<double>(vrec) / static_cast<double>(allrec) : 0.0,
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
                "[perf-debug receiver] d{}/s{} atomic blocks: {} calls, {} recs, {:.2f} recs/call (max 16) | scalar by type: {}",
                s.dev, s.sock_idx, d.vec_atomic_calls, d.vec_atomic_recs,
                d.vec_atomic_calls ? double(d.vec_atomic_recs) / double(d.vec_atomic_calls) : 0.0, bt);
        }
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
