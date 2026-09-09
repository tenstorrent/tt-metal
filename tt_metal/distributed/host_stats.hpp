// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Per-stage instrumentation.
//
//   one way (3 stages)
//     1  t6 -> host                     Tensix posts payload + control into its arena
//     2  host -> remote host            libfabric moves it to the peer's RX arena
//     3  remote host -> remote t6       the peer writes it into the destination L1
//
//   round trip adds the mirror image (6 stages)
//     4  remote t6 -> remote host       the far core posts its reply
//     5  remote host -> host            libfabric brings it back
//     6  host -> t6                     we write it into the originating core's L1
//
// plus an aggregate for each. Named after physical events rather than code locations, so
// a number stays meaningful when a function moves.
//
// A host timestamp and a Tensix cycle counter are not the same clock. Subtracting them
// raw is arithmetic on unrelated quantities -- the same mistake as subtracting two
// unsynchronised hosts' CLOCK_MONOTONIC_RAW. An earlier version of this file dodged it by
// declaring stage 1 unmeasurable and reporting only "when the host noticed", which is not
// what was asked for and quietly understates the leg.
//
// So the device clock is calibrated against the host clock the same way the peer host's
// is (see host_clock.hpp): the kernel stamps its own cycle counter into an operand
// register, and a startup calibration establishes the cycles->ns scale and the offset,
// with an uncertainty bound carried alongside. Every cross-domain stage is reported WITH
// that bound.
//
#pragma once
#include <thread>
#include <array>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental {

inline uint64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t measure_clock_overhead_ns();

// The stages, in the order bytes pass through them. Dense enum with a name table so
// adding one cannot silently shift another's column in a CSV.
enum Hop : uint32_t {
    // --- one way ---
    kHopT6ToHost = 0,
    kHopHostToRemoteHost,
    kHopRemoteHostToRemoteT6,
    kHopOneWayTotal,
    kHopNotice,      // armed word visible -> a worker picked it up
    kHopDecode,      // validate + operand snapshot + route
    kHopStealWait,
    kHopL1Write,     // the noc_write half of stages 3/6, without the doorbell
    kHopDoorbell,    // the doorbell half
    kHopH2HRetire,
    kHopSendQueueWait,
    kHopPullWait,
    kHopD2HVisibility,
    kHopH2HPayloadAtPeer,
    kHopH2HCreditRaw,
    kHopH2HNet,
    kHopD2HFenced,
    kHopCount
};

constexpr uint32_t kHopStageCount = kHopOneWayTotal + 1;

inline const char* hop_name(uint32_t h) {
    switch (h) {
        case kHopT6ToHost: return "t6->host";
        case kHopHostToRemoteHost: return "host->remote_host";
        case kHopRemoteHostToRemoteT6: return "remote_host->remote_t6";
        case kHopOneWayTotal: return "ONEWAY_TOTAL";
        case kHopNotice: return "diag:notice";
        case kHopDecode: return "diag:decode";
        case kHopStealWait: return "diag:steal-wait";
        case kHopL1Write: return "diag:l1-write";
        case kHopDoorbell: return "diag:doorbell";
        case kHopH2HRetire: return "diag:h2h-retire";
        case kHopSendQueueWait: return "diag:sendq-wait";
        case kHopPullWait: return "diag:pull-wait";
        case kHopD2HVisibility: return "diag:d2h-visibility";
        case kHopH2HPayloadAtPeer: return "h2h:payload-at-peer";
        case kHopH2HCreditRaw: return "h2h:credit-raw";
        case kHopH2HNet: return "h2h:net";
        case kHopD2HFenced: return "diag:d2h-fenced";
        default: return "?";
    }
}

inline bool hop_crosses_device_clock(uint32_t h) {
    return h == kHopT6ToHost || h == kHopD2HVisibility || h == kHopD2HFenced ||
           h == kHopRemoteHostToRemoteT6;
}

inline bool hop_crosses_host_clock(uint32_t h) { return h == kHopH2HNet; }

inline bool hop_rate_is_bandwidth(uint32_t h) {
    switch (h) {
        case kHopT6ToHost:              // the producer's own copy into its arena
        case kHopRemoteHostToRemoteT6:  // MMIO write into the destination L1
        case kHopL1Write:               // the noc_write half of the above
        case kHopH2HRetire:             // post -> completion: the transfer, by construction
        case kHopD2HFenced:             // kHopT6ToHost with the read-back probe taken back out
            return true;
        default:
            return false;
    }
}

inline bool hop_samples_warmup_gated(uint32_t h) {
    return h < kHopCount;
}

struct Dist {
    uint64_t n = 0;
    uint64_t min = UINT64_MAX;
    uint64_t max = 0;
    uint64_t sum = 0;
    double mean = 0.0;
    double m2 = 0.0;

    void add(uint64_t v) {
        ++n;
        min = std::min(min, v);
        max = std::max(max, v);
        sum += v;
        mean = static_cast<double>(sum) / static_cast<double>(n);
    }

    void merge(const Dist& o) {
        if (o.n == 0) {
            return;
        }
        n += o.n;
        sum += o.sum;
        min = std::min(min, o.min);
        max = std::max(max, o.max);
        mean = static_cast<double>(sum) / static_cast<double>(n);
    }

    double stddev() const { return n > 1 ? std::sqrt(m2 / static_cast<double>(n - 1)) : 0.0; }
    double rel_stddev() const { return mean > 0.0 ? stddev() / mean : 0.0; }
};

struct Span {
    uint64_t first = UINT64_MAX;
    uint64_t last = 0;

    void add(uint64_t open_ns, uint64_t close_ns) {
        first = std::min(first, open_ns);
        last = std::max(last, close_ns);
    }

    void merge(const Span& o) {
        if (o.empty()) {
            return;
        }
        first = std::min(first, o.first);
        last = std::max(last, o.last);
    }

    bool empty() const { return first == UINT64_MAX || last <= first; }

    uint64_t ns() const { return empty() ? 0 : last - first; }
};

constexpr uint32_t kTraceBuckets = 2048;
constexpr uint32_t kTraceShiftDefault = 24;  // 2^24 ns = 16.777 ms per bucket

struct TraceBucket {
    uint64_t bytes = 0;   // payload confirmed in this bucket
    uint64_t ns_sum = 0;  // sum of the accompanying stage durations, for a per-bucket mean
    uint32_t n = 0;       // samples, so the mean has a denominator
    uint32_t reserved = 0;
};

struct alignas(64) WorkerStats {
    Dist hop[kHopCount];
    uint64_t hop_wire_bytes[kHopCount] = {};
    uint64_t hop_payload_bytes[kHopCount] = {};
    Span hop_window[kHopCount];
    uint64_t scanned = 0;
    uint64_t found = 0;
    uint64_t stolen = 0;
    uint64_t donated = 0;
    uint64_t bytes = 0;
    uint64_t timed_bytes = 0;
    uint64_t rejected[8] = {};
    uint64_t idle_spins = 0;
    uint64_t delivered = 0;   // messages written into a Tensix L1
    uint64_t tx_credit_skips = 0;
    TraceBucket trace[kTraceBuckets];
    uint64_t trace_clamped = 0;  // samples that landed past the last bucket

    std::vector<std::array<Dist, kHopCount>> ladder_windows;
    std::vector<uint64_t> ladder_window_bytes;  // payload sealed into each window
    const struct VolumeLadder* ladder_cfg = nullptr;
    struct LadderSync* ladder_sync = nullptr;  // non-null only when quiescing
    uint32_t ladder_workers = 0;
    uint64_t ladder_bytes = 0;                  // this worker's cumulative payload
    uint64_t ladder_window_start = 0;           // ladder_bytes when the current window opened
    uint32_t ladder_next = 0;                   // index of the next mark to cross
    char pad[64];
};

inline uint64_t ladder_now_ns() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

inline void ladder_seal_window(WorkerStats& ws) {
    ws.ladder_windows.emplace_back();
    std::copy(std::begin(ws.hop), std::end(ws.hop), ws.ladder_windows.back().begin());
    ws.ladder_window_bytes.push_back(ws.ladder_bytes - ws.ladder_window_start);
    ws.ladder_window_start = ws.ladder_bytes;
    for (uint32_t h = 0; h < kHopCount; ++h) {
        ws.hop[h] = Dist{};
    }
    ++ws.ladder_next;
}

struct LadderSync {
    static constexpr uint64_t kQuiesceBudgetNs = 100ull * 1000 * 1000;  // 100 ms

    std::atomic<uint64_t> bytes{0};       // pool-wide recorded payload
    std::atomic<uint32_t> arrived{0};     // workers sealed at the current checkpoint
    std::atomic<uint32_t> generation{0};  // bumped when a checkpoint completes
    std::atomic<uint32_t> next_mark{0};
    std::atomic<uint32_t> clean{0};       // checkpoints where every worker arrived in budget
    std::atomic<uint32_t> degraded{0};    // checkpoints that timed out and proceeded anyway
};

struct VolumeLadder {
    bool enabled = false;
    bool quiesced = false;       // pause at each checkpoint so boundaries are exact
    uint64_t chunk_bytes = 0;    // one message's payload
    uint64_t total_bytes = 0;    // the RECORDED volume this ladder spans
    uint64_t discarded_bytes = 0;  // what --steady dropped before the counters started
    std::vector<uint64_t> marks; // nominal cumulative thresholds, doubling from chunk_bytes
    uint32_t quiesce_clean = 0;
    uint32_t quiesce_degraded = 0;

    void build(uint64_t chunk, uint64_t total, uint32_t workers) {
        chunk_bytes = chunk;
        total_bytes = total;
        marks.clear();
        const uint64_t first = chunk * (workers == 0 ? 1u : workers);
        for (uint64_t v = first; v > 0 && v < total; v <<= 1) {
            marks.push_back(v);
        }
        if (total > 0) {
            marks.push_back(total);
        }
        enabled = !marks.empty();
    }

    uint64_t worker_mark(size_t i, uint32_t workers) const {
        const uint64_t w = workers == 0 ? 1u : workers;
        return marks[i] / w;
    }
};

inline void ladder_note_quiesced(WorkerStats& ws, LadderSync& sync, const VolumeLadder& cfg,
                                 uint32_t workers, uint64_t payload_bytes) {
    const uint64_t total = sync.bytes.fetch_add(payload_bytes, std::memory_order_acq_rel) + payload_bytes;
    ws.ladder_bytes += payload_bytes;

    const uint32_t mark = sync.next_mark.load(std::memory_order_acquire);
    if (mark >= cfg.marks.size() || total < cfg.marks[mark]) {
        return;
    }

    ladder_seal_window(ws);

    const uint32_t gen = sync.generation.load(std::memory_order_acquire);
    const uint32_t n = sync.arrived.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (n >= workers) {
        sync.clean.fetch_add(1, std::memory_order_relaxed);
        sync.arrived.store(0, std::memory_order_release);
        sync.next_mark.store(mark + 1, std::memory_order_release);
        sync.generation.fetch_add(1, std::memory_order_acq_rel);
        return;
    }

    const uint64_t start = ladder_now_ns();
    while (sync.generation.load(std::memory_order_acquire) == gen) {
        if (ladder_now_ns() - start > LadderSync::kQuiesceBudgetNs) {
            sync.degraded.fetch_add(1, std::memory_order_relaxed);
            sync.arrived.store(0, std::memory_order_release);
            sync.next_mark.store(mark + 1, std::memory_order_release);
            sync.generation.fetch_add(1, std::memory_order_acq_rel);
            return;
        }
        std::this_thread::yield();
    }
}

inline void ladder_note_message(WorkerStats& ws, bool recording, uint64_t payload_bytes) {
    if (!recording || ws.ladder_cfg == nullptr || !ws.ladder_cfg->enabled) {
        return;
    }
    const VolumeLadder& cfg = *ws.ladder_cfg;
    const uint32_t workers = ws.ladder_workers;

    if (cfg.quiesced && ws.ladder_sync != nullptr) {
        ladder_note_quiesced(ws, *ws.ladder_sync, cfg, workers, payload_bytes);
        return;
    }
    ws.ladder_bytes += payload_bytes;
    while (ws.ladder_next < cfg.marks.size() &&
           ws.ladder_bytes >= cfg.worker_mark(ws.ladder_next, workers)) {
        ladder_seal_window(ws);
    }
}

struct RunStats {
    std::vector<WorkerStats> per_worker;
    VolumeLadder ladder;
    uint64_t clock_overhead_ns = 0;
    uint64_t wall_ns = 0;

    uint64_t timed_ns = 0;
    uint32_t timed_iters = 0;    // iterations inside the bracket: iters - warmup
    uint32_t xfers_per_iter = 1;  // fabtests' show_perf() argument: 1 one-way, 2 round trip

    uint32_t window = 0;
    std::string sender_shape = "none";

    uint64_t device_clock_uncertainty_ns = 0;
    uint64_t host_clock_uncertainty_ns = 0;
    bool device_clock_valid = false;
    bool host_clock_valid = false;

    uint32_t payload_bytes = 0;
    uint32_t cores = 0;
    uint32_t iters = 0;
    std::string provider = "none";
    std::string mode = "oneway";  // oneway | roundtrip | local

    std::string run_id;           // unique per process
    std::string run_started_utc;

    double ns_per_cycle = 0.0;

    uint32_t warmup = 0;
    bool warmup_applied = false;

    uint32_t host_ident = 0;
    std::string role = "local";  // server | peer | local
    bool symmetric = false;
    bool tx_side = true;

    // H2D Leg implementation. "write" is the host pushing
    // into L1 over a WC TLB window; "socket" is one tt-metal H2DSocket per core in
    // DEVICE_PULL mode. Recorded per row because it lives NOWHERE ELSE in the file -- `mode`
    // is the run role (sym-tx/sym-rx) and `tag` is whatever the caller passed. Until this
    // column existed the only record of which approach a measurement came from was a `_sock`
    // suffix the sweep script put on the FILENAME, so renaming a file, merging two, or
    // running the binary by hand lost it with nothing to notice.
    //
    // Stages 1 and 2 are unaffected by this choice, but the column is written on every row
    // anyway: a reader grouping by run_id gets one answer for the whole run, and a row that
    // omitted it would be a row you could not attribute.
    std::string h2d = "socket";

    Dist ladder_window(size_t i, uint32_t h) const {
        Dist d;
        for (const auto& w : per_worker) {
            if (i < w.ladder_windows.size()) {
                d.merge(w.ladder_windows[i][h]);
            }
        }
        return d;
    }

    Dist ladder_cumulative(size_t i, uint32_t h) const {
        Dist d;
        for (const auto& w : per_worker) {
            const size_t n = std::min<size_t>(i + 1, w.ladder_windows.size());
            for (size_t k = 0; k < n; ++k) {
                d.merge(w.ladder_windows[k][h]);
            }
        }
        return d;
    }

    uint64_t ladder_bytes_at(size_t i) const {
        uint64_t total = 0;
        for (const auto& w : per_worker) {
            const size_t n = std::min<size_t>(i + 1, w.ladder_window_bytes.size());
            for (size_t k = 0; k < n; ++k) {
                total += w.ladder_window_bytes[k];
            }
        }
        return total;
    }

    void ladder_seal_final() {
        if (!ladder.enabled) {
            return;
        }
        for (auto& w : per_worker) {
            bool any = false;
            for (uint32_t h = 0; h < kHopCount && !any; ++h) {
                any = w.hop[h].n != 0;
            }
            if (!any) {
                continue;
            }
            w.ladder_windows.emplace_back();
            std::copy(std::begin(w.hop), std::end(w.hop), w.ladder_windows.back().begin());
            w.ladder_window_bytes.push_back(w.ladder_bytes - w.ladder_window_start);
            w.ladder_window_start = w.ladder_bytes;
            for (uint32_t h = 0; h < kHopCount; ++h) {
                w.hop[h] = Dist{};
            }
        }
    }

    size_t ladder_points() const {
        size_t n = 0;
        for (const auto& w : per_worker) {
            n = std::max(n, w.ladder_windows.size());
        }
        return n;
    }

    Dist merged(uint32_t h) const {
        Dist d;
        for (const auto& w : per_worker) {
            for (const auto& win : w.ladder_windows) {
                d.merge(win[h]);
            }
            d.merge(w.hop[h]);
        }
        return d;
    }

    uint32_t trace_shift = kTraceShiftDefault;

    std::vector<TraceBucket> merged_trace() const {
        std::vector<TraceBucket> t(kTraceBuckets);
        for (const auto& w : per_worker) {
            for (uint32_t i = 0; i < kTraceBuckets; ++i) {
                t[i].bytes += w.trace[i].bytes;
                t[i].ns_sum += w.trace[i].ns_sum;
                t[i].n += w.trace[i].n;
            }
        }
        return t;
    }
    uint64_t total_trace_clamped() const { return total(&WorkerStats::trace_clamped); }

    uint64_t merged_wire_bytes(uint32_t h) const {
        uint64_t t = 0;
        for (const auto& w : per_worker) {
            t += w.hop_wire_bytes[h];
        }
        return t;
    }
    uint64_t merged_payload_bytes(uint32_t h) const {
        uint64_t t = 0;
        for (const auto& w : per_worker) {
            t += w.hop_payload_bytes[h];
        }
        return t;
    }

    uint64_t merged_hop_window_ns(uint32_t h) const {
        Span s;
        for (const auto& w : per_worker) {
            s.merge(w.hop_window[h]);
        }
        return s.ns();
    }
    uint64_t total(uint64_t WorkerStats::*field) const {
        uint64_t t = 0;
        for (const auto& w : per_worker) {
            t += w.*field;
        }
        return t;
    }
    uint64_t total_found() const { return total(&WorkerStats::found); }
    uint64_t total_bytes() const { return total(&WorkerStats::bytes); }
    uint64_t total_timed_bytes() const { return total(&WorkerStats::timed_bytes); }
    double timed_mb_per_s() const {
        return (timed_ns > 0 && total_timed_bytes() > 0)
                   ? static_cast<double>(total_timed_bytes()) * 1000.0 / static_cast<double>(timed_ns)
                   : 0.0;
    }
    uint64_t total_scanned() const { return total(&WorkerStats::scanned); }
    uint64_t total_stolen() const { return total(&WorkerStats::stolen); }
    uint64_t total_tx_credit_skips() const { return total(&WorkerStats::tx_credit_skips); }
    uint64_t total_delivered() const { return total(&WorkerStats::delivered); }
};

std::string format_table(const RunStats& s);
std::string format_csv(const RunStats& s, const std::string& tag);
std::string csv_header();
std::string basic_csv_header();
std::string format_basic_csv(const RunStats& s, const std::string& tag);

std::string sample_count_warning(const RunStats& s);

std::string format_trace_csv(const RunStats& s, const std::string& tag);

inline uint32_t trace_shift() { return kTraceShiftDefault; }

inline void trace_add(WorkerStats& ws, bool rec, uint64_t timed_start_ns, uint64_t bytes, uint64_t ns) {
    if (!rec || timed_start_ns == 0 || bytes == 0) {
        return;
    }
    const uint64_t now = now_ns();
    if (now <= timed_start_ns) {
        return;
    }
    const uint64_t b = (now - timed_start_ns) >> trace_shift();
    if (b >= kTraceBuckets) {
        ++ws.trace_clamped;
        TraceBucket& t = ws.trace[kTraceBuckets - 1];
        t.bytes += bytes;
        t.ns_sum += ns;
        ++t.n;
        return;
    }
    TraceBucket& t = ws.trace[b];
    t.bytes += bytes;
    t.ns_sum += ns;
    ++t.n;
}

std::string make_run_id();
std::string utc_now_iso();

std::string csv_schema_error(const std::string& path, const std::string& want_header);

std::string rotate_csv(const std::string& path, std::string& error);

std::string ladder_csv_header();
std::string ladder_csv_rows(const RunStats& s, const std::string& tag);

}  // namespace tt::tt_metal::experimental
