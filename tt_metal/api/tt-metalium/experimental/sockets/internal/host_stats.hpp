// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Per-stage instrumentation. 3 stages and 1 aggregate -- there is no return half, and
// the Hop enum below carries only these plus diagnostics:
//
//   1  t6 -> host                  Tensix posts payload + control into its arena
//   2  host -> remote host         MPI RMA moves it to the peer's RX arena
//   3  remote host -> remote t6    the peer writes it into the destination L1
//      ONEWAY_TOTAL                the aggregate
//
// Named after physical events rather than code locations, so a number stays meaningful when
// a function moves.
//
// A host timestamp and a Tensix cycle counter are NOT the same clock; subtracting them raw is
// arithmetic on unrelated quantities, the same error as differencing two unsynchronised hosts'
// CLOCK_MONOTONIC_RAW. So the device clock is calibrated against the host clock the way the
// peer host's is (see host_clock.hpp): the kernel stamps its cycle counter into an operand
// register, a startup calibration fixes the cycles->ns scale and the offset, and every
// cross-domain stage is reported WITH the resulting uncertainty bound.
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
    kHopNotice,  // armed word visible -> a worker picked it up
    kHopDecode,  // validate + operand snapshot + route
    kHopStealWait,
    kHopL1Write,   // the noc_write half of stages 3/6, without the doorbell
    kHopDoorbell,  // the doorbell half
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

struct HopSpec {
    const char* name;
    bool dev_clock;   // crosses device<->host
    bool host_clock;  // crosses host<->host
    bool rate_is_bw;
};

constexpr HopSpec kHopSpec[] = {
    /* T6ToHost             */ {"t6->host", true, false, true},               // producer's copy into its arena
    /* HostToRemoteHost     */ {"host->remote_host", false, false, false},
    /* RemoteHostToRemoteT6 */ {"remote_host->remote_t6", true, false, true},  // MMIO write into destination L1
    /* OneWayTotal          */ {"ONEWAY_TOTAL", false, false, false},
    /* Notice               */ {"diag:notice", false, false, false},
    /* Decode               */ {"diag:decode", false, false, false},
    /* StealWait            */ {"diag:steal-wait", false, false, false},
    /* L1Write              */ {"diag:l1-write", false, false, true},          // the noc_write half of the above
    /* Doorbell             */ {"diag:doorbell", false, false, false},
    /* H2HRetire            */ {"diag:h2h-retire", false, false, true},        // post->completion: the transfer
    /* SendQueueWait        */ {"diag:sendq-wait", false, false, false},
    /* PullWait             */ {"diag:pull-wait", false, false, false},
    /* D2HVisibility        */ {"diag:d2h-visibility", true, false, false},
    /* H2HPayloadAtPeer     */ {"h2h:payload-at-peer", false, false, false},
    /* H2HCreditRaw         */ {"h2h:credit-raw", false, false, false},
    /* H2HNet               */ {"h2h:net", false, true, false},
    /* D2HFenced            */ {"diag:d2h-fenced", true, false, true},         // T6ToHost less the read-back probe
};
static_assert(sizeof(kHopSpec) / sizeof(kHopSpec[0]) == kHopCount, "kHopSpec needs one row per Hop");

inline const char* hop_name(uint32_t h) { return h < kHopCount ? kHopSpec[h].name : "?"; }
inline bool hop_crosses_device_clock(uint32_t h) { return h < kHopCount && kHopSpec[h].dev_clock; }
inline bool hop_crosses_host_clock(uint32_t h) { return h < kHopCount && kHopSpec[h].host_clock; }
inline bool hop_rate_is_bandwidth(uint32_t h) { return h < kHopCount && kHopSpec[h].rate_is_bw; }

// n is a count; sum is a total of nanoseconds
struct Dist {
    uint64_t n = 0;
    uint64_t sum = 0;  // nanoseconds, totalled over n samples

    double mean_ns() const { return n ? static_cast<double>(sum) / static_cast<double>(n) : 0.0; }

    void add(uint64_t v) {
        ++n;
        sum += v;
    }

    // Exact and order-independent: a count and a total both add.
    void merge(const Dist& o) {
        n += o.n;
        sum += o.sum;
    }
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

struct alignas(64) WorkerStats {
    Dist hop[kHopCount];
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
    uint64_t delivered = 0;  // messages written into a Tensix L1
    // Armed TX work passed over because the peer has not returned enough credit and
    // send_blocking_ is off. Incremented in D2H2H2DSocket::send_try_start(), surfaced by
    // total_tx_credit_skips() in format_table's credit warning.
    uint64_t tx_credit_skips = 0;

    // False when pin_this_thread() failed: the work is still correct, but the scheduler could
    // migrate the thread mid-run, so its hop timings are not comparable with the pinned
    // workers'. Written at thread start, read at collect(); both sit before pad[64] so the
    // false-sharing padding still ends the struct.
    bool pinned = true;
    std::string pin_error;  // strerror text from the failed pthread_setaffinity_np

    char pad[64];
};

struct RunStats {
    std::vector<WorkerStats> per_worker;
    uint64_t clock_overhead_ns = 0;
    uint64_t wall_ns = 0;

    uint64_t timed_ns = 0;
    uint32_t timed_iters = 0;     // iterations inside the bracket: iters - warmup
    // dead-code review -- `xfers_per_iter` removed 2026-09-14: never assigned, so always 1.

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

    std::string run_id;  // unique per process
    std::string run_started_utc;

    double ns_per_cycle = 0.0;

    uint32_t warmup = 0;
    bool warmup_applied = false;

    uint32_t host_ident = 0;
    std::string role = "local";  // server | peer | local
    bool symmetric = false;
    bool tx_side = true;

    // 1 x H2DSocket per core in DEVICE_PULL mode -- because nothing assigns the flag that would
    // select "write" (host CPU storing into L1 over a WC TLB window), and that path is not
    // offered. Written on every row because it lives nowhere else: `mode` is the run role and
    // `tag` is caller text, so the only prior record was a filename suffix from the sweep
    // script, which renaming or merging files lost silently.
    std::string h2d = "socket";

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

    Dist merged(uint32_t h) const {
        Dist d;
        for (const auto& w : per_worker) {
            d.merge(w.hop[h]);
        }
        return d;
    }


    uint64_t total(uint64_t WorkerStats::* field) const {
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
    uint64_t total_stolen() const { return total(&WorkerStats::stolen); }
    uint64_t total_tx_credit_skips() const { return total(&WorkerStats::tx_credit_skips); }
    uint64_t total_delivered() const { return total(&WorkerStats::delivered); }

    // Not total(): that folds uint64_t event counters, and this counts threads.
    uint32_t workers_unpinned() const {
        uint32_t n = 0;
        for (const auto& w : per_worker) {
            n += w.pinned ? 0u : 1u;
        }
        return n;
    }
};

std::string format_table(const RunStats& s);
std::string basic_csv_header();
std::string format_basic_csv(const RunStats& s, const std::string& tag);

std::string sample_count_warning(const RunStats& s);

// Empty unless a worker failed to pin. An unpinned worker in a pinned pool is the one row
// that cannot be compared with the others, so the run says so up front rather than leaving
// it to be found later as an unexplained outlier.
std::string pinning_warning(const RunStats& s);


std::string make_run_id();
std::string utc_now_iso();

std::string csv_schema_error(const std::string& path, const std::string& want_header);



}  // namespace tt::tt_metal::experimental
