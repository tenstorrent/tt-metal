// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf-debug host receiver: D2H socket streams -> in-place frame decode -> 24 B raw records
// emitted DIRECTLY into per-stream BroadcastRings -> per-consumer pairing -> registered consumers.
//
// Decode threads fuse poll + in-place decode + ack: frames are decoded out of the pinned
// FIFO itself (no staging copy), records are stored straight into ring slots (no batch
// copy), and the ack is issued only after the peeked frames are decoded -- credit return
// is decode-paced and the receiver is lossless by construction. One ring per socket
// stream keeps each ring single-writer. Each registered consumer gets its own thread with
// one reader per ring; a lagging consumer drops its own oldest records (counted) -- the
// only place on the host where records can drop.
//
// TWO record types, deliberately: the RING carries the raw 24 B record the decode hot path emits
// (the AVX2 packer and the all-NT-store discipline depend on this layout, do not widen it), while
// consumers receive the PUBLIC 32 B PerfDebugRec. Worker zones arrive from the device WHOLE
// (ZoneAtomic: end + duration) and convert 1:1 -- the PRODUCER-STALL zone included, since it ships
// atomically too. Only the >3.2s long-zone fallback still emits legacy start/end pairs, which go
// through the per-(dev, lane) pairing stack on each consumer's delivery thread (no shared state, no
// locks, never on decode).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tools/profiler/perf_debug_consumer.hpp"
#include "tools/profiler/spsc_marker_decode.hpp"

namespace tt::tt_metal {

namespace distributed {
class D2HSocket;
}
template <typename T>
class BroadcastRing;

namespace perf_debug {

// ---- RAW ring record: receiver-internal, NOT the consumer contract --------------------------------
//
// This is the record the decode hot path emits and the BroadcastRing carries: the device's own
// start/end markers, unpaired, 24 B. Its field and bit layout (lane at 16, dev at 26, type at 29;
// ZoneStart=1 / ZoneEnd=2) are pinned by the receiver's vectorized packer and by the all-NT-store
// discipline in broadcast_ring.hpp -- widening or reordering it is a measured multi-x decode
// regression, which is why the public PerfDebugRec is a SEPARATE type built after the ring.
// Receiver-internal only: every consumer, the built-in Tracy sink included, receives the public
// paired record.
enum class PerfDebugRawRecType : uint32_t {
    ZoneStart = 1,  // legacy pair halves: >3.2s fallback only
    ZoneEnd = 2,
    // 3 recycled: it was ZoneTotal (SUM zones, removed). Safe to reuse HERE because this enum is
    // in-process only -- the never-reuse rule binds wire values (PP_*), which a stale JIT ELF can emit.
    ZoneAtomic = 3,  // one complete zone: ts = END, dur = duration (start = ts - dur)
    Data = 4,
    Event = 5,
    Ext = 6,
    Cont = 7,
};

struct PerfDebugRawRecMeta {
    uint32_t spare : 16;
    uint32_t lane : 10;
    uint32_t dev : 3;
    PerfDebugRawRecType type : 3;
};
static_assert(sizeof(PerfDebugRawRecMeta) == 4);

struct PerfDebugRawRec {
    uint64_t ts;  // ZoneAtomic: zone END; everything else: the marker's timestamp
    uint32_t id;  // full 27-bit structural zone id
    PerfDebugRawRecMeta meta;
    uint32_t prog;
    // ZoneAtomic duration in cycles; 0 on every other type. Occupies what used to be tail padding, so
    // field offsets are unchanged -- the AVX2 legacy-pair packer's third quadword (zero-extended prog)
    // already writes these bytes as 0.
    uint32_t dur;
};
static_assert(sizeof(PerfDebugRawRec) == 24);
static_assert(std::is_trivially_copyable_v<PerfDebugRawRec>);

struct ReceiverDeviceConfig {
    uint32_t chip_id = 0;
    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets;
    uint32_t num_cores = 0;
    std::vector<PerfDebugLaneInfo> lane_table;          // size num_cores * 5
    std::unordered_map<uint32_t, uint32_t> core_of_xy;  // incl. DRISC self-zone cores
    bool clock_synced = false;
    double frequency_ghz = 0.0;
};

// One device-to-device clock-sync sample (a PP_SYNC packet), routed OUT of the record stream at the
// decode stage: decode is the single point every packet passes exactly once, so the sync aggregator
// sees each sample once, while per-consumer delivery threads never see them at all. dev/lane are
// capture-context indices -- the aggregator maps them to (chip, eth core) via capture_context().
struct PerfDebugSyncSample {
    uint32_t dev = 0;
    uint32_t lane = 0;
    uint32_t which = 0;  // 0 = T0 (initiator send), 1 = T1 (responder stamp), 2 = T2 (initiator echo)
    uint32_t round = 0;  // 17-bit round sequence (wraps); join key with idx
    uint32_t idx = 0;    // sample index within the round
    uint64_t ts = 0;     // full device timestamp (59-bit, timer_hi from the lane's sticky)
};
using PerfDebugSyncSink = std::function<void(const PerfDebugSyncSample&)>;

struct ReceiverConfig {
    // Optional: called once per stream if it starves for the watchdog window mid-run while
    // its producers are not done (control plane can dump drainer state; receiver has no MMIO).
    std::function<void(uint32_t device_index, uint32_t socket_index)> starvation_diagnostic;
};

class PerfDebugReceiver {
public:
    PerfDebugReceiver(ReceiverConfig config, std::vector<ReceiverDeviceConfig> devices);

    // Late-arriving frequency for a device whose anchor is COMPOSED from the root AFTER the receiver
    // was built. ReceiverDeviceConfig::frequency_ghz is snapshotted at construction, and for a
    // non-root device it is still 0 at that point -- the host-fit fallback that used to fill it in
    // was deliberately removed (it injected us-scale cross-device skew). Without this refresh the
    // device keeps frequency 0 forever: its per-device stats are skipped by the `freq > 0.0` gate
    // and nothing downstream can turn its cycles into time.
    void set_device_frequency(uint32_t chip_id, double freq_ghz);
    ~PerfDebugReceiver();

    PerfDebugReceiver(const PerfDebugReceiver&) = delete;
    PerfDebugReceiver& operator=(const PerfDebugReceiver&) = delete;

    void start();

    // Sync-sample sink: invoked on the DECODE thread for every PP_SYNC packet, so it must be cheap
    // and must never block (enqueue-and-return). Set BEFORE start(); unset means sync packets are
    // counted and dropped.
    void set_sync_sink(PerfDebugSyncSink sink) { sync_sink_ = std::move(sink); }

    PerfDebugConsumerHandle add_consumer(std::string name, PerfDebugRecordCallback cb);
    void remove_consumer(PerfDebugConsumerHandle handle);

    // Every drainer owning (device, socket) has published done, which implies the device saw
    // all its bytes acked -- so the stream retires itself after one final empty check.
    void notify_producers_done(uint32_t device_index, uint32_t socket_index);

    void shutdown();

    const PerfDebugCaptureContext& capture_context() const { return ctx_; }
    // Final per-lane words-consumed mirrors; valid after shutdown(). Feeds the control
    // plane's completeness check against the workers' own tails.
    std::vector<uint32_t> final_lane_heads(uint32_t device_index) const;
    void log_report() const;

private:
    // The PRODUCER-STALL zone ids, mirrored from the zone-meta registry BY NAME: the stall counter is
    // the one thing the decode identifies per marker, and it may not do that by id VALUE -- a structural
    // id legitimately moves whenever kernel_profiler.hpp does, and the stall zone genuinely has MANY ids,
    // one per kernel TU, because that header declares it at namespace scope. Refreshed by cursor once per
    // decode pass.
    //
    // The membership test runs PER MARKER on the decode hot path, so it is an open-addressed table (one
    // L1 load on the overwhelming miss path), not a binary search -- and not a [min,max] pre-screen
    // either: stall ids carry tu_id in their high bits like every other zone id, so their range spans
    // the id space and a range screen admits everything (measured ~2x on the decode walk).
    struct StallIdMirror {
        uint32_t cursor = 0;
        std::vector<uint32_t> ids;    // insertion order, source for rebuilds
        std::vector<uint32_t> table;  // open addressing, linear probe; 0xFFFFFFFF = empty
        uint32_t mask = 0;
        void refresh();
        bool contains(uint32_t id) const {
            if (table.empty()) {
                return false;
            }
            uint32_t slot = (id * 0x9E3779B9u) & mask;
            while (true) {
                const uint32_t v = table[slot];
                if (v == id) {
                    return true;
                }
                if (v == 0xFFFFFFFFu) {
                    return false;
                }
                slot = (slot + 1) & mask;
            }
        }
    };

    struct Stream {
        distributed::D2HSocket* sock = nullptr;
        uint32_t dev = 0;
        uint32_t sock_idx = 0;
        std::unique_ptr<BroadcastRing<PerfDebugRawRec>> ring;
        profiler::SpanDecodeState decode;
        StallIdMirror stall_ids;
        std::vector<uint64_t> last_zone_ts;  // per lane, order invariant (must never regress)
        std::atomic<bool> producers_done{false};
        bool retired = false;

        uint64_t passes = 0, frames = 0, pages = 0, records = 0, zone_markers = 0, stall_zones = 0;
        uint64_t sync_packets = 0;  // PP_SYNC packets routed to the sync sink (never delivered)
        uint64_t decode_ticks = 0;
        uint64_t order_regressions = 0, bad_frames = 0;
        uint64_t first_data_tsc = 0, last_commit_tsc = 0;
        uint64_t min_zone_ts = 0, max_zone_ts = 0;
        uint64_t checksum = 0;  // READ_ONLY ablation: defeats elision of the bandwidth-probe reads
        bool desync_warned = false;
        bool watchdog_fired = false;
        std::chrono::steady_clock::time_point last_progress;
    };

    struct Consumer {
        std::string name;
        PerfDebugRecordCallback cb;
        PerfDebugConsumerHandle handle = 0;
        std::atomic<int> mode{0};  // 0 = run, 1 = drain-then-stop, 2 = stop-now
        uint64_t delivered = 0;
        uint64_t dropped = 0;
        std::thread thread;
    };

    void decode_thread(std::vector<Stream*> streams);
    // One poll+decode+ack pass over a stream. Returns true if it moved data; sets
    // s.retired when the stream is finished.
    bool decode_pass(Stream& s);
    void consumer_thread(Consumer& c);

    ReceiverConfig cfg_;
    std::vector<ReceiverDeviceConfig> devices_;
    PerfDebugCaptureContext ctx_;
    std::vector<std::unique_ptr<Stream>> streams_;
    std::vector<std::thread> decode_threads_;

    std::mutex consumers_mu_;
    std::vector<std::unique_ptr<Consumer>> consumers_;
    std::vector<std::unique_ptr<Consumer>> consumers_report_;
    uint64_t next_handle_ = 1;

    bool no_decode_ = false;
    bool read_only_ = false;  // peek + line-stride read + pop: isolates pinned-FIFO read bandwidth
    bool stall_only_ = false;
    PerfDebugSyncSink sync_sink_;  // set before start(); decode threads call it, never mutate it
    uint32_t die_after_ = 0;
    std::chrono::seconds watchdog_{120};

    std::atomic<bool> stop_{false};
    std::atomic<bool> shutdown_done_{false};
    bool started_ = false;
};

}  // namespace perf_debug
}  // namespace tt::tt_metal
