// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// D2H2H2DSocket -- a device-to-device channel whose middle is two hosts.
//
//   chip A  --D2H-->  host A  --H2H-->  host B  --H2D-->  chip B
//           <--H2D--                            <--D2H--
//
// A Tensix posts its payload into pinned host RAM and arms its control word; this host
// sweeps the banks from a work-stealing pool; a host-to-host transport carries the middle
// hop; the far host writes the last one into L1.
//
// ---------------------------------------------------------------------------
//
// HostRegion (the pinned register file and per-core arenas), Deliverer (the H2D leg),
// Transport (the H2H leg), BankScanner (the work-stealing sweep), PeerTable, and the stats,
// clock, UVA and wire-layout headers. Nothing in this file reaches outside
// tt::tt_metal::experimental.
//
// ---------------------------------------------------------------------------
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tt_metal/distributed/host_clock.hpp"
#include "tt_metal/distributed/host_deliver.hpp"
#include "tt_metal/distributed/host_region.hpp"
#include "tt_metal/distributed/host_scan.hpp"
#include "tt_metal/distributed/host_stats.hpp"
#include "tt_metal/distributed/host_transport.hpp"
#include "tt_metal/distributed/host_uva.hpp"
#include "tt_metal/distributed/host_uva_layout.hpp"

namespace tt::tt_metal::experimental {

// Operand layout
enum : uint32_t {
    kArgDestUva = 0,     // TX: where these bytes are going
    kArgLength = 1,      // TX and RX: how many bytes
    kArgElapsed = 2,     // running sum of every stage measured so far
    // Slot 3 is overloaded. On the TX/immediate form the kernel
    // writes its own core index here (host_scan.cpp reads region reg 3); on an RX store
    // notice the same slot carries the DESTINATION UVA (host_scan.cpp reads it from
    // kNoticeUvaOffset, service_rx reads it back as dest_uva).
    kArgOriginCore = 3,
    kArgCount = 4
};

struct SocketConfig {
    // volume ladder, or null. Owned by the caller and must outlive the socket -- it is
    // handed straight to ScanConfig, which hands it to every worker.
    const VolumeLadder* ladder = nullptr;
    LadderSync* ladder_sync = nullptr;  // non-null only when quiescing

    // Payload. Needed only to size the receive-slot sweep: the number of
    // slots a message of this size leaves in an arena is what the scanner should look at, not
    // the maximum.
    // commented out b/c deadcode
    // uint32_t payload_bytes = 0;
    uint32_t chip = 0;     // our own chip index, for the origin selector a notice carries
    uint32_t cores = 0;    // cores in use; bounds the stall dump and the local check
    uint32_t workers = 0;  // 0 => one per CPU, capped at cores
    bool pin = true;
    // commented out b/c deadcode
    // bool roundtrip = false;

    // `send_window` 0 means unset -> cores-in-use; `send_blocking` selects
    // post-and-wait and implies a window of 1. The window caps concurrency; the shape decides
    // whether the thread parks or spins. A window of 1 is NOT the blocking sender -- it gives
    // the blocking sender's concurrency while still spinning in try_wait() and still skipping
    // on an unreturned credit. Keeping them separate is what lets one measurement tell the two
    // causes apart.
    uint32_t send_window = 0;
    bool send_blocking = false;
    bool record_from_start = true;  // false while a warmup is pending

    // 0 means "the caller owns the gate" and set_recording() is the only thing that will ever open it.
    //
    // Needed because set_recording() is driven from a PRODUCER loop, which knows the iteration
    // number -- and a side that only receives has no producer loop. Where the roles are split
    // that is half the run: the receiving side set recording_ false at construction and
    // nothing ever flipped it, so every stage recorded zero samples and the table printed a
    // dash on every row. Two of its rows survived only because they bypassed the gate
    // entirely, which is what hid it.
    uint64_t warmup_msgs = 0;

    // Credit RTT (h2h:credit-raw and h2h:net). Off by default, for the same reason
    // TransportConfig::measure_retire is: it adds work to the SENDER THREAD, which is this
    // path's measured bottleneck (a bandwidth run must not pay for a number it does not print)
    //
    // The cost is one credit_total() read per IN-FLIGHT message per lap -- the pass skips
    // unarmed watches on a bool. At --send-window 1, where h2h:net is the only place it means
    // anything, that is one cache line per lap.
    //
    // hop_window_ns is an elapsed span (max close - min open), not a sum of accounted work; there
    // is no line item to remove, and removing a per-thread duration from a wall-clock envelope is
    // the S/T-into-S/W confusion this file was rebuilt to eliminate.
    bool measure_credit = false;

    double ns_per_cycle = 0.0;  // 0 => a cycles-flagged accumulator contributes no sample
};

// Everything the path counts. Atomics because scan workers, the sender thread and the caller's
// shutdown wait all read and write them concurrently.
//
//   tx_done    outbound TX jobs finished servicing -- what a producer waits on before
//              re-arming. NOT the scanner's job total: one message produces two serviced jobs
//              (the TX and the RX delivery), so a producer watching the total sees its target
//              met at double rate and never waits (measured 134/160, 149/160).
//   delivered  bytes written into a Tensix L1.
//   (home_done / replies REMOVED -- the round-trip barrier they served went with libfabric.
//
struct SocketCounters {
    std::atomic<uint64_t> routed_local{0};
    std::atomic<uint64_t> routed_remote{0};
    std::atomic<uint64_t> routed_nowhere{0};
    std::atomic<uint64_t> delivered{0};
    std::atomic<uint64_t> tx_done{0};
    std::atomic<uint64_t> errors{0};
    // flush_slots/flush_calls is how many payloads one MPI_Win_flush retired. The flush is a full
    // round trip, so the whole point of the kPayloadLocal phase is to divide that cost across a
    // lap's worth of messages -- but nothing checked that it actually does. A ratio near 1 means
    // every message is paying its own round trip and the batching is not happening.
    std::atomic<uint64_t> flush_calls{0};
    std::atomic<uint64_t> flush_slots{0};
    //   credit_net_no_resp   the credit carried a zero turnaround -- an old peer, or a credit
    //                        posted from a path that has no measurement to report.
    //   credit_net_skew      resp >= raw: the peer claims to have spent longer on the delivery
    //                        than we spent waiting for the whole round trip. Physically
    //                        impossible, so it is evidence of clock skew and the sample is
    //                        DROPPED rather than published as a negative or clamped to zero.
    std::atomic<uint64_t> credit_net_no_resp{0};
    std::atomic<uint64_t> credit_net_skew{0};
    std::atomic<uint64_t> rejects[8];

    SocketCounters() {
        for (auto& r : rejects) {
            r.store(0, std::memory_order_relaxed);
        }
    }
    SocketCounters(const SocketCounters&) = delete;
    SocketCounters& operator=(const SocketCounters&) = delete;
};

#if defined(TT_METAL_HOST_BRIDGE)

// ---------------------------------------------------------------------------
// Operates with symmetry
//
//   open()        refuses unless the topology really is two hosts.
//   service_tx()  refuses a LOCAL destination. A UVA resolving to this host names a core that
//                 is already a source, so delivering into it would put a second writer on the
//                 buffer that core is sending from -- overwriting a payload it had not finished
//                 sending, with nothing anywhere reporting it.
//
// ---------------------------------------------------------------------------
// Sending must not happen on a scan thread. The send path waits on completions, and a worker
// inside it is a worker not scanning -- so it cannot DELIVER the peer's inbound traffic, so
// the peer never returns the credit it is waiting for.
//
// ONE sender thread, not one per worker: the transport is ONE endpoint with ONE request table,
// already serialised inside itself, so a second thread would add contention rather than
// throughput. Scaling that (an endpoint per worker) is a different measurement.
//
// It also keeps the RMA calls single-threaded in practice. MPI_THREAD_MULTIPLE is required and
// provided, but concurrent one-sided operations on a single window are the least-exercised
// corner of most runtimes -- so the shape that was chosen for throughput happens to be the one
// that stays on the well-trodden path, and it should not be traded away casually.
// ---------------------------------------------------------------------------
class D2H2H2DSocket {
public:
    // THE FAULT DOMAIN FOR STORES. A store's offset arrives from the wire, so an executor that
    // trusts it is an arbitrary-write primitive. Set by the driver from the same L1 map the
    // kernels were compiled against.
    struct StoreGuard {
        uint32_t lo = 0;
        uint32_t hi = 0;
        uint32_t signal_addr = 0;
        uint32_t completion_addr = 0;
        uint32_t stop_addr = 0;
    };
    void set_store_guard(const StoreGuard& g) { store_guard_ = g; }
    uint64_t store_faults() const { return store_faults_.load(std::memory_order_relaxed); }

    D2H2H2DSocket(HostRegion& region, Deliverer* deliverer, HostTopology topo,
                  ClockSync clock, SocketConfig cfg, Transport& transport);
    ~D2H2H2DSocket();

    D2H2H2DSocket(const D2H2H2DSocket&) = delete;
    D2H2H2DSocket& operator=(const D2H2H2DSocket&) = delete;

    bool open(std::string& err);

    void stop();

    const SocketCounters& counters() const { return counters_; }
    std::string first_error() const;

    bool transport_failed() const { return transport_failed_.load(std::memory_order_acquire); }

    void set_recording(bool on) {
        recording_.store(on, std::memory_order_relaxed);
        transport_.set_recording(on);
    }

    Transport& transport() const { return transport_; }

    // Registered before open()
    void add_peer(Transport* t) { extra_peers_.push_back(t); }

    std::vector<Transport*> peers_for_barrier() const { return peers_.all(); }

    void stamp_timed_start() {
        uint64_t expect = 0;
        timed_start_ns_.compare_exchange_strong(expect, now_ns(), std::memory_order_relaxed);
    }
    void stamp_timed_end() {
        uint64_t expect = 0;
        timed_end_ns_.compare_exchange_strong(expect, now_ns(), std::memory_order_relaxed);
    }
    uint64_t timed_start_ns() const { return timed_start_ns_.load(std::memory_order_relaxed); }

    void open_recording_gate() {
        set_recording(true);
        stamp_timed_start();
    }

    RunStats collect() const;

    std::string stall_dump(const char* where) const;

    HostRegion& region() const { return region_; }

private:

    uint64_t service_one(const Job& job, WorkerStats& ws);
    uint64_t service_rx(const Job& job, WorkerStats& ws, bool rec);
    uint64_t service_tx(const Job& job, WorkerStats& ws, bool rec);

    uint64_t deliver_to_l1(const Job& job, WorkerStats& ws, uint32_t stage, uint64_t& stage_ns,
                           bool rec);

    uint64_t elapsed_ns_of(const Job& job, bool& usable, uint64_t& visibility_ns) const;

    uint64_t deliver_remote(const Job& job, WorkerStats& ws, uint32_t dest_core, uint64_t length,
                            uint64_t accumulated_ns);

    bool start_transport(std::string& err);
    void stop_transport();
    void append_transport_stats(RunStats& s) const;
    void dump_transport(std::string& into) const;

    void return_credit(uint32_t origin_selector, uint64_t turnaround_ns);

    // ---- shared helpers --------------------------------------------------
    void fail(const std::string& what);
    void retire_tx(uint32_t core);
    static void add_sample(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns);
    static void add_sample_with_size(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns, uint64_t amt);
    static void add_sample_with_payload(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns, uint64_t payload);
    static void add_sample_with_window(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns,
                                       uint64_t payload, uint64_t open_ns, uint64_t close_ns);
    bool recording() const { return recording_.load(std::memory_order_relaxed); }

    bool recording_now();

    static uint32_t my_stage_slot();

    struct SendReq {
        uint32_t src_core = 0;
        uint32_t dest_core = 0;
        uint64_t length = 0;
        uint64_t accumulated_ns = 0;
        uint64_t t_queued = 0;
        // The effective address, forwarded unmodified. Zero means "not a store".
        uint64_t dest_uva = 0;
        // Which peer this goes to, decoded once on the scan worker. dest_uva is 0 for a
        // non-store message, so the sender could not recover it.
        uint32_t dest_host = 0;
    };

    // A message in flight, per core.
    struct SendSlot {
        enum Phase : uint8_t { kIdle = 0, kAwaitPayload = 1, kAwaitNotice = 2, kPayloadLocal = 3 };
        uint8_t phase = kIdle;
        SendReq r{};
        OpHandle payload_op{};
        OpHandle notice_op{};
        uint64_t t0 = 0;
        uint64_t deadline = 0;
        Transport* tp = nullptr;
        // Which of the destination core's receive slots this sender owns. DERIVED from the
        // source, not claimed -- so a slot has one lifetime source and needs no ticket, no
        // tail pointer and no lap check.
        // uint32_t rx_slot = 0;
    };

    struct CreditWatch {
        bool armed = false;
        bool timed = false;      // the message passed the straddler test when it was posted
        uint64_t want = 0;       // credit_total(core) must reach this
        uint64_t t0 = 0;         // when the payload was posted -- the same t0 the h2h rows use
        uint64_t length = 0;
        uint32_t dest_host = 0;  // whose turnaround word to read back
    };
    std::vector<CreditWatch> credit_watch_;

    void sender_loop();
    bool send_try_start(SendSlot& slot, uint32_t core, WorkerStats& ws, bool rec);
    bool send_poll(SendSlot& slot, WorkerStats& ws, bool rec);

    // Arms the RX notice for a slot the flush pass has just made remotely visible. Split out of
    // send_poll() because the flush between the two is per-ENDPOINT and per-lap, while this is
    // per-slot: one flush releases every slot waiting on the same peer.
    bool send_arm_notice(SendSlot& slot);

    // Retires a slot whose message will not complete. Extracted so the flush pass can fail a
    // whole endpoint's worth of slots the same way send_poll() fails one -- a message that is
    // dropped without retiring leaves its producer blocked on tx_done to the drain deadline,
    // reporting a timeout instead of the error that actually happened.
    void send_fail_slot(SendSlot& slot);

    // ---- construction state ----------------------------------------------
    HostRegion& region_;
    Deliverer* deliverer_ = nullptr;
    HostTopology topo_{};
    ClockSync clock_{};
    SocketConfig cfg_{};
    SocketCounters counters_;
    Transport& transport_;

    std::vector<std::vector<std::atomic<uint64_t>>> credit_out_;  // remote deliveries echoed back
    std::vector<std::atomic<uint64_t>> delivered_per_core_;       // the value rdma_signal carries
    // Single delivery at a time per core.
    std::vector<std::mutex> deliver_m_;
    std::vector<std::atomic<uint64_t>> notice_sent_;  // notices armed in the peer's bank
    std::vector<std::atomic<uint64_t>> tx_retired_;   // the value rdma_completion carries

    // 0 idle, 1 credit-wait, 2 payload, 3 notice. Written by the sender thread, read by the
    // stall dump. A sender parked in a state it cannot report reads as `idle`, which is worse
    // than no dump at all.
    std::atomic<int> sender_state_{0};

    StoreGuard store_guard_{};
    std::atomic<uint64_t> store_faults_{0};

    std::atomic<bool> recording_{true};
    std::atomic<uint64_t> timed_start_ns_{0};
    std::atomic<uint64_t> timed_end_ns_{0};
    std::atomic<bool> stopped_{false};
    std::unique_ptr<BankScanner> scanner_;
    mutable std::mutex err_mutex_;
    std::string first_error_;

    std::atomic<bool> transport_failed_{false};

    // Each core has a single queue. A destination core has one RX control word, so at most one message
    // may be outstanding to it; (per-core queues) lets the sender hold a window across cores. It also
    // removes the head-of-line stall where one core's unreturned credit parked every other core's sends.
    std::mutex send_m_;
    std::condition_variable send_cv_;
    std::vector<std::deque<SendReq>> send_q_;  // indexed by src_core
    std::vector<std::atomic<uint32_t>> send_pending_;
    std::atomic<uint64_t> send_depth_{0};
    uint32_t send_rr_ = 0;
    bool send_stop_ = false;

    PeerTable peers_;
    std::vector<Transport*> extra_peers_;  // mesh peers beyond transport_; see add_peer()

    uint32_t send_window_ = 0;
    bool send_blocking_ = false;

    // Credit flush deadline. Owned by the sender thread alone (last_credit_flush_) except for
    // the counter, which the stall dump reads from another thread. See the credit flush in
    // sender_loop(): flushing only on an idle lap starves at high core counts, so a deadline
    // bounds credit latency by wall time instead of by whether this host happens to be idle.
    static constexpr std::chrono::microseconds kCreditFlushInterval{100};
    std::chrono::steady_clock::time_point last_credit_flush_{};
    std::atomic<uint64_t> credit_flushes_{0};
    // rx-side credit gate instrumentation. deliver_remote() only returns a credit
    // when the arriving ctrl word carries kFlagRemoteNotice.
    std::atomic<uint64_t> rx_deliveries_{0};        // times deliver_remote() reached the gate
    std::atomic<uint64_t> rx_remote_notice_{0};     // ... of those, how many had the flag
    std::atomic<uint64_t> rx_first_ctrl_{0};        // the first ctrl word consumed, verbatim
    std::atomic<uint64_t> rx_credits_posted_{0};    // times return_credit() was ENTERED
    std::atomic<uint64_t> rx_credits_done_{0};      // ... and times it RETURNED
    std::atomic<uint64_t> rx_origin_host_{~0ull};   // origin_host the last credit resolved to
    std::atomic<uint64_t> rx_origin_sel_{0};        // the selector it came from

    std::thread sender_;
    WorkerStats sender_stats_{};
};

#endif  // TT_METAL_HOST_BRIDGE

}  // namespace tt::tt_metal::experimental
