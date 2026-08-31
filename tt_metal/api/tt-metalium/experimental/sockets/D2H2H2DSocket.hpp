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
// THE NAME SPELLS EVERY HOP ON PURPOSE. Each host holds a pair of endpoints against its own
// device -- one it reads and one it writes -- and the two pairs are joined by a host-to-host
// link: D2H, H2D, H2H, H2D, D2H. It is not shortened to D2D because D2D already means
// device-to-device over Tenstorrent's own ethernet fabric (MeshSocket, in these same
// headers), and the defining property of this path is the opposite -- the host is IN it,
// twice, because the destination is not in any chip's NOC address space.
//
// ---------------------------------------------------------------------------
// WHERE THIS SITS, AND WHAT IT STILL DEPENDS ON
//
// Declared in tt::tt_metal::experimental for review. Note that the sibling socket classes --
// D2HSocket, H2DSocket, MeshSocket -- are declared in tt::tt_metal::distributed even though
// their headers live under experimental/sockets/, so if the intent is for this to sit beside
// them rather than in the experimental namespace proper, this is the line to change.
//
// Everything it is built on now lives in this same namespace, in this same directory:
// HostRegion (the pinned register file and per-core arenas), Deliverer (the H2D leg),
// Transport (the H2H leg), BankScanner (the work-stealing sweep), PeerTable, and the stats,
// clock, UVA and wire-layout headers. Nothing in this file reaches outside
// tt::tt_metal::experimental.
//
// THOSE TYPES ARE COPIES, NOT MOVES. tests/t6_host_uva/ still holds the originals in namespace
// t6_host_uva, because two shipping programs and -- critically -- the bare-metal RISC-V kernels
// still compile against them. So there are now TWO definitions of the wire contract, and
// nothing in the tree detects a copy that was missed. See host_uva_layout.hpp here for which
// half the kernels actually read.
//
// WHAT IT DOES NOT OWN, and should not acquire: the producer (a kernel, or a test's stand-in
// thread), the PASS/FAIL verdict, and the CSV. Those belong to a driver, not to a path.
//
// ---------------------------------------------------------------------------
// ONE CLASS, NO BASE, AND THE TRANSPORT IS REQUIRED
//
// In tests/t6_host_uva/host_socket.hpp this type derives from D2H2DSocket, which does three
// jobs at once: it holds the shared engine, it IS the one-host configuration, and it is the
// type the driver holds polymorphically. Here there is no base. `Transport&` is a constructor
// argument rather than an optional member, so a socket named for a five-hop path still cannot
// be constructed without the hop in the middle -- the guarantee is carried by the absence of
// a nullable pointer, not by a check. What it gives up is the ability to serve the
// no-transport case; that case is simply not this class's job.
//
// Six things follow, and each REMOVES something that previously had to be explained:
//
//   1. NO VIRTUALS AT ALL. The six former seams (deliver_remote, start_transport,
//      stop_transport, append_transport_stats, dump_transport, return_credit) are ordinary
//      private methods. Their NAMES are unchanged on purpose, so `grep -c` still lines this
//      file up against host_socket.cpp and t6_host_uva.cpp.
//   2. THE open() VTABLE HAZARD IS GONE. host_socket.hpp had to explain that open() cannot
//      live in the constructor because the pool's callback dispatches through the vtable, and
//      a virtual call from a base constructor resolves to the base. With no base there is no
//      such hazard. open() stays separate for the honest reason: it can FAIL, and it reports
//      why through a string a constructor could only throw.
//   3. deliver_remote() HAS ONE BODY. The base's refusal went with the base; the routing
//      switch calls the real thing directly.
//   4. set_recording() IS ONE METHOD that both flips the gate and forwards it to the
//      transport, instead of a base method plus an override that calls it.
//   5. TWO DEAD DECLARATIONS DROPPED. `send_remote()` was declared at host_socket.hpp:436 and
//      never defined anywhere -- a leftover of the pre-2026-08-28 blocking sender. `tx_mutex_`
//      was declared and never used; its serialisation moved inside the transport when the
//      three-threads-one-endpoint bug was fixed. The reasons they existed are recorded here
//      instead, which is what keeping the unused member was for.
//   6. ONE BEHAVIOUR FIX. See set_recording() below -- it is the only one in this file.
//
// ---------------------------------------------------------------------------
// WHY THERE IS NO service()
//
// A caller-driven non-blocking pass over both directions is the right shape when a single
// embedded core walks every bank and the host is a one-threaded servant of it. It cannot
// carry over here: the bank sweep is a work-stealing pool of pinned threads, and the
// bank-sweep term in `interval = S + F/D` is absent precisely BECAUSE the sweep is parallel.
// A caller-driven service() would collapse the pool back to one thread and throw away the
// reason this path exists.
//
// So this class owns the pool and exposes a lifecycle -- open/stop -- with the service
// function as an internal callback. What survives from that other shape is the part about
// correctness rather than control flow: both directions are serviced by ONE function on the
// SAME threads, because any arrangement that finishes one direction before starting the other
// stops at the first iteration with this host waiting for records from cores that are waiting
// for this host.
//
// ---------------------------------------------------------------------------
// BUILDING
//
// The includes are bare names and every one of them is a file in THIS directory, so
// `finalized/` alone on the include path is enough -- verified by compiling every translation
// unit here with nothing else on it. Do not add tests/t6_host_uva to the include path as well:
// the headers there have the same filenames, and a quoted include that resolved to the wrong
// copy would put two namespaces' worth of the same wire contract into one binary.
//
// REQUIRES A HOST-TO-HOST TRANSPORT. The class is a middle hop and there is no version of it
// without one, so the declaration sits under HAS_LIBFABRIC and the translation
// unit is empty without it. Do not add a second libfabric probe -- a second thing that can
// disagree about whether libfabric exists is how the two halves of a pair end up built
// differently.
// ---------------------------------------------------------------------------
#pragma once

#include <atomic>
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

// WHAT THIS CLASS IS BUILT ON, all of it now in this namespace:
//
//   HostRegion    the pinned register file + per-core arenas    host_region.hpp
//   Deliverer     the H2D leg (an H2DSocket on the socket route) host_deliver.hpp
//   Transport     the H2H leg -- post / wait / try_wait          host_transport.hpp
//   BankScanner   the work-stealing sweep                        host_scan.hpp
//   PeerTable     host id -> endpoint                            host_transport.hpp
//   HostTopology, ClockSync, Job, WorkerStats, RunStats, OpHandle, VolumeLadder, LadderSync
//
// Transport is the interesting one for review: it is the only one that is already an
// ABSTRACTION rather than a concrete type, which makes it the natural place to meet tt-metal's
// own multi-host plumbing -- DistributedContext is the closest fit, and it covers the
// bootstrap, rank identity and barrier half cleanly. It does NOT cover the data path: it is
// message-passing only, with no one-sided put/get anywhere in it, and this hot path is RMA.

// Operand layout. Named rather than indexed at every use site: "operand[2]" appearing in four
// files is how a producer and a consumer end up disagreeing about which register holds the
// length. These are the SAME indices the device kernel writes and the tt-direct driver reads
// -- the kernel is shared between programs, so they are not this file's to choose.
enum : uint32_t {
    kArgDestUva = 0,     // TX: where these bytes are going
    kArgLength = 1,      // TX and RX: how many bytes
    kArgElapsed = 2,     // running sum of every stage measured so far
    kArgOriginCore = 3,  // RX: which core to reply to on a round trip
    kArgCount = 4,
};

struct SocketConfig {
    // THE VOLUME LADDER, or null. Owned by the caller and must outlive the socket -- it is
    // handed straight to ScanConfig, which hands it to every worker.
    const VolumeLadder* ladder = nullptr;
    LadderSync* ladder_sync = nullptr;  // non-null only when quiescing

    // THE PAYLOAD THIS RUN CARRIES. Needed only to size the receive-slot sweep: the number of
    // slots a message of this size leaves in an arena is what the scanner should look at, not
    // the maximum.
    uint32_t payload_bytes = 0;
    uint32_t chip = 0;     // our own chip index, for the origin selector a notice carries
    uint32_t cores = 0;    // cores in use; bounds the stall dump and the local check
    uint32_t workers = 0;  // 0 => one per CPU, capped at cores
    bool pin = true;
    bool roundtrip = false;

    // THE TWO SENDER KNOBS, AND THEY ARE INDEPENDENT. `send_window` 0 means unset ->
    // cores-in-use, the behaviour before the knob existed; `send_blocking` selects
    // post-and-wait and implies a window of 1. The window caps CONCURRENCY; the shape decides
    // whether the thread parks or spins. A window of 1 is NOT the blocking sender -- it gives
    // the blocking sender's concurrency while still spinning in try_wait() and still SKIPPING
    // on an unreturned credit. Keeping them separate is what lets one measurement tell the two
    // causes apart.
    uint32_t send_window = 0;
    bool send_blocking = false;
    bool record_from_start = true;  // false while a warmup is pending

    // MESSAGES THIS SIDE MUST HANDLE BEFORE THE GATE OPENS BY ITSELF. 0 means "the caller owns
    // the gate" and set_recording() is the only thing that will ever open it.
    //
    // Needed because set_recording() is driven from a PRODUCER loop, which knows the iteration
    // number -- and a side that only receives has no producer loop. Where the roles are split
    // that is half the run: the receiving side set recording_ false at construction and
    // nothing ever flipped it, so every stage recorded zero samples and the table printed a
    // dash on every row. Two of its rows survived only because they bypassed the gate
    // entirely, which is what hid it.
    uint64_t warmup_msgs = 0;

    double ns_per_cycle = 0.0;  // 0 => a cycles-flagged accumulator contributes no sample
};

// Everything the path counts. Atomics because scan workers, the sender thread and the caller's
// shutdown wait all read and write them concurrently.
//
// THREE SEPARATE BARRIERS, AND COLLAPSING ANY PAIR ENDS RUNS SHORT.
//   tx_done    outbound TX jobs finished servicing -- what a producer waits on before
//              re-arming. NOT the scanner's job total: one message produces two serviced jobs
//              (the TX and the RX delivery), so a producer watching the total sees its target
//              met at double rate and never waits (measured 134/160, 149/160).
//   delivered  bytes written into a Tensix L1.
//   home_done  replies delivered back to the ORIGINATING core -- the round-trip barrier.
//              tx_done cannot serve: the turnaround arms a TX word of its own, and on one host
//              the reply re-uses the origin core's TX register.
struct SocketCounters {
    std::atomic<uint64_t> routed_local{0};
    std::atomic<uint64_t> routed_remote{0};
    std::atomic<uint64_t> routed_nowhere{0};
    std::atomic<uint64_t> delivered{0};
    std::atomic<uint64_t> tx_done{0};
    std::atomic<uint64_t> home_done{0};
    std::atomic<uint64_t> replies{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<uint64_t> rejects[8];

    SocketCounters() {
        for (auto& r : rejects) {
            r.store(0, std::memory_order_relaxed);
        }
    }
    SocketCounters(const SocketCounters&) = delete;
    SocketCounters& operator=(const SocketCounters&) = delete;
};

#if defined(HAS_LIBFABRIC)

// ---------------------------------------------------------------------------
// SYMMETRIC OPERATION IS NATIVE. There is no flag for it, because there is no other way this
// socket works.
//
// The name spells two hosts -- chip A, host A, host B, chip B -- so there are exactly two. One
// side's cores are sources and the other's are destinations, and that is what lets a core hold
// ONE L1 buffer at a shared address (`deliver_addr == payload_addr`) instead of two, roughly
// doubling the payload a core can carry.
//
// THE INVARIANT IS PER-CORE, NOT PER-HOST, and saying it that way is what makes it checkable:
// **a core must never be simultaneously a source and a destination.** Splitting roles by host
// is merely the crude way to guarantee it. Both consequences are enforced here rather than left
// to a driver to remember, because a driver that forgets produces a run that passes:
//
//   open()        refuses unless the topology really is two hosts.
//   service_tx()  refuses a LOCAL destination. A UVA resolving to this host names a core that
//                 is already a source, so delivering into it would put a second writer on the
//                 buffer that core is sending from -- overwriting a payload it had not finished
//                 sending, with nothing anywhere reporting it.
//
// ---------------------------------------------------------------------------
// SENDING MUST NOT HAPPEN ON A SCAN THREAD. The send path waits on completions, and a worker
// inside it is a worker not scanning -- so it cannot DELIVER the peer's inbound traffic, so
// the peer never returns the credit it is waiting for. At one core the pool has one worker and
// the stall is total: measured delivered=0 with ~129k scans in 15 s, where an unblocked worker
// does 100M+. Hence one dedicated sender thread.
//
// ONE sender thread, not one per worker: the transport is ONE endpoint with ONE completion
// queue, already serialised inside itself, so a second thread would add contention rather than
// throughput. Scaling that (an endpoint per worker) is a different measurement.
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

    // A socket named for a five-hop path cannot be constructed without the hop in the middle.
    // `transport` is held by reference and must outlive the socket.
    D2H2H2DSocket(HostRegion& region, Deliverer* deliverer, HostTopology topo,
                  ClockSync clock, SocketConfig cfg, Transport& transport);
    ~D2H2H2DSocket();

    D2H2H2DSocket(const D2H2H2DSocket&) = delete;
    D2H2H2DSocket& operator=(const D2H2H2DSocket&) = delete;

    // Builds the peer table, starts the sender thread, then starts the scan pool.
    //
    bool open(std::string& err);

    // Scanner first, then the sender. That order is not cosmetic: the workers are what FILL
    // the send queue, so stopping them first is what lets the queue drain to empty instead of
    // being abandoned with work in it.
    void stop();

    // --- what the caller waits on -----------------------------------------
    const SocketCounters& counters() const { return counters_; }
    std::string first_error() const;

    // True once the middle hop has failed a post or a drain. A faulted transport does not
    // recover, so a caller spinning on counters would otherwise sit there until its own
    // deadline.
    bool transport_failed() const { return transport_failed_.load(std::memory_order_acquire); }

    void set_recording(bool on) {
        recording_.store(on, std::memory_order_relaxed);
        transport_.set_recording(on);
    }

    Transport& transport() const { return transport_; }

    // EXTRA PEERS FOR A MESH, registered before open(). The constructor takes one transport
    // because at two hosts there is exactly one; a mesh passes the first as that primary and
    // the rest through here. Must be called before open() -- open() builds the table from them
    // and the sender thread reads it without synchronisation afterwards.
    void add_peer(Transport* t) { extra_peers_.push_back(t); }

    // Every connected peer, for the driver's end-of-send / end-of-receive barriers. ASCENDING
    // host id -- PeerTable::all() walks its entries in index order -- because a common order
    // across ranks is what makes N pairwise barriers deadlock-free.
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
    // ---- the service path, on the scan workers ---------------------------

    uint64_t service_one(const Job& job, WorkerStats& ws);
    uint64_t service_rx(const Job& job, WorkerStats& ws, bool rec);
    uint64_t service_tx(const Job& job, WorkerStats& ws, bool rec);

    uint64_t deliver_to_l1(const Job& job, WorkerStats& ws, uint32_t stage, uint64_t& stage_ns,
                           bool rec);

    uint64_t elapsed_ns_of(const Job& job, bool& usable) const;

    // ---- the middle hop --------------------------------------------------

    uint64_t deliver_remote(const Job& job, WorkerStats& ws, uint32_t dest_core, uint64_t length,
                            uint64_t accumulated_ns, bool reply);

    bool start_transport(std::string& err);
    void stop_transport();
    void append_transport_stats(RunStats& s) const;
    void dump_transport(std::string& into) const;

    // A credit for a remotely-armed notice: the sender cannot see its slot is free any other
    // way. The credit has to name a peer as well as a core.
    void return_credit(uint32_t origin_selector);

    // ---- shared helpers --------------------------------------------------
    void fail(const std::string& what);
    void retire_tx(uint32_t core);
    static void add_sample(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns);
    static void add_sample_with_size(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns, uint64_t amt);
    bool recording() const { return recording_.load(std::memory_order_relaxed); }

    bool recording_now();

    // A staging slot per thread. post_notice must source its bytes from our own registered
    // region, and two threads staging into the same bytes would interleave their notices.
    static uint32_t my_stage_slot();

    // ---- what a scan worker hands to the sender --------------------------
    struct SendReq {
        uint32_t src_core = 0;
        uint32_t dest_core = 0;
        uint64_t length = 0;
        uint64_t accumulated_ns = 0;
        bool reply = false;
        uint64_t t_queued = 0;
        // The effective address, forwarded unmodified. Zero means "not a store".
        uint64_t dest_uva = 0;
        // Which peer this goes to, decoded once on the scan worker. dest_uva is 0 for a
        // non-store message, so the sender could not recover it.
        uint32_t dest_host = 0;
    };

    // A message in flight, per core. Two phases because the payload must have COMPLETED before
    // its notice is posted -- that is what puts the bytes ahead of the trigger.
    struct SendSlot {
        enum Phase : uint8_t { kIdle = 0, kAwaitPayload = 1, kAwaitNotice = 2 };
        uint8_t phase = kIdle;
        SendReq r{};
        OpHandle payload_op{};
        OpHandle notice_op{};
        uint64_t t0 = 0;
        uint64_t deadline = 0;
        // The endpoint this message was posted on -- a completion must be reaped on the same
        // transport that produced it, or with several peers the poll either never sees it or
        // reaps somebody else's and attributes the host-to-host stage to the wrong message.
        Transport* tp = nullptr;
        // Which of the destination core's receive slots this sender owns. DERIVED from the
        // source, not claimed -- so a slot has one lifetime source and needs no ticket, no
        // tail pointer and no lap check.
        uint32_t rx_slot = 0;
    };

    void sender_loop();
    bool send_try_start(SendSlot& slot, uint32_t core, WorkerStats& ws, bool rec);
    bool send_poll(SendSlot& slot, WorkerStats& ws, bool rec);

    // ---- construction state ----------------------------------------------
    HostRegion& region_;
    Deliverer* deliverer_ = nullptr;
    HostTopology topo_{};
    ClockSync clock_{};
    SocketConfig cfg_{};
    SocketCounters counters_;
    Transport& transport_;

    // Per-core state. Sized to the provisioned maximum rather than cfg_.cores so an
    // out-of-range core index from a corrupt operand indexes a real slot instead of running
    // off the end. PER (PEER, CORE) since wire v2: a credit is an ABSOLUTE count written by
    // the receiver, so with one shared word several peers report the last writer and never the
    // total, and the gate stops opening the moment a core changes destination.
    std::vector<std::vector<std::atomic<uint64_t>>> credit_out_;  // remote deliveries echoed back
    std::vector<std::atomic<uint64_t>> delivered_per_core_;       // the value rdma_signal carries
    // ONE DELIVERY AT A TIME PER CORE. The H2D endpoint, its write pointer and the per-core
    // acked snapshot the delivery wait measures against are all per-core state with NO
    // per-message identity; letting two of a core's receive slots be delivered concurrently by
    // two stealing workers raced all three. A converged end state cannot see a delivery that
    // returned before its own bytes landed.
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

    // ---- the sender -------------------------------------------------------
    std::atomic<bool> transport_failed_{false};

    // ONE QUEUE PER CORE. A destination core has one RX control word, so at most one message
    // may be outstanding to it; per-core queues make that STRUCTURAL rather than a check, which
    // is what lets the sender hold a window across cores. It also removes the head-of-line
    // stall where one core's unreturned credit parked every other core's sends.
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

    std::thread sender_;
    WorkerStats sender_stats_{};
};

#endif  // HAS_LIBFABRIC

}  // namespace tt::tt_metal::experimental
