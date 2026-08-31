// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/D2H2H2DSocket.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace tt::tt_metal::experimental {

#if defined(HAS_LIBFABRIC)

namespace {

// The transport waits. Named so a change to one is not silently a change to a different kind
// of wait: these are per-operation completion budgets, not run budgets.
constexpr uint32_t kCompletionTimeoutMs = 30000;
constexpr uint64_t kCreditWaitNs = 30ull * 1000 * 1000 * 1000;

}  // namespace

// ===========================================================================
// Lifecycle
// ===========================================================================

D2H2H2DSocket::D2H2H2DSocket(HostRegion& region, Deliverer* deliverer, HostTopology topo,
                             ClockSync clock, SocketConfig cfg, Transport& transport)
    : region_(region),
      deliverer_(deliverer),
      topo_(topo),
      clock_(clock),
      cfg_(cfg),
      transport_(transport),
      credit_out_(topo.num > 0 ? topo.num : 1),
      delivered_per_core_(kProvisionedCores),
      deliver_m_(kProvisionedCores),
      notice_sent_(kProvisionedCores),
      tx_retired_(kProvisionedCores) {
    for (auto& per_host : credit_out_) {
        per_host = std::vector<std::atomic<uint64_t>>(kProvisionedCores);
    }
    // Sized to the provisioned maximum like every other per-core vector, so the enqueue path
    // can index by src_core without a bounds decision on the hot path. Missing this compiles
    // perfectly and indexes an empty vector at run time.
    send_q_.resize(kProvisionedCores);
    send_pending_ = std::vector<std::atomic<uint32_t>>(kProvisionedCores);

    set_recording(cfg_.record_from_start);
}

D2H2H2DSocket::~D2H2H2DSocket() { stop(); }

bool D2H2H2DSocket::open(std::string& err) {
    // EXACTLY TWO HOSTS - needs to be changed. the ticket system adresses this issue
    if (topo_.num != 2) {
        err = "D2H2H2DSocket needs exactly two hosts, not " + std::to_string(topo_.num) +
              " -- the path is chip->host->host->chip, and its symmetric operation (one shared "
              "L1 buffer per core) is only coherent when every destination is on the other host";
        return false;
    }
    // Transport first: the pool's callback can hand work to the sender the moment it starts.
    if (!start_transport(err)) {
        return false;
    }

    ScanConfig sc;
    sc.ladder = cfg_.ladder;
    sc.ladder_sync = cfg_.ladder_sync;
    sc.workers = cfg_.workers;
    sc.pin_threads = cfg_.pin;
    // DELIVERY IS DRIVEN BY THE RX CONTROL WORD, IN EVERY MODE. An armed, valid ctrl_rx that is
    // never scanned looks exactly like one that is rejected, and those are different bugs -- so
    // this is not a mode flag, it is always on.
    sc.scan_rx = true;
    // Only the slots this payload can use.
    // NO DERIVED JOB TOTAL. Stopping after `msgs * 2` serviced jobs counts two different things
    // in one budget, so a side receiving faster than it sends reaches the total while still
    // owing sends -- it stops, and its peer starves waiting for a tail that never comes.
    // The caller stops us on the actual conditions instead.
    sc.stop_after_messages = 0;
    // The scanner's own decode/steal-wait samples join the warmup gate. recording_ is a member
    // of this object, which owns the scanner, so it outlives it by construction.
    sc.recording = &recording_;

    scanner_ = std::make_unique<BankScanner>(
        region_, sc, [this](const Job& j, WorkerStats& w) { return service_one(j, w); },
        [this](uint32_t v) { counters_.rejects[v & 7].fetch_add(1, std::memory_order_relaxed); });
    scanner_->start();
    return true;
}

void D2H2H2DSocket::stop() {
    if (stopped_.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    // Workers first: they are what fills the send queue, so stopping them is what lets the
    // queue drain rather than be abandoned with work still in it.
    if (scanner_) {
        scanner_->stop();
        scanner_->join();
    }
    stop_transport();
}

std::string D2H2H2DSocket::first_error() const {
    std::lock_guard<std::mutex> g(err_mutex_);
    return first_error_;
}

void D2H2H2DSocket::fail(const std::string& what) {
    {
        std::lock_guard<std::mutex> g(err_mutex_);
        if (first_error_.empty()) {
            first_error_ = what;
        }
    }
    counters_.errors.fetch_add(1, std::memory_order_relaxed);
}

// ===========================================================================
// Shared helpers
// ===========================================================================

void D2H2H2DSocket::add_sample(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns) {
    // A single choke point rather than an `if (rec)` at each call site: a stage added later
    // would otherwise silently bypass the warmup gate and be the one row in the table that
    // includes the startup transient.
    if (rec) {
        ws.hop[hop].add(ns);
    }
}

void D2H2H2DSocket::add_sample_with_size(WorkerStats& ws, bool rec, uint32_t hop, uint64_t ns,
                                         uint64_t amt) {
    if (rec) {
        ws.hop[hop].add(ns);
        ws.hop_wire_bytes[hop] += amt;
    }
}

// Rings the completion doorbell for a core whose TX control word we have finished with. Called
// on EVERY path out of the TX handler, including the error paths -- a kernel blocked on a
// completion that never comes is a hang, and a hang says far less than an error count.
void D2H2H2DSocket::retire_tx(uint32_t core) {
    const uint64_t n = tx_retired_[core].fetch_add(1, std::memory_order_relaxed) + 1;
    if (deliverer_ != nullptr) {
        (void)deliverer_->ring_completion(core, static_cast<uint32_t>(n));
    }
}

// THE ACCUMULATOR. kArgElapsed carries the running sum of every stage measured so far. Each
// party brackets its OWN stage on its OWN clock, adds its measurement, and passes the total on;
// the last hop reports the sum. No stage is ever computed by subtracting one machine's
// timestamp from another's.
//
// The Tensix is the one party that cannot produce nanoseconds -- it publishes CYCLES, flagged.
// A cycle count read as nanoseconds is wrong by roughly the clock rate and still looks like a
// plausible duration, so with no measured rate this contributes no sample rather than a
// converted guess.
uint64_t D2H2H2DSocket::elapsed_ns_of(const Job& job, bool& usable) const {
    const uint64_t raw = job.operand_count > kArgElapsed ? job.operand[kArgElapsed] : 0;
    if ((ctrl_flags(job.ctrl) & kFlagCycles) == 0) {
        usable = true;
        return raw;  // already nanoseconds
    }
    if (cfg_.ns_per_cycle <= 0.0) {
        usable = false;
        return 0;
    }
    usable = true;
    return static_cast<uint64_t>(static_cast<double>(raw) * cfg_.ns_per_cycle);
}

bool D2H2H2DSocket::recording_now() {
    if (recording_.load(std::memory_order_relaxed)) {
        return true;
    }
    if (cfg_.warmup_msgs == 0) {
        return false;  // the caller owns the gate -- do not second-guess its producer loop
    }
    // max(), NOT a sum: where the roles are split one side only sends and the other only
    // receives, so one of these counters never advances at all. "How many messages has THIS
    // side handled, in whichever direction it handles."
    const uint64_t handled = std::max(counters_.tx_done.load(std::memory_order_relaxed),
                                      counters_.delivered.load(std::memory_order_relaxed));
    if (handled < cfg_.warmup_msgs) {
        return false;
    }
    // open_recording_gate(), NOT set_recording(true): the gate opening and the interval
    // starting are the same event. This was set_recording alone once, which is half of why
    // timed_ns came out empty -- a device producer opened the gate and nothing stamped t0.
    open_recording_gate();
    return true;
}

uint32_t D2H2H2DSocket::my_stage_slot() {
    static std::atomic<uint32_t> next{0};
    thread_local const uint32_t slot = next.fetch_add(1, std::memory_order_relaxed) % kNoticeStageSlots;
    return slot;
}

// ===========================================================================
// The service path -- runs on the scan workers
// ===========================================================================

uint64_t D2H2H2DSocket::deliver_to_l1(const Job& job, WorkerStats& ws, uint32_t stage,
                                      uint64_t& stage_ns, bool rec) {
    if (deliverer_ == nullptr) {
        fail("an RX notice arrived but no deliverer is configured -- the H2D leg cannot run");
        return 0;
    }
    if (job.operand_count < 1) {
        fail("RX notice with no length operand");
        return 0;
    }
    const uint64_t length = job.operand[0];  // base was kArgLength, so operand[0] IS the length
    if (length == 0 || length > kArenaBytes) {
        fail("RX notice with an out-of-range length");
        return 0;
    }

    // THE EFFECTIVE ADDRESS. Zero for kOpSendUva, which takes the layout's fixed destination.
    uint32_t dst_l1 = 0;
    if (ctrl_op_is_store(ctrl_opcode(job.ctrl))) {
        if (job.operand_count < 4) {
            fail("store notice without its destination UVA -- the sender is older than this build");
            return 0;
        }
        const uint64_t dest_uva = job.operand[3];
        const uint32_t off = uva_offset(dest_uva);
        const char* why = nullptr;
        if (off < store_guard_.lo) {
            why = "below the allocator base -- that L1 belongs to tt-metal";
        } else if (off > store_guard_.hi || static_cast<uint64_t>(off) + length > store_guard_.hi) {
            why = "runs past the end of this core's L1";
        } else if ((off < store_guard_.signal_addr + 4 && off + length > store_guard_.signal_addr) ||
                   (off < store_guard_.completion_addr + 4 && off + length > store_guard_.completion_addr) ||
                   (off < store_guard_.stop_addr + 4 && off + length > store_guard_.stop_addr)) {
            // A FORGED DOORBELL RELEASES A KERNEL FOR BYTES THAT NEVER ARRIVED, which is why
            // this is a fault and not a clamp.
            why = "overlaps a doorbell word (rdma_signal / rdma_completion / stop)";
        }
        if (why != nullptr) {
            store_faults_.fetch_add(1, std::memory_order_relaxed);
            std::ostringstream m;
            m << "store fault: core " << job.core << " offset 0x" << std::hex << off << std::dec
              << " length " << length << " -- " << why;
            fail(m.str());
            return 0;
        }
        dst_l1 = off;
    }

    // SERIALISE THIS CORE'S DELIVERIES -- see deliver_m_ in the header. Taken BEFORE the stage
    // clock so the lock wait is billed as queueing rather than appearing inside this stage
    // under a name that says PCIe.
    std::lock_guard<std::mutex> deliver_guard(deliver_m_[job.core]);

    const uint64_t t0 = now_ns();
    if (const std::string e =
            // From the SLOT the notice came from.
            deliverer_->write_payload(job.core, region_.rx_slot(job.core, job.slot, length) + dst_l1,
                                      static_cast<uint32_t>(length), dst_l1);
        !e.empty()) {
        fail("L1 write: " + e);
        return 0;
    }
    const uint64_t t1 = now_ns();
    // A MONOTONIC PER-CORE COUNT, not the message's sequence number. The kernel waits for
    // `doorbell == i + 1` after its i-th message, and the sequence comes from a global counter
    // with no relationship to any one core's iteration -- ringing it would leave the kernel
    // waiting for a value that never arrives.
    const uint64_t bell = delivered_per_core_[job.core].fetch_add(1, std::memory_order_relaxed) + 1;
    if (const std::string e = deliverer_->ring_doorbell(job.core, static_cast<uint32_t>(bell)); !e.empty()) {
        fail("doorbell: " + e);
        return 0;
    }
    // INSIDE THE STAGE, DELIBERATELY. A no-op on the push path; on the socket path it is what
    // keeps this stage meaning "bytes are in L1" rather than "bytes are in host RAM and a core
    // will fetch them shortly".
    const uint64_t t1b = now_ns();
    if (const std::string e = deliverer_->wait_delivered(job.core, static_cast<uint32_t>(bell)); !e.empty()) {
        fail("delivery: " + e);
        return 0;
    }
    const uint64_t t2 = now_ns();

    add_sample(ws, rec, kHopL1Write, t1 - t0);
    add_sample(ws, rec, kHopDoorbell, t1b - t1);
    // Zero on the push path (wait_delivered returns immediately); the device pull on the socket
    // path. Sampled unconditionally so the row's population matches every other stage's -- a
    // diagnostic that only appears in one mode cannot be compared across modes.
    add_sample(ws, rec, kHopPullWait, t2 - t1b);
    add_sample(ws, rec, stage, t2 - t0);
    stage_ns = t2 - t0;
    ws.delivered++;
    // ONE PER DELIVERED MESSAGE -- the receiving side's contribution to the volume ladder.
    ladder_note_message(ws, rec, length);
    counters_.delivered.fetch_add(1, std::memory_order_relaxed);

    // A remotely-armed notice needs a credit returned: the sender cannot see that this slot is
    // free any other way, and without it the next message overwrites an RX control word we have
    // not read yet. Locally-armed notices need none -- the producer is in this process and can
    // see the counter directly.
    if ((ctrl_flags(job.ctrl) & kFlagRemoteNotice) != 0) {
        // The ORIGIN SELECTOR (wire v2) -- it names the peer as well as its core, which is what
        // a bare core index could not do once there was more than one peer.
        const uint32_t origin_sel =
            job.operand_count > 2 ? static_cast<uint32_t>(job.operand[2])
                                  : t6_global_selector(topo_.ident, cfg_.chip, job.core, topo_.chips_per_host);
        return_credit(origin_sel);
    }
    return length;
}

uint64_t D2H2H2DSocket::service_rx(const Job& job, WorkerStats& ws, bool rec) {
    // The bytes are already in our RX arena. Deliver them into L1 and add our stage to the
    // running total the message carries.
    if (job.operand_count < 2) {
        fail("RX notice published fewer operands than the delivery needs");
        return 0;
    }
    const uint64_t carried_ns = job.operand[1];

    // THE PATH IS ONE WAY, so there is no homeward leg and this delivery is always the
    // remote-host-to-remote-T6 stage. The reply arm and the host-side turnaround that used to
    // live here are gone: the turnaround armed `ctrl_tx(job.core)`, the far core's OWN TX
    // control register, which that core's kernel also writes -- two writers on one word. The
    // full text of what was removed is in host_socket.cpp's service_rx().
    const uint32_t stage = kHopRemoteHostToRemoteT6;

    uint64_t stage_ns = 0;
    const uint64_t moved = deliver_to_l1(job, ws, stage, stage_ns, rec);
    if (moved == 0) {
        return 0;
    }
    // THE RECEIVING SIDE'S CONTRIBUTION TO THE TIMED NUMERATOR: bytes that reached an L1. This
    // is the only byte count a receive-only side has, and it is the site whose absence made
    // that side report a confident zero.
    if (rec) {
        ws.timed_bytes += moved;
        // Receiving side: the trace is a DELIVERY curve here, where the tx side's is a send
        // curve. Same distinction the two roles' timed_bytes already carry.
        trace_add(ws, rec, timed_start_ns(), moved, stage_ns);
    }
    const uint64_t total_ns = carried_ns + stage_ns;

    // This was the last hop: the sum that arrived plus our delivery.
    add_sample(ws, rec, kHopOneWayTotal, total_ns);
    return moved;
}

uint64_t D2H2H2DSocket::service_tx(const Job& job, WorkerStats& ws, bool rec) {
    // A TX job that cannot be serviced must STILL count as done. The producer blocks on this
    // counter, so a message dropped here without incrementing it would stall the run until the
    // drain deadline and report a timeout instead of the actual error.
    //
    // The immediate form publishes ONE operand register -- the destination UVA -- because its
    // length rides in the control word. Requiring two would reject it as malformed.
    const bool store_imm = ctrl_op_has_imm(ctrl_opcode(job.ctrl));
    if (job.operand_count < (store_imm ? 1u : 2u)) {
        fail("TX message published fewer operands than its opcode needs");
        counters_.tx_done.fetch_add(1, std::memory_order_release);
        retire_tx(job.core);
        return 0;
    }
    const uint64_t dest_uva = job.operand[kArgDestUva];
    // WHERE THE LENGTH LIVES IS THE OPCODE'S CHOICE -- an immediate in the instruction, or a
    // 64-bit operand register. The one line that has to know; every hop after it carries a
    // plain byte count.
    const uint64_t length = store_imm ? static_cast<uint64_t>(ctrl_imm(job.ctrl)) : job.operand[kArgLength];
    if (length == 0 || length > kArenaBytes) {
        fail("TX message length out of range");
        counters_.tx_done.fetch_add(1, std::memory_order_release);
        retire_tx(job.core);
        return 0;
    }

    // STAGE 1: the sender's OWN measurement of its push, taken on the sender's clock and
    // published with the message. Nothing is subtracted across a domain.
    bool usable = false;
    uint64_t accumulated = elapsed_ns_of(job, usable);
    const bool returning = (ctrl_flags(job.ctrl) & kFlagReply) != 0;
    if (!usable) {
        accumulated = 0;
    } else if (!returning && accumulated > 0) {
        // Outbound: the whole carried value IS this sender's push, so it is stage 1. On a return
        // leg the carried value is a running total of everything so far, not this leg alone, so
        // it is not a stage-4 sample -- the far sender's own push cost would have to arrive as
        // an increment, which needs an echo kernel that does not exist. Stage 4 therefore stays
        // empty rather than being filled with a running total mislabelled as one leg.
        add_sample(ws, rec, kHopT6ToHost, accumulated);
    }

    const uint32_t reach = uva_host_reach(dest_uva, topo_);
    const uint32_t dest_core = uva_t6_core(dest_uva);

    switch (reach) {
        case kHostReachNoSuchHost:
            // A provisioning answer, not a corrupt-address answer -- counted apart for the same
            // reason a resolver splits "no table entry" from "bad region".
            counters_.routed_nowhere.fetch_add(1, std::memory_order_relaxed);
            counters_.tx_done.fetch_add(1, std::memory_order_release);
            retire_tx(job.core);
            return 0;

        case kHostReachLocal: {
            // REFUSED, AND THIS IS THE PER-CORE HALF OF THE SYMMETRIC INVARIANT.
            //
            // Symmetric operation is native here (see the header): one side's cores are
            // sources, the other's are destinations, and each core therefore holds ONE L1
            // buffer at a shared address instead of two. A UVA that resolves to THIS host
            // names a core that is already a source, so delivering into it would put a second
            // writer on the buffer that core is sending from -- overwriting a payload it had
            // not finished sending.
            //
            // NAMED RATHER THAN SILENTLY DROPPED, because this is the shape a misroute takes.
            // Destinations are pre-filled with the complement of what should arrive, so a
            // payload delivered to the wrong place still verifies as correct wherever it did
            // land; without this refusal a wrong selector would pass as a clean run. Counted as
            // routed_local so the count still says how many arrived, and tx_done/retire_tx
            // still advance -- a producer blocked on a message dropped without them stalls to
            // its deadline and reports a timeout instead of the error that actually happened.
            counters_.routed_local.fetch_add(1, std::memory_order_relaxed);
            std::ostringstream m;
            m << "UVA resolves to this host (core " << dest_core
              << "), but this socket is symmetric: a core is a source or a destination, never "
                 "both, because the two share one L1 buffer. A destination must name the other "
                 "host";
            fail(m.str());
            counters_.tx_done.fetch_add(1, std::memory_order_release);
            retire_tx(job.core);
            return 0;
        }

        case kHostReachRemote:
            counters_.routed_remote.fetch_add(1, std::memory_order_relaxed);
            return deliver_remote(job, ws, dest_core, length, accumulated, returning);

        default:
            fail("unclassifiable host reach");
            counters_.tx_done.fetch_add(1, std::memory_order_release);
            retire_tx(job.core);
            return 0;
    }
}

uint64_t D2H2H2DSocket::service_one(const Job& job, WorkerStats& ws) {
    const uint64_t t_service = now_ns();
    const bool rec = recording_now();
    if (rec) {
        ws.hop[kHopNotice].add(t_service > job.t_notice ? t_service - job.t_notice : 0);
    }
    // ONE FUNCTION, BOTH DIRECTIONS, ON THE SAME THREADS. Any arrangement that finishes one
    // direction before starting the other stops at the first iteration with this host waiting
    // for records from cores that are waiting for this host.
    return job.dir == Dir::Rx ? service_rx(job, ws, rec) : service_tx(job, ws, rec);
}

// ===========================================================================
// The middle hop
// ===========================================================================

bool D2H2H2DSocket::start_transport(std::string& err) {
    // THE PEER TABLE, built from the transport this class was constructed with plus any mesh
    // peers. Sized to the whole topology so a host inside it with no entry reports a
    // provisioning gap rather than being indistinguishable from a host that does not exist.
    peers_.configure(topo_.num, topo_.ident);
    if (const std::string e = peers_.connect_peer(transport_.peer().host_id, &transport_); !e.empty()) {
        err = e;
        return false;
    }
    for (Transport* t : extra_peers_) {
        if (const std::string e = peers_.connect_peer(t->peer().host_id, t); !e.empty()) {
            err = e;
            return false;
        }
    }

    const uint32_t structural = cfg_.cores ? cfg_.cores : 1u;
    send_blocking_ = cfg_.send_blocking;
    if (send_blocking_) {
        if (cfg_.send_window != 0 && cfg_.send_window != 1) {
            err = "send-blocking implies a window of 1, but window " + std::to_string(cfg_.send_window) +
                  " was requested";
            return false;
        }
        send_window_ = 1;
    } else if (cfg_.send_window == 0) {
        send_window_ = structural;  // unset is not zero: the pre-knob behaviour
    } else if (cfg_.send_window > structural) {
        err = "send window " + std::to_string(cfg_.send_window) + " exceeds cores in use (" +
              std::to_string(structural) +
              "); a destination core has ONE RX control word, so the window is across cores, "
              "never within one";
        return false;
    } else {
        // REFUSED BY NAME, NEVER SILENTLY REDUCED: a run labelled window=8 that ran at 4 is a
        // file that lies. The constants come from the shared layout header rather than being
        // repeated here -- a local 3 and 8 would be free to drift.
        //
        // THE TIGHTEST PEER'S QUEUE, because one sender thread posts to all of them, so a depth
        // the narrowest endpoint cannot honour is a depth the sender cannot use. tx_depth() is 0
        // where the provider has no such number, which means "do not clamp", not "clamp to
        // zero".
        uint64_t depth = 0;
        for (Transport* t : peers_.all()) {
            const uint64_t d = t->tx_depth();
            if (d != 0 && (depth == 0 || d < depth)) {
                depth = d;
            }
        }
        if (depth > kTxDepthReserve) {
            const uint64_t allowed = (depth - kTxDepthReserve) / kTxDepthPerMessage;
            if (static_cast<uint64_t>(cfg_.send_window) > allowed) {
                err = "send window " + std::to_string(cfg_.send_window) +
                      " exceeds what this endpoint's TX queue allows (" + std::to_string(allowed) +
                      "; tx_attr->size=" + std::to_string(depth) + ")";
                return false;
            }
        }
        send_window_ = cfg_.send_window;
    }

    // ANNOUNCED, because a run whose shape is only discoverable by reading the CSV afterwards is
    // how four pair runs came to set a variable the binary did not have. stderr, not stdout,
    // because a sweep driver captures stdout into a shell variable and prints only its own
    // verdict line.
    std::cerr << "  sender      "
              << (send_blocking_ ? "BLOCKING (post-and-wait, credit waited)"
                                 : "windowed (post-and-poll, credit skipped)")
              << ", window " << send_window_ << " of " << structural << " core"
              << (structural == 1 ? "" : "s");
    // The TIGHTEST peer's depth -- the same number the clamp used, so the line reports what
    // actually bounded the window rather than one arbitrary endpoint's figure.
    uint64_t announce_depth = 0;
    for (Transport* t : peers_.all()) {
        const uint64_t d = t->tx_depth();
        if (d != 0 && (announce_depth == 0 || d < announce_depth)) {
            announce_depth = d;
        }
    }
    if (announce_depth > 0) {
        std::cerr << ", tx_depth " << announce_depth;
    }
    std::cerr << "\n";

    sender_ = std::thread([this] { sender_loop(); });
    return true;
}

void D2H2H2DSocket::stop_transport() {
    if (!sender_.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> g(send_m_);
        send_stop_ = true;
    }
    send_cv_.notify_all();
    sender_.join();
}

void D2H2H2DSocket::return_credit(uint32_t origin_selector) {
    const uint32_t origin_host = t6_selector_host(origin_selector, topo_.chips_per_host);
    const uint32_t origin_core = t6_selector_core(origin_selector);
    if (origin_host >= credit_out_.size()) {
        fail("credit: origin selector names host " + std::to_string(origin_host) +
             " which is outside the configured topology");
        return;
    }
    const uint64_t n = credit_out_[origin_host][origin_core].fetch_add(1, std::memory_order_relaxed) + 1;
    // ONE PEER OR REFUSE. With several peers a credit sent to the wrong one stalls the real
    // sender forever, and an uncredited sender reads as a hang over there.
    uint32_t why = kPeerOk;
    Transport* const back = peers_.for_host(origin_host, why);
    if (back == nullptr) {
        // The message arrived, so its sender exists; a lookup failure means our table disagrees
        // with what the peer says about itself.
        fail("credit: origin host " + std::to_string(origin_host) + ": " + peer_why_name(why));
        return;
    }
    if (const std::string e = back->post_credit(origin_core, topo_.ident, n, my_stage_slot()); !e.empty()) {
        fail("credit: " + e);
        transport_failed_.store(true, std::memory_order_release);
    }
}

uint64_t D2H2H2DSocket::deliver_remote(const Job& job, WorkerStats& ws, uint32_t dest_core,
                                       uint64_t length, uint64_t accumulated_ns, bool reply) {
    (void)ws;
    // THE DESTINATION HOST MUST BE ONE WE ARE CONNECTED TO.
    //
    // host_reach() says only local/remote, so at three hosts a UVA naming host 2 classifies
    // exactly like one naming host 1 and would be posted down whichever transport exists. The
    // bytes land on the WRONG HOST and the run still passes: the destination is pre-filled with
    // the complement of what should arrive, so a payload delivered to the wrong place verifies
    // as correct where it did arrive, and no counter moves. Counted as routed_nowhere because it
    // is a provisioning gap, not a corrupt address.
    const uint32_t want_host = uva_target_host(job.operand[kArgDestUva], topo_);
    {
        uint32_t why = kPeerOk;
        if (peers_.for_host(want_host, why) == nullptr) {
            std::ostringstream m;
            m << "UVA names host " << want_host << ": " << peer_why_name(why);
            fail(m.str());
            counters_.routed_nowhere.fetch_add(1, std::memory_order_relaxed);
            counters_.tx_done.fetch_add(1, std::memory_order_release);
            retire_tx(job.core);
            return 0;
        }
    }
    // HAND IT TO THE SENDER THREAD -- DO NOT SEND ON THIS THREAD. The send path waits on
    // completions, and a worker inside it cannot scan, so it cannot deliver the peer's inbound
    // traffic, so the peer never returns the credit it is waiting for. At one core the pool has
    // one worker and the stall is total.
    //
    // tx_done and the completion doorbell move to the sender thread WITH the work: both mean
    // "these bytes have left", so incrementing them here -- before anything has been sent --
    // would free the kernel to overwrite an arena the transport has not read.
    SendReq r;
    r.src_core = job.core;
    r.dest_core = dest_core;
    r.length = length;
    r.accumulated_ns = accumulated_ns;
    r.reply = reply;
    r.t_queued = now_ns();
    // Register kArgDestUva holds the destination in BOTH store encodings: the immediate form
    // spends its saved register on the length, not on the address.
    r.dest_uva = ctrl_op_is_store(ctrl_opcode(job.ctrl)) ? job.operand[kArgDestUva] : 0ull;
    r.dest_host = want_host;
    {
        std::lock_guard<std::mutex> g(send_m_);
        send_q_[r.src_core].push_back(r);
        send_pending_[r.src_core].fetch_add(1, std::memory_order_release);
        send_depth_.fetch_add(1, std::memory_order_release);
    }
    send_cv_.notify_one();
    // Zero bytes: nothing has moved yet. The sender counts the payload once it has actually
    // gone, so a queued message is never reported as a transferred one.
    return 0;
}

// ===========================================================================
// The sender, as a state machine
// ===========================================================================
//
// One thread that posted a payload and waited on it inline held at most ONE RMA in flight on
// the whole machine -- measured at 2.8 of 8 available with the fabric at 99% of line rate. The
// payload must still COMPLETE before its notice is posted, so each message walks two phases and
// the loop polls all of them instead of blocking on one.

bool D2H2H2DSocket::send_try_start(SendSlot& slot, uint32_t core, WorkerStats& ws, bool rec) {
    if (slot.phase != SendSlot::kIdle) {
        return false;
    }
    // THE CREDIT CHECK IS A SKIP, NOT A WAIT, in the windowed shape. It used to park this thread
    // for up to kCreditWaitNs, so one core's unreturned credit stalled every other core's sends.
    // Checked BEFORE dequeuing: a request pulled off and found unsendable would have to be
    // pushed back, reordering it against its own core's traffic.
    //
    // CHEAPEST CHECK FIRST -- one relaxed atomic load for the common "nothing queued" lap.
    if (send_pending_[core].load(std::memory_order_acquire) == 0) {
        return false;
    }
    const uint64_t already = notice_sent_[core].load(std::memory_order_acquire);
    if (already > 0) {
        if (credit_total(region_, core) < already) {
            if (!send_blocking_) {
                return false;
            }
            // THE BLOCKING SHAPE PARKS HERE. Safe only because this is the dedicated sender
            // thread: the scan workers keep delivering while it sleeps, and delivering is what
            // makes the peer return the credit. It reinstates the head-of-line stall on purpose
            // -- that is the behaviour under measurement, not a defect.
            //
            // STATE 1 IS "credit-wait", and the stall dump advertises it by name. Without these
            // two stores a sender parked here reports `idle` -- a dump that says the opposite of
            // what is happening.
            sender_state_.store(1, std::memory_order_relaxed);
            const uint64_t dl = now_ns() + kCreditWaitNs;
            while (credit_total(region_, core) < already && now_ns() < dl) {
                std::this_thread::yield();
            }
            sender_state_.store(0, std::memory_order_relaxed);
            if (credit_total(region_, core) < already) {
                fail("credit wait: timed out on core " + std::to_string(core) + " (wanted " +
                     std::to_string(already) + ")");
                return false;
            }
        }
    }

    SendReq r;
    {
        std::lock_guard<std::mutex> g(send_m_);
        if (send_q_[core].empty()) {
            return false;
        }
        r = send_q_[core].front();
        send_q_[core].pop_front();
        send_pending_[core].fetch_sub(1, std::memory_order_release);
        send_depth_.fetch_sub(1, std::memory_order_release);
    }

    add_sample(ws, rec, kHopSendQueueWait, now_ns() - r.t_queued);

    const uint64_t local_off = HostRegion::tx_arena_off(r.src_core);
    const uint32_t store_off = (r.dest_uva != 0) ? uva_offset(r.dest_uva) : 0u;
    // CHECKED BY THE SENDER TOO, not only by the receiver: past the arena is the next core's TX
    // arena, and that is this side's memory.
    if (static_cast<uint64_t>(store_off) + r.length > kArenaBytes) {
        std::ostringstream m;
        m << "store fault: destination offset 0x" << std::hex << store_off << std::dec << " + length "
          << r.length << " runs past the " << kArenaBytes << " B arena and into the next core's";
        fail(m.str());
        if (!r.reply) {
            counters_.tx_done.fetch_add(1, std::memory_order_release);
            retire_tx(r.src_core);
        }
        return false;
    }
    // ONE RECEIVE SLOT. The aliased ring's data region is exactly one payload, so the arena
    // start is the only place a payload can land.
    slot.rx_slot = 0;
    const uint64_t remote_off = HostRegion::rx_arena_off(r.dest_core) + store_off;

    // The endpoint for this message's destination, resolved on the posting thread.
    uint32_t why = kPeerOk;
    Transport* const tp = peers_.for_host(r.dest_host, why);
    if (tp == nullptr) {
        fail("send: host " + std::to_string(r.dest_host) + ": " + peer_why_name(why));
        if (!r.reply) {
            counters_.tx_done.fetch_add(1, std::memory_order_release);
            retire_tx(r.src_core);
        }
        return false;
    }

    sender_state_.store(2, std::memory_order_relaxed);
    slot.t0 = now_ns();
    slot.deadline = slot.t0 + kCreditWaitNs;
    slot.r = r;
    slot.tp = tp;

    if (const std::string e = tp->post(local_off, remote_off, r.length, r.src_core, slot.payload_op);
        !e.empty()) {
        fail("transport post: " + e);
        transport_failed_.store(true, std::memory_order_release);
        sender_state_.store(0, std::memory_order_relaxed);
        if (!r.reply) {
            counters_.tx_done.fetch_add(1, std::memory_order_release);
            retire_tx(r.src_core);
        }
        return false;
    }
    slot.phase = SendSlot::kAwaitPayload;
    return true;
}

bool D2H2H2DSocket::send_poll(SendSlot& slot, WorkerStats& ws, bool rec) {
    if (slot.phase == SendSlot::kIdle) {
        return false;
    }
    const bool timed_out = now_ns() > slot.deadline;
    Completion c;
    bool failed = false;

    // ONE PHASE ADVANCE, AND IT IS THE ONLY PLACE THE TWO SHAPES DIFFER -- everything after it
    // (the ordering, the samples, tx_done, retire_tx) is the same code either way, which is what
    // makes the switch a switch rather than two implementations free to drift. In the blocking
    // shape this always returns true: a wait() that times out comes back with c.ok false and its
    // own message, so the `timed_out` arms are unreachable there and the error still names the
    // operation.
    auto advance = [this, &slot](OpHandle& op, Completion& out) -> bool {
        if (send_blocking_) {
            out = slot.tp->wait(op, kCompletionTimeoutMs);
            return true;
        }
        // try_wait(), not wait(op, 0): a zero timeout means THIRTY SECONDS, and wait() consumes
        // the handle on entry, so a caller cannot peek and come back.
        return slot.tp->try_wait(op, out);
    };

    if (slot.phase == SendSlot::kAwaitPayload) {
        if (!advance(slot.payload_op, c)) {
            if (!timed_out) {
                return false;
            }
            fail("transport completion: payload timed out on core " + std::to_string(slot.r.src_core));
            failed = true;
        } else if (!c.ok) {
            fail("transport completion: " + c.error);
            failed = true;
        }
        if (!failed) {
            // The RX notice, posted only now: the payload has COMPLETED, so a peer that sees the
            // trigger is guaranteed the bytes behind it.
            sender_state_.store(3, std::memory_order_relaxed);
            if (const std::string e = slot.tp->post_notice(
                    slot.r.dest_core, slot.rx_slot, slot.r.length,
                    t6_global_selector(topo_.ident, cfg_.chip, slot.r.src_core, topo_.chips_per_host),
                    slot.r.accumulated_ns + (now_ns() - slot.t0), slot.r.reply, my_stage_slot(),
                    slot.notice_op, slot.r.dest_uva);
                !e.empty()) {
                fail("transport notice: " + e);
                failed = true;
            } else {
                slot.phase = SendSlot::kAwaitNotice;
                return true;
            }
        }
    } else {
        if (!advance(slot.notice_op, c)) {
            if (!timed_out) {
                return false;
            }
            fail("notice completion: timed out on core " + std::to_string(slot.r.src_core));
            failed = true;
        } else if (!c.ok) {
            fail("notice completion: " + c.error);
            failed = true;
        }
        if (!failed) {
            notice_sent_[slot.r.src_core].fetch_add(1, std::memory_order_release);
            sender_state_.store(0, std::memory_order_relaxed);
            const uint64_t stage_ns = now_ns() - slot.t0;
            add_sample_with_size(ws, rec, kHopHostToRemoteHost, stage_ns,
                                 static_cast<uint64_t>(slot.r.length) + kNoticeBytes);
            const uint64_t moved = slot.r.length;
            ws.bytes += moved;
            ladder_note_message(ws, rec, moved);
            if (rec) {
                ws.timed_bytes += moved;
                trace_add(ws, rec, timed_start_ns(), moved, stage_ns);
            }
            if (!slot.r.reply) {
                counters_.tx_done.fetch_add(1, std::memory_order_release);
                retire_tx(slot.r.src_core);
            }
            slot.phase = SendSlot::kIdle;
            return true;
        }
    }

    // A failed message must STILL retire, or the producer blocks on tx_done to the drain
    // deadline and reports a timeout instead of the error that actually happened.
    transport_failed_.store(true, std::memory_order_release);
    sender_state_.store(0, std::memory_order_relaxed);
    if (!slot.r.reply) {
        counters_.tx_done.fetch_add(1, std::memory_order_release);
        retire_tx(slot.r.src_core);
    }
    slot.phase = SendSlot::kIdle;
    return true;
}

void D2H2H2DSocket::sender_loop() {
    std::vector<SendSlot> slots(kProvisionedCores);
    WorkerStats& ws = sender_stats_;
    const uint32_t n = cfg_.cores ? cfg_.cores : 1u;

    for (;;) {
        const bool rec = recording();
        bool progressed = false;
        uint32_t in_flight = 0;

        // POLL FIRST, so a completion frees its slot before the start pass looks at it -- the
        // other order costs a full lap of latency per message.
        for (uint32_t c = 0; c < n; ++c) {
            if (slots[c].phase != SendSlot::kIdle) {
                progressed |= send_poll(slots[c], ws, rec);
                if (slots[c].phase != SendSlot::kIdle) {
                    ++in_flight;
                }
            }
        }
        // Rotating origin so a core with a permanently full queue cannot starve the others -- a
        // fixed order would reintroduce on the send side the imbalance the work-stealing pool
        // exists to avoid.
        for (uint32_t k = 0; k < n; ++k) {
            // THE WINDOW -- the one condition that separates N=1 from N=8. Everything downstream
            // is identical, which is what makes a sweep over it measure concurrency rather than
            // two code paths.
            if (in_flight >= send_window_) {
                break;
            }
            const uint32_t c = (send_rr_ + k) % n;
            if (send_try_start(slots[c], c, ws, rec)) {
                progressed = true;
                ++in_flight;
            }
        }
        send_rr_ = (send_rr_ + 1) % n;

        if (progressed) {
            continue;
        }
        if (in_flight != 0) {
            std::this_thread::yield();
            continue;
        }
        std::unique_lock<std::mutex> g(send_m_);
        if (send_stop_ && send_depth_.load(std::memory_order_acquire) == 0) {
            return;
        }
        send_cv_.wait_for(g, std::chrono::microseconds(50), [this] {
            return send_stop_ || send_depth_.load(std::memory_order_acquire) != 0;
        });
        if (send_stop_ && send_depth_.load(std::memory_order_acquire) == 0) {
            return;
        }
    }
}

// ===========================================================================
// Reporting
// ===========================================================================

void D2H2H2DSocket::append_transport_stats(RunStats& s) const {
    // The sender thread's samples ARE the host-to-host stages; without this they are simply
    // absent from the table.
    s.per_worker.push_back(sender_stats_);

    // THE RESOLVED SHAPE, not what was requested.
    s.window = send_window_;
    s.sender_shape = send_blocking_ ? "blocking" : "windowed";

    // The progress thread's post->completion samples, when the transport was built to measure
    // them. FOLDED OVER EVERY PEER with the same Welford merge the distributions use -- taking
    // one endpoint's figures would report a fraction of the run's completions as all of them.
    //
    // NOT w.bytes: total_bytes() sums that field and the sender has already counted these same
    // payloads, so setting it would double the wall-clock throughput column.
    RetireStats rs{};
    for (Transport* t : peers_.all()) {
        const RetireStats one = t->retire_stats();
        if (one.n == 0) {
            continue;
        }
        if (rs.n == 0) {
            rs = one;
            continue;
        }
        const double delta = one.mean_ns - rs.mean_ns;
        const uint64_t n = rs.n + one.n;
        rs.m2 += one.m2 +
                 delta * delta * static_cast<double>(rs.n) * static_cast<double>(one.n) / static_cast<double>(n);
        rs.mean_ns += delta * static_cast<double>(one.n) / static_cast<double>(n);
        rs.min_ns = std::min(rs.min_ns, one.min_ns);
        rs.max_ns = std::max(rs.max_ns, one.max_ns);
        rs.n = n;
    }
    if (rs.n > 0) {
        WorkerStats w{};
        Dist& d = w.hop[kHopH2HRetire];
        d.n = rs.n;
        d.min = rs.min_ns;
        d.max = rs.max_ns;
        d.mean = rs.mean_ns;
        d.m2 = rs.m2;
        s.per_worker.push_back(w);
    }
}

void D2H2H2DSocket::dump_transport(std::string& into) const {
    // WHAT THE TRANSPORT THINKS IS OUTSTANDING. A sender parked in state 2 or 3 is waiting for
    // one specific operation, and these say whether the provider was given it (posted), whether
    // anything came back (retired), and whether something came back that belonged to nobody
    // (unmatched). A stall with outstanding=1 and unmatched=0 is a completion the provider never
    // produced; unmatched>0 is one we failed to attribute.
    //
    // ONE LINE PER PEER, ACCUMULATED into a single stream rather than assigned per peer --
    // assigning would leave only the last one, which is exactly the shape of bug this dump
    // exists to catch. Summing them would hide an endpoint that stopped completing behind ones
    // that did not.
    std::ostringstream m;
    for (Transport* tp_i : peers_.all()) {
        const TransportDiag d = tp_i->diag();
        m << "    transport[host " << tp_i->peer().host_id << "]: posted=" << d.posted
          << " retired=" << d.retired << " injected=" << d.injected << " outstanding=" << d.outstanding
          << " unmatched=" << d.unmatched << " abandoned=" << d.abandoned;
        if (d.outstanding != 0) {
            m << " oldest_tag=" << d.oldest_tag;
        }
        m << "\n";
        if (!d.last_error.empty()) {
            m << "    transport[host " << tp_i->peer().host_id << "] last CQ error: " << d.last_error << "\n";
        }
    }
    into = m.str();
}

RunStats D2H2H2DSocket::collect() const {
    RunStats s = scanner_ ? scanner_->collect() : RunStats{};
    // THE BRACKET: a duration only when both stamps landed and the end is after the start, so an
    // aborted run writes an EMPTY throughput cell rather than a 0.0 that a sweep would average
    // in.
    const uint64_t t0 = timed_start_ns_.load(std::memory_order_relaxed);
    const uint64_t t1 = timed_end_ns_.load(std::memory_order_relaxed);
    s.timed_ns = (t0 > 0 && t1 > t0) ? (t1 - t0) : 0;
    // The width the trace actually used, so the trace file is self-describing.
    s.trace_shift = trace_shift();
    append_transport_stats(s);
    return s;
}

std::string D2H2H2DSocket::stall_dump(const char* where) const {
    std::ostringstream m;
    m << "\n  [STALL @ " << where << "]\n";
    m << "    tx_done=" << counters_.tx_done.load() << " delivered=" << counters_.delivered.load()
      << " home=" << counters_.home_done.load() << " routed_remote=" << counters_.routed_remote.load()
      << " errors=" << counters_.errors.load() << "\n";
    m << "    sender: " << sender_state_.load() << "  (0=idle 1=credit-wait 2=payload 3=notice)\n";
    std::string t;
    dump_transport(t);
    m << t;
    // Is the RX direction being scanned, and is the scanner rejecting what it finds? An armed,
    // valid ctrl_rx that is never serviced is either not scanned or rejected, and those are
    // different bugs. scan_rx is unconditional here, so only the rejects vary.
    m << "    scan_rx=yes  rejects:";
    for (uint32_t i = 0; i < 8; ++i) {
        const uint64_t n = counters_.rejects[i].load();
        if (n) {
            m << " " << ctrl_verdict_name(i) << "=" << n;
        }
    }
    m << "\n";
    const uint32_t show = cfg_.cores < 8 ? cfg_.cores : 8;
    for (uint32_t c = 0; c < show; ++c) {
        m << "    core " << c << ": notice_sent=" << notice_sent_[c].load()
          << " credit_in=" << credit_total(region_, c) << " tx_retired=" << tx_retired_[c].load()
          << " delivered=" << delivered_per_core_[c].load() << " ctrl_tx=0x" << std::hex
          << load_acquire(region_.ctrl_tx(c)) << " ctrl_rx=0x" << load_acquire(region_.ctrl_rx(c))
          << std::dec << "\n";
    }
    const std::string fe = first_error();
    if (!fe.empty()) {
        m << "    first error: " << fe << "\n";
    }
    return m.str();
}

#endif  // HAS_LIBFABRIC

}  // namespace tt::tt_metal::experimental
