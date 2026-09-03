// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host leg: one interface over MPI one-sided RMA.
//
//   single host, two processes:  osc/sm or osc/rdma, chosen by the runtime
//   two hosts,   two processes:  osc/rdma or osc/ucx, chosen by the runtime
//
// MPI_Win_create() is called exactly once, over the same span PinnedMemory pinned. Both are
// independent, refcounted pins of the same pages -- one for the TT device, one for the NIC --
// so a payload a Tensix core pushed over PCIe is already inside the window when the transport
// goes to send it. No bounce buffer, no per-message registration, and the send is described by
// an OFFSET into the region rather than by a pointer, which is exactly how a window is
// addressed: (target rank, displacement).
//
// A one-sided MPI_Rput into the peer's RX arena is the operation this design wants. What MPI
// will NOT give -- and libfabric did -- is remote completion from the operation's own handle:
// MPI_Rput retires when the ORIGIN buffer is reusable. See flush() below; it is the single
// semantic difference the whole port turns on.
//
// Transport selection is the launcher's job, not this file's. The one component that would
// break the design is osc/pt2pt, which needs the TARGET to enter MPI for a put to progress
// while the receiving host here only polls memory -- and it excludes itself, refusing to build
// a window at all under MPI_THREAD_MULTIPLE. describe() names what was actually chosen.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental {

struct TransportConfig {
    // The MPI rank this endpoint talks to. There is no address to configure and no listen/
    // connect asymmetry: the window is collective, so both sides are already introduced by the
    // time any Transport exists.
    uint32_t peer_rank = 0;
    uint32_t timeout_ms = 30000;

    // Geometry, sent in the bootstrap hello so the peer can check we agree before a byte moves.
    //
    // host_id IS NOT HERE. It is the MPI rank, read from DistributedContext by connect_mesh(),
    // because two sources for one fact is how the peer table and the communicator end up
    // disagreeing about which host this process is.
    uint32_t host_id = 0;  // set by connect_mesh() from the rank; not for callers to choose
    uint32_t chips_per_host = 1;
    uint32_t grid_width = 0;
    uint32_t cores_in_use = 0;

    // Off by default because it puts a completion-slot acquire/release on a data path that
    // currently takes no such lock, and because it changes what the diag counters mean (a
    // measured payload retires into `retired` instead of `unmatched`). Both are harmless and
    // both are visible, which is exactly why they should be opted into rather than assumed.
    bool measure_retire = false;
};

struct RetireStats {
    uint64_t n = 0;
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
    double mean_ns = 0.0;
    double m2 = 0.0;
    uint64_t bytes = 0;        // payload bytes across the measured operations
    uint64_t unmeasured = 0;
};

// What the two sides must tell each other before a byte moves. Exchanged with a pairwise
// MPI_Sendrecv once the window exists.
//
// No region base and no memory key, which is the part worth noticing: a window is addressed by
// (target rank, byte displacement), so the peer's virtual address is not merely unnecessary, it
// is not knowable and not wanted. Everything here is a geometry cross-check -- the offsets the
// two sides compute independently must mean the same thing, and a disagreement produces a
// VALID offset naming the wrong core rather than an error.
struct PeerInfo {
    uint64_t region_bytes = 0;
    uint32_t host_id = 0;
    uint32_t chips_per_host = 0;
    uint32_t grid_width = 0;
    uint32_t cores_in_use = 0;
    uint32_t provisioned_cores = 0;
    uint64_t arena_stride = 0;
    uint64_t arena_bytes = 0;
};

// A completion the caller is waiting on.
struct Completion {
    bool ok = false;
    std::string error;
};

// ONE POSTED OPERATION, NAMED. A caller waits for ITS OWN operation to retire, not for
// "some operation" to retire.
//
// The version this replaced counted completions into one global counter and had each waiter
// draw a ticket from it. That is not an accounting detail, it is the whole difference
// between a wait that means something and a wait that does not: any op whose completion the
// counter saw satisfied whichever waiter happened to hold the next ticket, and any op whose
// completion never arrived desynchronised every wait that followed it, permanently. It
// stalled a run whose payload was already in the peer's memory, and no counter could have
// said which transfer the missing completion belonged to.
//
// The handle names a slot in the transport's own request table, one MPI_Request per slot. MPI
// hands back a request per operation and asks nothing of the caller's storage, so unlike the
// provider-context arrangement this replaced there is nothing the transport can get wrong by
// inventing a pointer -- but the table is still per-operation rather than a shared counter, for
// the reason above.
struct OpHandle {
    uint32_t slot = 0;       // 0 == no completion to wait for
    bool inline_done = false;  // the transfer completed synchronously; see below
    bool valid() const { return slot != 0 || inline_done; }
};

struct TransportDiag {
    uint64_t posted = 0;
    uint64_t retired = 0;
    uint64_t outstanding = 0;
    uint64_t unmatched = 0;
    uint64_t abandoned = 0;   // ops a waiter timed out on and left to the provider
    uint64_t injected = 0;    // sent inline, no completion expected -- see OpHandle
    uint64_t oldest_tag = 0;  // caller tag of the longest-outstanding op
    std::string last_error;
};

class Transport {
public:
    virtual ~Transport() = default;

    // Joins the region to the process-wide window -- created collectively on the first call,
    // shared by every endpoint after it -- and exchanges the geometry hello with the peer.
    // Returns an error string; empty means connected.
    virtual std::string connect(uint8_t* region_base, uint64_t region_bytes) = 0;

    // One-sided write of `bytes` from our region at `local_offset` into the peer's window at
    // `remote_offset`. MPI_Rput: the handle retires on LOCAL completion, so the bytes are not
    // the peer's until flush() has been called.
    //
    // `op` names the posted operation; pass it to wait() to block on THIS transfer. `tag` is
    // carried for diagnostics only (the caller's core index), so a stalled operation can be
    // attributed to a core.
    virtual std::string post(uint64_t local_offset, uint64_t remote_offset, uint64_t bytes, uint64_t tag,
                             OpHandle& op) = 0;

    // REMOTE completion of everything posted on this endpoint since the last flush -- the bytes
    // are in the peer's memory, not merely handed to the local NIC.
    //
    // Pure virtual rather than a defaulted no-op, because the two backends answer it for opposite
    // reasons and a forgotten override is the one bug that does not announce itself. Under
    // libfabric a write completion ALREADY means remote arrival, so the override is empty. Under
    // MPI it does not: MPI_Rput's request retires when the ORIGIN buffer is reusable and says
    // nothing about the target, so the override is MPI_Win_flush and skipping it lets a peer see
    // an armed trigger whose payload is still on the wire.
    //
    // True while this endpoint has one-sided writes that have not been flushed -- so the caller
    // can find the endpoints that owe a flush without tracking every unwaited post itself.
    //
    // EXISTS FOR THE CREDIT PATH. post_credit() and post_word() deliberately wait for nothing, and
    // under MPI an unsynchronised put has no guaranteed completion at ALL: it lands at the next
    // synchronisation call, which on a host that is only receiving never comes, because its
    // sender loop has no payloads of its own to flush. The peer then parks waiting for a credit
    // that is sitting in the local runtime. This is what lets the sender flush those endpoints on
    // its idle path instead. See sender_loop().
    virtual bool needs_flush() const = 0;

    // Costs a round trip, so it is called ONCE PER SENDER LAP over the set of endpoints with
    // payloads waiting -- not once per message. MPI_Win_flush retires every operation outstanding
    // to its target, so one call covers the whole in-flight window; see sender_loop().
    virtual std::string flush() = 0;

    // Arms the RX control word in the PEER's bank for `dest_core`: three operand registers
    // then the trigger, the same operands-before-trigger discipline the T6 uses on its own
    // side, carried by RMA instead of PCIe. Called only after the payload write has been FLUSHED, so a
    // peer that sees the trigger is guaranteed the bytes behind it.
    virtual std::string post_notice(uint32_t dest_core, uint32_t rx_slot, uint64_t length,
                                    uint32_t origin_selector,
                                    uint64_t elapsed_ns, bool reply, uint32_t stage_slot, OpHandle& op,
                                    uint64_t dest_uva) = 0;

    // Blocks until THE operation named by `op` completes, or the timeout expires. The handle
    // is consumed: it is cleared on return and must not be waited on twice.
    virtual Completion wait(OpHandle& op, uint32_t timeout_ms) = 0;

    // Returns true when the operation has retired -- `out` holds the verdict and the handle is
    // consumed, exactly as wait() leaves it. Returns false while it is still outstanding, with
    // the handle UNTOUCHED so the caller can poll it again.
    virtual bool try_wait(OpHandle& op, Completion& out) = 0;

    virtual const PeerInfo& peer() const = 0;
    virtual std::string describe() const = 0;
    virtual TransportDiag diag() const = 0;

    // How many operations may be OUTSTANDING on this endpoint at once, or 0 where there is no
    // such number. Zero means "unknown", not "none", so a caller must treat it as "do not clamp"
    // rather than "clamp to zero".
    //
    // ALWAYS ZERO ON MPI, and that is a real loss rather than a detail: MPI exposes no queue
    // depth and never refuses a post the way a provider returns EAGAIN -- it queues internally
    // instead, so there is no backpressure signal to read. The transport's own request-table
    // size is the only limit left, which makes it a real limit rather than an advisory one.
    virtual uint64_t tx_depth() const { return 0; }

    // Post->completion samples for the payload write. Empty (n == 0) unless
    // TransportConfig::measure_retire was set. Safe to call while the run is in flight.
    virtual RetireStats retire_stats() const { return {}; }

    virtual void set_recording(bool) {}

    // Needed before any cross-host verification: our
    // RX arenas are filled by the PEER, and our own scanner stopping says nothing about
    // whether the peer has finished sending. Without this a two-host run checks arenas
    // the peer is still writing and reports a failure that is really a race.
    virtual std::string barrier() = 0;

    // Writes `count` into the peer's credit register for `core`. One 8-byte RMA, no completion
    // waited: a credit is idempotent and monotonic, so a lost one is corrected by the next, and
    // blocking a delivery on a credit completion would put a round trip in the middle of the
    // receive path.
    //
    // UNWAITED IS NOT UNFLUSHED. Under MPI an unsynchronised put has no guaranteed completion at
    // all, so a host that only receives -- whose sender loop has no payloads of its own to flush
    // -- would leave every credit sitting in the runtime while the peer's sender parks waiting
    // for one. The sender loop flushes endpoints with pending credits on its idle path; that is
    // what makes "no completion waited" safe rather than a deadlock.
    virtual std::string post_credit(uint32_t core, uint32_t my_host, uint64_t count,
                                    uint32_t stage_slot) = 0;

    // The same mechanism post_credit() uses, exposed for any trigger word. The value is staged
    // inside the window rather than passed from the caller's stack -- MPI has no inject, so the
    // source must stay live until local completion -- and no completion is expected.
    //
    virtual std::string post_word(uint64_t remote_offset, uint64_t value) = 0;

    // claim a ticket: one-sided fetch-and-add of `add` at the peer's `remote_offset`, with the
    // value BEFORE the add returned in `out`. This is how a sender takes a receive slot without
    // a handshake and without the receiver holding any per-sender state -- which is the whole
    // reason the receive cost does not grow with the number of hosts.
    //
    // Blocking: the ticket is needed before the payload can be placed, so unlike post() there
    // is nothing useful to overlap with. MPI_Fetch_and_op with MPI_SUM on MPI_UINT64_T, which on
    // RoCE RC is a hardware FetchAdd on 8 bytes.
    //
    // Returns an error string if the runtime cannot do it, so a caller can fall back rather
    // than assume. `atomics_available()` reports the same fact before any traffic, so a run
    // can refuse at startup instead of failing on the first message.
    virtual std::string fetch_add(uint64_t remote_offset, uint64_t add, uint64_t& out) {
        (void)remote_offset; (void)add; (void)out;
        return "this transport does not implement fetch_add";
    }
    virtual bool atomics_available() const { return false; }
};

// Returns null and fills `error` when the MPI window could not be established, so a caller can
// report the reason rather than crashing on a null it did not expect.
std::unique_ptr<Transport> make_transport(const TransportConfig& cfg, std::string& error);

// Peer Table: host id -> the endpoint that reaches it.
enum : uint32_t {
    kPeerOk = 0,
    kPeerNoSuchHost = 1,  // the id is outside the configured topology: a bad address
    kPeerNotConnected = 2,  // inside the topology, no endpoint: a provisioning gap
    kPeerIsSelf = 3,        // our own id; routing should have taken the local arm before here
};

class PeerTable {
public:
    // `num_hosts` and `self` come from HostTopology, so a table can never disagree with the
    // topology the routing decision used.
    void configure(uint32_t num_hosts, uint32_t self) {
        entries_.assign(num_hosts, nullptr);
        self_ = self;
    }

    // Refused rather than overwritten. Two transports claiming one host id means the OOB
    // bootstrap paired something twice, and silently keeping the last one would send half a
    // run's traffic to whichever won.
    std::string connect_peer(uint32_t host_id, Transport* t) {
        if (host_id >= entries_.size()) {
            return "peer table: host " + std::to_string(host_id) + " is outside the configured " +
                   std::to_string(entries_.size()) + " hosts";
        }
        if (host_id == self_) {
            return "peer table: host " + std::to_string(host_id) + " is THIS host; a peer entry for "
                   "ourselves would be reached only by a routing bug";
        }
        if (entries_[host_id] != nullptr) {
            return "peer table: host " + std::to_string(host_id) + " already has a transport";
        }
        entries_[host_id] = t;
        return {};
    }

    Transport* for_host(uint32_t host_id, uint32_t& why) const {
        if (host_id >= entries_.size()) {
            why = kPeerNoSuchHost;
            return nullptr;
        }
        if (host_id == self_) {
            why = kPeerIsSelf;
            return nullptr;
        }
        if (entries_[host_id] == nullptr) {
            why = kPeerNotConnected;
            return nullptr;
        }
        why = kPeerOk;
        return entries_[host_id];
    }

    // Every connected peer, for the aggregate calls -- diag, retire_stats, set_recording,
    // barrier -- which are per-run rather than per-message and must reach all of them.
    std::vector<Transport*> all() const {
        std::vector<Transport*> v;
        for (Transport* t : entries_) {
            if (t != nullptr) {
                v.push_back(t);
            }
        }
        return v;
    }
    uint32_t connected() const { return static_cast<uint32_t>(all().size()); }
    bool empty() const { return connected() == 0; }

private:
    std::vector<Transport*> entries_;
    uint32_t self_ = 0;
};

// One endpoint per peer rank, each registered in `table`. There is no address list, no
// listen/connect asymmetry, and NO IDENTITY ARGUMENT: a rank IS the host id, taken from
// DistributedContext, so the caller cannot tell this a different story than the communicator.
std::string connect_mesh(uint8_t* region_base, uint64_t region_bytes, const TransportConfig& base_cfg,
                         std::vector<std::unique_ptr<Transport>>& owned, PeerTable& table);

inline const char* peer_why_name(uint32_t why) {
    switch (why) {
        case kPeerOk: return "ok";
        case kPeerNoSuchHost: return "no such host (bad address)";
        case kPeerNotConnected: return "no transport connected (provisioning gap)";
        case kPeerIsSelf: return "this host (routing bug: the local arm should have taken it)";
        default: return "?";
    }
}

bool transport_available();

}  // namespace tt::tt_metal::experimental
