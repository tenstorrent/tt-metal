// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The host-to-host leg: one interface, two libfabric providers.
//
//   single host, two processes:  tcp (loopback)
//   two hosts,   two processes:  tcp, verbs
//
// fi_mr_reg() is called exactly once, over the same span PinnedMemory pinned. Both are
// independent, refcounted pins of the same pages -- one for the TT device, one for the
// NIC -- so a payload a Tensix core pushed over PCIe is already inside a registered MR
// when the transport goes to send it. No bounce buffer, no per-message registration, and
// the send is described by an OFFSET into the region rather than by a pointer.
//
// A one-sided fi_write into
// the peer's RX arena is the operation this design actually wants, and it is what verbs
// gives.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tt::tt_metal::experimental {

enum class Provider : uint32_t { Tcp, Verbs };

const char* provider_name(Provider p);      // "tcp", "verbs"
const char* provider_label(Provider p);     // short tag for CSV rows
bool provider_from_string(const std::string& s, Provider& out);

struct TransportConfig {
    Provider provider = Provider::Tcp;
    bool is_server = false;         // the side that binds and waits
    std::string peer_host;          // client: where the server is
    uint16_t oob_port = 18515;      // out-of-band bootstrap port
    std::string bind_addr;          // server: routable local address for the fabric
                                    // endpoint. Empty => this host's own hostname.
    bool prefer_rma = true;
    uint32_t timeout_ms = 30000;

    // This host's identity and geometry, sent in the bootstrap hello so the peer can
    // check we agree before a byte moves. Supplied by the caller rather than read from
    // the region here, so the transport stays usable without one (the loopback and
    // no-device modes exercise it that way).
    uint32_t host_id = 0;
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

// What the two sides must tell each other before a byte moves. Exchanged over a plain
// TCP socket rather than over the fabric itself, for the obvious reason: the fabric is
// not usable until this has happened. The same socket then carries the clock sync and
// the end-of-run verdict, so there is exactly one bootstrap channel to get right.
struct PeerInfo {
    uint64_t region_base = 0;   // the peer's region VA -- for RMA target addressing
    uint64_t region_bytes = 0;
    uint64_t mr_key = 0;
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
// The handle IS the libfabric context. Providers that report FI_CONTEXT in their mode --
// verbs is one -- require the caller to hand over a context object the provider owns for the
// lifetime of the operation. The old code passed a core index cast to a pointer and, for the
// notice, a plain nullptr; under FI_CONTEXT that is the provider writing its state through a
// pointer we invented.
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

    // Registers the region and completes the bootstrap handshake with the peer.
    // Returns an error string; empty means connected.
    virtual std::string connect(uint8_t* region_base, uint64_t region_bytes) = 0;

    // One-sided write of `bytes` from our region at `local_offset` into the peer's
    // region at `remote_offset`. Falls back to a message send when the provider has no
    // RMA; rma_in_use() says which happened.
    //
    // `op` names the posted operation; pass it to wait() to block on THIS transfer. `tag` is
    // carried for diagnostics only (the caller's core index), so a stalled operation can be
    // attributed to a core.
    virtual std::string post(uint64_t local_offset, uint64_t remote_offset, uint64_t bytes, uint64_t tag,
                             OpHandle& op) = 0;

    // Arms the RX control word in the PEER's bank for `dest_core`: three operand registers
    // then the trigger, the same operands-before-trigger discipline the T6 uses on its own
    // side, carried by RMA instead of PCIe. Called only after the payload write has
    // COMPLETED, so a peer that sees the trigger is guaranteed the bytes behind it.
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

    // Receive side for the message fallback; a no-op under RMA, where the bytes land
    // without the target's involvement and arrival is signalled out of band.
    virtual std::string post_recv(uint64_t local_offset, uint64_t bytes) = 0;

    virtual bool rma_in_use() const = 0;
    virtual const PeerInfo& peer() const = 0;
    virtual std::string describe() const = 0;
    virtual TransportDiag diag() const = 0;

    // provider's TX QUEUE DEPTH (`tx_attr->size`), or 0 where there is no such number.
    //
    // How many operations may be OUTSTANDING on this endpoint at once. Read by the sender's
    // Zero means "unknown", not "none", so a caller must treat it as "do not clamp" rather
    // than "clamp to zero" -- a local transport has no queue to exhaust.
    virtual uint64_t tx_depth() const { return 0; }

    // Post->completion samples for the payload write. Empty (n == 0) unless
    // TransportConfig::measure_retire was set. Safe to call while the run is in flight.
    virtual RetireStats retire_stats() const { return {}; }

    virtual void set_recording(bool) {}

    // The bootstrap socket, reused for clock sync and the final verdict exchange.
    virtual int oob_fd() const = 0;

    // Rendezvous on the bootstrap socket. Needed before any cross-host verification: our
    // RX arenas are filled by the PEER, and our own scanner stopping says nothing about
    // whether the peer has finished sending. Without this a two-host run checks arenas
    // the peer is still writing and reports a failure that is really a race.
    virtual std::string barrier(uint32_t timeout_ms = 60000) = 0;

    // Writes `count` into the peer's credit register for `core`. One 8-byte RMA, no
    // completion waited: a credit is idempotent and monotonic, so a lost one is corrected
    // by the next, and blocking a delivery on a credit completion would put a round trip
    // in the middle of the receive path.
    virtual std::string post_credit(uint32_t core, uint32_t my_host, uint64_t count,
                                    uint32_t stage_slot) = 0;

    // The same mechanism post_credit() uses, exposed for any trigger word: the data is copied
    // inline so the source is free on return, and no completion is expected.
    //
    virtual std::string post_word(uint64_t remote_offset, uint64_t value) = 0;

    // claim a ticket: one-sided fetch-and-add of `add` at the peer's `remote_offset`, with the
    // value BEFORE the add returned in `out`. This is how a sender takes a receive slot without
    // a handshake and without the receiver holding any per-sender state -- which is the whole
    // reason the receive cost does not grow with the number of hosts.
    //
    // Blocking: the ticket is needed before the payload can be placed, so unlike post() there
    // is nothing useful to overlap with. On RoCE RC this is a hardware FetchAdd on 8 bytes.
    //
    // Returns an error string if the provider cannot do it, so a caller can fall back rather
    // than assume. `atomics_available()` reports the same fact before any traffic, so a run
    // can refuse at startup instead of failing on the first message.
    virtual std::string fetch_add(uint64_t remote_offset, uint64_t add, uint64_t& out) {
        (void)remote_offset; (void)add; (void)out;
        return "this transport does not implement fetch_add";
    }
    virtual bool atomics_available() const { return false; }
};

// Returns null and fills `error` when libfabric is not compiled in, so a caller can
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

inline uint16_t pair_port(uint16_t base, uint32_t a, uint32_t b, uint32_t num) {
    const uint32_t lo = a < b ? a : b;
    const uint32_t hi = a < b ? b : a;
    return static_cast<uint16_t>(base + lo * num + hi);
}

inline std::vector<std::string> split_csv(const std::string& in) {
    std::vector<std::string> out;
    std::string cur;
    for (const char c : in) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

std::string connect_mesh(uint32_t num_hosts, uint32_t self, const std::string& addr_csv, uint16_t base_port,
                         const TransportConfig& base_cfg, uint8_t* region_base, uint64_t region_bytes,
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
