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
// will NOT give is remote completion from the operation's own handle:
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
    // commented out b/c the wedge markers that fed it are commented out -- see staged_word()
    // in host_transport.cpp. Uncomment this, the stores there, and the diag() line together.
    // Which step of a credit write is in flight, 0 when none. Nonzero in a stall dump means a
    // credit is wedged: 1 entered, 2 holds credit_m_, 3 past MPI_Put, 4 past flush_local.
    // uint32_t word_phase = 0;
    std::string last_error;
};

class Transport {
public:
    virtual ~Transport() = default;

    virtual std::string connect(uint8_t* region_base, uint64_t region_bytes) = 0;

    virtual std::string post(uint64_t local_offset, uint64_t remote_offset, uint64_t bytes, uint64_t tag,
                             OpHandle& op) = 0;

    // MPI Progress. An idle sender makes no MPI calls at all, and one-sided operations
    // still need the TARGET side to progress before they complete -- so a peer's credit
    // MPI_Put blocks until this side happens to call into MPI. Cheap, and safe to call from
    // any thread on a transport that permits it.
    //
    // Empty by intent to conform to an inherited interface.
    //
    virtual std::string progress() { return {}; }

    virtual bool needs_flush() const = 0;

    virtual std::string flush() = 0;

    virtual std::string post_notice(uint32_t dest_core, uint32_t rx_slot, uint64_t length,
                                    uint32_t origin_selector,
                                    uint64_t elapsed_ns, bool reply, uint32_t stage_slot, OpHandle& op,
                                    uint64_t dest_uva) = 0;

    virtual Completion wait(OpHandle& op, uint32_t timeout_ms) = 0;

    virtual bool try_wait(OpHandle& op, Completion& out) = 0;

    virtual const PeerInfo& peer() const = 0;
    virtual std::string describe() const = 0;
    virtual TransportDiag diag() const = 0;

    virtual uint64_t tx_depth() const { return 0; }

    virtual RetireStats retire_stats() const { return {}; }

    virtual void set_recording(bool) {}

    virtual std::string barrier() = 0;

    virtual std::string post_credit(uint32_t core, uint32_t my_host, uint64_t count,
                                    uint64_t turnaround_ns, uint32_t stage_slot) = 0;

    virtual std::string post_word(uint64_t remote_offset, uint64_t value) = 0;

    virtual std::string fetch_add(uint64_t remote_offset, uint64_t add, uint64_t& out) {
        (void)remote_offset; (void)add; (void)out;
        return "this transport does not implement fetch_add";
    }
    virtual bool atomics_available() const { return false; }
};

std::unique_ptr<Transport> make_transport(const TransportConfig& cfg, std::string& error);

enum : uint32_t {
    kPeerOk = 0,
    kPeerNoSuchHost = 1,  // the id is outside the configured topology: a bad address
    kPeerNotConnected = 2,  // inside the topology, no endpoint: a provisioning gap
    kPeerIsSelf = 3,        // our own id; routing should have taken the local arm before here
};

class PeerTable {
public:
    void configure(uint32_t num_hosts, uint32_t self) {
        entries_.assign(num_hosts, nullptr);
        self_ = self;
    }

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
