// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/sockets/d2h2h2d_socket.hpp>

namespace tt::tt_metal {
class IDevice;
}
namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental {

#if defined(TT_METAL_HOST_BRIDGE)

struct L1Map {
    static constexpr uint32_t kStageSlotBytes = 16;
    static constexpr uint32_t kStageSlots = 5;

    // A doorbell gets its own cache line, so ringing one never disturbs a neighbour's meaning.
    static constexpr uint32_t kDoorbellBytes = 64;

    // The receive SCR. host_deliver.cpp writes it as ONE 8-byte strict-ordered UC store and
    // kernels/test_kernel_pull.cpp reads it back as a uint64_t, so 8 is exact, not a round-up.
    static constexpr uint32_t kDestWordBytes = 8;

    uint32_t l1_size = 0;  // bytes of L1 per core; the guard's upper bound

    uint32_t payload_addr = 0;     // what the sender kernel pushes
    uint32_t stage_addr = 0;       // its register staging slots
    uint32_t signal_addr = 0;      // rdma_signal: bytes arrived for this core
    uint32_t completion_addr = 0;  // rdma_completion: this core's request retired
    uint32_t stop_addr = 0;        // where a DEVICE_PULL receiver kernel is told to exit
    uint32_t dest_word_addr = 0;   // a store's per-message destination, for the pull kernel

    uint32_t deliver_addr = 0;

    // deliver_addr is its own buffer rather than aliased onto payload_addr. Aliased costs one
    // payload of L1 but a core can then send or receive, not both -- its kernel and host
    // delivery would write the same bytes. Separate costs two, and is what the ring needs.
    bool bidirectional = false;

    static L1Map compute(uint32_t l1_base, uint32_t l1_size, uint32_t payload_bytes, bool bidirectional = false);

    // How many payload-sized buffers this layout claims. The only place the 1-vs-2 lives.
    uint32_t payload_copies() const { return bidirectional ? 2u : 1u; }

    // One past the last byte compute() claimed. The control words are the top of the aliased
    // layout; when delivery has its own buffer that buffer sits above them and is the top.
    uint32_t end() const { return bidirectional ? deliver_end : (dest_word_addr + kDestWordBytes); }

    // Everything compute() stacks ABOVE the payload: the staging slots, the three doorbell
    // lines, the dest word. Deliberately not end() - stage_addr: it must stay independent of
    // payload_bytes, and under bidirectional end() includes a payload.
    uint32_t control_bytes() const { return dest_word_addr + kDestWordBytes - stage_addr; }

    // end() of the delivery buffer, set by compute() only when bidirectional. No trailing
    // underscore: every other member of this struct is public and unadorned, and in this tree
    // the underscore means private.
    uint32_t deliver_end = 0;

    std::string fits(uint32_t payload_bytes) const;

    L1Layout l1_layout() const {
        return L1Layout{deliver_addr, signal_addr, completion_addr, stop_addr, dest_word_addr};
    }

    D2H2H2DSocket::StoreGuard store_guard() const {
        return D2H2H2DSocket::StoreGuard{payload_addr,      l1_size,  signal_addr,
                                         completion_addr,  stop_addr, /*ring_bytes=*/0u,
                                         dest_word_addr};
    }

    uint32_t store_dest(uint32_t dest_offset, bool is_store) const {
        return is_store ? (payload_addr + dest_offset) : 0u;
    }

    std::string describe() const;
};

struct D2DSocketConfig {
    // --- identity and topology ---
    uint32_t host_ident = 0;
    uint32_t host_num = 0;
    uint32_t chips_per_host = 1;
    uint32_t chip = 0;

    // --- geometry ---
    uint32_t cores = 0;
    uint32_t grid_width = 0;
    uint32_t grid_height = 0;

    // --- the payload ---
    //
    // one size for the socket's life -- the H2D ring's requirement. The T6 signals each message's
    // length at run time and the host uses it, but the aliased ring is kNumAliasRingSlots * this
    // and write_payload() refuses anything that is not an exact divisor of fifo_size.
    uint32_t payload_bytes = 0;

    // --- knobs, forwarded to SocketConfig ---
    uint32_t workers = 0;
    uint32_t send_window = 0;
    bool send_blocking = false;
    bool pin = true;

    // Give delivery its own L1 buffer instead of aliasing it onto the send buffer, so a core
    // can hold an outbound and an inbound payload at once. Costs a second payload of L1,
    // halving the largest payload that fits. See L1Map::bidirectional.
    bool bidirectional = false;
};

// what A measurement needs and A data mover does not. None of it changes where a byte goes:
// a socket built without it moves traffic and records nothing, and never runs a clock sync.
// measure() is the opt-in, which keeps socket creation independent of any timing prerequisite.
struct D2DMeasurementConfig {
    // Iterations per core discarded before the counters start. 0 records from the first
    // message. Multiplied by `cores` to get the message count the gate actually counts.
    uint32_t warmup = 0;

    // Sample the credit round trip: h2h:credit-raw (post -> credit visible), h2h:net (that
    // minus the peer's own reported turnaround) and h2h:payload-at-peer. The credit is this
    // socket's analogue of tt-fabric's ping-pong reply, so it is the only shape yielding a
    // latency comparable to theirs -- and only at send_window 1, where one message is alone in
    // flight. Costs a credit_total() read per in-flight message per lap on the sender thread,
    // so a run carrying it does not also produce a quotable bandwidth.
    bool measure_credit = false;

    // Time each payload write from post to local completion, reported as diag:h2h-retire.
    // Costs work on the sender thread, which is this path's measured bottleneck, so a
    // bandwidth run must not pay for a number it does not print.
    bool measure_retire = false;

    // Device cycles -> ns. 0 measures it, which takes ~50 ms and touches the device.
    double ns_per_cycle_override = 0.0;

    // One box, one hardware clock: the offset is 0 by construction and estimating it would
    // substitute noise for a known value. Only ever consumed by the clock sync.
    bool same_host = false;
};

class D2DSocket {
public:
    static std::unique_ptr<D2DSocket> create(
        std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device, tt::tt_metal::IDevice* device,
        const D2DSocketConfig& cfg, std::string& err);

    ~D2DSocket();

    D2DSocket(const D2DSocket&) = delete;
    D2DSocket& operator=(const D2DSocket&) = delete;

    const L1Map& l1() const { return l1_; }
    // Both are only populated once measure() has run; before that the clock is invalid and
    // ns_per_cycle is 0, which is what a cycles-flagged sample treats as "no scale".
    const ClockSync& clock() const { return clock_; }
    double ns_per_cycle() const { return ns_per_cycle_; }

    // opt IN TO measurement. Before open(), because the scan pool and sender read these after.
    // collective -- it runs the clock sync, so every rank must call it at the same point.
    // Returns an error string; empty means measuring.
    std::string measure(const D2DMeasurementConfig& m);
    bool measuring() const { return measuring_; }
    const std::string& clock_rate_detail() const { return clock_rate_detail_; }
    HostRegion& region() const;
    Transport* primary_transport() const { return primary_.get(); }
    Deliverer* deliverer() const { return deliverer_.get(); }
    uint32_t peer_count() const { return static_cast<uint32_t>(1 + mesh_peers_.size()); }
    std::string deliverer_describe() const;
    std::string transport_describe() const;

    // No kernel compile-arg builders here: a kernel's argument order belongs to whoever owns
    // the kernel. Callers assemble their own from l1() and region().device().

    // --- pass-throughs ----------------------------------------------------
    bool open(std::string& err);
    void stop();
    const SocketCounters& counters() const;
    RunStats collect() const;
    std::vector<Transport*> peers_for_barrier() const;
    void set_recording(bool on);
    void open_recording_gate();
    void stamp_timed_end();
    uint64_t timed_start_ns() const;
    bool transport_failed() const;
    bool peer_refused() const;
    std::string first_error() const;
    std::string stall_dump(const char* where) const;
    uint64_t store_faults() const;

private:
    D2DSocket() = default;

    D2DSocketConfig cfg_{};
    D2DMeasurementConfig measure_cfg_{};
    bool measuring_ = false;
    bool opened_ = false;
    L1Map l1_{};

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;

    std::unique_ptr<Deliverer> deliverer_;
    double ns_per_cycle_ = 0.0;
    std::string clock_rate_detail_;
    HostRegion* region_ = nullptr;

    std::unique_ptr<Transport> primary_;
    std::vector<std::unique_ptr<Transport>> mesh_peers_;

    ClockSync clock_{};  // 5

    std::unique_ptr<D2H2H2DSocket> inner_;
};

#endif  // TT_METAL_HOST_BRIDGE

}  // namespace tt::tt_metal::experimental
