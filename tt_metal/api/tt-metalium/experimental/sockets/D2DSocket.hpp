// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/experimental/sockets/D2H2H2DSocket.hpp>

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

    static L1Map compute(uint32_t l1_base, uint32_t l1_size, uint32_t payload_bytes);

    // One past the last byte compute() claimed. dest_word_addr is the highest field, so this
    // is the number the L1 bound has to be taken against -- not payload_addr + payload_bytes,
    // which leaves control_bytes() of tail unchecked.
    uint32_t end() const { return dest_word_addr + kDestWordBytes; }

    // Everything compute() stacks ABOVE the payload: the staging slots, the three doorbell
    // lines, the dest word. Measured from stage_addr, so it is independent of payload_bytes,
    // and a field added anywhere below end() is accounted for without editing a tally.
    uint32_t control_bytes() const { return end() - stage_addr; }

    std::string fits(uint32_t payload_bytes) const;

    L1Layout l1_layout() const {
        return L1Layout{deliver_addr, signal_addr, completion_addr, stop_addr, dest_word_addr};
    }

    D2H2H2DSocket::StoreGuard store_guard() const {
        return D2H2H2DSocket::StoreGuard{payload_addr, l1_size, signal_addr, completion_addr, stop_addr};
    }

    uint32_t store_dest(uint32_t dest_offset, bool is_store) const {
        return is_store ? (payload_addr + dest_offset) : 0u;
    }
    // commented out b/c deadcode -- never called, and byte-identical to store_dest() above
    // uint32_t verify_at(uint32_t dest_offset, bool is_store) const {
    //     return is_store ? (payload_addr + dest_offset) : 0u;
    // }

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
    uint32_t payload_bytes = 0;

    // --- the run, for the warmup gate and the ladder ---
    uint32_t iters = 0;
    uint32_t warmup = 0;

    // --- knobs, forwarded to SocketConfig ---
    uint32_t workers = 0;
    uint32_t send_window = 0;
    bool send_blocking = false;
    bool pin = true;

    double ns_per_cycle_override = 0.0;

    bool h2d_socket = true;

    bool measure_retire = false;
    bool same_host = false;

    bool ladder_enabled = false;
    bool ladder_quiesce = false;
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
    const ClockSync& clock() const { return clock_; }
    double ns_per_cycle() const { return ns_per_cycle_; }
    const std::string& clock_rate_detail() const { return clock_rate_detail_; }
    const VolumeLadder& ladder() const { return ladder_; }
    HostRegion& region() const;
    // commented out b/c deadcode
    // D2H2H2DSocket& inner() const { return *inner_; }
    Transport* primary_transport() const { return primary_.get(); }
    Deliverer* deliverer() const { return deliverer_.get(); }
    uint32_t peer_count() const { return static_cast<uint32_t>(1 + mesh_peers_.size()); }
    std::string deliverer_describe() const;
    std::string transport_describe() const;

    std::vector<uint32_t> sender_compile_args(
        uint32_t iterations, uint32_t opcode, uint32_t flags, bool await_completion) const;
    std::vector<uint32_t> receiver_compile_args() const;

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
    std::string first_error() const;
    std::string stall_dump(const char* where) const;
    uint64_t store_faults() const;

private:
    D2DSocket() = default;

    D2DSocketConfig cfg_{};
    L1Map l1_{};

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;

    std::unique_ptr<Deliverer> deliverer_;
    double ns_per_cycle_ = 0.0;
    std::string clock_rate_detail_;
    HostRegion* region_ = nullptr;

    std::unique_ptr<Transport> primary_;
    std::vector<std::unique_ptr<Transport>> mesh_peers_;

    ClockSync clock_{};  // 5

    VolumeLadder ladder_{};
    LadderSync ladder_sync_{};

    std::unique_ptr<D2H2H2DSocket> inner_;
};

#endif  // TT_METAL_HOST_BRIDGE

}  // namespace tt::tt_metal::experimental
