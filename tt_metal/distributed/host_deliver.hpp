// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// The H2D leg: getting bytes out of a host RX arena and into a Tensix core's L1.
//
//   payload   noc_write()            Cluster::write_core, WC TLB window, Relaxed ordering.
//                                    CHUNKED AT 32 KiB -- see kMaxHostWrite in the .cpp
//
//   doorbell  noc_write_immediate()  Cluster::write_core_immediate, UC window, STRICT
//                                    ordering.
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tt::tt_metal {
class IDevice;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

// Where a delivered message lands in the destination core's L1. These are the same
// addresses the kernel was compiled with, so the host and the kernel cannot disagree about
// them -- they come from one place and are passed to both.
//
//   rdma_completion  "the request YOU issued retired"      -- rung by the LOCAL host after
//                    it services this core's TX control word. Purely local.
//
//   rdma_signal      "bytes SOMEBODY ELSE sent landed"     -- rung on delivery into L1.
//
// One counter cannot say which event it was. A kernel pacing itself on the delivery
// counter can only issue its next request once a REMOTE peer has sent it something, which
// -- combined with the sender-side credit -- makes two interlocking depth-1 ladders that
// deadlock at the tail.
//
struct L1Layout {
    uint32_t payload_addr = 0;     // where delivered bytes are written
    uint32_t signal_addr = 0;      // rdma_signal: incremented on delivery into this core
    uint32_t completion_addr = 0;  // rdma_completion: incremented when this core's request retires
    // Where the host tells a DEVICE_PULL receiver kernel to exit. Unused by the push path,
    // which has no kernel of its own to stop.
    uint32_t stop_addr = 0;
    // payload_addr above is one address for the whole run, which is all kOpSendUva ever
    // needs. A UVA store names its own destination per message, and on the DEVICE_PULL path
    // the thing doing the writing is the receiver KERNEL -- whose destination is a compile
    // arg. So the host drops the address here, on the strict-ordered UC path, immediately
    // before it advertises the message; the kernel reads it and writes there instead.
    //
    uint32_t dest_word_addr = 0;
};

class Deliverer {
public:
    virtual ~Deliverer() = default;

    // Writes `bytes` into `core`'s L1 and then rings its doorbell. Returns an error
    // string; empty on success. The two halves are timed separately by the caller, which
    // is why they are also exposed separately below.
    //
    // `dst_l1` IS THE EFFECTIVE ADDRESS, and 0 means "the layout's fixed payload_addr".
    //
    virtual std::string write_payload(uint32_t core, const uint8_t* src, uint32_t bytes, uint32_t dst_l1) = 0;
    virtual std::string ring_doorbell(uint32_t core, uint32_t value) = 0;   // rdma_signal
    virtual std::string ring_completion(uint32_t core, uint32_t value) = 0;  // rdma_completion

    // Reads a core's doorbell back. Used by the round-trip mode to see the far core
    // acknowledge, and to verify delivery happened at all.
    virtual uint32_t read_doorbell(uint32_t core) = 0;

    // Returns an error string; empty on success, non-empty on timeout.
    virtual std::string wait_delivered(uint32_t core, uint32_t expected) {
        (void)core;
        (void)expected;
        return {};
    }

    virtual std::vector<uint8_t> read_payload(uint32_t core, uint32_t bytes, uint32_t src_l1) = 0;

    // Raw 32-bit register read on a core. Used for exactly one thing: sampling the Tensix
    // wall clock to measure a cycles->ns RATE. Not a data path.
    virtual uint32_t read_reg32(uint32_t core, uint64_t addr) = 0;

    // The per-core socket config-buffer addresses, indexed by core, or empty for a
    // deliverer that has no sockets. A DEVICE_PULL receiver kernel needs its OWN core's
    // address as a runtime arg -- each socket's config buffer is a separate allocation, so
    // they differ per core and a shared compile arg would aim every core at one ring.
    virtual std::vector<uint32_t> socket_config_addresses() const { return {}; }

    // Clears the stop word on every core, so a receiver kernel launched next will actually
    // run. A no-op on the push path.
    virtual std::string arm_receivers() { return {}; }

    // Tells every receiver kernel to exit. A no-op on the push path. MUST be called before
    // waiting on the workload: the receiver loops until this word is set, so a run that
    // finishes its messages and then waits for the program would wait forever.
    virtual std::string stop_receivers() { return {}; }

    virtual std::string describe() const = 0;
};

std::unique_ptr<Deliverer> make_device_deliverer(
    tt::tt_metal::IDevice* device, uint32_t grid_width, uint32_t cores, L1Layout layout, std::string& error);

struct H2DSocketConfig {
    uint32_t fifo_size = 1u << 20;  // per core. reserve_bytes blocks once this fills.
    uint32_t page_size = 0;         // 0 = use the PCIe alignment, i.e. finest legal grain.
    bool device_pull = true;        // false = HOST_PUSH; see the L1 warning above.
    bool socket_is_the_doorbell = true;
    // RING ALIASING. each core's socket ring is mapped MAP_FIXED over
    // rx_arena_offset(core) within this region, so the peer's RMA -- which already targets
    // exactly that offset (host_socket.cpp:664) -- lands directly in the ring and the
    // RX-arena memcpy stops existing. The peer needs no change: same offsets, same MR, same
    // key. Requires fifo_size == the payload size; see write_payload().
    //
    // MUST BE SET BEFORE THE REGION IS PINNED AND REGISTERED. MAP_FIXED replaces the physical
    // pages behind these addresses, and an MR registered first would go on naming the old
    // ones -- the NIC then DMAs into pages that have been unmapped, with nothing reporting
    // it. The deliverer therefore has to be constructed ahead of Transport::connect().
    uint8_t* alias_region_base = nullptr;
};

// `mesh_device` rather than IDevice*: H2DSocket's constructor takes the mesh, and the
// sockets must outlive it, so the deliverer holds a share of it.
std::unique_ptr<Deliverer> make_h2d_socket_deliverer(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device, uint32_t grid_width, uint32_t cores,
    L1Layout layout, H2DSocketConfig cfg, std::string& error);

// The Tensix wall clock, as the host addresses it. Blackhole:
// RISCV_DEBUG_REGS_START_ADDR (0xFFB12000) | 0x1F0. Duplicated here rather than included
// because tensix.h is a device-side header; the value is pinned by the static_assert in
// the kernel's own use of the macro, and a mismatch shows up as a nonsensical clock rate
// at startup rather than as a wrong latency later.
constexpr uint64_t kWallClockLo = 0xFFB12000ull | 0x1F0ull;
constexpr uint64_t kWallClockHi = 0xFFB12000ull | 0x1F8ull;

// Measures cycles per second by sampling the wall clock around a known host interval.
// A RATE, not an epoch -- which is all the differential timing design needs, and is why
// there is no Cristian handshake here. Returns 0 if the sample looks implausible.
double measure_ns_per_cycle(Deliverer& deliverer, uint32_t core, uint32_t sample_ms, std::string& detail);

}  // namespace tt::tt_metal::experimental
