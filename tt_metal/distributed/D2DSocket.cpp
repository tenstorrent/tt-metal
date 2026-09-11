// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/D2DSocket.hpp>

#include <algorithm>
#include <sstream>
#include <thread>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace tt::tt_metal::experimental {

#if defined(TT_METAL_HOST_BRIDGE)

namespace {

constexpr uint32_t align64(uint32_t v) { return (v + 0x3Fu) & ~0x3Fu; }

}  // namespace

// ===========================================================================
// L1Map
// ===========================================================================

L1Map L1Map::compute(uint32_t l1_base, uint32_t l1_size, uint32_t payload_bytes) {
    L1Map m;
    m.l1_size = l1_size;
    m.payload_addr = align64(l1_base);
    m.stage_addr = m.payload_addr + align64(payload_bytes);
    m.signal_addr = m.stage_addr + kStageSlots * kStageSlotBytes;
    m.completion_addr = m.signal_addr + kDoorbellBytes;
    m.stop_addr = m.completion_addr + kDoorbellBytes;
    m.dest_word_addr = m.stop_addr + kDoorbellBytes;
    m.deliver_addr = m.payload_addr;
    return m;
}

std::string L1Map::fits(uint32_t payload_bytes) const {
    // compute() stacks the control words ABOVE the aligned payload, so bounding the payload
    // span alone leaves control_bytes() of tail unchecked -- and every word up there gets
    // written, by the sender kernel, the pull kernel, or host delivery. Bound the map's top.
    if (end() > l1_size || deliver_addr + payload_bytes > l1_size) {
        // The alignment pad this payload carries. Defensive: the map was built from
        // payload_bytes, so the two agree at the one call site, but a mismatched call would
        // otherwise wrap a uint32 here.
        const uint32_t aligned = stage_addr - payload_addr;  // align64(payload_bytes), as built
        const uint32_t pad = aligned > payload_bytes ? aligned - payload_bytes : 0u;
        const uint32_t overhead = control_bytes() + pad;
        // The largest payload that WOULD fit is 64-aligned and so carries no pad of its own,
        // which makes control_bytes() alone the floor -- min_l1 is what a zero-byte payload
        // already costs. Guarded: a high allocator base can put min_l1 above l1_size, and
        // unguarded this wrapped and advertised a ~4 GB ceiling.
        const uint32_t min_l1 = payload_addr + control_bytes();
        const uint32_t ceiling = l1_size > min_l1 ? ((l1_size - min_l1) & ~0x3Fu) : 0u;
        std::ostringstream o;
        o << "payload " << payload_bytes << " B does not fit L1 on this core.\n"
          << "  L1 per core        " << l1_size << " B\n"
          << "  allocator base     " << payload_addr << " B\n"
          << "  needed             " << payload_bytes << " B (one shared buffer) + " << overhead
          << " B of control words\n"
          << "  largest payload    " << ceiling << " B\n"
          << "The 1.5 MiB arena is the HOST-side buffer; L1 has to hold what the device holds.";
        return o.str();
    }
    return {};
}

std::string L1Map::describe() const {
    std::ostringstream o;
    o << "payload 0x" << std::hex << payload_addr << " stage 0x" << stage_addr << " signal 0x" << signal_addr
      << " completion 0x" << completion_addr << " stop 0x" << stop_addr << " dest_word 0x" << dest_word_addr
      << " deliver 0x" << deliver_addr << std::dec << " (L1 " << l1_size << " B)";
    return o.str();
}

// ===========================================================================
// D2DSocket -- bringup
// ===========================================================================

std::unique_ptr<D2DSocket> D2DSocket::create(
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device, tt::tt_metal::IDevice* device,
    const D2DSocketConfig& cfg, std::string& err) {
    namespace mh = tt::tt_metal::distributed::multihost;

    if (mesh_device == nullptr || device == nullptr) {
        err = "D2DSocket::create: a mesh device and a device are both required";
        return nullptr;
    }
    if (cfg.cores == 0) {
        err = "D2DSocket::create: cores is 0";
        return nullptr;
    }
    if (cfg.cores > cfg.grid_width * cfg.grid_height) {
        err = "D2DSocket::create: cores " + std::to_string(cfg.cores) + " exceeds the " +
              std::to_string(cfg.grid_width * cfg.grid_height) + " cores on a " + std::to_string(cfg.grid_width) +
              "x" + std::to_string(cfg.grid_height) + " grid";
        return nullptr;
    }
    if (cfg.host_num < 2) {
        err = "D2DSocket::create: the path is chip->host->host->chip and needs two hosts; host_num is " +
              std::to_string(cfg.host_num);
        return nullptr;
    }

    std::unique_ptr<D2DSocket> s(new D2DSocket());
    s->cfg_ = cfg;
    s->mesh_device_ = std::move(mesh_device);

    // ---- 1. the L1 map ---------------------------------------------------
    const uint32_t l1_base =
        static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    const uint32_t l1_size = static_cast<uint32_t>(device->l1_size_per_core());
    s->l1_ = L1Map::compute(l1_base, l1_size, cfg.payload_bytes);
    if (const std::string e = s->l1_.fits(cfg.payload_bytes); !e.empty()) {
        err = e;
        return nullptr;
    }

    const L1Layout layout = s->l1_.l1_layout();
    std::string derr;
    // Hoisted out of the branch below so the store guard can be given the ring it must bound
    // against. 0 on the direct-write path, which has no ring.
    uint32_t h2d_ring_bytes = 0;
    if (cfg.h2d_socket) {
        H2DSocketConfig scfg;
        scfg.page_size = cfg.payload_bytes;
        scfg.fifo_size = cfg.payload_bytes;
        scfg.alias_region_base = HostRegion::reserved_base();
        h2d_ring_bytes = scfg.fifo_size;
        s->deliverer_ = make_h2d_socket_deliverer(s->mesh_device_, cfg.grid_width, cfg.cores, layout, scfg, derr);
    } else {
        s->deliverer_ = make_device_deliverer(device, cfg.grid_width, cfg.cores, layout, derr);
    }
    if (!s->deliverer_) {
        err = "H2D delivery unavailable: " + derr;
        return nullptr;
    }

    if (cfg.ns_per_cycle_override > 0.0) {
        s->ns_per_cycle_ = cfg.ns_per_cycle_override;
        s->clock_rate_detail_ = "supplied by the caller";
    } else {
        s->ns_per_cycle_ = measure_ns_per_cycle(*s->deliverer_, /*core=*/0, /*sample_ms=*/50,
                                                s->clock_rate_detail_);
    }

    // provision() REPORTS BY EXCEPTION; create() reports through `err`. A caller should not
    // have to handle both, so the throw is converted here. It can fire on a first call --
    // validate_shape() rejects a bad core count or grid, and the RLIMIT_MEMLOCK check rejects
    // a pin this process cannot afford -- and both used to escape create() as an exception
    // while every other failure returned nullptr.
    //
    // NOT AN ATTACH PATH, DELIBERATELY. The region is process-global by construction
    // (host_region.cpp:32: the arena offsets stay compile-time constants because the Tensix
    // kernel and the peer host compute the same ones), so provision() throws on a second call
    // and names attached() as the alternative. Calling it here would be wrong: a second socket
    // in this process cannot have its rings aliased -- map_rings() refuses once the region is
    // pinned (host_deliver.cpp:623) and that refusal is FATAL by a 2026-08-27 correction,
    // because an unaliased run would be filed as an aliased one. So the second socket is
    // already refused above, at the deliverer, with a message that says why. Attaching here
    // would route around that and hand back a socket that is quietly not the one asked for.
    // One D2DSocket per process; recovery from a failed bring-up is a new process.
    HostRegion* region_p = nullptr;
    try {
        region_p = &HostRegion::provision(
            s->mesh_device_, cfg.chip, cfg.cores, HostTopology{cfg.host_ident, cfg.host_num, cfg.chips_per_host},
            HostRegion::Grid{cfg.grid_width, cfg.grid_height});
    } catch (const std::exception& ex) {
        err = std::string("host region unavailable: ") + ex.what();
        return nullptr;
    }
    HostRegion& region = *region_p;
    if (const std::string e = region.verify_header(); !e.empty()) {
        err = "region header check failed: " + e;
        return nullptr;
    }
    if (region.base() != HostRegion::reserved_base()) {
        err = "the region was provisioned at an address other than the reserved base; the deliverer's ring "
              "overlay was built against the wrong one";
        return nullptr;
    }
    s->region_ = &region;

    TransportConfig tc;
    tc.chips_per_host = cfg.chips_per_host;
    tc.grid_width = cfg.grid_width;
    tc.cores_in_use = cfg.cores;
    tc.measure_retire = cfg.measure_retire;

    std::vector<std::unique_ptr<Transport>> owned;
    PeerTable bringup_table;
    if (const std::string e = connect_mesh(region.base(), region.pinned_bytes(), tc, owned, bringup_table);
        !e.empty()) {
        err = "mesh bringup failed: " + e;
        return nullptr;
    }
    if (owned.empty()) {
        err = "no peers: this is a one-rank job and the path is chip->host->host->chip";
        return nullptr;
    }
    s->primary_ = std::move(owned.front());
    owned.erase(owned.begin());
    s->mesh_peers_ = std::move(owned);

    {
        const auto& ctx = mh::DistributedContext::get_current_world();
        const uint32_t self = static_cast<uint32_t>(*ctx->rank());
        const uint32_t peer = (self == 0) ? 1u : 0u;
        s->clock_ = sync_clocks(ctx, mh::Rank{static_cast<int>(peer)}, /*initiator=*/self < peer, cfg.same_host);
        if (!s->clock_.valid) {
            err = "refusing to report cross-host hop timings without a clock offset: " + s->clock_.error;
            return nullptr;
        }
    }

    if (cfg.ladder_enabled) {
        const uint32_t ladder_workers =
            cfg.workers != 0 ? cfg.workers : std::max(1u, std::thread::hardware_concurrency());
        const uint64_t recorded =
            static_cast<uint64_t>(cfg.iters - cfg.warmup) * cfg.cores * cfg.payload_bytes;
        s->ladder_.build(cfg.payload_bytes, recorded, ladder_workers);
        s->ladder_.quiesced = cfg.ladder_quiesce;
        s->ladder_.discarded_bytes = static_cast<uint64_t>(cfg.warmup) * cfg.cores * cfg.payload_bytes;
    }

    SocketConfig sc;
    sc.ladder = s->ladder_.enabled ? &s->ladder_ : nullptr;
    sc.ladder_sync = (s->ladder_.enabled && s->ladder_.quiesced) ? &s->ladder_sync_ : nullptr;
    // commented out b/c deadcode
    // sc.payload_bytes = cfg.payload_bytes;
    sc.chip = cfg.chip;
    sc.cores = cfg.cores;
    sc.workers = cfg.workers;
    sc.pin = cfg.pin;
    // commented out b/c deadcode
    // sc.roundtrip = false;
    sc.send_window = cfg.send_window;
    sc.send_blocking = cfg.send_blocking;
    sc.ns_per_cycle = s->ns_per_cycle_;
    sc.record_from_start = cfg.warmup == 0;
    sc.warmup_msgs = static_cast<uint64_t>(cfg.warmup) * static_cast<uint64_t>(cfg.cores);

    s->inner_ = std::make_unique<D2H2H2DSocket>(
        region, s->deliverer_.get(), HostTopology{cfg.host_ident, cfg.host_num, cfg.chips_per_host}, s->clock_,
        sc, *s->primary_);

    for (const auto& t : s->mesh_peers_) {
        s->inner_->add_peer(t.get());
    }

    {
        // THE STORE FAULT DOMAIN. A store's offset arrives from another machine, so an
        // executor that trusts it is an arbitrary-write primitive. L1Map supplies the L1
        // bounds; the ring span is the socket's, so it is filled in here.
        D2H2H2DSocket::StoreGuard g = s->l1_.store_guard();
        g.ring_bytes = h2d_ring_bytes;
        s->inner_->set_store_guard(g);
    }

    return s;
}

D2DSocket::~D2DSocket() = default;

HostRegion& D2DSocket::region() const { return *region_; }

std::string D2DSocket::deliverer_describe() const { return deliverer_->describe(); }
std::string D2DSocket::transport_describe() const { return primary_->describe(); }

std::vector<uint32_t> D2DSocket::sender_compile_args(
    uint32_t iterations, uint32_t opcode, uint32_t flags, bool await_completion) const {
    const DeviceView& dev = region_->device();
    return {
        dev.pcie_xy_enc,
        static_cast<uint32_t>(dev.io_base & 0xFFFFFFFFull),
        static_cast<uint32_t>(dev.io_base >> 32),
        cfg_.grid_width,
        l1_.payload_addr,
        l1_.stage_addr,
        l1_.signal_addr,
        cfg_.payload_bytes,
        iterations,
        opcode,
        flags,
        // AWAIT THE DOORBELL. The kernel's control word is a single slot; without this it arms
        // iteration i+1 before a host worker has read iteration i, and the duplicate filter
        // drops the skipped message.
        await_completion ? 1u : 0u,
        l1_.completion_addr,
        // ARGS 13 AND 14, AND THEY ARE NOT OPTIONAL. test_kernel.cpp declares both as
        // unconditional function-scope constexpr (lines 166-167), so get_ct_arg<13>() is
        // instantiated no matter what the probe is set to -- and get_ct_arg static_asserts its
        // index against the size of THIS vector (compile_time_args.h:27). Stopping at 13 args
        // made the D2D sender kernel fail its JIT compile, which a green host build cannot
        // show: the assert fires on the device, at run time.
        //
        // verify_landing is 0 because it cannot be anything else here. The probe reads back
        // into landing_addr, and D2DSocket's L1Map has no landing slot -- compute() ends at
        // dest_word_addr. Arming it needs a new field there, which also moves control_bytes()
        // from 280 to 352 and with it the fits() bound. A real gap against
        // test_oneway_volume.cpp, which does carry --verify-landing.
        0u,  // verify_landing
        0u,  // landing_addr -- never read: the probe branch is if constexpr'd away
    };
}

std::vector<uint32_t> D2DSocket::receiver_compile_args() const {
    return {l1_.deliver_addr, cfg_.payload_bytes, l1_.signal_addr, 1u, l1_.stop_addr, l1_.dest_word_addr};
}

// ===========================================================================
// Pass-throughs
// ===========================================================================

bool D2DSocket::open(std::string& err) { return inner_->open(err); }
void D2DSocket::stop() { inner_->stop(); }
const SocketCounters& D2DSocket::counters() const { return inner_->counters(); }
std::vector<Transport*> D2DSocket::peers_for_barrier() const { return inner_->peers_for_barrier(); }
void D2DSocket::set_recording(bool on) { inner_->set_recording(on); }
void D2DSocket::open_recording_gate() { inner_->open_recording_gate(); }
void D2DSocket::stamp_timed_end() { inner_->stamp_timed_end(); }
uint64_t D2DSocket::timed_start_ns() const { return inner_->timed_start_ns(); }
bool D2DSocket::transport_failed() const { return inner_->transport_failed(); }
std::string D2DSocket::first_error() const { return inner_->first_error(); }
std::string D2DSocket::stall_dump(const char* where) const { return inner_->stall_dump(where); }
uint64_t D2DSocket::store_faults() const { return inner_->store_faults(); }

RunStats D2DSocket::collect() const {
    RunStats s = inner_->collect();
    s.ladder = ladder_;
    s.ladder.quiesce_clean = ladder_sync_.clean.load(std::memory_order_relaxed);
    s.ladder.quiesce_degraded = ladder_sync_.degraded.load(std::memory_order_relaxed);
    return s;
}

#endif  // TT_METAL_HOST_BRIDGE

}  // namespace tt::tt_metal::experimental
