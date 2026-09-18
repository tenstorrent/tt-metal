// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/sockets/d2d_socket.hpp>

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

L1Map L1Map::compute(uint32_t l1_base, uint32_t l1_size, uint32_t payload_bytes, bool bidirectional) {
    L1Map m;
    m.l1_size = l1_size;
    m.bidirectional = bidirectional;
    m.payload_addr = align64(l1_base);
    m.stage_addr = m.payload_addr + align64(payload_bytes);
    m.signal_addr = m.stage_addr + L1Map::kStageSlots * L1Map::kStageSlotBytes;
    m.completion_addr = m.signal_addr + L1Map::kDoorbellBytes;
    m.stop_addr = m.completion_addr + L1Map::kDoorbellBytes;
    m.dest_word_addr = m.stop_addr + L1Map::kDoorbellBytes;
    if (bidirectional) {
        // Its own buffer, above the control words, so a core can hold an outbound payload and
        // an inbound one at once. The pull kernel already takes this as a compile arg and
        // never reads payload_addr, so nothing below the map changes.
        m.deliver_addr = align64(m.dest_word_addr + L1Map::kDestWordBytes);
        m.deliver_end = m.deliver_addr + align64(payload_bytes);
    } else {
        m.deliver_addr = m.payload_addr;
        m.deliver_end = 0;
    }
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
        const uint32_t copies = payload_copies();
        const uint32_t overhead = control_bytes() + copies * pad;
        // The largest payload that WOULD fit is 64-aligned and so carries no pad of its own,
        // which makes control_bytes() alone the floor -- min_l1 is what a zero-byte payload
        // already costs. Guarded: a high allocator base can put min_l1 above l1_size, and
        // unguarded this wrapped and advertised a ~4 GB ceiling.
        const uint32_t min_l1 = payload_addr + control_bytes();
        const uint32_t ceiling = l1_size > min_l1 ? (((l1_size - min_l1) / copies) & ~0x3Fu) : 0u;
        std::ostringstream o;
        o << "payload " << payload_bytes << " B does not fit L1 on this core.\n"
          << "  L1 per core        " << l1_size << " B\n"
          << "  allocator base     " << payload_addr << " B\n"
          << "  needed             " << copies << " x " << payload_bytes << " B ("
          << (bidirectional ? "send + deliver" : "one shared buffer") << ") + " << overhead
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
    s->l1_ = L1Map::compute(l1_base, l1_size, cfg.payload_bytes, cfg.bidirectional);
    if (const std::string e = s->l1_.fits(cfg.payload_bytes); !e.empty()) {
        err = e;
        return nullptr;
    }

    const L1Layout layout = s->l1_.l1_layout();
    std::string derr;
    // Needed by the store guard, which must be given the ring it bounds against.
    H2DSocketConfig scfg;
    scfg.page_size = cfg.payload_bytes;
    // from kNumAliasRingSlots. The layout header names four consumers that must derive from
    // that constant and warns a mismatch "corrupts silently"; this is one of the four.
    scfg.fifo_size = kNumAliasRingSlots * cfg.payload_bytes;
    scfg.alias_region_base = HostRegion::reserved_base();
    const uint32_t h2d_ring_bytes = scfg.fifo_size;
    s->deliverer_ = make_h2d_socket_deliverer(s->mesh_device_, cfg.grid_width, cfg.cores, layout, scfg, derr);
    if (!s->deliverer_) {
        err = "H2D delivery unavailable: " + derr;
        return nullptr;
    }

    // A measurement input costing ~50 ms on the device, so it belongs to measure(). 0 here: a
    // cycles-flagged sample treats that as "no scale" and contributes nothing, which is the
    // honest answer for a socket nobody is measuring.
    s->clock_rate_detail_ = "not measured -- measure() was not called";

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
    // Off at creation; measure() opts in through set_measure_retire(), so a socket that is
    // only moving bytes never pays for the timing.
    tc.measure_retire = false;

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

    SocketConfig sc;
    sc.chip = cfg.chip;
    sc.cores = cfg.cores;
    sc.workers = cfg.workers;
    sc.pin = cfg.pin;
    sc.send_window = cfg.send_window;
    sc.send_blocking = cfg.send_blocking;

    s->inner_ = std::make_unique<D2H2H2DSocket>(
        region, s->deliverer_.get(), HostTopology{cfg.host_ident, cfg.host_num, cfg.chips_per_host}, sc, *s->primary_);

    for (const auto& t : s->mesh_peers_) {
        s->inner_->add_peer(t.get());
    }

    {
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

// ===========================================================================
// Pass-throughs
// ===========================================================================

std::string D2DSocket::measure(const D2DMeasurementConfig& m) {
    // Before the collective below, so a late call fails on every rank without any of them
    // entering a clock sync they would discard.
    if (opened_) {
        return "measure: the socket is already open; measurement must be configured before open()";
    }
    if (measuring_) {
        return "measure: already measuring; call it once";
    }
    measure_cfg_ = m;

    // Device cycles -> ns, for the stage the kernel reports in Tensix cycles.
    if (m.ns_per_cycle_override > 0.0) {
        ns_per_cycle_ = m.ns_per_cycle_override;
        clock_rate_detail_ = "supplied by the caller";
    } else {
        ns_per_cycle_ = measure_ns_per_cycle(*deliverer_, /*core=*/0, /*sample_ms=*/50, clock_rate_detail_);
    }

    // collective. Every rank must reach this at the same point: rank 0 probes each peer in
    // turn and the others answer. It lives here so a socket that is not being measured neither
    // runs it nor fails to be built when it cannot.
    {
        namespace mh = tt::tt_metal::distributed::multihost;
        const auto& ctx = mh::DistributedContext::get_current_world();
        clock_ = sync_clocks_to_hub(ctx, m.same_host);
        if (!clock_.valid) {
            return "refusing to report cross-host hop timings without a clock offset: " + clock_.error;
        }
    }

    // Refuses if open() has already run -- checked inside the socket rather than trusted here.
    if (const std::string e = inner_->configure_measurement(
            static_cast<uint64_t>(m.warmup) * static_cast<uint64_t>(cfg_.cores), ns_per_cycle_,
            m.measure_credit);
        !e.empty()) {
        return e;
    }
    inner_->set_measure_retire(m.measure_retire);

    measuring_ = true;
    return {};
}

bool D2DSocket::open(std::string& err) {
    opened_ = true;
    return inner_->open(err);
}
void D2DSocket::stop() { inner_->stop(); }
const SocketCounters& D2DSocket::counters() const { return inner_->counters(); }
std::vector<Transport*> D2DSocket::peers_for_barrier() const { return inner_->peers_for_barrier(); }
void D2DSocket::set_recording(bool on) { inner_->set_recording(on); }
void D2DSocket::open_recording_gate() { inner_->open_recording_gate(); }
void D2DSocket::stamp_timed_end() { inner_->stamp_timed_end(); }
uint64_t D2DSocket::timed_start_ns() const { return inner_->timed_start_ns(); }
bool D2DSocket::transport_failed() const { return inner_->transport_failed(); }
bool D2DSocket::peer_refused() const { return inner_->peer_refused(); }
std::string D2DSocket::first_error() const { return inner_->first_error(); }
std::string D2DSocket::stall_dump(const char* where) const { return inner_->stall_dump(where); }
uint64_t D2DSocket::store_faults() const { return inner_->store_faults(); }

RunStats D2DSocket::collect() const {
    RunStats s = inner_->collect();
    return s;
}

#endif  // TT_METAL_HOST_BRIDGE

}  // namespace tt::tt_metal::experimental
