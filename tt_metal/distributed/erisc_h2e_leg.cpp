// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/erisc_h2e_leg.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_env_impl.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/distributed/erisc_bridge_doorbell.hpp"
#include "tt_metal/distributed/erisc_bridge_packet.hpp"
#include "tt_metal/distributed/erisc_bridge_placement.hpp"

namespace tt::tt_metal::experimental {

using namespace tt::tt_fabric;
using namespace tt::tt_metal::distributed;
namespace eb = tt::tt_fabric::erisc_bridge;

namespace {
using HeaderT = tt::tt_fabric::LowLatencyPacketHeaderT<0>;
}  // namespace

struct H2ELeg::Impl {
    Config cfg{};
    std::shared_ptr<MeshDevice> mesh;
    IDevice* dev = nullptr;
    CoreCoord eth_logical{0, 0};

    uint32_t rx_base = 0;  // the receiver channel in L1 this leg injects into
    uint32_t rx_slots = 0;
    uint32_t slot_bytes = 0;
    uint32_t bell = 0;   // host writes +1 here
    uint32_t avail = 0;  // router writes -1; reading this gives OCCUPANCY
    // A live link settles wherever it settles, so occupancy is a modular difference against a
    // baseline taken before the first frame. §7.2, open item 9.
    uint32_t avail_baseline = 0;
    eb::Credit credit{};
    uint32_t next_slot = 0;

    uint64_t injected = 0;
    uint64_t outstanding = 0;
    uint64_t credit_stalls = 0;
    uint64_t oversize = 0;
    uint64_t slot_mismatch = 0;

    // Only used when a task carries no header of its own.
    std::vector<uint32_t> scratch;

    std::vector<std::chrono::steady_clock::time_point> inject_at;
    uint64_t timed_upto = 0;
    std::vector<double> lat_us;
    std::string err;

    const Hal& hal() const { return mesh->impl().metal_env().get_hal(); }
    void fail(const std::string& w) {
        if (err.empty()) {
            err = w;
        }
    }
    // Frames the router has taken: it writes -1 per packet processed, so what it has NOT taken
    // is `outstanding`.
    uint64_t consumed() const { return injected >= outstanding ? injected - outstanding : 0; }
};

H2ELeg::H2ELeg() : impl_(std::make_unique<Impl>()) {}
H2ELeg::~H2ELeg() = default;

std::unique_ptr<H2ELeg> H2ELeg::create(const std::shared_ptr<MeshDevice>& mesh, const Config& cfg, std::string& err) {
    err.clear();
    if (mesh == nullptr || cfg.page_bytes == 0) {
        err = "H2ELeg::create: mesh and page_bytes are required";
        return nullptr;
    }
    std::unique_ptr<H2ELeg> leg(new H2ELeg());
    auto& im = *leg->impl_;
    im.cfg = cfg;
    im.mesh = mesh;
    im.dev = mesh->get_device(MeshCoordinate(cfg.mesh_row, cfg.mesh_col));
    if (im.dev == nullptr) {
        err = "H2ELeg::create: no device at mesh coordinate (" + std::to_string(cfg.mesh_row) + ", " +
              std::to_string(cfg.mesh_col) + ")";
        return nullptr;
    }

    auto& env = mesh->impl().metal_env();
    const auto& hal = env.get_hal();
    const auto& cp = env.get_control_plane();
    const auto& soc = env.get_cluster().get_soc_desc(im.dev->id());
    const auto node = cp.get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(im.dev->id()));

    std::vector<uint32_t> chans;
    for (const auto& c : cp.get_active_fabric_eth_channels(node)) {
        chans.push_back(static_cast<uint32_t>(c.first));
    }
    if (cfg.eth_chan == Config::kUnsetChan && (chans.empty() || cfg.link_idx >= chans.size())) {
        err = "H2ELeg::create: link_idx out of range for the active eth channels";
        return nullptr;
    }
    // The caller's channel wins, as in E2HLeg: link_idx into the active list can name a
    // different router than the one the caller resolved as bridged.
    const uint32_t chan = cfg.eth_chan != Config::kUnsetChan ? cfg.eth_chan : chans.at(cfg.link_idx);
    im.eth_logical = soc.get_eth_core_for_channel(chan, CoordSystem::LOGICAL);

    // The receiver channel the remote ERISC would have written -- derived from fabric's own
    // allocator, never a constant.
    const eb::EriscBridgePlacement place(eb::router_config(cp));
    // One receiver channel: VC0 ch0, stream 0.
    im.rx_base = place.receiver_channel_base(0, 0);
    im.rx_slots = place.receiver_channel_slots(0, 0);
    im.slot_bytes = place.channel_slot_bytes();
    if (im.rx_base == 0 || im.rx_slots == 0) {
        err = "H2ELeg::create: receiver channel unreadable";
        return nullptr;
    }
    // The router's own receiver-0 counter: fabric assigns stream ids, so a fixed id rings the wrong one.
    const uint32_t stream = eb::receiver_pkts_sent_stream(cp, *node.mesh_id, 0);
    if (!eb::stream_id_valid_for_eth(stream)) {
        err = "H2ELeg::create: doorbell stream id " + std::to_string(stream) + " is outside the ERISC's 32";
        return nullptr;
    }
    im.bell = eb::doorbell_addr(hal, stream);
    im.avail = eb::available_addr(hal, stream);
    // Read BEFORE the first frame: the accumulator need not start at 0.
    im.avail_baseline = eb::available_value(
        eb::reg_read(im.dev, im.eth_logical, CoreType::ETH, im.avail), eb::kWordsFreeWidthBlackhole);

    im.next_slot = static_cast<uint32_t>(cfg.frames_before % im.rx_slots);
    im.credit.slots = im.rx_slots;
    im.credit.margin = cfg.credit_margin;
    if (cfg.collect_timing) {
        // One lap of the channel bounds what can be outstanding, so it bounds the ring.
        im.inject_at.assign(im.rx_slots, std::chrono::steady_clock::time_point{});
    }
    return leg;
}

bool H2ELeg::publish(const BridgeDeliverTask& task) {
    Impl& im = *impl_;
    if (im.dev == nullptr) {
        return false;
    }
    if (!im.credit.has_room(static_cast<uint32_t>(im.outstanding))) {
        ++im.credit_stalls;
        return false;  // the ROUTER has not kept up; the caller re-offers
    }
    if (task.length == 0) {
        im.fail("H2ELeg::publish: empty frame");
        return false;
    }
    const uint32_t hdr_bytes = sizeof(HeaderT);
    const uint32_t slot_addr = im.rx_base + (im.next_slot % im.rx_slots) * im.slot_bytes;
    // The sending router's view of where this frame belongs. Counted, not refused: a disagreement
    // means the ends drifted. Widened explicitly -- uint16_t promotes to int, which -Werror rejects.
    if (task.far_slot.has_value() && static_cast<uint32_t>(*task.far_slot) != im.next_slot % im.rx_slots) {
        ++im.slot_mismatch;
    }
    bool ok = false;

    if (im.cfg.frames_carry_header) {
        // THE ZERO-REBUILD PATH. The task names a page in the Rx arena that already holds
        // [header | payload] exactly as the channel expects.
        if (task.length > im.slot_bytes) {
            ++im.oversize;
            im.fail("H2ELeg::publish: frame larger than the receiver slot");
            return false;
        }
        if (im.cfg.alias_region_base == nullptr) {
            im.fail("H2ELeg::publish: frames_carry_header but no alias_region_base to read from");
            return false;
        }
        // REGION-relative, as H2H delivers it: slot_off() already includes the Rx segment, so
        // adding bridge_segment_offset() here read a whole segment past the frame.
        const uint8_t* src = im.cfg.alias_region_base + task.page_offset;
        ok = eb::write_bytes_to_l1(im.dev, im.eth_logical, slot_addr, src, task.length);
    } else {
        // Synthesised: a caller with bare payload bytes and no fabric header of its own.
        const uint32_t payload = std::min<uint32_t>(task.length, im.cfg.page_bytes);
        if (hdr_bytes + payload > im.slot_bytes) {
            ++im.oversize;
            im.fail("H2ELeg::publish: header + payload larger than the receiver slot");
            return false;
        }
        // No staging copy when aliased: the arena is ordinary host memory the MMIO path reads
        // directly. Safe to split -- the doorbell arms the slot, so only "both before it" matters.
        const bool aliased = im.cfg.alias_region_base != nullptr;
        // Header-sized when the payload is written separately; header+payload when there is
        // no source, so an unaliased caller still gets the zeroed slot it always got.
        im.scratch.assign((aliased ? hdr_bytes : hdr_bytes + payload) / sizeof(uint32_t), 0);
        if (!eb::make_local_write_header<HeaderT>(
                im.scratch.data(), im.hal(), im.cfg.dst_noc_x, im.cfg.dst_noc_y, im.cfg.dst_l1_addr, payload)) {
            im.fail("H2ELeg::publish: header build rejected the payload size");
            return false;
        }
        ok = eb::write_bytes_to_l1(
            im.dev, im.eth_logical, slot_addr, im.scratch.data(), aliased ? hdr_bytes : hdr_bytes + payload);
        if (ok && aliased) {
            const uint8_t* src = im.cfg.alias_region_base + task.page_offset;  // region-relative
            ok = eb::write_bytes_to_l1(im.dev, im.eth_logical, slot_addr + hdr_bytes, src, payload);
        }
    }
    if (!ok) {
        im.fail("H2ELeg::publish: write into the receiver channel failed");
        return false;
    }

    // DOORBELL STRICTLY AFTER THE PAYLOAD -- §5.2 step 4 after step 3, reproducing the router's
    // own eth_txq_is_busy discipline. The hardware ACCUMULATES this write.
    if (!eb::reg_write(
            im.dev, im.eth_logical, CoreType::ETH, im.bell, eb::doorbell_value(1, eb::kWordsFreeIncShiftBlackhole))) {
        im.fail("H2ELeg::publish: doorbell write failed");
        return false;
    }
    if (im.cfg.collect_timing && !im.inject_at.empty()) {
        im.inject_at[im.injected % im.inject_at.size()] = std::chrono::steady_clock::now();
    }
    ++im.next_slot;
    ++im.injected;
    ++im.outstanding;  // our own +1, until the next drained() says the router took it
    return true;
}

uint64_t H2ELeg::drained(uint32_t arena) {
    Impl& im = *impl_;
    (void)arena;  // one receiver channel per leg today; the parameter keeps the H2DLeg shape
    if (im.dev == nullptr) {
        return 0;
    }
    // Against the baseline, never the raw register: the raw value is a running total, and
    // comparing it to a slot count stalls the credit gate forever on a meaningless number.
    const uint32_t raw = eb::available_value(
        eb::reg_read(im.dev, im.eth_logical, CoreType::ETH, im.avail), eb::kWordsFreeWidthBlackhole);
    const uint32_t since = eb::occupancy_since(raw, im.avail_baseline, eb::kWordsFreeWidthBlackhole);
    // We cannot have more outstanding than we injected. More means something else moves this
    // accumulator, and a wrapped difference would stall forever.
    im.outstanding = std::min<uint64_t>(since, im.injected);

    // One occupancy read retires a whole run of samples: the router takes frames in order.
    if (im.cfg.collect_timing && !im.inject_at.empty()) {
        const uint64_t done = im.consumed();
        const auto now = std::chrono::steady_clock::now();
        for (uint64_t k = im.timed_upto; k < done; ++k) {
            // Only while the stamp is still in the ring; a wrong sample is worse than none.
            // And only past warmup, so the percentiles describe steady state.
            if (k >= im.cfg.timing_from_frame && done - k < im.inject_at.size()) {
                im.lat_us.push_back(
                    std::chrono::duration<double, std::micro>(now - im.inject_at[k % im.inject_at.size()]).count());
            }
        }
        im.timed_upto = done;
    }
    return im.consumed();
}

std::uint64_t H2ELeg::credit_stalls() const { return impl_->credit_stalls; }
std::uint64_t H2ELeg::oversize() const { return impl_->oversize; }
std::uint64_t H2ELeg::slot_mismatch() const { return impl_->slot_mismatch; }
const std::vector<double>& H2ELeg::latency_us() const { return impl_->lat_us; }

std::string H2ELeg::first_error() const { return impl_->err; }

}  // namespace tt::tt_metal::experimental
