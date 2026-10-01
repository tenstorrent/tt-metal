// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/pcie_p2p.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt_stl/assert.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/tlb.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental::pcie_p2p {

namespace {

// Blackhole constants (mirror tt-kmd blackhole.c and hw/inc/blackhole/noc_parameters.h).
constexpr uint64_t kTlb2M = 2ull << 20;
constexpr uint32_t kTlb2MCount = 202;
constexpr uint32_t kIatuBase = 0x1000;  // relative to BAR2
constexpr uint32_t kIatuRegions = 16;
constexpr uint32_t kIatuRegionStride = 0x100;
constexpr uint32_t kIatuCtrl1 = 0x00, kIatuCtrl2 = 0x04, kIatuLowerBase = 0x08, kIatuUpperBase = 0x0C;
constexpr uint32_t kIatuLowerLimit = 0x10, kIatuLowerTarget = 0x14, kIatuUpperTarget = 0x18, kIatuCtrl3 = 0x1C;
constexpr uint32_t kIatuUpperLimit = 0x20;
constexpr uint32_t kIatuIncreaseRegionSize = 1u << 13;
constexpr uint32_t kIatuRegionEn = 1u << 31;
constexpr uint64_t kNocPcieSelect = 1ull << 60;
constexpr uint32_t kNocAddrLocalBits = 36;
constexpr uint32_t kNocNodeIdBits = 6;
constexpr uint64_t kNocLocalLimit = 1ull << kNocAddrLocalBits;
constexpr uint64_t kPeerStride = 512ull << 20;  // one whole BAR0 per peer

struct Iatu {
    volatile uint32_t* regs = nullptr;
    uint32_t rd(uint32_t region, uint32_t reg) const {
        return regs[(kIatuBase + (2 * region + 0) * kIatuRegionStride + reg) / 4];
    }
    void wr(uint32_t region, uint32_t reg, uint32_t v) const {
        regs[(kIatuBase + (2 * region + 0) * kIatuRegionStride + reg) / 4] = v;
    }
    bool enabled(uint32_t region) const { return rd(region, kIatuCtrl2) & kIatuRegionEn; }
    uint64_t base(uint32_t region) const {
        return (uint64_t(rd(region, kIatuUpperBase)) << 32) | rd(region, kIatuLowerBase);
    }
    uint64_t target(uint32_t region) const {
        return (uint64_t(rd(region, kIatuUpperTarget)) << 32) | rd(region, kIatuLowerTarget);
    }
    void program(uint32_t region, uint64_t base, uint64_t size, uint64_t target) const {
        const uint64_t limit = base + size - 1;
        wr(region, kIatuCtrl2, 0);
        wr(region, kIatuLowerBase, uint32_t(base));
        wr(region, kIatuUpperBase, uint32_t(base >> 32));
        wr(region, kIatuLowerTarget, uint32_t(target));
        wr(region, kIatuUpperTarget, uint32_t(target >> 32));
        wr(region, kIatuLowerLimit, uint32_t(limit));
        wr(region, kIatuUpperLimit, uint32_t(limit >> 32));
        wr(region, kIatuCtrl1, kIatuIncreaseRegionSize);
        wr(region, kIatuCtrl3, 0);
        wr(region, kIatuCtrl2, kIatuRegionEn);
        (void)rd(region, kIatuCtrl2);
    }
    void disable(uint32_t region) const {
        wr(region, kIatuCtrl2, 0);
        (void)rd(region, kIatuCtrl2);
    }
};

struct ChipState {
    ChipId id = 0;
    uint32_t index = 0;  // position in the mesh device list
    IDevice* dev = nullptr;
    tt::umd::PCIDevice* pci = nullptr;
    std::string bdf;
    uint64_t bar_phys[6] = {0};
    uint64_t bar_size[6] = {0};
    Iatu iatu{};
    uint64_t pcie_noc_base = 0;
    std::vector<std::unique_ptr<tt::umd::TlbHandle>> tlbs;  // one per core, in cores() order
    std::vector<uint64_t> window_off;                       // BAR0 offset per core
    std::vector<uint32_t> programmed_regions;
};

struct State {
    Config cfg;
    std::vector<CoreCoord> cores;
    std::vector<ChipId> order;
    std::map<ChipId, ChipState> chips;
};

std::mutex g_mutex;
std::unique_ptr<State> g_state;

uint64_t tlb_ordering(const std::string& s) {
    if (s == "relaxed") {
        return tt::umd::tlb_data::Relaxed;
    }
    if (s == "strict") {
        return tt::umd::tlb_data::Strict;
    }
    if (s == "posted") {
        return tt::umd::tlb_data::Posted;
    }
    TT_THROW("pcie_p2p: bad TLB ordering '{}' (relaxed|strict|posted)", s);
}

void read_sysfs_bars(ChipState& c) {
    std::string bdf = c.bdf;
    if (bdf.size() == 7) {
        bdf = "0000:" + bdf;
    }
    std::ifstream f("/sys/bus/pci/devices/" + bdf + "/resource");
    TT_FATAL(f.good(), "pcie_p2p: cannot read /sys/bus/pci/devices/{}/resource", bdf);
    std::string line;
    for (int i = 0; i < 6 && std::getline(f, line); ++i) {
        uint64_t start = 0, end = 0, flags = 0;
        sscanf(line.c_str(), "%lx %lx %lx", &start, &end, &flags);
        c.bar_phys[i] = start;
        c.bar_size[i] = (end >= start && start) ? (end - start + 1) : 0;
    }
}

void init_chip(ChipState& c, IDevice* dev, uint32_t index) {
    auto& cluster = MetalContext::instance().get_cluster();
    c.id = dev->id();
    c.index = index;
    c.dev = dev;
    TT_FATAL(dev->is_mmio_capable(), "pcie_p2p: chip {} is not PCIe attached", c.id);
    auto* ttdev = cluster.get_driver()->get_tt_device(c.id);
    c.pci = ttdev->get_pci_device();
    TT_FATAL(c.pci && c.pci->bar2_uc, "pcie_p2p: chip {} has no BAR2 mapping in UMD", c.id);
    c.bdf = c.pci->get_device_info().pci_bdf;
    read_sysfs_bars(c);
    TT_FATAL(
        c.bar_size[0] >= kTlb2M * kTlb2MCount,
        "pcie_p2p: chip {} BAR0 is {} bytes, expected 512 MiB",
        c.id,
        c.bar_size[0]);
    c.iatu.regs = reinterpret_cast<volatile uint32_t*>(c.pci->bar2_uc);
    const auto pcie_cores = cluster.get_soc_desc(c.id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    TT_FATAL(!pcie_cores.empty(), "pcie_p2p: chip {} has no PCIe core in its SoC descriptor", c.id);
    const uint64_t xy = (uint64_t(pcie_cores.front().y) << kNocNodeIdBits) | uint64_t(pcie_cores.front().x);
    c.pcie_noc_base = (xy << kNocAddrLocalBits) | kNocPcieSelect;
}

// One 2 MiB inbound window on `c` pointing at the L1 of logical worker `core` (offset 0). Returns BAR0 offset.
uint64_t make_inbound_window(ChipState& c, const CoreCoord& core, uint64_t ordering) {
    const CoreCoord v = c.dev->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto tlb = c.pci->allocate_tlb(kTlb2M, tt::umd::TlbMapping::UC);
    tt::umd::tlb_data cfg{};
    cfg.local_offset = 0;
    cfg.x_end = v.x;
    cfg.y_end = v.y;
    cfg.noc_sel = 0;
    cfg.ordering = ordering;
    tlb->configure(cfg);
    const int id = tlb->get_tlb_id();
    TT_FATAL(id >= 0 && uint32_t(id) < kTlb2MCount, "pcie_p2p: expected a 2 MiB BAR0 window, got tlb id {}", id);
    const uint64_t bar_off = uint64_t(id) * kTlb2M;
    // Non-destructive proof that the window reaches this core: the first 64 B of L1 read the same through the
    // window (a PCIe read of the core) and through UMD's own path.
    {
        std::vector<uint32_t> via_umd;
        detail::ReadFromDeviceL1(c.dev, core, 0, 64, via_umd);
        auto* win = reinterpret_cast<volatile uint32_t*>(tlb->get_base());
        for (size_t w = 0; w < 16; ++w) {
            TT_FATAL(
                win[w] == via_umd[w],
                "pcie_p2p: chip {} window {} for core {} reads 0x{:x} but UMD reads 0x{:x} at L1+{}",
                c.id,
                id,
                core.str(),
                win[w],
                via_umd[w],
                w * 4);
        }
    }
    c.tlbs.push_back(std::move(tlb));
    return bar_off;
}

void make_outbound(ChipState& src, uint32_t region, uint64_t noc_base, const ChipState& dst) {
    TT_FATAL(region < kIatuRegions, "pcie_p2p: iATU region {} out of range", region);
    const uint64_t size = dst.bar_size[0];
    if (src.iatu.enabled(region)) {
        // A process that died without teardown() leaves its regions programmed (the iATU only resets with the
        // chip). Re-use a region that already maps exactly what we are about to program; anything else is
        // someone else's mapping and must not be clobbered.
        TT_FATAL(
            src.iatu.base(region) == noc_base && src.iatu.target(region) == dst.bar_phys[0],
            "pcie_p2p: chip {} iATU region {} already enabled with a different mapping (base 0x{:x} target 0x{:x}); "
            "choose another first_iatu_region or reset the chip",
            src.id,
            region,
            src.iatu.base(region),
            src.iatu.target(region));
        log_warning(
            tt::LogMetal,
            "pcie_p2p: chip {} iATU region {} was left enabled by an earlier run; reprogramming it",
            src.id,
            region);
    }
    TT_FATAL(
        noc_base + size <= kNocLocalLimit,
        "pcie_p2p: outbound window 0x{:x}+0x{:x} exceeds the 36-bit NoC local space",
        noc_base,
        size);
    src.iatu.program(region, noc_base, size, dst.bar_phys[0]);
    TT_FATAL(
        src.iatu.enabled(region) && src.iatu.base(region) == noc_base && src.iatu.target(region) == dst.bar_phys[0],
        "pcie_p2p: chip {} iATU region {} readback mismatch",
        src.id,
        region);
    src.programmed_regions.push_back(region);
    log_info(
        tt::LogMetal,
        "pcie_p2p: chip {} iATU region {}: NoC 0x{:x}..0x{:x} -> chip {} BAR0 (PCIe 0x{:x})",
        src.id,
        region,
        noc_base,
        noc_base + size - 1,
        dst.id,
        dst.bar_phys[0]);
}

void teardown_locked() {
    if (!g_state) {
        return;
    }
    for (auto& [id, c] : g_state->chips) {
        for (auto r : c.programmed_regions) {
            c.iatu.disable(r);
        }
        c.programmed_regions.clear();
        c.tlbs.clear();
    }
    g_state.reset();
}

}  // namespace

void use_isolated_mesh(const std::vector<int>& chips) {
    TT_FATAL(!chips.empty(), "pcie_p2p::use_isolated_mesh: no chips");
    char path[] = "/tmp/pcie_p2p_isolated_XXXXXX.textproto";
    int fd = mkstemps(path, 10);
    TT_FATAL(fd >= 0, "pcie_p2p: cannot create temp mesh graph descriptor");
    std::string d = fmt::format(
        "mesh_descriptors {{\n  name: \"M0\"\n  arch: BLACKHOLE\n  device_topology {{ dims: [ 1, {} ] }}\n"
        "  host_topology   {{ dims: [ 1, 1 ] }}\n  channels {{ count: 1 policy: RELAXED }}\n}}\n"
        "top_level_instance {{ mesh {{ mesh_descriptor: \"M0\" mesh_id: 0 }} }}\n",
        chips.size());
    TT_FATAL(write(fd, d.data(), d.size()) == ssize_t(d.size()), "pcie_p2p: short write of mesh descriptor");
    close(fd);
    std::map<tt::tt_fabric::FabricNodeId, ChipId> mapping;
    for (size_t i = 0; i < chips.size(); ++i) {
        mapping[tt::tt_fabric::FabricNodeId(tt::tt_fabric::MeshId{0}, uint32_t(i))] = chips[i];
    }
    log_info(
        tt::LogMetal,
        "pcie_p2p: isolated 1x{} mesh over chips ({}), descriptor {}",
        chips.size(),
        fmt::join(chips, ","),
        path);
    MetalContext::instance().set_custom_fabric_topology(path, mapping);
}

void setup(distributed::MeshDevice& mesh, const std::vector<CoreCoord>& logical_cores, const Config& cfg) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto devices = mesh.get_devices();
    TT_FATAL(devices.size() >= 2, "pcie_p2p::setup: need at least 2 devices, mesh has {}", devices.size());
    TT_FATAL(!logical_cores.empty(), "pcie_p2p::setup: no cores");
    TT_FATAL(
        cfg.noc_base + devices.size() * kPeerStride <= kNocLocalLimit,
        "pcie_p2p::setup: {} devices x 512 MiB from 0x{:x} exceeds the 36-bit NoC local space",
        devices.size(),
        cfg.noc_base);
    TT_FATAL(
        cfg.first_iatu_region + devices.size() - 1 <= kIatuRegions,
        "pcie_p2p::setup: not enough iATU regions from {} for {} peers",
        cfg.first_iatu_region,
        devices.size() - 1);
    if (g_state) {
        bool same = g_state->cores == logical_cores && g_state->order.size() == devices.size();
        for (size_t i = 0; same && i < devices.size(); ++i) {
            same = g_state->order[i] == devices[i]->id();
        }
        if (same) {
            return;
        }
        teardown_locked();
    }
    TT_FATAL(MetalContext::instance().get_cluster().arch() == tt::ARCH::BLACKHOLE, "pcie_p2p: Blackhole only");

    auto st = std::make_unique<State>();
    st->cfg = cfg;
    st->cores = logical_cores;
    const uint64_t ord = tlb_ordering(cfg.ordering);
    for (size_t i = 0; i < devices.size(); ++i) {
        ChipState c;
        init_chip(c, devices[i], uint32_t(i));
        for (const auto& core : logical_cores) {
            c.window_off.push_back(make_inbound_window(c, core, ord));
        }
        st->order.push_back(c.id);
        st->chips.emplace(c.id, std::move(c));
    }
    // Outbound: on every chip, one region per peer, mapping the peer's whole BAR0 at noc_base + peer_index * 512 MiB.
    try {
        for (auto& [id, src] : st->chips) {
            uint32_t region = cfg.first_iatu_region;
            for (auto peer_id : st->order) {
                if (peer_id == id) {
                    continue;
                }
                const auto& dst = st->chips.at(peer_id);
                make_outbound(src, region++, cfg.noc_base + uint64_t(dst.index) * kPeerStride, dst);
            }
        }
    } catch (...) {
        for (auto& [id, c] : st->chips) {
            for (auto r : c.programmed_regions) {
                c.iatu.disable(r);
            }
        }
        throw;
    }
    log_info(
        tt::LogMetal,
        "pcie_p2p: windows ready for {} devices x {} cores (noc_base 0x{:x}, ordering {})",
        devices.size(),
        logical_cores.size(),
        cfg.noc_base,
        cfg.ordering);
    g_state = std::move(st);
}

bool is_setup() {
    std::lock_guard<std::mutex> lk(g_mutex);
    return g_state != nullptr;
}

void teardown() {
    std::lock_guard<std::mutex> lk(g_mutex);
    teardown_locked();
}

uint64_t remote_l1_noc_addr(ChipId from_chip, ChipId to_chip, const CoreCoord& core, uint32_t l1_addr) {
    std::lock_guard<std::mutex> lk(g_mutex);
    TT_FATAL(g_state, "pcie_p2p: setup() has not run");
    TT_FATAL(from_chip != to_chip, "pcie_p2p: {} -> {} is not a peer transfer", from_chip, to_chip);
    const auto& src = g_state->chips.at(from_chip);
    const auto& dst = g_state->chips.at(to_chip);
    size_t ci = 0;
    for (; ci < g_state->cores.size(); ++ci) {
        if (g_state->cores[ci] == core) {
            break;
        }
    }
    TT_FATAL(ci < g_state->cores.size(), "pcie_p2p: core {} was not part of setup()", core.str());
    TT_FATAL(l1_addr < kTlb2M, "pcie_p2p: L1 address 0x{:x} outside the 2 MiB window", l1_addr);
    const uint64_t local = g_state->cfg.noc_base + uint64_t(dst.index) * kPeerStride + dst.window_off[ci] + l1_addr;
    return src.pcie_noc_base | local;
}

uint32_t mesh_index(ChipId chip) {
    std::lock_guard<std::mutex> lk(g_mutex);
    TT_FATAL(g_state, "pcie_p2p: setup() has not run");
    return g_state->chips.at(chip).index;
}

const std::vector<CoreCoord>& cores() {
    std::lock_guard<std::mutex> lk(g_mutex);
    TT_FATAL(g_state, "pcie_p2p: setup() has not run");
    return g_state->cores;
}

}  // namespace tt::tt_metal::experimental::pcie_p2p
