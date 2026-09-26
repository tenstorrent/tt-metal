// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared helpers for the Blackhole PCIe peer-to-peer microbenchmarks: BAR discovery, outbound
// iATU programming through BAR2, inbound TLB windows, and L1 access.

#pragma once

#include <fmt/format.h>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>
#include <llrt/tt_cluster.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/pcie/pci_device.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/tt_device/tt_device.hpp>
#include <umd/device/types/tlb.hpp>

namespace tt::tt_metal::pcie_p2p {

// ---- Blackhole constants (mirrors tt-kmd blackhole.c and hw/inc/blackhole/noc_parameters.h) ----
constexpr uint64_t kTlb2M = 2ull << 20;
constexpr uint32_t kTlb2MCount = 202;   // 2 MiB windows live in BAR0 at id * 2 MiB
constexpr uint32_t kIatuBase = 0x1000;  // relative to BAR2
constexpr uint32_t kIatuRegions = 16;
constexpr uint32_t kIatuRegionStride = 0x100;
constexpr uint32_t kIatuCtrl1 = 0x00, kIatuCtrl2 = 0x04, kIatuLowerBase = 0x08, kIatuUpperBase = 0x0C;
constexpr uint32_t kIatuLowerLimit = 0x10, kIatuLowerTarget = 0x14, kIatuUpperTarget = 0x18, kIatuCtrl3 = 0x1C;
constexpr uint32_t kIatuUpperLimit = 0x20;
constexpr uint32_t kIatuIncreaseRegionSize = 1u << 13;
constexpr uint32_t kIatuRegionEn = 1u << 31;
constexpr uint64_t kNocPcieSelect = 1ull << 60;  // bit 60 selects the PCIe outbound path
constexpr uint32_t kNocAddrLocalBits = 36;
constexpr uint32_t kNocNodeIdBits = 6;
constexpr uint64_t kNocLocalLimit = 1ull << kNocAddrLocalBits;  // iATU base must stay below this
constexpr uint32_t kL1Size = 1536u * 1024u;

inline uint64_t tlb_ordering(const std::string& s) {
    if (s == "relaxed") {
        return tt::umd::tlb_data::Relaxed;
    }
    if (s == "strict") {
        return tt::umd::tlb_data::Strict;
    }
    if (s == "posted") {
        return tt::umd::tlb_data::Posted;
    }
    TT_THROW("bad ordering {}", s);
}

// ---- outbound iATU access through BAR2 ----
struct Iatu {
    volatile uint32_t* regs = nullptr;  // BAR2 base
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
    uint64_t limit(uint32_t region) const {
        return (uint64_t(rd(region, kIatuUpperLimit)) << 32) | rd(region, kIatuLowerLimit);
    }
    uint64_t target(uint32_t region) const {
        return (uint64_t(rd(region, kIatuUpperTarget)) << 32) | rd(region, kIatuLowerTarget);
    }
    void program(uint32_t region, uint64_t base, uint64_t size, uint64_t target) const {
        const uint64_t limit = base + size - 1;
        wr(region, kIatuCtrl2, 0);  // disable while reprogramming
        wr(region, kIatuLowerBase, uint32_t(base));
        wr(region, kIatuUpperBase, uint32_t(base >> 32));
        wr(region, kIatuLowerTarget, uint32_t(target));
        wr(region, kIatuUpperTarget, uint32_t(target >> 32));
        wr(region, kIatuLowerLimit, uint32_t(limit));
        wr(region, kIatuUpperLimit, uint32_t(limit >> 32));
        wr(region, kIatuCtrl1, kIatuIncreaseRegionSize);
        wr(region, kIatuCtrl3, 0);
        wr(region, kIatuCtrl2, kIatuRegionEn);
        (void)rd(region, kIatuCtrl2);  // flush posted MMIO writes
    }
    void disable(uint32_t region) const {
        wr(region, kIatuCtrl2, 0);
        (void)rd(region, kIatuCtrl2);
    }
};

struct Chip {
    int id = -1;
    std::shared_ptr<distributed::MeshDevice> mesh;
    IDevice* dev = nullptr;
    tt::umd::TTDevice* ttdev = nullptr;
    tt::umd::PCIDevice* pci = nullptr;
    std::string bdf;
    uint64_t bar_phys[6] = {0};
    uint64_t bar_size[6] = {0};
    Iatu iatu{};
    uint64_t pcie_noc_base = 0;  // (xy << 36) | (1 << 60) for this chip's PCIe core
    uint32_t l1_unreserved = 0;
    std::vector<std::unique_ptr<tt::umd::TlbHandle>> tlbs;
    std::vector<uint32_t> programmed_regions;

    ~Chip() {
        for (auto r : programmed_regions) {
            if (iatu.regs) {
                iatu.disable(r);
            }
        }
    }
};

inline void read_sysfs_bars(Chip& c) {
    std::string bdf = c.bdf;
    if (bdf.size() == 7) {
        bdf = "0000:" + bdf;  // "01:00.0" -> "0000:01:00.0"
    }
    std::ifstream f("/sys/bus/pci/devices/" + bdf + "/resource");
    TT_FATAL(f.good(), "cannot read /sys/bus/pci/devices/{}/resource", bdf);
    std::string line;
    for (int i = 0; i < 6 && std::getline(f, line); ++i) {
        uint64_t start = 0, end = 0, flags = 0;
        sscanf(line.c_str(), "%lx %lx %lx", &start, &end, &flags);
        c.bar_phys[i] = start;
        c.bar_size[i] = (end >= start && start) ? (end - start + 1) : 0;
    }
}

// Fill in everything about a PCIe-attached Blackhole chip that the benchmarks need.
inline void init_chip(Chip& c, int id, std::shared_ptr<distributed::MeshDevice> mesh) {
    auto& cluster = MetalContext::instance().get_cluster();
    c.id = id;
    c.mesh = std::move(mesh);
    c.dev = c.mesh->get_devices()[0];
    TT_FATAL(c.dev->is_mmio_capable(), "chip {} is not PCIe attached", id);
    c.ttdev = cluster.get_driver()->get_tt_device(id);
    c.pci = c.ttdev->get_pci_device();
    TT_FATAL(c.pci && c.pci->bar2_uc, "chip {}: no BAR2 mapping in UMD", id);
    c.bdf = c.pci->get_device_info().pci_bdf;
    read_sysfs_bars(c);
    c.iatu.regs = reinterpret_cast<volatile uint32_t*>(c.pci->bar2_uc);
    const auto pcie_cores = cluster.get_soc_desc(id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    TT_FATAL(!pcie_cores.empty(), "chip {}: no PCIe core", id);
    const uint64_t xy = (uint64_t(pcie_cores.front().y) << kNocNodeIdBits) | uint64_t(pcie_cores.front().x);
    c.pcie_noc_base = (xy << kNocAddrLocalBits) | kNocPcieSelect;
    c.l1_unreserved = c.dev->allocator()->get_base_allocator_addr(HalMemType::L1);
}

inline void dump_chip(const Chip& c) {
    fmt::print(
        "chip {} pci {}  BAR0 phys 0x{:x} size {} MiB | BAR2 phys 0x{:x} size {} KiB | BAR4 phys 0x{:x} size {} GiB\n",
        c.id,
        c.bdf,
        c.bar_phys[0],
        c.bar_size[0] >> 20,
        c.bar_phys[2],
        c.bar_size[2] >> 10,
        c.bar_phys[4],
        c.bar_size[4] >> 30);
    fmt::print("  PCIe core NoC base 0x{:016x}, L1 unreserved base 0x{:x}\n", c.pcie_noc_base, c.l1_unreserved);
    fmt::print("  outbound iATU regions (BAR2 + 0x1000):\n");
    uint32_t free_cnt = 0;
    for (uint32_t r = 0; r < kIatuRegions; ++r) {
        if (!c.iatu.enabled(r)) {
            ++free_cnt;
            continue;
        }
        fmt::print(
            "    region {:2}  EN  base 0x{:012x}  limit 0x{:012x}  target 0x{:012x}  ({} MiB)  ctrl1 0x{:x}\n",
            r,
            c.iatu.base(r),
            c.iatu.limit(r),
            c.iatu.target(r),
            (c.iatu.limit(r) - c.iatu.base(r) + 1) >> 20,
            c.iatu.rd(r, kIatuCtrl1));
    }
    fmt::print("    {} of {} regions free\n", free_cnt, kIatuRegions);
}

// Point a fresh 2 MiB window on `c` at NoC (x,y) + local_offset. Returns the window's BAR0 offset.
inline uint64_t make_inbound_window(
    Chip& c, uint32_t x, uint32_t y, uint64_t local_offset, uint64_t ordering, const char* what) {
    auto tlb = c.pci->allocate_tlb(kTlb2M, tt::umd::TlbMapping::UC);
    tt::umd::tlb_data cfg{};
    cfg.local_offset = local_offset;
    cfg.x_end = x;
    cfg.y_end = y;
    cfg.noc_sel = 0;
    cfg.ordering = ordering;
    tlb->configure(cfg);
    const int id = tlb->get_tlb_id();
    TT_FATAL(id >= 0 && uint32_t(id) < kTlb2MCount, "expected a 2 MiB BAR0 window, got tlb id {}", id);
    const uint64_t bar_off = uint64_t(id) * kTlb2M;
    TT_FATAL(bar_off + kTlb2M <= c.bar_size[0], "window beyond BAR0");
    fmt::print(
        "  chip {} inbound window [{}]: tlb id {} -> BAR0 + 0x{:x} (phys 0x{:x}) -> NoC ({},{}) @ 0x{:x}, ordering "
        "{}\n",
        c.id,
        what,
        id,
        bar_off,
        c.bar_phys[0] + bar_off,
        x,
        y,
        local_offset,
        ordering);
    c.tlbs.push_back(std::move(tlb));
    return bar_off;
}

// Program outbound region `region` on `src` so that NoC addresses noc_base .. +2 MiB hit dst's BAR0 + bar_off.
inline void make_outbound(
    Chip& src, uint32_t region, uint64_t noc_base, const Chip& dst, uint64_t bar_off, bool force) {
    TT_FATAL(region < kIatuRegions, "bad region {}", region);
    if (src.iatu.enabled(region) && !force) {
        TT_THROW(
            "chip {} iATU region {} already enabled (base 0x{:x} target 0x{:x}); pick another region or --force",
            src.id,
            region,
            src.iatu.base(region),
            src.iatu.target(region));
    }
    const uint64_t target = dst.bar_phys[0] + bar_off;
    src.iatu.program(region, noc_base, kTlb2M, target);
    src.programmed_regions.push_back(region);
    fmt::print(
        "  chip {} outbound iATU region {}: NoC 0x{:x}..0x{:x} -> PCIe 0x{:x} (chip {} BAR0 + 0x{:x})  readback en={} "
        "base=0x{:x} target=0x{:x}\n",
        src.id,
        region,
        noc_base,
        noc_base + kTlb2M - 1,
        target,
        dst.id,
        bar_off,
        src.iatu.enabled(region),
        src.iatu.base(region),
        src.iatu.target(region));
    TT_FATAL(
        src.iatu.enabled(region) && src.iatu.base(region) == noc_base && src.iatu.target(region) == target,
        "iATU readback mismatch");
}

inline std::vector<uint32_t> read_l1(IDevice* dev, CoreCoord core, uint32_t addr, uint32_t bytes) {
    std::vector<uint32_t> v;
    detail::ReadFromDeviceL1(dev, core, addr, bytes, v);
    return v;
}
inline void write_l1(IDevice* dev, CoreCoord core, uint32_t addr, const std::vector<uint32_t>& v) {
    std::vector<uint32_t> copy = v;
    detail::WriteToDeviceL1(dev, core, addr, copy);
}

}  // namespace tt::tt_metal::pcie_p2p
