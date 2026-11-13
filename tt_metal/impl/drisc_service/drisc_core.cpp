// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/drisc_service/drisc_core.hpp"

#include <tt_stl/assert.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <tt-umd/io_window/io_window.hpp>
#include <tt-umd/pcie/pci_device.hpp>
#include <tt-umd/tt_device/tt_device.hpp>

#include "internal/tt-1xx/blackhole/dev_mem_map.h"
#include "internal/tt-1xx/blackhole/noc/noc_parameters.h"
#include "llrt/tt_elffile.hpp"

namespace tt::tt_metal::drisc_service {

namespace {

// The low 8 GB of a DRAM tile's NIU space is GDDR, so the DRISC's L1 sits behind a tag.
constexpr uint64_t drisc_l1_noc_offset = 0x2000000000ull;
constexpr uint32_t niu_cfg_0_noc2axi = 1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE;

constexpr uint64_t niu_cfg_0_addr(size_t niu) { return NOC_CFG(NIU_CFG_0) + niu * NOC_INSTANCE_OFFSET; }

}  // namespace

DriscCore::DriscCore(tt::umd::Cluster& cluster, ChipId chip, const tt::umd::CoreCoord& translated) :
    cluster_(&cluster), chip_(chip), translated_(translated) {
    const auto* desc = cluster.get_cluster_description();
    TT_FATAL(desc->get_arch(chip) == tt::ARCH::BLACKHOLE, "drisc_service: chip {} is not Blackhole", chip);
    TT_FATAL(desc->is_chip_mmio_capable(chip), "drisc_service: chip {} is not MMIO-attached", chip);
}

void DriscCore::force_stop(uint8_t noc2axi_mask) const {
    cluster_->assert_risc_reset(chip_, translated_, tt::umd::RiscType::BRISC);
    const std::array<uint32_t, 2> cfg = read_niu_cfg();
    for (size_t niu = 0; niu < cfg.size(); niu++) {
        const uint32_t want =
            ((noc2axi_mask >> niu) & 1u) != 0 ? cfg[niu] | niu_cfg_0_noc2axi : cfg[niu] & ~niu_cfg_0_noc2axi;
        if (want != cfg[niu]) {
            cluster_->write_to_device_reg(&want, sizeof(want), chip_, translated_, niu_cfg_0_addr(niu));
        }
    }
}

void DriscCore::load(const ll_api::ElfFile& image) const {
    struct Placement {
        uint32_t addr, lo, hi;
    };
    // As tt-metal's firmware loader: text goes to the firmware window, and the local-memory data image to the L1
    // scratch do_crt1 copies it from.
    const auto place = [](uint32_t address) {
        if ((address & MEM_LOCAL_BASE) == MEM_LOCAL_BASE) {
            constexpr uint32_t lo = MEM_DRISC_INIT_LOCAL_L1_BASE_SCRATCH;
            return Placement{(address & ~MEM_LOCAL_BASE) + lo, lo, lo + MEM_DRISC_INIT_LOCAL_L1_SCRATCH_SIZE};
        }
        return Placement{address, MEM_DRISC_FIRMWARE_BASE, MEM_DRISC_MAP_END};
    };
    for (const auto& seg : image.GetSegments()) {
        const Placement p = place(seg.address);
        TT_FATAL(
            p.addr >= p.lo && p.addr + seg.contents.size_bytes() <= p.hi,
            "drisc_service: segment at {:#x} ({} B) lies outside its L1 window [{:#x}, {:#x})",
            seg.address,
            seg.contents.size_bytes(),
            p.lo,
            p.hi);
    }
    const uint32_t entry = MEM_DRISC_FIRMWARE_BASE;
    cluster_->write_to_device_reg(&entry, sizeof(entry), chip_, translated_, DRISC_RESET_PC);
    for (const auto& seg : image.GetSegments()) {
        if (!seg.contents.empty()) {
            write_l1(place(seg.address).addr, seg.contents.data(), static_cast<uint32_t>(seg.contents.size_bytes()));
        }
    }
}

void DriscCore::start() const {
    cluster_->l1_membar(chip_, {translated_});
    cluster_->deassert_risc_reset(chip_, translated_, tt::umd::RiscType::BRISC);
}

void DriscCore::write_l1(uint32_t addr, const void* data, uint32_t size) const {
    cluster_->write_to_device_reg(data, size, chip_, translated_, drisc_l1_noc_offset + addr);
}

void DriscCore::read_l1(uint32_t addr, void* data, uint32_t size) const {
    cluster_->read_from_device_reg(data, chip_, translated_, drisc_l1_noc_offset + addr, size);
}

uint32_t DriscCore::read_l1_u32(uint32_t addr) const {
    uint32_t value = 0;
    read_l1(addr, &value, sizeof(value));
    return value;
}

void DriscCore::write_l1_u32(uint32_t addr, uint32_t value) const { write_l1(addr, &value, sizeof(value)); }

std::unique_ptr<tt::umd::IoWindow> DriscCore::map_l1(tt::umd::HostMemoryCaching caching, uint32_t size) const {
    try {
        return cluster_->create_io_window(
            chip_,
            translated_,
            drisc_l1_noc_offset,
            tt::umd::HostIoWindowConfig{caching, size},
            tt::umd::IoOrdering::Strict,
            std::nullopt,
            tt::umd::WindowFlags::UnicastWrite);
    } catch (const std::exception& e) {
        log_debug(tt::LogMetal, "drisc_service: no L1 window on {}: {}", translated_.str(), e.what());
        return nullptr;
    }
}

std::array<uint32_t, 2> DriscCore::read_niu_cfg() const {
    std::array<uint32_t, 2> cfg{};
    for (size_t niu = 0; niu < cfg.size(); niu++) {
        cluster_->read_from_device_reg(&cfg[niu], chip_, translated_, niu_cfg_0_addr(niu), sizeof(uint32_t));
    }
    return cfg;
}

ExitWrite::ExitWrite(const DriscCore& core, uint32_t addr, uint32_t value) :
    cleanup_(
        core.cluster().get_tt_device(core.chip())->get_pci_device()->get_device_num(),
        static_cast<uint8_t>(core.translated().x),
        static_cast<uint8_t>(core.translated().y),
        0,
        drisc_l1_noc_offset + addr,
        value) {}

}  // namespace tt::tt_metal::drisc_service
