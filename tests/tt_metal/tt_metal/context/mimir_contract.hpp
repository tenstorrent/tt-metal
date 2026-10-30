// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for Mimir/CCE tests. Contract tests call these instead of a raw Cluster or a
// chippy transport. Case sizes come from the selected package descriptor.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "hostdev/cce_spa_routes.h"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/slow_dispatch.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "umd/device/soc_arch_descriptor.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal {
namespace mimir_contract {

inline constexpr uint64_t kCceSramTestOffset = 0x800;
inline constexpr uint32_t kDramOffset = 0x800;
inline constexpr uint64_t kGddrSlotStride = 1ull << 30;
// Mimir/MMK emu GDDR is an 8GiB SPA with a 1GiB sparse backing store. Unique 32KiB pages persist
// until reset.
inline constexpr uint64_t kEmuGddrBackingBudget = 1ull << 30;
inline constexpr uint64_t kEmuGddrBackingPage = 32ull << 10;

struct GddrBackingTracker {
    std::set<uint64_t> pages;

    void note(uint32_t mimir_index, uint64_t address, uint64_t size) {
        if (size == 0) {
            return;
        }
        const uint64_t first = address / kEmuGddrBackingPage;
        const uint64_t last = (address + size - 1) / kEmuGddrBackingPage;
        for (uint64_t page = first; page <= last; ++page) {
            pages.insert((static_cast<uint64_t>(mimir_index) << 32) | page);
        }
        const uint64_t used = pages.size() * kEmuGddrBackingPage;
        ASSERT_LE(used, kEmuGddrBackingBudget)
            << "EMU GDDR unique backing " << used << " B exceeds the 1GiB sparse budget";
    }
};

inline thread_local GddrBackingTracker* t_gddr_backing = nullptr;

struct GddrBackingScope {
    GddrBackingTracker tracker;
    GddrBackingScope() { t_gddr_backing = &tracker; }
    ~GddrBackingScope() { t_gddr_backing = nullptr; }
};

// The descriptor the runtime will open. A chippy server names its own file. With no server,
// TT_METAL_SOC_DESC names the package for a transport that discovers the device, such as PCIe.
inline const char* selected_soc_desc_path() {
    if (std::getenv("TT_METAL_GRENDEL_JTAG_SERVER") != nullptr) {
        return std::getenv("TT_METAL_GRENDEL_JTAG_SOC_DESC");
    }
    if (std::getenv("TT_METAL_EMU_SERVER") != nullptr) {
        return std::getenv("TT_METAL_EMU_SOC_DESC");
    }
    return std::getenv("TT_METAL_SOC_DESC");
}

inline bool device_transport_configured() {
    if (std::getenv("TT_METAL_EMU_SERVER") != nullptr || std::getenv("TT_METAL_GRENDEL_JTAG_SERVER") != nullptr) {
        return true;
    }
    const char* path = std::getenv("TT_METAL_SOC_DESC");
    return path != nullptr && path[0] != '\0';
}

// The runtime treats any value of this variable, including 0, as slow dispatch. Unset is fast dispatch.
inline bool slow_dispatch_requested() { return std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr; }

// The package file, not the transport. Opening the device is what makes a skipped case slow, so
// callers decide from this before opening it.
inline const metal_SocDescriptor& configured_soc_desc() {
    static const metal_SocDescriptor desc = [] {
        const char* path = selected_soc_desc_path();
        TT_FATAL(path != nullptr, "No package descriptor selected");
        auto arch = std::make_shared<tt::umd::SocArchDescriptor>(std::string(path));
        return metal_SocDescriptor(tt::umd::SocDescriptor(std::move(arch)), tt::BoardType::UNKNOWN);
    }();
    return desc;
}

inline bool package_has_programmable_dram() {
    const char* path = selected_soc_desc_path();
    if (path == nullptr) {
        return false;
    }
    return configured_soc_desc().has_programmable_dram_views();
}

inline bool package_has_tensix_cores() {
    if (selected_soc_desc_path() == nullptr) {
        return false;
    }
    return !configured_soc_desc().get_cores(tt::CoreType::TENSIX).empty();
}

// GTEST_SKIP returns from the function that invokes it, so this has to be a macro. Contract tests
// run for any selected package that declares programmable DRAM cores and has a transport to open it.
#define skip_unless_programmable_dram_package()                                                                     \
    do {                                                                                                            \
        if (!::tt::tt_metal::mimir_contract::package_has_programmable_dram()) {                                     \
            GTEST_SKIP() << "Selected package has no programmable DRAM cores.";                                     \
        }                                                                                                           \
        if (!::tt::tt_metal::mimir_contract::device_transport_configured()) {                                       \
            GTEST_SKIP() << "Selected package has programmable DRAM cores, but no device transport is configured."; \
        }                                                                                                           \
        if (!::tt::tt_metal::mimir_contract::package_has_tensix_cores() &&                                          \
            !::tt::tt_metal::mimir_contract::slow_dispatch_requested()) {                                           \
            GTEST_SKIP() << "This Mimir-only package has no Tensix cores for fast dispatch. Set "                   \
                            "TT_METAL_SLOW_DISPATCH_MODE=1.";                                                       \
        }                                                                                                           \
    } while (0)

inline uint32_t locations_per_channel(const metal_SocDescriptor& soc_desc) {
    return std::max<uint32_t>(static_cast<uint32_t>(soc_desc.get_grid_size(tt::CoreType::DRAM).y), 1u);
}

inline uint32_t mimir_index_for_cce(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    return cce_index / locations_per_channel(soc_desc);
}

inline uint32_t location_for_cce(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    return cce_index % locations_per_channel(soc_desc);
}

inline uint32_t mimir_index_for_cce(uint32_t cce_index) {
    return mimir_index_for_cce(configured_soc_desc(), cce_index);
}

inline uint32_t location_for_cce(uint32_t cce_index) { return location_for_cce(configured_soc_desc(), cce_index); }

inline uint32_t cce_index_for(const metal_SocDescriptor& soc_desc, uint32_t mimir_index, uint32_t location) {
    return mimir_index * locations_per_channel(soc_desc) + location;
}

inline uint32_t cce_index_for(uint32_t mimir_index, uint32_t location) {
    return cce_index_for(configured_soc_desc(), mimir_index, location);
}

inline uint32_t num_cces_from_soc(const metal_SocDescriptor& soc_desc) {
    return static_cast<uint32_t>(soc_desc.get_num_dram_channels() * locations_per_channel(soc_desc));
}

inline const std::vector<uint32_t>& cce_indices() {
    static const std::vector<uint32_t> indices = [] {
        std::vector<uint32_t> built;
        if (selected_soc_desc_path() == nullptr) {
            built.push_back(0);
            return built;
        }
        const uint32_t cce_count = num_cces_from_soc(configured_soc_desc());
        built.reserve(cce_count);
        for (uint32_t cce_index = 0; cce_index < cce_count; ++cce_index) {
            built.push_back(cce_index);
        }
        return built;
    }();
    return indices;
}

inline const std::vector<uint32_t>& dram_view_indices() {
    static const std::vector<uint32_t> indices = [] {
        std::vector<uint32_t> built;
        if (selected_soc_desc_path() == nullptr) {
            built.push_back(0);
            return built;
        }
        const uint32_t view_count = static_cast<uint32_t>(configured_soc_desc().get_num_dram_views());
        built.reserve(view_count);
        for (uint32_t view = 0; view < view_count; ++view) {
            built.push_back(view);
        }
        return built;
    }();
    return indices;
}

inline const metal_SocDescriptor& soc_desc_of(const IDevice* device) {
    return MetalContext::instance().get_cluster().get_soc_desc(device->id());
}

inline uint32_t num_cces_on(const IDevice* device) { return num_cces_from_soc(soc_desc_of(device)); }

// One Mimir is one UMD DRAM channel. Device::num_dram_channels() is the Metal view count.
inline uint32_t num_mimirs_on(const IDevice* device) {
    return static_cast<uint32_t>(soc_desc_of(device).get_num_dram_channels());
}

struct GddrSweep {
    uint32_t num_slots = 0;
    uint32_t slot_stride = 0;
};

inline GddrSweep gddr_sweep_from_soc(const IDevice* device) {
    const uint64_t view_size = soc_desc_of(device).dram_view_size;
    // Views of at least 1 GiB are walked in 1 GiB slots. A smaller non-zero view is one slot of
    // that size. The slot must still divide the view's address_offset.
    uint64_t slot_stride = kGddrSlotStride;
    if (view_size == 0 || view_size > std::numeric_limits<uint32_t>::max()) {
        return {};
    }
    if (view_size < kGddrSlotStride) {
        slot_stride = view_size;
    } else if (view_size % kGddrSlotStride != 0) {
        return {};
    }
    return {static_cast<uint32_t>(view_size / slot_stride), static_cast<uint32_t>(slot_stride)};
}

struct GddrTarget {
    uint32_t host_view = 0;
    uint32_t mimir_index = 0;
    uint32_t location = 0;
    // Kernel offsets are mimir-relative. slot_base is the view's address_offset in slots.
    uint32_t slot_base = 0;
    uint32_t num_slots = 0;
    uint32_t slot_stride = 0;
};

struct CcePartition {
    uint32_t cce_index = 0;
    uint32_t partition = 0;
};

inline const std::vector<CcePartition>& cce_partition_params() {
    static const std::vector<CcePartition> params = [] {
        if (selected_soc_desc_path() == nullptr) {
            return std::vector<CcePartition>{{0, 0}};
        }
        const metal_SocDescriptor& soc_desc = configured_soc_desc();
        const uint32_t cce_count = num_cces_from_soc(soc_desc);
        const uint32_t partition_count = static_cast<uint32_t>(soc_desc.get_num_dram_views());
        std::vector<CcePartition> built;
        built.reserve(cce_count * partition_count);
        for (uint32_t cce_index = 0; cce_index < cce_count; ++cce_index) {
            for (uint32_t partition = 0; partition < partition_count; ++partition) {
                built.push_back({cce_index, partition});
            }
        }
        return built;
    }();
    return params;
}

inline std::string cce_param_name(const ::testing::TestParamInfo<uint32_t>& info) {
    return fmt::format("Cce{}", info.param);
}

inline std::string cce_partition_name(const ::testing::TestParamInfo<CcePartition>& info) {
    return fmt::format("Cce{}Partition{}", info.param.cce_index, info.param.partition);
}

inline bool is_cross_mimir_case() {
    const char* suite = ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
    return std::string_view(suite).find("CrossMimir") != std::string_view::npos;
}

inline std::vector<uint32_t> dram_views_for_channel(const metal_SocDescriptor& soc_desc, uint32_t channel) {
    std::vector<uint32_t> views;
    for (uint32_t view = 0; view < soc_desc.get_num_dram_views(); ++view) {
        if (soc_desc.get_channel_for_dram_view(static_cast<int>(view)) == channel) {
            views.push_back(view);
        }
    }
    return views;
}

inline uint32_t local_dram_view_for_cce(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    const std::vector<uint32_t> views = dram_views_for_channel(soc_desc, mimir_index_for_cce(soc_desc, cce_index));
    TT_FATAL(!views.empty(), "Mimir {} has no DRAM view", mimir_index_for_cce(soc_desc, cce_index));
    if (views.size() == 1) {
        return views.front();
    }
    const uint32_t location = location_for_cce(soc_desc, cce_index);
    TT_FATAL(
        location < views.size(),
        "CCE {} location {} has no DRAM view on mimir {}",
        cce_index,
        location,
        mimir_index_for_cce(soc_desc, cce_index));
    return views[location];
}

// Metal's logical DRAM coordinate is {view, index into that view's endpoints}, not {channel, location}.
inline CoreCoord logical_cce_core(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    const uint32_t view = local_dram_view_for_cce(soc_desc, cce_index);
    return soc_desc.get_logical_dram_core_for_subchannel(
        static_cast<int>(view), static_cast<int>(location_for_cce(soc_desc, cce_index)));
}

inline uint32_t dram_view_location(const metal_SocDescriptor& soc_desc, uint32_t view) {
    const uint32_t channel = static_cast<uint32_t>(soc_desc.get_channel_for_dram_view(static_cast<int>(view)));
    const std::vector<uint32_t> views = dram_views_for_channel(soc_desc, channel);
    for (uint32_t location = 0; location < views.size(); ++location) {
        if (views[location] == view) {
            return location;
        }
    }
    TT_FATAL(false, "DRAM view {} is not on channel {}", view, channel);
    return 0;
}

inline bool partition_in_case(const metal_SocDescriptor& soc_desc, uint32_t cce_index, uint32_t partition, bool cross) {
    if (partition >= soc_desc.get_num_dram_views()) {
        return false;
    }
    const bool local =
        soc_desc.get_channel_for_dram_view(static_cast<int>(partition)) == mimir_index_for_cce(soc_desc, cce_index);
    return cross ? !local : local;
}

inline bool cce_partition_case_applies(const CcePartition& param, bool cross) {
    const metal_SocDescriptor& soc_desc = configured_soc_desc();
    if (param.cce_index >= num_cces_from_soc(soc_desc) || soc_desc.get_num_dram_channels() == 0) {
        return false;
    }
    if (cross && soc_desc.get_num_dram_channels() < 2) {
        return false;
    }
    return partition_in_case(soc_desc, param.cce_index, param.partition, cross);
}

inline GddrTarget gddr_target_for_view(const metal_SocDescriptor& soc_desc, uint32_t view, const GddrSweep& sweep) {
    const uint64_t address_offset = soc_desc.get_address_offset(static_cast<int>(view));
    TT_FATAL(
        sweep.slot_stride != 0 && address_offset % sweep.slot_stride == 0,
        "DRAM view {} address_offset {} is not a multiple of the {} B slot",
        view,
        address_offset,
        sweep.slot_stride);
    return {
        view,
        static_cast<uint32_t>(soc_desc.get_channel_for_dram_view(static_cast<int>(view))),
        dram_view_location(soc_desc, view),
        static_cast<uint32_t>(address_offset / sweep.slot_stride),
        sweep.num_slots,
        sweep.slot_stride};
}

inline uint64_t host_slot_offset(const GddrTarget& target, uint32_t slot) {
    return static_cast<uint64_t>(slot) * target.slot_stride;
}

// A unit mesh's DRAM view through the slow-dispatch host API. A write below the allocator's DRAM base is refused.
inline void write_gddr_view(
    distributed::MeshDevice& unit_mesh, uint32_t view, uint64_t address, const std::vector<uint32_t>& data) {
    const metal_SocDescriptor& soc_desc = soc_desc_of(&slow_dispatch::physical_device_from_unit_mesh(unit_mesh));
    const uint32_t size = static_cast<uint32_t>(data.size() * sizeof(uint32_t));
    TT_FATAL(
        view < soc_desc.get_num_dram_views(),
        "DRAM view {} is outside the {} views",
        view,
        soc_desc.get_num_dram_views());
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(), "GDDR view offset 0x{:x} does not fit the host API", address);
    if (t_gddr_backing != nullptr) {
        const uint32_t channel = static_cast<uint32_t>(soc_desc.get_channel_for_dram_view(static_cast<int>(view)));
        t_gddr_backing->note(channel, soc_desc.get_address_offset(static_cast<int>(view)) + address, size);
    }
    TT_FATAL(
        slow_dispatch::WriteToDRAMChannel(
            unit_mesh,
            static_cast<int>(view),
            static_cast<uint32_t>(address),
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(data.data()), size)),
        "write to DRAM view {} offset 0x{:x} failed",
        view,
        address);
}

inline void read_gddr_view(
    distributed::MeshDevice& unit_mesh, uint32_t view, uint64_t address, uint32_t size, std::vector<uint32_t>& data) {
    const metal_SocDescriptor& soc_desc = soc_desc_of(&slow_dispatch::physical_device_from_unit_mesh(unit_mesh));
    TT_FATAL(size % sizeof(uint32_t) == 0, "DRAM read size {} is not a multiple of 4", size);
    TT_FATAL(
        view < soc_desc.get_num_dram_views(),
        "DRAM view {} is outside the {} views",
        view,
        soc_desc.get_num_dram_views());
    TT_FATAL(
        address <= std::numeric_limits<uint32_t>::max(), "GDDR view offset 0x{:x} does not fit the host API", address);
    if (t_gddr_backing != nullptr) {
        const uint32_t channel = static_cast<uint32_t>(soc_desc.get_channel_for_dram_view(static_cast<int>(view)));
        t_gddr_backing->note(channel, soc_desc.get_address_offset(static_cast<int>(view)) + address, size);
    }
    data.assign(size / sizeof(uint32_t), 0);
    TT_FATAL(
        slow_dispatch::ReadFromDRAMChannel(
            unit_mesh, static_cast<int>(view), static_cast<uint32_t>(address), size, data),
        "read from DRAM view {} offset 0x{:x} failed",
        view,
        address);
}

inline uint32_t base_dram_view_for_mimir(const metal_SocDescriptor& soc_desc, uint32_t mimir_index) {
    const std::vector<uint32_t> views = dram_views_for_channel(soc_desc, mimir_index);
    TT_FATAL(!views.empty(), "Mimir {} has no DRAM view", mimir_index);
    for (uint32_t view : views) {
        if (soc_desc.get_address_offset(static_cast<int>(view)) == 0) {
            return view;
        }
    }
    TT_FATAL(false, "Mimir {} has no DRAM view at offset 0", mimir_index);
    return views.front();
}

inline void write_mimir_gddr(
    distributed::MeshDevice& unit_mesh, uint32_t mimir_index, uint64_t address, const std::vector<uint32_t>& data) {
    const metal_SocDescriptor& soc_desc = soc_desc_of(&slow_dispatch::physical_device_from_unit_mesh(unit_mesh));
    write_gddr_view(unit_mesh, base_dram_view_for_mimir(soc_desc, mimir_index), address, data);
}

inline void read_mimir_gddr(
    distributed::MeshDevice& unit_mesh,
    uint32_t mimir_index,
    uint64_t address,
    uint32_t size,
    std::vector<uint32_t>& data) {
    const metal_SocDescriptor& soc_desc = soc_desc_of(&slow_dispatch::physical_device_from_unit_mesh(unit_mesh));
    read_gddr_view(unit_mesh, base_dram_view_for_mimir(soc_desc, mimir_index), address, size, data);
}

// Host view of one CCE's SRAM. The address is the HAL NOC address (base plus offset), which UMD
// resolves. Callers do not form a chiplet-local AXI address.
inline void write_cce_sram(
    IDevice* device, CoreCoord logical_dram_core, uint64_t noc_addr, const void* data, uint32_t size) {
    const CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    MetalContext::instance().get_cluster().write_core(data, size, {device->id(), virtual_core}, noc_addr);
}

inline void read_cce_sram(IDevice* device, CoreCoord logical_dram_core, uint64_t noc_addr, void* data, uint32_t size) {
    const CoreCoord virtual_core = device->virtual_core_from_logical_core(logical_dram_core, CoreType::DRAM);
    MetalContext::instance().get_cluster().read_core(data, size, {device->id(), virtual_core}, noc_addr);
}

inline uint64_t cce_sram_noc_addr(uint64_t offset) {
    return MetalContext::instance().hal().get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BASE) +
           offset;
}

// The SPA routes device initialization wrote to one CCE, right after its bank-to-NOC staging range.
inline CceSpaRoutes read_cce_spa_routes(IDevice* device, uint32_t cce_index) {
    const Hal& hal = MetalContext::instance().hal();
    const uint64_t routes_noc_addr =
        hal.get_dev_noc_addr(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BANK_TO_NOC_SCRATCH) +
        hal.get_dev_size(HalProgrammableCoreType::DRAM, HalL1MemAddrType::BANK_TO_NOC_SCRATCH);
    CceSpaRoutes routes{};
    read_cce_sram(device, logical_cce_core(soc_desc_of(device), cce_index), routes_noc_addr, &routes, sizeof(routes));
    return routes;
}

// The SPA a CCE kernel's cce_gddr_address() issues with these routes.
inline uint64_t cce_gddr_address(const CceSpaRoutes& routes, uint32_t mimir_index, uint64_t offset) {
    TT_FATAL(mimir_index < CCE_SPA_ROUTES_MAX_INSTANCES, "Mimir {} has no SPA route", mimir_index);
    const uint32_t port = routes.gddr_d2d[mimir_index];
    TT_FATAL(port < std::size(routes.gddr_d2d_base), "Mimir {} route names D2D port {}", mimir_index, port);
    return routes.gddr_d2d_base[port] + static_cast<uint64_t>(mimir_index) * routes.gddr_instance_stride + offset;
}

// The SPA a CCE kernel's cce_sram_address() issues with these routes.
inline uint64_t cce_sram_address(const CceSpaRoutes& routes, uint32_t cce_index, uint64_t offset) {
    return routes.sram_spa_base + static_cast<uint64_t>(cce_index) * routes.sram_tile_stride + offset;
}

// UMD logical DRAM coordinate {channel, location}. Backend checks that do not go through a Metal
// device use this. Contract tests use logical_cce_core().
inline CoreCoord translated_dram_core(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    const auto core = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(
            mimir_index_for_cce(soc_desc, cce_index),
            location_for_cce(soc_desc, cce_index),
            CoreType::DRAM,
            CoordSystem::LOGICAL),
        CoordSystem::TRANSLATED);
    return {core.x, core.y};
}

}  // namespace mimir_contract
}  // namespace tt::tt_metal
