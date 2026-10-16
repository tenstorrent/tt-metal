// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <fmt/format.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/host_api.hpp>
#include "umd/device/types/arch.hpp"
#include "umd/device/types/core_coordinates.hpp"

namespace tt::tt_metal::mimir_test {

inline constexpr uint32_t kCcesPerMimir = 2;
inline constexpr uint32_t kHartsPerCce = 8;
// Four instances cover the maximum CCE count of any descriptor; each instance skips when the descriptor has fewer CCEs.
inline constexpr uint32_t kMaxCces = 4;

inline uint32_t mimir_index_for_cce(uint32_t cce_index) { return cce_index / kCcesPerMimir; }

inline uint32_t location_for_cce(uint32_t cce_index) { return cce_index % kCcesPerMimir; }

inline uint32_t cce_index_for(uint32_t mimir_index, uint32_t location) {
    return mimir_index * kCcesPerMimir + location;
}

inline std::string cce_param_name(const ::testing::TestParamInfo<uint32_t>& info) {
    return fmt::format("Cce{}", info.param);
}

inline uint32_t num_cces_from_soc(const metal_SocDescriptor& soc_desc) {
    return static_cast<uint32_t>(soc_desc.get_num_dram_channels() * soc_desc.get_grid_size(tt::CoreType::DRAM).y);
}

inline const metal_SocDescriptor& soc_desc_of(const IDevice* device) {
    return MetalContext::instance().get_cluster().get_soc_desc(device->id());
}

// The runtime's descriptor. MetalContext::instance() opens the cluster on first use, so a skip decided here runs
// connected.
inline const metal_SocDescriptor& cluster_soc_desc(ChipId chip = 0) {
    return MetalContext::instance().get_cluster().get_soc_desc(chip);
}

// CCE locations per Mimir: the DRAM grid height, as num_cces_from_soc uses it.
inline uint32_t cces_per_mimir(const metal_SocDescriptor& soc_desc) {
    return static_cast<uint32_t>(soc_desc.get_grid_size(tt::CoreType::DRAM).y);
}

// kCcesPerMimir/kHartsPerCce stay constexpr for std::array sizes, gtest parameter lists and the index helpers.
// Checked once per process against the runtime's DRAM cores and the HAL hart count.
inline void check_cce_constants_once(const metal_SocDescriptor& soc_desc) {
    static const bool checked = [&soc_desc] {
        const size_t runtime_cces = soc_desc.get_metal_dram_cores(CoordSystem::TRANSLATED).size();
        TT_FATAL(
            runtime_cces == soc_desc.get_num_dram_channels() * kCcesPerMimir,
            "runtime has {} DRAM cores for {} Mimirs; tests assume {} CCEs per Mimir",
            runtime_cces,
            soc_desc.get_num_dram_channels(),
            kCcesPerMimir);
        TT_FATAL(
            cces_per_mimir(soc_desc) == kCcesPerMimir,
            "descriptor has {} CCEs per Mimir; tests assume {}",
            cces_per_mimir(soc_desc),
            kCcesPerMimir);
        const uint32_t harts = MetalContext::instance().hal().get_num_risc_processors(HalProgrammableCoreType::DRAM);
        TT_FATAL(harts == kHartsPerCce, "HAL reports {} harts per CCE; tests assume {}", harts, kHartsPerCce);
        return true;
    }();
    (void)checked;
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

// Each Mimir has one view per coordinate, in location order, so CCE 0 uses the first and CCE 1 the second.
inline uint32_t local_dram_view_for_cce(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    const std::vector<uint32_t> views = dram_views_for_channel(soc_desc, mimir_index_for_cce(cce_index));
    const uint32_t location = location_for_cce(cce_index);
    TT_FATAL(
        location < views.size(),
        "CCE {} location {} has no DRAM view on mimir {}",
        cce_index,
        location,
        mimir_index_for_cce(cce_index));
    return views[location];
}

// Metal's logical DRAM coordinate is {view, index into that view's endpoints}, not {channel, location}.
// Launching CCE n at {n/2, n%2} would be wrong: that pair is view 1, whose endpoints are Mimir 0's
// two CCEs, so CCE 2 and CCE 3 would be dispatched onto CCE 1 and CCE 0.
inline CoreCoord logical_cce_core(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    const uint32_t view = local_dram_view_for_cce(soc_desc, cce_index);
    return soc_desc.get_logical_dram_core_for_subchannel(
        static_cast<int>(view), static_cast<int>(location_for_cce(cce_index)));
}

// The start of a Mimir is the view at offset 0 on that channel. That is not view index == channel:
// view 1 is still Mimir 0, 4 GiB in.
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

inline CoreCoord translated_dram_core(const metal_SocDescriptor& soc_desc, uint32_t cce_index) {
    const auto core = soc_desc.translate_coord_to(
        tt::umd::CoreCoord(
            mimir_index_for_cce(cce_index), location_for_cce(cce_index), CoreType::DRAM, CoordSystem::LOGICAL),
        CoordSystem::TRANSLATED);
    return {core.x, core.y};
}

// The runtime's logical-to-virtual mapping, as IDevice::virtual_core_from_logical_core uses it.
inline CoreCoord cce_virtual_core(const Cluster& cluster, ChipId chip, uint32_t cce_index) {
    return cluster.get_virtual_coordinate_from_logical_coordinates(
        chip, logical_cce_core(cluster.get_soc_desc(chip), cce_index), CoreType::DRAM);
}

}  // namespace tt::tt_metal::mimir_test
