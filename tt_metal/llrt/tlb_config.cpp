// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tlb_config.hpp"

#include <tt_stl/assert.hpp>
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core_coord.hpp"
#include "metal_soc_descriptor.h"
#include "tt_backend_api_types.hpp"
#include <umd/device/arch/blackhole_implementation.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/arch/wormhole_implementation.hpp>

namespace ll_api {

namespace wormhole {

static constexpr uint32_t TENSIX_STATIC_TLB_START = 16;
static constexpr uint32_t ETH_STATIC_TLB_START = 0;

int32_t get_static_tlb_index_logical_eth(CoreCoord target) {
    if (target.y >= tt::umd::wormhole::ETH_LOCATIONS.size()) {
        TT_THROW("Invalid ETH core coordinate for generating static TLB index");
    }
    int32_t tlb_index = ETH_STATIC_TLB_START + target.y;
    return tlb_index;
}

int32_t get_static_tlb_index_logical_tensix(CoreCoord target) {
    if ((target.x >= tt::umd::wormhole::T6_X_LOCATIONS.size()) ||
        (target.y >= tt::umd::wormhole::T6_Y_LOCATIONS.size())) {
        TT_THROW("Invalid TENSIX core coordinate for generating static TLB index");
    }
    return TENSIX_STATIC_TLB_START + (target.y * 8) + target.x;
}

int32_t get_static_tlb_index(tt::umd::CoreCoord target) {
    if (target.coord_system != CoordSystem::LOGICAL) {
        TT_THROW("Invalid TENSIX core coordinate for generating static TLB index");
    }
    switch (target.core_type) {
        case CoreType::ETH: return get_static_tlb_index_logical_eth({target.x, target.y});
        case CoreType::TENSIX: return get_static_tlb_index_logical_tensix({target.x, target.y});
        default: TT_THROW("Invalid core type for generating static TLB index");
    }
}

}  // namespace wormhole

namespace blackhole {

static constexpr uint32_t NUM_PORTS_PER_DRAM_CHANNEL = 3;
static constexpr uint32_t NUM_DRAM_CHANNELS = 8;
// Values taken from blackhole.py in `src/t6ifc/t6py/packages/tenstorrent/chip/blackhole.py`
static constexpr uint32_t ETH_STATIC_TLB_START = 0;
static constexpr uint32_t TENSIX_STATIC_TLB_START = 38;

int32_t get_static_tlb_index_logical_eth(CoreCoord target) {
    if (target.y >= tt::umd::blackhole::ETH_LOCATIONS.size()) {
        TT_THROW("Invalid ETH core coordinate for generating static TLB index");
    }
    int tlb_index = ETH_STATIC_TLB_START + target.y;
    return tlb_index;
}

int32_t get_static_tlb_index_logical_tensix(CoreCoord target) {
    if ((target.x >= tt::umd::blackhole::T6_X_LOCATIONS.size()) ||
        (target.y >= tt::umd::blackhole::T6_Y_LOCATIONS.size())) {
        TT_THROW("Invalid TENSIX core coordinate for generating static TLB index");
    }
    return TENSIX_STATIC_TLB_START + (target.y * 14) + target.x;
}

int32_t get_static_tlb_index_logical_dram(CoreCoord target) {
    auto dram_tlb_index = target.x;
    if (dram_tlb_index > 7) {
        TT_THROW("Invalid DRAM core coordinate for generating static TLB index");
    }
    // Each DRAM channel has 3 ports, and the logical x coordinate directly corresponds to the channel.
    // Therefore, the x coordinate alone is sufficient for determining the TLB mapping.
    return tt::umd::blackhole::TLB_BASE_INDEX_4G + (dram_tlb_index);
}

int32_t get_static_tlb_index(tt::umd::CoreCoord target) {
    if (target.coord_system != CoordSystem::LOGICAL) {
        TT_THROW("Invalid TENSIX core coordinate for generating static TLB index");
    }
    switch (target.core_type) {
        case CoreType::ETH: return get_static_tlb_index_logical_eth({target.x, target.y});
        case CoreType::TENSIX: return get_static_tlb_index_logical_tensix({target.x, target.y});
        case CoreType::DRAM: return get_static_tlb_index_logical_dram({target.x, target.y});
        default: TT_THROW("Invalid core type for generating static TLB index");
    }
}

// Returns last port of dram channel passed as the argument to align with dram_preferred_worker_endpoint
// This core will be used for configuring 4GB TLB.
tt_xy_pair ddr_to_noc0(unsigned i) {
    return tt::umd::blackhole::DRAM_LOCATIONS[(NUM_PORTS_PER_DRAM_CHANNEL * i) + (NUM_PORTS_PER_DRAM_CHANNEL - 1)];
}

}  // namespace blackhole

void configure_static_tlbs(
    tt::ARCH arch, ChipId mmio_device_id, const metal_SocDescriptor& sdesc, tt::umd::Cluster& device_driver) {
    using get_static_tlb_index_ptr = std::int32_t (*)(tt::umd::CoreCoord);
    get_static_tlb_index_ptr get_static_tlb_index;

    const uint32_t dynamic_tlb_count = 16;
    uint32_t dynamic_tlb_base_index, dynamic_tlb_16m_size, dram_channel_0_peer2peer_region_start, dram_channel_0_x,
        dram_channel_0_y;

    // Need to set these values based on arch because UMD does not expose architecture_implementation
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0:
            get_static_tlb_index = wormhole::get_static_tlb_index;
            dynamic_tlb_base_index = tt::umd::wormhole::DYNAMIC_TLB_BASE_INDEX;
            dynamic_tlb_16m_size = tt::umd::wormhole::DYNAMIC_TLB_16M_SIZE;
            dram_channel_0_peer2peer_region_start = tt::umd::wormhole::DRAM_CHANNEL_0_PEER2PEER_REGION_START;
            dram_channel_0_x = tt::umd::wormhole::DRAM_CHANNEL_0_X;
            dram_channel_0_y = tt::umd::wormhole::DRAM_CHANNEL_0_Y;
            break;
        case tt::ARCH::BLACKHOLE:
            get_static_tlb_index = blackhole::get_static_tlb_index;
            dynamic_tlb_base_index = tt::umd::blackhole::DYNAMIC_TLB_BASE_INDEX;
            dynamic_tlb_16m_size = 0;
            dram_channel_0_peer2peer_region_start = tt::umd::blackhole::DRAM_CHANNEL_0_PEER2PEER_REGION_START;
            dram_channel_0_x = tt::umd::blackhole::DRAM_CHANNEL_0_X;
            dram_channel_0_y = tt::umd::blackhole::DRAM_CHANNEL_0_Y;
            break;
        default: TT_THROW("Configuring static TLBs is not supported for {}", tt::get_string(arch));
    }

    std::int32_t address = 0;
    // Setup static TLBs for all worker cores
    for (const tt::umd::CoreCoord& core : sdesc.get_cores(CoreType::TENSIX, tt::umd::CoordSystem::LOGICAL)) {
        auto tlb_index = get_static_tlb_index(core);
        // TODO
        // Note: see issue #10107
        // Strict is less performant than Posted, however, metal doesn't presently
        // use this on a perf path and the launch_msg "kernel config" needs to
        // arrive prior to the "go" message during device init and slow dispatch
        // Revisit this when we have a more flexible UMD api
        device_driver.configure_tlb(mmio_device_id, core, tlb_index, address, TLB_DATA::Strict);
    }
    // Setup static TLBs for all eth cores
    for (const tt::umd::CoreCoord& core : sdesc.get_cores(CoreType::ETH, tt::umd::CoordSystem::LOGICAL)) {
        auto tlb_index = get_static_tlb_index(core);
        device_driver.configure_tlb(mmio_device_id, core, tlb_index, address, TLB_DATA::Strict);
    }

    // TODO (#9932): Remove workaround for BH
    if (arch != tt::ARCH::BLACKHOLE) {
        // Setup static TLBs for MMIO mapped data space
        uint64_t peer_dram_offset = dram_channel_0_peer2peer_region_start;
        for (uint32_t tlb_id = dynamic_tlb_base_index; tlb_id < dynamic_tlb_base_index + dynamic_tlb_count; tlb_id++) {
            device_driver.configure_tlb(
                mmio_device_id,
                tt::umd::CoreCoord(dram_channel_0_x, dram_channel_0_y, CoreType::DRAM, CoordSystem::NOC0),
                tlb_id,
                peer_dram_offset);
            // Align address space of 16MB TLB to 16MB boundary
            peer_dram_offset += dynamic_tlb_16m_size;
        }
    } else {
        // Setup static 4GB tlbs for DRAM cores
        uint32_t dram_addr = 0;
        for (std::uint32_t dram_channel = 0; dram_channel < blackhole::NUM_DRAM_CHANNELS; dram_channel++) {
            tt::umd::CoreCoord dram_core =
                tt::umd::CoreCoord(blackhole::ddr_to_noc0(dram_channel), CoreType::DRAM, CoordSystem::NOC0);
            auto tlb_index = tt::umd::blackhole::TLB_COUNT_2M + dram_channel;
            device_driver.configure_tlb(mmio_device_id, dram_core, tlb_index, dram_addr, TLB_DATA::Posted);
        }
    }
}

}  // namespace ll_api
