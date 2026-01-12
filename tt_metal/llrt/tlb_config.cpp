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
#include "llrt/metal_soc_descriptor.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "common/tt_backend_api_types.hpp"
#include <umd/device/arch/blackhole_implementation.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <umd/device/arch/wormhole_implementation.hpp>

namespace ll_api {

namespace wormhole {

int32_t get_static_tlb_size() {
    return 1 << 20;
}

}  // namespace wormhole

namespace blackhole {

static constexpr uint32_t NUM_PORTS_PER_DRAM_CHANNEL = 3;
static constexpr uint32_t NUM_DRAM_CHANNELS = 8;

int32_t get_static_tlb_size() {
    return 2 * (1 << 20);
}

// Returns last port of dram channel passed as the argument to align with dram_preferred_worker_endpoint
// This core will be used for configuring 4GB TLB.
tt_xy_pair ddr_to_noc0(unsigned i) {
    return tt::umd::blackhole::DRAM_LOCATIONS[(NUM_PORTS_PER_DRAM_CHANNEL * i) + (NUM_PORTS_PER_DRAM_CHANNEL - 1)];
}

}  // namespace blackhole

void configure_static_tlbs(
    tt::ARCH arch, tt::ChipId mmio_device_id, const metal_SocDescriptor& sdesc, tt::umd::Cluster& device_driver) {
    using get_static_tlb_size_ptr = std::int32_t (*)();
    get_static_tlb_size_ptr get_static_tlb_size;

    // Need to set these values based on arch because UMD does not expose architecture_implementation
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0:
            get_static_tlb_size = wormhole::get_static_tlb_size;
            break;
        case tt::ARCH::BLACKHOLE:
            get_static_tlb_size = blackhole::get_static_tlb_size;
            break;
        default: TT_THROW("Configuring static TLBs is not supported for {}", tt::get_string(arch));
    }

    std::int32_t address = 0;
    // Setup static TLBs for all worker cores.
    for (const tt::umd::CoreCoord& core : sdesc.get_cores(tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED)) {
        // TODO
        // Note: see issue #10107
        // Strict is less performant than Posted, however, metal doesn't presently
        // use this on a perf path and the launch_msg "kernel config" needs to
        // arrive prior to the "go" message during device init and slow dispatch
        // Revisit this when we have a more flexible UMD api
        device_driver.configure_tlb(mmio_device_id, core, get_static_tlb_size(), address, tt::umd::tlb_data::Strict);
    }
    // Setup static TLBs for all eth cores
    for (const  tt::umd::CoreCoord& core : sdesc.get_cores(tt::CoreType::ETH, tt::CoordSystem::TRANSLATED)) {
        device_driver.configure_tlb(mmio_device_id, core, get_static_tlb_size(), address, tt::umd::tlb_data::Strict);
    }

    if (arch == tt::ARCH::BLACKHOLE) {
        // Setup static 4GB tlbs for DRAM cores.
        uint32_t dram_addr = 0;
        for (std::uint32_t dram_channel = 0; dram_channel < blackhole::NUM_DRAM_CHANNELS; dram_channel++) {
            tt::umd::CoreCoord dram_core =
                tt::umd::CoreCoord(blackhole::ddr_to_noc0(dram_channel), tt::CoreType::DRAM, tt::CoordSystem::NOC0);
            device_driver.configure_tlb(mmio_device_id, dram_core, 4ULL * (1ULL << 30), dram_addr, tt::umd::tlb_data::Posted);
        }
    }
}

}  // namespace ll_api
