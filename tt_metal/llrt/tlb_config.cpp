// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tlb_config.hpp"

#include <tt_stl/assert.hpp>
#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

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

uint64_t get_static_tlb_size() {
    return 1ULL << 20;
}

}  // namespace wormhole

namespace blackhole {

static constexpr uint32_t NUM_PORTS_PER_DRAM_CHANNEL = 3;
static constexpr uint32_t NUM_DRAM_CHANNELS = 8;

uint64_t get_static_tlb_size() {
    return 2 * (1ULL << 20);
}

// Returns last port of dram channel passed as the argument to align with dram_preferred_worker_endpoint
// This core will be used for configuring 4GB TLB.
tt_xy_pair ddr_to_noc0(unsigned i) {
    return tt::umd::blackhole::DRAM_LOCATIONS[(NUM_PORTS_PER_DRAM_CHANNEL * i) + (NUM_PORTS_PER_DRAM_CHANNEL - 1)];
}

}  // namespace blackhole

namespace quasar {

uint64_t get_static_tlb_size() { return 4ULL * (1ULL << 30); }

}  // namespace quasar

void configure_static_tlbs(
    tt::ARCH arch, tt::ChipId mmio_device_id, const metal_SocDescriptor& sdesc, tt::umd::Cluster& device_driver) {
    using get_static_tlb_size_ptr = uint64_t (*)();
    get_static_tlb_size_ptr get_static_tlb_size;

    // Need to set these values based on arch because UMD does not expose architecture_implementation
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0:
            get_static_tlb_size = wormhole::get_static_tlb_size;
            break;
        case tt::ARCH::BLACKHOLE:
            get_static_tlb_size = blackhole::get_static_tlb_size;
            break;
        case tt::ARCH::QUASAR: get_static_tlb_size = quasar::get_static_tlb_size; break;
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
    // Static TLBs for L2CPU tiles on Blackhole, anchored at the LIM base.
    //
    // Unlike Tensix/ETH (anchored at NOC address 0 because L1 lives in
    // [0, 2 MiB)), L2CPU LIM starts at 0x08000000, so a window anchored
    // at 0 would never cover any useful LIM address. We anchor each
    // L2CPU TLB at 0x08000000 instead, giving it coverage of
    // [0x08000000, 0x08200000) -- enough for the H2D config buffer,
    // D2H config buffer, and an H2D data FIFO whose end is below
    // 0x08200000.
    //
    // We use a 2 MiB window (get_static_tlb_size()), NOT a 4 GiB one.
    // Blackhole has only 8 × 4 GiB TLBs total (see UMD's
    // TLB_COUNT_4G[BLACKHOLE] = 8 in tt_kmd_lib.c) and the DRAM block
    // below consumes all 8 (one per channel). 2 MiB TLBs have 202 slots
    // and Tensix+ETH use < ~150, so 4 more for L2CPU is well within budget
    // and DRAM access is unaffected.
    //
    // Because the window is anchored at 0x08000000 rather than 0, the
    // H2DSocket/D2HSocket L2CPU pcie_writer must convert absolute LIM
    // addresses to window-relative offsets via
    // (device_addr - tlb->get_base_address()) before write_block. Tensix
    // gets away without that subtraction only because its base is 0.
    //
    // get_cores(TRANSLATED) returns only unharvested L2CPU tiles; on
    // Blackhole the set is {(8,3),(8,5),(8,7),(8,9)} with TRANSLATED == NOC0.
    //
    // Ordering is Strict (not the Posted DRAM uses): socket FIFO
    // bookkeeping (bytes_sent / bytes_acked / read_ptr) is order-sensitive
    // across the L2CPU NOC pipeline, and the FIFO data path already runs
    // at PCIe-limited throughput on this socket so a Posted relaxation
    // would not help.
    if (arch == tt::ARCH::BLACKHOLE) {
        constexpr uint64_t l2cpu_lim_base = 0x08000000ULL;
        for (const tt::umd::CoreCoord& core : sdesc.get_cores(tt::CoreType::L2CPU, tt::CoordSystem::TRANSLATED)) {
            device_driver.configure_tlb(
                mmio_device_id, core, get_static_tlb_size(), l2cpu_lim_base, tt::umd::tlb_data::Strict);
        }
    }

    if (arch == tt::ARCH::BLACKHOLE && sdesc.get_num_dram_channels() == blackhole::NUM_DRAM_CHANNELS) {
        uint32_t dram_addr = 0;
        for (std::uint32_t dram_channel = 0; dram_channel < blackhole::NUM_DRAM_CHANNELS; dram_channel++) {
            tt::umd::CoreCoord dram_core =
                tt::umd::CoreCoord(blackhole::ddr_to_noc0(dram_channel), tt::CoreType::DRAM, tt::CoordSystem::NOC0);
            device_driver.configure_tlb(mmio_device_id, dram_core, 4ULL * (1ULL << 30), dram_addr, tt::umd::tlb_data::Posted);
        }
    }
}

}  // namespace ll_api
