// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "emule_device_map.hpp"

#include <cstring>
#include <sstream>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/device.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "umd/device/chip/sw_emule_chip.hpp"
#include "tt_emule/device.hpp"

// The four bank arrays — global scope, unmangled, -rdynamic-exported for JIT kernels.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS] = {};
int32_t bank_to_dram_offset[MAX_NUM_BANKS] = {};
uint16_t l1_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS] = {};
int32_t bank_to_l1_offset[MAX_NUM_BANKS] = {};

namespace tt::tt_metal::emule {

void populate_bank_mapping(
    tt::umd::SWEmuleChip* sw_emu,
    IDevice* device,
    ChipId device_id,
    tt_emule::Core*& dram_core_out,
    uint32_t& num_dram_channels_out,
    uint32_t& num_l1_banks_out) {
    dram_core_out = nullptr;
    num_dram_channels_out = 0;
    num_l1_banks_out = 0;
    if (!sw_emu) {
        return;
    }

    auto& soc = sw_emu->get_soc_descriptor();
    auto dram_channels = soc.get_dram_cores();
    num_dram_channels_out = static_cast<uint32_t>(dram_channels.size());

    if (num_dram_channels_out > 0) {
        dram_core_out = sw_emu->get_dram_channel_backing(0);
    }

    // Populate bank mapping arrays using metal_SocDescriptor (matches host write path).
    auto& metal_soc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
    num_dram_channels_out = static_cast<uint32_t>(metal_soc.get_num_dram_views());
    TT_FATAL(
        num_dram_channels_out <= MAX_NUM_BANKS,
        "emule: num_dram_channels ({}) exceeds MAX_NUM_BANKS ({}); bump the constant or implement dynamic backing",
        num_dram_channels_out,
        MAX_NUM_BANKS);

    // noc_xy encoding: (y << 6) | x (matching Blackhole firmware encoding).
    // Per-NOC preferred coords: get_preferred_worker_core_for_dram_view's noc arg
    // selects the view's worker_endpoint[noc] subchannel. On Wormhole the two
    // subchannels coincide (worker_endpoint=[n,n]); on Blackhole they are distinct
    // NOC ports of the same physical bank.
    //
    // The kernel-side extern is declared [NUM_NOCS][NUM_DRAM_BANKS] where the JIT
    // define NUM_DRAM_BANKS == num_dram_channels_out, so the kernel's [noc][bank]
    // row stride is the actual bank count — NOT this array's static MAX_NUM_BANKS
    // dimension. Lay the table out flat with that same actual-count stride (matching
    // silicon's [noc*num_banks + bank] vector) so noc=1 rows align. A 2D [noc][bank]
    // write would stride by MAX_NUM_BANKS and the kernel's noc=1 reads would land on
    // uninitialized zeros → coord (0,0) → wrong (DRAM) backing.
    uint16_t* dram_tbl = &dram_bank_to_noc_xy[0][0];
    std::memset(dram_bank_to_noc_xy, 0, sizeof(dram_bank_to_noc_xy));
    std::memset(bank_to_dram_offset, 0, sizeof(bank_to_dram_offset));
    for (uint32_t ch = 0; ch < num_dram_channels_out && ch < MAX_NUM_BANKS; ch++) {
        auto dc0 = metal_soc.get_preferred_worker_core_for_dram_view(ch, 0 /* NOC 0 */);
        auto dc1 = metal_soc.get_preferred_worker_core_for_dram_view(ch, 1 /* NOC 1 */);
        uint16_t noc_xy0 = (static_cast<uint16_t>(dc0.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(dc0.x);
        uint16_t noc_xy1 = (static_cast<uint16_t>(dc1.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(dc1.x);
        dram_tbl[0 * num_dram_channels_out + ch] = noc_xy0;
        dram_tbl[1 * num_dram_channels_out + ch] = noc_xy1;
        bank_to_dram_offset[ch] = static_cast<int32_t>(metal_soc.get_address_offset(ch));

        log_debug(
            tt::LogMetal,
            "  DRAM bank[{}]: NOC0=({},{}) NOC1=({},{}) noc_xy0=0x{:04x} noc_xy1=0x{:04x} offset=0x{:x}",
            ch,
            dc0.x,
            dc0.y,
            dc1.x,
            dc1.y,
            noc_xy0,
            noc_xy1,
            bank_to_dram_offset[ch]);
    }

    // L1 bank mapping — mirror the host allocator's bank distribution so the
    // kernel-side `interleaved_addr_gen::get_bank_index<L1>(id)` lands on the
    // same worker core that `SWEmuleChip::write_to_device` wrote a given page
    // to.  Without this, every page maps to bank 0 (a single core) while the
    // host scatters across all worker cores — interleaved-L1 → sharded paths
    // read all zeros.
    // Flat actual-count stride, same rationale as dram_bank_to_noc_xy above: the
    // kernel reads l1_bank_to_noc_xy[noc][bank] with stride NUM_L1_BANKS (== the JIT
    // define == num_l1_banks_out), not MAX_NUM_BANKS.
    uint16_t* l1_tbl = &l1_bank_to_noc_xy[0][0];
    std::memset(l1_bank_to_noc_xy, 0, sizeof(l1_bank_to_noc_xy));
    std::memset(bank_to_l1_offset, 0, sizeof(bank_to_l1_offset));
    if (device) {
        const auto& allocator = device->allocator();
        num_l1_banks_out = allocator->get_num_banks(BufferType::L1);
        TT_FATAL(
            num_l1_banks_out <= MAX_NUM_BANKS,
            "emule: num_l1_banks ({}) exceeds MAX_NUM_BANKS ({}); bump the constant or implement dynamic backing",
            num_l1_banks_out,
            MAX_NUM_BANKS);
        for (uint32_t b = 0; b < num_l1_banks_out && b < MAX_NUM_BANKS; ++b) {
            auto logical = allocator->get_logical_core_from_bank_id(b);
            auto virt = device->virtual_core_from_logical_core(logical, CoreType::WORKER);
            uint16_t noc_xy = (static_cast<uint16_t>(virt.y) << NOC_NODE_ID_BITS) | static_cast<uint16_t>(virt.x);
            l1_tbl[0 * num_l1_banks_out + b] = noc_xy;  // NOC 0
            l1_tbl[1 * num_l1_banks_out + b] = noc_xy;  // NOC 1 (same target in emule)
            // Intentionally leave bank_to_l1_offset[b] = 0.  emule's per-core
            // L1 mmap starts at byte 0 with no firmware-reserved prefix, so
            // silicon's `allocator->get_bank_offset(L1, b)` isn't applicable.
        }
    }
}

void build_worker_coord_maps(IDevice* device, std::string& worker_col_map_str, std::string& worker_row_map_str) {
    static constexpr uint32_t MAX_LOGICAL_GRID_DIM = 64;
    auto grid = device->compute_with_storage_grid_size();
    std::ostringstream col_ss;
    for (uint32_t lx = 0; lx < MAX_LOGICAL_GRID_DIM; lx++) {
        if (lx) {
            col_ss << ',';
        }
        if (lx < grid.x) {
            auto virt = device->virtual_core_from_logical_core(CoreCoord(lx, 0), CoreType::WORKER);
            col_ss << virt.x;
        } else {
            col_ss << 0;
        }
    }
    worker_col_map_str = col_ss.str();

    std::ostringstream row_ss;
    for (uint32_t ly = 0; ly < MAX_LOGICAL_GRID_DIM; ly++) {
        if (ly) {
            row_ss << ',';
        }
        if (ly < grid.y) {
            auto virt = device->virtual_core_from_logical_core(CoreCoord(0, ly), CoreType::WORKER);
            row_ss << virt.y;
        } else {
            row_ss << 0;
        }
    }
    worker_row_map_str = row_ss.str();
}

}  // namespace tt::tt_metal::emule
