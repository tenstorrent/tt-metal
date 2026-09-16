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
#include "umd/device/chip_helpers/simulation_sysmem_manager.hpp"  // SysmemManager::get_pcie_base_for_arch
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

// Resolve the emulated chip backend for a device via the MetalContext cluster.
tt::umd::SWEmuleChip* get_sw_emulated_chip(tt::ChipId device_id) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto* umd_cluster = cluster.get_driver().get();
    if (!umd_cluster) {
        return nullptr;
    }
    auto* chip = umd_cluster->get_chip(device_id);
    return dynamic_cast<tt::umd::SWEmuleChip*>(chip);
}

// Per-device cache of pcie_base_ (the host-facing/PCIe address threshold), keyed by device_id —
// avoids a dynamic_cast + cluster lookup on every single NOC-address resolve. Rebuilt lazily; a
// device close+reopen mints a new SWEmuleChip with a stable arch, so the cached value never goes
// stale the way the core_map cache (which holds raw Core* into per-chip L1) can.
static std::mutex g_pcie_base_mutex;
static std::unordered_map<uint32_t, uint64_t> g_pcie_base_cache;

uint64_t get_pcie_base_cached(uint32_t device_id) {
    std::lock_guard<std::mutex> lock(g_pcie_base_mutex);
    auto it = g_pcie_base_cache.find(device_id);
    if (it != g_pcie_base_cache.end()) {
        return it->second;
    }
    auto* sw_emu = get_sw_emulated_chip(static_cast<tt::ChipId>(device_id));
    uint64_t pcie_base =
        sw_emu ? tt::umd::SysmemManager::get_pcie_base_for_arch(sw_emu->get_soc_descriptor().arch) : UINT64_MAX;
    g_pcie_base_cache[device_id] = pcie_base;
    return pcie_base;
}

// Per-device physical {x,y}->Core* maps. g_core_map_mutex/cache have EXTERNAL linkage (declared
// in the header) because the fabric teleport resolver reads them directly to resolve a remote
// chip's core (its map is built by that device's own concurrent run). See docs/fabric-ccl-emulation.md.
std::mutex g_core_map_mutex;
std::unordered_map<uint32_t, std::shared_ptr<std::unordered_map<uint64_t, tt_emule::Core*>>> g_core_map_cache;
// The SWEmuleChip each cached core_map was built against. A device close+reopen mints
// a NEW SWEmuleChip with fresh per-core L1 mmaps (single-process-galaxy L1 model), so a
// core_map cached from the prior chip holds Core* into a now-disjoint L1 region. The NOC
// path (this map) would then resolve a worker's semaphore to a different L1 backing than
// that worker's own fiber (built from the CURRENT chip in setup_core_state) reads —
// cross-core sems never observed → deadlock. Rebuild when the chip identity changes.
static std::unordered_map<uint32_t, tt::umd::SWEmuleChip*> g_core_map_sw_emu;

std::unordered_map<uint64_t, tt_emule::Core*>* build_core_map(
    tt::umd::SWEmuleChip* sw_emu, IDevice* device, ChipId device_id) {
    std::lock_guard<std::mutex> lock(g_core_map_mutex);
    auto& core_map = g_core_map_cache[device_id];
    if (core_map && g_core_map_sw_emu[device_id] != sw_emu) {
        core_map.reset();  // stale: built against a different (now-replaced) SWEmuleChip
    }
    if (!core_map && sw_emu) {
        g_core_map_sw_emu[device_id] = sw_emu;
        core_map = std::make_shared<std::unordered_map<uint64_t, tt_emule::Core*>>();
        // Add ALL worker cores from the device grid
        auto grid = device->compute_with_storage_grid_size();
        for (uint32_t lx = 0; lx < grid.x; lx++) {
            for (uint32_t ly = 0; ly < grid.y; ly++) {
                auto phys = device->virtual_core_from_logical_core(CoreCoord(lx, ly), tt::CoreType::WORKER);
                auto* core = sw_emu->get_core(tt_xy_pair(phys.x, phys.y));
                uint64_t key = (uint64_t(phys.x) << 32) | phys.y;
                (*core_map)[key] = core;
            }
        }
        // Add DRAM cores. Post-uplift get_core() is worker-only (mints a bogus
        // CoreRole::WORKER for a DRAM coord), so back DRAM per physical channel via
        // get_dram_channel_backing(channel): every NOC endpoint of a channel must
        // alias onto that one CoreRole::DRAM core so host writes and kernel NOC reads
        // hit the same memory. get_dram_cores() groups by LOGICAL channel (outer index).
        auto& umd_soc = sw_emu->get_soc_descriptor();
        auto dram_cores = umd_soc.get_dram_cores();
        for (uint32_t ch = 0; ch < dram_cores.size(); ch++) {
            auto* core = sw_emu->get_dram_channel_backing(ch);
            for (auto& dc : dram_cores[ch]) {
                uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                (*core_map)[key] = core;
            }
        }
        // Add DRAM cores (metal_SocDescriptor preferred worker coords). Register BOTH
        // NOC0 and NOC1 preferred coords (on Wormhole they differ per view) so
        // __emule_resolve_noc_addr can route either. Key the backing by the coord's
        // umd LOGICAL channel (the physical DRAM channel) — not the metal dram-view
        // index: several views alias one physical channel (at different offsets), and
        // the host write path resolves the same LOGICAL channel. Keying by view index
        // would split one channel across multiple backings → host/kernel read mismatch.
        {
            auto& umd = sw_emu->get_soc_descriptor();
            auto& msoc = MetalContext::instance().get_cluster().get_soc_desc(device_id);
            for (uint32_t view = 0; view < msoc.get_num_dram_views() && view < MAX_NUM_BANKS; view++) {
                for (uint32_t noc = 0; noc < NUM_NOCS; noc++) {
                    auto dc = msoc.get_preferred_worker_core_for_dram_view(view, noc);
                    auto lg =
                        umd.translate_coord_to(tt_xy_pair(dc.x, dc.y), CoordSystem::TRANSLATED, CoordSystem::LOGICAL);
                    auto* core = sw_emu->get_dram_channel_backing(static_cast<uint32_t>(lg.x));
                    uint64_t key = (uint64_t(dc.x) << 32) | dc.y;
                    (*core_map)[key] = core;
                }
            }
        }
    } else if (!core_map) {
        core_map = std::make_shared<std::unordered_map<uint64_t, tt_emule::Core*>>();
    }
    return core_map.get();
}

}  // namespace tt::tt_metal::emule
