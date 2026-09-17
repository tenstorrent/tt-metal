// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <tt-metalium/device.hpp>  // IDevice, ChipId, CoreCoord, CoreType, BufferType

namespace tt::umd {
class SWEmuleChip;
}
namespace tt_emule {
class Core;
struct SocView;  // the marshaller's flat device-geometry POD (emule_program_descriptor.hpp)
}

// Bank-mapping constants and the four bank arrays live at GLOBAL scope with unmangled
// names: JIT-compiled kernels resolve them by symbol via dlopen(-rdynamic), so a
// namespace or rename would break that lookup. They are defined in emule_device_map.cpp
// and exported through libtt_metal's dynamic symbol table (-rdynamic).
constexpr uint32_t NUM_NOCS = 2;
// L1 banks scale with worker grid: 64 on WH-N150, 140 on BH P100/P150.  Must match the
// array size declared by the JIT side in include/jit_hw/internal/dataflow/dataflow_api_addrgen.h.
constexpr uint32_t MAX_NUM_BANKS = 256;
constexpr uint32_t NOC_NODE_ID_BITS = 6;  // noc_xy encoding: (y << NOC_NODE_ID_BITS) | x
// NOC-address decode constants (encoded 64-bit: y[47:42] x[41:36] addr[35:0]), shared by
// the NOC bridge and the fabric resolvers.
constexpr uint32_t NOC_LOCAL_BITS = 36;
constexpr uint64_t NOC_LOCAL_MASK = (1ULL << NOC_LOCAL_BITS) - 1;
constexpr uint32_t NOC_NODE_MASK = (1u << NOC_NODE_ID_BITS) - 1;

extern uint16_t dram_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS];
extern int32_t bank_to_dram_offset[MAX_NUM_BANKS];
extern uint16_t l1_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS];
extern int32_t bank_to_l1_offset[MAX_NUM_BANKS];

namespace tt::tt_metal::emule {

// Populate the DRAM/L1 bank arrays above from the marshaller's SocView (built by
// tt_emule::build_soc_view), so a kernel's interleaved-address bank index resolves to the
// worker/DRAM core the host wrote. A runtime diff-guard asserts SocView == the direct
// metal_SocDescriptor/allocator reads this used to do.
void populate_bank_mapping(
    tt::umd::SWEmuleChip* sw_emu,
    const tt_emule::SocView& soc,
    tt_emule::Core*& dram_core_out,
    uint32_t& num_dram_channels_out,
    uint32_t& num_l1_banks_out);

// Build the logical->virtual worker column/row CSV maps (JIT defines) for a device.
void build_worker_coord_maps(IDevice* device, std::string& worker_col_map_str, std::string& worker_row_map_str);

// Resolve the emulated chip backend for a device (via the MetalContext cluster). Used by the
// extern-C NOC bridge and the fabric resolver in the runner.
tt::umd::SWEmuleChip* get_sw_emulated_chip(tt::ChipId device_id);

// Cached host-facing/PCIe address threshold per device (pre-filter for on-chip vs host NOC addr).
uint64_t get_pcie_base_cached(uint32_t device_id);

// Build (cached) the physical {x,y}->Core* map for a device's NOC resolution. The cache globals
// are exposed below because the fabric resolver reads them directly to resolve a peer chip's map.
std::unordered_map<uint64_t, tt_emule::Core*>* build_core_map(
    tt::umd::SWEmuleChip* sw_emu, IDevice* device, ChipId device_id, const tt_emule::SocView& soc);

extern std::mutex g_core_map_mutex;
extern std::unordered_map<uint32_t, std::shared_ptr<std::unordered_map<uint64_t, tt_emule::Core*>>> g_core_map_cache;

}  // namespace tt::tt_metal::emule
