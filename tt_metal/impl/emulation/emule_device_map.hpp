// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>

#include <tt-metalium/device.hpp>  // IDevice, ChipId, CoreCoord, CoreType, BufferType

namespace tt::umd {
class SWEmuleChip;
}
namespace tt_emule {
class Core;
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

extern uint16_t dram_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS];
extern int32_t bank_to_dram_offset[MAX_NUM_BANKS];
extern uint16_t l1_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS];
extern int32_t bank_to_l1_offset[MAX_NUM_BANKS];

namespace tt::tt_metal::emule {

// Populate the DRAM/L1 bank arrays above from the SoC descriptor + host allocator, so a
// kernel's interleaved-address bank index resolves to the worker/DRAM core the host wrote.
void populate_bank_mapping(
    tt::umd::SWEmuleChip* sw_emu,
    IDevice* device,
    ChipId device_id,
    tt_emule::Core*& dram_core_out,
    uint32_t& num_dram_channels_out,
    uint32_t& num_l1_banks_out);

// Build the logical->virtual worker column/row CSV maps (JIT defines) for a device.
void build_worker_coord_maps(IDevice* device, std::string& worker_col_map_str, std::string& worker_row_map_str);

}  // namespace tt::tt_metal::emule
