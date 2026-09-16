// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Per-core CB / DFB / semaphore setup. setup_core_state builds the per-core CoreSetup list
// (CB-sync state, DFB allocation, semaphores) consumed by the engine's launch_cores;
// build_per_thread_dfb_interfaces builds the per-RISC DFB views. The shared setup->engine
// value types (KernelInfo, DFBAllocInfo, CoreSetup) live here.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "emule_program_model.hpp"      // kRtaCrtaNoArgsSentinel
#include "tt_emule/device.hpp"          // tt_emule::Core
#include "tt_emule/dfb_sync_state.hpp"  // tt_emule::EmuleDFBInterface

namespace tt::tt_metal {
class IDevice;
namespace detail {
class ProgramImpl;
}
namespace experimental::dfb {
struct DataflowBufferConfig;
}
}  // namespace tt::tt_metal
namespace tt::umd {
class SWEmuleChip;
}

namespace tt::tt_metal::emule {

struct KernelInfo {
    // size 1 for normal kernels; size 4 for Quasar compute (one per TRISC).
    // Either 4 distinct compiled variants (compile-time TRISC_* guards) or 4
    // copies of one function (runtime TRISC_ID). When run_all_variants is true,
    // the launcher iterates and sets __emule_trisc_id per variant.
    std::vector<std::function<void()>> variants;
    bool run_all_variants = false;
    uint8_t processor_id = 0;  // RISC-V processor ID (mhartid); used for DFB role resolution
    uint8_t thread_idx = 0;    // Index within this kernel's processor list → __emule_my_thread_id
    bool is_tensix = false;    // true for Tensix/compute kernels (DFB mask uses bits 8-23)
    uint32_t num_threads = 1;  // number of engines (for get_num_threads())
    // L1 address of rt-args = kernel_config_base + rta_offset_in_kc (per-RISC,
    // read from kg->launch_msg). Sentinel = kernel has no args on this RISC.
    uint32_t kernel_config_base = 0;
    uint16_t rta_offset_in_kc = kRtaCrtaNoArgsSentinel;
    uint16_t crta_offset_in_kc = kRtaCrtaNoArgsSentinel;
    // Runtime-arg values handed to this kernel on its core (see PendingKernelInfo).
    std::vector<uint32_t> rt_arg_values;
    // Kernel source path; owns the string __emule_kernel_name points at during
    // this kernel's launch (used by the ASAN trace to name the offending kernel).
    std::string kernel_name;
    // Count of unique (per-core) runtime-arg words = size of the rta_offset region.
    // rt_arg_values is [unique..(this many).., common..]; used to bounds-check
    // out-of-range per-core arg reads to 0 (silicon zero-pad).
    uint32_t num_unique_rt_args = 0;
};

// DFB allocation info for a single DFB on a core. Only device_slot and base_addr
// are genuinely new per-core state; everything else (entry_size, num_entries,
// risc masks, num_producers/consumers, cap) is read from the borrowed config
// pointer, whose backing DataflowBufferImpl is owned by ProgramImpl and
// outlives one program execution.
struct DFBAllocInfo {
    // Matches the dfb::<name> accessor value the emulated kernel uses, so all per-core tables
    // (CB sync, tile counters, interface slots) are keyed by slot rather than by program-wide id.
    uint32_t device_slot = 0;
    uint32_t base_addr = 0;
    const tt::tt_metal::experimental::dfb::DataflowBufferConfig* cfg = nullptr;
};

struct CoreSetup {
    CoreCoord logical_core;
    tt_emule::Core* core;
    std::vector<KernelInfo>* ki_list;
    uint8_t phys_x;
    uint8_t phys_y;
    std::vector<DFBAllocInfo> dfb_allocs;
    bool has_tc_dfbs = false;  // Quasar: DFBs here are tile-counter-backed, not CB-backed
    uint32_t sem_base;
    uint32_t sem_size;
    // Globally-allocated (persistent) CB extents on this core, packed (start<<32|end);
    // Object-Intent exempts kernel writes to them (§12).
    std::vector<uint64_t> persistent_cb_ranges;
};

// Definitions in emule_cb_dfb_setup.cpp.
void setup_core_state(
    detail::ProgramImpl& impl,
    IDevice* device,
    tt::umd::SWEmuleChip* sw_emu,
    std::map<CoreCoord, std::vector<KernelInfo>>& core_kernels,
    uint32_t emule_sem_base,
    std::vector<CoreSetup>& core_setups);

std::vector<std::unique_ptr<tt_emule::EmuleDFBInterface[]>> build_per_thread_dfb_interfaces(
    const std::vector<KernelInfo>& ki_list, const std::vector<DFBAllocInfo>& dfb_allocs);

}  // namespace tt::tt_metal::emule
