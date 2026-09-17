// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
// collect_kernels — walks the descriptor's kernels and builds the JIT variant set. Produces
// PendingKernelInfo (consumed by the engine's launch path) and DeferredCompile tasks (emule_jit).
// Reads only EmuleProgramDescriptor + SocView — no private tt-metal Program/Kernel. See docs.

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "emule_jit.hpp"  // DeferredCompile

namespace tt_emule {
struct EmuleProgramDescriptor;
struct SocView;
}  // namespace tt_emule

namespace tt::tt_metal::emule {

struct PendingKernelInfo {
    // Parallels KernelInfo::variants but holds cache keys pending compile-resolution.
    std::vector<std::string> variant_cache_keys;
    bool run_all_variants = false;
    uint8_t processor_id = 0;
    uint8_t thread_idx = 0;  // Index within this kernel's processor list
    bool is_tensix = false;
    uint32_t num_threads = 1;
    uint32_t kernel_config_base = 0;
    uint16_t rta_offset_in_kc = tt_emule::kRtaCrtaNoArgsSentinel;
    uint16_t crta_offset_in_kc = tt_emule::kRtaCrtaNoArgsSentinel;
    // Runtime-arg values (unique + common); buffer L1 addresses appear verbatim, so
    // Object-Intent uses them to find this kernel's I/O tensors (§12).
    std::vector<uint32_t> rt_arg_values;
    std::string kernel_name;          // kernel source path, for the ASAN trace
    uint32_t num_unique_rt_args = 0;  // size of the per-core (rta) region; see KernelInfo
};

// Definition in emule_program_model.cpp. Reads only the descriptor + SocView (no private
// tt-metal Program/Kernel); the marshaller (build_emule_descriptor) is the sole private reader.
void collect_kernels(
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base,
    const std::string& extra_inc,
    std::map<CoreCoord, std::vector<PendingKernelInfo>>& pending_core_kernels,
    std::map<std::string, DeferredCompile>& deferred_compiles,
    std::unordered_map<std::string, std::function<void()>>& resolved_fns,
    std::vector<std::string>& inline_src_temps,
    const tt_emule::EmuleProgramDescriptor& desc,
    const tt_emule::SocView& soc);

}  // namespace tt::tt_metal::emule
