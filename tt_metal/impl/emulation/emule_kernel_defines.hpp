// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "emule_program_descriptor.hpp"  // tt_emule::KernelDescriptor / CoreDescriptor / SocView

namespace tt::tt_metal::emule {

// Semaphore alignment in L1 (must match firmware layout). Shared: build_kernel_defines
// emits it as the EMULE_SEM_ALIGN define; the runner's sem-region setup uses it for addressing.
static constexpr uint32_t EMULE_SEM_ALIGN = 16;

// Build the full defines map for a kernel from the POD: subclass-derived process_defines +
// arch/fabric/alignments (SocView) + banking/worker-maps/sem (scalar params) + the EMULE_TILE_*
// CB/DFB geometry tables (the kernel's first-core CoreDescriptor). Reads no tt-metal Kernel/
// ProgramImpl. first_core_desc is null when the kernel occupies no cores.
std::map<std::string, std::string> build_kernel_defines_from_desc(
    const tt_emule::KernelDescriptor& kd,
    const tt_emule::CoreDescriptor* first_core_desc,
    const tt_emule::SocView& soc,
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base);

// Quasar compute TRISC-guard scan result (compile-4-variants vs runtime-TRISC vs single).
struct TriscMode {
    bool needs_trisc_compile = false;
    bool needs_runtime_trisc = false;
};
TriscMode detect_quasar_trisc_mode(bool is_quasar_compute, const std::string& src_path);

}  // namespace tt::tt_metal::emule
