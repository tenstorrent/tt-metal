// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "impl/kernels/kernel.hpp"        // Kernel, experimental::quasar::Quasar*Kernel
#include "impl/program/program_impl.hpp"  // detail::ProgramImpl

namespace tt::tt_metal::emule {

// Semaphore alignment in L1 (must match firmware layout). Shared: build_kernel_defines
// emits it as the EMULE_SEM_ALIGN define; the runner's sem-region setup uses it for addressing.
static constexpr uint32_t EMULE_SEM_ALIGN = 16;

// Build the full defines map for a kernel: subclass-derived + arch + emulator
// constants (banking, alignments, worker maps, sem base, CB tile sizes).
std::map<std::string, std::string> build_kernel_defines(
    Kernel& kernel,
    detail::ProgramImpl& impl,
    uint32_t num_dram_channels,
    uint32_t num_l1_banks,
    const std::string& worker_col_map_str,
    const std::string& worker_row_map_str,
    uint32_t emule_sem_base);

// Per-kernel thread count and the processor ids each thread runs as.
struct ProcIdList {
    std::vector<uint8_t> proc_ids;
    uint32_t num_threads;
};
ProcIdList compute_proc_ids_and_thread_count(
    Kernel& kernel,
    experimental::quasar::QuasarDataMovementKernel* qdm,
    experimental::quasar::QuasarComputeKernel* qck);

// Quasar compute TRISC-guard scan result (compile-4-variants vs runtime-TRISC vs single).
struct TriscMode {
    bool needs_trisc_compile = false;
    bool needs_runtime_trisc = false;
};
TriscMode detect_quasar_trisc_mode(bool is_quasar_compute, const std::string& src_path);

}  // namespace tt::tt_metal::emule
