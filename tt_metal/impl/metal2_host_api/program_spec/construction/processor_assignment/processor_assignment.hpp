// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>
#include <cstdint>
#include <unordered_map>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "impl/metal2_host_api/helpers.hpp"
#include "impl/metal2_host_api/program_spec/collection/collect_metadata.hpp"

namespace tt::tt_metal::experimental {

// Processor allocation on a node: bit i set = core i in use.
using DMProcessorMask = std::bitset<QUASAR_DM_CORES_PER_NODE>;
using ComputeEngineMask = std::bitset<QUASAR_TENSIX_ENGINES_PER_NODE>;

// Kernel -> DFB risc mask (passed to MakeDataflowBufferConfig)
//   Gen1: bit 0 = RISCV_0 (BRISC), bit 1 = RISCV_1 (NCRISC), bit 2 = Tensix compute
//   Gen2: bits 0-7 = DM processors, bits 8-15 = Tensix compute engines
using KernelRiscMaskMap = std::unordered_map<const KernelSpec*, uint16_t>;

// Build each kernel's risc mask (arch-specific):
//  - Gen2: backtracking solver assigns DM cores automatically
//  - Gen1: processor is user-specified in Gen1Config
KernelRiscMaskMap SolveKernelRiscMasks(const ProgramSpec& spec, const CollectedSpecData& collected, const Hal& hal);

}  // namespace tt::tt_metal::experimental
