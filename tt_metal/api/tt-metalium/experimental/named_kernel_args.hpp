// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// EXPERIMENTAL: Named kernel-args — temporary, Blaze-only.
//
// This header defines the host-side named-arg structs that were previously
// inlined in `KernelDescriptor` (program_descriptors.hpp).  They are now
// quarantined in `namespace tt::tt_metal::experimental` as a sub-struct
// (`NamedKernelArgs`) aggregated by `KernelDescriptor`.
//
// This feature will be deleted when Blaze migrates to the Metal 2.0
// `args::` system.  See:
//   tt_metal/api/tt-metalium/experimental/README_named_kernel_args.md

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

class Program;
struct KernelDescriptor;

}  // namespace tt::tt_metal

namespace tt::tt_metal::experimental {

// Named runtime args use "ns.field" convention (e.g. "demo.num_tiles").
// The name is split on '.' to produce namespace hierarchy in the generated header.
// Common: same value on all cores. Per-core: different value per core.
struct NamedCommonRuntimeArg {
    std::string name;
    uint32_t value;
};
using NamedCommonRuntimeArgs = std::vector<NamedCommonRuntimeArg>;

struct NamedPerCoreRuntimeArg {
    std::string name;
    std::vector<std::pair<CoreCoord, uint32_t>> core_values;
};
using NamedPerCoreRuntimeArgs = std::vector<NamedPerCoreRuntimeArg>;

// Array variant: named RT arg that occupies N contiguous slots.
// Generates ArrayArg (with length) in the kernel header instead of Arg.
struct NamedCommonRuntimeArgArray {
    std::string name;
    std::vector<uint32_t> values;
};
using NamedCommonRuntimeArgArrays = std::vector<NamedCommonRuntimeArgArray>;

// Per-core array variant: each core gets its own array of N contiguous RT arg slots.
struct NamedPerCoreRuntimeArgArray {
    std::string name;
    std::vector<std::pair<CoreCoord, std::vector<uint32_t>>> core_values;
};
using NamedPerCoreRuntimeArgArrays = std::vector<NamedPerCoreRuntimeArgArray>;

// Aggregates the 4 named-arg vector fields.  Included as a sub-struct member
// on `KernelDescriptor` so that the named-arg fields are quarantined in the
// `experimental` namespace while `KernelDescriptor` itself stays intact.
struct NamedKernelArgs {
    NamedCommonRuntimeArgs named_common_runtime_args;
    NamedPerCoreRuntimeArgs named_per_core_runtime_args;
    NamedCommonRuntimeArgArrays named_common_runtime_arg_arrays;
    NamedPerCoreRuntimeArgArrays named_per_core_runtime_arg_arrays;

    bool empty() const {
        return named_common_runtime_args.empty() && named_per_core_runtime_args.empty() &&
               named_common_runtime_arg_arrays.empty() && named_per_core_runtime_arg_arrays.empty();
    }
};

// Processes named runtime args for a kernel: merges named values into
// positional runtime/common arg vectors, validates identifiers, and builds
// the namespace maps used by the JIT header generator.
// Called from the Program constructor when `kernel_descriptor.named_args` is non-empty.
void process_named_args(Program& program, const KernelDescriptor& kernel_descriptor, uint32_t kernel_handle);

}  // namespace tt::tt_metal::experimental
