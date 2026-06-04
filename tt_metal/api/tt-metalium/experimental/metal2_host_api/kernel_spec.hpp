// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/advanced_options.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/group.hpp>
#include <tt-metalium/experimental/metal2_host_api/utility/table.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_metal::experimental {

// ============================================================================
//  KernelSpec API
// ============================================================================
//
// A *kernel* is a function — a kernel_main() — that runs on a node's baby RISC-V
// cores: device code, in the GPU-programming-model sense. A Tenstorrent kernel is
// specifically either a *compute* kernel or a *data-movement* kernel.
//
// A KernelSpec describes a *compiled kernel*: it specializes a kernel for
// compilation, baking in the kernel's compile-time arguments and compiler options.
//
// A compiled kernel may run as multiple threads (see num_threads), following the
// SPMD (single-program, multiple-data) model: a small number of independent
// threads that each run the whole kernel function, each with its own thread
// index, coordinating explicitly. How those threads map onto the node's physical
// RISC-V cores — and how many binaries the kernel compiles to — is an implementation
// detail the programming model hides.
//
// The KernelSpec describes all the properties of a kernel:
//  - Source code
//  - Compiler options for generating the kernel binary(ies)
//  - Resource bindings (access to DFBs, semaphores, etc.)
//  - Kernel argument schema (for arguments specified when the Program is enqueued)
//  - Kernel argument bindings (for compile-time constant arguments)
//  - The configuration of any hardware resources controlled by the kernel
//
// SPECIALIZATION: A single kernel source may be represented by multiple KernelSpecs
//   in the same ProgramSpec — for example with different CTA bindings, different DFB
//   endpoint bindings, different semaphore bindings, etc. Each KernelSpec compiles
//   independently and is placed independently, via its WorkUnitSpec membership.
//
// INSTANCING: At runtime, one *kernel instance* runs on each node where the kernel
//   is placed. Each instance is a copy of the compiled kernel, fed its own per-node
//   runtime arguments (see ProgramRunArgs) — so sibling instances can do different
//   work from the same binary.
//
// PLACEMENT: The nodes the kernel runs on is derived from WorkUnitSpec membership.
//
// ============================================================================

// A name identifying a KernelSpec within a ProgramSpec.
using KernelSpecName = ttsl::StrongType<std::string, struct KernelSpecNameTag>;

//------------------------------------------------
// KernelSpec
//------------------------------------------------

struct KernelSpec {
    ///////////////////////////////////////////////////////////////////
    // Basic kernel info
    ///////////////////////////////////////////////////////////////////

    // Kernel identifier: used to reference this kernel within the ProgramSpec
    KernelSpecName unique_id;

    // Kernel source: either a path to a source file, or the source code itself.
    // To pass inline source code, wrap it in KernelSpec::SourceCode{...}.
    // (A string literal binds directly to the path variant alternative.)
    struct SourceCode {
        std::string code;
    };
    std::variant<std::filesystem::path, SourceCode> source;

    // NOTE: The kernel's target node set is a DERIVED property, based on the
    //       WorkUnitSpec(s) that include this kernel.

    // Kernel threading: the number of SPMD threads this kernel has.
    //
    // The legality rules for num_threads are architecture and kernel-type dependent:
    //  - Gen1 architectures (Wormhole, Blackhole) support single-threaded kernels only.
    //  - Gen2 architectures (Quasar) support num_threads > 1.
    //    Different rules apply for compute vs data-movement kernels.
    uint32_t num_threads = 1;

    // Kernel type (methods)
    bool is_data_movement_kernel() const { return std::holds_alternative<DataMovementHardwareConfig>(hw_config); }
    bool is_compute_kernel() const { return std::holds_alternative<ComputeHardwareConfig>(hw_config); }

    ///////////////////////////////////////////////////////////////////
    // Kernel compiler options
    ///////////////////////////////////////////////////////////////////
    struct CompilerOptions {
        using IncludePaths = std::vector<std::filesystem::path>;
        using Defines = Table<std::string, std::string>;
        using OptLevel = tt::tt_metal::KernelBuildOptLevel;

        IncludePaths include_paths;         // -I <path>
        Defines defines;                    // -D <name>=<value>
        OptLevel opt_level = OptLevel::O2;  // -O<level>
        // Can add more options here as needed
    };
    CompilerOptions compiler_options = {};

    ///////////////////////////////////////////////////////////////////
    // Program-scope resource bindings
    ///////////////////////////////////////////////////////////////////

    // DFB bindings
    // Declares that this kernel requires a DFB resource (declared at the ProgramSpec level)
    // The kernel constructs the accessor via DataflowBuffer(dfb::<accessor_name>)
    struct DFBBinding {
        // Endpoint role this binding plays for the DFB.
        enum class EndpointType { PRODUCER, CONSUMER };
        // How the kernel's threads iterate over the DFB's entries. (Only meaningful
        // for multi-threaded kernels; at num_threads == 1 all patterns are equivalent.)
        //   STRIDED: a kernel thread accesses every N-th entry (where N = num_threads)
        //   ALL:     each kernel thread accesses every DFB entry
        //   BLOCKED: a kernel thread accesses blocks of N entries, in strides of N blocks
        //            (NOT YET SUPPORTED — currently rejected at runtime)
        enum class AccessPattern { STRIDED, ALL, BLOCKED };

        DFBSpecName dfb_spec_name;   // identify the DFB within the ProgramSpec
        std::string accessor_name;   // DFB accessor name (used in the kernel source code)
        EndpointType endpoint_type;  // producer or consumer
        AccessPattern access_pattern = AccessPattern::STRIDED;
        // Tiles per block; meaningful (and required) only when access_pattern == BLOCKED.
        // A block is the contiguous strip of entries a thread accesses before striding by
        // block_size * num_threads. Must be 0 for STRIDED/ALL.
        uint32_t block_size = 0;
    };
    Group<DFBBinding> dfb_bindings;

    // Semaphore bindings
    // Declares that this kernel accesses a semaphore resource (declared at the ProgramSpec level)
    // The kernel constructs the accessor via Semaphore(sem::<accessor_name>)
    struct SemaphoreBinding {
        SemaphoreSpecName semaphore_spec_name;  // identify the semaphore within the ProgramSpec
        std::string accessor_name;              // semaphore accessor name (used in the kernel source code)
    };
    Group<SemaphoreBinding> semaphore_bindings;

    // Scratchpad bindings
    // Declares that this kernel uses a scratchpad resource (declared at the ProgramSpec level)
    // The kernel constructs the accessor via Scratchpad(scratch::<accessor_name>)
    struct ScratchpadBinding {
        ScratchpadSpecName scratchpad_spec_name;  // identify the scratchpad within the ProgramSpec
        std::string accessor_name;                // scratchpad accessor name (used in the kernel source code)
    };
    Group<ScratchpadBinding> scratchpad_bindings;

    ///////////////////////////////////////////////////////////////////
    // Program parameter bindings (user-managed resources)
    ///////////////////////////////////////////////////////////////////

    // Tensor bindings
    // Declares that this kernel accesses a tensor parameter (declared at the ProgramSpec level)
    // The kernel constructs the accessor via TensorAccessor(tensor::<accessor_name>)
    struct TensorBinding {
        TensorParamName tensor_parameter_name;  // identify the TensorParameter within the ProgramSpec
        std::string accessor_name;              // tensor accessor name (used in the kernel source code)
    };
    Group<TensorBinding> tensor_bindings;

    // Additional program parameter binding types (coming soon):
    //  - GlobalSemaphore bindings
    //  - GlobalDataflowBuffer bindings
    //  - MeshBuffer bindings

    //////////////////////////////////////////////////////////////////////////////
    // Kernel arguments
    //////////////////////////////////////////////////////////////////////////////

    //----------------------------------------------------------------------------
    // Compile time arguments
    // (Bound argument values cannot be changed between Program executions)
    using CompileTimeArgs = Table<std::string, uint32_t>;
    CompileTimeArgs compile_time_args;
    // TODO -- extend to support arbitrary POD types, including user-defined structs.

    //----------------------------------------------------------------------------
    // Runtime argument schema (declaration)

    // Schema (names) for the runtime arguments declared by this kernel.
    // (The values of these arguments are set as ProgramRunArgs.)
    // Currently, only arguments of uint32_t are supported.

    struct RuntimeArgSchema {
        // Runtime argument names (must be unique, valid C++ identifiers.)
        Group<std::string> runtime_arg_names;

        // Common runtime argument names (must be unique, valid C++ identifiers.)
        Group<std::string> common_runtime_arg_names;
    };
    RuntimeArgSchema runtime_arg_schema{};

    // For vararg-style positional arguments, see KernelAdvancedOptions.

    //////////////////////////////////////////////////////////////////////////////
    // Kernel-controlled hardware resource configuration
    //////////////////////////////////////////////////////////////////////////////
    std::variant<DataMovementHardwareConfig, ComputeHardwareConfig> hw_config;

    //////////////////////////////////////////////////////////////////////////////
    // Advanced options (see advanced_options.hpp)
    //////////////////////////////////////////////////////////////////////////////
    KernelAdvancedOptions advanced_options;
};

//------------------------------------------------
// Convenience aliases
//------------------------------------------------

// These aliases lift commonly-used nested enums to the namespace level
using DFBEndpointType = KernelSpec::DFBBinding::EndpointType;
using DFBAccessPattern = KernelSpec::DFBBinding::AccessPattern;

// These aliases lift the kernel resource-binding types to the namespace level
using DFBBinding = KernelSpec::DFBBinding;
using TensorBinding = KernelSpec::TensorBinding;
using SemaphoreBinding = KernelSpec::SemaphoreBinding;
using ScratchpadBinding = KernelSpec::ScratchpadBinding;

//------------------------------------------------
// Convenience factories for DFBBinding
//------------------------------------------------

// Ergonomic alternatives to writing a designated-initializer DFBBinding{...}

// Creates a DFB producer binding with a STRIDED access pattern
// (All DFB producers are STRIDED)
inline DFBBinding ProducerOf(DFBSpecName dfb_spec_name, std::string accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .accessor_name = std::move(accessor_name),
        .endpoint_type = DFBEndpointType::PRODUCER,
        .access_pattern = DFBAccessPattern::STRIDED};
}

// Creates a DFB consumer binding (with a default-STRIDED access pattern)
// Use this for single-threaded kernels, where the access pattern doesn't matter.
// For multi-threaded kernels (Quasar), prefer the explicit access pattern
// helper factories below.
inline DFBBinding ConsumerOf(DFBSpecName dfb_spec_name, std::string accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .accessor_name = std::move(accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        // access pattern defaults to STRIDED
    };
}

// Creates a DFB consumer binding with a STRIDED access pattern
// (The common case for multi-threaded DFB consumers)
inline DFBBinding StridedConsumerOf(DFBSpecName dfb_spec_name, std::string accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .accessor_name = std::move(accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::STRIDED,
    };
}

// Creates a DFB consumer binding with an ALL access pattern
inline DFBBinding AllConsumerOf(DFBSpecName dfb_spec_name, std::string accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .accessor_name = std::move(accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::ALL,
    };
}

// Creates a DFB consumer binding with a BLOCKED access pattern.
// block_size is the number of contiguous entries a thread accesses before striding
// by block_size * num_threads; it must be > 0.
inline DFBBinding BlockedConsumerOf(DFBSpecName dfb_spec_name, std::string accessor_name, uint32_t block_size) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .accessor_name = std::move(accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::BLOCKED,
        .block_size = block_size,
    };
}

}  // namespace tt::tt_metal::experimental
