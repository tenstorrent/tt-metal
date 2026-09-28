// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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

// Maximum length of a resource binding's accessor_name (DFB, semaphore, scratchpad, tensor).
//
// MAINTAINER: This constant is in sync with MAX_TEMPLATE_STRING_LEN on device side.
inline constexpr std::size_t MAX_ACCESSOR_NAME_LENGTH = 64;

//------------------------------------------------
// KernelSpec
//------------------------------------------------

struct KernelSpec {
    // Note on the inline invariant comments:
    // The invariant listed in the comments describes the local invariant of the field from the perspective of the
    // KernelSpec and KernelSpec alone. More invariants about how the bindings and references interact with the rest of
    // the ProgramSpec are listed in the ProgramSpec header.

    ///////////////////////////////////////////////////////////////////
    // Basic kernel info
    ///////////////////////////////////////////////////////////////////

    // Kernel identifier: used to reference this kernel within the ProgramSpec
    KernelSpecName unique_id;

    // Kernel source: either a path to a source file, or the source code itself.
    // To pass inline source code, wrap it in KernelSpec::SourceCode{...}.
    // (A string literal binds directly to the path variant alternative.)
    //
    // When source is a path, the lookup order is as follows:
    //   - Must be non-empty.
    //   - An absolute path is used as given.
    //   - A relative path is resolved against the first location where the file exists:
    //       1. The current working directory
    //       2. TT_METAL_KERNEL_PATH, when that variable is set
    //       3. The system kernel directory (/usr/share/tenstorrent/kernels/)
    //       4. TT_METAL_HOME, or the directory set with SetRootDir
    //
    // Invariant for the path:
    // - Must be non-empty.
    // - Must point to a file that exists.
    // - The file must be readable.
    struct SourceCode {
        // Invariant:
        // - Must be non-empty.
        std::string code;
    };
    std::variant<std::filesystem::path, SourceCode> source;

    // NOTE: The kernel's target node set is a DERIVED property, based on the
    //       WorkUnitSpec(s) that include this kernel.

    // Kernel threading: the number of SPMD threads this kernel has.
    //
    // Invariant on Gen1 architectures (Wormhole, Blackhole): must be 1.
    // Invariant on Gen2 architecture (Quasar):
    //   - If is_data_movement_kernel(), the valid range is [1, 6]
    //   - If is_compute_kernel(), the valid values are [1, 2, 4]
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
    // The kernel constructs a DataflowBuffer from the binding token:
    //   DataflowBuffer(dfb::<accessor_name>)
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

        // identify the DFB within the ProgramSpec
        DFBSpecName dfb_spec_name;

        // DFB accessor name (used in the kernel source code)
        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;

        // producer or consumer
        EndpointType endpoint_type;

        // See above for more documentation.
        //
        // Invariant:
        // - Cannot be blocked (not yet supported).
        // - For a producer binding, must be STRIDED.
        AccessPattern access_pattern = AccessPattern::STRIDED;
    };
    // Local Invariant:
    // - Each DFB has at most one PRODUCER binding and at most one CONSUMER binding.
    //   (A kernel that binds a DFB in both roles "self-loops" it.)
    // - Two bindings may share an accessor_name only if they are the PRODUCER and CONSUMER
    //   bindings of the same DFB. (A self-loop may also use two different accessor_names.)
    // - Gen2: a data-movement kernel must not self-loop a DFB.
    // - A compute kernel that self-loops a DFB must use STRIDED on its CONSUMER binding.
    // - A CONSUMER binding with access_pattern ALL requires num_threads <= 4.
    Group<DFBBinding> dfb_bindings;

    // Semaphore bindings
    // Declares that this kernel accesses a semaphore resource (declared at the ProgramSpec level)
    // The kernel constructs a Semaphore from the emitted id: Semaphore(sem::<accessor_name>)
    struct SemaphoreBinding {
        // identify the semaphore within the ProgramSpec
        SemaphoreSpecName semaphore_spec_name;

        // semaphore accessor name (used in the kernel source code)
        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;
    };
    // Local Invariant:
    // - semaphore_spec_name must be unique across all semaphore_bindings.
    // - accessor_name must be unique across all semaphore_bindings.
    // - Gen 2 & wormhole: Must be empty if is_compute_kernel().
    Group<SemaphoreBinding> semaphore_bindings;

    // Scratchpad bindings
    // Declares that this kernel uses a scratchpad resource (declared at the ProgramSpec level)
    // The kernel constructs a Scratchpad from the binding token, naming the element type:
    //   Scratchpad<uint32_t>(scratch::<accessor_name>)
    struct ScratchpadBinding {
        // identify the scratchpad within the ProgramSpec
        ScratchpadSpecName scratchpad_spec_name;

        // scratchpad accessor name (used in the kernel source code)
        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;
    };
    // Local Invariant:
    // - scratchpad_spec_name must be unique across all scratchpad_bindings.
    // - accessor_name must be unique across all scratchpad_bindings.
    Group<ScratchpadBinding> scratchpad_bindings;

    ///////////////////////////////////////////////////////////////////
    // Program parameter bindings (user-managed resources)
    ///////////////////////////////////////////////////////////////////

    // Tensor bindings
    // Declares that this kernel accesses a tensor parameter (declared at the ProgramSpec level)
    // The kernel constructs a TensorAccessor (or LocalTensorAccessor) from the binding token:
    //   TensorAccessor(tensor::<accessor_name>)
    struct TensorBinding {
        // identify the TensorParameter within the ProgramSpec
        TensorParamName tensor_parameter_name;

        // tensor accessor name (used in the kernel source code)
        // Invariant: A valid C++ identifier shorter than (or equal to) MAX_ACCESSOR_NAME_LENGTH.
        std::string accessor_name;
    };
    // Local Invariant:
    // - accessor_name must be unique across all tensor_bindings.
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
    //
    // Table key represents the accessor name of the CTA.
    // Invariant:
    // - The key must be a valid C++ identifier.
    // - Must not have repeated name with runtime_arg_schema.
    using CompileTimeArgs = Table<std::string, uint32_t>;
    CompileTimeArgs compile_time_args;
    // TODO -- extend to support arbitrary POD types, including user-defined structs.

    //----------------------------------------------------------------------------
    // Runtime argument schema (declaration)

    // Schema (names) for the runtime arguments declared by this kernel.
    // (The values of these arguments are set as ProgramRunArgs.)
    // Currently, only arguments of uint32_t are supported.

    struct RuntimeArgSchema {
        // Runtime argument names
        //
        // Invariant:
        // - Must not have repeated names.
        // - All argument names must be valid C++ identifiers.
        Group<std::string> runtime_arg_names;

        // Common runtime argument names
        //
        // Invariant:
        // - Must not have repeated names.
        // - All argument names must be valid C++ identifiers.
        Group<std::string> common_runtime_arg_names;
    };
    // Invariant:
    // - No repeated names across runtime_arg_names and common_runtime_arg_names.
    // - Must not have repeated names with compile_time_args.
    RuntimeArgSchema runtime_arg_schema{};

    // For vararg-style positional arguments, see KernelAdvancedOptions.

    //////////////////////////////////////////////////////////////////////////////
    // Kernel-controlled hardware resource configuration
    //////////////////////////////////////////////////////////////////////////////

    // Invariant for ComputeHardwareConfig:
    // - Every unpack_modes key names a DFB in this kernel's dfb_bindings (either role).
    // - Gen1: an UnpackToDest entry for a DFB this kernel consumes requires enable_32_bit_dest.
    //
    // Invariant for DataMovementHardwareConfig:
    // - Every config_2xx->disable_dfb_implicit_sync_for entry names a DFB in this kernel's dfb_bindings.
    //
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

// Creates a DFB consumer binding with a BLOCKED access pattern
// Uncomment when BLOCKED support is added (currently TT_FATALs)
/*
inline DFBBinding BlockedConsumerOf(DFBSpecName dfb_spec_name, std::string accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .accessor_name = std::move(accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::BLOCKED,
    };
}
*/

}  // namespace tt::tt_metal::experimental
