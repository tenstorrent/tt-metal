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
#include <tt-metalium/experimental/metal2_host_api/compute_configuration.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_configuration.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// A name identifying a KernelSpec within a ProgramSpec.
using KernelSpecName = std::string;

// A KernelSpec is a descriptor for a Tenstorrent kernel:
// A single computational task compiled into one or more executable files that work
// collaboratively on a single node.
//
// The KernelSpec describes the properties of a compute or data movement kernel:
//  - Source code
//  - Compiler options for generating the kernel binary/binaries
//  - Resource bindings (access to DFBs, semaphores, etc.)
//  - Kernel argument schema (for arguments specified when the Program is enqueued)
//  - Kernel argument bindings (for compile-time constant arguments)
//  - The configuration of any hardware resources controlled by the kernel
//
// Specialization: A single kernel source may be represented by multiple KernelSpecs in
// the same ProgramSpec — for example with different CTA bindings, different DFB endpoint
// bindings, different semaphore bindings, etc. Each KernelSpec compiles independently
// and is placed independently via WorkUnitSpec membership.
//
// Instancing: A KernelSpec is a *per-node template*. At runtime, one independent
// instance runs on each node where the kernel is placed, with its own runtime arguments.
//
// Placement: The nodes the kernel runs on is derived from WorkUnitSpec membership.
//
struct KernelSpec {
    ///////////////////////////////////////////////////////////////////
    // Basic kernel info
    ///////////////////////////////////////////////////////////////////

    // Kernel identifier: used to reference this kernel within the ProgramSpec
    KernelSpecName unique_id;

    // Kernel source: either a path to a source file, or the source code itself.
    // String literals bind directly to the path variant alternative; wrap inline
    // source code with SourceCode{...}.
    struct SourceCode {
        std::string code;
    };
    std::variant<std::filesystem::path, SourceCode> source;

    // NOTE: The kernel's target node set is a DERIVED property, based on the
    //       WorkUnitSpec(s) that include this kernel.

    // Kernel threading:
    // Number of kernel threads
    uint32_t num_threads = 1;

    // Kernel type (methods)
    bool is_data_movement_kernel() const { return std::holds_alternative<DataMovementConfiguration>(config); }
    bool is_compute_kernel() const { return std::holds_alternative<ComputeConfiguration>(config); }

    ///////////////////////////////////////////////////////////////////
    // Kernel compiler options
    ///////////////////////////////////////////////////////////////////
    struct CompilerOptions {
        using IncludePaths = std::vector<std::filesystem::path>;
        using Defines = std::vector<std::pair<std::string, std::string>>;
        using OptLevel = tt::tt_metal::KernelBuildOptLevel;

        IncludePaths include_paths;         // -I <path>
        Defines defines;                    // -D <name>=<value>
        OptLevel opt_level = OptLevel::O2;  // -O<level>
        // Can add more options here as needed
    };
    CompilerOptions compiler_options = {};

    ///////////////////////////////////////////////////////////////////
    // Resource bindings (immutable Program parameters)
    //////////////////////////////////////////////////////////////////

    // DFB bindings
    // Declares that this kernel requires a DFB resource (declared at the ProgramSpec level)
    // The kernel constructs the accessor via DataflowBufferAccessor(dfb::<local_accessor_name>)
    enum class DFBEndpointType { PRODUCER, CONSUMER };
    struct DFBBinding {
        DFBSpecName dfb_spec_name;        // identify the DFB within the ProgramSpec
        std::string local_accessor_name;  // DFB accessor name (used in the kernel source code)
        DFBEndpointType endpoint_type;    // producer or consumer
        DFBAccessPattern access_pattern = DFBAccessPattern::STRIDED;  // strided, all, or blocked
    };
    std::vector<DFBBinding> dfb_bindings;

    // Semaphore bindings
    // Declares that this kernel accesses a semaphore resource (declared at the ProgramSpec level)
    // The kernel constructs the accessor via SemaphoreAccessor(sem::<local_accessor_name>)
    struct SemaphoreBinding {
        SemaphoreSpecName semaphore_spec_name;  // identify the semaphore within the ProgramSpec
        std::string accessor_name;              // semaphore accessor name (used in the kernel source code)
    };
    std::vector<SemaphoreBinding> semaphore_bindings;

    // Tensor bindings
    // Declares that this kernel accesses a tensor parameter (declared at the ProgramSpec level)
    // The kernel constructs the accessor via TensorAccessor(ta::<accessor_name>)
    struct TensorBinding {
        TensorParameterName tensor_parameter_name;  // identify the TensorBinding within the ProgramSpec
        std::string accessor_name;                  // tensor accessor name (used in the kernel source code)
    };
    std::vector<TensorBinding> tensor_bindings;

    // TODO -- GlobalSemaphore bindings
    // TODO -- GlobalDataflowBuffer bindings

    //////////////////////////////////////////////////////////////////////////////
    // Kernel arguments
    //////////////////////////////////////////////////////////////////////////////

    //----------------------------------------------------------------------------
    // Compile time arguments
    // (Bound argument values cannot be changed between Program executions)
    using CompileTimeArgs = std::vector<std::pair<std::string, uint32_t>>;
    CompileTimeArgs compile_time_args;
    // TODO -- extend to support arbitrary POD types, including user-defined structs.

    //----------------------------------------------------------------------------
    // Runtime argument schema (declaration)

    // Schema for the named runtime arguments declared by this kernel.
    // (The VALUES of these arguments are set as ProgramRunArgs.)
    //
    // Named runtime args: referenced by name in kernel code via `args::<name>`.
    // (Currently, only uint32_t type is supported.)
    //
    // For vararg-style positional arguments, see KernelAdvancedOptions.
    struct RuntimeArgSchema {
        // Runtime argument names (must be unique, valid C++ identifiers.)
        std::vector<std::string> runtime_arg_names;

        // Common runtime argument names (must be unique, valid C++ identifiers.)
        std::vector<std::string> common_runtime_arg_names;
    };
    RuntimeArgSchema runtime_arg_schema{};

    //////////////////////////////////////////////////////////////////////////////
    // Kernel-controlled hardware resource configuration
    //////////////////////////////////////////////////////////////////////////////
    std::variant<DataMovementConfiguration, ComputeConfiguration> config;

    //////////////////////////////////////////////////////////////////////////////
    // Advanced options (see advanced_options.hpp)
    //////////////////////////////////////////////////////////////////////////////
    KernelAdvancedOptions advanced_options;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
