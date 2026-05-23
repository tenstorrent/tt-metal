// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/compute_configuration.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_configuration.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>

namespace tt::tt_metal::experimental {

// A name identifying a KernelSpec within a ProgramSpec.
// String literals work directly; misnamed references fail at validation.
using KernelSpecName = std::string;

// A KernelSpec is a descriptor for a Tenstorrent kernel:
// A single computational task compiled into one or more executable files that work
// collaboratively on a single node.
//
// The KernelSpec describes the properties of a compute or data movement kernel:
//  - Source code
//  - Compiler options for generating the kernel binary/binaries
//  - Resource bindings (access to DFBs, semaphores, tensors)
//  - Kernel argument schema (for arguments specified when the Program is enqueued)
//  - Compile-time kernel arguments (constexpr args specified at Program compile time)
//  - The configuration of any hardware resources controlled by the kernel
//
// Specialization: A single kernel source may be represented by multiple KernelSpecs in
// the same ProgramSpec — for example with different CTA bindings, different DFB endpoint
// bindings, different semaphore bindings, etc. Each KernelSpec compiles independently
// and is placed independently via WorkUnitSpec membership.
//
// Instancing: A KernelSpec is a *per-node template*. At runtime, one independent
// instance runs on each node where the kernel is placed, with its own runtime arguments.
// (Compile-time arguments and common runtime arguments are across all instances.)
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
    // String literals are accepted directly as a file path (the common case);
    // wrap inline source code with SourceCode{...}.
    struct SourceCode {
        std::string code;
    };
    std::variant<std::filesystem::path, SourceCode> source;

    // NOTE: The kernel's target node set is a DERIVED property, based on the
    //       WorkUnitSpec(s) that include this kernel.

    // Number of kernel threads
    int num_threads = 1;

    // (Optional) Per-node thread count specification
    // The default threading is num_threads. However, you may override this on a per-node basis.
    // NOTE: This feature is currently unsupported. It's an open question if we EVER want to support it.
    //       Here as a placeholder; specifying it will trigger a runtime error.
    using NodeSpecificThreadCount = std::pair<Nodes, int>;  // {node_set, num_threads}
    using NodeSpecificThreadCounts = std::vector<NodeSpecificThreadCount>;
    std::optional<NodeSpecificThreadCounts> node_specific_thread_counts = std::nullopt;

    // Kernel type (methods)
    bool is_dm_kernel() const { return std::holds_alternative<DataMovementConfiguration>(config_spec); }
    bool is_compute_kernel() const { return std::holds_alternative<ComputeConfiguration>(config_spec); }

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
    // Compile time argument bindings
    // (Bound argument values cannot be changed between Program executions)
    using CompileTimeArgs = std::vector<std::pair<std::string, uint32_t>>;
    CompileTimeArgs compile_time_args;
    // TODO -- extend to support arbitrary POD types, including user-defined structs.

    //----------------------------------------------------------------------------
    // Runtime argument schema (declaration)

    // Schema for runtime arguments (RTA) and common runtime arguments (CRTA)
    // (The VALUES of these arguments are set as ProgramRunParams.)
    //
    // Two mechanisms are supported per kernel:
    //   - Named RTAs/CRTAs: referenced by name in kernel code via `args::<name>`.
    //     (Currently, only uint32_t type is supported.)
    //   - Vararg RTAs/CRTAs: positional, variable-count, always uint32_t.
    //     Indexed from 0 in kernel code via `get_vararg(idx)` / `get_common_vararg(idx)`.
    //     Vararg indices are stable across schema changes (e.g., moving a named arg from RTA→CRTA).
    struct RuntimeArgSchema {
        // Named RTAs: names in declaration order. Must be unique valid C++ identifiers.
        std::vector<std::string> runtime_args;

        // Named CRTAs: names in declaration order. Must be unique valid C++ identifiers.
        std::vector<std::string> common_runtime_args;

        //----------------------
        // Advanced options

        // Runtime varargs: dynamic RTAs
        // Some kernels are designed to take a variable number of arguments.
        //  e.g. N arguments representing the dimensions of an N-dimensional tensor,
        //       where N is passed to the kernel as a CTA.
        // Varargs are accessed positionally, since the kernel does not know how many to expect.
        // The vararg schema specifies the number of RTA varargs for this kernel.
        // Use ProgramRunParams to set the vararg values (per node).
        size_t num_runtime_varargs = 0;

        // Per-node vararg number override: different per-node vararg counts
        // In very rare cases, the kernel running on different nodes requires a DIFFERENT
        // number of varargs on different nodes.
        // Use num_runtime_varargs_per_node to override the number of varargs.
        // Any kernel target node not specified in the override defaults to num_runtime_varargs.
        using NumVarargsPerNode = std::vector<std::pair<Nodes, size_t>>;  // {nodes, num_varargs}
        std::optional<NumVarargsPerNode> num_runtime_varargs_per_node = std::nullopt;
        // TODO: This feature is truly bizarre. Investigate removing it from the API.

        // Common runtime varargs: dynamic number of CRTAs
        // These are similar to runtime varargs. However, when specifying the argument values
        // (in ProgramRunParams), all nodes of the kernel receive the common values.
        size_t num_common_runtime_varargs = 0;
    };
    RuntimeArgSchema runtime_arguments_schema{};

    //////////////////////////////////////////////////////////////////////////////
    // Kernel-controlled hardware resource configuration
    //////////////////////////////////////////////////////////////////////////////
    using ConfigSpec = std::variant<DataMovementConfiguration, ComputeConfiguration>;
    ConfigSpec config_spec;

    //////////////////////////////////////////////////////////////////////////////
    // Advanced options / niche use cases
    //////////////////////////////////////////////////////////////////////////////

    // Niche use case: Self-loop DFBs on compute kernels only
    // This applies only to compute kernels that bind BOTH the producer and consumer
    // endpoints of the same DFB (self-loop).
    //
    // The compute kernel threads can communicate via the DFB in two topologies:
    //
    //   INTRA (intra-thread): Each kernel thread uses the DFB in its own self-loop.
    //         (no cross-thread communication). This is the common case.
    //   INTER (inter-thread): Within the kernel, some threads produce data for other
    //          threads to consume.
    //
    // Only the INTRA case is currently supported. INTER will trigger a validation error.
    // There are currently no known use cases for an INTER-thread self-loop. This option
    // is present in the API for completeness, to surface any use cases that may arise.
    //
    struct DFBComputeSelfLoopScope {
        DFBSpecName dfb_spec_name;
        enum class Scope { INTRA, INTER };
        Scope scope = Scope::INTRA;
        // If the INTER case were enabled, we would need an additional field to describe
        // the inter-thread communication pattern here.
    };
    std::vector<DFBComputeSelfLoopScope> dfb_compute_self_loop_scopes;
};

//////////////////////////////////////////////////////////////////////////////
// Namespace-level aliases for commonly-used nested types
//////////////////////////////////////////////////////////////////////////////

// The qualified (KernelSpec::Foo) and unqualified (Foo) forms refer to the same type.
using DFBBinding = KernelSpec::DFBBinding;
using DFBEndpointType = KernelSpec::DFBEndpointType;
using SemaphoreBinding = KernelSpec::SemaphoreBinding;
using TensorBinding = KernelSpec::TensorBinding;
using SourceCode = KernelSpec::SourceCode;

//////////////////////////////////////////////////////////////////////////////
// Convenience factories for DFBBinding
//////////////////////////////////////////////////////////////////////////////

// Ergonomic alternatives to writing a designated-init DFBBinding{...}

// Creates a DFB producer binding with a STRIDED access pattern
// (All DFB producers are STRIDED)
inline DFBBinding ProducerOf(DFBSpecName dfb_spec_name, std::string local_accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .local_accessor_name = std::move(local_accessor_name),
        .endpoint_type = DFBEndpointType::PRODUCER,
        .access_pattern = DFBAccessPattern::STRIDED};
}

// Creates a DFB consumer binding (with a default-STRIDED access pattern)
// Use this for single-threaded kernels, where the access pattern doesn't matter.
// For multi-threaded kernels (Quasar), prefer the explicit access pattern
// helper factories below.
inline DFBBinding ConsumerOf(DFBSpecName dfb_spec_name, std::string local_accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .local_accessor_name = std::move(local_accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        // access pattern defaults to STRIDED
    };
}

// Creates a DFB consumer binding with a STRIDED access pattern
// (The common case for multi-threaded DFB consumers)
inline DFBBinding StridedConsumerOf(DFBSpecName dfb_spec_name, std::string local_accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .local_accessor_name = std::move(local_accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::STRIDED,
    };
}

// Creates a DFB consumer binding with a ALL access pattern
inline DFBBinding AllConsumerOf(DFBSpecName dfb_spec_name, std::string local_accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .local_accessor_name = std::move(local_accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::ALL,
    };
}

// Creates a DFB consumer binding with a BLOCKED access pattern
// Uncomment when BLOCKED support is added (currently TT_FATALs)
/*
inline DFBBinding BlockedConsumerOf(
    DFBSpecName dfb_spec_name,
    std::string local_accessor_name) {
    return DFBBinding{
        .dfb_spec_name = std::move(dfb_spec_name),
        .local_accessor_name = std::move(local_accessor_name),
        .endpoint_type = DFBEndpointType::CONSUMER,
        .access_pattern = DFBAccessPattern::BLOCKED,
    };
}
*/

//////////////////////////////////////////////////////////////////////////////
// Convenience factories for SemaphoreBinding and TensorBinding
//////////////////////////////////////////////////////////////////////////////

// Creates a SemaphoreBinding (kernel uses a Semaphore declared at ProgramSpec level)
inline SemaphoreBinding UseSemaphore(SemaphoreSpecName semaphore_spec_name, std::string accessor_name) {
    return SemaphoreBinding{
        .semaphore_spec_name = std::move(semaphore_spec_name),
        .accessor_name = std::move(accessor_name),
    };
}

// Creates a TensorBinding (kernel accesses a TensorParameter declared at ProgramSpec level)
inline TensorBinding UseTensor(TensorParameterName tensor_parameter_name, std::string accessor_name) {
    return TensorBinding{
        .tensor_parameter_name = std::move(tensor_parameter_name),
        .accessor_name = std::move(accessor_name),
    };
}

}  // namespace tt::tt_metal::experimental
