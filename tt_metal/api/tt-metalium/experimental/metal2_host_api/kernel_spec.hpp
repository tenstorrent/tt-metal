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

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/base_types.hpp>    // For MathFidelity, UnpackToDestMode (global scope)
#include <tt-metalium/kernel_types.hpp>  // For DataMovementProcessor, NOC, etc.

namespace tt::tt_metal::experimental::metal2_host_api {

struct ComputeConfiguration {
    // Tensix hardware resource configuration (configured by compute kernels)
    // Gen1 and Gen2 configurations are currently identical.

    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;

    // "Unpack to dest" mode must be specified on a per-DFB basis
    // unpack_to_dest_mode maps DFB identifier to UnpackToDestMode
    using UnpackToDestModeEntry = std::pair<DFBSpecName, tt::tt_metal::UnpackToDestMode>;
    std::vector<UnpackToDestModeEntry> unpack_to_dest_mode;
};

struct DataMovementConfiguration {
    // The DM configuration is different for Gen1 and Gen2.
    // You can provide either a Gen1 config, a Gen2 config, or both.
    // If your host code is intended to be architecture-agnostic, provide both.

    struct Gen1DataMovementConfig {
        tt::tt_metal::DataMovementProcessor processor = tt::tt_metal::DataMovementProcessor::RISCV_0;
        tt::tt_metal::NOC noc = tt::tt_metal::NOC::RISCV_0_default;
        tt::tt_metal::NOC_MODE noc_mode = tt::tt_metal::NOC_MODE::DM_DEDICATED_NOC;
    };
    std::optional<Gen1DataMovementConfig> gen1_data_movement_config = std::nullopt;

    struct Gen2DataMovementConfig {
        // Currently, no configuration is needed for Gen2!
        // The empty struct is still used to express a Gen2 DM kernel.
    };
    std::optional<Gen2DataMovementConfig> gen2_data_movement_config = std::nullopt;
};

// A name identifying a KernelSpec within a ProgramSpec.
//
// CONVENTION: define names as `constexpr const char*` constants, e.g.:
//   constexpr const char* READER_KERNEL = "reader";
//   KernelSpec{.unique_id = READER_KERNEL, ...};
// Reusing a single constant helps catch typos and errors at compile time.
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
    // (Force callers to choose explicitly between path and inline code.)
    struct SourceFilePath {
        std::filesystem::path path;
    };
    struct SourceCode {
        std::string code;
    };
    std::variant<SourceFilePath, SourceCode> source;

    // NOTE: The kernel's target node set is a DERIVED property, based on the
    //       WorkUnitSpec(s) that include this kernel.

    // Kernel threading:
    // Number of kernel threads
    uint8_t num_threads = 1;

    // (Optional) Per-node thread count specification
    // The default threading is num_threads. However, you may override this on a per-node basis.
    // NOTE: This feature is currently unsupported. It's an open question if we EVER want to support it.
    //       Here as a placeholder; specifying it will trigger a runtime error.
    using Nodes = std::variant<NodeCoord, NodeRange, NodeRangeSet>;
    using NodeSpecificThreadCount = std::pair<Nodes, uint8_t>;  // {node_set, num_threads}
    using NodeSpecificThreadCounts = std::vector<NodeSpecificThreadCount>;
    std::optional<NodeSpecificThreadCounts> node_specific_thread_counts = std::nullopt;

    // Kernel type (methods)
    bool is_dm_kernel() const { return std::holds_alternative<DataMovementConfiguration>(config_spec); }
    bool is_compute_kernel() const { return std::holds_alternative<ComputeConfiguration>(config_spec); }

    ///////////////////////////////////////////////////////////////////
    // Kernel compiler options
    ///////////////////////////////////////////////////////////////////
    struct CompilerOptions {
        using IncludePaths = std::vector<std::string>;
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
    enum class DFBEndpointType { PRODUCER, CONSUMER, RELAY };
    struct DFBBinding {
        DFBSpecName dfb_spec_name;        // identify the DFB within the ProgramSpec
        std::string local_accessor_name;  // DFB accessor name (used in the kernel source code)
        DFBEndpointType endpoint_type;    // producer, consumer, or relay
        DFBAccessPattern access_pattern;  // strided, blocked, or contiguous
    };
    std::vector<DFBBinding> dfb_bindings;

    // Semaphore bindings
    struct SemaphoreBinding {
        SemaphoreSpecName semaphore_spec_name;  // identify the semaphore within the ProgramSpec
        std::string accessor_name;              // semaphore accessor name (used in the kernel source code)
    };
    std::vector<SemaphoreBinding> semaphore_bindings;

    // TODO -- GlobalSemaphore bindings
    // TODO -- GlobalDataflowBuffer bindings

    //////////////////////////////////////////////////////////////////////////////
    // Kernel arguments
    //////////////////////////////////////////////////////////////////////////////

    //----------------------------------------------------------------------------
    // Compile time argument bindings
    // (Bound argument values cannot be changed between Program executions)
    using CompileTimeArgBindings = std::vector<std::pair<std::string, uint32_t>>;
    CompileTimeArgBindings compile_time_arg_bindings;
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
        std::vector<std::string> named_runtime_args;

        // Named CRTAs: names in declaration order. Must be unique valid C++ identifiers.
        std::vector<std::string> named_common_runtime_args;

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
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
