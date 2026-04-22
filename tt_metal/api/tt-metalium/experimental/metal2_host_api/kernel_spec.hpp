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
        // Might want to revisit the API if so....
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

struct KernelSpec {
    ///////////////////////////////////////////////////////////////////
    // Basic kernel info
    ///////////////////////////////////////////////////////////////////

    // Kernel identifier: used to reference this kernel within the ProgramSpec
    KernelSpecName unique_id;

    // Kernel source: either a path to a source file, or the source code itself.
    // (Wrapper types disambiguate the string-constructible variant alternatives,
    // ensuring compile-time enforcement.)
    struct SourceFilePath {
        std::filesystem::path path;
    };
    struct SourceCode {
        std::string code;
    };
    std::variant<SourceFilePath, SourceCode> source;

    // Target nodes
    // The logical coordinates for the set of device nodes on which the kernel will run
    using Nodes = std::variant<NodeCoord, NodeRange, NodeRangeSet>;
    Nodes target_nodes;

    // Threading
    // Number of kernel threads (this can be specified globally or per-node)
    uint8_t num_threads = 1;
    // Optional per-node thread count specification (overrides global num_threads)
    // This is currently unsupported, and an open question if we ever want to support it.
    using NodeSpecificThreadCount = std::pair<Nodes, uint8_t>;  // {node, num_threads}
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

    // Namespace for the kernel argument accessors in the kernel source code, e.g.
    //   auto my_arg = get_arg(my_args_namespace::my_arg_name);
    // Use a custom namespace to avoid identifier collisions when fusing kernels;
    // otherwise, use the default "args" namespace.
    std::string args_namespace = "args";

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
    //   - Named RTAs/CRTAs: referenced by name in kernel code via `<args_namespace>::<name>`.
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

        // Vararg RTAs: dynamic number of RTAs
        // The RTA varargs count is dynamic with respect to the kernel source code, but immutable
        // with respect to the Program. (It cannot be changed in ProgramRunParams.)
        // Every node the kernel runs on gets this same vararg count, but unique vararg values.
        // Values are specified in the ProgramRunParams.
        // NOTE: RTA varargs can also also be useful for legacy migration.
        size_t num_runtime_varargs = 0;

        // Per-node RTA vararg override: different per-node vararg counts
        // Still immutable with respect to the Program, but varargs count can be specified per-node.
        // Any kernel target node not specified in the override defaults to num_runtime_varargs.
        using NumVarargsPerNode = std::vector<std::pair<Nodes, size_t>>;  // {nodes, num_varargs}
        std::optional<NumVarargsPerNode> num_runtime_varargs_per_node = std::nullopt;

        // Vararg CRTAs: dynamic number of CRTAs
        // The CTRA varargs count is dynamic with respect to the kernel source code, but immutable
        // with respect to the Program. (It cannot be changed in ProgramRunParams.)
        // Values are specified in the ProgramRunParams; they are common to all kernel nodes.
        // NOTE: RTA varargs can also also be useful for legacy migration.
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
