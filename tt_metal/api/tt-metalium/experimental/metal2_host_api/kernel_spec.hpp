// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
    using UnpackToDestModeEntry = std::pair<DFBSpecName, ::UnpackToDestMode>;
    std::vector<UnpackToDestModeEntry> unpack_to_dest_mode = {};  // empty vector means default mode
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

using KernelSpecID = uint32_t;
using KernelSpecName = std::string;

struct KernelSpec {
    ///////////////////////////////////////////////////////////////////
    // Basic kernel info
    ///////////////////////////////////////////////////////////////////

    // Kernel identifier: used to reference this kernel within the ProgramSpec
    KernelSpecName unique_id;

    // Kernel source
    std::string source;
    enum class SourceType { FILE_PATH, SOURCE_CODE };
    SourceType source_type = SourceType::FILE_PATH;

    // Target nodes
    // The set of device nodes on which the kernel will run
    using Nodes = std::variant<NodeCoord, NodeRange, NodeRangeSet>;
    Nodes target_nodes;

    // Threading
    // Number of kernel threads (this can be specified globally or per-node)
    uint8_t num_threads = 1;
    using ThreadNodeMap = std::unordered_map<Nodes, uint8_t>;  // node -> number of kernel threads
    std::optional<ThreadNodeMap> thread_node_map = std::nullopt;

    // Kernel type (methods)
    bool is_dm_kernel() const { return std::holds_alternative<DataMovementConfiguration>(config_spec); }
    bool is_compute_kernel() const { return std::holds_alternative<ComputeConfiguration>(config_spec); }

    ///////////////////////////////////////////////////////////////////
    // Kernel compiler options
    ///////////////////////////////////////////////////////////////////
    struct CompilerOptions {
        using IncludePaths = std::vector<std::string>;
        using Defines = std::vector<std::pair<std::string, std::string>>;
        using Macros = std::vector<std::string>;
        using OptLevel = tt::tt_metal::KernelBuildOptLevel;

        IncludePaths include_paths = {};    // -I <path>
        Defines defines = {};               // -D <name>=<value>
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
    // TODO -- Socket bindings

    // Compile time argument bindings (values cannot be changed between Program executions)
    using CompileTimeArgBindings = std::unordered_map<std::string, uint32_t>;
    CompileTimeArgBindings compile_time_arg_bindings;
    // TODO -- extend to support arbitrary POD types, including user-defined structs.

    //////////////////////////////////////////////////////////////////////////////
    // Runtime argument schema / declaration
    //////////////////////////////////////////////////////////////////////////////

    // Schema for runtime and common runtime arguments
    // (The VALUES of these arguments are set as ProgramRunParams.)
    struct RuntimeArgSchema {
        // Schema for named and typed RTAs + CRTAs
        // (These must be fully specified in the kernel code.)
        //   TODO

        // Schema for unnamed/variable RTAs + CRTAs
        // (Must be of uint32_t; can be treated as varargs in the kernel code)
        using NumRTAsPerNode = std::vector<std::pair<NodeCoord, size_t>>;  // {node, num_rtas}
        NumRTAsPerNode num_runtime_args_per_node;
        size_t num_common_runtime_args;
    };
    RuntimeArgSchema runtime_arguments_schema;

    //////////////////////////////////////////////////////////////////////////////
    // Kernel-controlled hardware resource configuration
    //////////////////////////////////////////////////////////////////////////////
    using ConfigSpec = std::variant<DataMovementConfiguration, ComputeConfiguration>;
    ConfigSpec config_spec;
};

}  // namespace tt::tt_metal::experimental::metal2_host_api
