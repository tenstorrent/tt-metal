// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp> 

namespace tt::tt_metal::experimental::metal2_host_api {

typedef KernelSpecID = uint_32;
typedef KernelSpecName = std::string;

struct KernelSpec {

    ///////////////////////////////////////////////////////////////////
    // Basic kernel info
    ///////////////////////////////////////////////////////////////////

    // Kernel identifier
    // A handle used to reference this kernel within the ProgramSpec
    std::variant<KernelSpecID, KernelSpecName> unique_id;

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
    using ThreadNodeMap = std::unordered_map<Nodes, uint8_t>; // node -> number of kernel threads
    std::optional<ThreadNodeMap> thread_node_map = std::nullopt;


    ///////////////////////////////////////////////////////////////////
    // Kernel compiler options
    ///////////////////////////////////////////////////////////////////
    struct CompilerOptions {
        using IncludePaths = std::vector<std::string>;
        using Defines = std::vector<std::pair<std::string, std::string>>;
        using Macros = std::vector<std::string>;

        IncludePaths include_paths = {};   // -I <path>
        Defines defines = {};              // -D <name>=<value>
        OptLevel opt_level = OptLevel::O2; // -O<level>
        // Can add more options here as needed
    };
    CompilerOptions compiler_options = {};


    ///////////////////////////////////////////////////////////////////
    // Resource bindings (immutable Program parameters)
    //////////////////////////////////////////////////////////////////

    // DFB bindings
    enum class DFBEndpointType { PRODUCER, CONSUMER, RELAY };
    struct DFBBinding {
        std::variant<DFBSpecID, DFBSpecName> dfb_spec_id;  // identify the DFB within the ProgramSpec
        std::string local_accessor_name;                   // DFB accessor name (used in the kernel source code)
        DFBEndpointType endpoint_type;                     // producer, consumer, or relay
        DFBAccessPattern access_pattern;                   // strided, blocked, or contiguous
    };
    std::vector<DFBBinding> dfb_bindings;

    // Semaphore bindings
    struct SemaphoreBinding {
        std::variant<SemaphoreSpecId, SemaphoreSpecName> semaphore_spec_id; // identify the semaphore within the ProgramSpec
        std::string accessor_name;                                          // semaphore accessor name (used in the kernel source code)
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
        using NumRTAsPerNode = std::vector<std::pair<NodeCoord, size_t>>; // {node, num_rtas}
        NumRTAsPerNode num_runtime_args_per_node;
        size_t num_common_runtime_args
    };
    RuntimeArgsSchema runtime_arguments_schema;

};

struct ComputeKernelSpec : public KernelSpec {
    // Everything in the base KernelSpec, plus:

    //////////////////////////////////////////////////////////////////////////////
    // Hardware resource configuration
    // Compute kernels control Tensix hardware resources
    //////////////////////////////////////////////////////////////////////////////

    // We will need architecture-specific variants for the hardware resource configs.
    // (It's possible that Quasar compute config will be identical to Gen1...)
    // (If so, we'll merge them.)

    // You can provide a Gen1 config, a Gen2 config, or both.
    // If your host code is intended to be architecture-agnostic, provide both.

    struct Gen1TensixComputeConfig {
        MathFidelity math_fidelity = MathFidelity::HiFi4;
        bool fp32_dest_acc_en = false;
        bool dst_full_sync_en = false;
        bool bfp8_pack_precise = false;
        bool math_approx_mode = false;

        // "Unpack to dest" mode must be specified on a per-DFB basis
        using UnpackToDestMode = std::pair<std::variant<DFBSpecId, DFBSpecName>, UnpackToDestMode>; 
        std::vector<UnpackToDestMode> unpack_to_dest_mode = {}; // empty vector means default mode
    };
    std::optional<Gen1TensixComputeConfig> gen1_tensix_compute_config = std::nullopt;
    

    // Currently looking identical to Gen1... but we'll see!
    struct Gen2TensixComputeConfig {
        MathFidelity math_fidelity = MathFidelity::HiFi4;
        bool fp32_dest_acc_en = false;
        bool dst_full_sync_en = false;
        bool bfp8_pack_precise = false;
        bool math_approx_mode = false;

        // "Unpack to dest" mode must be specified on a per-DFB basis
        using UnpackToDestMode = std::pair<std::variant<DFBSpecId, DFBSpecName>, UnpackToDestMode>; 
        std::vector<UnpackToDestMode> unpack_to_dest_mode = {}; // empty vector means default mode
    };
    std::optional<Gen2TensixComputeConfig> gen2_tensix_compute_config = std::nullopt;
};

struct DataMovementKernelSpec : public KernelSpec {
    // Everything in the base KernelSpec, plus:

    //////////////////////////////////////////////////////////////////////////////
    // Hardware resource configuration
    // Compute kernels control Tensix hardware resources
    //////////////////////////////////////////////////////////////////////////////

    // We will need architecture-specific variants for the hardware resource configs.

    // You can provide either a Gen1 config, a Gen2 config, or both.
    // If your host code is intended to run on Gen1 or Gen2, you should provide both.

    struct Gen1DataMovementConfig {
        DataMovementProcessor processor = DataMovementProcessor::RISCV_0;  // TODO: Paul wanted to be rid of this
        NOC noc = NOC::RISCV_0_default;                                    // TODO: This too
        NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;
    };

    struct Gen2DataMovementConfig {
        bool is_dm = true; // placeholder so I don't have an empty struct
        // TODO
        //   The user doesn't specify the DM processor (the runtime chooses).
        //   There's only one NOC.
        //   Is there anything in here? It's possible we don't need this.
    };
};

}  // namespace tt::tt_metal::experimental::metal2_host_api