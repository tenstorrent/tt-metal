
enum class DFBAccessPattern {STRIDED, BLOCKED, CONTIGUOUS}; // this appears twice, fix it
typedef KernelID = uint_32;
typedef KernelName = std::string;

struct KernelSpec {

    ///////////////////////////////////////////////////////////////////
    // Basic kernel info
    ///////////////////////////////////////////////////////////////////

    // Kernel identifier
    // A handle used to reference this kernel within the ProgramSpec
    std::variant<KernelID, KernelName> unique_id;
    // (I intend to remove either the string or uint32_t option. Having both is annoying. Thoughts?)

    // Kernel source
    std::string source;
    enum class SourceType { FILE_PATH, SOURCE_CODE };
    SourceType  source_type = SourceType::FILE_PATH;

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
    // Passed to the kernel compiler at Program creation
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
    // Resource bindings
    // Bindings (unlike arguments) are immutable Program parameters
    //////////////////////////////////////////////////////////////////

    // Compile time argument bindings
    // CTAs are bound at Program creation; they cannot be changed between Program executions
    using CompileTimeArgBindings = std::unordered_map<std::string, uint32_t>;
    CompileTimeArgBindings compile_time_arg_bindings;
        // TODO -- We want to permit arbitrary POD types, including user-defined structs.


    // DFB bindings
    enum class DFBEndpointType { PRODUCER, CONSUMER, RELAY };
    struct DFBBinding {
        std::variant<DFBid, DFBName> dfb_id;  // identify the DFB within the ProgramSpec
        std::string local_accessor_name;      // DFB accessor name (used in the kernel source code, via DFBAccessor)
        DFBEndpointType endpoint_type;        // producer, consumer, or relay
        DFBAccessPattern access_pattern;      // strided, blocked, or contiguous
    };
    std::vector<DFBBinding> dfb_bindings;


    // Semaphore bindings
    struct SemaphoreBinding {
        std::variant<SemaphoreID, std::string> semaphore_id; // identify the semaphore within the ProgramSpec
        std::string accessor_name;                           // semaphore accessor name (used in the kernel source code)
    };
    std::vector<SemaphoreBinding> semaphore_bindings;


    // TODO -- GlobalSemaphore bindings
    // TODO -- GlobalDataflowBuffer bindings
    // TODO -- Socket bindings (might replace GlobalDataflowBuffer?)


    //////////////////////////////////////////////////////////////////////////////
    // Runtime argument schema
    // RTAs and CTAs are declared here
    // (Their values are set as program execution parameters)
    //////////////////////////////////////////////////////////////////////////////

    // "Legacy" specification for runtime and common runtime arguments
    // (No support for named or typed arguments; just a vector of uint32_t)
    // Recall that "Spec" represents the immutable aspect of the Program -- 
    // for legacy-style kernel arguments, only the number of RTAs and CRTAs are specified. 
    // The argument VALUES are set as program execution parameters.
    struct LegaryKernelArgumentsSpec {
        using NumRTAsPerNode = std::vector<std::pair<NodeCoord, size_t>>; // {node, num_rtas}
        NumRTAsPerNode num_runtime_args_per_node;
        size_t num_common_runtime_args
    };
    LegacyKernelArgumentsSpec legacy_kernel_args_spec;
    
    // "New" specification for runtime and common runtime arguments
    // Supports:
    //   - named and typed arguments, including user-defined structs
    //   - additional varargs
    struct KernelArgumentsSpec {
        // TODO
    };
    NewKernelArgumentsSpec new_kernel_args_spec;

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
        using UnpackToDestMode = std::pair<std::variant<DFBid, std::string>, UnpackToDestMode>; 
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
        using UnpackToDestMode = std::pair<std::variant<DFBid, std::string>, UnpackToDestMode>; 
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
        // TODO
        //   The user doesn't specify the DM processor; the runtime chooses.
        //   There's only one NOC.
        //   Is there anything in here?
    };
};