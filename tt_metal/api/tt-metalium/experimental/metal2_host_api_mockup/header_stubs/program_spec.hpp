// WorkerSpec describes the configuration of a worker node.
struct WorkerSpec {
    // Kernels, DFBs, and semaphores associated with this worker
    // (They must all have compatible target nodes)
    std::vector<KernelSpec> kernels;
    std::vector<DataflowBufferSpec> dataflow_buffers;
    std::vector<SemaphoreSpec> semaphores;

    // The set of nodes this worker will run on
    std::variant<NodeCoord, NodeRange, NodeRangeSet> nodes;
};


// ProgramSpec describes the immutable aspect of the Program,
// its defining "structure".
struct ProgramSpec {
    std::vector<WorkerSpec> workers;
}


// ProgramRunParams describes the mutable properties of a Program.
// Analogous to function arguments, these are specified for each Program execution.
struct ProgramRunParams {

    ////////////////////////////////////////////////////////////////////////
    // Kernel runtime arguments (legacy style)
    ////////////////////////////////////////////////////////////////////////
    struct KernelRunParams {
        // Kernel identifier
        std::variant<KernelID, KernelName> kernel_id;

        // Runtime arguments (specified per-node)
        using NodeRuntimeArgs = std::vector<uint32_t>; 
        using RuntimeArgs = std::vector<std::pair<CoreCoord, NodeRuntimeArgs>>; 
        RuntimeArgs runtime_args;

        // Common runtime arguments (shared by all cores of a kernel)
        using CommonRuntimeArgs = std::vector<uint32_t>; 
        CommonRuntimeArgs common_runtime_args;
    };
    std::vector<KernelRunParams> kernel_run_params;

    ////////////////////////////////////////////////////////////////////////
    // Kernel runtime arguments (new)
    ////////////////////////////////////////////////////////////////////////
    
    // TODO


    ////////////////////////////////////////////////////////////////////////
    // DFB parameters (optional, advanced use cases)
    ////////////////////////////////////////////////////////////////////////

    struct DFBRunParams {
        // DFB identifier
        std::variant<DFBid, DFBName> dfb_id;

        // DFB size overrides
        // The DFB size specified in the ProgramSpec can be (optionally) overridden per Program execution.
        std::optional<uint32_t> entry_size = std::nullopt;        
        std::optional<uint32_t> num_entries = std::nullopt;

        // DFB borrowed memory
        // For DFBs build on borrowed memory, the underlying memory is passed (as a non-owning view). 
        // (If the DFB is not build on borrowed memory, specifying the borrowed memory will throw an error.)
        using BorrowedMemory = std::variant<BufferView, MeshTensorView>;
        std::optional<BorrowedMemory> borrowed_memory = std::nullopt;
    };

};

