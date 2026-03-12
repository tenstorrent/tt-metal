#if 0

////////////////////////////////////////////////////////////
// Simple Program with multi-threaded kernels and local DFBs.

// This is a trivial example, intended mainly to illustrate the 
// new DFB syntax. It has only one type of worker.
// ////////////////////////////////////////////////////////////


// Note: The overloaded meaning of "core" is especially confusing 
// on Quasar! We reserve the term "core" for RISC-V cores.
// We'll  use the term "Node" for the physical NOC endpoints.



////////////////////////////////////////////////////////////
// Basis worker nodes
////////////////////////////////////////////////////////////

// All the nodes in this set will be identically configured.
NodeRange type1_nodes = NodeRange(NodeCoord{0, 0}, NodeCoord{3, 1});


//-------------------
// Dataflow Buffers

// Basic local DFB (DM -> compute)
DataflowBufferSpec compute_feeder_dfb_spec{   
    .identifier = "compute_feeder", 
    .target_nodes = {
        .nodes = type1_nodes,
        // for remote DFB, the producer-consumer map would be here.
    }
    .backing_memory = {
        .entry_size = 32, // bytes
        .num_entries = 8
    }
    // Endpoint info is specified when the DFB is bound to a kernel.
};

// Basic local DFB (compute -> DM)
DataflowBufferSpec compute_output_dfb_spec{   
    .identifier = "compute_output",
    .target_nodes = {
        .nodes = type1_nodes,
    }
    .backing_memory = {
        .entry_size = 32, // bytes
        .num_entries = 8
    },
    // Endpoint info is specified when the DFB is bound to a kernel.
};

////////////////////////////////////////////////////////////
// Kernels
////////////////////////////////////////////////////////////

// Reader kernel
KernelSpec reader_kernel_spec{
    .identifier = "reader_kernel",
    .source = "reader_kernel.cpp",
    .target_nodes = {         
        .nodes = type1_nodes, 
        .num_threads = 4,      
    }
    .compiler_options = {
        .include_paths = {"../somewhere/my_kernels"}
    },
    .dfb_bindings = {
        DFBBinding{.dfb_id = "compute_feeder", 
                   .local_accessor_name = "dfb_out", 
                   .endpoint_type = DFBEndpointType::PRODUCER,
                   .access_pattern = DFBAccessPattern::STRIDED}
    },
    .kernel_args_spec = {
        // Coming soon...
    },
    .hardware_config = Gen2DataMovementConfig{};
};


// Compute kernel
KernelSpec consumer_kernel_spec{
    .source = "compute_kernel.cpp",
    .target_nodes = {          // different from Option 2
        .nodes = type1_nodes, 
        .num_threads = 4,
    }
    .compiler_options = {
        .include_paths = {"../somewhere/my_kernels"}
    },
    .dfb_bindings = {
        DFBBinding{.dfb_id = "compute_feeder", 
                   .local_accessor_name = "dfb_in",  
                   .endpoint_type = DFBEndpointType::CONSUMER,
                   .access_pattern = DFBAccessPattern::BLOCKED}
        DFBBinding{.dfb_id = "compute_output", 
                   .local_accessor_name = "dfb_out", 
                   .endpoint_type = DFBEndpointType::PRODUCER,
                   .access_pattern = DFBAccessPattern::STRIDED}
    },
    .kernel_args_spec = {
        // Coming soon...
    },  
    .hardware_config = Gen2TensixComputeConfig{}; // just accept defaults
};

// Writer kernel
KernelSpec writer_kernel_spec{
    .source = "writer_kernel.cpp",
    .target_nodes = {       
        .nodes = type1_nodes, 
        .num_threads = 4,
    }
    .compiler_options = {
        .include_paths = {"../somewhere/my_kernels"}
    },
    .dfb_bindings = {
        DFBBinding{.dfb_id = "compute_output", 
                   .local_accessor_name = "dfb_in", 
                   .endpoint_type = DFBEndpointType::CONSUMER,
                   .access_pattern = DFBAccessPattern::STRIDED}
    },
    .kernel_args_spec = {
        // Coming soon...
    },
    .hardware_config = Gen2DataMovementConfig{};
};

////////////////////////////////////////////////////////////
// Program
////////////////////////////////////////////////////////////

// Create a worker spec for this node type
WorkerSpec type1_worker_spec{
    .kernels = { 
        reader_kernel_spec,
        compute_kernel_spec,
        writer_kernel_spec,
    },
    .dataflow_buffers = {
        compute_feeder_dfb_spec,
        compute_output_dfb_spec,
    }
    .nodes = type1_nodes,
};

// Note: A KernelSpec, DataflowBufferSpec, and SemaphoreSpec can be used in multiple
//   WorkerSpecs. But, all of the WorkerSpec's constituent objects target nodes must
//   include the WorkerSpec's target nodes.
//   This is enforced by the legality check in the Program constructor.


// The ProgramSpec just holds a vector of WorkerSpecs.
ProgramSpec program_spec{
    .workers = {
        type1_worker_spec
    }
};


// Create the program object
Program simple_program(simple_program_spec);
// Alternatively, we could just pass in a vector of WorkerSpecs directly.
// Program simple_program({worker_type1_spec, worker_type2_spec, ...});

// Add to MeshWorkload 
// This is a transfer of ownership -- note the std::move!
MeshWorklad mesh_workload;
mesh_workload.add_program(target_device_range, std::move(simple_program));

// This is dumb... should change it to:
// my_mesh_workload.add_program(target_device_range, simple_program_spec);


////////////////////////////////////////////////////////////
// Program run parameters
////////////////////////////////////////////////////////////

ProgramRunParams program_run_params{
    .kernel_run_params = {
        KernelRunParams{.kernel_id = "reader_kernel",
                        .runtime_args = {/*TODO*/},
                        .common_runtime_args = {/*TODO*/},
                        {.kernel_id = "compute_kernel",
                         .runtime_args = {/*TODO*/  },
                         .common_runtime_args = {/*TODO*/},
                        },
                        {.kernel_id = "writer_kernel",
                         .runtime_args = {/*TODO*/},
                         .common_runtime_args = {/*TODO*/},
                        },
    }
    // No DFB run params needed
    // (Only used in rare/niche use cases.)
};

// The syntax to retrieve the program is a bit clunky....
auto& programs = mesh_workload.get_programs();
auto& [coord_range, program] = *programs.begin(); 


// The descriptor-based syntax is nice, but it performs a copy of
// the kernel runtime arguments vector into the dispatch buffer.
program.set_run_parameters(program_run_params);

// Alternative, you can use more verbose APIs to construct
// the runtime arguments vector in place. (TBD)

#endif