# Matmul (Single Core)

We'll build a program that will perform matmul operations on two tensors with equal-size inner dimension. We will then go through specific sections of the program.

The full example program is in
[matmul_single_core.cpp](matmul_single_core.cpp)

To build and execute, you may use the following commands:
Then run the following:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_matmul_single_core
```

## Mesh setup (host)

We use the Mesh API: create a 1x1 mesh device, get the mesh command queue, construct a workload and device range, and describe the program with a `ProgramSpec`. Then prepare inputs, tilize, call `matmul_single_core`, untilize, and validate.

## Main blocks in matmul_single_core function

We will go through sections of the `matmul_single_core` function:

-   Program, enqueue and core range settings
-   Create DRAM tensors based on input and output vectors
-   Create L1 dataflow buffers
-   Kernels declarations and related compile and runtime arguments
-   Program launch and reading data from DRAM output tensor to result vector

## Create Program, Enqueue initialization, and core range definition

We want a just a single core, so we will restrict the core range to be just one core at (0, 0).

``` cpp
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
// Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
tt_metal::experimental::NodeCoord core({0, 0});
```

## Create DRAM tensors & dataflow buffers

In terms of DRAM tensors, we need two source tensors and one destination tensor.
Each name below identifies a kernel, a dataflow buffer, or a tensor parameter within the program spec.

``` cpp
// Calcaulate the number of tiles for each dimension.
uint32_t Mt = M / TILE_HEIGHT;
uint32_t Kt = K / TILE_WIDTH;
uint32_t Nt = N / TILE_WIDTH;

// Create DRAM tensors for the input and output data.
uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
uint32_t elements_per_tile = single_tile_size / sizeof(bfloat16);

auto make_dram_tensor_spec = [&](size_t num_elements) {
    return TensorSpec(
        Shape({static_cast<uint32_t>(num_elements / elements_per_tile), elements_per_tile}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
};
TensorSpec src0_tensor_spec = make_dram_tensor_spec(a.size());
TensorSpec src1_tensor_spec = make_dram_tensor_spec(b.size());
TensorSpec dst_tensor_spec = make_dram_tensor_spec(output.size());

MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src0_tensor_spec);
MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, src1_tensor_spec);
MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_tensor_spec);

// Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
const tt_metal::experimental::KernelSpecName reader_kernel{"reader"};
const tt_metal::experimental::KernelSpecName writer_kernel{"writer"};
const tt_metal::experimental::KernelSpecName compute_kernel{"compute"};
const tt_metal::experimental::DFBSpecName src0_dfb{"src0"};
const tt_metal::experimental::DFBSpecName src1_dfb{"src1"};
const tt_metal::experimental::DFBSpecName output_dfb{"output"};
const tt_metal::experimental::TensorParamName src0_tensor_param{"src0"};
const tt_metal::experimental::TensorParamName src1_tensor_param{"src1"};
const tt_metal::experimental::TensorParamName dst_tensor_param{"dst"};

tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
uint32_t num_input_tiles = 2;
tt_metal::experimental::DataflowBufferSpec src0_dfb_spec{
    .unique_id = src0_dfb,
    .entry_size = single_tile_size,
    .num_entries = num_input_tiles,
    .data_format_metadata = cb_data_format,
};

tt_metal::experimental::DataflowBufferSpec src1_dfb_spec{
    .unique_id = src1_dfb,
    .entry_size = single_tile_size,
    .num_entries = num_input_tiles,
    .data_format_metadata = cb_data_format,
};

uint32_t num_output_tiles = 2;
tt_metal::experimental::DataflowBufferSpec output_dfb_spec{
    .unique_id = output_dfb,
    .entry_size = single_tile_size,
    .num_entries = num_output_tiles,
    .data_format_metadata = cb_data_format,
};
```

## Compile-time kernels arguments

We have to declare some compile-time arguments for the compute kernel.
They are named compile-time arguments in its `KernelSpec`, and the kernel reads each one by name (for example `get_arg(args::Mt)`).

``` cpp
tt_metal::experimental::KernelSpec compute_kernel_spec{
    .unique_id = compute_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/compute/mm.cpp",
    .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
    .dfb_bindings =
        {tt_metal::experimental::ConsumerOf(src0_dfb, "in0"),
         tt_metal::experimental::ConsumerOf(src1_dfb, "in1"),
         tt_metal::experimental::ProducerOf(output_dfb, "out0")},
    .compile_time_args = {{"Mt", Mt}, {"Kt", Kt}, {"Nt", Nt}},
    .hw_config =
        tt_metal::experimental::ComputeHardwareConfig{
            .fpu_math_fidelity = math_fidelity,
            .sfpu_precision_mode = Precision::Precise,
        },
};
```

## Compute kernel declaration and compile-time defines

We're using a special reader kernel to take in data from DRAM into L1, and a special writer kernel to write out results from the compute engine back to the destination DRAM tensor.
Each kernel is a `KernelSpec` in the `ProgramSpec`, and `MakeProgramFromSpec` builds the `Program` from it.

``` cpp
tt_metal::experimental::KernelSpec reader_kernel_spec{
    .unique_id = reader_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
    .dfb_bindings = {tt_metal::experimental::ProducerOf(src0_dfb, "in0"), tt_metal::experimental::ProducerOf(src1_dfb, "in1")},
    .tensor_bindings =
        {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
         {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
    .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Kt", "Nt"}},
    .hw_config =
        tt_metal::experimental::DataMovementHardwareConfig{
            .config_1xx =
                tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        },
};

tt_metal::experimental::KernelSpec writer_kernel_spec{
    .unique_id = writer_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
    .dfb_bindings = {tt_metal::experimental::ConsumerOf(output_dfb, "out0")},
    .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
    .runtime_arg_schema = {.runtime_arg_names = {"Mt", "Nt"}},
    .hw_config =
        tt_metal::experimental::DataMovementHardwareConfig{
            .config_1xx =
                tt_metal::experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};

tt_metal::experimental::ProgramSpec program_spec{
    .name = "matmul_single_core",
    .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
    .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, output_dfb_spec},
    .tensor_parameters =
        {{.unique_id = src0_tensor_param, .spec = src0_tensor_spec},
         {.unique_id = src1_tensor_param, .spec = src1_tensor_spec},
         {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
    .work_units =
        {{.name = "matmul_single_core",
          .kernels = {reader_kernel, writer_kernel, compute_kernel},
          .target_nodes = core}},
};
Program program = tt_metal::experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

## Runtime arguments and program launch

We will now set named runtime arguments for the reader and writer kernels on the single core in `ProgramRunArgs`, and bind the DRAM tensors to the tensor parameters, to run the matmul operation on a single core and a single tile at a time.

``` cpp
tt_metal::experimental::ProgramRunArgs run_args{
    .kernel_run_args =
        {{.kernel = reader_kernel,
          .runtime_arg_values =
              tt_metal::experimental::MakeRuntimeArgsForSingleNode(core, {{"Mt", Mt}, {"Kt", Kt}, {"Nt", Nt}})},
         {.kernel = writer_kernel,
          .runtime_arg_values = tt_metal::experimental::MakeRuntimeArgsForSingleNode(core, {{"Mt", Mt}, {"Nt", Nt}})}},
    .tensor_args =
        {{src0_tensor_param, tt_metal::experimental::TensorArgument{src0_dram_tensor}},
         {src1_tensor_param, tt_metal::experimental::TensorArgument{src1_dram_tensor}},
         {dst_tensor_param, tt_metal::experimental::TensorArgument{dst_dram_tensor}}},
};
tt_metal::experimental::SetProgramRunArgs(program, run_args);
```

Launch program, enqueue & read in output tensor result into the host vector.

``` cpp
HostTensor src0_host_tensor = HostTensor::from_vector(a, src0_tensor_spec);
HostTensor src1_host_tensor = HostTensor::from_vector(b, src1_tensor_spec);
cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
output = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
```

## Conclusion

Those are the additional steps for getting `matmul_single_core` operations up and running on the compute engine. To see a more complicated example using as many cores as possible, please refer to the
`Matmul multi-core` example.
