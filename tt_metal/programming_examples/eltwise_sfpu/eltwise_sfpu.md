# Eltwise SFPU

We now build a program that will perform an eltwise SFPU unary operation on a single tensor.

We'll go through any new code section by section. This builds on top of previous examples. Note that we have this exact, full example program in [eltwise_sfpu.cpp](../../../tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
./build_metal.sh --build-programming-examples
./build/programming_examples/metal_example_eltwise_sfpu
```

## Mesh setup

- Create a 1x1 mesh on device 0 and obtain the mesh command queue.
- Create a workload and a coordinate range spanning the mesh.

```cpp
constexpr int device_id = 0;
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
constexpr experimental::NodeCoord core = {0, 0};
```

## DRAM tensors (replicated)

Allocate one input and one output DRAM-backed `MeshTensor`. Each tensor is replicated across the mesh. On a unit mesh this is a single device allocation. Each row of the row-major `TensorSpec` holds one tile's worth of elements, so the page size is a single tile and transfers happen tile-by-tile.

```cpp
constexpr uint32_t n_tiles = 64;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

TensorSpec dram_tensor_spec(
    Shape({n_tiles, elements_per_tile}),
    TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));

MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
```

Upload host data into the input tensor (non-blocking to overlap with host setup):

```cpp
std::vector<bfloat16> src0_vec(n_tiles * elements_per_tile);
// ... fill src0_vec with random values ...
HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
```

## Dataflow buffers (L1 FIFOs)

Describe two L1 dataflow buffers: one for input tiles to the compute kernel and one for output tiles from compute to writer. Use two entries for double-buffering. Kernels, dataflow buffers and tensor parameters are identified by name within the program spec.

```cpp
const experimental::KernelSpecName unary_reader_kernel{"unary_reader"};
const experimental::KernelSpecName unary_writer_kernel{"unary_writer"};
const experimental::KernelSpecName eltwise_sfpu_kernel{"eltwise_sfpu"};
const experimental::DFBSpecName src0_dfb{"src0"};
const experimental::DFBSpecName output_dfb{"output"};
const experimental::TensorParamName src0_tensor_param{"src0"};
const experimental::TensorParamName dst_tensor_param{"dst"};

constexpr uint32_t num_input_tiles = 2;
experimental::DataflowBufferSpec src0_dfb_spec{
    .unique_id = src0_dfb,
    .entry_size = tile_size_bytes,
    .num_entries = num_input_tiles,
    .data_format_metadata = tt::DataFormat::Float16_b,
};
experimental::DataflowBufferSpec output_dfb_spec{
    .unique_id = output_dfb,
    .entry_size = tile_size_bytes,
    .num_entries = num_input_tiles,
    .data_format_metadata = tt::DataFormat::Float16_b,
};
```

## Kernels and runtime arguments

Describe the data-movement and compute kernels as `KernelSpec`s, collect them in a `ProgramSpec`, and build the program with `MakeProgramFromSpec`. The reader pulls tiles from DRAM into `src0`; compute applies SFPU exp and pushes to `output`; writer flushes tiles back to DRAM. Each kernel binds dataflow buffers and tensor parameters to the names it uses in its source (for example `in0`), and declares its named runtime args.

```cpp
experimental::KernelSpec unary_reader_kernel_spec{
    .unique_id = unary_reader_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/read_tile.cpp",
    .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0")},
    .tensor_bindings = {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"}},
    .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        },
};

experimental::KernelSpec unary_writer_kernel_spec{
    .unique_id = unary_writer_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/dataflow/write_tile.cpp",
    .dfb_bindings = {experimental::ConsumerOf(output_dfb, "out0")},
    .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
    .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};

experimental::KernelSpec eltwise_sfpu_kernel_spec{
    .unique_id = eltwise_sfpu_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "eltwise_sfpu/kernels/compute/eltwise_sfpu.cpp",
    .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
    .dfb_bindings =
        {experimental::ConsumerOf(src0_dfb, "in0"), experimental::ProducerOf(output_dfb, "out0")},
    .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
    .hw_config =
        experimental::ComputeHardwareConfig{
            .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .sfpu_precision_mode = Precision::Precise,
        },
};

experimental::ProgramSpec program_spec{
    .name = "eltwise_sfpu",
    .kernels = {unary_reader_kernel_spec, unary_writer_kernel_spec, eltwise_sfpu_kernel_spec},
    .dataflow_buffers = {src0_dfb_spec, output_dfb_spec},
    .tensor_parameters =
        {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
         {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
    .work_units =
        {{.name = "eltwise_sfpu",
          .kernels = {unary_reader_kernel, unary_writer_kernel, eltwise_sfpu_kernel},
          .target_nodes = core}},
};
Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

Then set the named runtime arguments for each kernel and bind the DRAM tensors to the tensor parameters:

```cpp
experimental::ProgramRunArgs run_args{
    .kernel_run_args =
        {{.kernel = eltwise_sfpu_kernel,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
         {.kernel = unary_reader_kernel,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
         {.kernel = unary_writer_kernel,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
    .tensor_args =
        {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
         {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
};
experimental::SetProgramRunArgs(program, run_args);
```

## Launch and read back

Enqueue the program in a mesh workload (non-blocking), then wait for completion and read back the output tensor (shard `{0, 0}` on a unit mesh).

```cpp
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);

std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
```

Finally, close the mesh device after validation.
