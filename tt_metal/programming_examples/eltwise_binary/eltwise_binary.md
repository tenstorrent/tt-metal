# Eltwise binary


We now build a program that will perform eltwise binary operations on a some equal-sized tensors.

We'll go through any new code section by section. This builds on top of previous examples. Note that we have this exact, full example program in [eltwise_binary.cpp](../../../tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_eltwise_binary
```
## New buffers

In terms of DRAM tensors, We just need a new tensor for a 2nd source, because we have two source tensors (vectors).

## Mesh Setup

- **Mesh device**: Create a 1x1 mesh on device 0. The same API scales to larger meshes.
- **Command queue**: All uploads, downloads, and program executions are enqueued here.
- **Workload and range**: A workload wraps one or more programs and a device coordinate range to run them on.

```cpp
constexpr int device_id = 0;
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
constexpr experimental::NodeCoord core = {0, 0};
```

### DRAM tensors (replicated)

We allocate three DRAM-backed `MeshTensor`s: two inputs and one output. Each tensor is replicated across the mesh, so each device in the mesh gets its own identical allocation. On a unit mesh, this is just one device allocation. Each row of the row-major `TensorSpec` holds one tile's worth of elements, so the page size is a single tile and NoC transfers operate tile-by-tile.

```cpp
constexpr uint32_t n_tiles = 64;
constexpr uint32_t elements_per_tile = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

TensorSpec dram_tensor_spec(
    Shape({n_tiles, elements_per_tile}),
    TensorLayout(
        DataType::BFLOAT16,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM})); // This is a DRAM tensor.

MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
```

Upload host data into the input tensors:

```cpp
std::vector<bfloat16> a_data(elements_per_tile * n_tiles);
// ... fill a_data with random values ...
constexpr float val_to_add = -1.0f;
std::vector<bfloat16> b_data(elements_per_tile * n_tiles, bfloat16(val_to_add));

HostTensor a_host_tensor = HostTensor::from_vector(a_data, dram_tensor_spec);
HostTensor b_host_tensor = HostTensor::from_vector(b_data, dram_tensor_spec);
cq.enqueue_write_tensor(a_host_tensor, src0_dram_tensor);
cq.enqueue_write_tensor(b_host_tensor, src1_dram_tensor);
```

### Dataflow buffers (L1 FIFOs)

Describe three L1-backed dataflow buffers to move tiles between kernels. Use two entries per dataflow buffer for double-buffering to overlap producer and consumer work. Kernels, dataflow buffers and tensor parameters are identified by name within the program spec.

```cpp
const experimental::KernelSpecName reader{"reader"};
const experimental::KernelSpecName writer{"writer"};
const experimental::KernelSpecName compute{"compute"};
const experimental::DFBSpecName src0_dfb{"src0"};
const experimental::DFBSpecName src1_dfb{"src1"};
const experimental::DFBSpecName dst_dfb{"dst"};
const experimental::TensorParamName src0_tensor_param{"src0"};
const experimental::TensorParamName src1_tensor_param{"src1"};
const experimental::TensorParamName dst_tensor_param{"dst"};

constexpr uint32_t tiles_per_cb = 2;
experimental::DataflowBufferSpec src0_dfb_spec{
    .unique_id = src0_dfb,
    .entry_size = tile_size_bytes,
    .num_entries = tiles_per_cb,
    .data_format_metadata = tt::DataFormat::Float16_b,
};
experimental::DataflowBufferSpec src1_dfb_spec{
    .unique_id = src1_dfb,
    .entry_size = tile_size_bytes,
    .num_entries = tiles_per_cb,
    .data_format_metadata = tt::DataFormat::Float16_b,
};
experimental::DataflowBufferSpec dst_dfb_spec{
    .unique_id = dst_dfb,
    .entry_size = tile_size_bytes,
    .num_entries = tiles_per_cb,
    .data_format_metadata = tt::DataFormat::Float16_b,
};
```

### Compute kernel declaration and compile-time defines

We describe three kernels as `KernelSpec`s, collect them in a `ProgramSpec`, and build the program with `MakeProgramFromSpec`:
- Reader (RISCV_0): reads DRAM tiles into `src0` and `src1`
- Compute: pops two tiles, adds them, pushes result to `dst`
- Writer (RISCV_1): writes tiles from `dst` back to DRAM

Each kernel binds dataflow buffers and tensor parameters to the names it uses in its source (for example `in0`), and declares its named runtime args.

```cpp
experimental::KernelSpec reader_spec{
    .unique_id = reader,
    .source = OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/read_tiles.cpp",
    .dfb_bindings = {experimental::ProducerOf(src0_dfb, "in0"), experimental::ProducerOf(src1_dfb, "in1")},
    .tensor_bindings =
        {{.tensor_parameter_name = src0_tensor_param, .accessor_name = "in0"},
         {.tensor_parameter_name = src1_tensor_param, .accessor_name = "in1"}},
    .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};
experimental::KernelSpec writer_spec{
    .unique_id = writer,
    .source = OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/write_tile.cpp",
    .dfb_bindings = {experimental::ConsumerOf(dst_dfb, "out0")},
    .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "out0"}},
    .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
        },
};
experimental::KernelSpec compute_spec{
    .unique_id = compute,
    .source = OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/tiles_add.cpp",
    .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
    .dfb_bindings =
        {experimental::ConsumerOf(src0_dfb, "in0"),
         experimental::ConsumerOf(src1_dfb, "in1"),
         experimental::ProducerOf(dst_dfb, "out0")},
    .runtime_arg_schema = {.runtime_arg_names = {"n_tiles"}},
    .hw_config =
        experimental::ComputeHardwareConfig{
            .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        },
};

experimental::ProgramSpec program_spec{
    .name = "eltwise_binary",
    .kernels = {reader_spec, writer_spec, compute_spec},
    .dataflow_buffers = {src0_dfb_spec, src1_dfb_spec, dst_dfb_spec},
    .tensor_parameters =
        {{.unique_id = src0_tensor_param, .spec = dram_tensor_spec},
         {.unique_id = src1_tensor_param, .spec = dram_tensor_spec},
         {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
    .work_units =
        {{.name = "eltwise_binary",
          .kernels = {reader, writer, compute},
          .target_nodes = core}},
};
Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

experimental::ProgramRunArgs run_args{
    .kernel_run_args =
        {{.kernel = reader,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
         {.kernel = writer,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})},
         {.kernel = compute,
          .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(core, {{"n_tiles", n_tiles}})}},
    .tensor_args =
        {{src0_tensor_param, experimental::TensorArgument{src0_dram_tensor}},
         {src1_tensor_param, experimental::TensorArgument{src1_dram_tensor}},
         {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
};
experimental::SetProgramRunArgs(program, run_args);
```

In this program, we have a second source tensor. We will be adding this to the first source tensor.

### Launch and read back

Enqueue the program as a mesh workload. Use non-blocking enqueue and explicitly wait with `Finish(cq)` before reading results.

```cpp
workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, false);
distributed::Finish(cq);
```

Read back the output tensor (the shard at mesh coordinate `{0, 0}` on a unit mesh) and validate.

```cpp
std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();
```

In this program, we have a second source tensor. We will be adding this to the first source tensor.

## Conclusion

Those are the additional steps for getting eltwise binary operations upmand running on the compute engine. We essentially repeat the same process to chain together two operations, with one DRAM read in the middle to get the intermediate result and hold it in a DRAM tensor. For an example involving matrix multiplication on a single core, please refer to the `Matmul single core` example.
