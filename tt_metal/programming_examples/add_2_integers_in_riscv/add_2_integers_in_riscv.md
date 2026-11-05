# Add 2 Integers in RISC-V

RISC-V processors 1 and 5 of a Tensix core are used for data movement, yet also have basic computing capabilities. In this example, we will build a TT-Metalium program to add two integers using these processors.

We'll go through this code section by section. Note that we have this exact, full example program in
[add_2_integers_in_riscv.cpp](../../../tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp),
so you can follow along.

To build and execute, you may use the following commands:
```bash
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_add_2_integers_in_riscv
```
## Set up device, command queue, workload, and core

```cpp
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
constexpr experimental::NodeCoord core = {0, 0};
```

We create a 1x1 `MeshDevice` at mesh coordinate {0, 0}, obtain its `MeshCommandQueue` to submit work, define a `MeshWorkload` and its device range, and indicate the `core` we intend to use at the coordinates `{0, 0}` for utilization in this example. The `Program` is created later from a program spec.

## Configure DRAM and L1 tensors

```cpp
constexpr uint32_t buffer_size = sizeof(uint32_t);
constexpr uint32_t elements_per_page = buffer_size / sizeof(uint32_t);
TensorSpec dram_tensor_spec(
    Shape({1, elements_per_page}),
    TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
TensorSpec l1_tensor_spec(
    Shape({1, elements_per_page}),
    TensorLayout(
        DataType::UINT32,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1}));
```

We configure DRAM and L1 tensors using `TensorSpec`. The `MemoryConfig` selects the buffer type (DRAM or L1), and each row of a row-major tensor is one page, so a row of one integer gives a page size of `buffer_size`. A `MeshTensor` allocated from a `TensorSpec` is replicated across the mesh, which allows the user to seamlessly port the configuration across an arbitrary-sized mesh of devices.

## Create tensors

```cpp
MeshTensor src0_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor src1_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);
MeshTensor src0_l1_tensor = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
MeshTensor src1_l1_tensor = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
MeshTensor dst_l1_tensor = MeshTensor::allocate_on_device(*mesh_device, l1_tensor_spec);
```

We create three DRAM tensors for inputs/outputs and three L1 tensors for on-core staging, all as `MeshTensor`s on the same device.

## Initialize source data and upload to DRAM

```cpp
std::vector<uint32_t> src0_vec = {14};
std::vector<uint32_t> src1_vec = {7};

HostTensor src0_host_tensor = HostTensor::from_vector(src0_vec, dram_tensor_spec);
HostTensor src1_host_tensor = HostTensor::from_vector(src1_vec, dram_tensor_spec);
cq.enqueue_write_tensor(src0_host_tensor, src0_dram_tensor);
cq.enqueue_write_tensor(src1_host_tensor, src1_dram_tensor);
```

We prepare two single-element host vectors, wrap them in `HostTensor`s, and upload them asynchronously to the device. The host tensors must stay alive until the write is done.

## Create the Data Movement kernel

```cpp
const experimental::KernelSpecName add_kernel{"reader_writer_add_in_riscv"};
experimental::KernelSpec add_kernel_spec{
    .unique_id = add_kernel,
    .source = OVERRIDE_KERNEL_PREFIX "add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
    .runtime_arg_schema =
        {.runtime_arg_names = {"src0_dram", "src1_dram", "dst_dram", "src0_l1", "src1_l1", "dst_l1"}},
    .hw_config =
        experimental::DataMovementHardwareConfig{
            .config_1xx =
                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
        },
};

experimental::ProgramSpec program_spec{
    .name = "add_2_integers_in_riscv",
    .kernels = {add_kernel_spec},
    .work_units = {{.name = "add_2_integers_in_riscv", .kernels = {add_kernel}, .target_nodes = core}},
};
Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);
```

The kernel runs on a RISC-V Data Movement core. It reads inputs from DRAM, adds them, and writes the result back to DRAM. The `KernelSpec` declares the names of the kernel's runtime arguments. The `ProgramSpec` lists the kernel and the work unit (the cores the kernel runs on), and `MakeProgramFromSpec` creates the `Program`.

## Set runtime arguments and execute

```cpp
experimental::ProgramRunArgs run_args{
    .kernel_run_args = {{
        .kernel = add_kernel,
        .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
            core,
            {
                {"src0_dram", static_cast<uint32_t>(src0_dram_tensor.address())},
                {"src1_dram", static_cast<uint32_t>(src1_dram_tensor.address())},
                {"dst_dram", static_cast<uint32_t>(dst_dram_tensor.address())},
                {"src0_l1", static_cast<uint32_t>(src0_l1_tensor.address())},
                {"src1_l1", static_cast<uint32_t>(src1_l1_tensor.address())},
                {"dst_l1", static_cast<uint32_t>(dst_l1_tensor.address())},
            }),
    }},
};
experimental::SetProgramRunArgs(program, run_args);

workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
```

We pass DRAM and L1 addresses directly to the kernel as named runtime arguments, add the program to a `MeshWorkload`, and enqueue it non-blockingly.

## Kernel function

``` cpp
// Arguments are read by the names declared in the host code.
uint32_t src0_dram = get_arg(args::src0_dram);
uint32_t src1_dram = get_arg(args::src1_dram);
uint32_t dst_dram = get_arg(args::dst_dram);
uint32_t src0_l1 = get_arg(args::src0_l1);
uint32_t src1_l1 = get_arg(args::src1_l1);
uint32_t dst_l1 = get_arg(args::dst_l1);

InterleavedAddrGen<true> src0 = {.bank_base_address = src0_dram, .page_size = sizeof(uint32_t)};
InterleavedAddrGen<true> src1 = {.bank_base_address = src1_dram, .page_size = sizeof(uint32_t)};
InterleavedAddrGen<true> dst = {.bank_base_address = dst_dram, .page_size = sizeof(uint32_t)};
```

We first read the DRAM and L1 addresses by their runtime argument names, and create address generators used to compute the NoC addresses for reading and writing data to/from DRAM. Each kernel will access a single page.

``` cpp
// Read data from DRAM -> local L1
uint64_t src0_dram_noc_addr = src0.get_noc_addr(0);
uint64_t src1_dram_noc_addr = src1.get_noc_addr(0);
noc_async_read(src0_dram_noc_addr, src0_l1, sizeof(uint32_t));
noc_async_read(src1_dram_noc_addr, src1_l1, sizeof(uint32_t));
noc_async_read_barrier();

// Do simple add in RiscV core
uint32_t* dat0 = (uint32_t*)src0_l1;
uint32_t* dat1 = (uint32_t*)src1_l1;
uint32_t* out0 = (uint32_t*)dst_l1;

DPRINT("Adding integers: {} + {}\n", *dat0, *dat1);

(*out0) = (*dat0) + (*dat1);

(void)ckernel::load_blocking(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1));

// Write data from L1 -> DRAM. Again this is a non-blocking operation.
uint64_t dst_dram_noc_addr = dst.get_noc_addr(0);
noc_async_write(dst_l1, dst_dram_noc_addr, sizeof(uint32_t));
noc_async_write_barrier();
```

In the kernel, the values of each of the source vectors will be read from the DRAM into the L1 buffers. These values will then be accessed and added together. The sum is stored in the destination L1 buffer temporarily before being written directly to DRAM to be accessed by the host. `load_blocking` makes sure the sum lands in L1 before the NoC reads it.

## Retrieve compute results

``` cpp
std::vector<uint32_t> result_vec =
    cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<uint32_t>();
// ... verify result_vec ...
mesh_device->close();
```

We synchronously read back from the destination `MeshTensor` using `enqueue_read_tensor`, verify the result, and close the `MeshDevice`.
