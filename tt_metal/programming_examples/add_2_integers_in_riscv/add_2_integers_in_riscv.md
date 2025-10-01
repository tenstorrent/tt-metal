# Add 2 Integers in RISC-V

RISC-V processors 1 and 5 of a Tensix core are used for data movement, yet also have basic computing capabilities. In this example, we will build a TT-Metalium program to add two integers using these processors.

We'll go through this code section by section. Note that we have this exact, full example program in
[add_2_integers_in_riscv.cpp](../../../tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp),
so you can follow along.

To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_add_2_integers_in_riscv
```
## Set up device, command queue, workload, and program

```cpp
std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
distributed::MeshWorkload workload;
distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
Program program = CreateProgram();
constexpr CoreCoord core = {0, 0};
```

We create a 1x1 `MeshDevice` at mesh coordinate {0, 0}, obtain its `MeshCommandQueue` to submit work, define a `MeshWorkload` and its device range, create a `Program`, and indicate the `core` we intend to use at the coordinates `{0, 0}` for utilization in this example.

## Configure DRAM and L1 buffers

```cpp
constexpr uint32_t buffer_size = sizeof(uint32_t);
distributed::DeviceLocalBufferConfig dram_config{
    .page_size = buffer_size,
    .buffer_type = BufferType::DRAM};
distributed::DeviceLocalBufferConfig l1_config{
    .page_size = buffer_size,
    .buffer_type = BufferType::L1};
distributed::ReplicatedBufferConfig buffer_config{
    .size = buffer_size,
};
```

We configure DRAM and L1 buffers using `DeviceLocalBufferConfig`, and size them with `ReplicatedBufferConfig` for use across the mesh. The `ReplicatedBufferConfig` allows the user to seamlessly port a buffer configuration across an arbitrary-sized mesh of devices.

## Create buffers

```cpp
auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
auto src0_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
auto src1_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
auto dst_l1_buffer = distributed::MeshBuffer::create(buffer_config, l1_config, mesh_device.get());
```

We create three DRAM buffers for inputs/outputs and three L1 buffers for on-core staging, all as `MeshBuffer`s on the same device.

## Initialize source data and upload to DRAM

```cpp
std::vector<uint32_t> src0_vec = {14};
std::vector<uint32_t> src1_vec = {7};

EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);
EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_vec, /*blocking=*/false);
```

We prepare two single-element host vectors and upload them asynchronously to the device.

## Create the Data Movement kernel

```cpp
KernelHandle kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
```

The kernel runs on a RISC-V Data Movement core. It reads inputs from DRAM, adds them, and writes the result back to DRAM.

## Set runtime arguments and execute

```cpp
SetRuntimeArgs(
    program,
    kernel_id,
    core,
    {
        src0_dram_buffer->address(),
        src1_dram_buffer->address(),
        dst_dram_buffer->address(),
        src0_l1_buffer->address(),
        src1_l1_buffer->address(),
        dst_l1_buffer->address(),
    });

workload.add_program(device_range, std::move(program));
distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
```

We pass DRAM and L1 addresses directly to the kernel, add the program to a `MeshWorkload`, and enqueue it non-blockingly.

## Kernel function

``` cpp
// NoC coords (x,y) depending on DRAM location on-chip
uint64_t src0_dram_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_dram);
uint64_t src1_dram_noc_addr = get_noc_addr_from_bank_id<true>(src1_bank_id, src1_dram);
uint64_t dst_dram_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_dram);

constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0; // index=0
constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1; // index=1

// single-tile ublocks
uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
```

We first define the NoC addresses used for reading and writing data to/from DRAM, as well as retrieve the L1 addresses to access for data movement. Each kernel will access a single tile.

``` cpp
// Read data from DRAM -> L1 circular buffers
noc_async_read(src0_dram_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
noc_async_read_barrier();
noc_async_read(src1_dram_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
noc_async_read_barrier();

// Do simple add in RiscV core
uint32_t* dat0 = (uint32_t*) l1_write_addr_in0;
uint32_t* dat1 = (uint32_t*) l1_write_addr_in1;

dat0[0] = dat0[0] + dat1[0];

// Write data from L1 circulr buffer (in0) -> DRAM
noc_async_write(l1_write_addr_in0, dst_dram_noc_addr, ublock_size_bytes_0);
noc_async_write_barrier();
```

In the kernel, tiles corresponding to each of the source vectors will be read from the DRAM into circular buffers. These tiles will then be accessed and added together. The sum is stored in one of the circular buffers temporarily before being written directly to DRAM to be accessed by the host.

## Retrieve compute results

``` cpp
std::vector<uint32_t> result_vec;
distributed::ReadShard(cq, result_vec, dst_dram_buffer, distributed::MeshCoordinate(0, 0), true);
mesh_device->close();
```

We synchronously read back from the destination `MeshBuffer` using `ReadShard`, verify the result, and close the `MeshDevice`.
