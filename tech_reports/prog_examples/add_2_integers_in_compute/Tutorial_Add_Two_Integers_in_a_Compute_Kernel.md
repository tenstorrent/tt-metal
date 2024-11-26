# Tutorial - Add Two Integers in a Compute Kernel 🚧

1. To build and execute use the following commands
```export ARCH_NAME=<arch name>
    export TT_METAL_HOME=<this repo dir>
    ./build_metal.sh  --build-tests
    ./build/programming_examples/add_2_integers_in_compute
```

2. Setup the host program:
```Device *device = CreateDevice(0);
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
constexpr CoreCoord core = {0, 0};
```

3. Create two integers:
```Int a=1
Int b=2
```

4. Configure and initialize the DRAM Buffer:
```constexpr uint32_t single_tile_size = 2 * 1024;
tt_metal::InterleavedBufferConfig dram_config{
            .device= device,
            .size = single_tile_size,
            .page_size = single_tile_size,
            .buffer_type = tt_metal::BufferType::DRAM
};
```

5. Define the tile size to fit BFloat16 values:
```std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
```

6.Allocate memory for each buffer:
```auto src0_dram_noc_coord = src0_dram_buffer->noc_coordinates();
auto src1_dram_noc_coord = src1_dram_buffer->noc_coordinates();
auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
uint32_t src0_dram_noc_x = src0_dram_noc_coord.x;
uint32_t src0_dram_noc_y = src0_dram_noc_coord.y;
uint32_t src1_dram_noc_x = src1_dram_noc_coord.x;
uint32_t src1_dram_noc_y = src1_dram_noc_coord.y;
uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;
```

7. Specify NoC Coordinates:
```constexpr uint32_t src0_cb_index = CB::c_in0;
constexpr uint32_t num_input_tiles = 1;
CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

constexpr uint32_t src1_cb_index = CB::c_in1;
CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, single_tile_size);
CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

constexpr uint32_t output_cb_index = CB::c_out0;
constexpr uint32_t num_output_tiles = 1;
CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}}).set_page_size(output_cb_index, single_tile_size);
CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
```

8. Create a data movement kernel:
```KernelHandle binary_reader_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

KernelHandle unary_writer_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
```

9. Create a compute kernel:
```vector<uint32_t> compute_kernel_args = {};
KernelHandle eltwise_binary_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
    core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compute_kernel_args,
    }
);
```

10. Create two source vectors:
```std::vector<uint32_t> src0_vec;
std::vector<uint32_t> src1_vec;
src0_vec = create_constant_vector_of_bfloat16(single_tile_size, 14.0f);
src1_vec = create_constant_vector_of_bfloat16(single_tile_size, 8.0f);

EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);
```

11. Setup corresponding runtime arguments:
```SetRuntimeArgs(program, binary_reader_kernel_id, core, { src0_dram_buffer->address(), src1_dram_buffer->address(), src0_dram_noc_x, src0_dram_noc_y, src1_dram_noc_x, src1_dram_noc_y});
SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_dram_noc_x, dst_dram_noc_y});

EnqueueProgram(cq, program, false);
Finish(cq);
```

12. Execute the Program:
```uint32_t ublock_size_bytes_0 = get_tile_size(cb_id_in0);
uint32_t ublock_size_bytes_1 = get_tile_size(cb_id_in1);

uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

cb_reserve_back(cb_id_in0, 1);
noc_async_read(src0_noc_addr, l1_write_addr_in0, ublock_size_bytes_0);
noc_async_read_barrier();
cb_push_back(cb_id_in0, 1);

cb_reserve_back(cb_id_in1, 1);
noc_async_read(src1_noc_addr, l1_write_addr_in1, ublock_size_bytes_1);
noc_async_read_barrier();
cb_push_back(cb_id_in1, 1);
```

13. Unpack, compute, and pack the data:
```binary_op_init_common(cb_in0, cb_in1, cb_out0);
add_tiles_init();

// wait for a block of tiles in each of input CBs
cb_wait_front(cb_in0, 1);
cb_wait_front(cb_in1, 1);

tile_regs_acquire(); // acquire 8 tile registers

add_tiles(cb_in0, cb_in1, 0, 0, 0);

tile_regs_commit(); // signal the packer

tile_regs_wait(); // packer waits here
pack_tile(0, cb_out0);
tile_regs_release(); // packer releases

cb_pop_front(cb_in0, 1);
cb_pop_front(cb_in1, 1);

cb_push_back(cb_out0, 1);
```

14. Write integer values to the DRAM:
```uint64_t dst_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_addr);

constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

cb_wait_front(cb_id_out0, 1);
noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
noc_async_write_barrier();
cb_pop_front(cb_id_out0, 1);
```

15. Close the device:
```CloseDevice(device);
```
