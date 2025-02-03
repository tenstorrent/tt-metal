# Data Multicasting in [matmul_multicore_reuse_mcast]

**Note**: This example only works on Grayskull.

Let\'s level up our code and show how you can leverage and fully customize METALIUM\'s core-to-core communication through a data broadcasting scheme. METALIUM offers you customizability for creating your very own compute fabric, allowing precise control over which cores disseminate, collect, or process segments of work.
This example builds off of the data_reuse one, so we employ the same intemediate (partial) results handling scheme on-core. However, rather than map tile-work statically to your coregrid, we map in0\'s rows and in1\'s columns to the coregrid\'s edges, and cascade work core-to-core dynamically.
A fun tidbit: \"torrent\" in Tenstorrent pays homage to this concept of tensor computation flowing like an ultra fast stream of water.

## Additional Compile-Time Argument

We introduced an out_block_num_tiles parameter in our reuse example, yet in our mcast example, we leverage it at compile time to navigate the complexity of multicasting partial results alongside fused operations \-- namely the bias and activation functions \-- which we will demonstrate further below. Let\'s pass it as follows:

``` cpp
vector<uint32_tcompute_kernel_args = {
...
out_block_tiles // out_block_num_tiles
};
log_info(tt::LogVerif, " -- out_block_tiles= {} --", out_block_tiles);
```

## Configuring Core Ranges for Tile Distribution

We can define our coregrid from any upper-left corner and define its range with any height and width that our problem calls for. The cores that make up the grid can be relegated specific broadcast roles (sender or receiver cores) and compute workloads (only certain tiles of in0 and in1).
We start with the following initializations for the grid itself, upper-left corner, and grid edge vectors:

``` cpp
CoreCoord start_core = {0, 0};
CoreCoord core_range = bmm_op_utils::get_core_range(num_blocks_y, num_blocks_x, num_cores_y, num_cores_x);
```

``` cpp
uint32_t start_core_x = start_core.x;
uint32_t start_core_y = start_core.y;
uint32_t num_cores_c = core_range.x;  // Core count along x-axis
uint32_t num_cores_r = core_range.y;  // Core count along y-axis
```

Next, we define the mcast role of the entire coregrid, the uppermost edge, and the leftmost edge vectors of cores like so:

-   **\`\`all_cores\`\`**: Outlines the boundary of all inclusive cores
    for the matmul kernel.

    ``` cpp
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    ```

-   **\`\`left_column\`\`**: Isolates the left column of cores for
    specialized tasks; in this case, multicasting in0 tiles.

    ``` cpp
    CoreRange left_column(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});
    ```

-   **\`\`all_except_left_column\`\`**: Designates which cores will peform their share of in0 and in1 tile work, and calculate their partial results.

    ``` cpp
    CoreRange all_except_left_column(
        {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    ```

Then, we define the data flow framework. The basic idea is that we initiate in0 row and in1 column data flow from the upper-left core of the coregrid (0,0) and designate this as our master sender core.

``` cpp
CoreRange in0_sender_in1_sender(
    {(std::size_t)start_core_x, (std::size_t)start_core_y}, {(std::size_t)start_core_x, (std::size_t)start_core_y});
```

Then we mcast send in0 rows of work vertically down the coregrid's left_column (from DRAM into each of these core's L1). These left_column cores are responsible for disseminating the **same** in0 row tile data to each core, thereby leveraging the data reuse scheme as we mentioned in the last section. We also ensure they are desginated as receiver cores because they will also take on in1 column work.

``` cpp
CoreRange in0_sender_in1_receiver(
    {(std::size_t)start_core_x, (std::size_t)start_core_y + 1},
    {(std::size_t)start_core_x, (std::size_t)start_core_y + num_cores_r - 1});
```

We also mcast send in1 columns of work horizontally across the coregrid (left to right) into left_column and all_except_left_column ranges of cores. You can imagine the top row of our coregrid (minus the master sender core) will be responsible for disseminating all the **different** in1 columns of work.

``` cpp
CoreRange in0_receiver_in1_sender(
    {(std::size_t)start_core_x + 1, (std::size_t)start_core_y},
    {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y});
```

The remining tiles act as receivers for both in0 and in1 tile data.
Essentially we are computing output_tile work (partial results of our output matrix) on each core, wherein each core has been simultaneously mcasted a unique chunk of in0 and in1 tile data to compute on.

``` cpp
CoreRange in0_receiver_in1_receiver(
    {(std::size_t)start_core_x + 1, (std::size_t)start_core_y + 1},
    {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
```

This leaves each core with exactly the work it needs to compute its partial results of the output matrix. We will end up using 4 dataflow kernels:

    in0 sender
    in0 receiver
    in1 sender+writer
    in1 receiver+writer

## Circular Buffer Creation for CoreGrid

Recall in our data reuse example, we created our L1 circular buffers for all the cores like so:

``` cpp
auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
```

METALIUM also allows us to pass all of our CoreRanges defined above through a `CoreRangeSet(...)` function call as the 2nd argument. Let's do so with the following:

``` cpp
auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);
```

In fact, you can instantiate circular buffers on any one of these three options: `const std::variant<CoreCoord, CoreRange, CoreRangeSet>`.
Please refer to the CircularBuffers page for further details.

## Multicast Reader/Writer Kernel Setup

In datareuse, we spawned reader and writer kernels per core. In mcast, we have desginated core ranges (or more generally speaking, `groups`), and METALIUM gives us functionality to relegate a certain type of reader/writer kernel to a group.

Below, let's set some core ID\'s associated with a specific sender-receiver kernel. Take note that each ID is designated as one of two data movement processors, NCRISC (loading data from DRAM to L1) or BRISC (storing data from L1 to DRAM), as defined in the
[tt_metal/impl/kernels/data_types.hpp](../../../tt_metal/impl/kernels/data_types.hpp) file.

``` cpp
// Create reader and writer kernels per core group

auto mm_reader_kernel_in0_sender_in1_sender_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp",
    in0_sender_in1_sender,
    tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_compile_time_args});

auto mm_reader_kernel_in0_sender_in1_receiver_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp",
    in0_sender_in1_receiver,
    tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_compile_time_args});

auto mm_reader_kernel_in0_receiver_in1_sender_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp",
    in0_receiver_in1_sender,
    tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

auto mm_reader_kernel_in0_receiver_in1_receiver_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp",
    in0_receiver_in1_receiver,
    tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

auto unary_writer_kernel_noc0_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
    all_except_left_column,
    tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

auto unary_writer_kernel_noc1_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
    left_column,
    tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = writer_compile_time_args});
```

If you are interested in further details on how these work, we implore you to check out the exact dataflow kernels located in the
[tt_metal/programming_examples/matmul_common/kernels/dataflow](../../../tt_metal/programming_examples/matmul_common/kernels/dataflow) file.
You can see there are many arguments with which to experiment with, such as mcast destination nocs. You can imagine defining your own mcast scheme.

    in0_mcast_dest_noc_start_x
    in0_mcast_dest_noc_start_y
    in0_mcast_dest_noc_end_x
    in0_mcast_dest_noc_end_y

## New Compute Kernel: Fused Bias Addition and Activation Functions

Like all the examples preceeding, we call our compute kernel as usual, except here we introduce a new one called `bmm_large_block_zm_fused_bias_activation`.

``` cpp
auto mm_kernel_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
    all_cores,
    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args}
);
```

a.  **Flow Control through Conditionals**

When bias fusion is enabled ([FUSE_BIAS]{ title-ref}), intermediate results are directly packed and may not require reloading for subsequent operations within the same batch, indicated by [enable_reload = false]. We can employ this as a means of minimizing mem-to-mem operations.

For kernels without bias fusion or when the [PACKER_L1_ACC] is not defined, we determine whether intermediate results need to be reloaded based on the computation phase (ie. our [spill] condition and the current [block]{.title-ref}). This ensures that for operations that accumulate results over multiple blocks, intermediate data is correctly managed across iterations.

b.  **Bias Broadcasting Mechanism**

    ``` cpp
    add_bcast_rows_init_short(mm_partials_cb_id, bias_cb_id);
    for (uint32_t i = 0, j = 0; j < out_subblock_h; j++) {
        uint32_t bcast_tile_idx = in1_index_subblock_offset;
        for (uint32_t k = 0; k < out_subblock_w; k++, i++) {
            add_tiles_bcast_rows(mm_partials_cb_id, bias_cb_id, i, bcast_tile_idx, i);
            bcast_tile_idx++;
        }
    }
    ```

c.  **In-place Activation Function**

    ``` cpp
    #ifdef SFPU_OP_INIT_ACTIVATION
    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        SFPU_OP_FUNC_ACTIVATION
    }
    #endif
    ```

d.  **Handling Partial Results**

    ``` cpp
    if (enable_reload) {
        reload_from_cb_to_dst(in0_cb_id, in1_cb_id, mm_partials_cb_id, out_subblock_num_tiles, out_subblock_w, out_subblock_h, in0_block_w);
    }
    ```

## Semaphores

To cleanly coordinate the distribution and processing of in0 and in1 tiles in our mcast strategy, we should introduce semaphores:
- Without semaphores, we run the risk of mcast sending data from one Tensix core to another too early (before the first Tensix core's CB stream is fully populated), or mcast receiving too few packets of data and thus computing prematurely (before the second Tensix core\'s CB stream is fully populated).
- METALIUM makes this very simple, by allowing to call the CreateSemaphore function and simply passing the entire CoreGrid number of cores.
- Therefore, we define our sender and receiver core semaphores as follows, to maintain synchronization of compute across the device:

``` cpp
auto in0_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
auto in0_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
auto in1_mcast_sender_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
auto in1_mcast_receiver_semaphore = tt_metal::CreateSemaphore(program, all_cores, INVALID);
```

## Kernel Runtime Arguments

Recall that we just desginated NCRISCs to handle our DRAM-\>CoreGrid L1 data movement. METALIUM lets us pass in a buffer of tensors and dereference them with a stride by multiples of core coordintates.

``` cpp
std::vector<uint32_tmm_reader_args = {
    (std::uint32_t)  src0_dram_buffer->address(), // in0_buffer_addr
    (std::uint32_t)  Kt * per_core_M * core_idx_y, // in0_buffer_start_tile_id
    (std::uint32_t)  1, // in0_buffer_stride_w
    (std::uint32_t)  Kt, // in0_buffer_stride_h
    (std::uint32_t)  in0_block_w, // in0_buffer_next_block_stride

    (std::uint32_t)  in0_block_w, // in0_block_w
    (std::uint32_t)  per_core_M, // in0_block_h
    (std::uint32_t)  in0_block_w * per_core_M, // in0_block_num_tiles

    (std::uint32_t)  src1_dram_buffer->address(), // in1_buffer_addr
    (std::uint32_t)  per_core_N * core_idx_x, //in1_buffer_start_tile_id
    (std::uint32_t)  1, // in1_buffer_stride_w
    (std::uint32_t)  Nt, // in1_buffer_stride_h
    (std::uint32_t)  in0_block_w * Nt, //in1_buffer_next_block_stride
    ...
```

For runtime, we need to set a few more IDs on corner cores of our CoreGrid, that will act solely as worker cores.

``` cpp
std::vector<KernelHandlereader_kernel_ids;
std::vector<KernelHandlewriter_kernel_ids;
for(int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
    for(int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
        CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

        CoreCoord left_core    = {(std::size_t) start_core_x, (std::size_t) core.y};
        CoreCoord left_core_plus_one    = {(std::size_t) start_core_x + 1, (std::size_t) core.y};
        CoreCoord right_core   = {(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) core.y};
        CoreCoord top_core     = {(std::size_t) core.x, (std::size_t) start_core_y};
        CoreCoord top_core_plus_one     = {(std::size_t) core.x, (std::size_t) start_core_y + 1};
        CoreCoord bottom_core  = {(std::size_t) core.x, (std::size_t) start_core_y + num_cores_r - 1};

        auto left_core_physical = device->worker_core_from_logical_core(left_core);
        auto left_core_plus_one_physical = device->worker_core_from_logical_core(left_core_plus_one);
        auto right_core_physical = device->worker_core_from_logical_core(right_core);
        auto top_core_physical = device->worker_core_from_logical_core(top_core);
        auto top_core_plus_one_physical = device->worker_core_from_logical_core(top_core_plus_one);
        auto bottom_core_physical = device->worker_core_from_logical_core(bottom_core);
```

At this point we can specificy exactly which worker core plays which role for mcasting in0 and in1 data. Here we can map the physical core on device with:

``` cpp
(std::uint32_t)  right_core_physical.x, // in0_mcast_dest_noc_start_x
(std::uint32_t)  right_core_physical.y, // in0_mcast_dest_noc_start_y
(std::uint32_t)  left_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
(std::uint32_t)  left_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
(std::uint32_t)  (num_cores_c - 1), // in0_mcast_num_dests
(std::uint32_t)  left_core_physical.x, // in0_mcast_sender_noc_x
(std::uint32_t)  left_core_physical.y, // in0_mcast_sender_noc_y
(std::uint32_t)  in0_mcast_sender_semaphore,
(std::uint32_t)  in0_mcast_receiver_semaphore,

(std::uint32_t)  bottom_core_physical.x, // in0_mcast_dest_noc_start_x
(std::uint32_t)  bottom_core_physical.y, // in0_mcast_dest_noc_start_y
(std::uint32_t)  top_core_plus_one_physical.x, // in0_mcast_dest_noc_end_x
(std::uint32_t)  top_core_plus_one_physical.y, // in0_mcast_dest_noc_end_y
(std::uint32_t)  (num_cores_r - 1), // in0_mcast_num_dests
(std::uint32_t)  top_core_physical.x, // in0_mcast_sender_noc_x
(std::uint32_t)  top_core_physical.y, // in0_mcast_sender_noc_y
(std::uint32_t)  in1_mcast_sender_semaphore,
(std::uint32_t)  in1_mcast_receiver_semaphore,
...
```

Finally, we push our IDs into our reader and writer kernel handler vectors, and targets our NCRISC (RISCV_0) and BRISC (RISCV_1) processors. For our master send core (0,0), which initiates data movement for both matrices in0 and in1:

``` cpp
if(core_idx_x == 0 and core_idx_y == 0) {
    tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_sender_id, core, mm_reader_args); // RISCV_0_default
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args); // RISCV_1_default
    reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_sender_id);
    writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
}
```

For the left_column cores, we task them with receiving in1 columns from the top and sending in0 rows to the right:

``` cpp
else if (core_idx_x == 0 and core_idx_y != 0) {
    tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_sender_in1_receiver_id, core, mm_reader_args); // RISCV_0_default
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc1_id, core, writer_args); // RISCV_1_default
    reader_kernel_ids.push_back(mm_reader_kernel_in0_sender_in1_receiver_id);
    writer_kernel_ids.push_back(unary_writer_kernel_noc1_id);
}
```

For the upper_row cores (minus the upper-left master send core), we task them with receiving matrix in0 rows from the left, and sending in1 columns upwards.

``` cpp
else if (core_idx_x != 0 and core_idx_y == 0) {
    tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_sender_id, core, mm_reader_args); // RISCV_1_default
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args); // RISCV_0_default
    reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_sender_id);
    writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
}
```

For all other cores (between the left_column and upper_row cores, minus the master send core), we task these with receiving in0 rows from the left and in1 columns from the top, thereby dividing work appropriately and commencing the partial results computation process.

``` cpp
else {
    tt_metal::SetRuntimeArgs(program, mm_reader_kernel_in0_receiver_in1_receiver_id, core, mm_reader_args); // RISCV_1_default
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_noc0_id, core, writer_args); // RISCV_0_default
    reader_kernel_ids.push_back(mm_reader_kernel_in0_receiver_in1_receiver_id);
    writer_kernel_ids.push_back(unary_writer_kernel_noc0_id);
}
```
