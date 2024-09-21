# Data Reuse in [matmul_multicore_reuse]

## Fine-Grained Block Size Control

Advanced matrix dimension controls are found in the Programming Example's matmul_common directory, namely Block Matrix Multiply Ops (bmm_op.hpp). 
Including this header allows us advanced dynamic means of defining and retrieving matrix parameters. 
Our matmul kernels that work out-of-the-box perform on row-major and tile-major layouts, so you have the power to define your own outer-dimensional tile sizes, desired core grid dimensions, as well as your own input block width, all depending on your problem at hand.

In our reuse example, we can employ the `get_large_matmul_params(...)` function and pass our inputs as described above. By doing so, we let METALIUM\'s bmm op utility functions do the heavy lifting for us mathematically, and calculate our matmul\'s exact work-per-core size and work output size seamlessly. (You can consult the header for the prime factorization method used, plus many other details).

``` cpp
auto matmul_params = bmm_op_utils::get_large_matmul_params(Mt, Nt, num_cores_y, num_cores_x, in0_block_w);
uint32_t per_core_M = std::get<0(matmul_params);
uint32_t per_core_N = std::get<1(matmul_params);
uint32_t out_subblock_h = std::get<2(matmul_params);
uint32_t out_subblock_w = std::get<3(matmul_params);
```

Take note of the example\'s use of \"subblocks\" above. Recall that until now, we have optimized matmul by dividing matrices into blocks and subdivided those into tiles, which are laid out neatly on our compute cores. 
A key optimization here in [matmul_multicore_reuse] is the introduction of an intermediate subdivision of blocks, called subblocks. Below are some optimal subblock layouts already provided for you in the header, which run efficiently on our hardware.

``` cpp
constexpr std::array<std::tuple<uint32_t, uint32_t, 20SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8},
    {7, 1}, {1, 7},
    {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5},
    {2, 2}, {4, 1}, {1, 4},
    {3, 1}, {1, 3},
    {2, 1}, {1, 2},
    {1, 1},
}};
```

## Intermediate Circular Buffer Configuration

In addition to our double-buffer config, we introduce a third circular buffer denoted as `interm0_cb_index`. Out of the 32 possible circular buffers provided by the API (which you can view in the [tt_metal/hostdevcommon/kernel_structs.h](../../../tt_metal/hostdevcommon/kernel_structs.h), this one belongs to a subset of intermediate CBs. 
This buffer acts as a temporary storage for the intermediate results of matrix multiplication before they are combined into the final output.

``` cpp
uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
uint32_t interm0_cb_index = 24; // Index for the intermediate circular buffer
std::map<uint8_t, tt::DataFormatoutput_cb_data_format_spec {
    {output_cb_index, cb_data_format}, // Output buffer configuration
    {interm0_cb_index, cb_data_format} // Intermediate buffer configuration
};
CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
    .set_page_size(output_cb_index, single_tile_size)
    .set_page_size(interm0_cb_index, single_tile_size);
auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);
```

## Stride Kernel Arguments

The runtime arguments for the read, write, and compute kernels are set up in a certain way to employ data reuse through the intermediate circular buffer. This setup aligns with the execution model of the `bmm_tile_layout.cpp` reader and writer kernels, and `bmm_large_block_zm.cpp` compute kernel.

``` cpp
/*
* Create Kernels (Reader, Writer, Compute)
*/
// Create reader and writer kernels per core
auto reader_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_tile_layout.cpp",
    all_cores,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

auto writer_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_tile_layout.cpp",
    all_cores,
    tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

// Create compute kernel
auto mm_kernel_id = tt_metal::CreateKernel(
    program,
    "tt_metal/programming_examples/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
    all_cores,
    tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args}
);
```

Recall our compile-time kernel compute args:

``` cpp
vector<uint32_tcompute_kernel_args = {
    in0_block_w, // in0_block_w
    in0_num_subblocks, // in0_num_subblocks
    in0_block_num_tiles, // in0_block_num_tiles
    in0_subblock_num_tiles, // in0_subblock_num_tiles

    in1_num_subblocks, // in1_num_subblocks
    in1_block_num_tiles, // in1_block_num_tiles
    in1_per_core_w, // in1_per_core_w

    num_blocks, // num_blocks

    out_subblock_h, // out_subblock_h
    out_subblock_w, // out_subblock_w
    out_subblock_num_tiles, // out_subblock_num_tiles
    B // batch
};
```

To properly run the reader and writer kernels, we must set up the runtime arguments with this information. For each block of in0 and in1 matrices, we read the tiles pertaining to a certain subblock from DRAM into that core\'s L1, and we perform the bmm_large_block_zm on tiles therein using stride arguments. 

Recall each tile is a member of a certain subblock, and subblocks are distributed across different cores in the core grid (specifically, in each core\'s L1). The writer kernel then stores the partial matmul results into its corresponding output subblock.

Reader:

``` cpp
    std::vector<uint32_tmm_reader_args = {
    (std::uint32_t)  src0_dram_buffer-address(), // in0_tensor_addr
    (std::uint32_t)  Kt * per_core_M * output_idx_y, // in0_tensor_start_tile_id
    (std::uint32_t)  1, // in0_tensor_stride_w
    (std::uint32_t)  Kt, // in0_tensor_stride_h
    (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride

    (std::uint32_t)  in0_block_w, // in0_block_w
    (std::uint32_t)  per_core_M, // in0_block_h
    (std::uint32_t)  in0_block_w * per_core_M, //in0_block_num_tiles

    (std::uint32_t)  src1_dram_buffer-address(), // in1_tensor_addr
    (std::uint32_t)  per_core_N * output_idx_x, //in1_tensor_start_tile_id
    (std::uint32_t)  1, // in1_tensor_stride_w
    (std::uint32_t)  Nt, // in1_tensor_stride_h
    (std::uint32_t)  in0_block_w * Nt, //in1_tensor_next_block_stride

    (std::uint32_t)  per_core_N, // in1_block_w
    (std::uint32_t)  in0_block_w, //in1_block_h
    (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

    (std::uint32_t)  Kt / in0_block_w, // num_blocks

    (std::uint32_t)  Mt * Kt, // MtKt
    (std::uint32_t)  Kt * Nt, // KtNt
    (std::uint32_t)  B, // batch
    (std::uint32_t)  bcast_batch // bcast_B
};
```

Writer:

``` cpp
std::vector<uint32_twriter_args = {
    (std::uint32_t) dst_dram_buffer-address(), // out_buffer_addr
    (std::uint32_t) output_idx_x * per_core_N + output_idx_y * per_core_M * Nt, // out_tensor_start_tile_id
    (std::uint32_t) 1, // out_tensor_stride_w
    (std::uint32_t) Nt,  // out_tensor_stride_h
    (std::uint32_t) out_subblock_w, // out_tensor_next_subblock_stride_w
    (std::uint32_t) out_subblock_h * Nt, // out_tensor_next_subblock_stride_h

    (std::uint32_t) out_subblock_w, // out_subblock_w
    (std::uint32_t) out_subblock_h, // out_subblock_h
    (std::uint32_t) out_subblock_w * out_subblock_h, // out_subblocks_w * out_subblocks_h
    (std::uint32_t) per_core_N / out_subblock_w, // out_num_subblocks_w
    (std::uint32_t) per_core_M / out_subblock_h, // out_num_subblocks_h

    (std::uint32_t) Mt * Nt, // MtNt
    (std::uint32_t) B // batch
};
```

# Intermediate Results Handling

In `bmm_large_block_zm.cpp`,

a.  **Preparing the Intermediate Buffer**:

**Reserving Partial Results Space**: For a given block (excluding the last block), we reserve space for intermediate (ie. partial) results in the rear of the intermediate circular buffer with `cb_reserve_back(...)`. 
Each consecutive subblock within this block will access this space, and contribute their partial results.
    
```cpp
cb_reserve_back(tt::CB::c_intermed0, out_subblock_num_tiles);
```
    
**Storing Partial Results**: Partial results are stored via a packing mechanism with `pack_tile(...)` into the above reserved space.

``` cpp
for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
    pack_tile(i, tt::CB::c_intermed0);
}
cb_push_back(tt::CB::c_intermed0, out_subblock_num_tiles);
```

b.  **Computing with Partial Results**:

  **Result Retrieval**: During block computations after the first block, we retrieve the stored results `cb_wait_front(...)` for further computation. This retrieval, also known as \"reloading\" data, is the heart of our data reuse concept. 

  It is leveraged only when our flag `enable_reload` is set to true. Recall from our understanding of circular buffers that there needs be synchronization that all tile work thus far be finished before contributing more partial results.
    
``` cpp
if (enable_reload) {
    cb_wait_front(tt::CB::c_intermed0, out_subblock_num_tiles);
    for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
        copy_tile(tt::CB::c_intermed0, i, i);
    }
    cb_pop_front(tt::CB::c_intermed0, out_subblock_num_tiles);
}
```
    
-   **Execution with \`matmul_tiles\`**: Now we are ready to compute partial results and integrate them back into the computation stream (or for the last block of computation, culminate our data reuse to produce the final output tensor).

We call the `matmul_tiles(...)` function to execute our matmul on the core\'s subblocks of tiles.

```cpp
// Compute output sub-block from in0_subblock x in1_subblock
int dst_index = 0;
int in0_index_h_offset = 0;
for (uint32_t h = 0; h < out_subblock_h; h++) {
    for (uint32_t w = 0; w < out_subblock_w; w++) {
        int in1_index_inner_dim_offset = 0;
        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
            int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
            int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
            matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, in0_index, in1_index, dst_index, false /* transpose */);
            in1_index_inner_dim_offset += in1_per_core_w;
        }
        dst_index++;
    }
    in0_index_h_offset += in0_block_w;
}
```

c.  **Wrapping Up the Intermediate Buffer**:

 **Freeing Up Space**: After all partial results have been computed and stored in our output subblock, we have completed the cycle of reuse, so now we free up the space in the intermediate circular buffer with `cb_pop_front(...)`.

## Conclusion

Those are the additional steps for getting `matmul_multicore_data_reuse` operations up and running on the compute engine. To see a more complicated example using core-to-core data movement, please refer to the `Matmul multi-core data mcast` example.
