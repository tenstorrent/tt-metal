# Eltwise binary


We now build a program that will perform eltwise binary operations on a some equal-sized tensors.

We'll go through any new code section by section. This builds on top of previous examples. Note that we have this exact, full example program in [eltwise_binary.cpp](../../../tt_metal/programming_examples/eltwise_binary/eltwise_binary.cpp), so you can follow along.

To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_eltwise_binary
```
## New buffers

In terms of DRAM buffers, We just need a new buffer for a 2nd source, because we have two source tensors (vectors).

We already have set the circular buffers needed for compute data communication.

``` cpp
constexpr uint32_t src0_cb_index = CBIndex::c_0;
constexpr uint32_t src0_cb_addr = 200 * 1024;
constexpr uint32_t num_input_tiles = 2;
constexpr uint32_t input_cb_size = num_input_tiles * single_tile_size;
CircularBufferConfig cb_src0_config = CircularBufferConfig(input_cb_size, {{src0_cb_index, tt::DataFormat::Float16_b}}, src0_cb_addr).set_page_size(src0_cb_index, single_tile_size);
CBHandle cb_src0 = CreateCircularBuffer(program, core, cb_src0_config);

constexpr uint32_t src1_cb_index = CBIndex::c_1;
constexpr uint32_t src1_cb_addr = 300 * 1024;
CircularBufferConfig cb_src1_config = CircularBufferConfig(input_cb_size, {{src1_cb_index, tt::DataFormat::Float16_b}}, src1_cb_addr).set_page_size(src1_cb_index, single_tile_size);
CBHandle cb_src1 = CreateCircularBuffer(program, core, cb_src1_config);

constexpr uint32_t output_cb_index = CBIndex::c_16;
constexpr uint32_t output_cb_addr = 400 * 1024;
constexpr uint32_t num_output_tiles = 2;
constexpr uint32_t input_cb_size = num_input_tiles * single_tile_size;
CircularBufferConfig cb_output_config = CircularBufferConfig(input_cb_size, {{output_cb_index, tt::DataFormat::Float16_b}}, output_cb_addr).set_page_size(output_cb_index, single_tile_size);
CBHandle cb_output = CreateCircularBuffer(program, core, cb_output);
```

We will create two input circular buffers to accommodate our two input tensors, and an output one for the result of the eltwise binary operation.

## Compute kernel declaration and compile-time defines

``` cpp
KernelHandle eltwise_binary_kernel_id = CreateKernel(
    program,
    "tt_metal/kernels/compute/eltwise_binary.cpp",
    core,
    ComputeConfig{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = math_approx_mode,
        .compile_args = compute_kernel_args,
        .defines = get_defines(BinaryOpType::ADD)
    }
);
```

We will declare what kind of compute kernel we\'re using and further specify we want to use the `add_tiles` eltwise binary op, for eltwise adding.

## Extra source tensor

``` cpp
constexpr float val_to_add = -1.0f;
std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_add);

detail::WriteToBuffer(src1_dram_buffer, src1_vec);
```

In this program, we have a second source tensor. We will be adding this to the first source tensor.

## Conclusion

Those are the additional steps for getting eltwise binary operations upmand running on the compute engine. We essentially repeat the same process to chain together two operations, with one DRAM read in the middle to get the intermediate result and hold it in a DRAM buffer. For an example involving matrix multiplication on a single core, please refer to the `Matmul single core` example.
