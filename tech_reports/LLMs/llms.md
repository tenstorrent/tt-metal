# LLMs in TT-NN
Authors: 
## Contents
- [LLMs in TT-NN](#llms-in-tt-nn)
  - [Contents](#contents)
  - [1. Overview](#1-overview)
  - [2. Modules](#2-modules)
    - [2.1 Embedding](#21-embedding)
    - [2.2 RoPE](#22-rope)
    - [2.3 Norm](#23-norm) 
    - [2.4 Attention](#24-attention)
    - [2.5 MLP](#25-mlp)
    - [2.6 Decoder](#26-decoder)
    - [2.7 LM Head](#27-lm-head)
  - [3. Features](#3-features)
    - [3.1 Generative Decoding](#31-generative-decoding)
    - [3.2 Prefill and Decode](#32-prefill-and-decode)
    - [3.3 Multi-Device](#33-multi-device)
    - [3.4 Continuous Batching](#34-continuous-batching)
    - [3.5 vLLM Integration](#34-vllm-integration)
  - [4. Best Practices and Optimizations](#4-best-practices-and-optimizations)
    - [4.1 Tracing](#41-tracing)
    - [4.2 Async Mode](#42-async-mode)
    - [4.3 Multiple CQs](#43-multiple-cqs)
    - [4.4 Op Configs](#44-op-configs)
    - [4.5 Accuracy](#45-accuracy)
    - [4.6 Performance Analysis](#46-performance-analysis)
    - [4.7 Misc. Performance Optimizations](#47-misc-performance-optimizations)
    - [4.8 Module Tests](#48-module-tests)
    - [4.9 Performance Testing](#49-performance-testing)
    - [4.10 Common Pitfalls](#410-common-pitfalls)
      - [4.10.1 Error Messages](#4101-error-messages)
      - [4.10.2 Shard Spec Mismatches](#4102-shard-spec-mismatches)
      - [4.10.3 Ethernet Dispatch Cores](#4103-ethernet-dispatch-cores)
      - [4.10.4 Hangs](#4104-hangs)
        - [4.10.4.1 Tracing](#41041-tracing)
        - [4.10.4.2 Large Matmuls](#41042-large-matmuls)

## 1. Overview
## 2. Modules
### 2.1 Embedding
### 2.2 RoPE
  - Iterative update system
  - When to use our fused op
### 2.3 Norm
  - Replicated layernorm vs distributed layernorm
    - Layernorm/rmsnorm weights in row major / wrapped around tile size trick
### 2.4 Attention
  - Flash Attention and Flash Decode
    - general description
    - limitations
    - which dims are parallelized
### 2.5 MLP
### 2.6 Decoder
### 2.7 LM Head
## 3. Features
### 3.1 Generative Decoding
### 3.2 Prefill and Decode
  - submodules, tests
  - how to combine prefill and decode, 
  - slicing prefill to fit in L1
### 3.3 Multi-Device
  - device mesh
  - column parallel followed by row parallel
  - sharding, CCL ops, reducing CCL overheads, etc.
### 3.4 Continuous Batching
  - quick intro and how it is implemented in demos.
### 3.5 vLLM Integration
  - Our vLLM repo and what's needed to integrate with it.
## 4. Best Practices and Optimizations
### 4.1 Tracing
  - link to existing doc, why it helps decode more
### 4.2 Async Mode
### 4.3 Multiple CQs
  - how to feed back output to input and read output asyncronously

### 4.4 Op Configs

Program configs and memory configs are your greatest levers for performance. As a prerequisite for this section, you should understand [Tensor and Memory Layouts](../tensor_layouts/tensor_layouts.md) and the concepts in [ViT-TTNN](../VIT-TTNN/vit.md).

Most `ttnn` operations have arguments for `program_config` and `memory_config`. You should optimize these for best performance. 
`memory_config` is used to determine the layout of the output tensor. 
`program_config` configures the op with some hyperparameters like block size, core grid, etc. You should be intentional when setting up `memory_config` and `program_config`. Not only should you make each particular op execute fast, but ideally each op in the model should produce its output in a layout that is most efficient for the next op.

Let's look at `ttnn.matmul` as an example.
```python
output = ttnn.linear(
  act,
  weight,
  compute_kernel_config=compute_kernel_config,
  dtype=ttnn.bfloat16,
  program_config=program_config,
  memory_config=memory_config,
)
```
When you don't pass memory configs or program configs the operation will choose default values. These defaults are often sub-optimal. `memory_config` typically defaults to a DRAM interleaved configuration, while `program_config` defaults to something reasonable but still sub-optimal.
See [Matrix Engine](../matrix_engine/matrix_engine.md) for background on `compute_kernel_config`.

#### Memory Configs
For the LLM context, memory configs are not as important in prefill mode, where activations are large (due to the long sequence lengths) and thus should generally be DRAM interleaved (otherwise wouldn't fit on L1). In prefill mode, each op should consume DRAM interleaved inputs and produce DRAM interleaved output(s).

Memory configs are most important in decode mode. For some operation like `ttnn.matmul`, both the activation and the output will be sharded according to their memory configs. Decode mode activations are of shape `[batch_size, hidden_size]` and should be width-sharded in L1 (sharding the `hidden_size` dimension). By keeping activations and outputs width-sharded in L1 we reduce DRAM traffic and get better performance. The Llama3 codebase has examples of how to create a width-sharded memory config (see [Llama3 model config](../../models/demos/llama3/tt/model_config.py)).

```python
input_memcfg = ttnn.create_sharded_memory_config(
  (
    batch_size, # The HEIGHT of a single shard
    hidden_dim // core_grid.num_cores, # The WIDTH of a single shard
  ),
  core_grid, # Core grid to shard over (e.g. 8x2)
  ttnn.ShardStrategy.WIDTH, # WIDTH sharding (as opposed to HEIGHT or BLOCK)
  ttnn.ShardOrientation.ROW_MAJOR, # Shards are laid out in a row-major order over the core grid
  use_height_and_width_as_shard_shape=True,
)
```
Now that we know activations should be width-sharded, the only design decision to make is the `core_grid` on which to shard over. This is where you pay attention to 1) any constraints that an op might have on the input core grid, 2) how the input core grid affects the speed of the op, and 3) how the input core grid interplays with the output core grid.

There are some cases where you don't need to create a specific sharded memory config. In these cases, you can instead pass one of the following:
1. `ttnn.DRAM_MEMORY_CONFIG` when you just want DRAM interleaved.
2. `ttnn.L1_MEMORY_CONFIG` when you want L1 interleaved.
3. `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG` when you want width-sharded and the op can infer the core grid and shard shape.

As always, you should try running your `ttnn` op in a unit test with whichever settings you provide. You may find that the op produces incorrect outputs because it's missing some validation or different shard specs are used between input/output and the op itself (as TT-Metal matures, the sharding logic will get better at detecting these edge cases). You may also find that your memory config is not optimal and you can improve performance with a different configuration.

Be careful when your memory config creates shards that require padding (i.e, the shard shape does not divide evenly into 32x32 tiles). Padded shards and padded ops are under active development and can be sources of bugs. When your memory config requires padding, you probably want to instead find a core grid which divides evenly into the tensor shape.

#### Program Configs and Picking the Right Matmul
Each `ttnn` operation has its own unique program config class. In general, program configs configure the op with hyperparameters that affects their functionality and performance. There are too many ops and program configs to cover in detail. We will focus on `ttnn.matmul` since it has multiple variants and it usually requires the most care.

Picking a matmul variant is a key decision in optimizing a model. The choice depends on the shapes of the inputs and outputs and how the matmul fits into the rest of the model. You choose a variant by providing a specific `program_config` to `ttnn.matmul`. The following presents three matmul variants that are commonly used in LLMs.

##### Matmul 2D
Matmul 2D gets its name because it parallelizes an `(M x K) @ (K x N)` matmul over the M and N dimensions. It is useful to have this 2D parallelization when M and N are large (usually >= 256). Rule of thumb: use matmul 2D for all matmuls in prefill mode. Generally, inputs and output to matmul 2D will be interleaved in DRAM because these matmuls should be compute bound rather than memory bound and the inputs may be too large to fit in L1. NOTE: the weights can be DRAM sharded and still work with matmul 2D.

The following is a description of the program config for matmul 2D.
Given your input tensors of shape `(M x K)` and `(K x N)` and a core grid of shape `(cores_x, cores_y)`:

```python
matmul_2d_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
  compute_with_storage_grid_size=(cores_x, cores_y),
  in0_block_w=1,
  out_subblock_h=1, # Must be divisible by per_core_M
  out_subblock_w=1, # Must be divisible by per_core_N
  per_core_M=math.ceil(M / 32 / cores_y),  # M / TILE_HEIGHT / Grid_Size
  per_core_N=math.ceil(N / 32 / cores_x),  # N / TILE_WIDTH / grid width
  transpose_mcast=False,
  fused_activation=None,
  fuse_batch=False,
)
```
Line by line, this is what the program config means.

- `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig`: Selects the matmul 2D variant.

- `compute_with_storage_grid_size=(cores_x, cores_y)`: Determines how many cores to execute the matmul on. Note that M is parallelized over `cores_y` and N is parallelized over `cores_x`.

```python
in0_block_w=1,
out_subblock_h=1, # Must be divisible by per_core_M
out_subblock_w=1, # Must be divisible by per_core_N
```
- `in0_block_w` should divide evenly into K. Higher is better. 
- `out_subblock_h` and `out_subblock_w` should divide evenly into M and N respectively. Higher is better. 
  - When FP32 accumulate is enabled, `out_subblock_h * out_subblock_w` should be <= 4. Otherwise, `out_subblock_h * out_subblock_w` should be <= 8.

```python
per_core_M=math.ceil(M / 32 / cores_y),  # M / TILE_HEIGHT / Grid_Size
per_core_N=math.ceil(N / 32 / cores_x),  # N / TILE_WIDTH / grid width
```
- These parameters tell the matmul how many tiles of output each core is responsible for. Therefore, divide M and N by 32 (the tile size) and the core grid size. Round up because you may have padding.

```python
transpose_mcast=False,
fused_activation=None,
fuse_batch=False,
```
- If this matmul is part of an MLP with an activation, `fused_activation` will tell the kernel which activation to apply.
- `fuse_batch` should generally be set to `False`.

Since we use matmul 2D for large matmuls, there may be some issues where we run out of L1 just to store intermediate values in the kernel. When this happens, try reducing `in0_block_w` and `out_subblock_h` and `out_subblock_w`.

##### DRAM-Sharded Matmul
DRAM-Sharded matmul should be used in decode mode, where activations are small and DRAM-bandwidth to read weights is the limiting factor in op performance. This matmul gets its name because rather than having weights interleaved in DRAM, they are sharded across DRAM banks to optimally collocate weights with compute. See the [DRAM-Sharded Matmul](../Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md) writeup for details on the implementation.

We use DRAM-Sharded matmul for all matmuls in decode mode. The activation and output are width-sharded in L1, and the weights are width-sharded in DRAM. 

To use DRAM-Sharded matmul, create your weight memory config with this helper function we created in [`model_config.py`](../../models/demos/llama3/tt/model_config.py):

```python
weights_memory_config = create_dram_sharded_mem_config(k=K, n=N)
```

This function takes care of padding weights to fit evenly into the 12 DRAM banks.

You will also have to create a program config. We have another helper function in `model_config.py` which does this for you:

```python
matmul_program_config = dram_matmul_config(
  m=M,
  k=K,
  n=N,
  num_cores=core_grid.num_cores,
)
```

The `core_grid` should be the same core grid that the activation is width-sharded on. The output will end up width-sharded on this core grid as well. Call the matmul like this:
```python
output = ttnn.linear(
  activation,
  weights,
  compute_kernel_config=compute_kernel_config,
  core_grid=None,
  dtype=ttnn.bfloat16,
  program_config=matmul_program_config,
  memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
)
```

Be careful that the core grid evenly divides both the activations and the output. Padding functionality is not yet implemented for DRAM-Sharded matmuls.

#### Matmul 1D
Matmul 1D is the final variant to cover. Before ttnn implemented DRAM-Sharded matmul, this was the matmul of choice for decode mode. Now that DRAM-Sharded matmul exists and is much faster, matmul 1D is less often used. 
Matmul 1D gets its name because it only parallelizes over the N dimension. The activation and output(s) should be width-sharded in L1. Weights should be DRAM interleaved.

To use matmul 1D, create a program config like this: 

```python
model_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
  compute_with_storage_grid_size=(cores_x, cores_y),
  in0_block_w=in0_block_w,
  out_subblock_h=out_subblock_h,
  out_subblock_w=out_subblock_w,
  per_core_M=shard_height / 32, # Shard height in tiles
  per_core_N=shard_width / 32, # Shard width in tiles
  fuse_batch=True,
  fused_activation=None,
  mcast_in0=True,
)
```

The parameters of this matmul config have the same meaning as in matmul 2D. The only difference is that each core is responsible for some width shard of the output, rather than some 2D shard of the output.
When creating a matmul 1D program config, maximize the `in0_block_w` and `out_subblock` parameters. In addition, sweep the `compute_with_storage_grid_size` to find the fastest core grid.


### 4.5 Accuracy
  - How we measure it (PCC, perplexity, top-1/top-5, end-user tests, benchmarking)
  - How much PCC is enough? Rules of thumb.
  - Accuracy tests
  - Debugging PCC issues
### 4.6 Performance Analysis
  - Performance tooling, tracy
### 4.7 Misc. Performance Optimizations
  - Which dim to shard matmuls on
  - DRAM-sharding
  - Avoiding sharded to interleaved calls
### 4.8 Module Tests
### 4.9 Performance Testing
### 4.10 Common Pitfalls
#### 4.10.1 Error Messages
  - Running out of L1
  - Shard spec and program config mismatches
  - For some TTNN ops (e.g. ttnn.all_gather) it's not supported to pass -1 in the dim argument. 
    - You'll see an error related to op invocation where the arguments don't match
#### 4.10.2 Shard Spec Mismatches
#### 4.10.3 Ethernet Dispatch Cores
  - link to any other description, and mention it is needed for N300 and T3K
#### 4.10.4 Hangs
##### 4.10.4.1 Tracing
  - Host communications cause tracing to hang
  - Running without async mode enabled causes tracing to hang
  - Careful with print in tracing
##### 4.10.4.2 Large Matmuls
  - Large matmuls hanging? Link to appropriate ticket with workaround
  - Issue is being investigated with a workaround of setting the output subblock to 1,1 and grid size to 8x7
