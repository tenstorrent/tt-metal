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

For added performance, our implementation of Llama uses a fused operation to apply the Rotary Position Embeddings (RoPE), which can be accessed via `ttnn.experimental.rotary_embedding_llama` API. In the Attention module, this API is called twice, one for the queries and one for the keys respectively.

Here is an example of how the fused RoPE op is used in the attention module:

```py
q_heads = ttnn.experimental.rotary_embedding_llama(
    q_heads_pre_rot,
    cos_matrix,
    sin_matrix,
    transformation_matrix,
    is_decode_mode="False"
)

k_heads = ttnn.experimental.rotary_embedding_llama(
    k_heads_pre_rot,
    cos_matrix,
    sin_matrix,
    transformation_matrix,
    is_decode_mode="False"
)
```

#### Setting up inputs to RoPE

The fused operation uses a different parallelization scheme internally depending on of the model is in *prefill* or *decode* mode. Across these modes, the shapes and memory configs of the inputs vary and this table summararizes them below:

|     is_decode_mode    |                         True                        |                        False                       |
|:---------------------:|:---------------------------------------------------:|:--------------------------------------------------:|
|         Input         | [1, batch, n_heads, head_dim], <br>HEIGHT_SHARDED in L1 | [1, n_heads, seq_len, head_dim],  <br>INTERLEAVED in L1 |
|     Cos/Sin Matrix    |    [1, batch, 1, head_dim],  <br>HEIGHT_SHARDED in L1    |    [1, 1, seq_len, head_dim],  <br>INTERLEAVED in L1    |
| Transformation Matrix |     [1, 1, TH * batch, TW],  <br>HEIGHT_SHARDED in L1    |          [1, 1, TH, TW],  <br>INTERLEAVED in L1         |

*Note: (TH, TW) = (TILE_HEIGHT, TILE_WIDTH)*


#### Decode mode specifics
The cos/sin matrices, are generated in two slightly different ways, depending on the mode of operation. For *prefill* mode, the cos/sin matrices are computed once at intialization using the *prefill* sequence length, and then passed into the RoPE op. However, in *decode* mode, since the position index of each user is updated from token-to-token, the cos/sin matrices need to be updated across iterations. Here, we leverage our `TtLlamaRotarySetup` module, that can be used at each decode iteration to get the corresponding cos/sin matrices.

This is an example of how `TtLlamaRotarySetup` can be used in decode mode:
```py
from llama_rope import TtLlamaRotarySetup

# Step 1: Create the setup object
rope_setup_decode = TtLlamaRotarySetup(
    mesh_device,
    head_dim,
    max_seq_len,
    rope_theta,
    use_scaled_rope
)

transformation_mats_decode = rope_setup_decode.get_trans_mats()


# Step 2: Get user position ids
# For example, batch number of users, each with different position ids
position_ids = torch.arange(batch)


# Step 3: Retreive the relevant cos/sin matrices
cos_sin_matrices = rope_setup_decode.get_rot_mats(position_ids)
cos_matrix, sin_matrix = cos_sin_matrices


# Step 4: Perform the RoPE operation
out = ttnn.experimental.rotary_embedding_llama(
    x,  # example input
    cos_matrix
    sin_matrix,
    transformation_mats_decode,
    is_decode_mode=True
)

```
<br>

#### Quick note about the transformation matrix
Due to the sparse nature of the transformation matrix, the fused RoPE op takes as input a tile-sized transformation matrix, and then reuses that tile across all subsequent operations. In *decode* mode, this matrix is replicated *batch* times, and then sharded over *batch* number of cores. As a result, each core receives a single, tile-sized transformation matrix. In contrast, the *prefill* mode implementation requires only a single tile-sized transformation matrix, and then distributes it across all the cores internally.


### 2.3 Norm
  - Replicated layernorm vs distributed layernorm
    - Layernorm/rmsnorm weights in row major / wrapped around tile size trick
### 2.4 Attention
  - Flash Attention and Flash Decode
    - general description
    - limitations
    - which dims are parallelized
### 2.5 MLP

The MLP for the Llama models is implemented in the the `TtLlamaMLP` module class. The tests for this module are available in the `test_llama_mlp.py` file.

As an overview, the MLP performs the following operations on an input `x`:
```
w1_out = FF1(x)
w3_out = FF3(x)
w2_in = SiLU(w1_out) * w3_out
y = FF2(w2_in)
```
where FF1, FF2, and FF3 are linear transformations (matmuls) with weights `w1`, `w2`, and `w3` respectively. Since FF1 and FF3 share the same inputs, their optimizations are shared as well.


Let's dive into our implementation of MLP, and discuss what makes it performant across different WH systems.

#### 0. Setup
When used in the model by the `TtLlamaDecoder` module class, the MLP class is initialzed at the start, where the weights for `w1`, `w2`, and `w3` are loaded and fractured across devices in specific schemes, as outlined in section 3.3 Multi-Device. Specifically, in n300 and T3000 systems the weights are 1D column fractured, and in TG systems the weights are 2D fractured.

```py
self.feed_forward = TtLlamaMLP(
    mesh_device=mesh_device,
    args=args,
    state_dict=state_dict,
    weight_cache_path=weight_cache_path,
    layer_num=layer_num,
    dtype=dtype,
    model_config=self.model_config,
)
```

#### 1. Inputs
Then, at runtime, the `forward` function of `TtLlamaMLP` is called with a mode (*'prefill'* or *'decode'*), with inputs that are replicated across devices, for all WH system configurations. Note, in the actual model, the input `ff_in` is the output of the `norm` step prior to MLP (See norm section below).

**Decode mode**

In *decode* mode, the inputs have a maximum batch of 32, where each user only has a single token. As such, the inputs in *decode* mode are considered to be much smaller compared to in *prefill* mode, where the sequence length can be up to 128k. To make our subsequent matmul operations faster in *decode* mode, we can shard the input across L1, where they can be processed by the matmul, without any extra time for loading. The specific core grid to shard on, `mlp_core_grid` is chosen to be the lowest number of cores that the input can be width sharded on, while maintaining tile size. This is so we can minimize any communication delay over the NOC, when moving around the activations during the matmul.


```py
# ff_in shape: [1, 1, m, k] => [1, 1, batch, dim]
ff_in_memory_config = ttnn.create_sharded_memory_config(
    (m, k // mlp_core_grid.num_cores),
    mlp_core_grid,
    ttnn.ShardStrategy.WIDTH,
    ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
```

**Prefill mode**

As mentioned before, the input in prefill mode can be very large, and may not fit in the available L1 space. As such, the inputs are stored in DRAM.

```py
# ff_in shape: [1, 1, m, k] => [1, 1, seq_len, dim]
ff_in_memory_config = ttnn.DRAM_MEMORY_CONFIG
```

Note, similar to the Attention module, the matmul operation can exceed memory if the inputs are too large, and as a workaround, we push part of the sequence length into the batch dimension.
```py
# Reshape input to to fit on device and parallelize computation
if seq_len >= 1024:
    ff_in = ttnn.reshape(ff_in, [1, seq_len // 1024, 1024, -1])
```



#### 2. Setting up program configs for the matmuls
Depending on the mode of operation, the `forward` function of `TtLlamaMLP` instantiates different program configs for the matmuls of FF1/FF3, and FF2.


**Decode mode**

Since the weights are much larger than the activations, and the weights must be loaded from DRAM, these matmul operations are DRAM-bound. This means that loading the weights from DRAM is a bottleneck, rather than the computation itself. As such, we use DRAM sharded matmuls in decode mode, which are more performant than regular mamtuls (See section _ for details).

```py
_, _, m, k = ff_in.shape
n = hidden_dim // num_devices # Since w1/w3 are fractured on outer dim
pc1 = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    in0_block_w=math.ceil(k / (tile_size * ff1_num_cores)),
    per_core_M=math.ceil(m / tile_size),
    per_core_N=math.ceil(n / (tile_size * ff1_num_cores)),
    fused_activation=None,
)

k, n = n, k  # Since FF1 is up projection and FF2 is down projection
pc2 = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
    in0_block_w=math.ceil(k / (tile_size * ff2_num_cores)),
    per_core_M=math.ceil(m / tile_size),
    per_core_N=math.ceil(n / (tile_size * ff2_num_cores)),
    fused_activation=None,
)
```

**Prefill mode**

In prefill mode, since the activation and weights are similarly shaped, loading activations and weights from DRAM is no longer a bottleneck. Instead, for these compute bound matmul operations, we utilize a 2D matmul.

The specific paramers for the program configs are chosen to maximize matmul performance, based on the shapes of the inputs. See section _ for more details.

```py
# TODO: Move this function to a different section and just refer to it
def matmul_config(
    m: int,
    k: int,
    n: int,
    grid_size: Tuple[int, int],
    in0_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    per_core_M = math.ceil(m / (tile_size * grid_size[1]))
    per_core_N = math.ceil(n / (tile_size * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = get_out_subblock_w(per_core_N, out_subblock_h)

    if in0_block_w is None:
        in0_block_w = min(4, max(1, k // (tile_size * grid_size[0])))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


_, _, m, k = ff_in.shape
n = hidden_dim // num_devices
pc1 = matmul_config(
    m=m, k=k, n=n, grid_size=(8, 8)
)

k, n = n, k  # Since FF1 is up projection and FF2 is down projection
pc1 = matmul_config(
    m=m, k=k, n=n, grid_size=(8, 8)
)
```


#### 3. FF1/FF3 matmul
The first set of operations in the MLP are:
```py
w1_out = FF1(x)
w3_out = FF3(x)
```
Based on the program configs we computed beforehand, we perform the FF1/FF3 matmuls, making sure that the ouputs are L1 sharded in in decode mode, and interleaved in DRAM if in prefill mode. For the `compute_kernel_config`, we use `ttnn.MathFidelity.HiFi2` to retain accuracy while still being performance. Using `ttnn.MathFidelity.HiFi4` instead, would mean that this matmul would become compute bound.

```py
compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

w1_out = ttnn.linear(
    ff_in,
    w1,
    compute_kernel_config=args.compute_kernel_config_hifi2,
    core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
    dtype=ttnn.bfloat16,
    program_config=pc_1,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
)

w3_out = ttnn.linear(
    ff_in,
    w3,
    compute_kernel_config=args.compute_kernel_config_hifi2,
    core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_1 else None,
    dtype=ttnn.bfloat16,
    program_config=pc_1,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
)
```

#### 3.1 FF1/FF3 Matmul with 2D Weight Fracturing

In the case of TG systems, where have have access to a 2D device mesh, we can leverage 2D weight fracturing. For a weight tensor with shape `[1, 1, K, N]`, using 2D weight fracturing on a `(8, 4)` device mesh, the resulting shape on each device would be: `[1, 1, K / 4, N / 8]`. In other words, the inner dimension (K) of the matmul is spread out across 4 devices, and to complete the entire matmul operation, a reduction step across the partials is necessary. We do this using an all-reduce operation along the 4 devices in `cluster_axis=1` of the device mesh.
```py
  w1_out = tt_all_reduce(
      w1_out,
      self.mesh_device,
      cluster_axis=1,
      num_links=2,
      sharded=True if mode == "decode" else False,
      memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
  )
  w3_out = tt_all_reduce(
      w3_out,
      self.mesh_device,
      cluster_axis=1,
      num_links=2,
      sharded=True if mode == "decode" else False,
      memory_config=self.model_config["FF1_OUT_GATHERED_MEMCFG"] if mode == "decode" else None,
  )
```

#### 4. Multiply + fused SiLU activation

The output of the FF1/FF3 matmuls are column fractured tensors (the extra all-reduce operation for TG systems ensures this). The next operation is:
```py
w2_in = SiLU(w1_out) * w3_out
```
In ttnn, we have access to binary operations that can apply activations to any of the inputs, in a fused manner, leading to better performance as the inputs are only getting loaded/processed once. As such, the fused SiLU operation with the element-wise multiplication can be performed as follows:
```py
w2_in = ttnn.multiply(
    w1_out,
    w3_out,
    memory_config=(
        self.model_config["SHARDED_MLP2_INPUT_MEMCFG"] if TG else ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    )
    if mode == "decode"
    else ttnn.DRAM_MEMORY_CONFIG,
    input_tensor_a_activation=ttnn.UnaryOpType.SILU,
    dtype=ttnn.bfloat8_b,
)
```

Following our pattern mentioned before, the outputs are L1 sharded in `decode` mode and DRAM interleaved in `prefill` mode.

#### 5. FF2 Matmul
The last computation in MLP is:
```py
y = FF2(w2_in)
```
FF2 is a row-parallel matmul, meaning that that the weights are fractured across devices in the inner dim. The inputs of FF2, produced by FF1/FF3, are also fractured across devices in the same dimension and as a result, FF2 produces partial outputs across all devices.

Here's what the call for the FF2 matmul looks like. Note, that once the matmul operations are completed, we can undo the reshape operation we performed on the inputs of MLP to fit the matmuls on device in `prefill`.
```py
w2_out = ttnn.linear(
    w2_in,
    self.w2,
    compute_kernel_config=self.args.compute_kernel_config_hifi2_fp16,
    core_grid=ttnn.CoreGrid(y=1, x=8) if not pc_2 else None,
    dtype=ttnn.bfloat16,
    program_config=pc_2,
    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
)

# Undo the reshape operation used to fit the matmul on device
if seq_len >= 1024:  # Reshape back to intended shape
    w2_out = ttnn.reshape(w2_out, [1, 1, seq_len, -1])
```

5.1 Accumulating the partial outputs of FF2

Since the output of FF2 is the correct shape, but only a partial on each device. The output of the MLP module is required to be fractured, where each device has fully accumulated the inner dim of the matmul, but only has a fraction of the outer dim. There are two different cases to handle this, depending on if the WH system has a 1D or 2D device mesh.

1. 1D Device Mesh (n300, T3000): reduce-scatter operation across all devices, resulting in outputs fractued in the outer dim.
    ```py
    w2_out_reduced = ttnn.reduce_scatter(
        w2_out,
        scatter_dim=3,
        math_op=ttnn.ReduceType.Sum,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if mode == "prefill" else ttnn.L1_MEMORY_CONFIG,
    )
    ```
2. 2D Device Mesh (TG): all-reduce operation along the same cluster axis as which the inner dimension is fractured on. The FF2 matmul inner dim is fractured across cluster axis 0 (row-parallel across 8 device), and the outer dim is fractured across cluster axis 1 (4 devices). Then an all-reduce performed on cluster axis 0 will accumulate the partials across the inner dim of the matmul and replicate them along all the devices in that axis, while still keeping them fractured across cluster axis 1 (4 devices).
    ```py
    w2_out_reduced = tt_all_reduce(
        w2_out,
        self.mesh_device,
        cluster_axis=0,
        num_links=2,
        dim=0,
        memory_config=(self.model_config["FF2_OUT_GATHERED_MEMCFG"],
        sharded=(mode == "decode"),
    )
    ```

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
  - Writing correct program configs and shard specs
  - Deciding how many cores to run an op on
    - Why did we use 16 cores for MLP
  - Which matmul to use when @Colman Glagovich
    - 1d, 2d, dram-sharded, ...
  - Implicitly padding weights in program config for matmuls
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
