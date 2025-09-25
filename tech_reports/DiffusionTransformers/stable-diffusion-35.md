# Stable Diffusion 3.5 on TT Hardware

**Authors**: Dorsa Rohani, Colman Glagovich


## Contents

-  [1. Overview](#1-overview)
- [2. Modules](#2-modules)
   - [2.1 Encoder](#21-encoder)
     - [2.1.1 CLIP](#211-clip)
     - [2.1.2 T5](#212-t5)
   - [2.2 Transformer](#22-transformer)
     - [2.2.1 Architecture](#221-architecture)
     - [2.2.2 Decoder](#222-decoder)
     - [2.2.3 Attention](#223-attention)
     - [2.2.4 MLP](#224-mlp)
   - [2.3 VAE](#23-vae)
-  [3. Parallelization](#3-parallelization)
   - [3.1 CFG Parallelism](#31-cfg-parallelism)
   - [3.2 Non-attention Parallelism (TP, SP)](#32-non-attention-parallelism-tp-sp)
   - [3.3 Attention Parallelism (Ring + Head)](#33-attention-parallelism-ring--head)
   - [3.4 Topology Mapping](#34-topology-mapping)
- [4. Hardware Optimizations](#4-hardware-optimizations)
   - [4.1 Memory Management](#41-memory-management)
- [5. Testing and Demo](#5-testing-and-demo)
   - [5.1 Performance Testing](#51-performance-testing)
   - [5.2 Demo Usage](#52-demo-usage)



## Terminology

- **Spatial**: Image data
- **Prompt**: Text input provided by the user
- **CFG Parallelism**: Runs the model twice in parallel - once with a prompt (conditional, learns text-image correlations), and once without (unconditional, learns natural image distribution). Combines outputs using a guidance scale
- **Positive/Negative CFG**:
  - *Positive*: e.g., "a red cat on the moon"
  - *Negative*: e.g., "a blue dog on the sun"
- **Tensor Parallelism (TP)**: Splits model weights and computation across multiple devices
- **Sequence Parallelism (SP)**: Splits the tokens themselves so each device processes a different portion
- **Tile Layout**: Data in 32x32 blocks
- **Row Major Layout**: Data stored row by row
- **Patch Embeddings**: Converts image into token-like patches for ViTs
- **Latent Space**: Compressed image representation


## 1. Overview

This document describes the full technical implementation of Stable Diffusion (SD) 3.5 on Tenstorrent (TT) hardware.

### Requirements

- **Access to TT hardware** (Wormhole N300, T3K, TG)
- Familiarity with **PyTorch, Transformers, Metal, TT-NN**
- [HuggingFace access](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) to SD 3.5 (medium or large) weights



## 2. Modules

### 2.1 Encoder

### 2.1.1 CLIP

**Two separate CLIP encoders** `text_encoder_1` and `text_encoder_2` process identical prompts. Both are instances of `CLIPTextModelWithProjection` from Huggingface transformers, and run on CPU using PyTorch.

Each encoder takes the raw user prompt as input, and outputs:
- **Sequence embeddings**: one embedding vector per token in the prompt, used for token-level attention in the transformer so each word can be attended to.
- **Pooled embedding**: a single embedding vector representing the entire prompt, used for time conditioning.

```py
self._text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(checkpoint, subfolder="text_encoder")

self._text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(checkpoint, subfolder="text_encoder_2")
```

Both outputs are converted to TT-NN tensors using `from_torch_fast()`.

The two **sequence embeddings** are concatenated along the sequence dimension. The two **pooled embeddings** are concatenated to form pooled projection for time conditioning.


### 2.1.2 T5

The T5 encoder implementation `TtT5Encoder` is a custom module built for TT hardware using TT-NN. It processes the same raw user prompt as the CLIP encoders and outputs an extended **sequence embedding**, up to 256 tokens.

Each T5 block contains two sequential layers:
- **Self-attention layer**: `TtT5LayerSelfAttention`
- **Feed-forward layer**: `TtT5LayerFF`

```py
class TtT5Block:
   def __init__(self, parameters: TtT5BlockParameters, *, num_heads: int, layer_norm_epsilon: float) -> None:
       self._attention = TtT5LayerSelfAttention(
           parameters.attention, num_heads=num_heads, layer_norm_epsilon=layer_norm_epsilon)

       self._ff = TtT5LayerFF(parameters.ff, layer_norm_epsilon=layer_norm_epsilon)
```
The encoder outputs:
- **Sequence embedding:** one embedding vector for each token in the prompt (up to 256 tokens)

This sequence embedding is concatenated with the two CLIP sequence embeddings. The merged embedding is then projected to match the transformer's expected input dimension.



### 2.2 Transformer

### 2.2.1 Architecture

The architecture consists of **24 transformer blocks** processing multimodal inputs (spatial + text) through joint attention mechanisms.

Each transformer block takes as input:
- **Spatial embeddings**: Derived from the VAE latent space (the image's tokens).
	- The latent space is divided into fixed-size patches and linearly projected into **spatial tokens**, each representing a localized region of the compressed image.
- **Prompt embeddings**: Token-level embeddings produced by the CLIP encoders.
- **Time embedding** (`time_embed`): A conditioning vector representing the current diffusion timestep, formed by summing two learned projections.

Each block applies time conditioning using AdaLN-Zero, a method that injects timestep information into various parts of the network by chunking and linearly projecting the activated time embedding.

The `chunk_time()` function splits the embedding along the last dimension.

If the block includes spatial self-attention, the spatial time embedding is chunked into **9 components**:
- **Dual attention** (shift, scale, gate)
- **Feed-forward** (shift, scale, gate)
- **Spatial attention** (shift, scale, gate)
    - If the block has no spatial attention, only 6 chunks are used (dual attention and feed-forward only).

In context-pre-only mode (used in later blocks), prompt tokens receive only 2 chunks: shift and scale for attention. In full mode, prompt tokens receive 6 chunks (shift, scale, gate for both attention and feed-forward).

This chunking method supports multiple SD3.5 variants with minimal branching in code.

The output of the transformer stack is a set of **denoised spatial tokens**, i.e., the cleaned image in latent space.

### 2.2.2 Decoder

The decoder converts the output of the transformer blocks - **spatial tokens** - back into a 2D spatial format suitable for the VAE.

It takes a flat sequence of 4096 spatial tokens per image as input, each with a transformer dimension of 2432 (SD 3.5 large).

A final timestep embedding is applied to adjust the output. This allows conditioning using AdaLN.

```py
spatial = sd_layer_norm(spatial, parameters.norm_out) * (1 + scale) + shift
return sd_linear(spatial, parameters.proj_out)
```

### 2.2.3 Attention


Each transformer block has **dual attention paths**: one for **spatial tokens** (image) and one for **prompt tokens** (text). These paths are processed separately but interact through joint attention operations.

Some blocks have an additional **spatial self-attention**, where spatial tokens attend only to each other. Before attention, the spatial tokens are conditioned with time-dependent parameters. The attention output is then gated. This path uses **spatial** specific **time embedding chunks**, and is only active in blocks where `parameters.spatial_attn` is not None.

The core of each block is the **dual attention** module, which jointly attends over both spatial and prompt tokens. The joint attention is implemented in `sd_joint_attention()` which processes both spatial and prompt tokens in parallel.

Each attention stream (spatial and prompt) goes through:
- **QKV projection**: a single fused matmul (`_merge_qkv_proj`) for all Q, K, V weights, followed by splitting. This is an optimization that reduces matmul overhead and improves perf 3x over separate matmuls.
- **Multi-head reshape**: The qkv tensor is split and reshaped for multihead attention.

```py
num_local_heads = num_heads // parallel_config.tensor_parallel.factor

q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
   qkv,
   num_heads=num_local_heads,
   transpose_key=False)
```

### 2.2.4 MLP

Each transformer block has **separate MLPs for spatial and prompt tokens**. These MLPs are defined in the block parameters:

```py
class TtTransformerBlockParameters:
    # MLP for spatial/image tokens
   spatial_ff: TtFeedForwardParameters
   # MLP for prompt/text tokens (optional in context-pre-only mode)
   prompt_ff: TtFeedForwardParameters | None
```

The MLP architecture is a standard feed-forward structure with a **4x hidden dimension expansion** and a fused GELU activation in the input projection. After attention, each token stream is normalized and passed through its respective MLP.

### 2.3 VAE

The VAE reconstructs the final image from the denoised latent representation, performing an **8x upsampling**.

The mid block uses a ResNet for local feature refinement with residual connections, self attention for text integration across spatial dimensions, and another ResNet for further feature refinement.



## Parallelization

All parallelism options are configured with `DiTParallelConfig`, which defines how each axis is split across the device mesh.

```python
class DiTParallelConfig(NamedTuple):
   cfg_parallel: ParallelConfig
   tensor_parallel: ParallelConfig
   sequence_parallel: ParallelConfig
   ring_parallel: ParallelConfig
   ulysses_parallel: ParallelConfig
```

```py
spatial_time = sd_linear(ttnn.silu(time_embed), parameters.time_embed_out)
   [scale, shift] = chunk_time(spatial_time, 2)
   if parallel_config.tensor_parallel.factor > 1:
       spatial = utils.all_gather(spatial, dim=-1)
   spatial = sd_layer_norm(spatial, parameters.norm_out) * (1 + scale) + shift
   return sd_linear(spatial, parameters.proj_out)
```

## 3. Parallelization
### 3.1 CFG Parallelism

CFG parallelism lets us run both the *conditional* and the *unconditional* model passes simultaneously by splitting the workload across two device groups. CFG parallelism applies to DiT, not VAE. If there is a negative prompt, it might apply to CLIP/T5.

- **Conditional pass**: uses the user prompt.
- **Unconditional pass**: runs with an empty or null prompt to model natural image priors.

These are run in parallel and their outputs are later combined using the guidance scale to steer the final image. For example, when `DiTParallelConfig.cfg_parallel.factor = 2`, the device mesh is divided into 2 groups:
- **Group 0** → runs the conditional pass
- **Group 1** → runs the unconditional pass

This reduces the **batch size per device group** for better memory utilization, to be discussed in [4.1 Memory Management](#41-memory-management).

The final CFG is done after both passes are complete:
```py
latents = unconditional + guidance_scale * (conditional - unconditional)
```

### 3.2 Non-attention Parallelism (TP, SP)

Non-attention operations are parallelized using **tensor parallelism (TP) and sequence parallelism (SP)**. Together, computation is sharded across the device mesh.

> **__NOTE__**: "Non-attention parallelism" here refers to both SP and TP as applied to feed-forward layers. In contrast, attention uses different parallel axes.

Each transformer block's MLP consists of two main projections:
- **Input projection** (`in_proj`): expands from 2432 → 9728
- **Output projection** (`out_proj`): projects back from 9728 → 2432

These are sharded using two types of TP (megatron-LM style parallelism):
- `in_proj`: column parallel → shard along output features
- `out_proj`: row parallel → shard along input features

This setup lets each device compute a **partial result** of the MLP, which is then combined with reduce-scatter across devices.

```python
class TtFeedForwardParameters:
   # input projection: 2432 → 9728 (4x expansion)
   in_proj: TtLinearParameters
   # output projection: 9728 → 2432 (4x reduction)
   out_proj: TtLinearParameters

   def from_torch(cls, state: dict[str, torch.Tensor], *, dtype: ttnn.DataType | None = None, device: ttnn.MeshDevice, parallel_config: DiTParallelConfig) -> TtFeedForwardParameters:
       return cls(
           in_proj=TtLinearParameters.from_torch(
               substate(state, "net.0.proj"), dtype=dtype, device=device, shard_dim=-1  # column parallel
           ),
           out_proj=TtLinearParameters.from_torch(
               substate(state, "net.2"), dtype=dtype, device=device, shard_dim=-2      # row parallel
           ),
       )
```

```python
def sd_feed_forward(x: ttnn.Tensor, parameters: TtFeedForwardParameters, parallel_config: DiTParallelConfig) -> ttnn.Tensor:
   grid_size = x.device().compute_with_storage_grid_size()
   core_grid = ttnn.CoreGrid(x=grid_size.x, y=grid_size.y)

   x3 = sd_linear(x, parameters.in_proj, core_grid=core_grid, activation="gelu")

   result = sd_linear(x3, parameters.out_proj, core_grid=core_grid)
   ttnn.deallocate(x3)

   if parallel_config.tensor_parallel.factor > 1:
       result = ttnn.reduce_scatter(
           result,
           dim=-1,  # sum along feature dimension
           math_op=ttnn.ReduceType.Sum,  # sum all partial results
           num_links=1,  # single comm link per device
           memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
           topology=ttnn.Topology.Ring,  # ring pattern
       )
   return result
```

While **TP slices each token's embedding** (i.e., shards across hidden dim), **SP slices the tokens themselves** (i.e., shards across sequence dim) so each device processes a different portion of the sequence.

### 3.3 Attention Parallelism (Ring + Head)

Due to TP, head and hidden dimensions are padded to be divisible by the number of parallel shards. In SD 3.5 large, 38 heads are padded to 40. Similarly, 9728 hidden dim in MLP may be split across devices (e.g., 2432 // 2 = 1216). This ensures equal workload across shards and simplifies kernel implementations.

The SD3.5 large model has 38 attention heads, which cannot be evenly divided across the device mesh sizes. To solve this, **head count and hidden dimensions are padded** to a more suitable value.

In this case:
- `num_heads = 40`
- 8 devices → 40 / 8 = 5 heads per device

```python
if os.environ["MESH_DEVICE"] == "T3K" and embedding_dim == 2432:
   pad_embedding_dim = True
   hidden_dim_padding = (
       ((embedding_dim // num_devices // TILE_SIZE) + 1) * TILE_SIZE
   ) * num_devices - embedding_dim
   num_heads = 40
else:
   hidden_dim_padding = 0
   num_heads = torch_transformer.config.num_attention_heads
```

Heads are split across devices for tensor parallelism through `ttnn.transformer.split_query_key_value_and_split_heads`:

```python
n_local_heads = num_heads // device.get_num_devices()
return cls(
           spatial=TtAttentionPartParameters(
               qkv_proj=TtLinearParameters.from_torch_col_parallel(
                   state=spatial_qkv_proj,
                   n_local_heads=n_local_heads,
                   unpadded_num_heads=unpadded_num_heads,
                   hidden_dim_padding=hidden_dim_padding,
                   dtype=dtype,
                   device=device,
               ),

q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
       qkv, num_heads=num_local_heads, transpose_key=False
   )
```

To enable head-level tensor parallelism, **QKV weights must be reshuffled** so that each device holds full Q, K, V weight matrices only for the heads it owns. This transformation allows column-parallel QKV matmuls where each device independently computes attention for its own heads.

```python
def shuffle_heads(tensor):
   in_dim = tensor.shape[0]
   tensor = tensor.reshape(in_dim, 3, device.get_num_devices(), n_local_heads, -1)
   tensor = tensor.permute(0, 2, 1, 3, 4)
   tensor = tensor.reshape(in_dim, -1)
   return tensor
```

### 3.4 Topology Mapping

#### Device Mesh Layouts

| System | Mesh Shape | Devices | Notes |
|--------|------------|---------|-------|
| N300   | (1, 2)     | 2       | Linear topology only; no ring |
| T3K    | (1, 8)     | 8       | Can use ring topology; linear layout |
| TG     | (8, 4)     | 32      | 2D grid; only linear ops allowed; no ring in 2D |

**Mesh shape**: tuple (y, x) is a grid of devices where x is the number of horizontal tiles and y is the vertical. Communication axes (used for all-gather, reduce-scatter, etc) come from this layout and used in defining parallel axes (`mesh_axes`) in `ParallelConfig`.

The topology selection logic is as follows:

```python
# from TT-NN CCL all_gather:
if num_devices == 1:
   # no communication needed
   return input_tensor
elif num_devices == 2:
   topology = ttnn.Topology.Linear   # force linear for 2 devices
elif cluster_axis is not None:
   topology = ttnn.Topology.Linear   # required for 2D mesh operations
else:
   topology = ttnn.Topology.Ring     # ring for >2 devices
```

T3K (1x8) supports ring topology, used in reduce-scatter operations for MLP, attention, and more. TG (8x4) must use linear topology along a single axis due to its 2D layout; ring not supported. N300 (1x2) can only use linear, no ring support.

`DiTParallelConfig` configures which mesh axis and factor split for each parallelism mode:

```python
parallel_config = DiTParallelConfig(
    cfg_parallel=ParallelConfig(mesh_shape=(1, 8), factor=2, mesh_axis=0),       # split mesh into 2 for CFG
    tensor_parallel=ParallelConfig(mesh_shape=(1, 8), factor=4, mesh_axis=1),    # shard across devices in X
)
```

- `mesh_shape`: must match the physical layout of the system
- `factor`: number of groups/partitions along the axis
- `mesh_axis`: axis along which the partitioning occurs (0 = vertical, 1 = horizontal)

## 4. Hardware Optimizations

### 4.1 Memory Management

CFG parallelism, i.e. using `cfg_parallel` with a factor >1, reduces the batch size per device group where (number of groups) = `cfg_parallel.factor`. This results in lower memory per device group and the ability to run conditional and unconditional passes in parallel.

As well, TT hardware supports two major data layouts:
- **Tile Layout**: use cases in matrix ops (matmul), attention layers, linear projections
- **Row-major Layout**: use cases in embedding lookup, tensor reshaping, sequential access ops

The tile layout is optimized for 32x32 compute tiles on Tensix cores. This is used extensively in matmul kernels, QKV projections and attention computation, MLP layers, and more, as it enables high compute utilization and avoids DRAM bottlenecks.

The row-major layout is used for memory-aligned operations such as token embedding (e.g., in T5 encoder), positional encodings, and tensor reshaping. They are better suited for read-heavy, sequential patterns.

The dynamic memory configuration is selected based on sequence length. This ensures memory is allocated flexibly for large prompts without overloading SRAM:

```python
if sequence_length > 1024:
   mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG
elif sequence_length >= 512:
   mm_a_y, mm_a_x = 8, 8
   mm_a_x_memory_config = ttnn.DRAM_MEMORY_CONFIG
```

Below are the specialized memory settings per module:

**T5 encoder**:
- Uses row-major layout for token embedding lookups
- Embeddings are sequential reads → DRAM friendly
- Relative position bias matrices are cached for reuse
- All multihead attention ops are fused using TTNN's optimized primitives

**Transformer blocks**:
- All attention and MLP matmuls use tile layout
- Outputs are stored in tile layout for reuse in downstream layers
- Some operations (e.g., time embedding processing) fallback to row-major

**VAE**:
- Uses `ttnn.DRAM_MEMORY_CONFIG` across all layers
- Suitable for large feature maps

While tt-metal supports tensor sharding, it is not used in this implementation. Sharding is typically used to reduce memory-bound operation costs by distributing data across devices. However, in SD3.5, the bottleneck is compute-bound, not memory-bound - so sharding provides no added benefit.

Instead, DRAM tensors are interleaved automatically across tiles/devices using the runtime's memory allocator.

## 5. Testing and Demo

For instructions on running the demo, refer to [here](https://github.com/tenstorrent/tt-metal/blob/main/models/experimental/stable_diffusion_35_large/README.md).

### 5.1 Performance Testing

TODO - sometime soon we'll insert tables for each system, each parallel config, expected perf and desired perf

### 5.2 Demo Usage

The demo script tests SD inference using TT hardware. It supports both automated and interactive image generation workflows and is designed to highlight performance across various device configs.

The pipeline encompasses `TtStableDiffusion3Pipeline` from `tt/pipeline.py`. Fallback behaviour includes the T5 encoder running only on devices with >=4 cores and the VAE decoding running on CPU by default.

There are two modes:

**Non-interactive**: generates an image using a fixed prompt
- For example, "A snowy cabin in the woods, cinematic lighting"
- No user input required
- Saves output to disk

**Interactive Mode** (default): prompts user for input in a loop
- Accepts natural language prompts
- Generates an image for each input
- Exits when the user types "q"
