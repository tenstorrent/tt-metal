# TTNN Functions Reference for LLM Transformers

This document provides a comprehensive list of all `ttnn` functions used in the LLM transformer codebase, with descriptions and context for each function.

It is auto-generated for the benefit of LLM assistants writing code. It may be incorrect or incomplete and has not been verified by a human.

## Tensor Creation and Conversion Functions

### `ttnn.from_torch(tensor, device=None, dtype=None, layout=None, mesh_mapper=None, memory_config=None, cache_file_name=None)`
Converts a PyTorch tensor to a ttnn tensor. Used for loading weights and activations onto device.

### `ttnn.as_tensor(tensor, device=None, dtype=None, layout=None, mesh_mapper=None, memory_config=None, cache_file_name=None)`
Creates a ttnn tensor from a PyTorch tensor with optional device placement and configuration. Similar to `from_torch` but with different API semantics.

### `ttnn.to_torch(tensor, mesh_composer=None)`
Converts a ttnn tensor back to PyTorch tensor on host. Used for reading outputs and debugging.

### `ttnn.to_device(tensor, device, memory_config=None)`
Moves a tensor to a specific device with optional memory configuration.

### `ttnn.to_memory_config(tensor, memory_config, dtype=None)`
Changes the memory configuration of a tensor (e.g., from DRAM to L1, or changing sharding).

### `ttnn.typecast(tensor, dtype)`
Changes the data type of a tensor (e.g., from bfloat16 to bfloat8_b).

### `ttnn.to_layout(tensor, layout)`
Changes the tensor layout (e.g., from TILE_LAYOUT to ROW_MAJOR_LAYOUT).

## Tensor Shape Manipulation

### `ttnn.reshape(tensor, shape, padded_shape=None)`
Reshapes a tensor to a new shape, with optional padded shape for memory alignment.

### `ttnn.unsqueeze_to_4D(tensor)`
Adds dimensions to make a tensor 4D, commonly used for preparing inputs for attention and MLP operations.

### `ttnn.slice(tensor, start, end)`
Extracts a slice from a tensor. Used to get the last token from prefill outputs.

### `ttnn.transpose(tensor, dim0, dim1)`
Transposes two dimensions of a tensor. Used in RoPE operations and attention reshaping.

### `ttnn.clone(tensor)`
Creates a copy of a tensor.

## Linear Algebra Operations

### `ttnn.linear(input, weight, bias=None, memory_config=None, program_config=None, compute_kernel_config=None, dtype=None, core_grid=None)`
Performs matrix multiplication (linear transformation). Core operation for QKV projections, MLP layers, and output projections.

### `ttnn.matmul(tensor_a, tensor_b, memory_config=None, program_config=None, compute_kernel_config=None, dtype=None, core_grid=None)`
General matrix multiplication operation. Used for attention computations and some specialized matrix operations.

## Element-wise Operations

### `ttnn.add(tensor_a, tensor_b, memory_config=None, dtype=None)`
Element-wise addition. Used for residual connections and bias addition.

### `ttnn.mul(tensor_a, tensor_b, input_tensor_a_activations=None, memory_config=None, dtype=None)`
Element-wise multiplication. Used in MLP with optional fused activation (like SiLU) on first input.

### `ttnn.multiply(tensor_a, tensor_b, input_tensor_a_activation=None, memory_config=None, dtype=None)`
Alternative element-wise multiplication API with fused activation support.

## Tensor Layout and Memory Operations

### `ttnn.interleaved_to_sharded(tensor, memory_config)`
Converts an interleaved tensor to sharded layout across cores. Used for optimizing decode operations.

### `ttnn.sharded_to_interleaved(tensor, memory_config, dtype=None)`
Converts a sharded tensor to interleaved layout. Used when operations require interleaved inputs.

### `ttnn.untilize(tensor, use_multicore=False)`
Converts from tiled layout to row-major layout. Used before host transfer or certain operations.

### `ttnn.tilize(tensor, use_multicore=False)`
Converts from row-major layout to tiled layout for device operations.

## Attention Operations

### `ttnn.transformer.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=True, scale=None, program_config=None, compute_kernel_config=None)`
Flash attention implementation for prefill mode. Computes attention scores for full sequences.

### `ttnn.transformer.scaled_dot_product_attention_decode(query, key, value, cur_pos_tensor=None, cur_pos=None, scale=None, program_config=None, compute_kernel_config=None, memory_config=None, is_causal=True, attn_mask=None)`
Flash attention implementation optimized for decode mode with single token queries.

### `ttnn.transformer.paged_scaled_dot_product_attention_decode(query, key, value, cur_pos_tensor, page_table_tensor, scale=None, program_config=None, compute_kernel_config=None, memory_config=None)`
Paged version of decode attention that supports dynamic memory allocation with page tables.

### `ttnn.transformer.chunked_scaled_dot_product_attention(query, key, value, page_table, chunk_start_idx, compute_kernel_config=None, program_config=None)`
Chunked attention for processing very long sequences in smaller chunks during prefill.

## Experimental Operations

### `ttnn.experimental.nlp_create_qkv_heads(fused_qkv, num_heads, num_kv_heads, transpose_k_heads=False, memory_config=None)`
Splits fused QKV tensor into separate Q, K, V tensors for prefill mode with proper head reshaping.

### `ttnn.experimental.nlp_create_qkv_heads_decode(fused_qkv, num_heads, num_kv_heads, memory_config=None)`
Splits fused QKV tensor for decode mode with height-sharded output across batch dimension.

### `ttnn.experimental.nlp_concat_heads(tensor, memory_config=None)`
Concatenates attention heads back to original dimension for prefill mode.

### `ttnn.experimental.nlp_concat_heads_decode(tensor, num_heads)`
Concatenates attention heads for decode mode with proper batch handling.

### `ttnn.experimental.rotary_embedding_llama(tensor, cos_matrix, sin_matrix, transformation_matrix, is_decode_mode=False)`
Applies rotary position embeddings (RoPE) with fused implementation optimized for both prefill and decode modes.

### `ttnn.experimental.paged_fill_cache(cache_tensor, fill_tensor, page_table, batch_idx=0)`
Fills KV cache using page table for memory management. Used during prefill to populate cache.

### `ttnn.experimental.paged_update_cache(cache_tensor, update_tensor, update_idxs_tensor=None, update_idxs=None, page_table=None)`
Updates KV cache at specific positions using page table. Used during decode to add new K/V values.

### `ttnn.experimental.all_gather_matmul(input_tensor, weight, dim, all_gather_core_grid_offset=None, num_links=1, memory_config_ag=None, memory_config_mm=None, program_config=None, compute_kernel_config=None)`
Fused operation that combines all-gather and matrix multiplication for efficient communication and computation.

### `ttnn.experimental.fast_reduce_nc(tensor, dims, output=None, compute_kernel_config=None, memory_config=None)`
Fast reduction operation across specified dimensions.

## KV Cache Operations

### `ttnn.fill_cache(cache_tensor, fill_tensor, batch_idx)`
Fills KV cache at a specific batch index. Simpler version without page table support.

## Collective Communication Operations

### `ttnn.all_gather(tensor, dim, num_links=1, cluster_axis=None, mesh_device=None, topology=None, memory_config=None)`
Gathers tensors from all devices along specified dimension. Used for collecting distributed computations.

### `ttnn.reduce_scatter(tensor, dim, math_op=None, num_links=1, cluster_axis=None, mesh_device=None, topology=None, memory_config=None)`
Reduces tensors across devices and scatters result. Used for distributed weight computations.

## Normalization Operations

### `ttnn.rms_norm(input, epsilon=1e-5, weight=None, bias=None, memory_config=None, program_config=None, compute_kernel_config=None)`
Root Mean Square Layer Normalization. Standard normalization for single-device or replicated inputs.

### `ttnn.rms_norm_pre_all_gather(input, compute_kernel_config=None, dtype=None, program_config=None)`
First stage of distributed RMS norm that computes local statistics before gathering.

### `ttnn.rms_norm_post_all_gather(input, stats=None, epsilon=1e-5, weight=None, compute_kernel_config=None, program_config=None)`
Second stage of distributed RMS norm that applies normalization using gathered statistics.

### `ttnn.layer_norm(input, epsilon=1e-5, weight=None, bias=None, memory_config=None, program_config=None, compute_kernel_config=None)`
Standard Layer Normalization operation.

## Embedding Operations

### `ttnn.embedding(input_ids, weight, layout=None, memory_config=None)`
Looks up embeddings from weight table using input token IDs. Used for token embeddings and RoPE position lookups.

## Sampling and Output Operations

### `ttnn.argmax(tensor, dim, keepdim=False, use_multicore=False, output_tensor=None)`
Finds indices of maximum values along specified dimension. Used for greedy decoding on device.

### `ttnn.concat(tensors, dim, memory_config=None)`
Concatenates tensors along specified dimension. Used in LM head to combine output splits.

## Device and Memory Management

### `ttnn.deallocate(tensor)`
Explicitly deallocates tensor memory. Used for memory management in tight memory scenarios.

### `ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)`
Copies data from host tensor to device tensor in-place. Used for efficient tensor updates.

### `ttnn.get_device_tensors(multi_device_tensor)`
Extracts individual device tensors from a multi-device tensor.

### `ttnn.aggregate_as_tensor(tensors)`
Combines individual tensors into a multi-device tensor.

## Core and Memory Configuration Functions

### `ttnn.create_sharded_memory_config(shape, core_grid, strategy, orientation=None, use_height_and_width_as_shard_shape=False)`
Creates memory configuration for sharded tensors across cores.

### `ttnn.CoreGrid(y, x)`
Creates a core grid specification for operations.

### `ttnn.CoreRange(start_coord, end_coord)`
Defines a range of cores for operations.

### `ttnn.CoreCoord(x, y)`
Specifies coordinates of a core.

### `ttnn.CoreRangeSet(core_ranges)`
Creates a set of core ranges for complex core patterns.

### `ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)`
Converts number of cores to a core range set specification.

## Memory Configuration Constants

### `ttnn.DRAM_MEMORY_CONFIG`
Standard DRAM interleaved memory configuration.

### `ttnn.L1_MEMORY_CONFIG`
Standard L1 interleaved memory configuration.

### `ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG`
L1 memory configuration with width sharding.

### `ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG`
L1 memory configuration with height sharding.

## Layout Constants

### `ttnn.TILE_LAYOUT`
Tiled tensor layout optimized for device operations.

### `ttnn.ROW_MAJOR_LAYOUT`
Row-major tensor layout, required for some operations and host transfers.

## Data Type Constants

### `ttnn.bfloat16`
16-bit brain floating point format, standard precision for most operations.

### `ttnn.bfloat8_b`
8-bit brain floating point format for memory-efficient operations.

### `ttnn.bfloat4_b`
4-bit brain floating point format for ultra-low precision weights.

### `ttnn.uint32`
32-bit unsigned integer, used for token IDs and position indices.

### `ttnn.int32`
32-bit signed integer, used for position tracking and page tables.

## Mesh and Device Mapping

### `ttnn.ReplicateTensorToMesh(mesh_device)`
Replicates tensor across all devices in mesh.

### `ttnn.ShardTensorToMesh(mesh_device, dim)`
Shards tensor along specified dimension across mesh devices.

### `ttnn.ShardTensor2dMesh(mesh_device, dims, mesh_shape)`
2D sharding of tensor across mesh with specified dimensions and mesh shape.

### `ttnn.ConcatMeshToTensor(mesh_device, dim)`
Mesh composer for concatenating tensors from mesh devices.

### `ttnn.ConcatMesh2dToTensor(mesh_device, dims, mesh_shape)`
2D mesh composer for concatenating tensors with 2D device layout.

## Topology and Reduction Types

### `ttnn.Topology.Linear`
Linear device topology for collective communication.

### `ttnn.Topology.Ring`
Ring device topology for collective communication.

### `ttnn.ReduceType.Sum`
Sum reduction operation for collective communications.

## Sharding Strategies

### `ttnn.ShardStrategy.WIDTH`
Width-based sharding strategy.

### `ttnn.ShardStrategy.HEIGHT`
Height-based sharding strategy.

### `ttnn.ShardStrategy.BLOCK`
Block-based sharding strategy.

### `ttnn.ShardOrientation.ROW_MAJOR`
Row-major orientation for sharded tensors.

### `ttnn.ShardOrientation.COL_MAJOR`
Column-major orientation for sharded tensors.

## Compute Kernel Configuration

### `ttnn.WormholeComputeKernelConfig(math_fidelity, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)`
Compute kernel configuration for Wormhole architecture with math fidelity settings.

### `ttnn.MathFidelity.HiFi4`
Highest fidelity math mode for maximum accuracy.

### `ttnn.MathFidelity.HiFi2`
Medium fidelity math mode balancing accuracy and performance.

### `ttnn.MathFidelity.LoFi`
Low fidelity math mode for maximum performance.

## Program Configuration Classes

### `ttnn.MatmulMultiCoreReuseMultiCastProgramConfig`
2D matmul program configuration for large matmuls in prefill mode.

### `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`
DRAM-sharded matmul program configuration optimized for decode mode.

### `ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig`
1D matmul program configuration for specific use cases.

### `ttnn.LayerNormShardedMultiCoreProgramConfig`
Program configuration for sharded layer normalization operations.

## Activation Types

### `ttnn.UnaryOpType.SILU`
SiLU (Swish) activation function for fused operations.

## Tensor Size Constants

### `ttnn.TILE_SIZE`
Standard tile size (32) used throughout the codebase.

### `ttnn.TILE_HEIGHT`
Height of a tile (32).

### `ttnn.TILE_WIDTH`
Width of a tile (32).

## Utility Functions

### `ttnn.get_arch_name()`
Returns the architecture name of the current device.

## Tracing Functions

### `ttnn.begin_trace_capture(device, cq_id=0)`
Begins capturing operations for tracing optimization.

### `ttnn.end_trace_capture(device, trace_id, cq_id=0)`
Ends trace capture and returns trace ID.

### `ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)`
Executes a previously captured trace.

## Usage Patterns

### Typical MLP Forward Pass
```python
# Upsample projections (w1, w3)
w1_out = ttnn.linear(x, w1, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
w3_out = ttnn.linear(x, w3, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

# SiLU activation + multiply
w2_in = ttnn.multiply(w1_out, w3_out, input_tensor_a_activation=ttnn.UnaryOpType.SILU)

# Downsample projection
w2_out = ttnn.linear(w2_in, w2, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
```

### Attention Forward Pass - Prefill Mode
```python
# Input shape: (1, 1, seq_len, dim)
# QKV projection - larger sequence length operations use DRAM
xqkv_fused = ttnn.linear(x, wqkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Split into Q, K, V with shape (1, num_heads, seq_len, head_dim)
q, k, v = ttnn.experimental.nlp_create_qkv_heads(
    xqkv_fused,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    transpose_k_heads=False
)

# Apply RoPE for prefill (full sequence)
q_rot = ttnn.experimental.rotary_embedding_llama(
    q, cos_matrix, sin_matrix, transformation_matrix, is_decode_mode=False
)
k_rot = ttnn.experimental.rotary_embedding_llama(
    k, cos_matrix, sin_matrix, transformation_matrix, is_decode_mode=False
)

# Fill KV cache for the user (prefill populates entire cache)
ttnn.experimental.paged_fill_cache(kv_cache_k, k_rot, page_table, batch_idx=user_id)
ttnn.experimental.paged_fill_cache(kv_cache_v, v_rot, page_table, batch_idx=user_id)

# Compute attention for full sequence with causal mask
attn_output = ttnn.transformer.scaled_dot_product_attention(
    q_rot, k_rot, v_rot,
    is_causal=True,
    scale=1.0 / math.sqrt(head_dim),
    program_config=prefill_progcfg,
    compute_kernel_config=compute_kernel_config
)

# Concat heads back to (1, 1, seq_len, dim)
attn_concat = ttnn.experimental.nlp_concat_heads(attn_output)

# Output projection
output = ttnn.linear(attn_concat, wo, memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

### Attention Forward Pass - Decode Mode
```python
# Input shape: (1, 1, batch_size, dim) - batch_size users, 1 token each
# QKV projection - small activations use L1 sharded memory
xqkv_fused = ttnn.linear(x, wqkv, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)

# Split into Q, K, V with shape (1, batch_size, num_heads, head_dim)
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    xqkv_fused,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    memory_config=sharded_mem_config
)

# Apply RoPE for decode (single position per user)
q_rot = ttnn.experimental.rotary_embedding_llama(
    q, cos_matrix, sin_matrix, transformation_matrix, is_decode_mode=True
)
k_rot = ttnn.experimental.rotary_embedding_llama(
    k, cos_matrix, sin_matrix, transformation_matrix, is_decode_mode=True
)

# Update KV cache at current positions (one position per user)
ttnn.experimental.paged_update_cache(
    kv_cache_k, k_rot, update_idxs_tensor=current_pos, page_table=page_table
)
ttnn.experimental.paged_update_cache(
    kv_cache_v, v_rot, update_idxs_tensor=current_pos, page_table=page_table
)

# Compute attention using cached KV values
attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
    q_rot, kv_cache_k, kv_cache_v,
    cur_pos_tensor=current_pos,
    page_table_tensor=page_table,
    scale=1.0 / math.sqrt(head_dim)
)

# Concat heads for decode
attn_concat = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads)

# Output projection with sharded output
output = ttnn.linear(attn_concat, wo, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
```

### Key Differences Between Prefill and Decode
- **Input shapes**: Prefill processes full sequences `(1, 1, seq_len, dim)`, decode processes batches of single tokens `(1, 1, batch_size, dim)`
- **Memory configs**: Prefill uses DRAM for large activations, decode uses L1 sharding for small activations
- **QKV splitting**: Different ops for different parallelization patterns
- **KV cache**: Prefill fills entire cache, decode updates single positions
- **Attention computation**: Prefill computes full attention matrix, decode uses cached KV values
- **Performance focus**: Prefill is compute-bound, decode is memory-bandwidth-bound

This reference provides the core ttnn functions needed to implement transformer models, with sufficient detail to understand their purpose and write mock implementations.

---

# Tenstorrent Wormhole Architecture and TT-Metal/TTNN: A Model Writer's Perspective

## Wormhole Hardware Architecture

### Core Architecture
The Wormhole architecture consists of an **8x10 grid of Tensix cores** connected in a 2D torus via a Network-on-Chip (NoC). Each Tensix core contains:

- **1.5 MB of L1 memory (SRAM)**
- **5 programmable RISC-V cores**:
  - **RISC0 and RISC1**: Capable of issuing NoC transfers for data movement (L1↔L1 and L1↔DRAM)
  - **RISC 2, 3, and 4**: Responsible for issuing instruction streams to the tile-based matrix processing engine
- **Concurrent execution**: All RISCs operate concurrently, enabling native pipelining and data movement/compute overlap

### Memory and Interconnect Specifications

**DRAM Subsystem:**
- **12 channels of 1GB GDDR6**
- **Total DRAM bandwidth: 288 GB/s**

**Ethernet Connectivity:**
- **16 Ethernet cores** connected to the NoC
- Each Ethernet core contains:
  - Single RISC-V processor
  - 256KB of L1 space
  - Ethernet subsystem for inter-chip communication
- **Per-core capability: 100Gbps bidirectional**
- **Aggregate bandwidth: 1600Gbps per direction (3200 Gbps/400 GB/s bidirectional)**

**Ethernet Packet Specifications:**
- Minimum packet size: 16B
- Maximum packet size: 1500B
- Packet overheads: 50+B (includes headers and CRC)

### Hardware Configurations

**N150:**
- Single Wormhole chip PCIe card
- Operating power: up to 160W
- Direct PCIe host connection

**N300:**
- **2-chip configuration** on single board
- **Local chip**: Direct PCIe connection to host
- **Remote chip**: Accessible only through Ethernet via local chip
- **Inter-chip bandwidth**: 200 Gbps per direction (with 100 Gbps available for user kernels after dispatcher overhead)
- 6 additional external Ethernet ports

**T3000:**
- **8 Wormhole chips** in 2x4 mesh configuration
- Assembled from 4 N300 parts using external ports
- **PCIe-accessible chips**: 0, 1, 2, 3
- Same Ethernet performance characteristics as N300

**Galaxy:**
- **32 Wormhole chips** in 4x8 2D mesh
- Non-edge/corner chips utilize all 16 Ethernet links
- Edge/corner chips have unused Ethernet cores for inter-Galaxy connectivity
- Link speed: 12.5 GB/s per direction per link (25 GB/s bidirectional)

## Matrix Engine Performance Characteristics

### Core Compute Specifications
- **Matrix engine operation**: 8x16 × 16x16 = 8x16 per cycle
- **Operations per cycle**: 2×8×16×16 = 4,096 multiply-adds
- **At 1GHz**: 4 TFLOPS per matrix engine (theoretical peak)

### Math Fidelity Impact on Performance
- **LoFi**: ~4 TFLOPS
- **HiFi2**: ~2 TFLOPS
- **HiFi3**: ~1.33 TFLOPS
- **HiFi4**: ~1 TFLOPS

### Performance Scaling Factors
- **Shape dependency**: Square matrices achieve best performance
- **Data format impact**: Significant performance variation between bfloat16, bfloat8_b, bfloat4_b
- **Memory hierarchy**: SRAM vs DRAM access patterns significantly affect achieved performance
- **Utilization**: Peak utilization up to 83% observed in benchmarks for HiFi4 bfloat16

## TT-Metal Programming Model

### Execution Model
TT-Metal executes one **program** at a time on a 2D grid of Tensix cores. A program consists of:

**Kernel Types:**
- **Reader kernel**: Targets RISC0 for data input
- **Writer kernel**: Targets RISC1 for data output
- **Compute kernel**: Compiled for RISCs 2, 3, 4 for mathematical operations

**Parallelization:** Programs are mapped onto 2D grids of Tensix cores, with each core executing its kernels asynchronously.

### Tile-Based Computing
- **Tile size**: 32×32 matrix elements
- **Tilized tensors**: Last two dimensions shuffled so 32×32 tile elements are contiguous in memory
- **NoC optimization**: Tile reads/writes implemented as large contiguous bursts
- **Compute engine**: Natively operates on 32×32 tiles

### Data Layout Considerations
- **Sliding window operations**: Run in channels-last ordering (NHWC vs PyTorch's NCHW)
- **Required transformations**: Most models need CHW→HWC transformation at input and HWC→CHW at output

## Advanced Performance Optimizations

### Metal Trace
- **Purpose**: Removes host overhead by recording operation dispatch commands in DRAM buffer
- **Benefit**: Eliminates gaps between operations for compute-bound workloads
- **Requirements**: Requires `trace_region_size` parameter during device creation
- **APIs**: `ttnn.begin_trace_capture()`, `ttnn.end_trace_capture()`, `ttnn.execute_trace()`

### Multiple Command Queues
- **Capacity**: Up to 2 independent command queues per device
- **Use case**: Parallel dispatch of commands (e.g., separate I/O and compute queues)
- **Synchronization**: Event-based coordination between queues

### Sharding Strategies
**Available types:**
- **Height sharding**: Input matrix height divided equally across cores
- **Width sharding**: Input matrix width divided equally across cores
- **Block sharding**: Both height and width divided across 2D core grid

## Data Format Support

### Supported Formats
- **FLOAT32, BFLOAT16, BFLOAT8_B, BFLOAT4_B, UINT8**
- **Block-float formats**: 8-bit, 4-bit, 2-bit with shared exponents
- **Runtime reconfiguration**: Hardware can switch between formats using `reconfig_data_format()` API

### Special Value Handling
- **NaN/Infinity detection**: Available through status flags
- **Denormal handling**: Flushed to zero
- **Detection granularity**: Per-FPU/SFPU lane with sticky status bits

## Multi-Chip Scaling Architecture

### Collective Communication Library (CCL)
- **Supported topologies**: Ring and line configurations
- **Available operations**: All-gather, reduce-scatter (all-reduce in development)
- **Bandwidth efficiency**: Optimized for 4KB-16KB packet sizes

### ERISC Data Mover (EDM)
- **Purpose**: Multi-channel, bidirectional data movement over Ethernet links
- **Flow control**: End-to-end acknowledgment system
- **Buffering**: Support for double-buffering and multiple virtual channels

## Performance Benchmarks and Examples

### FlashAttention Results
- **Speedup**: 20x average over baseline (range: 9x-44x)
- **Architecture utilization**: Leverages L1 memory effectively for large sequence lengths
- **Implementation features**: Causality-aware load balancing, pipelined execution

### Matrix Multiplication Performance
- **Peak observed**: 86.31 TFLOPS (bfloat16, HiFi2, 16K×16K×16K with trace)
- **Utilization**: Up to 83% of theoretical peak for HiFi4
- **Memory bound regions**: Performance significantly lower when exceeding L1 capacity

## Missing Information Requiring Further Investigation

### Unspecified Architecture Details
1. **Exact NoC bandwidth specifications** (internal chip interconnect speeds)
2. **PCIe generation and lane configuration** for different products
3. **Detailed L1 memory bank organization** and access patterns
4. **Precise RISC-V core specifications** (instruction sets, clock frequencies)
5. **Memory controller specifications** and access patterns
6. **Detailed power consumption** breakdown by component
7. **Thermal characteristics** and throttling behavior

### Performance Specifications Needing Clarification
1. **Maximum achievable NoC bandwidth** in practice
2. **L1-to-L1 transfer speeds** within and between cores
3. **DRAM access latency** characteristics
4. **Context switching overhead** between kernels
5. **Ethernet link establishment latency**
6. **Program compilation and dispatch overhead**

### Programming Model Gaps
1. **Maximum grid sizes** supported for different operation types
2. **Memory allocation strategies** and fragmentation handling
3. **Debugging and profiling tool specifications**
4. **Error handling and recovery mechanisms**
5. **Resource scheduling** and multi-tenancy support
6. **Kernel memory footprint** optimization guidelines

This architecture represents a significant departure from traditional GPU programming models, with its emphasis on explicit memory management, tile-based computing, and distributed execution across a mesh of specialized cores.
