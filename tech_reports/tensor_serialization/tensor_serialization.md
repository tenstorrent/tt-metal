# Tensor Serialization

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Key APIs](#2-key-apis)
  - [2.1 ttnn.dump_tensor](#21-ttnndump_tensor)
  - [2.2 ttnn.load_tensor](#22-ttnnload_tensor)
  - [2.3 ttnn.as_tensor](#23-ttnnas_tensor)
- [3. File Format](#3-file-format)
  - [3.1 FlatBuffer Schema](#31-flatbuffer-schema)
  - [3.2 File Layout](#32-file-layout)
- [4. Multi-Host Support](#4-multi-host-support)
- [5. Best Practices](#5-best-practices)
- [6. Understanding Cache Hits and Misses](#6-understanding-cache-hits-and-misses)
  - [6.1 Common Reasons for Cache Misses](#61-common-reasons-for-cache-misses)
- [7. Examples](#7-examples)
  - [7.1 Basic Save and Load](#71-basic-save-and-load)
  - [7.2 Using `ttnn.as_tensor` with Caching](#72-using-ttnnas_tensor-with-caching)

## 1. Introduction

TT-NN provides a robust tensor serialization mechanism that allows users to save and load tensors to/from disk. The serialization format uses FlatBuffers for metadata and supports single-device / multi-device / multi-host environments. All serialized tensor files use the `.tensorbin` extension.

The key features include:
- Efficient binary format with FlatBuffer metadata
- Support for all TT-NN data types and layouts
- Multi-host/multi-device tensor support
- Memory-mapped file loading for efficient access
- Guaranteed 8-byte alignment for data regions

## 2. Key APIs

The tensor serialization APIs are defined in `ttnn/ttnn/operations/core.py`. Here's a brief overview with examples:

### 2.1 `ttnn.dump_tensor`

Saves a tensor to disk in the TT-NN binary format. Files must use the `.tensorbin` extension.

### 2.2 `ttnn.load_tensor`

Loads a tensor from disk, optionally placing it directly on a device.

### 2.3 `ttnn.as_tensor`

Converts a PyTorch tensor to a ttnn.Tensor with optional caching. When `cache_file_name` is provided, the actual file name is generated as: `{cache_file_name}_dtype_{dtype}_layout_{layout}.tensorbin`

**Basic example:**
```python
import ttnn
import torch

# Save a tensor
tensor = ttnn.from_torch(torch.randn(64, 128), dtype=ttnn.bfloat16)
ttnn.dump_tensor("weights.tensorbin", tensor)

# Load it back to host
loaded = ttnn.load_tensor("weights.tensorbin")

# Or load directly to device
device = ttnn.open_device(0)
loaded_on_device = ttnn.load_tensor("weights.tensorbin", device=device)

# Use as_tensor for automatic caching
cached_tensor = ttnn.as_tensor(
    torch.randn(1024, 1024),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    cache_file_name="activations",  # Creates activations_dtype_bfloat16_layout_TILE.tensorbin
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG
)
```

## 3. File Format

### 3.1 FlatBuffer Schema

The serialization format uses FlatBuffers to store tensor metadata. The schema is defined in `ttnn/core/tensor/flatbuffer/tensor.fbs` and includes tensor specifications, mesh shapes, and shard information for distributed tensors.

### 3.2 File Layout

The `.tensorbin` file format follows a simple structure:

```
┌─────────────────────────────────────────────────────────────┐
│                     File Header (8 bytes)                   │
│                   uint64_t header_size                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                  FlatBuffer Metadata                        │
│                   (header_size bytes)                       │
│              Aligned to 8 bytes (uint64_t)                  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      Data Region                            │
│                   (Tensor Data Buffers)                     │
│                  Guaranteed 8-byte aligned                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key characteristics:**
- Fixed 8-byte header containing the size of the FlatBuffer metadata
- FlatBuffer region is aligned to 8 bytes for safe memory-mapped access
- Data region immediately follows the metadata and is guaranteed to be 8-byte aligned
- Individual tensor buffers are aligned according to their element size

The 8-byte alignment guarantee enables efficient memory-mapped file loading using `mmap`, allowing the tensor data to be accessed directly from the file without copying to RAM first.

## 4. Multi-Host Support

The tensor serialization APIs work seamlessly in multi-host environments. From a user's perspective, there is no difference between single-host and multi-host usage:

1. **Unified File Format**: A single `.tensorbin` file represents the global tensor data across all hosts
2. **Single File Output**: When saving tensors in a multi-host environment, only one host writes the tensor data to disk, containing the global tensor information
3. **Mesh Device Support**: The `load_tensor` API accepts a `MeshDevice` parameter to load tensors directly onto distributed devices
4. **Transparent Sharding**: The format stores individual shards with their mesh coordinates, enabling proper reconstruction of distributed tensors

The key advantage is that users don't need to handle multi-host complexity - the APIs abstract away the distributed nature and present a simple interface regardless of the deployment configuration.

## 5. Best Practices

### 5.1 Reproducible Random Tensors

When generating tensors with random numbers, use manual seeds for reproducibility and include the seed in the filename:

```python
import torch
import ttnn

# Set manual seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Include seed in filename using "seed_<>" format
tensor = ttnn.from_torch(torch.randn(1024, 1024), dtype=ttnn.bfloat16)
ttnn.dump_tensor(f"random_weights_seed_{seed}.tensorbin", tensor)
```

This allows distinguishing between different random tensors and ensures reproducible model initialization.

### 5.2 Prefer `ttnn.as_tensor` API

The `ttnn.as_tensor` API is generally preferred because it:
- Automatically handles caching
- Allows straightforward regeneration of weights
- Maintains consistency between data type and layout in filenames

If not using `ttnn.as_tensor`, ensure the weight-regeneration flow is clearly documented so that someone unfamiliar with the model code can regenerate the tensors.

### 5.3 Organize Tensor Files

Use descriptive filenames and directory structure to clarify tensor purposes:

```
model_cache/
├── weights/                              # Model parameters (static, rarely change)
│   ├── embeddings/
│   │   ├── word_embeddings.weight_bfloat16.tensorbin
│   │   ├── position_embeddings.weight_bfloat16.tensorbin
│   │   └── token_type_embeddings.weight_bfloat16.tensorbin
│   ├── attention/
│   │   ├── layer_0.qkv.weight_bfloat16.tensorbin
│   │   ├── layer_0.qkv.bias_bfloat16.tensorbin
│   │   ├── layer_0.dense.weight_bfloat16.tensorbin
│   │   └── layer_0.dense.bias_bfloat16.tensorbin
│   ├── ffn/
│   │   ├── layer_0.ff1.weight_bfloat16.tensorbin
│   │   ├── layer_0.ff1.bias_bfloat16.tensorbin
│   │   ├── layer_0.ff2.weight_bfloat16.tensorbin
│   │   └── layer_0.ff2.bias_bfloat16.tensorbin
│   └── layernorm/
│       ├── layer_0.ln_attn.weight_bfloat16.tensorbin
│       └── layer_0.ln_mlp.weight_bfloat16.tensorbin
├── activations/                          # Cached intermediate tensors (may change)
│   ├── cos_cached_bfloat16.tensorbin    # Rotary position embeddings
│   ├── sin_cached_bfloat16.tensorbin
│   └── attention_mask_causal_seed_42.tensorbin
└── preprocessed/                         # Preprocessed inputs (dataset-specific)
    ├── input_ids_batch_32.tensorbin
    └── segment_ids_batch_32.tensorbin
```

This organization helps users understand:
- Which tensors are model weights that remain static across runs
- Which tensors are intermediate activations that may need regeneration
- The data type and purpose of each tensor from its filename
- Layer-specific organization for multi-layer models

## 6. Understanding Cache Hits and Misses

When using `ttnn.as_tensor` with caching or `ttnn.load_tensor`, understanding cache behavior is crucial for debugging and performance optimization.

### 6.1 Common Reasons for Cache Misses

Cache misses occur when the cached tensor file cannot be used, forcing regeneration or recomputation. The most common causes are:

1. **Cache Not Generated**: The cache file doesn't exist yet (first run or after cleanup)
   ```python
   # First call - cache miss, generates activations_dtype_bfloat16_layout_TILE.tensorbin
   tensor = ttnn.as_tensor(data, cache_file_name="activations", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
   ```

2. **Shape Mismatch**: The requested tensor shape differs from the cached tensor
   ```python
   # Cache miss if previous tensor was different shape
   tensor1 = ttnn.as_tensor(torch.randn(1024, 1024), cache_file_name="weights", ...)  # Creates cache
   tensor2 = ttnn.as_tensor(torch.randn(2048, 2048), cache_file_name="weights", ...)  # Cache miss!
   ```

3. **Data Type or Layout Mismatch**: Different dtype or layout parameters create different cache files
   ```python
   # These create different cache files due to naming convention
   tensor_bf16 = ttnn.as_tensor(data, cache_file_name="weights", dtype=ttnn.bfloat16)  # weights_dtype_bfloat16_layout_ROW_MAJOR.tensorbin
   tensor_fp32 = ttnn.as_tensor(data, cache_file_name="weights", dtype=ttnn.float32)   # weights_dtype_float32_layout_ROW_MAJOR.tensorbin
   ```

4. **File Corruption**: The cache file exists but is corrupted or incomplete
   - Partial writes from interrupted processes
   - Disk errors or filesystem issues
   - Manual modification of tensor files

## 7. Examples

### 7.1 Basic Save and Load

```python
import ttnn
import torch

# Create and save a tensor
tensor = ttnn.from_torch(torch.randn(64, 128), dtype=ttnn.bfloat16)
ttnn.dump_tensor("model_weights.tensorbin", tensor)

# Load the tensor back
loaded_tensor = ttnn.load_tensor("model_weights.tensorbin")
```

### 7.2 Using `ttnn.as_tensor` with Caching

```python
import ttnn
import torch

# First call generates and caches the tensor
tensor = ttnn.as_tensor(
    torch.randn(1024, 1024),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    cache_file_name="activations",
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG
)
# Creates: activations_dtype_bfloat16_layout_TILE.tensorbin

# Subsequent calls load from cache
tensor2 = ttnn.as_tensor(
    torch.randn(1024, 1024),  # Input is ignored when cache exists
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    cache_file_name="activations",
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG
)
```

The tensor serialization system provides a robust and efficient way to persist tensors across sessions. The same APIs work identically whether you're using a single device or complex multi-host deployments, making it easy to scale applications without code changes.
