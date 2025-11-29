# Conv2D Weight Preparation in TT-Metal

## Table of Contents
1. [Introduction](#introduction)
2. [Why Weight Preparation is Necessary](#why-weight-preparation-is-necessary)
3. [Weight Preparation Process](#weight-preparation-process)
4. [Weight Layout Transformations](#weight-layout-transformations)
5. [Implementation Details](#implementation-details)
6. [Special Cases](#special-cases)
7. [Performance Considerations](#performance-considerations)

## Introduction

Conv2D weight preparation is a critical preprocessing step in TT-Metal's convolution implementation that transforms standard PyTorch-format weight tensors into hardware-optimized layouts. This transformation is performed once before convolution operations and the prepared weights can be reused across multiple convolution invocations.

## Why Weight Preparation is Necessary

### 1. Hardware Architecture Requirements

TT-Metal's hardware accelerator operates on **tiles** as the fundamental unit of computation. Each tile is:
- **32x32 elements** in size (TILE_HEIGHT x TILE_WIDTH)
- The smallest granularity for matrix operations
- Required to be properly aligned for efficient hardware execution

Standard PyTorch convolution weights in OIHW format (Output channels, Input channels, Height, Width) are not directly compatible with this tiled architecture and must be transformed.

### 2. Memory Layout Optimization

The hardware requires specific memory layouts for efficient data access:
- **Tile Layout**: Data must be organized in tiles rather than row-major format
- **Memory Alignment**: Weights must be aligned to tile boundaries for optimal memory bandwidth utilization
- **Sharding Schemes**: Different sharding strategies (HEIGHT_SHARDED, BLOCK_SHARDED, WIDTH_SHARDED) require different weight layouts

### 3. Computational Efficiency

Conv2D is implemented as a matrix multiplication operation where:
- **Activations** are transformed into a matrix form (im2col-like transformation)
- **Weights** are reshaped and tiled to match the activation matrix layout
- The convolution becomes a highly optimized matrix multiplication

Without proper weight preparation, this transformation would need to happen for every convolution operation, creating significant overhead.

### 4. Data Type Optimization

Weight preparation handles data type conversions:
- Supports multiple data types: BFLOAT16, FLOAT32, BFLOAT8_B, BFLOAT4_B
- Performs quantization/packing for compressed formats
- Ensures weights are in the optimal format for the target hardware

## Weight Preparation Process

The weight preparation follows these main steps:

### Step 1: Initial Shape Analysis
```
Original Weight Shape: [out_channels, in_channels, kernel_h, kernel_w]
```

### Step 2: Special Transformations (if applicable)

#### Grouped Convolution
For grouped convolutions (groups > 1):
- Weights are reorganized with zero-padding between groups
- Each group's weights are isolated to enable parallel processing
- Output shape: `[out_channels, in_channels * groups, kernel_h, kernel_w]`

#### Depthwise Convolution
For depthwise convolutions (groups == in_channels):
- Weights are broadcasted to match activation block height
- Enables efficient depthwise processing in a single pass
- Broadcasting factor: `act_block_h_ntiles * TILE_HEIGHT`

#### Kernel Stride Folding
For strided convolutions:
- Weights can be pre-folded to incorporate stride information
- Reduces runtime computation requirements

### Step 3: Padding and Alignment

Weights are padded to ensure:
- **Channel alignment**: Input/output channels aligned to tile boundaries
- **Core distribution**: Channels divisible by number of processing cores
- **Memory efficiency**: Padding optimized for the specific sharding scheme

```cpp
in_channels_padded = round_up(in_channels, input_cores * input_alignment);
out_channels_padded = round_up(out_channels, output_cores * TILE_WIDTH);
```

### Step 4: Layout Transformation

Based on the convolution configuration, weights are converted to one of several layouts:

#### Interleaved MM Layout
```
Transform: [Co, Ci, Kh, Kw] → [1, 1, Kh*Kw*Ci, Co]
```
Used for standard matrix multiplication-based convolution.

#### Tiled Layout
```
Transform: [Co, Ci, Kh, Kw] → [1, 1, Ci*Kh*Kw, Co] → Tiled
```
Standard tiled layout with configurable block sizes.

#### Special Padding Tiled Layout
For HEIGHT_SHARDED configurations:
- Additional padding between weight blocks
- Optimized for activation reuse patterns
- Block dimensions: `weight_block_h_ntiles x weight_block_w_ntiles`

#### Block Sharded Layout
For BLOCK_SHARDED configurations:
- Weights distributed across multiple cores
- Zero padding inserted between channel shards
- Optimized for parallel processing

### Step 5: Data Type Conversion

Final conversion to target data type:
- Float32 → BFloat16 (common case)
- Float32 → BFloat8_B/BFloat4_B (quantized formats)
- Includes packing for compressed formats

### Step 6: Device Transfer

Prepared weights are moved to device memory with appropriate memory configuration.

## Weight Layout Transformations

### 1. `convert_conv_weight_tensor_to_interleaved_mm_layout`

**Purpose**: Converts weights for interleaved matrix multiplication
**Input Shape**: `[out_channels, in_channels, kernel_h, kernel_w]`
**Output Shape**: `[1, 1, kernel_h * kernel_w * in_channels, out_channels]`
**Use Case**: Standard conv2d operations using matmul

### 2. `convert_conv_weight_tensor_to_tiled_layout`

**Purpose**: Basic tiled layout conversion
**Parameters**:
- `in1_block_h`: Height of weight blocks in tiles
- `in1_block_w`: Width of weight blocks in tiles
**Use Case**: WIDTH_SHARDED convolutions

### 3. `convert_conv_weight_tensor_to_special_padding_tiled_layout`

**Purpose**: Tiled layout with special padding for activation reuse
**Key Features**:
- Padding between weight blocks
- Optimized for HEIGHT_SHARDED operations
- Supports activation reuse optimization
**Use Case**: Memory-efficient convolutions with activation caching

### 4. `convert_conv_weight_tensor_to_tiled_layout_block_sharded`

**Purpose**: Optimized for block-sharded parallel processing
**Parameters**:
- `in_num_channel_shards`: Number of input channel shards
- `out_num_channel_shards`: Number of output channel shards
- `full_inner_dim`: Whether to use full inner dimension
**Use Case**: BLOCK_SHARDED high-performance convolutions

### 5. `convert_conv_weight_tensor_to_grouped_layout`

**Purpose**: Handles grouped convolutions
**Process**:
- Separates weights by groups
- Adds zero padding between groups
- Maintains group isolation for parallel processing
**Output Shape**: `[out_channels, in_channels * num_groups, kernel_h, kernel_w]`

### 6. `convert_conv_weight_tensor_to_depthwise_layout`

**Purpose**: Optimizes depthwise convolutions
**Process**:
- Broadcasts single-channel weights
- Replicates to match activation block height
- Enables efficient depthwise processing
**Replication Factor**: `act_block_h_ntiles * TILE_HEIGHT`

## Implementation Details

### Threading and Parallelization

The weight preparation uses the `WeightLayoutThreader` class for efficient multi-threaded processing:

```cpp
class WeightLayoutThreader {
    // Calculates optimal thread configuration based on hardware
    static ThreadConfig calculate_thread_config(
        uint32_t out_ch, uint32_t in_ch,
        int MIN_WORK_PER_THREAD = 16
    );

    // Parallel execution across channel dimensions
    template <typename Func>
    static void parallel_for_channels(...);
};
```

- Automatically determines thread count based on hardware concurrency
- Balances work across output and input channel dimensions
- Minimizes thread overhead with minimum work thresholds

### Memory Allocation Strategy

Weight preparation minimizes memory allocation through:
- Pre-calculated buffer sizes
- Direct tensor transformations when possible
- Efficient host buffer management

### Configuration Parameters

The `Conv2dWeightsBiasPrepConfig` structure encapsulates all preparation parameters:

```cpp
struct Conv2dWeightsBiasPrepConfig {
    uint32_t input_channels_alignment;      // Channel alignment requirement
    std::optional<DataType> weights_bias_dtype;  // Target data type
    uint32_t weight_block_h_ntiles;         // Weight block height in tiles
    uint32_t weight_block_w_ntiles;         // Weight block width in tiles
    ParallelConfig input_parallel_config;   // Input sharding configuration
    ParallelConfig output_parallel_config;  // Output sharding configuration
    uint32_t groups;                         // Number of conv groups
    bool enable_kernel_stride_folding;      // Stride folding optimization
    bool full_inner_dim;                    // Full inner dimension flag
    bool enable_activation_reuse;           // Activation reuse optimization
    // ... additional parameters
};
```

## Special Cases

### 1D Convolution
- Treated as 2D convolution with height = 1
- Special handling for depthwise 1D convolutions
- Optimized weight broadcasting for 1D depthwise case

### Dilated Convolution
- Dilation handled during activation transformation
- Weights remain in standard format
- No special weight preparation required

### Transposed Convolution
- Uses separate `prepare_conv_transpose2d_weights` function
- Different weight layout requirements
- Handles fractional stride through weight manipulation

## Performance Considerations

### 1. One-Time Preparation
- Weight preparation is performed once per model
- Prepared weights are cached and reused
- Amortizes preparation cost over multiple inferences

### 2. Memory vs Compute Tradeoff
- Padding increases memory usage but improves compute efficiency
- Block sizes chosen to maximize hardware utilization
- Balance between memory bandwidth and computational throughput

### 3. Sharding Strategy Impact

| Sharding Type | Weight Layout | Best For |
|--------------|---------------|----------|
| HEIGHT_SHARDED | Special padding layout | Large activation maps |
| BLOCK_SHARDED | Block distributed | Balanced workloads |
| WIDTH_SHARDED | Standard tiled | Channel-heavy operations |
| Interleaved MM | Flattened matrix | Small kernels |

### 4. Data Type Selection
- BFloat16: Standard precision-performance balance
- BFloat8_B: Higher performance, lower precision
- BFloat4_B: Maximum performance, minimum precision
- Float32: Maximum precision, lower performance

### 5. Optimization Guidelines

1. **Group/Depthwise Detection**: Automatically optimizes for special convolution types
2. **Alignment Requirements**: Ensures all dimensions meet hardware constraints
3. **Parallel Processing**: Multi-threaded preparation for large weight tensors
4. **Memory Locality**: Layout chosen to maximize cache efficiency during execution

## Example Usage

```python
# Python API example
import ttnn

# Original PyTorch weights
weight_tensor = torch.randn(out_channels, in_channels, kernel_h, kernel_w)

# Prepare weights for conv2d
prepared_weights = ttnn.prepare_conv_weights(
    weight_tensor=weight_tensor,
    input_memory_config=input_memory_config,
    input_layout=ttnn.TILE_LAYOUT,
    weights_format="OIHW",
    in_channels=in_channels,
    out_channels=out_channels,
    batch_size=batch_size,
    input_height=input_height,
    input_width=input_width,
    kernel_size=(kernel_h, kernel_w),
    stride=(stride_h, stride_w),
    padding=(pad_h, pad_w),
    dilation=(1, 1),
    has_bias=False,
    groups=groups,
    input_dtype=ttnn.bfloat16,
    conv_config=conv_config
)

# Reuse prepared weights for multiple convolutions
output = ttnn.conv2d(
    input_tensor=input_tensor,
    weight_tensor=prepared_weights,  # Use prepared weights
    ...
)
```

## Conclusion

Weight preparation is an essential optimization in TT-Metal's conv2d implementation that:
1. **Transforms** weights from standard formats to hardware-optimized layouts
2. **Enables** efficient tiled matrix multiplication operations
3. **Optimizes** memory access patterns and computational efficiency
4. **Supports** various convolution types through specialized transformations
5. **Amortizes** preparation cost through weight reuse

The preparation process is highly configurable and automatically adapts to different convolution parameters, hardware configurations, and performance requirements, making it a crucial component for achieving high-performance convolution operations on TT-Metal hardware.
