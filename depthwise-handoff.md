# Depthwise Conv2d via Pool2d Approach

## Idea

Depthwise conv2d is a special case of grouped convolution where `groups == input_channels == output_channels`. Each input channel is convolved with its own filter, with no cross-channel interaction.

### Problem with Conventional Approach

In the standard grouped conv2d implementation, weights are padded with zeros to fit the grouped format. This results in many matrix multiplications with zeros, which is inefficient.

### Solution

Implement depthwise conv2d using a pool2d-like approach:
- Reuse the pool2d kernel infrastructure
- Each input channel is processed independently with its corresponding filter
- Element-wise multiplication and reduce replaces the matmul-based convolution

This approach is more efficient, especially for large channel counts.

See issue for details: https://github.com/tenstorrent/tt-metal/issues/25523

---

## What's Done on This Branch

### New Files
- `conv2d_op_depthwise_program_factory.cpp/.hpp` - New program factory that creates pool2d-based programs for depthwise conv2d

### Modified Components

**Conv2d Device Operation Routing**
- `conv2d_device_operation.cpp` - Added `is_2d_depthwise_conv()` check in `select_program_factory()` to route depthwise convolutions to `Conv2dDepthwiseProgramFactory` instead of the regular matmul-based path

**Weight Preparation (`prepare_conv2d_weights.cpp/.hpp`)**
- Added `conv_2d_depthwise_weight_layout_helper<T>()` - New templated function that transforms weights from [out_ch, 1, kH, kW] to a face-based tiled layout suitable for element-wise multiplication in the pool2d kernel
- Layout handles all sharding schemes (HEIGHT, WIDTH, BLOCK) by distributing channels across cores:
  - HEIGHT_SHARDED: all cores use same weights (num_channel_shards=1)
  - WIDTH_SHARDED: each core gets different channels (num_channel_shards=total_cores)
  - BLOCK_SHARDED: each column gets different channels (num_channel_shards=num_cores_c)
- Each tile packs kernel positions into rows (ceil(channels/16) rows per kernel position)

**Conv2d Utils (`conv2d_utils.cpp/.hpp`)**
- Added `is_2d_depthwise_conv()` function to detect depthwise convolutions (groups == in_channels == out_channels) that are truly 2D (not 1D)
- Modified `determine_parallel_config()` to accept `groups` parameter and force TILE_WIDTH (32) channel alignment for depthwise
- Modified `determine_input_memory_config()` and `get_conv_padded_input_shape_and_memory_config()` to use tile-aligned channels for depthwise

**Pool Compute Kernel (`compute_pool_2d.cpp`)**
- Added `IS_DEPTHWISE` compile-time define that enables depthwise convolution mode
- Added new compile-time args for depthwise: `weight_cb_id`, `mul_cb_id`, `has_bias`, `bias_cb_id`, `bias_ntiles`, `clear_value_cb_id`
- Depthwise path performs element-wise multiply (`mul_tiles`) of input windows with weights instead of reduce operations
- Added support for activation functions (RELU6, SILU, GELU) via SFPU
- Added bias addition to output tiles

**Pool Reader Kernel (`reader_pool_2d.cpp`)**
- Added depthwise-specific compile-time args: `weight_cb_id`, `has_bias`, `bias_cb_id`, `bias_ntiles`, `num_shards_c`
- Depthwise path implements sender/receiver multicast pattern for weights:
  - Sender cores read weights from DRAM using TensorAccessor and multicast to receivers
  - Receiver cores wait for weight multicast via semaphores
- Weight tile reading uses 2D iteration over (row, col) to handle the face-based layout
- Bias tiles are similarly read and multicast

**Pool Program Factory (`pool_multi_core_program_factory.cpp`)**
- Added compile-time args to pass depthwise parameters to kernels
- Minor updates to support depthwise buffer allocation

### Tests (`test_conv2d.py`)
- Added `test_groups_vs_pool2` parametrized test covering:
  - All sharding layouts (HEIGHT, WIDTH, BLOCK)
  - Various data types (bfloat16, bfloat8_b, bfloat4_b)
  - Activations (RELU6, SILU, GELU)
  - MobileNetV2 layer configurations
  - YOLOv10x layer configurations
  - EfficientNet-B0 layer configurations (3x3 and 5x5 kernels)

### Demo
- `ttnn_mobilenetv2.py` - Updated to use depthwise conv2d path

---

## Current State

### Nightly Unit Tests
```bash
pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d.py
```
- **3 failed tests** - all due to wide reduction (`num_channels_per_core > 256`)

### Sweep Tests
```bash
pytest tests/ttnn/nightly/unit_tests/operations/conv/test_conv2d_sweeps.py
```
- **6 failed tests:**
  - 1 forge wide reduction - need to support wide reduction (kernels support it, weight prep needs changes)
  - 5 pytorch 5x5 kernel - need to allocate larger `in_cb` for reader/compute synchronization (same as pool: `TILE_HEIGHT * in_c` instead of `k_h * k_w * in_c`)

### MobileNetV2 Demo
- Passes with program cache disabled (investigate why)
- **Performance issues** (~4x slower overall, worst conv ~10x slower than baseline)

### TODO / Next Steps
1. Investigate SFPU activation function overhead (becoming math bound)
2. Investigate bias performance
3. Utilize DST in pool - batch tiles instead of processing 1 tile/channel per core (e.g., do 8 at once to reduce NOC time and init overhead)
4. Implement fused MUL + REDUCE
5. Fix UTs (wide reduction + `in_cb` allocation)
6. Check other models and its perf
