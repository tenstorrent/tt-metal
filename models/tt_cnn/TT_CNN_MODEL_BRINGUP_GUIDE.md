# TT-CNN Model Bringup Guide

This guide provides a comprehensive walkthrough for bringing up CNN models using the TT-CNN library, using the Vanilla UNet implementation as a practical example.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture Patterns](#architecture-patterns)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Configuration Management](#configuration-management)
6. [Performance Optimization](#performance-optimization)
7. [Testing and Validation](#testing-and-validation)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

## Overview

The TT-CNN library provides three main components for CNN model development:

- **Builder**: Configuration-driven layer composition (`TtConv2d`, `TtMaxPool2d`, `TtUpsample`)
- **Pipeline**: High-performance execution framework with tracing and multi-CQ support
- **Executor**: Various execution strategies for optimal throughput and latency

### Key Benefits

- **Simplified Development**: Configuration-based approach reduces boilerplate
- **Performance Optimization**: Built-in tracing, multi-command queues, and memory management
- **Flexible Sharding**: Support for height, width, block, and auto-sharding strategies
- **Memory Efficiency**: Automatic tensor slicing and memory configuration

## Prerequisites

- TT-NN environment setup
- Understanding of CNN architectures
- Basic knowledge of tensor sharding concepts
- Familiarity with PyTorch (for reference models)

## Architecture Patterns

### 1. Model Structure Organization

Organize your model implementation into these key files:

```
models/demos/your_model/
├── tt/
│   ├── model.py          # TT-NN model implementation
│   ├── config.py         # Configuration management
│   └── common.py         # Utilities and preprocessing
├── reference/
│   └── model.py          # PyTorch reference implementation
├── tests/
│   ├── test_model.py     # Accuracy tests
│   └── test_perf.py      # Performance benchmarks
└── demo/
    └── demo.py           # End-to-end demo
```

### 2. Layer Hierarchy

Structure complex models using modular components:

```python
# Example from vanilla_unet/tt/model.py
class TtUNetEncoder:
    def __init__(self, conv1, conv2, pool, device):
        self.conv1 = TtConv2d(conv1, device)
        self.conv2 = TtConv2d(conv2, device)
        self.pool = TtMaxPool2d(pool, device)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.pool(x)
        return x, skip
```

## Step-by-Step Implementation

### Step 1: Define Configuration Classes

Create configuration dataclasses for your model:

```python
# config.py
@dataclass
class ModelLayerConfigs:
    l1_input_memory_config: ttnn.MemoryConfig

    # Define configurations for each layer
    conv1: Conv2dConfiguration
    conv2: Conv2dConfiguration
    pool1: MaxPool2dConfiguration
    # ... additional layers

class ModelConfigBuilder:
    def __init__(self, parameters, input_height, input_width, batch_size):
        self.parameters = parameters
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size

    def build_configs(self) -> ModelLayerConfigs:
        return ModelLayerConfigs(
            l1_input_memory_config=self._create_input_memory_config(),
            conv1=self._create_conv_config_from_params(...),
            # ... configure all layers
        )
```

### Step 2: Implement Model Class

Create your main model class:

```python
# model.py
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d

class TtModel:
    def __init__(self, configs: ModelLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        # Initialize layers using TT-CNN builder components
        self.conv1 = TtConv2d(configs.conv1, device)
        self.conv2 = TtConv2d(configs.conv2, device)
        self.pool1 = TtMaxPool2d(configs.pool1, device)

    def preprocess_input_tensor(self, x: ttnn.Tensor, deallocate_input_activation: bool = True):
        # Convert input format as needed (e.g., CHW to HWC)
        output = ttnn.experimental.convert_to_hwc(x)
        if deallocate_input_activation:
            ttnn.deallocate(x)
        return output

    def __call__(self, input_tensor: ttnn.Tensor, deallocate_input_activation: bool = True) -> ttnn.Tensor:
        input_tensor = self.preprocess_input_tensor(input_tensor, deallocate_input_activation)

        # Forward pass
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        x = self.pool1(x)

        # Convert output format as needed
        return ttnn.experimental.convert_to_chw(x, dtype=ttnn.bfloat16)

def create_model_from_configs(configs: ModelLayerConfigs, device: ttnn.Device) -> TtModel:
    return TtModel(configs, device)
```

### Step 3: Create Weight Preprocessing

Implement parameter preprocessing to convert PyTorch weights to TT-NN format:

```python
# common.py
def create_model_preprocessor(device, mesh_mapper=None):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if isinstance(model, YourPyTorchModel):
            # Convert conv layers with batch norm folding
            from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.conv1, model.bn1
            )
            parameters["conv1"] = {
                "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                "bias": ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                    mesh_mapper=mesh_mapper
                )
            }
            # ... process remaining layers

        return parameters

    return custom_preprocessor
```

### Step 4: Integration with Pipeline API (Optional)

For high-performance inference, integrate with the Pipeline API:

```python
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Configure pipeline for optimal performance
config = PipelineConfig(
    use_trace=True,
    num_command_queues=2,
    all_transfers_on_separate_command_queue=False
)

# Create pipeline
pipeline = create_pipeline_from_config(
    config=config,
    model=your_model_callable,
    device=device,
    dram_input_memory_config=dram_memory_config,
    l1_input_memory_config=l1_memory_config
)

# Compile and run
pipeline.compile(sample_input)
outputs = pipeline.enqueue(inputs).pop_all()
pipeline.cleanup()
```

## Configuration Management

### Sharding Strategy Selection

Choose appropriate sharding strategies based on your model's characteristics:

```python
# For large spatial dimensions - use HeightShardedStrategy
sharding_strategy = HeightShardedStrategyConfiguration(
    act_block_h_override=64,  # Adjust based on tensor dimensions
    reshard_if_not_optimal=False
)

# For models with many channels - use WidthShardedStrategy
sharding_strategy = WidthShardedStrategyConfiguration(
    act_block_w_div=1,
    reshard_if_not_optimal=False
)

# For optimal performance - use AutoShardedStrategy
sharding_strategy = AutoShardedStrategyConfiguration()
```

### Memory Configuration

Set up appropriate memory configurations:

```python
# DRAM memory config for persistent tensors
dram_memory_config = get_memory_config_for_persistent_dram_tensor(
    shape=input_shape,
    shard_strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    dram_grid_size=device.dram_grid_size()
)

# L1 memory config for working tensors
l1_shard_spec = ttnn.ShardSpec(
    core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR
)
l1_memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, l1_shard_spec
)
```

## Performance Optimization

### 1. Data Type Selection

Use appropriate data types for optimal performance:

```python
conv_config = Conv2dConfiguration(
    # ... other parameters
    activation_dtype=ttnn.bfloat8_b,  # Lower precision for activations
    weights_dtype=ttnn.bfloat8_b,     # Lower precision for weights
    output_dtype=ttnn.bfloat16,       # Higher precision for output
)
```

### 2. Memory Management

Implement proper memory management:

```python
# Enable double buffering for better throughput
conv_config = Conv2dConfiguration(
    # ... other parameters
    enable_act_double_buffer=True,
    enable_weights_double_buffer=True,
    deallocate_activation=True,  # Deallocate immediately after use
)
```

### 3. Compute Configuration

Optimize compute settings:

```python
conv_config = Conv2dConfiguration(
    # ... other parameters
    math_fidelity=ttnn.MathFidelity.LoFi,  # Use LoFi for better performance
    fp32_dest_acc_en=True,                 # Enable for better accuracy when needed
    packer_l1_acc=True,                    # Enable L1 accumulation
)
```

## Testing and Validation

### 1. Accuracy Testing

Create comprehensive accuracy tests:

```python
# test_model.py
def test_model_accuracy(device, reset_seeds, model_location_generator):
    # Load reference model
    reference_model = load_reference_model()

    # Create TT-NN model
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        custom_preprocessor=create_model_preprocessor(device),
        device=None,
    )
    configs = create_model_configs_from_parameters(parameters)
    tt_model = create_model_from_configs(configs, device)

    # Compare outputs
    torch_output = reference_model(torch_input)
    tt_output = tt_model(ttnn_input)

    assert_with_pcc(torch_output, ttnn.to_torch(tt_output), pcc=0.97)
```

### 2. Performance Testing

Implement performance benchmarks:

```python
# test_perf.py
def test_model_performance(device):
    # ... setup model

    # Measure end-to-end performance
    start_time = time.time()
    for _ in range(num_iterations):
        output = model(input_tensor)
        ttnn.synchronize_device(device)
    end_time = time.time()

    throughput = num_iterations / (end_time - start_time)
    logger.info(f"Throughput: {throughput:.2f} inferences/second")
```

## Common Patterns

### 1. Skip Connections (UNet Pattern)

Implement skip connections with proper memory management:

```python
def concatenate_skip_connection(upsampled: ttnn.Tensor, skip: ttnn.Tensor) -> ttnn.Tensor:
    # Reshard skip connection to match upsampled tensor's memory config
    if not skip.is_sharded():
        input_memory_config = upsampled.memory_config()
        skip = ttnn.to_memory_config(skip, input_memory_config)

    # Concatenate with appropriate output memory config
    output_memory_config = # ... calculate based on input shapes
    concatenated = ttnn.concat([upsampled, skip], dim=3, memory_config=output_memory_config)

    return concatenated
```

### 2. Transpose Convolution

Handle transpose convolutions for upsampling:

```python
def transpose_conv2d(input_tensor: ttnn.Tensor, upconv_config: UpconvConfiguration) -> ttnn.Tensor:
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
    )

    output, [weight, bias] = ttnn.conv_transpose2d(
        input_tensor=input_tensor,
        weight_tensor=upconv_config.weight,
        bias_tensor=upconv_config.bias,
        # ... other parameters
        conv_config=conv_config,
        return_weights_and_bias=True,
    )
    return output
```

### 3. Channel Slicing

Implement channel slicing for large models:

```python
# In your configuration
slice_strategy = ChannelSliceStrategyConfiguration(num_slices=4)

conv_config = Conv2dConfiguration(
    # ... other parameters
    slice_strategy=slice_strategy
)

# The TtConv2d layer will automatically handle slicing
layer = TtConv2d(conv_config, device)
output = layer(input)  # Automatically sliced execution
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Allocation Errors**
   - Reduce `act_block_h_override` values
   - Use channel slicing for large models
   - Verify shard dimensions are compatible with tensor shapes

2. **Shape Mismatch Errors**
   - Ensure input preprocessing matches expected format
   - Check that skip connection tensors have matching shapes
   - Validate memory config compatibility between operations

3. **Performance Issues**
   - Enable tracing for repeated inference
   - Use appropriate sharding strategies
   - Consider pipeline API for high-throughput scenarios

4. **Accuracy Issues**
   - Verify weight preprocessing is correct
   - Check data type configurations
   - Ensure batch norm folding is applied correctly

### Debug Utilities

```python
# Add debugging information
logger.info(f"Tensor shape: {tensor.shape}")
logger.info(f"Memory config: {tensor.memory_config()}")
logger.info(f"Storage type: {tensor.storage_type()}")

# Validate tensor properties
assert tensor.is_sharded(), "Expected sharded tensor"
assert tensor.memory_config().buffer_type == ttnn.BufferType.L1, "Expected L1 tensor"
```

## Example: Complete Vanilla UNet Walkthrough

The Vanilla UNet implementation (`models/demos/vanilla_unet/`) demonstrates all these patterns:

### Key Files Reference

- `tt/model.py:112-185`: Main UNet model implementation using TT-CNN components
- `tt/config.py:75-452`: Configuration builder with comprehensive layer setup
- `tt/common.py:46-167`: Weight preprocessing with batch norm folding
- `demo/demo.py:97-162`: End-to-end demo showing complete inference pipeline
- `tests/test_unet_model.py:26-61`: Accuracy validation against PyTorch reference

### Usage Pattern

```python
# 1. Load and preprocess PyTorch model
reference_model = load_reference_model()
parameters = preprocess_model_parameters(
    initialize_model=lambda: reference_model,
    custom_preprocessor=create_unet_preprocessor(device),
)

# 2. Create configurations
configs = create_unet_configs_from_parameters(
    parameters=parameters,
    input_height=480,
    input_width=640,
    batch_size=1,
)

# 3. Initialize TT-NN model
tt_model = create_unet_from_configs(configs, device)

# 4. Run inference
ttnn_input = prepare_ttnn_input(input_data, configs.l1_input_memory_config)
output = tt_model(ttnn_input)
```

This pattern provides a solid foundation for implementing any CNN architecture using the TT-CNN library.
