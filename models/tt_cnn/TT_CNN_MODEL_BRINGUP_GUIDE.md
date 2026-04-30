# TT-CNN - Vision Model Bring-up Guide

This document demonstrates how to bring-up new vision models using the TT-CNN library.

The TT-CNN library comprises several components designed to assist the development of vision models on Tenstorrent devices. At a high level, the main TT-CNN modules include:

- **Pipeline**: Provides abstractions for high-performance end-to-end model inference, with out-of-the-box support for tracing, and multi-command queue pipelining.
- **Builder**: Provides configuration-driven API for defining and instantiating convolution and pooling layers, making it easy to compose models from reusable configuration classes.

Each module is documented and illustrated with working examples. To dive deeper into:
- **Pipeline API, memory layouts, and performance tuning**, see the [README.md](README.md) "Pipeline" and "Performance Optimization Techniques" sections.
- **Builder API, configuration patterns, and layer composition**, refer to the [README.md](README.md) "Builder" section.

Throughout this document we will refer to the Vanilla UNet implementation as a concrete example of how to use the available TT-CNN modules to create a model. Take a look at the implementation:
- [UNet README](../demos/vanilla_unet/README.md)
- [Reference PyTorch Model](../demos/vanilla_unet/reference/model.py)
- [TT-NN Implementation](../demos/vanilla_unet/tt/model.py)
- [Configuration Builder](../demos/vanilla_unet/tt/config.py)
- [Weight Preprocessing](../demos/vanilla_unet/tt/common.py)

For further details, practical tips, and reference patterns, please explore the in-depth documentation in [README.md](README.md).

## Table of Contents

- [Overview](#overview)
- [1. Model Structure Organization](#1-model-structure-organization)
- [2. Reference Model (PyTorch)](#2-reference-model-pytorch)
- [3. Weight Preprocessing](#3-weight-preprocessing)
  - [Understanding the Preprocessing Flow](#understanding-the-preprocessing-flow)
  - [Using preprocess_model_parameters](#using-preprocess_model_parameters)
- [4. Building Layer Configurations](#4-building-layer-configurations)
  - [Supported Operations](#supported-operations)
  - [Choosing Sharding Strategies](#choosing-sharding-strategies)
- [5. Creating the TT-NN Model](#5-creating-the-tt-nn-model)
  - [Handling Skip Connections and Concatenation](#handling-skip-connections-and-concatenation)
- [6. Put it all Together](#6-put-it-all-together)
- [7. Applying Tracing and Multi-CQ Pipeline](#7-applying-tracing-and-multi-cq-pipeline)
  - [Understanding Pipeline Optimizations](#understanding-pipeline-optimizations)
    - [What Tracing Does](#what-tracing-does)
    - [Why 2 Command Queues Improve Performance](#why-2-command-queues-improve-performance)
    - [When to Use all_transfers_on_separate_command_queue](#when-to-use-all_transfers_on_separate_command_queue)
  - [Implementation Example](#implementation-example)
- [8. Performance Analysis and Optimization](#8-performance-analysis-and-optimization)
  - [Understanding Device vs End-to-End Performance](#understanding-device-vs-end-to-end-performance)
  - [Identifying Performance Bottlenecks](#identifying-performance-bottlenecks)
  - [Performance Optimization Techniques](#performance-optimization-techniques)
    - [Use Sharding Everywhere](#use-sharding-everywhere)
    - [Sliding Window Configuration](#sliding-window-configuration)
    - [Data Types](#data-types)
    - [Multi-Device Support](#multi-device-support)
- [Testing and Validation](#testing-and-validation)
  - [Layer-by-Layer Correctness Testing](#layer-by-layer-correctness-testing)
  - [Device Performance Testing](#device-performance-testing)
- [Troubleshooting](#troubleshooting)
  - [Common Issues and Solutions](#common-issues-and-solutions)

## Overview

The general flow for bringing up a vision model with TT-CNN is:

1. **Reference Model**: Define your model reference using standard PyTorch modules (`nn.Module`) or use an available open source model. This reference implementation will serve as your correctness/accuracy benchmark to ensure that the TT-NN version of this model is correct.

2. **Parameter Preprocessing**: Extract, convert, and optionally fuse your PyTorch weights (e.g., batch norm folding). Also extract other layer parameters (kernel size, stride, etc.) from the reference model.

3. **Layer Configurations**: Using the parameters from the reference model, generate the TT-NN model's configurations for each layer, which contain the preprocessed weights and TT-specific layer attributes.

4. **TT-NN Model Construction**: Create the TT-NN model instance from the model configuration.

5. **Pipeline Optimization**: Wrap your TT-NN model in the Pipeline API to ensure good end-to-end performance

The flow can be visualized as:

```
PyTorch Model
    ↓
Weight Preprocessing
    ↓
Layer Configurations
    ↓
TT-NN Model
    ↓
Pipeline Optimization (compile, enqueue, run)
```

Let's examine each step using the Vanilla UNet as our example.

## 1. Model Structure Organization

Every TT-CNN model follows roughly this standard structure:

```
models/demos/your_model/
├── reference/
│   └── model.py          # Original PyTorch model
├── tt/
│   ├── common.py         # Weight preprocessing & utilities
│   ├── config.py         # Layer configuration builder
│   └── model.py          # TT-NN model implementation
├── tests/
│   ├── test_model.py     # Correctness validation
│   └── test_perf.py      # Performance benchmarks for device and end-to-end performance
│   └── test_*.py         # Additional unit tests for testing correctness at a more granular level - useful for complex models!
│                         # Consider using TT-CNN test utilities for layer verification
└── demo/
    └── demo.py           # End-to-end demonstration
└── README.md
```

## 2. Reference Model (PyTorch)

We start with a standard PyTorch model. Here's the Vanilla UNet structure:

```python
# reference/model.py
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        # Encoder blocks with Conv+BN+ReLU
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ... more encoder/decoder blocks abbreviated ...
        # See full implementation: ../demos/vanilla_unet/reference/model.py
```

## 3. Weight Preprocessing

After defining your PyTorch reference model, you need to extract and convert its weights to TT-NN format. The `preprocess_model_parameters` function handles this conversion process, walking through your model and transforming each layer's weights appropriately.

For simple models with standard layers (Linear, LayerNorm, etc.), TTNN's default preprocessing may be sufficient. However, for complex architectures like UNet that use batch normalization or custom layer arrangements, you'll need to provide a custom preprocessor function.

Here's how the Vanilla UNet implements its custom preprocessing:

```python
# tt/common.py
def create_unet_preprocessor(device, mesh_mapper=None):
    """
    Creates a custom preprocessor function for the UNet model.

    The preprocessor is called by preprocess_model_parameters for each module in the model
    hierarchy. It allows you to intercept specific layers and apply custom weight transformations.

    Args:
        device: TT device for tensor placement
        mesh_mapper: Optional mapper for multi-device distribution

    Returns:
        custom_preprocessor function that processes model layers
    """
    def custom_preprocessor(model, name, ttnn_module_args):
        """
        Custom preprocessor that handles UNet-specific layer transformations.

        This function is called recursively for each module in the PyTorch model:
        - model: The current PyTorch module being processed
        - name: Hierarchical name (e.g., "encoder1.0" for first conv in encoder1)
        - ttnn_module_args: Additional args inferred from model (optional)

        Returns:
            Dict of preprocessed parameters, or None to use default preprocessing
        """
        parameters = {}

        # Only process the top-level UNet model
        if isinstance(model, UNet):
            # Process each encoder block
            for i in range(1, 5):
                parameters[f"encoder{i}"] = {}

                # Fold BatchNorm into Conv weights
                from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

                # First conv in block - fold conv + batchnorm
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                    getattr(model, f"encoder{i}")[0],  # Conv layer
                    getattr(model, f"encoder{i}")[1]   # BatchNorm layer
                )
                parameters[f"encoder{i}"][0] = {
                    "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper
                    )
                }

                # Second conv in block
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                    getattr(model, f"encoder{i}")[3],
                    getattr(model, f"encoder{i}")[4]
                )
                parameters[f"encoder{i}"][1] = {
                    "weight": ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
                    "bias": ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper
                    )
                }

            # Process decoder blocks similarly (bottleneck, decoder4-1)
            # Full preprocessing: ../demos/vanilla_unet/tt/common.py:82-156

            # Process upconv layers (no batch norm to fold)
            for i in range(4, 0, -1):
                parameters[f"upconv{i}"] = {
                    "weight": ttnn.from_torch(
                        getattr(model, f"upconv{i}").weight,
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper
                    ),
                    "bias": ttnn.from_torch(
                        torch.reshape(getattr(model, f"upconv{i}").bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                        mesh_mapper=mesh_mapper
                    )
                }

        return parameters

    return custom_preprocessor
```

### Understanding the Preprocessing Flow

The weight preprocessing involves two key components:

1. **`preprocess_model_parameters`** - A TTNN function that orchestrates the entire preprocessing:
   - Takes your PyTorch model and recursively walks through all modules
   - For each module, checks if a custom preprocessor should handle it
   - If no custom handling, uses default preprocessing (e.g., for Linear, LayerNorm)
   - Returns a nested dictionary structure matching your model hierarchy

2. **`custom_preprocessor`** - Your custom function for handling specific layers:
   - Called for EVERY module in the model (not just the top level)
   - Return a dict to override default preprocessing for that module
   - Return None/empty dict to use TTNN's default preprocessing
   - Commonly used for:
     - Folding batch normalization into convolutions
     - Custom weight transformations
     - Handling non-standard layer types

### Using preprocess_model_parameters

Here's how to use the preprocessing in practice:

```python
from ttnn.model_preprocessing import preprocess_model_parameters

# Load your PyTorch model
reference_model = load_reference_model()  # Returns PyTorch UNet model

# Extract and convert weights to TT-NN format
parameters = preprocess_model_parameters(
    initialize_model=lambda: reference_model,
    custom_preprocessor=create_unet_preprocessor(device),
    device=None,  # Weights are not placed on device yet
)

# The returned 'parameters' is a nested dictionary matching your model structure:
# parameters["encoder1"][0]["weight"]  # First conv weight in encoder1
# parameters["encoder1"][0]["bias"]    # First conv bias in encoder1
# parameters["encoder1"][1]["weight"]  # Second conv weight in encoder1
# ... and so on for all layers
```

Key points about weight preprocessing:
- **Batch Norm Folding**: Conv+BN pairs are fused into single Conv operations
- **Tensor Conversion**: PyTorch tensors → TT-NN tensors with appropriate dtype
- **Bias Reshaping**: Biases must be reshaped to (1, 1, 1, channels) for TT-NN's expected format
  - PyTorch bias shape: (out_channels,)
  - TT-NN bias shape: (1, 1, 1, out_channels)
- **Mesh Mapper**: For multi-device data parallelism, weights can be replicated across devices

## 4. Building Layer Configurations

The configuration builder takes preprocessed weights and creates detailed configs for each layer:

```python
# tt/config.py
class TtUNetConfigBuilder:
    def __init__(self, parameters: Dict, input_height: int, input_width: int, batch_size: int):
        self.parameters = parameters
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size

    def build_configs(self) -> TtUNetLayerConfigs:
        return TtUNetLayerConfigs(
            # Input memory configuration
            l1_input_memory_config=self._create_input_memory_config(),

            # Encoder 1 configuration
            encoder1_conv1=self._create_conv_config_from_params(
                self.input_height,
                self.input_width,
                self.in_channels,
                self.features,
                self.parameters["encoder1"][0],  # Preprocessed weights
                sharding_strategy=HeightShardedStrategyConfiguration(act_block_h_override=15 * 32),
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
            ),
            # ... encoder2-4, bottleneck, decoder, upconv configs ...
            # Full configuration: ../demos/vanilla_unet/tt/config.py:104-347
        )

    def _create_conv_config_from_params(
        self,
        input_height: int,
        input_width: int,
        in_channels: int,
        out_channels: int,
        parameters: Dict,  # Contains preprocessed weight and bias
        sharding_strategy: ShardingStrategy,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        **kwargs
    ) -> Conv2dConfiguration:
        """Create Conv2d config with preprocessed weights"""
        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=self.batch_size,
            weight=parameters["weight"],  # TT-NN tensor from preprocessing
            bias=parameters["bias"],      # TT-NN tensor from preprocessing
            sharding_strategy=sharding_strategy,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
            **kwargs
        )
```

### Supported Operations

The builder API currently provides high-level APIs for:
- **Conv2d**: Standard 2D convolution with various configurations
- **MaxPool2d**: Max pooling operations
- **Upsample**: Upsampling operations (bilinear and nearest interpolation)

For operations not yet in TT-CNN, use the TT-NN API directly:
- Transpose convolution: `ttnn.conv_transpose2d` (will be added to TT-CNN in the future)
- Normalization: `ttnn.layer_norm`, `ttnn.group_norm`, `ttnn.batch_norm`
- Activations (unless fused with conv2d): `ttnn.relu`, `ttnn.gelu`, `ttnn.silu`, `ttnn.leaky_relu`
- Other operations: See the full TT-NN API documentation

### Testing Utilities

TT-CNN also provides testing utilities to simplify model validation:
- **Input tensor creation**: `create_random_input_tensor()` supports various formats (NCHW/NHWC, folded/unfolded, padded/unpadded)
- **Layer verification**: `verify_conv2d_from_config()` and `verify_maxpool2d_from_config()` for automatic correctness testing
- **Performance testing**: `run_device_perf_test()` for declarative device performance benchmarking

See the [Testing and Validation](#testing-and-validation) section for detailed usage.


### Choosing Sharding Strategies

Sharding determines how tensors are distributed across device cores:

- **HeightShardedStrategyConfiguration**: Best for tensors with large spatial dimensions (H×W)
  - `act_block_h_override`: Controls activation block height (must be multiple of 32)
  - Larger values = more L1 memory per core -> you want to maximize this value without running out of L1

- **WidthShardedStrategyConfiguration**: Best for inputs/outputs with deep channels
  - `act_block_w_div`: Divides the width dimension for sharding
  - Use when channels >> spatial dimensions

- **BlockShardedStrategyConfiguration**: Shards across both height and width
  - Combines both parameters for 2D sharding

- **AutoShardedStrategyConfiguration**: Automatically selects optimal strategy (minimizing L1 usage)
  - Good starting point, but manual tuning often yields better performance

## 5. Creating the TT-NN Model

The model uses TT-CNN builder components with the configurations:

```python
# tt/model.py
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d

class TtUNet:
    def __init__(self, configs: TtUNetLayerConfigs, device: ttnn.Device):
        self.device = device
        self.configs = configs

        # Create encoder blocks using configurations
        self.downblock1 = TtUNetEncoder(
            configs.encoder1_conv1,  # Conv config with weights
            configs.encoder1_conv2,  # Conv config with weights
            configs.encoder1_pool,   # Pool config
            device
        )
        # ... encoder2-4, bottleneck, decoder blocks ...
        # Full model: ../demos/vanilla_unet/tt/model.py:112-141

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        # Convert input format from CHW to HWC
        # TT-NN convolutions expect HWC format (Height-Width-Channel)
        # while PyTorch uses CHW format (Channel-Height-Width)
        input_tensor = ttnn.experimental.convert_to_hwc(input_tensor)

        # Encoder path with skip connections
        enc1, skip1 = self.downblock1(input_tensor)
        enc2, skip2 = self.downblock2(enc1)
        # ... enc3, enc4, bottleneck processing ...

        # Decoder path with upsampling and concatenation
        dec4 = transpose_conv2d(bottleneck, self.upconv4_config)
        dec4 = concatenate_skip_connection(dec4, skip4)
        # ... dec3, dec2, dec1, final_conv processing ...
        # Full forward pass: ../demos/vanilla_unet/tt/model.py:149-184

        # Convert output format back to CHW for PyTorch compatibility
        return ttnn.experimental.convert_to_chw(output, dtype=ttnn.bfloat16)

class TtUNetEncoder:
    def __init__(self, conv1, conv2, pool, device):
        self.conv1 = TtConv2d(conv1, device)  # Uses weights from config
        self.conv2 = TtConv2d(conv2, device)  # Uses weights from config
        self.pool = TtMaxPool2d(pool, device)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)  # Save for skip connection
        x = self.pool(x)
        return x, skip
```

### Handling Skip Connections and Concatenation

For architectures with skip connections (ResNet, UNet, etc.), you may need to:

1. **Store skip connections in DRAM** to free L1 memory:
   ```python
   skip = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
   ```

2. **Implement concatenation** for feature fusion:
   ```python
   def concatenate_skip_connection(upsampled, skip):
       # Ensure both tensors have same sharding
       if not skip.is_sharded():
           skip = ttnn.to_memory_config(skip, upsampled.memory_config())

       # Concatenate along channel dimension
       return ttnn.concat([upsampled, skip], dim=3)  # HWC format: dim=3 is channels
   ```

This is a common pattern for any architecture with residual connections or feature fusion.

See the UNet implementation:
- [concatenate_skip_connection()](../demos/vanilla_unet/tt/model.py:10-53)

## 6. Put it all Together

Here's how everything comes together in the demo (adapted from [demo.py](../demos/vanilla_unet/demo/demo.py:97-162)):

```python
# demo/demo.py
# Initialize device (typically provided by pytest fixtures)
import ttnn
device = ttnn.open_device(device_id=0)

# Step 1: Load PyTorch model
reference_model = load_reference_model()

# Step 2: Preprocess weights
parameters = preprocess_model_parameters(
    initialize_model=lambda: reference_model,
    custom_preprocessor=create_unet_preprocessor(device),
    device=None,
)

# Step 3: Create configurations with weights
configs = create_unet_configs_from_parameters(
    parameters=parameters,  # Contains all preprocessed weights
    input_height=480,
    input_width=640,
    batch_size=1,
)

# Step 4: Build TT-NN model
tt_model = create_unet_from_configs(configs, device)

# Run inference
# Option 1: Manual input preparation
ttnn_input = prepare_ttnn_input(input_data, configs.l1_input_memory_config)

# Option 2: Use TT-CNN testing utility (supports various formats)
from models.tt_cnn.tt.testing import create_random_input_tensor
torch_input, ttnn_input = create_random_input_tensor(
    batch_size=1,
    input_channels=3,
    input_height=480,
    input_width=640,
    channel_order="first",  # "first" for NCHW, "last" for NHWC
    fold=True,             # Flatten HW dimensions for conv
    pad=True,              # Pad to tile size
    ttnn_dtype=ttnn.bfloat16,
    memory_config=configs.l1_input_memory_config
)

output = tt_model(ttnn_input)

# Cleanup
ttnn.close_device(device)
```

## 7. Applying Tracing and Multi-CQ Pipeline

For optimal end-to-end performance, wrap your model with the Pipeline API.

### Understanding Pipeline Optimizations

#### What Tracing Does

Tracing captures and optimizes the execution of your model:

1. **Command Recording**: During the first run (compile), TT-NN records all device commands (kernels, data movements) into a command buffer
2. **Optimization**: The trace eliminates CPU-device communication overhead by pre-recording the entire execution sequence
3. **Replay**: Subsequent runs simply replay the recorded commands without Python/C++ overhead

Without tracing, each operation involves:
```
Python → C++ → Command Generation → Device Execution
```

With tracing:
```
First run: Python → C++ → Record Commands → Device Execution
Later runs: Replay Recorded Commands → Device Execution (no Python/C++ overhead)
```

#### Why 2 Command Queues Improve Performance

Command queues (CQs) enable overlapping of different operations. With a single command queue, operations execute sequentially - transfers and compute cannot overlap. With two command queues, you can overlap data transfers with model execution, significantly improving throughput.

The key benefit: While CQ0 runs the model on batch N, CQ1 can transfer batch N+1, hiding transfer latency.

For detailed diagrams and explanations of how multiple command queues work, see the [Multiple Command Queues section](../../tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md#2-multiple-command-queues) in the Advanced Performance Optimizations guide.

#### When to Use `all_transfers_on_separate_command_queue`

This option puts ALL transfers (input AND output) on a separate CQ:

**Use when:**
- Your model has significant output transfer time
- You want maximum overlap of compute and I/O

**Don't use when:**
- Output transfer time is negligible

```python
# Standard 2CQ: Input transfers on CQ1, compute+output on CQ0
config = PipelineConfig(use_trace=True, num_command_queues=2)

# Full separation: All transfers on CQ1, only compute on CQ0
config = PipelineConfig(
    use_trace=True,
    num_command_queues=2,
    all_transfers_on_separate_command_queue=True
)
```

### Implementation Example

See [test_unet_perf.py](../demos/vanilla_unet/tests/test_unet_perf.py:72-177) for complete example.

```python
# test_perf.py
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

def create_unet_pipeline_model(model):
    """Wrapper to make model compatible with pipeline"""
    def run(input_l1_tensor: ttnn.Tensor):
        # Model receives L1 tensors from pipeline
        return model(input_l1_tensor, deallocate_input_activation=False)
    return run

# Configure pipeline with tracing and 2 command queues
pipeline = create_pipeline_from_config(
    config=PipelineConfig(
        use_trace=True,  # Enable tracing for performance
        num_command_queues=2,  # Use 2 CQs for overlapped execution
        all_transfers_on_separate_command_queue=True  # Separate I/O queue
    ),
    model=create_unet_pipeline_model(tt_model),
    device=device,
    dram_input_memory_config=dram_input_memory_config,
    dram_output_memory_config=dram_output_memory_config,
    l1_input_memory_config=configs.l1_input_memory_config,
)

# Compile once
pipeline.compile(sample_input)

# Run inference
input_tensors = [ttnn_input] * num_iterations
outputs = pipeline.enqueue(input_tensors).pop_all()

# Cleanup
pipeline.cleanup()
```

## 8. Performance Analysis and Optimization

### Understanding Device vs End-to-End Performance

When optimizing vision models, it's crucial to distinguish between device performance and end-to-end performance:

**Device Performance**: How fast the model runs on the accelerator (excluding data transfers)
- Measures only kernel time
- Best case scenario performance

**End-to-End Performance**: Total time including host-device transfers
- Includes PCI-e transfer overhead
- Can be significantly lower than device performance for vision models

### Identifying Performance Bottlenecks

Vision models can be bottlenecked by different factors:

#### 1. **Transfer Bottleneck**
**Symptoms**:
- Large gap between device and e2e performance

**Solutions**:
- Use Pipeline API with 2CQ to overlap transfers
- Reduce unnecessary padding in input/output tensors

### Performance Optimization Techniques

Once you've identified your bottlenecks, apply these optimization techniques:

#### Use Sharding Everywhere

Avoid going to L1 or DRAM interleaved wherever possible. For example, make sure residuals are sharded before performing channel-wise concatenation. Select sharding configs that will minimize the amount of reshards throughout the model.

#### Sliding Window Configuration

Tuning activation block sizes can make a significant improvement in performance in convolution layers. Experiment with increasing block size as much as possible (without running out of L1) on each convolution layer.

```python
# Tune convolution activation block sizes to improve performance
sharding_strategy = HeightShardedStrategyConfiguration(
    act_block_h_override=15 * 32,  # Must fit in L1 memory per core
    reshard_if_not_optimal=False
)

# Enable double buffering for throughput
enable_act_double_buffer=True,      # Double buffer activations
enable_weights_double_buffer=True,  # Double buffer weights
```

#### Data Types

Lower precision data types can improve performance at the cost of some accuracy. You can experiment with different activation and weight data types to determine the optimal value for each layer.

```python
# Example mixed precision strategy
encoder1_conv1=Conv2dConfiguration(
    activation_dtype=ttnn.bfloat16,    # First layer uses higher precision
    weights_dtype=ttnn.bfloat8_b,      # Weights in lower precision
    output_dtype=ttnn.bfloat16,
)

encoder2_conv1=Conv2dConfiguration(
    activation_dtype=ttnn.bfloat8_b,   # Intermediate layers use lower precision
    weights_dtype=ttnn.bfloat8_b,
    output_dtype=ttnn.bfloat8_b,
)
```

#### Multi-Device Support

For scaling to multiple devices:

```python
# For multi-device deployment
inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

parameters = preprocess_model_parameters(
    initialize_model=lambda: reference_model,
    custom_preprocessor=create_unet_preprocessor(mesh_device, mesh_mapper=weights_mesh_mapper),
    device=None,
)
```

## Testing and Validation

### Layer-by-Layer Correctness Testing

TT-CNN provides utilities to verify individual layers against PyTorch reference:

```python
from models.tt_cnn.tt.testing import verify_conv2d_from_config, verify_maxpool2d_from_config

# Test a single Conv2d layer
def test_encoder1_conv1(device):
    config = encoder1_conv1_config  # From your config.py
    pcc_threshold = 0.99

    # Automatically creates test input, runs PyTorch reference,
    # and compares with TT-NN implementation
    verify_conv2d_from_config(config, device, pcc_threshold)

# Test a MaxPool2d layer
def test_encoder1_pool(device):
    config = encoder1_pool_config
    verify_maxpool2d_from_config(config, device, pcc_threshold)
```

## Troubleshooting

### Common Issues and Solutions

1. **Memory Errors** (`Out of L1 memory`):
   - Reduce `act_block_h_override` or other L1 hungry configuration options
   - Use DRAM slicing for large layers that don't fit
   - Move intermediate tensors to DRAM
   - Use bfloat8_b activations to reduce memory pressure from intermediate tensors

2. **Accuracy Issues**:
   - Rule out correctness issues using [comparison model](../../tech_reports/ttnn/comparison-mode.md) or unit tests
   - Start with all activations in bfloat16 to establish baseline and then incrementally determine where you can reduce precision

4. **Performance Issues**:
   - Profile with tracy to identify bottlenecks
   - Is device performance much higher than end-to-end performance?
        - Vision models can be sensitive to host-to-device transfer overhead due to the large volume of data that must be sent over PCI-E
        - Use the pipeline API with tracing and 2CQs
        - Experiment with different pipeline configurations to find the best for your model
        - Eliminate any padding on host-to-device and device-to-host transfers
