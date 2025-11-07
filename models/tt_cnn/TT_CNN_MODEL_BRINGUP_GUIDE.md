# TT-CNN - Vision Model Bring-up Guide

This documents demonstrates how to bring-up new vision models using the TT-CNN library.

The TT-CNN library comprises several components designed to assist the development of vision models on Tenstorrent devices. At a high level, the main TT-CNN modules include:

- **Pipeline**: Provides abstractions for high-performance end-to-end model inference, with out-of-the-box support for tracing, and multi-command command queue pipelining.
- **Builder**: Provides configuration-driven API for defining and instantiating convolution and pooling layers, making it easy to compose models from reusable configuration classes.

Each module is documented and illustrated with working examples. To dive deeper into:
- **Pipeline API, memory layouts, and performance tuning**, see the [README.md](README.md) "Pipeline" and "Performance Optimization Techniques" sections.
- **Builder API, configuration patterns, and layer composition**, refer to the [README.md](README.md) "Builder" section.

Throughout this document we will refer to the Vanilla UNet implementation as a concrete example of how to use the available TT-CNN modules to create a model. Take a look at the implementation [here](../demos/vanilla_unet/README.md).

For further details, practical tips, and reference patterns, please explore the in-depth documentation in [README.md](README.md).

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
└── demo/
|    └── demo.py          # End-to-end demonstration
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
        # ... more encoder blocks ...
```

## 3. Weight Preprocessing

The `create_unet_preprocessor` function extracts and converts PyTorch weights to TT-NN format:

```python
# tt/common.py
def create_unet_preprocessor(device, mesh_mapper=None):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}

        if isinstance(model, UNet):
            # Process each encoder block
            for i in range(1, 5):
                parameters[f"encoder{i}"] = {}

                # Fold BatchNorm into Conv weights
                from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

                # First conv in block
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

            # Process decoder blocks similarly...
            # Process upconv layers
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
            # ... more layer configs ...
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


### Choosing Sharding Strategies

Sharding determines how tensors are distributed across device cores:

- **HeightShardedStrategyConfiguration**: Best for tensors with large spatial dimensions (H×W)
  - `act_block_h_override`: Controls activation block height (must be multiple of 32)
  - Larger values = more L1 memory per core -> you want to maxmize this value without running out of L1

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
        # ... more blocks ...

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        # Convert input format from CHW to HWC
        # TT-NN convolutions expect HWC format (Height-Width-Channel)
        # while PyTorch uses CHW format (Channel-Height-Width)
        input_tensor = ttnn.experimental.convert_to_hwc(input_tensor)

        # Encoder path with skip connections
        enc1, skip1 = self.downblock1(input_tensor)
        enc2, skip2 = self.downblock2(enc1)
        # ... encoder path ...

        # Decoder path with upsampling and concatenation
        dec4 = transpose_conv2d(bottleneck, self.upconv4_config)
        dec4 = concatenate_skip_connection(dec4, skip4)
        # ... decoder path ...

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

## 6. Put it all Together

Here's how everything comes together in the demo:

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
ttnn_input = prepare_ttnn_input(input_data, configs.l1_input_memory_config)
output = tt_model(ttnn_input)

# Cleanup
ttnn.close_device(device)
```

## 7. Applying Tracing and Multi-CQ Pipeline

For optimal end-to-end performance, wrap your model with the Pipeline API:

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

## 8. Performance Optimization

### Use sharding everywhere!

Try to avoid going to L1 or DRAM interleaved wherever possible. For example, make sure residuals are sharded before performance channel-wise concatenation. Select sharding configs that will minimize the about of reshards throughout the model.

### Sliding Window Configuration
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

### Data Types
Lower precision data types can improve performance at the cost of some accuracy. You can experiement with different activation and weight data types to determine the optimial value for each layer.
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

### Multi-Device Support
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

## Troubleshooting

### Common Issues and Solutions

1. **Memory Errors** (`Out of L1 memory`):
   - Reduce `act_block_h_override` to create smaller shards
   - Use DRAM slicing for large layers that don't fit
   - Move intermediate tensors to DRAM
   - Use bfloat8_b activations

2. **Accuracy Issues**:
   - Rule out correctness issues using (comparison model)[../../tech_reports/ttnn/comparison-mode.md] or unit tests
   - Start with all activations in bfloat16 to establish baseline

4. **Performance Issues**:
   - Use the Pipeline API with tracing enabled
   - Enable double buffering for weights and activations
   - Profile with `ttnn.DumpDeviceProfiler()` to identify bottlenecks
