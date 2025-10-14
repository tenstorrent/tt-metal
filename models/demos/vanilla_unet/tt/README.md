# TT-CNN UNet Implementation

Complete UNet implementation using the TT-CNN framework with preprocessed parameters from the original TTNN implementation.

## Key Features

- **Parameter-based configuration**: Creates TT-CNN configurations from preprocessed parameters with fused batch normalization
- **Complete UNet model**: Full encoder-decoder architecture with skip connections
- **480x640 default input size**: Optimized for the standard input dimensions
- **TT-CNN integration**: Uses modern TT-CNN builders for optimal performance
- **UpconvConfiguration**: Structured configuration for transpose convolutions with all parameters

## Files

- `unet_config.py`: Configuration builder for UNet layers and UpconvConfiguration dataclass
- `ttnn_unet.py`: Complete TtUNet model implementation with UpconvConfiguration support

## Usage

### Create UNet Model

```python
from models.demos.vanilla_unet.tt.unet_config import create_unet_configs_from_parameters
from models.demos.vanilla_unet.tt.ttnn_unet import create_unet_from_configs

# Create configurations from preprocessed parameters
configs = create_unet_configs_from_parameters(
    parameters=preprocessed_params,  # From original TTNN implementation
    input_height=480,                # Optional, defaults to 480
    input_width=640,                 # Optional, defaults to 640
    batch_size=1                     # Optional, defaults to 1
)

# Create the model
device = ttnn.CreateDevice(0)
model = create_unet_from_configs(configs, device)

# Run inference
output = model(input_tensor)  # Input: (batch, height, width, channels)
```

## Parameter Structure

The `parameters` dict should follow the original TTNN implementation structure:
- `parameters["encoder1"][0]`, `parameters["encoder1"][1]` - Encoder layer parameters
- `parameters["bottleneck"][0]`, `parameters["bottleneck"][1]` - Bottleneck parameters
- `parameters["decoder4"][0]`, `parameters["decoder4"][1]` - Decoder layer parameters
- `parameters["upconv4"]`, `parameters["upconv3"]`, etc. - Transpose convolution parameters
- `parameters["conv"]` - Final output layer parameters

Each parameter dict contains:
- `"weight"`: Preprocessed weight tensor with batch norm fused
- `"bias"`: Preprocessed bias tensor with batch norm fused

## UpconvConfiguration

The `UpconvConfiguration` dataclass stores transpose convolution parameters:

```python
@dataclass
class UpconvConfiguration:
    input_height: int
    input_width: int
    in_channels: int
    out_channels: int
    batch_size: int
    kernel_size: Tuple[int, int] = (2, 2)
    stride: Tuple[int, int] = (2, 2)
    padding: Tuple[int, int] = (0, 0)
    weight: ttnn.Tensor = None
    bias: ttnn.Tensor = None
```

## Configuration Output

Returns `TtUNetLayerConfigs` containing:
- All encoder convolution and pooling configurations (encoder1-encoder4)
- Bottleneck convolution configurations
- All decoder convolution configurations (decoder1-decoder4)
- Structured transpose convolution configurations (upconv1-upconv4)
- Final convolution configuration
