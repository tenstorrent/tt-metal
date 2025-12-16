# TT-Symbiote

PyTorch-to-TTNN acceleration framework for transparent hardware acceleration of neural networks on Tenstorrent devices.

## Overview

TT-Symbiote enables TTNN acceleration of pretrained PyTorch models by replacing standard PyTorch modules (e.g., `nn.Linear`, `nn.LayerNorm`) with TTNN-optimized equivalents. The framework automatically handles:
- Module replacement and weight conversion
- Device management and memory allocation
- Fallback to PyTorch when TTNN operations fail

## Quick Start

```python
import torch
from torch import nn
from transformers import AutoModelForImageClassification
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.modules.linear import TTNNLinear
from models.tt_symbiote.modules.normalization import TTNNLayerNorm

# Load model
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Define module replacement mapping
nn_to_ttnn = {
    nn.Linear: TTNNLinear,
    nn.LayerNorm: TTNNLayerNorm,
}

# Replace modules and set device
register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

#####
# Get TTNN device
ttnn_device = # Obtain TTNN device (e.g., through pytest fixture or ttnn.CreateDevice)
#####

set_device(model, ttnn_device)

# Run inference
model.eval()
torch.set_grad_enabled(False)
result = model(torch.randn(1, 3, 224, 224))


```

## Creating a New TTNN Module

All TTNN modules inherit from `TTNNModule` and implement:

```python
from models.tt_symbiote.core.module import TTNNModule
import ttnn
from torch import nn

class TTNNCustomLayer(TTNNModule):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    @classmethod
    def from_torch(cls, torch_layer):
        """Create TTNN module from PyTorch equivalent."""
        new_layer = TTNNCustomLayer(torch_layer.param1, torch_layer.param2)
        new_layer._fallback_torch_layer = torch_layer
        return new_layer

    def preprocess_weights_impl(self):
        """Convert PyTorch weights to TTNN format (called once)."""
        self.tt_weight = ttnn.from_torch(
            self.torch_layer.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to device."""
        self.tt_weight = ttnn.to_device(self.tt_weight, self.device)

    def move_weights_to_host_impl(self):
        """Move weights back to host."""
        self.tt_weight = self.tt_weight.cpu()

    def deallocate_weights_impl(self):
        """Deallocate device memory."""
        ttnn.deallocate(self.tt_weight)

    def forward(self, input_tensor):
        """TTNN forward implementation."""
        output = ttnn.custom_op(input_tensor, self.tt_weight)
        return output
```

**Key Methods:**
- `from_torch()`: Factory method to create from PyTorch module
- `preprocess_weights_impl()`: Convert weights to TTNN format (runs once)
- `move_weights_to_device_impl()`: Transfer weights to device
- `forward()`: TTNN implementation of the operation
- `deallocate_weights_impl()`: Free device memory

The base class handles:
- Automatic fallback to PyTorch on errors
- Tensor wrapping/unwrapping
- Weight lifecycle management
- Device placement

## Running Tests

Tests require manual invocation with a TTNN device:

```python
pytest tests/test_vit.py
pytest tests/test_llama.py
pytest tests/test_owl_vit.py
pytest tests/test_speech_t5.py
```

## Architecture

```
core/
├── module.py          # TTNNModule base class with auto-fallback
├── tensor.py          # TorchTTNNTensor wrapper for PyTorch dispatch
└── dispatcher.py      # TTNN operation dispatch handlers

modules/               # TTNN implementations
├── linear.py          # TTNNLinear, TTNNLinearLLama,...
├── attention.py       # TTNNViTSelfAttention,...
├── normalization.py   # TTNNLayerNorm,...
└── activation.py      # TTNNSilu,...

utils/
├── module_replacement.py  # Recursive module swapping
└── device_management.py   # Device configuration
```

## Examples

See [tests/](tests/) directory:
- [test_vit.py](tests/test_vit.py) - Vision Transformer with TTNN Linear, LayerNorm, Attention
- [test_llama.py](tests/test_llama.py) - LLaMA-3-8B with bfloat8 optimizations
- [test_owl_vit.py](tests/test_owl_vit.py) - OWL-ViT with TTNN Attention and Linear
- [test_speech_t5.py](tests/test_speech_t5.py) - SpeechT5 with TTNN Linear LLama and LayerNorm
