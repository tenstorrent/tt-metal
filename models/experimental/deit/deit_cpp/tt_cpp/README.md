# DeiT C++ Implementation - TT Components

This directory contains the C++ implementation of DeiT (Data-efficient Image Transformers) attention components, converted from the original Python implementation.

## Files Overview

### Header Files
- `deit_config.h` - Configuration structure for DeiT model parameters
- `deit_attention.h` - Main attention module combining self-attention and output layers
- `deit_embeddings.h` - Input embeddings with patch, position, and token embeddings
- `deit_self_attention.h` - Self-attention mechanism implementation
- `deit_self_output.h` - Output projection layer for attention
- `deit_pooler.h` - Pooler module for extracting CLS token representation

### Implementation Files
- `deit_attention.cpp` - Implementation of the main attention module
- `deit_embeddings.cpp` - Implementation of input embeddings layer
- `deit_self_attention.cpp` - Implementation of self-attention with Q, K, V projections
- `deit_self_output.cpp` - Implementation of the output projection layer
- `deit_pooler.cpp` - Implementation of the pooler module with linear transformation and tanh activation

### Build Files
- `CMakeLists.txt` - CMake configuration for building the C++ components
- `README.md` - This documentation file

## Architecture

The C++ implementation follows the same structure as the Python version:

```
TtDeiTAttention
├── TtDeiTSelfAttention (handles Q, K, V projections and attention computation)
└── TtDeiTSelfOutput (handles output projection)
```

## Key Features

1. **Memory Management**: Uses smart pointers for automatic memory management
2. **Error Handling**: Comprehensive error checking for null pointers and missing parameters
3. **TTNN Integration**: Full integration with TTNN tensor operations
4. **Configuration**: Flexible configuration through DeiTConfig structure
5. **State Dict Loading**: Support for loading pre-trained model weights

## Usage Example

```cpp
#include "deit_attention.h"

// Create configuration
DeiTConfig config;
config.hidden_size = 768;
config.num_attention_heads = 12;

// Load model parameters
std::unordered_map<std::string, ttnn::Tensor> state_dict = load_model_weights("path/to/weights");

// Create attention module
TtDeiTAttention attention(config, device, state_dict, "encoder.layer.0.attention");

// Forward pass
auto [output, attention_weights] = attention.forward(hidden_states, std::nullopt, false);
```

## Dependencies

- TTNN library for tensor operations
- TT-Metal for device management
- C++17 standard library

## Building

This module is built as part of the larger DeiT C++ project. The CMakeLists.txt file defines an object library that can be linked with other components.

## Notes

- The residual connection is handled in the DeiTLayer (not implemented here), following the original Python design
- All tensor operations use TTNN for optimal performance on Tenstorrent hardware
- The implementation maintains compatibility with the Python version's interface and behavior