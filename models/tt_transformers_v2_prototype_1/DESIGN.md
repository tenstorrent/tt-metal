# TTTv2 Design Document

## Overview

TT Transformers v2 (TTTv2) is a modular, composable library for implementing transformer models on Tenstorrent hardware. It addresses the scaling challenges of TTTv1 by providing clear boundaries between the core library and model implementations.

## Key Design Principles

### 1. Tightened Scope
- Only core modules are part of TTTv2
- Model implementations (LLaMA, Mistral, etc.) are **external** to TTTv2
- Clear separation of concerns between library and implementations

### 2. Minimal Dependencies
- Core modules depend only on TTNN
- No model-specific logic in core
- Clean, well-defined interfaces

### 3. Semantic Versioning
- Stable API with predictable versioning
- Model implementations pin to specific TTTv2 versions
- Backward compatibility within major versions

## Architecture

### Core Modules (`/core`)
Fundamental building blocks with minimal dependencies:
- **Attention**: Multi-head and grouped-query attention
- **MLP**: Standard and gated feedforward networks
- **Normalization**: RMSNorm, LayerNorm, GroupNorm
- **Embeddings**: Token, position, and vision embeddings
- **RoPE**: Rotary position embeddings with scaling
- **TransformerBlock**: Combines attention and MLP
- **LMHead**: Output projection for language models

### Interfaces (`/interfaces`)
Standard interfaces for integration:
- **Generator**: Text generation with various strategies
- **VLLMGenerator**: vLLM-compatible batch generation
- **HWConfig**: Hardware configuration and optimization
- **DemoBase**: Standardized demo framework

### Configuration (`/config`)
ML model configuration and optimization:
- **ModelConfig**: Base and specialized model configurations
- **OptimizationConfig**: Performance optimization settings
- **WeightLoader**: Weight loading and conversion utilities

## Usage Pattern

Model implementations follow this pattern:

```python
# Import specific TTTv2 version
from tt_transformers_v2.core import Attention, MLP, TransformerBlock
from tt_transformers_v2.config import TransformerConfig
from tt_transformers_v2.interfaces import HWConfig, Generator

# Build model using TTTv2 components
class MyModel(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        # Use TTTv2 modules
        self.attention = Attention(config, device)
        self.mlp = MLP(config, device)
        ...

# Model-specific logic stays in model repo
def create_my_model(...):
    # Model-specific setup
    ...
```

## Benefits

1. **Scalability**: Adding new models doesn't affect existing ones
2. **Maintainability**: Clear ownership boundaries
3. **Flexibility**: Models can override defaults as needed
4. **Stability**: Semantic versioning ensures predictable updates
5. **Performance**: Optimized for TT hardware with minimal overhead

## Migration from TTTv1

Models using TTTv1 can migrate by:
1. Replace TTTv1 imports with TTTv2
2. Update to new API where needed
3. Pin to specific TTTv2 version
4. Move model-specific logic out of core

## Example

See `/examples/example_llama.py` for a complete example of building a LLaMA model using TTTv2.
