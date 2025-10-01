# Model Specification Standards Analysis for TTTv2

## Overview of Existing Standards

### 1. **Architecture Description Standards**

These standards describe the computational graph and model structure:

#### ONNX (Open Neural Network Exchange)
- **Purpose**: Cross-framework interoperability
- **Scope**: Full computation graph + weights
- **Key Features**:
  - Includes complete DAG of operations
  - Vendor-neutral, supports many frameworks
  - Extensible operator set
- **Limitations**:
  - Can be verbose for simple architectures
  - Not optimized for specific hardware

#### CoreML (Apple)
- **Purpose**: Apple ecosystem deployment
- **Scope**: Architecture + weights + metadata
- **Key Features**:
  - Protobuf-based specification
  - ML Program format with MIL (Model Intermediate Language)
  - Optimized for Apple Silicon
- **Limitations**:
  - Platform-specific
  - Limited to Apple's supported operations

#### TensorRT (NVIDIA)
- **Purpose**: GPU inference optimization
- **Scope**: Network definition for GPU execution
- **Key Features**:
  - Two-phase: definition → optimization
  - Plugin system for custom ops
  - Aggressive fusion and optimization
- **Limitations**:
  - NVIDIA GPU specific
  - Version and hardware locked

### 2. **Weight Storage Standards**

These focus on efficient storage and loading of model parameters:

#### SafeTensors (HuggingFace)
- **Purpose**: Secure, efficient weight storage
- **Scope**: Tensors + metadata only
- **Key Features**:
  - Memory-mapped loading
  - Security focused (no arbitrary code execution)
  - Simple JSON header + binary data
- **Limitations**:
  - No architecture information
  - Requires separate model definition

#### GGUF (llama.cpp)
- **Purpose**: Quantized model storage for local inference
- **Scope**: Weights + minimal metadata
- **Key Features**:
  - Built-in quantization schemes
  - Optimized for CPU/consumer GPU
  - Self-contained format
- **Limitations**:
  - Primarily for transformer models
  - Limited to supported quantization methods

### 3. **High-Level Model Cards**

These describe models at a conceptual level:

#### HuggingFace Model Cards
- **Purpose**: Documentation and discovery
- **Scope**: High-level architecture description
- **Example**:
```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "vocab_size": 32000
}
```

## Comparison Matrix

| Standard | Architecture | Weights | Hardware-Specific | Quantization | Portability |
|----------|--------------|---------|-------------------|--------------|-------------|
| ONNX | Full DAG | ✓ | ✗ | Limited | High |
| CoreML | Full spec | ✓ | Apple only | Basic | Low |
| TensorRT | Network def | ✓ | NVIDIA only | Advanced | Low |
| SafeTensors | ✗ | ✓ | ✗ | ✗ | High |
| GGUF | Minimal | ✓ | ✗ | Built-in | Medium |
| HF Cards | High-level | ✗ | ✗ | ✗ | High |

## Key Insights

1. **No Single Standard Rules All**: Each standard serves different purposes
2. **Architecture vs Weights Separation**: Most modern formats separate these concerns
3. **Hardware Optimization Trade-offs**: Platform-specific formats offer better performance
4. **Quantization is Fracturing Standards**: Each format handles quantization differently

## Design Implications for TTTv2

### Making TTTv2 Standard-Agnostic

#### 1. **Adapter Pattern for Standards**

```python
# Define minimal interface for model specs
class ModelSpecAdapter(Protocol):
    """Minimal interface that any standard must provide"""
    def get_layers(self) -> List[LayerSpec]: ...
    def get_layer_config(self, layer_id: str) -> Dict[str, Any]: ...
    def get_connections(self) -> List[Tuple[str, str]]: ...

# Implement adapters for each standard
class ONNXAdapter(ModelSpecAdapter):
    def __init__(self, onnx_model):
        self.model = onnx_model

    def get_layers(self):
        # Convert ONNX nodes to LayerSpecs
        return [self._node_to_layer(node) for node in self.model.graph.node]

class HuggingFaceAdapter(ModelSpecAdapter):
    def __init__(self, hf_config):
        self.config = hf_config

    def get_layers(self):
        # Convert HF config to layer specs
        layers = []
        for i in range(self.config.num_hidden_layers):
            layers.append(LayerSpec(
                type="transformer",
                config={"hidden_size": self.config.hidden_size}
            ))
        return layers

class TTTModelFactory:
    @staticmethod
    def from_spec(spec: ModelSpecAdapter, device):
        """Build TTT model from any adapted spec"""
        layers = []
        for layer_spec in spec.get_layers():
            if layer_spec.type == "attention":
                layers.append(attention.MultiHeadAttention(**layer_spec.config))
            # ... handle other layer types
        return Model(layers)
```

#### 2. **Internal Canonical Format**

Create a minimal internal representation that captures only what TTTv2 needs:

```python
@dataclass
class TTTLayerSpec:
    """Minimal layer specification"""
    layer_type: str  # "attention", "ffn", "norm", etc.
    params: Dict[str, Any]  # Layer-specific parameters
    input_names: List[str]
    output_names: List[str]

@dataclass
class TTTModelSpec:
    """Minimal model specification"""
    layers: List[TTTLayerSpec]
    connections: List[Tuple[str, str]]  # (from_output, to_input)
    metadata: Dict[str, Any]  # Optional metadata
```

#### 3. **Registration System for Standards**

```python
class ModelSpecRegistry:
    _adapters = {}

    @classmethod
    def register(cls, format_name: str, adapter_class: Type[ModelSpecAdapter]):
        cls._adapters[format_name] = adapter_class

    @classmethod
    def load(cls, format_name: str, model_data: Any) -> ModelSpecAdapter:
        if format_name not in cls._adapters:
            raise ValueError(f"Unknown format: {format_name}")
        return cls._adapters[format_name](model_data)

# Register standard adapters
ModelSpecRegistry.register("onnx", ONNXAdapter)
ModelSpecRegistry.register("huggingface", HuggingFaceAdapter)
ModelSpecRegistry.register("safetensors", SafeTensorsAdapter)

# Users can register custom adapters
ModelSpecRegistry.register("custom_format", CustomAdapter)
```

#### 4. **Lazy Evaluation Strategy**

Don't require full conversion upfront:

```python
class LazyModelSpec:
    """Load only what's needed when it's needed"""
    def __init__(self, adapter: ModelSpecAdapter):
        self.adapter = adapter
        self._layer_cache = {}

    def get_layer(self, layer_id: str) -> TTTLayerSpec:
        if layer_id not in self._layer_cache:
            # Convert on-demand
            self._layer_cache[layer_id] = self._convert_layer(layer_id)
        return self._layer_cache[layer_id]
```

## Recommendations

1. **Don't Pick a Winner**: Support multiple standards through adapters
2. **Keep Internal Format Minimal**: Only what's needed for TTT operations
3. **Make Adapters Pluggable**: Let users add support for new formats
4. **Focus on Transformer Primitives**: Don't try to support every possible operation
5. **Separate Concerns**: Architecture spec ≠ weight format ≠ execution strategy

## Example Usage

```python
# Load from different sources transparently
from tt_transformers_v2 import ModelFactory, ModelSpecRegistry

# From ONNX
onnx_model = onnx.load("model.onnx")
spec = ModelSpecRegistry.load("onnx", onnx_model)
model = ModelFactory.from_spec(spec, device=ttnn_device)

# From HuggingFace
hf_config = AutoConfig.from_pretrained("meta-llama/Llama-3-8b")
spec = ModelSpecRegistry.load("huggingface", hf_config)
model = ModelFactory.from_spec(spec, device=ttnn_device)

# From custom format
custom_spec = load_my_format("model.custom")
ModelSpecRegistry.register("myformat", MyFormatAdapter)
spec = ModelSpecRegistry.load("myformat", custom_spec)
model = ModelFactory.from_spec(spec, device=ttnn_device)
```

This approach keeps TTTv2 flexible and future-proof without being tied to any particular standard.
