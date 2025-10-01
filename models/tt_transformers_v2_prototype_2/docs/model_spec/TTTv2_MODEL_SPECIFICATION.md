# TTTv2 Model Specification Design

## Motivation

Decouple model architecture specification from hardware-specific configurations to enable:
1. Direct conversion from HuggingFace model definitions
2. Hardware-agnostic model descriptions
3. Automatic optimization based on target device
4. Clean separation of concerns

## Design Overview

### 1. Model Specification API

```python
# tt_transformers_v2/src/interfaces/model_spec.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum

class AttentionType(Enum):
    MULTI_HEAD = "multi_head"
    GROUPED_QUERY = "grouped_query"
    MULTI_QUERY = "multi_query"
    SLIDING_WINDOW = "sliding_window"

class ActivationType(Enum):
    SILU = "silu"
    GELU = "gelu"
    RELU = "relu"
    SWISH = "swish"

class NormType(Enum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

@dataclass
class AttentionSpec:
    """Pure architecture specification for attention layer."""
    hidden_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None  # For GQA/MQA
    head_dim: Optional[int] = None  # If different from hidden_dim // num_heads
    attention_type: AttentionType = AttentionType.MULTI_HEAD
    use_rotary: bool = True
    rotary_base: float = 10000.0
    sliding_window_size: Optional[int] = None
    use_bias_q: bool = True
    use_bias_kv: bool = True
    use_bias_o: bool = False

@dataclass
class FFNSpec:
    """Pure architecture specification for FFN layer."""
    hidden_dim: int
    intermediate_dim: int
    activation: ActivationType = ActivationType.SILU
    use_gate: bool = True  # For gated variants like SwiGLU
    use_bias: bool = False

@dataclass
class LayerSpec:
    """Specification for a transformer layer."""
    attention: AttentionSpec
    ffn: FFNSpec
    norm_type: NormType = NormType.RMS_NORM
    norm_eps: float = 1e-6
    pre_norm: bool = True  # Pre-normalization vs post-normalization

@dataclass
class ModelSpec:
    """Complete model architecture specification."""
    vocab_size: int
    hidden_dim: int
    num_layers: int
    layers: List[LayerSpec]  # Can have different specs per layer
    max_seq_len: int = 2048
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_embeddings: bool = False

    @classmethod
    def uniform(cls, **kwargs) -> "ModelSpec":
        """Create a model with uniform layer specifications."""
        layer_spec = LayerSpec(
            attention=AttentionSpec(
                hidden_dim=kwargs["hidden_dim"],
                num_heads=kwargs["num_heads"],
                num_kv_heads=kwargs.get("num_kv_heads"),
                rotary_base=kwargs.get("rotary_base", 10000.0),
            ),
            ffn=FFNSpec(
                hidden_dim=kwargs["hidden_dim"],
                intermediate_dim=kwargs["intermediate_dim"],
                activation=kwargs.get("activation", ActivationType.SILU),
            ),
            norm_type=kwargs.get("norm_type", NormType.RMS_NORM),
        )

        return cls(
            vocab_size=kwargs["vocab_size"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            layers=[layer_spec] * kwargs["num_layers"],
            max_seq_len=kwargs.get("max_seq_len", 2048),
        )
```

### 2. Model Factory

```python
# tt_transformers_v2/src/interfaces/model_factory.py
from typing import Optional
from ..building_blocks import attention, ffn, normalization
from .model_spec import ModelSpec, AttentionType, ActivationType, NormType

class ModelFactory:
    """Factory for creating models from specifications."""

    @staticmethod
    def from_spec(spec: ModelSpec, device: Optional[Any] = None) -> Any:
        """Create a model from a pure architecture specification."""
        return ModelImplementation(spec, device)

    @staticmethod
    def from_huggingface(hf_config: Dict[str, Any]) -> ModelSpec:
        """Convert HuggingFace config to TTTv2 ModelSpec."""
        # Example for Qwen2.5 config
        if hf_config.get("model_type") == "qwen2_5_vl":
            return ModelSpec.uniform(
                vocab_size=hf_config["vocab_size"],
                hidden_dim=hf_config["hidden_size"],
                num_layers=hf_config["num_hidden_layers"],
                num_heads=hf_config["num_attention_heads"],
                num_kv_heads=hf_config.get("num_key_value_heads"),
                intermediate_dim=hf_config["intermediate_size"],
                activation=ActivationType.SILU,
                norm_type=NormType.RMS_NORM,
                rotary_base=hf_config.get("rope_theta", 10000.0),
            )
        # Add more model type conversions...

class ModelImplementation:
    """Actual model implementation from specification."""

    def __init__(self, spec: ModelSpec, device: Optional[Any] = None):
        self.spec = spec
        self.device = device

        # Build model from spec
        self.embedding = self._create_embedding(spec)
        self.layers = [self._create_layer(layer_spec) for layer_spec in spec.layers]
        self.final_norm = self._create_norm(spec.layers[0].norm_type, spec.hidden_dim)

        if not spec.tie_embeddings:
            self.lm_head = self._create_linear(spec.hidden_dim, spec.vocab_size)

    def _create_layer(self, layer_spec: LayerSpec):
        """Create a single layer from specification."""
        # Map spec to actual building blocks
        attn = self._create_attention(layer_spec.attention)
        ffn_block = self._create_ffn(layer_spec.ffn)
        norm = lambda dim: self._create_norm(layer_spec.norm_type, dim)

        # Use patterns for standard architectures
        from ..patterns import DecoderLayer
        return DecoderLayer(
            attention=attn,
            ffn=ffn_block,
            norm=norm,
            pre_norm=layer_spec.pre_norm,
            device=self.device,
        )

    def _create_attention(self, spec: AttentionSpec):
        """Create attention module from spec."""
        if spec.attention_type == AttentionType.MULTI_HEAD:
            return attention.MultiHeadAttention(
                hidden_dim=spec.hidden_dim,
                num_heads=spec.num_heads,
                rope_theta=spec.rotary_base,
                use_bias=(spec.use_bias_q, spec.use_bias_kv, spec.use_bias_o),
                device=self.device,
            )
        elif spec.attention_type == AttentionType.GROUPED_QUERY:
            return attention.GroupedQueryAttention(
                hidden_dim=spec.hidden_dim,
                num_heads=spec.num_heads,
                num_kv_heads=spec.num_kv_heads,
                device=self.device,
            )
        # Add more attention types...

    def _create_ffn(self, spec: FFNSpec):
        """Create FFN module from spec."""
        if spec.use_gate and spec.activation == ActivationType.SILU:
            return ffn.SwiGLU(
                hidden_dim=spec.hidden_dim,
                intermediate_dim=spec.intermediate_dim,
                use_bias=spec.use_bias,
                device=self.device,
            )
        # Add more FFN types...

    def _create_norm(self, norm_type: NormType, dim: int):
        """Create normalization module from spec."""
        if norm_type == NormType.RMS_NORM:
            return normalization.RMSNorm(dim, device=self.device)
        elif norm_type == NormType.LAYER_NORM:
            return normalization.LayerNorm(dim, device=self.device)
```

### 3. Usage Examples

```python
# Example 1: Create model from HuggingFace config
from transformers import AutoConfig
from tt_transformers_v2 import ModelFactory

# Load HF config
hf_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B")

# Convert to TTTv2 spec (pure architecture, no hardware details)
model_spec = ModelFactory.from_huggingface(hf_config.to_dict())

# Create model with specific hardware configuration
model = ModelFactory.from_spec(model_spec, device=ttnn_device)

# Example 2: Manual specification
from tt_transformers_v2.interfaces import ModelSpec, AttentionSpec, FFNSpec

custom_spec = ModelSpec(
    vocab_size=32000,
    hidden_dim=4096,
    num_layers=32,
    layers=[
        LayerSpec(
            attention=AttentionSpec(
                hidden_dim=4096,
                num_heads=32,
                attention_type=AttentionType.GROUPED_QUERY,
                num_kv_heads=8,
            ),
            ffn=FFNSpec(
                hidden_dim=4096,
                intermediate_dim=11008,
                activation=ActivationType.SILU,
            ),
        ) for _ in range(32)
    ],
)

model = ModelFactory.from_spec(custom_spec, device=ttnn_device)

# Example 3: Mixed layer specifications (e.g., different attention for some layers)
layers = []
for i in range(32):
    if i % 4 == 0:  # Every 4th layer uses sliding window
        attn_spec = AttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            attention_type=AttentionType.SLIDING_WINDOW,
            sliding_window_size=2048,
        )
    else:
        attn_spec = AttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            attention_type=AttentionType.MULTI_HEAD,
        )

    layers.append(LayerSpec(attention=attn_spec, ffn=ffn_spec))

mixed_spec = ModelSpec(
    vocab_size=32000,
    hidden_dim=4096,
    num_layers=32,
    layers=layers,
)
```

### 4. Hardware Configuration Separation

```python
# tt_transformers_v2/src/hardware/optimization.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareConfig:
    """Hardware-specific optimizations, separate from model architecture."""
    device_type: str  # "wormhole_b0", "grayskull", etc.

    # Memory optimizations
    use_flash_attention: bool = True
    kv_cache_dtype: Optional[str] = "bfloat16"
    activation_checkpointing: bool = False

    # Parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Quantization
    weight_dtype: str = "bfloat16"
    quantize_weights: bool = False
    quantize_activations: bool = False

    # Performance
    use_fused_kernels: bool = True
    optimize_for_inference: bool = True

class OptimizedModelFactory:
    """Factory that applies hardware optimizations to model specs."""

    @staticmethod
    def create(
        spec: ModelSpec,
        hardware_config: HardwareConfig,
    ) -> Any:
        """Create optimized model for specific hardware."""
        # First create base model from spec
        base_model = ModelFactory.from_spec(spec)

        # Then apply hardware-specific optimizations
        return HardwareOptimizer.optimize(base_model, hardware_config)
```

## Benefits

1. **Clean Separation**: Model architecture is completely separate from hardware details
2. **Portability**: Same ModelSpec can target different hardware
3. **HuggingFace Compatible**: Easy conversion from existing configs
4. **Flexibility**: Developers can still manually construct models
5. **Type Safety**: Strong typing for all specifications
6. **Extensibility**: Easy to add new model types and conversions

## Integration with Testing

```python
# Models can expose their spec for testing
class MyModel:
    def __init__(self, spec: ModelSpec, device):
        self.spec = spec
        # ... implementation

    def get_test_configs(self):
        """Auto-generate test configs from spec."""
        configs = []
        for i, layer_spec in enumerate(self.spec.layers):
            if layer_spec.attention.attention_type == AttentionType.MULTI_HEAD:
                configs.append({
                    "module": f"layer_{i}_attention",
                    "expected_latency": 5.0,
                })
        return configs
```

This design achieves complete separation between:
- **What** the model is (ModelSpec)
- **How** it's implemented (building blocks)
- **Where** it runs (HardwareConfig)
