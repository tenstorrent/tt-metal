"""
TTTv2 Standard-Agnostic Design

This module demonstrates how TTTv2 can work with any model specification standard
without being tied to a particular format.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Type


# Core TTTv2 Model Spec - Minimal and focused on transformers
@dataclass
class TTTLayerSpec:
    """Minimal layer specification for TTTv2"""

    layer_type: str  # "embedding", "attention", "ffn", "norm", "lm_head"
    layer_id: str  # Unique identifier
    params: Dict[str, Any]
    inputs: List[str]  # Input connection points
    outputs: List[str]  # Output connection points


@dataclass
class TTTModelSpec:
    """Complete model specification for TTTv2"""

    name: str
    layers: List[TTTLayerSpec]
    dataflow: List[tuple[str, str]]  # (output_id, input_id) connections
    metadata: Dict[str, Any] = None


# Protocol for spec adapters - what every adapter must implement
class SpecAdapter(Protocol):
    """Protocol that all specification adapters must implement"""

    def extract_layers(self) -> List[TTTLayerSpec]:
        """Extract layer specifications from the format"""
        ...

    def extract_dataflow(self) -> List[tuple[str, str]]:
        """Extract connections between layers"""
        ...

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract any additional metadata"""
        ...


# Concrete adapter implementations
class HuggingFaceAdapter:
    """Adapter for HuggingFace model configs"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def extract_layers(self) -> List[TTTLayerSpec]:
        layers = []

        # Embedding layer
        layers.append(
            TTTLayerSpec(
                layer_type="embedding",
                layer_id="embedding",
                params={
                    "vocab_size": self.config.get("vocab_size", 32000),
                    "hidden_size": self.config.get("hidden_size", 4096),
                },
                inputs=["input_ids"],
                outputs=["embeddings"],
            )
        )

        # Transformer layers
        num_layers = self.config.get("num_hidden_layers", 32)
        for i in range(num_layers):
            # Attention
            layers.append(
                TTTLayerSpec(
                    layer_type="attention",
                    layer_id=f"layer_{i}_attention",
                    params={
                        "hidden_size": self.config.get("hidden_size", 4096),
                        "num_heads": self.config.get("num_attention_heads", 32),
                        "num_kv_heads": self.config.get(
                            "num_key_value_heads", self.config.get("num_attention_heads", 32)
                        ),
                        "rope_theta": self.config.get("rope_theta", 10000.0),
                    },
                    inputs=[f"layer_{i}_input"],
                    outputs=[f"layer_{i}_attn_output"],
                )
            )

            # FFN
            layers.append(
                TTTLayerSpec(
                    layer_type="ffn",
                    layer_id=f"layer_{i}_ffn",
                    params={
                        "hidden_size": self.config.get("hidden_size", 4096),
                        "intermediate_size": self.config.get("intermediate_size", 11008),
                        "activation": self.config.get("hidden_act", "silu"),
                    },
                    inputs=[f"layer_{i}_attn_output"],
                    outputs=[f"layer_{i}_output"],
                )
            )

        # Output layer
        layers.append(
            TTTLayerSpec(
                layer_type="lm_head",
                layer_id="lm_head",
                params={
                    "hidden_size": self.config.get("hidden_size", 4096),
                    "vocab_size": self.config.get("vocab_size", 32000),
                },
                inputs=[f"layer_{num_layers-1}_output"],
                outputs=["logits"],
            )
        )

        return layers

    def extract_dataflow(self) -> List[tuple[str, str]]:
        """Build sequential dataflow for transformer"""
        connections = []
        layers = self.extract_layers()

        # Connect layers sequentially
        for i in range(len(layers) - 1):
            curr_outputs = layers[i].outputs
            next_inputs = layers[i + 1].inputs

            for out, inp in zip(curr_outputs, next_inputs):
                connections.append((out, inp))

        return connections

    def extract_metadata(self) -> Dict[str, Any]:
        return {
            "model_type": self.config.get("model_type", "unknown"),
            "source": "huggingface",
            "original_config": self.config,
        }


class ONNXAdapter:
    """Adapter for ONNX models"""

    def __init__(self, onnx_model):
        self.model = onnx_model
        self.graph = onnx_model.graph

    def extract_layers(self) -> List[TTTLayerSpec]:
        layers = []

        for node in self.graph.node:
            # Map ONNX ops to TTT layer types
            layer_type = self._map_onnx_op_to_ttt_type(node.op_type)
            if layer_type:
                layers.append(
                    TTTLayerSpec(
                        layer_type=layer_type,
                        layer_id=node.name or f"{node.op_type}_{len(layers)}",
                        params=self._extract_node_params(node),
                        inputs=list(node.input),
                        outputs=list(node.output),
                    )
                )

        return layers

    def _map_onnx_op_to_ttt_type(self, op_type: str) -> Optional[str]:
        """Map ONNX operators to TTT layer types"""
        mapping = {
            "Attention": "attention",
            "MultiHeadAttention": "attention",
            "LayerNormalization": "norm",
            "Add": "residual",
            "MatMul": "linear",
            "Gemm": "linear",
        }
        return mapping.get(op_type)

    def _extract_node_params(self, node) -> Dict[str, Any]:
        """Extract parameters from ONNX node attributes"""
        params = {}
        for attr in node.attribute:
            if attr.type == 1:  # FLOAT
                params[attr.name] = attr.f
            elif attr.type == 2:  # INT
                params[attr.name] = attr.i
            elif attr.type == 3:  # STRING
                params[attr.name] = attr.s.decode("utf-8")
        return params

    def extract_dataflow(self) -> List[tuple[str, str]]:
        """Extract connections from ONNX graph"""
        connections = []

        # Build a map of tensor producers and consumers
        tensor_producers = {}
        tensor_consumers = {}

        for node in self.graph.node:
            for output in node.output:
                tensor_producers[output] = node.name
            for input in node.input:
                if input not in tensor_consumers:
                    tensor_consumers[input] = []
                tensor_consumers[input].append(node.name)

        # Create connections
        for tensor, producer in tensor_producers.items():
            if tensor in tensor_consumers:
                for consumer in tensor_consumers[tensor]:
                    connections.append((tensor, consumer))

        return connections

    def extract_metadata(self) -> Dict[str, Any]:
        return {
            "source": "onnx",
            "ir_version": self.model.ir_version,
            "producer": self.model.producer_name,
            "imports": [imp.domain for imp in self.model.opset_import],
        }


class GGUFAdapter:
    """Adapter for GGUF format models"""

    def __init__(self, gguf_data: Dict[str, Any]):
        self.data = gguf_data
        self.metadata = gguf_data.get("metadata", {})

    def extract_layers(self) -> List[TTTLayerSpec]:
        """Extract layers from GGUF metadata"""
        layers = []

        # GGUF stores architecture info in metadata
        arch = self.metadata.get("general.architecture", "llama")

        # Extract architecture parameters
        hidden_size = self.metadata.get(f"{arch}.embedding_length", 4096)
        num_layers = self.metadata.get(f"{arch}.block_count", 32)
        num_heads = self.metadata.get(f"{arch}.attention.head_count", 32)
        vocab_size = self.metadata.get(f"{arch}.vocab_size", 32000)

        # Build standard transformer architecture
        # (Similar to HuggingFace adapter but reading from GGUF metadata)
        # ... implementation similar to HuggingFaceAdapter

        return layers

    def extract_dataflow(self) -> List[tuple[str, str]]:
        """GGUF assumes standard transformer dataflow"""
        # Similar sequential connection pattern
        return []

    def extract_metadata(self) -> Dict[str, Any]:
        return {
            "source": "gguf",
            "quantization": self.metadata.get("general.quantization_version"),
            "architecture": self.metadata.get("general.architecture"),
            **self.metadata,
        }


# Registry system for managing adapters
class ModelSpecRegistry:
    """Registry for model specification adapters"""

    _adapters: Dict[str, Type[SpecAdapter]] = {}

    @classmethod
    def register(cls, format_name: str, adapter_class: Type[SpecAdapter]):
        """Register a new adapter for a format"""
        cls._adapters[format_name] = adapter_class

    @classmethod
    def get_adapter(cls, format_name: str, data: Any) -> SpecAdapter:
        """Get an adapter instance for the given format"""
        if format_name not in cls._adapters:
            raise ValueError(f"Unknown format: {format_name}. " f"Available: {list(cls._adapters.keys())}")

        adapter_class = cls._adapters[format_name]
        return adapter_class(data)

    @classmethod
    def convert_to_ttt_spec(cls, format_name: str, data: Any) -> TTTModelSpec:
        """Convert any supported format to TTT spec"""
        adapter = cls.get_adapter(format_name, data)

        return TTTModelSpec(
            name=f"model_from_{format_name}",
            layers=adapter.extract_layers(),
            dataflow=adapter.extract_dataflow(),
            metadata=adapter.extract_metadata(),
        )


# Register built-in adapters
ModelSpecRegistry.register("huggingface", HuggingFaceAdapter)
ModelSpecRegistry.register("onnx", ONNXAdapter)
ModelSpecRegistry.register("gguf", GGUFAdapter)


# Factory for building models from specs
class TTTModelFactory:
    """Factory for creating TTT models from specifications"""

    @staticmethod
    def from_spec(spec: TTTModelSpec, device=None):
        """Build a model from TTT specification"""
        from tt_transformers_v2 import attention, embeddings, ffn

        modules = {}

        for layer_spec in spec.layers:
            if layer_spec.layer_type == "embedding":
                module = embeddings.TokenEmbedding(
                    vocab_size=layer_spec.params["vocab_size"],
                    embedding_dim=layer_spec.params["hidden_size"],
                    device=device,
                )
            elif layer_spec.layer_type == "attention":
                module = attention.MultiHeadAttention(
                    hidden_dim=layer_spec.params["hidden_size"],
                    num_heads=layer_spec.params["num_heads"],
                    num_kv_heads=layer_spec.params.get("num_kv_heads"),
                    device=device,
                )
            elif layer_spec.layer_type == "ffn":
                module = ffn.SwiGLU(
                    hidden_dim=layer_spec.params["hidden_size"],
                    intermediate_dim=layer_spec.params["intermediate_size"],
                    device=device,
                )
            # ... handle other layer types

            modules[layer_spec.layer_id] = module

        # Return a model wrapper that respects the dataflow
        return TTTModel(modules, spec.dataflow)

    @staticmethod
    def from_format(format_name: str, data: Any, device=None):
        """Convenience method to build from any format"""
        spec = ModelSpecRegistry.convert_to_ttt_spec(format_name, data)
        return TTTModelFactory.from_spec(spec, device)


# Example usage
def demo_usage():
    """Demonstrate the standard-agnostic design"""

    # From HuggingFace config
    hf_config = {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 11008,
        "vocab_size": 32000,
    }

    model = TTTModelFactory.from_format("huggingface", hf_config, device="ttnn")

    # From ONNX (would work with real ONNX model)
    # onnx_model = onnx.load("model.onnx")
    # model = TTTModelFactory.from_format("onnx", onnx_model, device="ttnn")

    # Register custom format
    class MyCustomAdapter:
        def __init__(self, data):
            self.data = data

        def extract_layers(self):
            # Custom extraction logic
            pass

        def extract_dataflow(self):
            # Custom dataflow logic
            pass

        def extract_metadata(self):
            return {"source": "custom"}

    ModelSpecRegistry.register("myformat", MyCustomAdapter)
    # model = TTTModelFactory.from_format("myformat", my_data, device="ttnn")


class TTTModel:
    """Simple model wrapper that respects dataflow"""

    def __init__(self, modules: Dict[str, Any], dataflow: List[tuple[str, str]]):
        self.modules = modules
        self.dataflow = dataflow

    def forward(self, inputs):
        # Execute modules according to dataflow
        # This is a simplified example
        pass


if __name__ == "__main__":
    demo_usage()
