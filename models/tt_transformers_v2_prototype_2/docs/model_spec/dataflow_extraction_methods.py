"""
Methods for extracting dataflow information from HuggingFace models.

This module provides several approaches to extract computational graphs,
forward pass traces, and dataflow information from transformer models.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch
from transformers import PreTrainedModel


class ModelDataflowTracer:
    """Extract dataflow information from HuggingFace models using forward hooks."""

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.execution_order = []
        self.layer_outputs = OrderedDict()
        self.layer_inputs = OrderedDict()
        self.hooks = []
        self.dataflow = {}

    def _create_forward_hook(self, name: str):
        """Create a forward hook for a specific layer."""

        def hook(module, input, output):
            self.execution_order.append(name)

            # Store input/output shapes and types
            input_info = self._extract_tensor_info(input)
            output_info = self._extract_tensor_info(output)

            self.layer_inputs[name] = input_info
            self.layer_outputs[name] = output_info

            # Track dataflow connections
            if len(self.execution_order) > 1:
                prev_layer = self.execution_order[-2]
                if prev_layer not in self.dataflow:
                    self.dataflow[prev_layer] = []
                self.dataflow[prev_layer].append(name)

        return hook

    def _extract_tensor_info(self, tensor_data):
        """Extract information about tensors (handling various input types)."""
        if isinstance(tensor_data, torch.Tensor):
            return {
                "type": "tensor",
                "shape": list(tensor_data.shape),
                "dtype": str(tensor_data.dtype),
                "device": str(tensor_data.device),
            }
        elif isinstance(tensor_data, tuple):
            return {"type": "tuple", "elements": [self._extract_tensor_info(t) for t in tensor_data]}
        elif isinstance(tensor_data, dict):
            return {"type": "dict", "elements": {k: self._extract_tensor_info(v) for k, v in tensor_data.items()}}
        else:
            return {"type": str(type(tensor_data)), "value": str(tensor_data)}

    def register_hooks(self):
        """Register forward hooks on all modules."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(self._create_forward_hook(name))
                self.hooks.append(hook)

    def trace_forward_pass(self, inputs: Dict[str, torch.Tensor]):
        """Trace a forward pass through the model."""
        # Clear previous trace
        self.execution_order = []
        self.layer_outputs = OrderedDict()
        self.layer_inputs = OrderedDict()
        self.dataflow = {}

        # Register hooks if not already done
        if not self.hooks:
            self.register_hooks()

        # Run forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        return {
            "execution_order": self.execution_order,
            "layer_inputs": self.layer_inputs,
            "layer_outputs": self.layer_outputs,
            "dataflow_graph": self.dataflow,
            "model_output": self._extract_tensor_info(outputs),
        }

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_dataflow_summary(self) -> Dict[str, Any]:
        """Generate a summary of the model's dataflow."""
        summary = {
            "total_layers": len(self.execution_order),
            "unique_layers": len(set(self.execution_order)),
            "execution_path": self.execution_order,
            "layer_connections": self.dataflow,
            "layer_io_shapes": {
                name: {"input": self.layer_inputs.get(name), "output": self.layer_outputs.get(name)}
                for name in self.execution_order
            },
        }
        return summary


class TorchScriptTracer:
    """Extract dataflow using TorchScript tracing."""

    @staticmethod
    def trace_model(model: PreTrainedModel, example_inputs: Dict[str, torch.Tensor]):
        """Create a TorchScript trace of the model."""
        model.eval()

        # Convert inputs to tuple for tracing
        input_values = tuple(example_inputs.values())

        try:
            traced_model = torch.jit.trace(model, input_values)
            return traced_model
        except Exception as e:
            print(f"TorchScript tracing failed: {e}")
            return None

    @staticmethod
    def get_graph_operations(traced_model):
        """Extract operations from TorchScript graph."""
        if traced_model is None:
            return []

        graph = traced_model.graph
        operations = []

        for node in graph.nodes():
            op = {
                "kind": node.kind(),
                "inputs": [str(i) for i in node.inputs()],
                "outputs": [str(o) for o in node.outputs()],
                "attributes": {k: node[k] for k in node.attributeNames()},
            }
            operations.append(op)

        return operations


class ModelArchitectureAnalyzer:
    """Analyze model architecture to infer dataflow patterns."""

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.architecture = {}

    def analyze_structure(self):
        """Analyze the model structure to understand potential dataflow."""

        def _analyze_module(module, prefix=""):
            module_info = {"type": module.__class__.__name__, "children": {}}

            # Extract parameters
            params = {}
            for name, param in module.named_parameters(recurse=False):
                params[name] = {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "requires_grad": param.requires_grad,
                }
            module_info["parameters"] = params

            # Recursively analyze children
            for name, child in module.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                module_info["children"][name] = _analyze_module(child, child_prefix)

            return module_info

        self.architecture = _analyze_module(self.model)
        return self.architecture

    def infer_dataflow_from_architecture(self):
        """Infer likely dataflow patterns based on architecture."""
        # This would contain model-specific patterns
        # For transformer models, we know the typical flow:
        # embedding -> layers -> norm -> lm_head

        dataflow_patterns = {
            "transformer_text": ["embeddings/embed_tokens", "encoder/layers/*", "encoder/norm", "lm_head"],
            "transformer_vision": ["patch_embed", "pos_embed", "blocks/*", "norm", "head"],
            "multimodal": ["visual/*", "language_model/*", "merger", "lm_head"],
        }

        return dataflow_patterns


def extract_qwen_dataflow(model):
    """
    Extract dataflow specifically for Qwen2.5 VL model.
    This function demonstrates how to trace the specific dataflow for the model you mentioned.
    """
    tracer = ModelDataflowTracer(model)

    # Create dummy inputs for the model
    batch_size = 1
    seq_length = 10
    image_size = 224

    dummy_inputs = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long),
        "pixel_values": torch.randn(batch_size, 3, image_size, image_size),
    }

    # Trace the forward pass
    trace_results = tracer.trace_forward_pass(dummy_inputs)

    # The dataflow for Qwen2.5 VL would typically be:
    # 1. Visual processing:
    #    - patch_embed -> rotary_pos_emb -> blocks[0-31] -> merger
    # 2. Language processing:
    #    - embed_tokens -> layers[0-35] (with self_attn and mlp) -> norm
    # 3. Final output:
    #    - lm_head

    # Clean up
    tracer.remove_hooks()

    return trace_results


# Example usage
if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    # Load a small model for demonstration
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Method 1: Using forward hooks
    tracer = ModelDataflowTracer(model)
    inputs = tokenizer("Hello world", return_tensors="pt")
    trace_results = tracer.trace_forward_pass(inputs)

    print("Execution Order:", trace_results["execution_order"][:10])  # First 10 layers
    print("\nDataflow Graph Sample:", list(trace_results["dataflow_graph"].items())[:5])

    tracer.remove_hooks()

    # Method 2: Architecture analysis
    analyzer = ModelArchitectureAnalyzer(model)
    architecture = analyzer.analyze_structure()
    print("\nModel Architecture (top level):", list(architecture["children"].keys()))

    # Method 3: TorchScript (if supported)
    traced = TorchScriptTracer.trace_model(model, inputs)
    if traced:
        ops = TorchScriptTracer.get_graph_operations(traced)
        print(f"\nTorchScript found {len(ops)} operations")
