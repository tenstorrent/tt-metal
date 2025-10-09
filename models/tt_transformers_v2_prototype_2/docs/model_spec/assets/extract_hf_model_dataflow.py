"""
Script to extract actual dataflow from HuggingFace models by intercepting forward() calls.

This approach uses Python's introspection to trace actual method calls during forward pass,
giving us the true execution flow including dynamic branching.
"""

import functools
import inspect
import json
from typing import Any, Dict

import torch
import torch.nn as nn


class ForwardCallTracer:
    """
    Traces forward() method calls throughout model execution.
    This captures the actual execution flow including dynamic paths.
    """

    def __init__(self):
        self.call_stack = []
        self.execution_trace = []
        self.module_registry = {}
        self.original_forwards = {}
        self.enabled = False

    def _wrap_forward(self, module: nn.Module, module_name: str):
        """Wrap a module's forward method to trace calls."""
        original_forward = module.forward

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            if not self.enabled:
                return original_forward(*args, **kwargs)

            # Record entry
            call_info = {
                "module_name": module_name,
                "module_type": module.__class__.__name__,
                "depth": len(self.call_stack),
                "order": len(self.execution_trace),
            }

            # Analyze inputs
            if args and isinstance(args[0], torch.Tensor):
                call_info["input_shape"] = list(args[0].shape)
                call_info["input_dtype"] = str(args[0].dtype)

            self.call_stack.append(module_name)
            self.execution_trace.append({**call_info, "event": "enter"})

            try:
                # Call original forward
                output = original_forward(*args, **kwargs)

                # Analyze outputs
                if isinstance(output, torch.Tensor):
                    call_info["output_shape"] = list(output.shape)
                    call_info["output_dtype"] = str(output.dtype)
                elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                    call_info["output_shape"] = list(output[0].shape)
                    call_info["output_dtype"] = str(output[0].dtype)
                    call_info["output_tuple_len"] = len(output)

                self.execution_trace.append({**call_info, "event": "exit"})

                return output

            finally:
                self.call_stack.pop()

        return wrapped_forward

    def register_model(self, model: nn.Module):
        """Register all modules in a model for tracing."""
        for name, module in model.named_modules():
            if module not in self.module_registry:
                self.module_registry[module] = name
                self.original_forwards[module] = module.forward
                module.forward = self._wrap_forward(module, name)

    def start_tracing(self):
        """Enable tracing."""
        self.enabled = True
        self.execution_trace = []
        self.call_stack = []

    def stop_tracing(self):
        """Disable tracing."""
        self.enabled = False

    def restore_original_forwards(self):
        """Restore original forward methods."""
        for module, original_forward in self.original_forwards.items():
            module.forward = original_forward

    def get_dataflow(self) -> Dict[str, Any]:
        """Extract dataflow information from the trace."""
        # Build execution order
        execution_order = []
        module_calls = {}
        call_graph = {}

        for event in self.execution_trace:
            if event["event"] == "enter":
                module_name = event["module_name"]
                execution_order.append(module_name)

                # Track module call information
                if module_name not in module_calls:
                    module_calls[module_name] = {
                        "type": event["module_type"],
                        "call_count": 0,
                        "input_shapes": [],
                        "output_shapes": [],
                    }

                module_calls[module_name]["call_count"] += 1

                if "input_shape" in event:
                    shape = tuple(event["input_shape"])
                    if shape not in module_calls[module_name]["input_shapes"]:
                        module_calls[module_name]["input_shapes"].append(shape)

                # Build call graph
                if len(self.call_stack) > 0 and event["depth"] > 0:
                    parent = self.call_stack[event["depth"] - 1]
                    if parent not in call_graph:
                        call_graph[parent] = []
                    if module_name not in call_graph[parent]:
                        call_graph[parent].append(module_name)

            elif event["event"] == "exit":
                module_name = event["module_name"]
                if "output_shape" in event and module_name in module_calls:
                    shape = tuple(event["output_shape"])
                    if shape not in module_calls[module_name]["output_shapes"]:
                        module_calls[module_name]["output_shapes"].append(shape)

        # Remove duplicates while preserving order
        seen = set()
        unique_execution_order = []
        for module in execution_order:
            if module not in seen:
                seen.add(module)
                unique_execution_order.append(module)

        return {
            "execution_order": unique_execution_order,
            "module_calls": module_calls,
            "call_graph": call_graph,
            "total_forward_calls": len([e for e in self.execution_trace if e["event"] == "enter"]),
        }


def extract_model_dataflow_with_trace(model, tokenizer, text_input: str = "Hello world"):
    """
    Extract dataflow by actually running the model with tracing.
    """
    tracer = ForwardCallTracer()

    # Register model for tracing
    tracer.register_model(model)

    # Prepare inputs
    inputs = tokenizer(text_input, return_tensors="pt")

    # Trace forward pass
    tracer.start_tracing()
    with torch.no_grad():
        outputs = model(**inputs)
    tracer.stop_tracing()

    # Get dataflow
    dataflow = tracer.get_dataflow()

    # Restore original forwards
    tracer.restore_original_forwards()

    return dataflow


def analyze_model_graph(model) -> Dict[str, Any]:
    """
    Analyze model structure without running it.
    Useful for understanding potential dataflow paths.
    """

    def get_forward_signature(module):
        """Extract forward method signature."""
        try:
            sig = inspect.signature(module.forward)
            params = []
            for name, param in sig.parameters.items():
                if name != "self":
                    params.append(
                        {
                            "name": name,
                            "default": str(param.default) if param.default != param.empty else None,
                            "annotation": str(param.annotation) if param.annotation != param.empty else None,
                        }
                    )
            return params
        except:
            return None

    module_info = {}
    hierarchy = {}

    def analyze_module(name, module, parent=None):
        # Get module information
        info = {
            "type": module.__class__.__name__,
            "parameters": {n: list(p.shape) for n, p in module.named_parameters(recurse=False)},
            "forward_signature": get_forward_signature(module),
            "has_children": len(list(module.children())) > 0,
        }

        # Special handling for specific layer types
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            info["dimensions"] = f"{module.in_features} -> {module.out_features}"
        elif hasattr(module, "num_embeddings") and hasattr(module, "embedding_dim"):
            info["dimensions"] = f"vocab={module.num_embeddings}, dim={module.embedding_dim}"

        module_info[name] = info

        # Build hierarchy
        if parent is not None:
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(name)

        # Recurse to children
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            analyze_module(full_name, child_module, name)

    # Start analysis from root
    analyze_module("", model)

    return {
        "modules": module_info,
        "hierarchy": hierarchy,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "total_modules": len(module_info),
    }


def generate_dataflow_visualization(dataflow: Dict[str, Any]) -> str:
    """
    Generate a simple text visualization of the dataflow.
    """
    lines = []
    lines.append("Model Dataflow Visualization")
    lines.append("=" * 50)

    # Execution order
    lines.append("\nExecution Order (first 20 modules):")
    for i, module in enumerate(dataflow["execution_order"][:20]):
        module_info = dataflow["module_calls"][module]
        lines.append(f"{i+1:3d}. {module} ({module_info['type']}) - {module_info['call_count']} calls")

    # Most called modules
    lines.append("\nMost Frequently Called Modules:")
    sorted_modules = sorted(dataflow["module_calls"].items(), key=lambda x: x[1]["call_count"], reverse=True)
    for module, info in sorted_modules[:10]:
        lines.append(f"  {module}: {info['call_count']} calls")

    # Call graph sample
    lines.append("\nCall Graph Sample (first 10 parent->children):")
    for i, (parent, children) in enumerate(list(dataflow["call_graph"].items())[:10]):
        lines.append(f"  {parent} ->")
        for child in children[:3]:  # Show first 3 children
            lines.append(f"    └─ {child}")
        if len(children) > 3:
            lines.append(f"    └─ ... and {len(children) - 3} more")

    lines.append(f"\nTotal forward() calls: {dataflow['total_forward_calls']}")

    return "\n".join(lines)


# Example usage for different model types
def demo_extraction():
    """Demonstrate extraction for different model architectures."""

    from transformers import AutoModel, AutoTokenizer

    print("Dataflow Extraction Demo")
    print("=" * 60)

    # Example 1: BERT model
    print("\n1. Extracting dataflow from BERT...")
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Method 1: Trace actual execution
    dataflow = extract_model_dataflow_with_trace(model, tokenizer, "Hello world")
    visualization = generate_dataflow_visualization(dataflow)
    print(visualization)

    # Method 2: Static analysis
    print("\n2. Static model analysis...")
    static_analysis = analyze_model_graph(model)
    print(f"Total modules: {static_analysis['total_modules']}")
    print(f"Total parameters: {static_analysis['total_parameters']:,}")

    # Export results
    with open("bert_dataflow.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "dataflow": dataflow,
                "static_analysis": {
                    "total_modules": static_analysis["total_modules"],
                    "total_parameters": static_analysis["total_parameters"],
                },
            },
            f,
            indent=2,
            default=str,
        )

    print("\nDataflow exported to bert_dataflow.json")


if __name__ == "__main__":
    demo_extraction()
