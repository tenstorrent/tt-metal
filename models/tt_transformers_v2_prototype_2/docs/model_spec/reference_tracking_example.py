"""
Reference Tracking for Module Debugging

This demonstrates how associating TTT modules with reference implementations
enables powerful debugging workflows.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

# =============================================================================
# Module Specs with Reference Tracking
# =============================================================================


@dataclass(frozen=True)
class ModuleSpec:
    """Base class for all module specifications"""

    module_id: str

    # Reference tracking
    reference_path: Optional[str] = None  # e.g., "model.layers.0.self_attn"
    reference_type: Optional[str] = None  # e.g., "huggingface", "pytorch", "custom"

    def validate(self):
        """Validate spec parameters"""


@dataclass(frozen=True)
class AttentionSpec(ModuleSpec):
    """Fixed public API for attention"""

    hidden_dim: int
    num_heads: int

    def validate(self):
        assert (
            self.hidden_dim % self.num_heads == 0
        ), f"hidden_dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}"


# =============================================================================
# Reference Manager for Debugging
# =============================================================================


class ReferenceManager:
    """Manages reference implementations for debugging"""

    def __init__(self):
        self.references: Dict[str, Any] = {}
        self.extractors: Dict[str, Callable] = {
            "huggingface": self._extract_hf_module,
            "pytorch": self._extract_pytorch_module,
            "custom": self._extract_custom_module,
        }

    def register_model(self, model: Any, model_type: str = "huggingface"):
        """Register a reference model"""
        self.reference_model = model
        self.model_type = model_type

    def _extract_hf_module(self, path: str):
        """Extract module from HuggingFace model"""
        # Navigate through model structure
        # e.g., "model.layers.0.self_attn" -> model.model.layers[0].self_attn
        parts = path.split(".")
        module = self.reference_model

        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)

        return module

    def _extract_pytorch_module(self, path: str):
        """Extract module from PyTorch model"""
        # Similar navigation logic
        return self._extract_hf_module(path)  # Often same structure

    def _extract_custom_module(self, path: str):
        """Extract from custom reference implementation"""
        # User-provided extraction logic
        if path in self.references:
            return self.references[path]
        raise KeyError(f"No custom reference found for {path}")

    def get_reference_module(self, spec: ModuleSpec):
        """Get reference module for a spec"""
        if spec.reference_path and spec.reference_type:
            extractor = self.extractors.get(spec.reference_type)
            if extractor:
                return extractor(spec.reference_path)
        return None

    def register_custom_reference(self, path: str, module: Callable):
        """Register a custom reference implementation"""
        self.references[path] = module


# =============================================================================
# TTT Module with Reference Support
# =============================================================================


class TTTModuleWithReference:
    """Base class for TTT modules with reference tracking"""

    def __init__(self, spec: ModuleSpec, reference_manager: Optional[ReferenceManager] = None):
        self.spec = spec
        self.reference_manager = reference_manager
        self.reference_module = None

        if reference_manager and spec.reference_path:
            self.reference_module = reference_manager.get_reference_module(spec)

    def compare_with_reference(self, input_data, rtol=1e-3, atol=1e-5, verbose=True):
        """Compare module output with reference implementation"""
        if not self.reference_module:
            return None

        # Get outputs
        with torch.no_grad():
            our_output = self.forward(input_data)
            ref_output = self._run_reference(input_data)

        # Compare
        comparison = {
            "max_abs_diff": torch.max(torch.abs(our_output - ref_output)).item(),
            "mean_abs_diff": torch.mean(torch.abs(our_output - ref_output)).item(),
            "relative_error": torch.mean(torch.abs((our_output - ref_output) / (ref_output + 1e-8))).item(),
            "allclose": torch.allclose(our_output, ref_output, rtol=rtol, atol=atol),
        }

        if verbose:
            print(f"Comparison for {self.spec.module_id}:")
            print(f"  Reference: {self.spec.reference_path}")
            print(f"  Max absolute difference: {comparison['max_abs_diff']:.2e}")
            print(f"  Mean absolute difference: {comparison['mean_abs_diff']:.2e}")
            print(f"  Relative error: {comparison['relative_error']:.2%}")
            print(f"  All close (rtol={rtol}, atol={atol}): {comparison['allclose']}")

        return comparison

    def _run_reference(self, input_data):
        """Run reference module (to be overridden by subclasses)"""
        raise NotImplementedError


class MultiHeadAttentionWithReference(TTTModuleWithReference):
    """Attention implementation with reference tracking"""

    def __init__(self, spec: AttentionSpec, device=None, reference_manager=None):
        super().__init__(spec, reference_manager)
        self.device = device

        # Create TTT implementation
        self._create_ttnn_ops()

    def _create_ttnn_ops(self):
        """Create TTNN operations based on spec"""
        # Simplified for example
        self.hidden_dim = self.spec.hidden_dim
        self.num_heads = self.spec.num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        # TTNN ops would go here
        # self.qkv_proj = ttnn.Linear(...)
        # etc.

    def forward(self, hidden_states, attention_mask=None):
        """TTT forward implementation"""
        # Simplified attention computation
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection (simplified)
        qkv = hidden_states @ torch.randn(self.hidden_dim, 3 * self.hidden_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention (simplified)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attention_mask is not None:
            scores += attention_mask

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return output

    def _run_reference(self, input_data):
        """Run reference HuggingFace/PyTorch module"""
        if self.spec.reference_type == "huggingface":
            # HuggingFace models expect specific input format
            return self.reference_module(hidden_states=input_data, attention_mask=None, output_attentions=False)[
                0
            ]  # Return just hidden states
        else:
            # Direct forward for PyTorch modules
            return self.reference_module(input_data)


# =============================================================================
# Model Building with Reference Tracking
# =============================================================================


def build_model_with_references(model_spec: Dict[str, Any], reference_model=None) -> Dict[str, Any]:
    """Build TTT model with reference tracking"""

    # Create reference manager
    ref_manager = ReferenceManager()
    if reference_model:
        ref_manager.register_model(reference_model, model_type="huggingface")

    # Build modules with specs
    modules = {}

    for layer_idx in range(model_spec["num_layers"]):
        # Create attention with reference
        attention_spec = AttentionSpec(
            module_id=f"layer_{layer_idx}_attention",
            hidden_dim=model_spec["hidden_dim"],
            num_heads=model_spec["num_heads"],
            reference_path=f"model.layers.{layer_idx}.self_attn",
            reference_type="huggingface",
        )

        modules[f"layer_{layer_idx}_attention"] = MultiHeadAttentionWithReference(
            spec=attention_spec, device="ttnn:0", reference_manager=ref_manager
        )

        # Add FFN, norm, etc. with similar pattern

    return modules


# =============================================================================
# Debugging Workflow Example
# =============================================================================


def debug_accuracy_issue():
    """Example debugging workflow with reference tracking"""

    print("=== Debugging Accuracy with Reference Tracking ===\n")

    # Load reference model
    from transformers import AutoModel

    reference_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Build TTT model with references
    model_spec = {"num_layers": 32, "hidden_dim": 4096, "num_heads": 32}

    ttt_modules = build_model_with_references(model_spec, reference_model)

    # Test specific layer
    layer_15_attention = ttt_modules["layer_15_attention"]

    # Create test input
    test_input = torch.randn(1, 128, 4096)  # [batch, seq, hidden]

    # Compare with reference
    print(f"Testing {layer_15_attention.spec.module_id}...")
    comparison = layer_15_attention.compare_with_reference(test_input, verbose=True)

    # If accuracy issue found
    if comparison and not comparison["allclose"]:
        print("\n⚠️  Accuracy issue detected!")
        print("Debugging steps:")

        # 1. Check intermediate values
        print("\n1. Checking intermediate values...")
        # Would check QKV projections, attention scores, etc.

        # 2. Try different precision
        print("\n2. Testing with higher precision...")
        # layer_15_attention.apply_config(compute_dtype='float32')

        # 3. Check for numerical instabilities
        print("\n3. Analyzing numerical stability...")
        # Check for overflow, underflow, etc.

    return comparison


# =============================================================================
# Advanced: Custom Reference Implementation
# =============================================================================


def register_custom_reference():
    """Example of registering custom reference implementation"""

    ref_manager = ReferenceManager()

    # Register custom attention implementation
    def custom_attention_reference(hidden_states):
        """Custom high-precision reference implementation"""
        # Implementation with double precision for accuracy testing
        hidden_states_fp64 = hidden_states.double()

        # ... compute attention in float64 ...

        return output.float()

    ref_manager.register_custom_reference("custom_attention_fp64", custom_attention_reference)

    # Create module with custom reference
    spec = AttentionSpec(
        module_id="test_attention",
        hidden_dim=4096,
        num_heads=32,
        reference_path="custom_attention_fp64",
        reference_type="custom",
    )

    attention = MultiHeadAttentionWithReference(spec=spec, reference_manager=ref_manager)

    return attention


# =============================================================================
# Benefits of This Approach
# =============================================================================

"""
1. **Automated Accuracy Testing**: Every module can be compared against reference
2. **Debugging Support**: When issues arise, reference is already available
3. **Progressive Validation**: Can validate module-by-module
4. **Multiple Reference Types**: Support HF, PyTorch, custom implementations
5. **Optional System**: References only needed when debugging

Example workflow:
1. Build model without references (fast iteration)
2. If accuracy issues, add reference tracking
3. Binary search to find problematic module
4. Use reference to debug specific module
5. Remove references for production
"""

if __name__ == "__main__":
    # Demo the debugging workflow
    debug_accuracy_issue()

    # Demo custom reference
    custom_attention = register_custom_reference()
    print(f"\nCreated module with custom reference: {custom_attention.spec.reference_type}")
