#!/usr/bin/env python3
"""Pure TTNN implementation of Qwen model (DeepSeek-R1-Distill-Qwen-1.5B)"""

import math
import os
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn

# ============================================================================
# Validation Decorator System
# ============================================================================


@dataclass
class ValidationResult:
    """Results from a single validation run"""

    function_name: str
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time_impl: float = 0.0
    execution_time_ref: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ValidationRegistry:
    """Global registry for validation results"""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.enabled = True

    def add_result(self, result: ValidationResult):
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all validations"""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0.0,
            "avg_speedup": sum(
                r.execution_time_ref / r.execution_time_impl for r in self.results if r.execution_time_impl > 0
            )
            / len(self.results)
            if self.results
            else 0.0,
        }

    def print_report(self):
        """Print detailed validation report"""
        summary = self.get_summary()
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        print(f"Total validations: {summary['total']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Average speedup: {summary['avg_speedup']:.2f}x")
        print()

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} - {result.function_name}")
            print(
                f"  Execution time: impl={result.execution_time_impl*1000:.2f}ms, ref={result.execution_time_ref*1000:.2f}ms"
            )

            if result.metrics:
                print(f"  Metrics:")
                for metric, value in result.metrics.items():
                    print(f"    {metric}: {value:.6f}")

            if result.errors:
                print(f"  Errors:")
                for error in result.errors:
                    print(f"    - {error}")
            print()
        print("=" * 80 + "\n")


# Global validation registry
_validation_registry = ValidationRegistry()


def validate_against(
    reference_fn: Callable,
    input_map: Optional[Callable] = None,
    output_map_impl: Optional[Callable] = None,
    output_map_ref: Optional[Callable] = None,
    metrics: Optional[Dict[str, Callable]] = None,
    tolerances: Optional[Dict[str, float]] = None,
    performance_metrics: bool = True,
    enabled: bool = True,
    match_signature: bool = False,
    auto_convert_outputs: bool = False,
):
    """
    Decorator to validate a function against a reference implementation.

    Args:
        reference_fn: Reference function to compare against
        input_map: Maps decorated function inputs to reference function inputs
                   Signature: (args, kwargs) -> (ref_args, ref_kwargs)
                   If None, inputs are passed as-is
        output_map_impl: Maps decorated function output for comparison
                         Signature: (output) -> comparable_output
                         If None, output is used as-is
        output_map_ref: Maps reference function output for comparison
                        Signature: (output) -> comparable_output
                        If None, output is used as-is
        metrics: Dictionary of metric_name -> metric_function(impl_out, ref_out) -> float
                 Default metrics: max_abs_error, mean_abs_error, cosine_similarity
        tolerances: Dictionary of metric_name -> max_acceptable_value
                    Validation fails if any metric exceeds its tolerance
        performance_metrics: Whether to collect execution time metrics
        enabled: Whether validation is enabled (can disable globally via registry)
        match_signature: If True, reference_fn has the same signature as the decorated
                        function and will be called with identical args/kwargs.
                        This allows using wrapper functions without complex input_map.
        auto_convert_outputs: If True, automatically converts TTNN tensors to torch tensors
                             for comparison. Applies to both impl and ref outputs.
                             Useful with match_signature when both return TTNN.

    Examples:
        # Pattern 1: Wrapper with same signature + auto_convert (cleanest!)
        @validate_against(
            reference_fn=lambda self, x: self._reference_impl(x),
            match_signature=True,
            auto_convert_outputs=True,  # No output_map_impl needed!
            tolerances={'max_abs_error': 1e-3}
        )
        def __call__(self, x):
            return ttnn.matmul(x, self.weight)  # Returns TTNN

        # Pattern 2: Wrapper with same signature (explicit mapping)
        @validate_against(
            reference_fn=lambda self, x: self._reference_impl(x),
            match_signature=True,
            output_map_impl=lambda x: ttnn.to_torch(x).squeeze(0),
            tolerances={'max_abs_error': 1e-3}
        )
        def __call__(self, x):
            return self.forward(x)

        # Pattern 3: Different signature with mappings
        @validate_against(
            reference_fn=torch.nn.functional.rms_norm,
            input_map=lambda args, kwargs: (
                (ttnn.to_torch(args[1]).squeeze(),),
                {'eps': args[0].eps}
            ),
            output_map_impl=lambda x: ttnn.to_torch(x).squeeze(),
            metrics={'max_error': lambda impl, ref: (impl - ref).abs().max().item()},
            tolerances={'max_error': 1e-3}
        )
        def __call__(self, x):
            return self.forward(x)
    """

    # Default metrics
    default_metrics = {
        "max_abs_error": lambda impl, ref: (impl - ref).abs().max().item() if torch.is_tensor(impl) else float("inf"),
        "mean_abs_error": lambda impl, ref: (impl - ref).abs().mean().item() if torch.is_tensor(impl) else float("inf"),
        "cosine_similarity": lambda impl, ref: torch.nn.functional.cosine_similarity(
            impl.flatten(), ref.flatten(), dim=0
        ).item()
        if torch.is_tensor(impl) and torch.is_tensor(ref)
        else 0.0,
    }

    if metrics:
        default_metrics.update(metrics)

    metrics_to_use = default_metrics
    tolerances = tolerances or {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if validation is enabled
            if not enabled or not _validation_registry.enabled:
                return func(*args, **kwargs)

            # Execute implementation
            start_time = time.perf_counter()
            impl_output = func(*args, **kwargs)
            impl_time = time.perf_counter() - start_time

            # Map inputs for reference function
            if match_signature:
                # Reference function has same signature, call with same args/kwargs
                ref_args, ref_kwargs = args, kwargs
            elif input_map:
                # Use custom input mapping
                ref_args, ref_kwargs = input_map(args, kwargs)
            else:
                # Pass through as-is
                ref_args, ref_kwargs = args, kwargs

            # Execute reference
            try:
                start_time = time.perf_counter()
                ref_output = reference_fn(*ref_args, **ref_kwargs)
                ref_time = time.perf_counter() - start_time
            except Exception as e:
                # If reference fails, just return impl output and log error
                result = ValidationResult(
                    function_name=f"{func.__module__}.{func.__qualname__}",
                    passed=False,
                    errors=[f"Reference execution failed: {str(e)}"],
                    execution_time_impl=impl_time,
                )
                _validation_registry.add_result(result)
                return impl_output

            # Map outputs for comparison
            try:
                if auto_convert_outputs:
                    # Auto-convert TTNN tensors to torch for comparison
                    def auto_convert(x):
                        """Auto-convert TTNN to torch, handling common cases"""
                        # Check if it's a TTNN tensor (has to_torch method or is from ttnn module)
                        if hasattr(x, "__module__") and x.__module__ and "ttnn" in x.__module__:
                            # It's a TTNN tensor, convert to torch
                            converted = ttnn.to_torch(x)
                            # Remove batch dimensions commonly used in TTNN
                            while converted.dim() > 0 and converted.shape[0] == 1:
                                converted = converted.squeeze(0)
                            return converted
                        return x

                    impl_comparable = auto_convert(impl_output)
                    ref_comparable = auto_convert(ref_output)
                else:
                    # Use explicit mapping functions
                    impl_comparable = output_map_impl(impl_output) if output_map_impl else impl_output
                    ref_comparable = output_map_ref(ref_output) if output_map_ref else ref_output
            except Exception as e:
                result = ValidationResult(
                    function_name=f"{func.__module__}.{func.__qualname__}",
                    passed=False,
                    errors=[f"Output mapping failed: {str(e)}"],
                    execution_time_impl=impl_time,
                    execution_time_ref=ref_time,
                )
                _validation_registry.add_result(result)
                return impl_output

            # Compute metrics
            computed_metrics = {}
            errors = []
            passed = True

            for metric_name, metric_fn in metrics_to_use.items():
                try:
                    value = metric_fn(impl_comparable, ref_comparable)
                    computed_metrics[metric_name] = value

                    # Check tolerance
                    if metric_name in tolerances:
                        if value > tolerances[metric_name]:
                            passed = False
                            errors.append(f"{metric_name}={value:.6e} exceeds tolerance {tolerances[metric_name]:.6e}")
                except Exception as e:
                    errors.append(f"Metric {metric_name} failed: {str(e)}")
                    passed = False

            # Record results
            result = ValidationResult(
                function_name=f"{func.__module__}.{func.__qualname__}",
                passed=passed,
                metrics=computed_metrics,
                errors=errors,
                execution_time_impl=impl_time,
                execution_time_ref=ref_time,
            )
            _validation_registry.add_result(result)

            return impl_output

        return wrapper

    return decorator


def get_validation_registry() -> ValidationRegistry:
    """Get the global validation registry"""
    return _validation_registry


def enable_validation(enabled: bool = True):
    """Enable or disable validation globally"""
    _validation_registry.enabled = enabled


def clear_validation_results():
    """Clear all validation results"""
    _validation_registry.results.clear()


# ============================================================================
# Model Implementation by claude-4.5-sonnet thinking
# ============================================================================


class RMSNorm:
    """RMS Normalization in TTNN"""

    def __init__(self, weight: torch.Tensor, eps: float, device):
        self.eps = eps
        self.weight = ttnn.from_torch(
            weight.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def __call__(self, x):
        # x shape: [1, seq_len, hidden_size]
        # Compute RMS: sqrt(mean(x^2) + eps)
        x_squared = ttnn.mul(x, x)
        mean_x_squared = ttnn.mean(x_squared, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_x_squared, self.eps))
        # Normalize and scale
        x_normed = ttnn.mul(x, ttnn.reciprocal(rms))
        return ttnn.mul(x_normed, self.weight)


class RotaryEmbedding:
    """Rotary Position Embedding"""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0, device=None):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        # Cache cos and sin
        self.cos_cached = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        self.sin_cached = torch.sin(freqs).unsqueeze(0).unsqueeze(0)

    def apply_rotary_emb(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Apply rotary embeddings to input tensor (on CPU/torch)"""
        # x: [batch, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = x.shape

        # Get cos/sin for the current position range
        cos = self.cos_cached[:, :, position : position + seq_len, :].to(x.device)
        sin = self.sin_cached[:, :, position : position + seq_len, :].to(x.device)

        # Split into even and odd dimensions
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Interleave back
        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)
        rotated = rotated.flatten(-2)

        return rotated


class Attention:
    """Multi-head attention with GQA support"""

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        device,
        max_seq_len: int = 2048,
    ):
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.hidden_size = hidden_size
        self.device = device

        # Convert weights to TTNN
        self.wq = ttnn.from_torch(
            wq.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wk = ttnn.from_torch(
            wk.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wv = ttnn.from_torch(
            wv.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.wo = ttnn.from_torch(
            wo.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len)

        # KV cache storage (on CPU as torch tensors for now)
        self.cache_k = torch.zeros((1, num_kv_heads, max_seq_len, head_dim))
        self.cache_v = torch.zeros((1, num_kv_heads, max_seq_len, head_dim))

    def __call__(self, x, start_pos: int, mask: Optional[torch.Tensor] = None):  # TTNN tensor [1, seq_len, hidden_size]
        # Project to Q, K, V
        xq = ttnn.matmul(x, self.wq)
        xk = ttnn.matmul(x, self.wk)
        xv = ttnn.matmul(x, self.wv)

        # Convert to torch for reshaping and RoPE
        xq_torch = ttnn.to_torch(xq).squeeze(0)  # [seq_len, hidden_size]
        xk_torch = ttnn.to_torch(xk).squeeze(0)
        xv_torch = ttnn.to_torch(xv).squeeze(0)

        batch_size, seq_len, _ = xq_torch.shape

        # Reshape to [batch, num_heads, seq_len, head_dim]
        xq_torch = xq_torch.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk_torch = xk_torch.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv_torch = xv_torch.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        xq_torch = self.rotary_emb.apply_rotary_emb(xq_torch, start_pos)
        xk_torch = self.rotary_emb.apply_rotary_emb(xk_torch, start_pos)

        # Update KV cache
        self.cache_k[:, :, start_pos : start_pos + seq_len] = xk_torch
        self.cache_v[:, :, start_pos : start_pos + seq_len] = xv_torch

        # Get keys and values up to current position
        keys = self.cache_k[:, :, : start_pos + seq_len]
        values = self.cache_v[:, :, : start_pos + seq_len]

        # Repeat KV heads for GQA
        if self.num_queries_per_kv > 1:
            keys = keys.repeat_interleave(self.num_queries_per_kv, dim=1)
            values = values.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Compute attention scores
        scores = torch.matmul(xq_torch, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        scores = torch.nn.functional.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(scores, values)

        # Reshape back to [batch, seq_len, hidden_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Convert back to TTNN and apply output projection
        output_tt = ttnn.from_torch(
            output.unsqueeze(0), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        output_tt = ttnn.matmul(output_tt, self.wo)

        return output_tt


class MLP:
    """Feed-forward network"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        gate_proj: torch.Tensor,
        up_proj: torch.Tensor,
        down_proj: torch.Tensor,
        device,
    ):
        self.gate_proj = ttnn.from_torch(
            gate_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.up_proj = ttnn.from_torch(
            up_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.down_proj = ttnn.from_torch(
            down_proj.T.unsqueeze(0).unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def __call__(self, x):
        # SwiGLU activation: gate(x) * up(x) then down projection
        gate = ttnn.matmul(x, self.gate_proj)
        gate = ttnn.silu(gate)

        up = ttnn.matmul(x, self.up_proj)

        hidden = ttnn.mul(gate, up)
        output = ttnn.matmul(hidden, self.down_proj)

        return output


class TransformerBlock:
    """Single transformer layer"""

    def __init__(self, layer_id: int, config, layer_weights, device, max_seq_len: int = 2048):
        self.layer_id = layer_id

        # Attention
        self.attention = Attention(
            layer_id=layer_id,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            wq=layer_weights["self_attn.q_proj.weight"],
            wk=layer_weights["self_attn.k_proj.weight"],
            wv=layer_weights["self_attn.v_proj.weight"],
            wo=layer_weights["self_attn.o_proj.weight"],
            device=device,
            max_seq_len=max_seq_len,
        )

        # MLP
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            gate_proj=layer_weights["mlp.gate_proj.weight"],
            up_proj=layer_weights["mlp.up_proj.weight"],
            down_proj=layer_weights["mlp.down_proj.weight"],
            device=device,
        )

        # Norms
        self.input_layernorm = RMSNorm(layer_weights["input_layernorm.weight"], config.rms_norm_eps, device)
        self.post_attention_layernorm = RMSNorm(
            layer_weights["post_attention_layernorm.weight"], config.rms_norm_eps, device
        )

    def __call__(self, x, start_pos: int, mask: Optional[torch.Tensor] = None):
        # Attention with residual
        h = self.input_layernorm(x)
        h = self.attention(h, start_pos, mask)
        x = ttnn.add(x, h)

        # MLP with residual
        h = self.post_attention_layernorm(x)
        h = self.mlp(h)
        x = ttnn.add(x, h)

        return x


class QwenModel:
    """Complete Qwen transformer model"""

    def __init__(self, model_name: str, device, max_seq_len: int = 2048):
        print(f"Loading {model_name}...")

        # Load HuggingFace model
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        self.config = hf_model.config
        self.device = device
        self.max_seq_len = max_seq_len

        print(f"Model config:")
        print(f"  Hidden size: {self.config.hidden_size}")
        print(f"  Num layers: {self.config.num_hidden_layers}")
        print(f"  Num attention heads: {self.config.num_attention_heads}")
        print(f"  Num KV heads: {self.config.num_key_value_heads}")
        print(f"  Intermediate size: {self.config.intermediate_size}")
        print(f"  Vocab size: {self.config.vocab_size}")

        # Extract weights
        state_dict = hf_model.state_dict()

        # Embedding (keep on CPU for now, will convert per-batch)
        self.embed_tokens = state_dict["model.embed_tokens.weight"]

        # Build transformer layers
        print("Building transformer layers...")
        self.layers = []
        for layer_id in range(self.config.num_hidden_layers):
            layer_weights = {
                key.replace(f"model.layers.{layer_id}.", ""): value
                for key, value in state_dict.items()
                if f"model.layers.{layer_id}." in key
            }

            layer = TransformerBlock(
                layer_id=layer_id,
                config=self.config,
                layer_weights=layer_weights,
                device=device,
                max_seq_len=max_seq_len,
            )
            self.layers.append(layer)

            if (layer_id + 1) % 4 == 0:
                print(f"  Loaded {layer_id + 1}/{self.config.num_hidden_layers} layers")

        # Final norm
        self.norm = RMSNorm(state_dict["model.norm.weight"], self.config.rms_norm_eps, device)

        # LM head (output projection)
        self.lm_head = ttnn.from_torch(
            state_dict["lm_head.weight"].T.unsqueeze(0).unsqueeze(0),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        print("Model loaded successfully!")

        # Free HF model
        del hf_model
        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):  # [batch, seq_len]
        batch_size, seq_len = tokens.shape

        # Embed tokens
        h = self.embed_tokens[tokens]  # [batch, seq_len, hidden_size]

        # Create causal mask
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        else:
            mask = None

        # Convert to TTNN
        h_tt = ttnn.from_torch(h.unsqueeze(0), device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        # Apply transformer layers
        for layer in self.layers:
            h_tt = layer(h_tt, start_pos, mask)

        # Final norm
        h_tt = self.norm(h_tt)

        # LM head
        logits = ttnn.matmul(h_tt, self.lm_head)

        return logits

    def reset_kv_cache(self):
        """Clear KV cache for all layers"""
        for layer in self.layers:
            layer.attention.cache_k.zero_()
            layer.attention.cache_v.zero_()


def generate(model: QwenModel, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0):
    """Generate text from prompt"""

    # Tokenize
    tokens = tokenizer.encode(prompt, return_tensors="pt")

    # Reset cache
    model.reset_kv_cache()

    # Prefill phase
    print(f"Prefill: {tokens.shape[1]} tokens")
    logits = model.forward(tokens, start_pos=0)

    # Get last token logits
    logits_torch = ttnn.to_torch(logits).squeeze(0)  # [batch, seq_len, vocab_size]
    next_token_id = torch.argmax(logits_torch[:, -1, :], dim=-1).item()  # Take last position

    generated = [next_token_id]
    current_pos = tokens.shape[1]

    # Decode phase
    print("Generating...")
    for i in range(max_new_tokens - 1):
        next_token = torch.tensor([[next_token_id]], dtype=torch.long)  # [1, 1]
        logits = model.forward(next_token, start_pos=current_pos)
        logits_torch = ttnn.to_torch(logits).squeeze(0)  # [batch, 1, vocab_size]

        # Sample next token
        if temperature > 0:
            probs = torch.nn.functional.softmax(logits_torch[:, -1, :] / temperature, dim=-1)
            next_token_id = torch.multinomial(probs.squeeze(0), num_samples=1).item()
        else:
            next_token_id = torch.argmax(logits_torch[:, -1, :], dim=-1).item()

        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated.append(next_token_id)
        current_pos += 1

        # Print token as it's generated
        if i % 5 == 0:
            print(".", end="", flush=True)

    print()
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    model_name = os.environ.get("HF_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    print(f"Pure TTNN implementation of {model_name}")
    print("=" * 80)

    # Setup device
    if os.environ.get("MESH_DEVICE") == "N150":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
    elif os.environ.get("MESH_DEVICE") == "N300":
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 2]))
    else:
        device_ids = ttnn.get_device_ids()
        num_devices = len(device_ids)
        if num_devices >= 1:
            mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]))
        else:
            raise RuntimeError("No devices found")

    print(f"Using {mesh_device.get_num_devices()} device(s)")
    print()

    # Load model
    model = QwenModel(model_name, mesh_device, max_seq_len=2048)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare prompt
    messages = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print()
    print("Prompt:", prompt)
    print()

    # Generate
    response = generate(model, tokenizer, prompt, max_new_tokens=50)

    print()
    print("Response:", response)
    print()

    # Print validation report if any validations were run
    registry = get_validation_registry()
    if registry.results:
        registry.print_report()

    # Cleanup
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
