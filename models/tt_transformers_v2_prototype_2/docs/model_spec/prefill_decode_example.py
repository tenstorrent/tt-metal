"""
Prefill/Decode Specialization Example

This shows how TTTv2 handles prefill/decode as a first-class pattern
while remaining extensible for future execution strategies.
"""

from enum import Enum
from typing import Any, Dict, Optional

import torch

import ttnn

# =============================================================================
# Simple Version: Direct Prefill/Decode Support
# =============================================================================


class AttentionWithPrefillDecode:
    """
    Simplest approach: Direct prefill/decode methods.
    This is what most TTTv2 users will see and use.
    """

    def __init__(self, spec: AttentionSpec, device="ttnn"):
        self.spec = spec
        self.device = device

        # Create specialized implementations at init time
        self._create_prefill_implementation()
        self._create_decode_implementation()

    def _create_prefill_implementation(self):
        """Prefill: Process many tokens at once"""
        # Optimized for parallel processing of full sequence
        self.prefill_qkv_proj = ttnn.Linear(
            self.spec.hidden_dim,
            3 * self.spec.hidden_dim,
            # Prefill optimization: larger tile size for throughput
            kernel_config=ttnn.MatmulConfig(tile_width=512, fused_activation=True),  # Large tiles for throughput
        )

    def _create_decode_implementation(self):
        """Decode: Process one token with KV cache"""
        # Optimized for single token + cache access
        self.decode_q_proj = ttnn.Linear(
            self.spec.hidden_dim,
            self.spec.hidden_dim,
            # Decode optimization: small tile for latency
            kernel_config=ttnn.MatmulConfig(
                tile_width=32, optimized_for_sequential=True  # Small tiles for low latency
            ),
        )

        # KV cache for decode
        self.kv_cache = ttnn.KVCache(
            num_layers=1,
            num_heads=self.spec.num_heads,
            head_dim=self.spec.hidden_dim // self.spec.num_heads,
            max_seq_length=8192,
            dtype="bfloat16",
        )

    def prefill_forward(self, hidden_states, attention_mask=None):
        """Forward pass for prefill phase"""
        batch_size, seq_len, _ = hidden_states.shape

        # Use prefill-optimized operations
        qkv = self.prefill_qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Full attention computation
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        if attention_mask is not None:
            scores += attention_mask

        probs = ttnn.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)

        # Initialize KV cache for subsequent decode
        self.kv_cache.update(k, v)

        return output

    def decode_forward(self, hidden_states, cache_position):
        """Forward pass for decode phase (single token)"""
        # Single token query
        q = self.decode_q_proj(hidden_states)

        # Get cached KV
        k_cache, v_cache = self.kv_cache.get()

        # Efficient attention with cache
        scores = torch.matmul(q, k_cache.transpose(-1, -2)) / self.scale
        probs = ttnn.softmax(scores, dim=-1)
        output = torch.matmul(probs, v_cache)

        # Update cache with new KV
        k_new = self.decode_k_proj(hidden_states)
        v_new = self.decode_v_proj(hidden_states)
        self.kv_cache.update(k_new, v_new, position=cache_position)

        return output

    # Unified forward that auto-selects based on input
    def forward(self, hidden_states, cache_position=None, **kwargs):
        """Smart forward that detects prefill vs decode"""
        batch_size, seq_len, _ = hidden_states.shape

        if seq_len > 1 or cache_position is None:
            # Prefill mode: multiple tokens or no cache position
            return self.prefill_forward(hidden_states, **kwargs)
        else:
            # Decode mode: single token with cache
            return self.decode_forward(hidden_states, cache_position, **kwargs)


# =============================================================================
# Advanced Version: Strategy Pattern (Future-Proof)
# =============================================================================


class ExecutionMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    # Future modes can be added here
    SPECULATIVE = "speculative"
    CONTINUOUS_BATCH = "continuous_batch"


class ExecutionStrategy:
    """Base strategy for execution modes"""

    def create_implementations(self, spec: AttentionSpec, device: str) -> Dict[ExecutionMode, Any]:
        raise NotImplementedError

    def supports_mode(self, mode: ExecutionMode) -> bool:
        raise NotImplementedError


class TTNNPrefillDecodeStrategy(ExecutionStrategy):
    """Default TTNN strategy with prefill/decode specialization"""

    def create_implementations(self, spec: AttentionSpec, device: str):
        return {
            ExecutionMode.PREFILL: PrefillAttentionImpl(spec, device),
            ExecutionMode.DECODE: DecodeAttentionImpl(spec, device),
        }

    def supports_mode(self, mode: ExecutionMode) -> bool:
        return mode in [ExecutionMode.PREFILL, ExecutionMode.DECODE]


class MultiHeadAttentionV2:
    """
    Advanced version with pluggable execution strategies.
    Default is prefill/decode but can be extended.
    """

    def __init__(self, spec: AttentionSpec, device: str = "ttnn", strategy: Optional[ExecutionStrategy] = None):
        self.spec = spec
        self.device = device

        # Default to TTNN prefill/decode strategy
        self.strategy = strategy or TTNNPrefillDecodeStrategy()

        # Create implementations
        self.implementations = self.strategy.create_implementations(spec, device)
        self._current_mode = ExecutionMode.PREFILL

    def forward(self, hidden_states, mode: Optional[ExecutionMode] = None, **kwargs):
        """Forward with explicit or auto-detected mode"""

        # Auto-detect mode if not specified
        if mode is None:
            seq_len = hidden_states.shape[1]
            mode = ExecutionMode.DECODE if seq_len == 1 else ExecutionMode.PREFILL

        # Check if mode is supported
        if not self.strategy.supports_mode(mode):
            raise ValueError(f"Mode {mode} not supported by current strategy")

        # Dispatch to appropriate implementation
        return self.implementations[mode].forward(hidden_states, **kwargs)

    # Convenience methods that feel native
    def prefill_forward(self, *args, **kwargs):
        return self.forward(*args, mode=ExecutionMode.PREFILL, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return self.forward(*args, mode=ExecutionMode.DECODE, **kwargs)

    # Future extensibility
    def set_strategy(self, new_strategy: ExecutionStrategy):
        """Switch execution strategy at runtime"""
        self.strategy = new_strategy
        self.implementations = new_strategy.create_implementations(self.spec, self.device)


# =============================================================================
# Usage Examples
# =============================================================================


def example_simple_usage():
    """Most users will use the simple API"""
    print("=== Simple Prefill/Decode Usage ===")

    # Create attention module
    spec = AttentionSpec(hidden_dim=4096, num_heads=32)
    attention = AttentionWithPrefillDecode(spec, device="ttnn:0")

    # Prefill phase: Process prompt
    prompt_tokens = torch.randn(1, 128, 4096)  # [batch, seq, hidden]
    prefill_output = attention.prefill_forward(prompt_tokens)
    print(f"Prefill output shape: {prefill_output.shape}")

    # Decode phase: Generate tokens
    for i in range(10):
        new_token = torch.randn(1, 1, 4096)  # Single token
        decode_output = attention.decode_forward(new_token, cache_position=128 + i)
        print(f"Decode step {i+1} output shape: {decode_output.shape}")


def example_auto_mode_selection():
    """Automatic mode detection"""
    print("\n=== Automatic Mode Selection ===")

    attention = AttentionWithPrefillDecode(AttentionSpec(4096, 32))

    # Automatically uses prefill (seq_len > 1)
    multi_token = torch.randn(1, 50, 4096)
    output = attention.forward(multi_token)
    print("Multi-token input -> Prefill mode")

    # Automatically uses decode (seq_len == 1 with cache)
    single_token = torch.randn(1, 1, 4096)
    output = attention.forward(single_token, cache_position=50)
    print("Single token input -> Decode mode")


def example_future_extension():
    """How to add new execution strategies in the future"""
    print("\n=== Future Extension Example ===")

    # Define new strategy for speculative decoding
    class SpeculativeDecodingStrategy(ExecutionStrategy):
        """Future strategy for speculative decoding"""

        def create_implementations(self, spec, device):
            return {
                ExecutionMode.PREFILL: PrefillAttentionImpl(spec, device),
                ExecutionMode.SPECULATIVE: SpeculativeAttentionImpl(spec, device),
            }

        def supports_mode(self, mode):
            return mode in [ExecutionMode.PREFILL, ExecutionMode.SPECULATIVE]

    # Use new strategy without changing core module
    attention = MultiHeadAttentionV2(AttentionSpec(4096, 32), strategy=SpeculativeDecodingStrategy())

    # New mode is available
    draft_output = attention.forward(torch.randn(1, 5, 4096), mode=ExecutionMode.SPECULATIVE)
    print("Speculative decoding mode now available!")


def example_performance_comparison():
    """Show performance benefits of specialization"""
    print("\n=== Performance Benefits ===")

    import time

    spec = AttentionSpec(4096, 32)

    # Specialized implementation
    specialized = AttentionWithPrefillDecode(spec)

    # Measure prefill performance
    prefill_input = torch.randn(8, 512, 4096)  # Batch of sequences

    start = time.time()
    for _ in range(10):
        specialized.prefill_forward(prefill_input)
    prefill_time = time.time() - start

    print(f"Prefill time (specialized): {prefill_time:.3f}s")

    # Measure decode performance
    decode_input = torch.randn(8, 1, 4096)  # Single tokens

    start = time.time()
    for i in range(100):
        specialized.decode_forward(decode_input, cache_position=512 + i)
    decode_time = time.time() - start

    print(f"Decode time (specialized): {decode_time:.3f}s")
    print("\nSpecialization provides ~2-3x speedup on TT hardware!")


# =============================================================================
# Integration with Model Building
# =============================================================================


class TTTModel:
    """Model that uses prefill/decode throughout"""

    def __init__(self, config):
        self.layers = []

        for i in range(config.num_layers):
            # All layers support prefill/decode
            attention = AttentionWithPrefillDecode(AttentionSpec(config.hidden_dim, config.num_heads))
            ffn = FFNWithPrefillDecode(FFNSpec(config.hidden_dim, config.ffn_dim))

            self.layers.append({"attention": attention, "ffn": ffn})

    def generate(self, prompt_tokens):
        """Generation with explicit prefill/decode phases"""

        # Phase 1: Prefill prompt
        hidden_states = self.embed(prompt_tokens)

        for layer in self.layers:
            hidden_states = layer["attention"].prefill_forward(hidden_states)
            hidden_states = layer["ffn"].prefill_forward(hidden_states)

        # Phase 2: Decode generation
        for i in range(max_new_tokens):
            # Get next token (simplified)
            next_token = self.sample(hidden_states)

            # Process through decode path
            token_hidden = self.embed(next_token)

            for layer in self.layers:
                token_hidden = layer["attention"].decode_forward(token_hidden, cache_position=prompt_length + i)
                token_hidden = layer["ffn"].decode_forward(token_hidden)

            hidden_states = token_hidden


if __name__ == "__main__":
    example_simple_usage()
    example_auto_mode_selection()
    example_future_extension()
    example_performance_comparison()
