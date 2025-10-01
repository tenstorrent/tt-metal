"""
Hardware Configuration Example

Shows the complete flow from module spec → execution strategy → hardware config → tensor caches
"""

from typing import Any, Dict, Optional

import torch

import ttnn

# =============================================================================
# Complete Example: Building Attention with Hardware Config
# =============================================================================


class AttentionModuleExample:
    """
    Example showing the complete hardware configuration flow.

    Flow: Spec → Specialize (prefill/decode) → Configure Hardware → Cache Tensors
    """

    # Device-specific default configurations
    DEVICE_DEFAULTS = {
        "ttnn:n150": {
            "prefill": {"matmul_tiles": 8, "use_dram": True, "dtype": "bfloat16", "accumulate_dtype": "float32"},
            "decode": {
                "matmul_tiles": 1,
                "use_l1": True,
                "dtype": "bfloat16",
                "accumulate_dtype": "bfloat16",  # Lower precision OK
            },
        },
        "ttnn:n300": {
            "prefill": {
                "matmul_tiles": 16,  # Larger device, larger tiles
                "use_dram": False,  # Enough L1 memory
                "dtype": "bfloat16",
                "accumulate_dtype": "float32",
            },
            "decode": {
                "matmul_tiles": 2,
                "use_l1": True,
                "dtype": "bfloat8_b",  # More aggressive quantization
                "accumulate_dtype": "bfloat16",
            },
        },
    }

    def __init__(self, spec: AttentionSpec, device: str, hw_config_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize attention module with hardware configuration.

        Args:
            spec: Pure mathematical specification
            device: Target TT device
            hw_config_overrides: Optional overrides for specific ops
        """
        self.spec = spec
        self.device = device
        self.hw_config_overrides = hw_config_overrides or {}

        # Step 1: Create specialized implementations
        print(f"Step 1: Specializing for prefill/decode...")
        self._create_specialized_implementations()

        # Step 2: Apply hardware configurations
        print(f"Step 2: Applying hardware configs for {device}...")
        self._apply_hardware_configs()

        # Step 3: Prepare for tensor caching
        print(f"Step 3: Preparing tensor cache structures...")
        self._prepare_tensor_caches()

    def _create_specialized_implementations(self):
        """Step 1: Create prefill and decode variants"""
        # These are created but not yet configured with hardware details
        self.prefill = {
            "name": "prefill_attention",
            "ops": ["qkv_matmul", "scores_matmul", "output_matmul"],
            "optimized_for": "throughput",
        }

        self.decode = {
            "name": "decode_attention",
            "ops": ["q_matmul", "kv_cache_read", "single_token_matmul"],
            "optimized_for": "latency",
            "has_kv_cache": True,
        }

    def _apply_hardware_configs(self):
        """Step 2: Configure each op with hardware-specific settings"""
        device_config = self.DEVICE_DEFAULTS.get(self.device, self.DEVICE_DEFAULTS["ttnn:n150"])

        # Configure prefill ops
        prefill_config = device_config["prefill"]
        self._configure_prefill_ops(prefill_config)

        # Configure decode ops
        decode_config = device_config["decode"]
        self._configure_decode_ops(decode_config)

    def _configure_prefill_ops(self, config):
        """Configure prefill operations with hardware settings"""
        # QKV projection for prefill
        qkv_config = self.hw_config_overrides.get("qkv_matmul", {})

        self.prefill["qkv_matmul"] = ttnn.Linear(
            self.spec.hidden_dim,
            3 * self.spec.hidden_dim,
            bias=False,
            # Hardware-specific configurations
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
                buffer_type=ttnn.BufferType.DRAM if config["use_dram"] else ttnn.BufferType.L1,
            ),
            kernel_config=ttnn.MatmulConfig(
                per_core_M=config["matmul_tiles"],
                per_core_N=config["matmul_tiles"],
                fp32_accumulation=(config["accumulate_dtype"] == "float32"),
            ),
            dtype=qkv_config.get("dtype", config["dtype"]),
        )

        print(
            f"  Configured prefill QKV: tiles={config['matmul_tiles']}, "
            f"memory={'DRAM' if config['use_dram'] else 'L1'}"
        )

    def _configure_decode_ops(self, config):
        """Configure decode operations with hardware settings"""
        # Single Q projection for decode
        q_config = self.hw_config_overrides.get("q_matmul", {})

        self.decode["q_matmul"] = ttnn.Linear(
            self.spec.hidden_dim,
            self.spec.hidden_dim,
            bias=False,
            # Decode-optimized configuration
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1 if config["use_l1"] else ttnn.BufferType.DRAM,
            ),
            kernel_config=ttnn.MatmulConfig(
                per_core_M=config["matmul_tiles"],
                per_core_N=config["matmul_tiles"],
                fp32_accumulation=(config["accumulate_dtype"] == "float32"),
            ),
            dtype=q_config.get("dtype", config["dtype"]),
        )

        # KV cache configuration
        self.decode["kv_cache"] = ttnn.KVCache(
            num_heads=self.spec.num_heads,
            head_dim=self.spec.hidden_dim // self.spec.num_heads,
            max_seq_length=8192,
            dtype=config["dtype"],
            layout=ttnn.Layout.TILE,
            device=self.device,
        )

        print(
            f"  Configured decode Q: tiles={config['matmul_tiles']}, " f"memory={'L1' if config['use_l1'] else 'DRAM'}"
        )
        print(f"  Configured KV cache: dtype={config['dtype']}")

    def _prepare_tensor_caches(self):
        """Step 3: Prepare structures for tensor caching"""
        self.weight_cache_config = {"dtype": "bfloat16", "layout": ttnn.Layout.TILE, "cache_on_device": True}

        self.activation_cache_config = {
            "cache_policy": "auto",  # Cache large activations
            "min_size_to_cache": 1024 * 1024,  # 1M elements
            "dtype": "bfloat16",
        }

        print(f"  Weight cache: dtype={self.weight_cache_config['dtype']}")
        print(f"  Activation cache: policy={self.activation_cache_config['cache_policy']}")

    def cache_weights_from_reference(self, reference_module):
        """Cache weights from reference model (happens at config time)"""
        print(f"\nCaching weights from reference model...")

        # Extract reference weights
        ref_weights = {"qkv_weight": reference_module.qkv_proj.weight, "out_weight": reference_module.out_proj.weight}

        # Convert to TT format with caching
        self.cached_weights = {}
        for name, weight in ref_weights.items():
            tt_weight = ttnn.from_torch(
                weight,
                device=self.device,
                dtype=self.weight_cache_config["dtype"],
                layout=self.weight_cache_config["layout"],
                memory_config=ttnn.MemoryConfig(
                    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM
                ),
            )

            self.cached_weights[name] = tt_weight
            print(f"  Cached {name}: shape={weight.shape}, " f"dtype={self.weight_cache_config['dtype']}")

        # Update ops to use cached weights
        self.prefill["qkv_matmul"].set_weight(self.cached_weights["qkv_weight"])

    def setup_activation_caches(self, batch_size: int, seq_length: int):
        """Setup activation caches (happens at compile time)"""
        print(f"\nSetting up activation caches for batch={batch_size}, seq={seq_length}...")

        # Pre-allocate frequently reused activation buffers
        self.activation_caches = {}

        # QKV output buffer
        qkv_shape = (batch_size, seq_length, 3 * self.spec.hidden_dim)
        if self._should_cache_activation(qkv_shape):
            self.activation_caches["qkv_output"] = ttnn.allocate_tensor(
                shape=qkv_shape,
                device=self.device,
                dtype=self.activation_cache_config["dtype"],
                layout=ttnn.Layout.TILE,
            )
            print(f"  Allocated QKV cache: {qkv_shape}")

        # Attention scores buffer
        scores_shape = (batch_size, self.spec.num_heads, seq_length, seq_length)
        if self._should_cache_activation(scores_shape):
            self.activation_caches["attention_scores"] = ttnn.allocate_tensor(
                shape=scores_shape,
                device=self.device,
                dtype="bfloat16",  # Lower precision for scores
                layout=ttnn.Layout.TILE,
            )
            print(f"  Allocated attention scores cache: {scores_shape}")

    def _should_cache_activation(self, shape):
        """Determine if activation should be cached based on size"""
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        return num_elements >= self.activation_cache_config["min_size_to_cache"]


# =============================================================================
# Usage Example
# =============================================================================


def demo_hardware_config_flow():
    """Demonstrate the complete hardware configuration flow"""

    print("=== TTTv2 Hardware Configuration Flow Demo ===\n")

    # 1. Start with pure spec
    spec = AttentionSpec(hidden_dim=4096, num_heads=32)
    print(f"Starting with spec: hidden_dim={spec.hidden_dim}, num_heads={spec.num_heads}\n")

    # 2. Create module for N150 device
    print("Creating module for N150 device:")
    attention_n150 = AttentionModuleExample(spec, device="ttnn:n150")

    # 3. Override specific op config
    print("\n\nCreating module with custom QKV config:")
    custom_config = {"qkv_matmul": {"dtype": "float32", "accumulate_dtype": "float32"}}  # Higher precision
    attention_custom = AttentionModuleExample(spec, "ttnn:n150", custom_config)

    # 4. Cache weights from reference
    print("\n\nSimulating weight caching:")

    # Create mock reference module
    class MockReference:
        def __init__(self, hidden_dim):
            self.qkv_proj = type("", (), {"weight": torch.randn(hidden_dim, 3 * hidden_dim)})()
            self.out_proj = type("", (), {"weight": torch.randn(hidden_dim, hidden_dim)})()

    ref_module = MockReference(4096)
    attention_n150.cache_weights_from_reference(ref_module)

    # 5. Setup activation caches at compile time
    print("\n\nSimulating compile-time activation cache setup:")
    attention_n150.setup_activation_caches(batch_size=8, seq_length=512)

    # 6. Show device differences
    print("\n\nComparing N150 vs N300 configurations:")
    attention_n300 = AttentionModuleExample(spec, device="ttnn:n300")
    print("\nN150 uses DRAM for prefill, N300 uses L1 (more memory available)")
    print("N300 uses more aggressive quantization (bfloat8_b) for decode")


def demo_override_patterns():
    """Show different ways to override hardware configs"""

    print("\n\n=== Hardware Config Override Patterns ===\n")

    spec = AttentionSpec(4096, 32)

    # Pattern 1: Override single op
    print("1. Override single operation:")
    config1 = {"qkv_matmul": {"dtype": "float32"}}
    attention1 = AttentionModuleExample(spec, "ttnn:n150", config1)

    # Pattern 2: Override multiple ops
    print("\n2. Override multiple operations:")
    config2 = {"qkv_matmul": {"dtype": "float32"}, "q_matmul": {"use_l1": False, "matmul_tiles": 4}}
    attention2 = AttentionModuleExample(spec, "ttnn:n150", config2)

    # Pattern 3: Performance tuning
    print("\n3. Performance-tuned configuration:")
    perf_config = {
        "qkv_matmul": {
            "matmul_tiles": 16,  # Larger tiles
            "use_dram": False,  # Force L1
            "dtype": "bfloat16",
            "accumulate_dtype": "float32",
        }
    }
    attention_perf = AttentionModuleExample(spec, "ttnn:n150", perf_config)


if __name__ == "__main__":
    demo_hardware_config_flow()
    demo_override_patterns()
