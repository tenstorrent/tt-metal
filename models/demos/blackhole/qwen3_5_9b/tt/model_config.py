# models/demos/blackhole/qwen3_5_9b/tt/model_config.py
"""Qwen3.5-9B model configuration for Blackhole P150.

Subclasses nothing — standalone config that loads from HuggingFace config.json.
Handles Qwen3.5-specific config:
- Hybrid attention layer types (Gated DeltaNet + Gated Full Attention)
- DeltaNet-specific parameters (key/value heads, conv kernel)
- Partial rotary factor for RoPE
"""
import json
import os
from pathlib import Path


class Qwen35ModelArgs:
    """Model configuration for Qwen3.5-9B on Blackhole P150."""

    def __init__(
        self,
        mesh_device=None,
        checkpoint_dir="/local/ttuser/atupe/Qwen9b",
        max_batch_size=1,
        max_seq_len=2048,
    ):
        self.mesh_device = mesh_device
        self.checkpoint_dir = checkpoint_dir
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        text_config = config.get("text_config", config)

        # Core dimensions
        self.dim = text_config["hidden_size"]
        self.n_layers = text_config["num_hidden_layers"]
        self.n_heads = text_config["num_attention_heads"]
        self.n_kv_heads = text_config["num_key_value_heads"]
        self.head_dim = text_config["head_dim"]
        self.hidden_dim = text_config["intermediate_size"]
        self.vocab_size = text_config["vocab_size"]
        self.norm_eps = text_config["rms_norm_eps"]

        # RoPE — rope_theta is nested under rope_parameters in config.json
        rope_params = text_config.get("rope_parameters", {})
        self.rope_theta = rope_params.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = text_config.get("partial_rotary_factor", 1.0)
        self.rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # DeltaNet-specific parameters
        self.linear_num_key_heads = text_config.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = text_config.get("linear_num_value_heads", 32)
        self.linear_key_head_dim = text_config.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = text_config.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = text_config.get("linear_conv_kernel_dim", 4)

        # Layer type list
        self.attention_type_list = text_config.get(
            "layer_types",
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
        )

        # Derived
        self.linear_q_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_v_dim = self.linear_num_value_heads * self.linear_value_head_dim

        # Blackhole P150 device config (lazy import to allow CPU-only testing)
        if mesh_device is not None:
            import ttnn

            self.weight_dtype = ttnn.bfloat8_b
            self.act_dtype = ttnn.bfloat16
        else:
            self.weight_dtype = None
            self.act_dtype = None

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "full_attention"

    def is_deltanet_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "linear_attention"

    def weight_cache_path(self, dtype=None):
        """Return cache directory path for converted weight tensors.

        Directory is created automatically by ttnn.as_tensor when first cache file is written.
        """
        if dtype is None:
            dtype = self.weight_dtype
        import ttnn

        if dtype == ttnn.bfloat8_b:
            suffix = "tensor_cache_bfp8"
        else:
            suffix = "tensor_cache_bf16"
        return Path(self.checkpoint_dir) / suffix
