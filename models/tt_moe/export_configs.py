#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Export MoE configurations to JSON files.

This script generates complete MoE configurations for both DeepSeek and GPT-OSS
backends in decode and prefill modes, then exports them as JSON files.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from loguru import logger

from models.tt_moe.moe_block import MoEBlock
from models.tt_moe.utils.json_export import export_config_to_json


def create_mock_mesh_device(shape=(4, 8)) -> object:
    """Create a mock mesh device for configuration generation.

    Args:
        shape: Mesh shape (y, x)

    Returns:
        Mock mesh device object
    """

    class MockMeshDevice:
        def __init__(self, shape):
            self.shape = shape

        def get_num_devices(self):
            return self.shape[0] * self.shape[1]

    return MockMeshDevice(shape)


def create_deepseek_hf_config() -> object:
    """Create a DeepSeek HuggingFace configuration object.

    Returns:
        HF config object with DeepSeek parameters
    """

    class DeepSeekConfig:
        def __init__(self):
            # DeepSeek V3 configuration
            self.hidden_size = 7168
            self.intermediate_size = 18432
            self.moe_intermediate_size = 2048
            self.num_hidden_layers = 61
            self.num_attention_heads = 128
            self.num_key_value_heads = 8
            self.n_routed_experts = 256  # Total experts
            self.n_shared_experts = 1
            self.num_experts_per_tok = 8
            self.rope_theta = 10000000
            self.vocab_size = 129280
            self.max_position_embeddings = 163840
            self.model_type = "deepseek_v3"

            # Additional fields that may be accessed
            self.norm_topk_prob = False
            self.scoring_func = "softmax"
            self.aux_loss_alpha = 0.001
            self.seq_aux = True
            self.topk_method = "grouped_limited_greedy"
            self.topk_group = 3
            self.n_group = 8  # Number of expert groups
            self.routed_scaling_factor = 1.0

    return DeepSeekConfig()


def create_gptoss_hf_config() -> object:
    """Create a GPT-OSS HuggingFace configuration object.

    Returns:
        HF config object with GPT-OSS MoE parameters
    """

    class GptOssConfig:
        def __init__(self):
            # GPT-OSS MoE configuration (based on PhiMoE)
            self.hidden_size = 2880
            self.intermediate_size = 11520  # For GPT-OSS experts
            self.num_hidden_layers = 32
            self.num_attention_heads = 32
            self.num_local_experts = 128  # Total experts (divisible by 32 devices)
            self.n_routed_experts = 128  # Alternative field name
            self.num_experts_per_tok = 2
            self.router_aux_loss_coef = 0.0
            self.output_router_logits = False
            self.vocab_size = 32064
            self.max_position_embeddings = 131072
            self.model_type = "phimoe"

            # GPT-OSS specific
            self.router_jitter_noise = 0.01

    return GptOssConfig()


def generate_deepseek_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate complete DeepSeek configurations for decode and prefill.

    Returns:
        Tuple of (decode_config, prefill_config)
    """
    hf_config = create_deepseek_hf_config()
    mesh_device = create_mock_mesh_device(shape=(4, 8))

    # Generate decode configuration
    decode_config = MoEBlock.decode_model_config(
        hf_config=hf_config, mesh_device=mesh_device, topk_fallback=False, backend="deepseek"
    )

    # Add additional configuration that's normally set at runtime
    decode_config.update(
        {
            "mode": "decode",
            "backend": "deepseek",
            "batch_size": 32,  # Example batch size
            "sequence_length": 1,
            "model_type": hf_config.model_type,
            "total_experts": hf_config.n_routed_experts,
            # Activation configuration for experts
            "activation": {
                "type": "swiglu",
                "swiglu_alpha": 1.0,  # DeepSeek default
                "swiglu_limit": float("inf"),  # No clamping
            },
            # Chunking configuration
            "enable_chunking": True,
            "chunk_size": 32,  # Default chunk size for decode
            "chunk_dim": 0,  # Batch dimension
        }
    )

    # Generate prefill configuration
    prefill_config = MoEBlock.prefill_model_config(
        hf_config=hf_config, mesh_device=mesh_device, topk_fallback=False, backend="deepseek"
    )

    # Add additional configuration for prefill
    prefill_config.update(
        {
            "mode": "prefill",
            "backend": "deepseek",
            "batch_size": 1,
            "sequence_length": 2048,  # Example sequence length
            "model_type": hf_config.model_type,
            "total_experts": hf_config.n_routed_experts,
            "prefill_chunk_size": 16384,  # Global token limit for chunking
            # Activation configuration
            "activation": {
                "type": "swiglu",
                "swiglu_alpha": 1.0,
                "swiglu_limit": float("inf"),
            },
            # Chunking configuration
            "enable_chunking": True,
            "chunk_size": 512,  # Chunk size for prefill
            "chunk_dim": 2,  # Sequence dimension
        }
    )

    return decode_config, prefill_config


def generate_gptoss_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate complete GPT-OSS configurations for decode and prefill.

    Returns:
        Tuple of (decode_config, prefill_config)
    """
    hf_config = create_gptoss_hf_config()
    mesh_device = create_mock_mesh_device(shape=(4, 8))

    # Generate decode configuration
    decode_config = MoEBlock.decode_model_config(
        hf_config=hf_config, mesh_device=mesh_device, topk_fallback=False, backend="gptoss"
    )

    # Add GPT-OSS specific configuration
    decode_config.update(
        {
            "mode": "decode",
            "backend": "gptoss",
            "batch_size": 32,
            "sequence_length": 1,
            "model_type": hf_config.model_type,
            "total_experts": hf_config.num_local_experts,
            "intermediate_size": hf_config.intermediate_size,
            # Activation configuration for clamped SwiGLU
            "activation": {
                "type": "clamped_swiglu",
                "swiglu_alpha": 1.702,  # GPT-OSS specific value
                "swiglu_limit": 7.0,  # Clamping limit
            },
            # Throughput experts configuration
            "use_throughput_experts": True,
            "use_fused_gate_up": False,  # Whether gate and up projections are fused
            # Weight precision
            "weight_dtype": "bfloat8_b",
            "activation_dtype": "bfloat16",
            # Chunking configuration
            "enable_chunking": False,  # GPT-OSS typically doesn't chunk in decode
        }
    )

    # Generate prefill configuration
    prefill_config = MoEBlock.prefill_model_config(
        hf_config=hf_config, mesh_device=mesh_device, topk_fallback=False, backend="gptoss"
    )

    # Add GPT-OSS specific configuration for prefill
    prefill_config.update(
        {
            "mode": "prefill",
            "backend": "gptoss",
            "batch_size": 1,
            "sequence_length": 2048,
            "model_type": hf_config.model_type,
            "total_experts": hf_config.num_local_experts,
            "intermediate_size": hf_config.intermediate_size,
            "prefill_chunk_size": 16384,
            # Activation configuration
            "activation": {
                "type": "clamped_swiglu",
                "swiglu_alpha": 1.702,
                "swiglu_limit": 7.0,
            },
            # Throughput experts configuration
            "use_throughput_experts": True,
            "use_fused_gate_up": False,
            # Weight precision
            "weight_dtype": "bfloat8_b",
            "activation_dtype": "bfloat16",
            # Chunking configuration for prefill
            "enable_chunking": True,
            "chunk_size": 512,  # Sequence chunk size
            "chunk_dim": 2,  # Sequence dimension
        }
    )

    return decode_config, prefill_config


def main():
    """Main function to generate and export all configurations."""
    logger.info("Generating MoE configurations...")

    # Generate DeepSeek configurations
    logger.info("Generating DeepSeek configurations...")
    deepseek_decode, deepseek_prefill = generate_deepseek_configs()

    # Generate GPT-OSS configurations
    logger.info("Generating GPT-OSS configurations...")
    gptoss_decode, gptoss_prefill = generate_gptoss_configs()

    # Create output directories
    deepseek_dir = Path("models/tt_moe/deepseek/config")
    gptoss_dir = Path("models/tt_moe/gpt-oss/config")
    deepseek_dir.mkdir(parents=True, exist_ok=True)
    gptoss_dir.mkdir(parents=True, exist_ok=True)

    # Export DeepSeek configuration
    deepseek_config = {"decode": deepseek_decode, "prefill": deepseek_prefill}
    export_config_to_json(deepseek_config, deepseek_dir / "deepseek.json")
    logger.info(f"DeepSeek configuration exported to {deepseek_dir / 'deepseek.json'}")

    # Export GPT-OSS configuration
    gptoss_config = {"decode": gptoss_decode, "prefill": gptoss_prefill}
    export_config_to_json(gptoss_config, gptoss_dir / "gpt-oss.json")
    logger.info(f"GPT-OSS configuration exported to {gptoss_dir / 'gpt-oss.json'}")

    # Also export individual files for easier access
    export_config_to_json(deepseek_decode, deepseek_dir / "deepseek_decode.json")
    export_config_to_json(deepseek_prefill, deepseek_dir / "deepseek_prefill.json")
    export_config_to_json(gptoss_decode, gptoss_dir / "gptoss_decode.json")
    export_config_to_json(gptoss_prefill, gptoss_dir / "gptoss_prefill.json")

    logger.info("Configuration export complete!")


if __name__ == "__main__":
    main()
