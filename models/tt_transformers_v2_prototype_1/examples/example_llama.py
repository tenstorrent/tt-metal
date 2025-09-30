# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Example LLaMA implementation using TTTv2.

This demonstrates how model implementations can use TTTv2 as a library
while remaining separate from the core codebase.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from tt_transformers_v2.config import OptimizationConfig, TransformerConfig, WeightLoader

# Import from TTTv2 - model implementations depend on specific version
from tt_transformers_v2.core import (
    Embedding,
    EmbeddingConfig,
    LMHead,
    LMHeadConfig,
    NormConfig,
    RMSNorm,
    RoPE,
    RoPEConfig,
    TransformerBlock,
    TransformerBlockConfig,
)
from tt_transformers_v2.interfaces import GenerationConfig, HWConfig, StandardGenerator

import ttnn


class LLaMAModel(torch.nn.Module):
    """
    Example LLaMA model implementation using TTTv2.

    This is a standalone implementation that uses TTTv2 modules.
    """

    def __init__(
        self,
        config: TransformerConfig,
        device: ttnn.Device,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.optimization_config = optimization_config or OptimizationConfig()

        # Initialize modules using TTTv2 components
        self._build_model()

    def _build_model(self):
        """Build model using TTTv2 modules"""
        # Embedding layer
        embed_config = EmbeddingConfig(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.hidden_size,
            scale_embeddings=self.config.custom_attrs.get("scale_embeddings", False),
        )
        self.embedding = Embedding(embed_config, self.device)

        # RoPE setup
        rope_config = RoPEConfig(
            dim=self.config.head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
            rope_type=self.config.custom_attrs.get("rope_type", "default"),
        )
        self.rope = RoPE(rope_config, self.device)

        # Transformer blocks
        self.blocks = torch.nn.ModuleList()
        for layer_idx in range(self.config.num_hidden_layers):
            block_config = TransformerBlockConfig(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                intermediate_size=self.config.intermediate_size,
                norm_type=self.config.norm_type,
                norm_eps=self.config.layer_norm_eps,
                activation=self.config.hidden_act,
                use_parallel_residual=self.config.use_parallel_residual,
            )
            block = TransformerBlock(block_config, self.device, layer_idx)
            self.blocks.append(block)

        # Final normalization
        norm_config = NormConfig(
            normalized_shape=self.config.hidden_size,
            eps=self.config.layer_norm_eps,
        )
        self.final_norm = RMSNorm(norm_config, self.device)

        # Language model head
        lm_head_config = LMHeadConfig(
            hidden_size=self.config.hidden_size,
            vocab_size=self.config.vocab_size,
            tie_word_embeddings=self.config.tie_word_embeddings,
        )
        self.lm_head = LMHead(lm_head_config, self.device)

    def setup_weights(self, state_dict: Dict[str, ttnn.Tensor]):
        """Setup model weights from state dict"""
        # Embedding weights
        self.embedding.setup_weight(state_dict["embedding.weight"])

        # Block weights
        for i, block in enumerate(self.blocks):
            block_weights = {
                "attention.wq": state_dict[f"layers.{i}.attention.wq"],
                "attention.wk": state_dict[f"layers.{i}.attention.wk"],
                "attention.wv": state_dict[f"layers.{i}.attention.wv"],
                "attention.wo": state_dict[f"layers.{i}.attention.wo"],
                "mlp.w1": state_dict[f"layers.{i}.mlp.w1"],
                "mlp.w2": state_dict[f"layers.{i}.mlp.w2"],
                "mlp.w3": state_dict.get(f"layers.{i}.mlp.w3"),  # Optional for gated
                "norm1.weight": state_dict[f"layers.{i}.norm1.weight"],
                "norm2.weight": state_dict[f"layers.{i}.norm2.weight"],
            }
            block.setup_weights(block_weights)

        # Final norm weight
        self.final_norm.setup_weight(state_dict["final_norm.weight"])

        # LM head weight
        if self.config.tie_word_embeddings:
            self.lm_head.setup_weight(state_dict["embedding.weight"])
        else:
            self.lm_head.setup_weight(state_dict["lm_head.weight"])

    def forward(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass of the model"""
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embedding(input_ids)

        # Position IDs
        if position_ids is None:
            position_ids = ttnn.arange(seq_len, device=self.device)
            position_ids = ttnn.broadcast_to(position_ids, [batch_size, seq_len])

        # Get RoPE embeddings
        cos, sin = self.rope(position_ids, seq_len)

        # Process through transformer blocks
        all_hidden_states = []
        all_self_attns = []
        present_key_values = []

        for i, block in enumerate(self.blocks):
            # Get past KV cache for this layer if available
            layer_past = past_key_values[i] if past_key_values else None

            # Forward through block
            hidden_states, layer_present = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                rotary_embeddings=(cos, sin),
                kv_cache=layer_past,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(layer_present)

            if self.optimization_config.output_hidden_states:
                all_hidden_states.append(hidden_states)

        # Final normalization
        hidden_states = self.final_norm(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Prepare outputs
        outputs = {
            "logits": logits,
            "hidden_states": all_hidden_states if all_hidden_states else None,
            "past_key_values": present_key_values if use_cache else None,
        }

        return outputs


def create_llama_model(
    model_size: str = "7b",
    device_arch: str = "wormhole_b0",
    optimization_level: str = "O2",
) -> Tuple[LLaMAModel, TransformerConfig]:
    """
    Factory function to create LLaMA models of different sizes.

    This demonstrates how model implementations can provide
    easy-to-use interfaces while leveraging TTTv2.
    """
    # Model configurations
    configs = {
        "7b": TransformerConfig(
            model_type="llama",
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
            hidden_act="silu",
            max_position_embeddings=4096,
            rope_theta=10000.0,
            norm_type="rmsnorm",
            layer_norm_eps=1e-6,
        ),
        "13b": TransformerConfig(
            model_type="llama",
            vocab_size=32000,
            hidden_size=5120,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=40,
            intermediate_size=13824,
            hidden_act="silu",
            max_position_embeddings=4096,
            rope_theta=10000.0,
            norm_type="rmsnorm",
            layer_norm_eps=1e-6,
        ),
        "70b": TransformerConfig(
            model_type="llama",
            vocab_size=32000,
            hidden_size=8192,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,  # GQA
            intermediate_size=28672,
            hidden_act="silu",
            max_position_embeddings=4096,
            rope_theta=10000.0,
            norm_type="rmsnorm",
            layer_norm_eps=1e-5,
        ),
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")

    config = configs[model_size]

    # Hardware configuration
    hw_config = HWConfig(
        device_arch=device_arch,
        num_devices=1,  # Can be increased for multi-chip
    )

    # Optimization configuration
    optimization_config = OptimizationConfig()
    if optimization_level == "O0":
        optimization_config.optimization_level = OptimizationLevel.O0
    elif optimization_level == "O1":
        optimization_config.optimization_level = OptimizationLevel.O1
    elif optimization_level == "O2":
        optimization_config.optimization_level = OptimizationLevel.O2
    elif optimization_level == "O3":
        optimization_config.optimization_level = OptimizationLevel.O3

    # Create model
    model = LLaMAModel(
        config=config,
        device=hw_config.mesh_device,
        optimization_config=optimization_config,
    )

    return model, config


class LLaMADemo:
    """
    Demo application showing how to use TTTv2 for inference.

    This would be part of the model-specific code, not TTTv2 core.
    """

    def __init__(
        self,
        model_path: str,
        model_size: str = "7b",
        device_arch: str = "wormhole_b0",
    ):
        # Create model
        self.model, self.config = create_llama_model(
            model_size=model_size,
            device_arch=device_arch,
        )

        # Setup hardware
        self.hw_config = HWConfig(device_arch=device_arch)

        # Load weights
        self._load_weights(model_path)

        # Create generator
        self.generator = StandardGenerator(
            model=self.model,
            tokenizer=self._load_tokenizer(),
            device=self.hw_config.mesh_device,
        )

    def _load_weights(self, model_path: str):
        """Load and setup model weights"""
        loader = WeightLoader(
            device=self.hw_config.mesh_device,
            dtype=ttnn.bfloat16,
        )

        # Load weights with conversion
        weights = loader.load_pretrained_weights(
            model_path,
            self.config,
            # Weight converter would handle HF -> TTT name mapping
        )

        self.model.setup_weights(weights)

    def _load_tokenizer(self):
        """Load tokenizer (placeholder)"""
        # Would load actual tokenizer
        return None

    def generate(self, prompt: str, max_length: int = 100):
        """Generate text from prompt"""
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        return self.generator.generate_from_prompt(
            prompt,
            generation_config=generation_config,
        )

    def run(self):
        """Run interactive demo"""
        print("LLaMA Demo using TTTv2")
        print("=" * 50)

        while True:
            prompt = input("\nEnter prompt (or 'quit' to exit): ")
            if prompt.lower() == "quit":
                break

            print("\nGenerating...")
            response = self.generate(prompt)
            print(f"\nResponse: {response}")


if __name__ == "__main__":
    # Example usage
    demo = LLaMADemo(
        model_path="/path/to/llama/weights",
        model_size="7b",
        device_arch="wormhole_b0",
    )
    demo.run()
