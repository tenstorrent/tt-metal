# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
vLLM integration for Qwen3-Coder-Next on T3K.

Registered as TTQwen3NextForCausalLM in tt-vllm-plugin/__init__.py:
    ModelRegistry.register_model(
        "TTQwen3NextForCausalLM",
        "models.demos.qwen3_coder_next.tt.generator_vllm:Qwen3NextForCausalLM",
    )

Serving: --override-tt-config '{"architectures": ["TTQwen3NextForCausalLM"]}'

Hybrid state management:
- GQA layers (12/48): paged KV cache via vLLM
- DeltaNet layers (36/48): fixed-size recurrent state per batch slot
"""


class Qwen3NextForCausalLM:
    """vLLM-compatible wrapper for Qwen3-Coder-Next."""

    model_capabilities = {
        "supports_prefix_caching": False,  # DeltaNet recurrent state doesn't support prefix caching
    }

    def __init__(self, model, model_args, mesh_device):
        self.model = model
        self.model_args = model_args
        self.mesh_device = mesh_device
        self._layer_states = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = "performance",
    ):
        """Initialize model for vLLM serving.

        Called by TTModelLoader.load_model() during server startup.
        """
        from models.demos.qwen3_coder_next.tt.tt_model_config import Qwen3CoderNextTTConfig

        config = Qwen3CoderNextTTConfig.from_hf_config_dict(
            hf_config.to_dict(),
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        if n_layers is not None:
            config.num_hidden_layers = n_layers

        # TODO: Build full TTNN Transformer model
        # from models.tt_transformers.tt.model import Transformer
        # state_dict = load_state_dict(...)
        # model = Transformer(args=config, mesh_device=mesh_device, ...)
        model = None  # Placeholder until full integration

        return cls(model, config, mesh_device)

    def prefill_forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        """Prefill: process prompt tokens."""
        # DeltaNet layers: sequential token-by-token (updates recurrent state)
        # GQA layers: parallel SDPA
        # MoE layers: parallel expert routing
        raise NotImplementedError("Prefill requires full model integration (Phase 5)")

    def decode_forward(self, input_ids, position_ids=None, **kwargs):
        """Decode: single token generation step."""
        raise NotImplementedError("Decode requires full model integration (Phase 5)")

    def allocate_kv_cache(self, *args, **kwargs):
        """Allocate KV cache for GQA layers only (12 of 48).
        DeltaNet layers use fixed-size recurrent state instead."""

    @property
    def cache_path(self):
        return getattr(self.model_args, "model_cache_path", None)
