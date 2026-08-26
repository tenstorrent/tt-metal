# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5-128B ModelArgs: HF config + weight loading + cache paths.

Mirrors ``gpt_oss_d_p/tt/model_config.py``. The checkpoint ingest itself (multimodal prefix strip,
per-tensor fp8 dequant, Meta RoPE swizzle, mechanism guards) lives in :mod:`checkpoint`, which is
pure torch and host-testable — see ``tests/unit/test_checkpoint_ingest.py``.
"""

import os
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.mistral_medium_d_p.tt.checkpoint import assert_supported, load_state_dict

# Bundled flattened text config (no network / checkpoint needed for the config-only path).
DEFAULT_HF_MODEL = "models/demos/mistral_medium_d_p/configs/Mistral-Medium-3.5-128B"


class ModelArgs:
    """Config + weight-loading front end for the Mistral-Medium-3.5 prefill stack."""

    def __init__(self, mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=1024 * 128, cache_hf=False):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.cache_hf = cache_hf

        hf_model = os.getenv("HF_MODEL") or DEFAULT_HF_MODEL
        self.model_path = hf_model
        self.weights_path = hf_model
        logger.info(
            f"Using Mistral-Medium-3.5 model from: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        # The published repo is Mistral3ForConditionalGeneration; the text backbone is text_config.
        # The bundled config is already flattened, so this is a no-op there.
        self.hf_config = getattr(cfg, "text_config", cfg)
        # Fail at construction time on any mechanism this stack does not implement, rather than at
        # accuracy-debug time three days later.
        assert_supported(self.hf_config)

        self.vocab_size = self.hf_config.vocab_size
        self.n_layers = self.hf_config.num_hidden_layers
        self.head_dim = getattr(
            self.hf_config, "head_dim", self.hf_config.hidden_size // self.hf_config.num_attention_heads
        )
        self.max_prefill_chunk_size = 128 * 1024
        self.model_name = Path(self.model_path).name
        self.max_context_len = max_seq_len

        if self.dummy_weights:
            self.tokenizer = None
        else:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True)
            except Exception as e:  # a config-only dir has no tokenizer
                logger.warning(f"No tokenizer at {self.weights_path} ({e}); tokenizer disabled")
                self.tokenizer = None

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True, head_dim=128):
        """Load + ingest the checkpoint. See :func:`checkpoint.load_state_dict`."""
        if dummy_weights:
            return {}
        return load_state_dict(weights_path, convert_to_meta_format=convert_to_meta_format, head_dim=head_dim)

    def weight_cache_path(self, dtype):
        """Weight-cache dir for this model + mesh."""
        cache_dir = os.getenv("TT_CACHE_PATH")
        cache_dir = Path(cache_dir) if cache_dir else Path(self.model_path)
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[dtype]
        cache_path = cache_dir / f"tensor_cache_{dtype_str}_{self.mesh_device.shape}"
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Weight cache: {cache_path}")
        return cache_path

    def get_state_dict_prefix(self, prefix, layer_idx):
        return prefix if layer_idx is None else f"{prefix}layers.{layer_idx}."
