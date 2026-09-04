# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B ModelArgs: config + checkpoint loading.

Copied from ``gpt_oss_d_p/tt/model_config.py``, with its MXFP4 dequantization removed — Llama 3.1
ships plain bf16 safetensors with no quantization and no packed expert blocks, so the loader reads
the shards directly rather than going through HF ``from_pretrained`` (which would materialise a live
model alongside the weights for no benefit).

The one transformation that IS applied is ``convert_hf_qkv_to_meta_format``: q/k projection rows are
``reverse_permute``d so the on-device interleaved RoPE reproduces HF's half-split rotation. Skipping
it does not raise — it produces a model whose Q/K are rotated in the wrong basis, which shows up as
a mid-0.x attention PCC.

Key layout: the checkpoint's keys are already ``model.*`` / ``lm_head.*``, matching what ``tt/model.py``
substates, so there is no key remapping.
"""

import gc
import json
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.llama3_1_8b_d_p.reference.config import CONFIG_DIR, LlamaConfig
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

# Bundled config.json — the config path needs no network and no checkpoint.
DEFAULT_CONFIG_DIR = str(CONFIG_DIR)


class ModelArgs:
    """Config + weight loading for the Llama 3.1 8B prefill package."""

    def __init__(
        self,
        mesh_device,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=128 * 1024,
    ):
        self.mesh_device = mesh_device
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        # HF_MODEL points at a real checkpoint dir; without it, the bundled config-only dir is used.
        hf_model = os.getenv("HF_MODEL") or DEFAULT_CONFIG_DIR
        self.model_path = hf_model
        self.weights_path = hf_model
        logger.info(
            f"Llama 3.1 8B model dir: {self.model_path}"
            f"{' (dummy weights — no checkpoint load)' if self.dummy_weights else ''}"
        )

        self.hf_config = LlamaConfig.from_json(Path(self.model_path) / "config.json")
        self.vocab_size = self.hf_config.vocab_size
        self.n_layers = self.hf_config.num_hidden_layers
        self.head_dim = self.hf_config.head_dim
        self.rope_theta = self.hf_config.rope_theta
        self.rope_scaling = self.hf_config.rope_scaling
        self.model_name = Path(self.model_path).name
        self.max_context_len = max_seq_len

        self.tokenizer = None
        if not self.dummy_weights:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.weights_path)
            except Exception as e:  # a config-only dir has no tokenizer; not an error for prefill
                logger.warning(f"No tokenizer at {self.weights_path} ({e}); tokenizer disabled")

    @staticmethod
    def load_state_dict(weights_path, dummy_weights=False, convert_to_meta_format=True):
        """Load the Llama 3.1 8B state dict from safetensors.

        Keys are already ``model.*`` / ``lm_head.*`` — no wrapper prefix to strip. Unless disabled,
        q/k projections are converted to Meta format for the on-device RoPE.
        """
        if dummy_weights:
            return {}
        state_dict = ModelArgs._load_safetensors(weights_path)
        if convert_to_meta_format:
            cfg = LlamaConfig.from_json(Path(weights_path) / "config.json")
            logger.info("Converting q/k projections from HuggingFace to Meta format for RoPE (full rotary)")
            state_dict = convert_hf_qkv_to_meta_format(state_dict, cfg.head_dim)
        return state_dict

    @staticmethod
    def _load_safetensors(weights_path):
        """Read every ``*.safetensors`` shard into one dict, keeping bf16.

        Deliberately NOT ``AutoModelForCausalLM.from_pretrained``: that path exists in the gpt-oss
        donor only to make transformers dequantize MXFP4 experts on the host. Llama has nothing to
        dequantize, and from_pretrained would hold a live nn.Module alongside the weights.
        """
        from safetensors.torch import load_file

        weights_path = Path(weights_path)
        shards = sorted(weights_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(
                f"no *.safetensors under {weights_path}. Set HF_MODEL to a downloaded "
                f"Llama-3.1-8B-Instruct checkpoint directory."
            )
        state_dict = {}
        for shard in shards:
            logger.info(f"Loading {shard.name}")
            state_dict.update(load_file(str(shard)))
        gc.collect()

        # Cast any fp32 stragglers; a no-op for a bf16 checkpoint.
        if any(v.dtype == torch.float32 for v in state_dict.values()):
            state_dict = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in state_dict.items()}
        return state_dict

    def weight_cache_path(self, dtype):
        """Weight-cache dir for this model + mesh shape. Keyed on dtype AND mesh: a cache built for a
        different TP split holds differently-sharded tensors and must not be reused."""
        cache_dir = os.getenv("TT_CACHE_PATH")
        cache_dir = Path(cache_dir) if cache_dir else Path(self.model_path)
        dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8", ttnn.bfloat4_b: "bfp4"}[dtype]
        cache_path = cache_dir / f"tensor_cache_{dtype_str}_{tuple(self.mesh_device.shape)}"
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Weight cache: {cache_path}")
        return cache_path

    def get_state_dict_prefix(self, prefix, layer_idx):
        if layer_idx is None:
            return prefix
        return f"{prefix}layers.{layer_idx}."
