# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice Model Configuration for tt_transformers framework.

CosyVoice3's LLM backbone is a Qwen2-0.5B model. This module provides:
  - CosyVoiceModelArgs: Extends ModelArgs to configure the framework for
    CosyVoice's non-standard weight layout (stored as llm.pt, not HF format).
  - Weight key remapping from CosyVoice's naming convention to what
    tt_transformers expects.
"""

import math
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn

# ---------------------------------------------------------------------------
# Weight key remapping
# ---------------------------------------------------------------------------


def remap_cosyvoice_llm_state_dict(state_dict):
    """
    Remap CosyVoice LLM state dict keys to the tt_transformers convention.

    CosyVoice stores Qwen2 weights under these prefixes:
        llm.model.model.embed_tokens.weight          -> tok_embeddings.weight
        llm.model.model.layers.N.self_attn.q_proj.*   -> layers.N.attention.wq.*
        llm.model.model.layers.N.self_attn.k_proj.*   -> layers.N.attention.wk.*
        llm.model.model.layers.N.self_attn.v_proj.*   -> layers.N.attention.wv.*
        llm.model.model.layers.N.self_attn.o_proj.*   -> layers.N.attention.wo.*
        llm.model.model.layers.N.mlp.gate_proj.*      -> layers.N.feed_forward.w1.*
        llm.model.model.layers.N.mlp.up_proj.*        -> layers.N.feed_forward.w3.*
        llm.model.model.layers.N.mlp.down_proj.*      -> layers.N.feed_forward.w2.*
        llm.model.model.layers.N.input_layernorm.*    -> layers.N.attention_norm.*
        llm.model.model.layers.N.post_attention_layernorm.* -> layers.N.ffn_norm.*
        llm.model.model.norm.*                        -> norm.*
        llm.model.lm_head.*                           -> output.*

    CosyVoice-specific keys (NOT remapped — extracted separately):
        speech_embedding.weight
        llm_decoder.weight
    """
    remapped = {}
    cosyvoice_specific = {}

    # Prefix to strip from Qwen2 backbone keys
    qwen2_prefix = "llm.model.model."
    lm_head_prefix = "llm.model.lm_head."

    for key, value in state_dict.items():
        # --- CosyVoice-specific keys (speech embedding, decoder head) ---
        if key.startswith("speech_embedding.") or key.startswith("llm_decoder."):
            cosyvoice_specific[key] = value
            continue

        # --- Qwen2 LM head ---
        if key.startswith(lm_head_prefix):
            suffix = key[len(lm_head_prefix) :]
            remapped[f"output.{suffix}"] = value
            continue

        # --- Qwen2 backbone ---
        if not key.startswith(qwen2_prefix):
            # Keep other keys as-is (e.g., criterion, sampling-related)
            cosyvoice_specific[key] = value
            continue

        suffix = key[len(qwen2_prefix) :]

        # Embedding
        if suffix.startswith("embed_tokens."):
            new_key = suffix.replace("embed_tokens.", "tok_embeddings.")
            remapped[new_key] = value
            continue

        # Final norm
        if suffix.startswith("norm."):
            remapped[suffix] = value
            continue

        # Transformer layers
        if suffix.startswith("layers."):
            # Extract layer number and remainder
            parts = suffix.split(".", 2)  # ['layers', 'N', 'rest']
            layer_num = parts[1]
            rest = parts[2]

            # Attention projections
            attn_map = {
                "self_attn.q_proj.": "attention.wq.",
                "self_attn.k_proj.": "attention.wk.",
                "self_attn.v_proj.": "attention.wv.",
                "self_attn.o_proj.": "attention.wo.",
            }
            # MLP projections
            mlp_map = {
                "mlp.gate_proj.": "feed_forward.w1.",
                "mlp.up_proj.": "feed_forward.w3.",
                "mlp.down_proj.": "feed_forward.w2.",
            }
            # Norms
            norm_map = {
                "input_layernorm.": "attention_norm.",
                "post_attention_layernorm.": "ffn_norm.",
            }

            mapped = False
            for old_prefix, new_prefix in {**attn_map, **mlp_map, **norm_map}.items():
                if rest.startswith(old_prefix):
                    new_rest = rest.replace(old_prefix, new_prefix, 1)
                    remapped[f"layers.{layer_num}.{new_rest}"] = value
                    mapped = True
                    break

            if not mapped:
                logger.warning(f"Unmapped key in layer: {key}")
                cosyvoice_specific[key] = value
            continue

        # Anything else
        logger.warning(f"Unmapped Qwen2 key: {key}")
        cosyvoice_specific[key] = value

    logger.info(f"Remapped {len(remapped)} Qwen2 keys, " f"{len(cosyvoice_specific)} CosyVoice-specific keys")
    return remapped, cosyvoice_specific


# ---------------------------------------------------------------------------
# Qwen2 configuration for CosyVoice3-0.5B
# ---------------------------------------------------------------------------

# These match Qwen2-0.5B as used inside CosyVoice3
COSYVOICE_QWEN2_CONFIG = {
    "dim": 896,
    "n_layers": 24,
    "n_heads": 14,
    "n_kv_heads": 2,
    "head_dim": 64,
    "hidden_dim": 4864,  # intermediate_size
    "vocab_size": 151936,  # Qwen2 tokenizer vocab
    "norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_seq_len": 4096,
    "speech_token_size": 6561,  # FSQ codebook size
}


from models.tt_transformers.tt.model_config import ModelArgs


class CosyVoiceModelConfig(ModelArgs):
    """
    Configuration container for CosyVoice3's Qwen2-0.5B backbone.

    This extends ModelArgs to provide all the parameters that tt_transformers
    needs to instantiate Transformer layers, while handling CosyVoice's
    specific checkpoint structure.
    """

    def __init__(
        self,
        mesh_device,
        max_batch_size=1,
        max_seq_len=None,
        weights_dir=None,
    ):
        # We must set HF_MODEL before calling super().__init__ because ModelArgs requires it
        if not os.getenv("HF_MODEL"):
            os.environ["HF_MODEL"] = "Qwen/Qwen2-0.5B"

        # Override Qwen2-0.5B architecture params since we don't rely on HF AutoConfig
        self.dim = COSYVOICE_QWEN2_CONFIG["dim"]
        self.n_layers = COSYVOICE_QWEN2_CONFIG["n_layers"]
        self.n_heads = COSYVOICE_QWEN2_CONFIG["n_heads"]
        self.n_kv_heads = COSYVOICE_QWEN2_CONFIG["n_kv_heads"]
        self.head_dim = COSYVOICE_QWEN2_CONFIG["head_dim"]
        self.hidden_dim = COSYVOICE_QWEN2_CONFIG["hidden_dim"]
        self.vocab_size = COSYVOICE_QWEN2_CONFIG["vocab_size"]
        self.norm_eps = COSYVOICE_QWEN2_CONFIG["norm_eps"]
        self.rope_theta = COSYVOICE_QWEN2_CONFIG["rope_theta"]
        self.model_name = "Qwen2-0.5B"
        self.is_multimodal = False

        self.rope_scaling = None  # Qwen2-0.5B doesn't use RoPE scaling
        self.rope_theta_local = None  # No local RoPE variant
        self.layer_types = ["attention"] * self.n_layers  # No sliding window attention
        self.sliding_window_pattern = [False] * self.n_layers
        self.query_pre_attn_scalar = None
        self.mlp_activation_type = ttnn.UnaryOpType.SILU
        self.unpadded_hidden_dim = self.hidden_dim
        self.use_sliding_window = False
        self.padded_vocab_size = self.vocab_size
        self.dummy_weights = False
        self.multiple_of = None
        self.ffn_dim_multiplier = None

        super().__init__(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len or COSYVOICE_QWEN2_CONFIG["max_seq_len"],
        )

        # CosyVoice speech-specific params
        self.speech_token_size = COSYVOICE_QWEN2_CONFIG["speech_token_size"]
        self.speech_vocab_size = self.speech_token_size + 200  # includes special tokens
        self.sos_token = self.speech_token_size + 0
        self.eos_token = self.speech_token_size + 1
        self.task_id_token = self.speech_token_size + 2
        self.fill_token = self.speech_token_size + 3

        # Derived params
        self.n_local_heads = self.n_heads  # No TP for 0.5B
        self.n_local_kv_heads = self.n_kv_heads
        self.padded_head_dim = math.ceil(self.head_dim / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)

        # Weight paths
        self.weights_dir = Path(weights_dir) if weights_dir else None

        # Compute kernel configs (set up when device is available)
        if mesh_device:
            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        else:
            self.compute_kernel_config_hifi2 = None

    def _set_hf_params(self, checkpoint_dir):
        # Override to do nothing, since we set parameters manually in __init__
        pass

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if module_name == "lm_head":
            return "output"

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",
        }

        return layer_prefix + module_map.get(module_name, "")

    def load_llm_weights(self):
        """Load and remap LLM weights from CosyVoice checkpoint."""
        if self.weights_dir is None:
            raise ValueError("weights_dir not set")
        llm_path = self.weights_dir / "llm.pt"
        logger.info(f"Loading LLM weights from {llm_path}")
        raw_state_dict = torch.load(llm_path, map_location="cpu", weights_only=True)
        qwen2_state_dict, cosyvoice_keys = remap_cosyvoice_llm_state_dict(raw_state_dict)
        return qwen2_state_dict, cosyvoice_keys

    def weight_cache_path(self, dtype=ttnn.bfloat16):
        """Return cache path for converted TTNN weights."""
        if self.weights_dir is None:
            return None
        cache_dir = self.weights_dir / "tt_cache" / str(dtype)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
