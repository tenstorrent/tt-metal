# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TT model configuration for Qwen3-TTS Talker (requires ttnn).

The Talker is essentially a causal LM with the same architecture as Qwen3-1.7B,
so we inherit from the shared ModelArgs and override config parsing to read from
the HF Qwen3-TTS config.json (which nests parameters under `talker_config`).

For ttnn-free configs (CodePredictor, SpeakerEncoder, Vocoder), see configs.py.
"""

import math
import os

from loguru import logger

from models.demos.qwen3_tts.tt.configs import CodePredictorConfig  # re-export
from models.tt_transformers.tt.model_config import ModelArgs


class TalkerModelArgs(ModelArgs):
    """
    ModelArgs subclass for the Qwen3-TTS Talker (main autoregressive LM).

    Overrides _set_hf_params to extract parameters from the talker_config
    section of the Qwen3-TTS HuggingFace config.json.
    """

    @staticmethod
    def _load_raw_config(checkpoint_dir):
        """Load config.json, falling back to HF Hub download if needed."""
        import json
        from pathlib import Path

        local_config = Path(checkpoint_dir) / "config.json"
        if local_config.exists():
            with open(local_config) as f:
                return json.load(f)

        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            checkpoint_dir,
            filename="config.json",
            local_files_only=os.getenv("CI") == "true",
        )
        with open(config_path) as f:
            return json.load(f)

    def _set_hf_params(self, checkpoint_dir):
        # Try AutoConfig first; fall back to raw JSON if transformers is too old
        try:
            from transformers import AutoConfig

            self.hf_config = AutoConfig.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
                local_files_only=os.getenv("CI") == "true",
            )
            raw_config = self.hf_config.to_dict()
        except (ValueError, ImportError):
            raw_config = self._load_raw_config(checkpoint_dir)
            self.hf_config = None
            logger.warning("AutoConfig failed for Qwen3-TTS; parsed config.json directly")

        talker_cfg = raw_config.get("talker_config", raw_config)

        self.dim = talker_cfg["hidden_size"]
        self.n_heads = talker_cfg["num_attention_heads"]
        self.n_kv_heads = talker_cfg["num_key_value_heads"]
        self.n_layers = talker_cfg["num_hidden_layers"]
        self.full_model_n_layers = self.n_layers
        self.head_dim = talker_cfg.get("head_dim", self.dim // self.n_heads)
        self.norm_eps = talker_cfg.get("rms_norm_eps", 1e-6)
        self.hidden_dim = talker_cfg["intermediate_size"]
        self.unpadded_hidden_dim = self.hidden_dim
        self.ffn_dim_multiplier = None
        self.multiple_of = None
        self.max_context_len = talker_cfg.get("max_position_embeddings", 32768)

        # Codec vocabulary (used for embedding and output head)
        self.vocab_size = talker_cfg["vocab_size"]  # 3072 for codec tokens
        tile_size = 32
        if self.num_devices == 0:
            self.padded_vocab_size = math.ceil(self.vocab_size / tile_size) * tile_size
        else:
            self.padded_vocab_size = math.ceil(self.vocab_size / (tile_size * self.num_devices)) * (
                tile_size * self.num_devices
            )
        self.pad_logits_to_power_of_2 = False

        # Text vocabulary (separate embedding for text input)
        self.text_vocab_size = talker_cfg.get("text_vocab_size", 151936)
        self.text_hidden_size = talker_cfg.get("text_hidden_size", 2048)

        # Code groups
        self.num_code_groups = talker_cfg.get("num_code_groups", 16)

        # RoPE — for non-streaming TTS all 3 MRoPE dims use sequential positions,
        # which is numerically identical to standard RoPE.
        self.rope_theta = talker_cfg.get("rope_theta", 1000000.0)
        rope_scaling = talker_cfg.get("rope_scaling", {})
        self.mrope_sections = rope_scaling.get("mrope_section", [24, 20, 20])
        self.mrope_interleaved = rope_scaling.get("interleaved", True)
        self.rope_scaling = None  # MRoPE handled separately
        self.rope_theta_local = None
        self.original_max_context_len = None
        self.use_sliding_window = False

        # Layer types and sliding window (disabled for Talker — all full attention)
        self.layer_types = None
        self.sliding_window = None
        self.sliding_window_pattern = [False] * self.n_layers

        # Special token IDs
        self.codec_bos_id = talker_cfg.get("codec_bos_id", 2149)
        self.codec_eos_token_id = talker_cfg.get("codec_eos_token_id", 2150)
        self.codec_pad_id = talker_cfg.get("codec_pad_id", 2148)
        self.codec_language_id = talker_cfg.get("codec_language_id", {})
        self.codec_think_id = talker_cfg.get("codec_think_id", 2154)
        self.codec_think_bos_id = talker_cfg.get("codec_think_bos_id", 2156)
        self.codec_think_eos_id = talker_cfg.get("codec_think_eos_id", 2157)
        self.codec_nothink_id = talker_cfg.get("codec_nothink_id", 2155)

        self.tts_bos_token_id = raw_config.get("tts_bos_token_id", 151672)
        self.tts_eos_token_id = raw_config.get("tts_eos_token_id", 151673)
        self.tts_pad_token_id = raw_config.get("tts_pad_token_id", 151671)

        # Speaker encoder dim
        spk_cfg = raw_config.get("speaker_encoder_config", {})
        self.spk_enc_dim = spk_cfg.get("enc_dim", 2048)

        # Code Predictor config (stored for Phase 2)
        self.code_predictor_config = talker_cfg.get("code_predictor_config", {})

        # Attention scaling (Qwen3 uses default head_dim**-0.5)
        self.query_pre_attn_scalar = None

        # Vision / multimodal flags (not applicable)
        self.is_multimodal = False
        self.vision_num_cross_attention_layers = 0
        self.image_token_index = None
        self.eos_token_id = None

        self.model_name = "Qwen3-TTS-12Hz-1.7B-Base"
        logger.info(
            f"Qwen3-TTS Talker config: dim={self.dim}, layers={self.n_layers}, "
            f"heads={self.n_heads}, kv_heads={self.n_kv_heads}, "
            f"ffn={self.hidden_dim}, codec_vocab={self.vocab_size}, "
            f"text_vocab={self.text_vocab_size}, code_groups={self.num_code_groups}"
        )

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "TransformerBlock": "",
            "Attention": "attention",
            "MLP": "feed_forward",
            "": "",
        }
        prefix = "talker."
        return prefix + layer_prefix + module_map.get(module_name, module_name)

    def load_state_dict(self):
        """Load Talker weights from safetensors, apply HF→meta key conversion.

        HF key structure (actual safetensors):
            talker.model.codec_embedding.weight      → talker.tok_embeddings.weight
            talker.model.text_embedding.weight        → (kept separately for host embedding)
            talker.model.layers.N.self_attn.q_proj    → talker.layers.N.attention.wq
            talker.model.layers.N.mlp.gate_proj       → talker.layers.N.feed_forward.w1
            talker.model.norm.weight                  → talker.norm.weight
            talker.codec_head.weight                  → talker.output.weight
            talker.text_projection.linear_fc1/fc2     → (kept for text projection MLP)

        Steps:
            1. Pre-remap: codec_embedding → tok_embeddings, codec_head → output
               (before map_hf_to_meta_keys which strips "model." and renames layers)
            2. convert_hf_to_meta_no_qkv_permute: handles model. strip + HF→meta key names
        """
        from models.tt_transformers.tt.load_checkpoints import (
            convert_hf_to_meta_no_qkv_permute,
            load_hf_state_dict_filtered,
        )

        state_dict = load_hf_state_dict_filtered(self.CKPT_DIR, ["talker."])

        # Pre-remap TTS-specific keys BEFORE the generic HF→meta conversion.
        # codec_embedding → tok_embeddings (so map_hf_to_meta_keys doesn't need to handle it)
        # codec_head → output (so LMHead can find it)
        pre_remapped = {}
        for k, v in state_dict.items():
            k = k.replace("codec_embedding", "tok_embeddings")
            k = k.replace("codec_head", "output")
            pre_remapped[k] = v

        # HF→meta key conversion: strips "model.", renames self_attn→attention, etc.
        state_dict = convert_hf_to_meta_no_qkv_permute(
            pre_remapped, self.head_dim, self.n_heads, self.n_kv_heads
        )

        return state_dict

    def weight_cache_path(self, dtype):
        return self.model_cache_path / f"talker_{dtype}"

    @property
    def base_model_name(self):
        return "Qwen3-TTS-1.7B"

    def get_code_predictor_config(self):
        return CodePredictorConfig.from_dict(self.code_predictor_config)


class CodePredictorModelArgs(TalkerModelArgs):
    """ModelArgs subclass for the Qwen3-TTS Code Predictor (5-layer Transformer).

    Inherits from TalkerModelArgs so we can share the same checkpoint path and
    most of the get_* infrastructure. Overrides dimensions to match the CP:
        hidden=1024, heads=16(q_dim=2048)/8(kv_dim=1024), ffn=3072, 5 layers.
    """

    def _set_hf_params(self, checkpoint_dir):
        super()._set_hf_params(checkpoint_dir)

        cp_cfg = self.code_predictor_config
        self.talker_hidden_size = self.dim

        self.dim = cp_cfg.get("hidden_size", 1024)
        self.n_heads = cp_cfg.get("num_attention_heads", 16)
        self.n_kv_heads = cp_cfg.get("num_key_value_heads", 8)
        self.n_layers = cp_cfg.get("num_hidden_layers", 5)
        self.full_model_n_layers = self.n_layers
        self.head_dim = cp_cfg.get("head_dim", 128)
        self.hidden_dim = cp_cfg.get("intermediate_size", 3072)
        self.unpadded_hidden_dim = self.hidden_dim
        self.norm_eps = cp_cfg.get("rms_norm_eps", 1e-6)
        self.max_context_len = cp_cfg.get("max_position_embeddings", 65536)

        self.vocab_size = cp_cfg.get("vocab_size", 2048)
        tile_size = 32
        if self.num_devices == 0:
            self.padded_vocab_size = math.ceil(self.vocab_size / tile_size) * tile_size
        else:
            self.padded_vocab_size = math.ceil(self.vocab_size / (tile_size * self.num_devices)) * (
                tile_size * self.num_devices
            )

        self.sliding_window_pattern = [False] * self.n_layers
        self.model_name = "Qwen3-TTS-CodePredictor"

        logger.info(
            f"Qwen3-TTS CodePredictor config: dim={self.dim}, layers={self.n_layers}, "
            f"heads={self.n_heads}, kv_heads={self.n_kv_heads}, ffn={self.hidden_dim}"
        )

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        """Return meta-style key prefix for Code Predictor components.

        After convert_hf_to_meta_no_qkv_permute, keys have 'model.' stripped:
            talker.code_predictor.layers.0.attention.wq.weight
        """
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "TransformerBlock": "",
            "Attention": "attention",
            "MLP": "feed_forward",
            "": "",
        }
        prefix = "talker.code_predictor."
        return prefix + layer_prefix + module_map.get(module_name, module_name)

    def load_state_dict(self):
        """Load Code Predictor weights from safetensors.

        HF key structure → meta key structure:
            talker.code_predictor.model.layers.N.self_attn.q_proj → talker.code_predictor.layers.N.attention.wq
            talker.code_predictor.model.norm.weight → talker.code_predictor.norm.weight
            talker.code_predictor.small_to_mtp_projection.* → kept as-is (no 'model.' prefix)
            talker.code_predictor.model.codec_embedding.{i}.weight → kept as-is
            talker.code_predictor.lm_head.{i}.weight → kept as-is (no 'model.' prefix)
        """
        from models.tt_transformers.tt.load_checkpoints import (
            convert_hf_to_meta_no_qkv_permute,
            load_hf_state_dict_filtered,
        )

        state_dict = load_hf_state_dict_filtered(self.CKPT_DIR, ["talker.code_predictor."])

        # Split: keys that go through HF→meta conversion vs keys kept as-is
        convertible_keys = {}
        passthrough_keys = {}
        for k, v in state_dict.items():
            if "model.layers." in k or k.endswith("model.norm.weight"):
                convertible_keys[k] = v
            else:
                passthrough_keys[k] = v

        if convertible_keys:
            converted = convert_hf_to_meta_no_qkv_permute(
                convertible_keys, self.head_dim, self.n_heads, self.n_kv_heads
            )
        else:
            converted = {}

        merged = {}
        merged.update(converted)
        merged.update(passthrough_keys)

        return merged

    def weight_cache_path(self, dtype):
        return self.model_cache_path / f"code_predictor_{dtype}"
