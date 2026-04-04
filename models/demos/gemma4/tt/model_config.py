# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os

from loguru import logger

import ttnn
from models.tt_transformers.tt.common import calculate_prefill_warmup_seq_lens, cap_seq_lens_to_max_prefill_chunk_size
from models.tt_transformers.tt.load_checkpoints import map_hf_to_meta_keys, split_hf_keys
from models.tt_transformers.tt.model_config import MathFidelitySetting
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs
from models.tt_transformers.tt.model_config import OpGroup, PrecisionSetting, TensorGroup

# File names for performance and accuracy mode override files
PERFORMANCE_DECODER_CONFIG_FILENAME = "performance_decoder_config.json"
ACCURACY_DECODER_CONFIG_FILENAME = "accuracy_decoder_config.json"


class ModelArgs(TTModelArgs):
    """
    Model configuration for Google Gemma 4 E4B (text-only).

    Key differences from Gemma 3:
    - Dual head_dim: 256 for sliding attention, 512 for global attention
    - Partial rotary embeddings (25% of dims for global attention)
    - Per-layer input gating mechanism
    - KV cache sharing (last 18 of 42 layers share KV)
    - Final logit soft-capping (tanh(logits/30)*30)
    - V-norm (RMSNorm on V without learnable scale)
    - Per-attention-type rope_parameters (different theta for sliding vs global)
    - embed_tokens_per_layer + per_layer_model_projection
    """

    OP_KEYS = (
        # Embedding
        "EMB_WEIGHTS",
        # Feed forward
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        # Attention
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "ATTN_OUTPUT",
        "ATTN_W_LAYOUT",
        # Decoder
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
    )

    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
    ):
        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            cache_hf=cache_hf,
            use_hf_rope=True,  # Gemma 4 uses HF-style rotary embeddings
        )

        # Gemma 4 does not use fused QK ops (rotary embedding + paged cache update)
        self.use_qk_fused = False
        self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.padded_vocab_size = 262144  # Gemma 4 vocab size (no padding needed, already power of 2)

        # Force BF16 for LM head output during bring-up (default bfloat8_b destroys logit precision)
        self.lm_head_dtype = ttnn.bfloat16

        # Force BF16 for ALL weights AND activations during bring-up to minimize numerical error.
        # Default settings use BFP8 for all weights and bfloat8_b for MLP activations,
        # which causes too much cumulative error across 42 layers with layer_scalar.
        for dec_id, dec_opt in self.optimizations.decoder_optimizations.items():
            # All weight tensors to BF16
            dec_opt._opt_settings["TensorPrecision"][TensorGroup.FF1_FF3] = PrecisionSetting.BF16
            dec_opt._opt_settings["TensorPrecision"][TensorGroup.FF2] = PrecisionSetting.BF16
            dec_opt._opt_settings["TensorPrecision"][TensorGroup.WQKV] = PrecisionSetting.BF16
            dec_opt._opt_settings["TensorPrecision"][TensorGroup.WO] = PrecisionSetting.BF16
            dec_opt._opt_settings["TensorPrecision"][TensorGroup.KV_CACHE] = PrecisionSetting.BF16
            dec_opt._opt_settings["TensorPrecision"][TensorGroup.ACTIVATION] = PrecisionSetting.BF16
            # Use HIFI4 for ALL ops to reduce rounding error
            dec_opt._opt_settings["OpFidelity"][OpGroup.LI_FF1_FF3] = MathFidelitySetting.HIFI4
            dec_opt._opt_settings["OpFidelity"][OpGroup.LI_FF2] = MathFidelitySetting.HIFI4
            dec_opt._opt_settings["OpFidelity"][OpGroup.LI_QKV_PREFILL] = MathFidelitySetting.HIFI4
            dec_opt._opt_settings["OpFidelity"][OpGroup.LI_O_PREFILL] = MathFidelitySetting.HIFI4
            dec_opt._opt_settings["OpFidelity"][OpGroup.SDPA_PREFILL] = MathFidelitySetting.HIFI4
            dec_opt._opt_settings["OpFidelity"][OpGroup.LI_QKV_DECODE] = MathFidelitySetting.HIFI4
            dec_opt._opt_settings["OpFidelity"][OpGroup.LI_O_DECODE] = MathFidelitySetting.HIFI4
            dec_opt._opt_settings["OpFidelity"][OpGroup.SDPA_DECODE] = MathFidelitySetting.HIFI4

        # Override compute_kernel_config_hifi4 for Wormhole: use HiFi3 with fp32 accumulation.
        # Wormhole has a known bug where HiFi4 + fp32_dest_acc_en=True produces WORSE accuracy
        # than HiFi3 + fp32_dest_acc_en=True. This config is used for SDPA.
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _set_model_specific_params(self):
        """Set Gemma 4 specific parameters."""
        # Gemma 4 RMSNorm uses direct scale (weight * normed), NOT the Gemma 2 offset pattern ((1+weight) * normed).
        # HF Gemma4RMSNorm initializes weights to ones and multiplies directly; checkpoint stores actual scale values.
        # Setting add_unit_offset=False avoids incorrectly adding 1 to already-correct weights.
        self.rms_norm_add_unit_offset = False
        self.embed_scale = self.dim**0.5

        # Parse Gemma 4 specific config values from the HF config
        if hasattr(self, "hf_config"):
            config = self.hf_config.to_dict()
            text_config = config.get("text_config", config)
        else:
            text_config = {}

        # Dual head_dim: global_head_dim for full attention layers
        self.global_head_dim = text_config.get("global_head_dim", self.head_dim)

        # Per-layer head dims (computed from layer_types)
        self.layer_head_dims = []
        if self.layer_types:
            for lt in self.layer_types:
                if lt == "full_attention":
                    self.layer_head_dims.append(self.global_head_dim)
                else:
                    self.layer_head_dims.append(self.head_dim)
        else:
            self.layer_head_dims = [self.head_dim] * self.n_layers

        # QKV sizes per layer (for attention weight loading)
        self.layer_qkv_sizes = []
        for hd in self.layer_head_dims:
            q_size = self.n_heads * hd
            kv_size = self.n_kv_heads * hd
            self.layer_qkv_sizes.append((q_size, kv_size, kv_size))

        # Final logit soft-capping
        self.final_logit_softcapping = text_config.get("final_logit_softcapping", None)

        # KV cache sharing
        self.num_kv_shared_layers = text_config.get("num_kv_shared_layers", 0)
        self.first_kv_shared_layer_idx = (
            self.n_layers - self.num_kv_shared_layers if self.num_kv_shared_layers > 0 else self.n_layers
        )

        # Build KV sharing map: for each shared layer, which non-shared layer provides KV
        self.kv_sharing_map = {}  # shared_layer_idx -> source_layer_idx
        if self.num_kv_shared_layers > 0:
            self._build_kv_sharing_map()

        # Per-layer input gating
        self.hidden_size_per_layer_input = text_config.get("hidden_size_per_layer_input", 0)

        # Rope parameters per attention type
        rope_parameters = text_config.get("rope_parameters", {})
        if rope_parameters:
            sliding_rope = rope_parameters.get("sliding_attention", {})
            full_rope = rope_parameters.get("full_attention", {})

            # Set local (sliding) rope theta
            self.rope_theta_local = sliding_rope.get("rope_theta", 10000.0)
            # Set global rope theta
            self.rope_theta = full_rope.get("rope_theta", 1000000.0)
            # Partial rotary factor for global attention
            self.partial_rotary_factor = full_rope.get("partial_rotary_factor", 1.0)
        else:
            self.partial_rotary_factor = 1.0

        # V-norm: applied without learnable scale (detected from state dict later)
        self.use_v_norm = True  # Gemma 4 always uses V-norm

        # Override query_pre_attn_scalar for Gemma 4
        # Gemma 4 uses head_dim (not a separate scalar) for attention scaling
        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim

        logger.info(
            f"Gemma 4 E4B config: dim={self.dim}, n_layers={self.n_layers}, "
            f"head_dim={self.head_dim}, global_head_dim={self.global_head_dim}, "
            f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, "
            f"num_kv_shared_layers={self.num_kv_shared_layers}, "
            f"hidden_size_per_layer_input={self.hidden_size_per_layer_input}, "
            f"final_logit_softcapping={self.final_logit_softcapping}, "
            f"partial_rotary_factor={self.partial_rotary_factor}"
        )

    def _build_kv_sharing_map(self):
        """
        Build a map from shared layers to their KV source layers.

        In Gemma 4, the last `num_kv_shared_layers` layers share KV caches.
        Each shared layer uses KV from the last non-shared layer of the same type
        (sliding_attention or full_attention).
        """
        # Find the last non-shared layer of each type
        last_non_shared = {}  # layer_type -> last non-shared layer index
        for i in range(self.first_kv_shared_layer_idx):
            lt = self.layer_types[i]
            last_non_shared[lt] = i

        # Map each shared layer to its source
        for i in range(self.first_kv_shared_layer_idx, self.n_layers):
            lt = self.layer_types[i]
            if lt in last_non_shared:
                self.kv_sharing_map[i] = last_non_shared[lt]
            else:
                logger.warning(f"Layer {i} (type={lt}) has no non-shared source for KV sharing")

        logger.info(f"KV sharing map: {self.kv_sharing_map}")

    def get_warmup_prefill_supported_seq_lens(self):
        DEFAULT_VALUE = self.capped_warmup_seq_len

        model_specific_ceil_warmup_lengths = {
            "gemma-4-E4B": 2048,
        }

        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(self.base_model_name, DEFAULT_VALUE)
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )

        return to_warmup_seq_lens

    def get_trace_prefill_supported_seq_lens(self):
        default_supported_seq_lens = {
            "N150": [],
            "N300": [],
            "T3K": [],
            "TG": [],
        }

        model_specific_supported_seq_lens = {}

        model_name = self.base_model_name
        device_name = self.device_name

        result = model_specific_supported_seq_lens.get(model_name, {}).get(
            device_name, default_supported_seq_lens.get(device_name)
        )

        if result is not None:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)
        else:
            return []

    def _set_params_from_dict(self, config):
        """Override to handle num_devices=0 for CPU-only config testing."""
        # Temporarily set num_devices=1 if it's 0, to avoid compute_padded_vocab_size error
        orig_num_devices = self.num_devices
        if self.num_devices == 0:
            self.num_devices = 1
        super()._set_params_from_dict(config)
        self.num_devices = orig_num_devices
        # Override padded vocab size for Gemma 4 (262144 is already tile-aligned)
        self.padded_vocab_size = 262144

    def _set_hf_params(self, checkpoint_dir):
        """Override to handle Gemma 4's merged text/vision/audio config."""

        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            # Merge non-nested keys into text_config (skip vision_config, audio_config)
            text_config.update(
                {k: v for k, v in base_config.items() if k not in ["text_config", "vision_config", "audio_config"]}
            )
            return text_config

        from transformers import AutoConfig

        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for Gemma 4 models yet.")
        else:
            self.hf_config = AutoConfig.from_pretrained(self.CKPT_DIR)

        config = self.hf_config.to_dict()

        if "text_config" in config:
            merged_text_config = merge_text_config(config)
            self._set_params_from_dict(merged_text_config)
        else:
            self._set_params_from_dict(config)

    def load_state_dict(self):
        """
        Load and convert Gemma 4 state dict.

        Handles:
        - model.language_model.* prefix stripping
        - Standard HF to Meta key conversion (q_proj->wq, etc.)
        - Gemma 4 specific keys (per_layer_input_gate, embed_tokens_per_layer, etc.)
        """
        if self.dummy_weights:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.CKPT_DIR)
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.n_layers

            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_config(config)
            state_dict = model.state_dict()
        else:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                self.CKPT_DIR,
                torch_dtype="auto",
                local_files_only=os.getenv("CI") == "true",
            )
            state_dict = model.state_dict()

        # Separate text, vision, and other keys
        text_state_dict = {}
        other_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("model.language_model.") or k.startswith("lm_head"):
                text_state_dict[k] = v
            else:
                # Skip vision/audio keys for text-only bring-up
                other_state_dict[k] = v

        # Extract and preserve Gemma 4 specific keys before standard conversion
        gemma4_keys = {}
        text_keys_to_remove = []

        gemma4_prefixes = [
            "per_layer_input_gate",
            "per_layer_projection",
            "post_per_layer_input_norm",
            "layer_scalar",
            "embed_tokens_per_layer",
            "per_layer_model_projection",
            "per_layer_projection_norm",
        ]
        for k, v in text_state_dict.items():
            for prefix in gemma4_prefixes:
                if prefix in k:
                    new_key = k.replace("model.language_model.", "")
                    gemma4_keys[new_key] = v
                    text_keys_to_remove.append(k)
                    break

        for k in set(text_keys_to_remove):  # Use set to avoid double-delete
            text_state_dict.pop(k, None)

        # Handle tied embeddings: Gemma 4 uses tie_word_embeddings=true
        # The lm_head weight IS the embed_tokens weight
        embed_key = "model.language_model.embed_tokens.weight"
        lm_head_key = "lm_head.weight"
        if embed_key in text_state_dict and lm_head_key not in text_state_dict:
            text_state_dict[lm_head_key] = text_state_dict[embed_key]
            del text_state_dict[embed_key]

        # For HF-style RoPE: skip QKV permutation (Gemma 4 uses HF-style RoPE)
        # Note: Gemma 4 has per-layer head_dim, so we need to handle this carefully
        # For now, use the default head_dim for the conversion (sliding head_dim)
        # The per-layer head_dim handling will be done in the attention layer
        text_state_dict = split_hf_keys(text_state_dict, self.n_heads, self.n_kv_heads)
        text_state_dict = map_hf_to_meta_keys(text_state_dict)

        # Merge back the Gemma 4 specific keys
        text_state_dict.update(gemma4_keys)

        # Remove layers beyond n_layers (if any)
        keys_to_remove = []
        for k in text_state_dict.keys():
            for i in range(self.n_layers, self.full_model_n_layers):
                if f"layers.{i}." in k:
                    keys_to_remove.append(k)
                    break
        for k in keys_to_remove:
            del text_state_dict[k]

        return text_state_dict

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        """Override to handle Gemma 4 state dict key prefixes."""
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        text_module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",
        }

        return layer_prefix + text_module_map[module_name]

    def get_layer_head_dim(self, layer_num):
        """Get the head_dim for a specific layer (varies between sliding and global attention)."""
        if layer_num < len(self.layer_head_dims):
            return self.layer_head_dims[layer_num]
        return self.head_dim

    def get_layer_qkv_size(self, layer_num):
        """Get the total QKV size for a specific layer."""
        hd = self.get_layer_head_dim(layer_num)
        return self.n_heads * hd + 2 * self.n_kv_heads * hd

    def is_kv_shared_layer(self, layer_num):
        """Check if a layer uses shared KV cache."""
        return layer_num in self.kv_sharing_map

    def get_kv_source_layer(self, layer_num):
        """Get the source layer index for KV sharing."""
        return self.kv_sharing_map.get(layer_num, layer_num)

    def get_layer_intermediate_size(self, layer_num):
        """Get intermediate_size for a layer.
        Note: E4B does NOT use double-wide MLP (use_double_wide_mlp=False).
        Larger models may have 2x intermediate_size for KV-shared layers.
        """
        return self.hidden_dim

    def get_layer_rotary_dim(self, layer_num):
        """Get the number of dimensions that receive RoPE for a specific layer."""
        hd = self.get_layer_head_dim(layer_num)
        lt = self.layer_types[layer_num] if self.layer_types else "sliding_attention"
        if lt == "full_attention":
            return int(hd * self.partial_rotary_factor)
        return hd  # Sliding layers get full rotation

    def is_sliding_layer(self, layer_num):
        """Check if a layer uses sliding window attention."""
        if self.layer_types:
            return self.layer_types[layer_num] == "sliding_attention"
        return True
