# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import calculate_prefill_warmup_seq_lens, cap_seq_lens_to_max_prefill_chunk_size
from models.tt_transformers.tt.load_checkpoints import (
    map_hf_to_meta_keys,
    reverse_permute,
    reverse_permute_1d,
    split_hf_keys,
    standardize_hf_keys_multimodal,
)
from models.tt_transformers.tt.model_config import HfAttentionWrapper, HfModelWrapper
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs


class ModelArgs(TTModelArgs):
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
        )

        # Override precision: generic accuracy config already sets WQKV/WO/KV_CACHE=BF16
        # and attention ops to HIFI4. Additionally force BF16 activations and MLP.
        from models.tt_transformers.tt.model_config import MathFidelitySetting, OpGroup, PrecisionSetting, TensorGroup

        opt = self.optimizations.decoder_optimizations[0]
        opt._opt_settings["TensorPrecision"][TensorGroup.ACTIVATION] = PrecisionSetting.BF16
        opt._opt_settings["TensorPrecision"][TensorGroup.FF1_FF3] = PrecisionSetting.BF16
        opt._opt_settings["TensorPrecision"][TensorGroup.FF2] = PrecisionSetting.BF16
        # Use HIFI4_FP32 (packer_l1_acc=False) for all ops to avoid BF16 cross-tile
        # accumulation precision loss in matmuls over 5376-dim inputs (168 tiles).
        for op in OpGroup:
            if op != OpGroup.ACCURACY:
                opt._opt_settings["OpFidelity"][op] = MathFidelitySetting.HIFI4_FP32

        self.use_qk_fused = False
        self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.padded_vocab_size = 262400
        self.lm_head_dtype = ttnn.bfloat16

        # Gemma 4-specific architecture params (set after HF config loaded in _set_model_specific_params)
        self.global_head_dim = 512
        self.num_global_kv_heads = 4
        self.attention_k_eq_v = True
        self.final_logit_softcapping = 30.0

    def _set_model_specific_params(self):
        self.rms_norm_add_unit_offset = False  # Gemma 4 uses weight*x, no (1+weight)*x
        self.embed_scale = self.dim**0.5
        # Gemma 4 uses q_norm/k_norm on Q/K, so attention scaling = 1.0 (not 1/sqrt(head_dim)).
        # HF Gemma4TextAttention hardcodes self.scaling = 1.0.
        self.query_pre_attn_scalar = 1.0
        self.use_hf_rope = False  # Use Meta-style RoPE (with weight permutation)

        # Parse Gemma 4-specific config from HF
        if hasattr(self, "hf_config") and self.hf_config is not None:
            text_config = self.hf_config.get("text_config", self.hf_config)
            self.global_head_dim = text_config.get("global_head_dim", 512)
            self.num_global_kv_heads = text_config.get("num_global_key_value_heads", 4)
            self.attention_k_eq_v = text_config.get("attention_k_eq_v", True)
            self.final_logit_softcapping = text_config.get("final_logit_softcapping", 30.0)

            # Parse dual RoPE parameters
            rope_params = text_config.get("rope_parameters", {})
            sliding_rope = rope_params.get("sliding_attention", {})
            full_rope = rope_params.get("full_attention", {})

            # Sliding attention RoPE (local) - standard RoPE
            self.rope_theta_local = sliding_rope.get("rope_theta", 10000.0)
            self.rope_theta = sliding_rope.get("rope_theta", 10000.0)

            # Full attention RoPE (global) - proportional with partial rotary
            self.rope_theta_global = full_rope.get("rope_theta", 1000000.0)
            self.rope_type_global = full_rope.get("rope_type", "proportional")
            self.partial_rotary_factor = full_rope.get("partial_rotary_factor", 0.25)

            # Sliding window from text config
            self.sliding_window = text_config.get("sliding_window", 1024)

    def is_layer_sliding(self, layer_num):
        """Check if a layer uses sliding window attention."""
        if self.layer_types is not None and layer_num < len(self.layer_types):
            return self.layer_types[layer_num] == "sliding_attention"
        return True  # Default to sliding

    def get_layer_head_dim(self, layer_num):
        """Get head dimension for a specific layer."""
        if self.is_layer_sliding(layer_num):
            return self.head_dim  # 256
        return self.global_head_dim  # 512

    def get_layer_n_kv_heads(self, layer_num):
        """Get number of KV heads for a specific layer."""
        if self.is_layer_sliding(layer_num):
            return self.n_kv_heads  # 16
        return self.num_global_kv_heads  # 4

    def get_layer_has_v_proj(self, layer_num):
        """Check if a layer has separate V projection (not K=V sharing)."""
        if self.is_layer_sliding(layer_num):
            return True  # Sliding layers always have v_proj
        return not self.attention_k_eq_v  # Full attention layers use K=V when enabled

    def get_layer_qkv_size(self, layer_num):
        """Get total QKV output size for a layer."""
        hd = self.get_layer_head_dim(layer_num)
        nkv = self.get_layer_n_kv_heads(layer_num)
        # V uses same dimension as K (even for K=V sharing, the concatenated weight includes V slot)
        return hd * (2 * nkv + self.n_heads)

    def get_warmup_prefill_supported_seq_lens(self):
        DEFAULT_VALUE = self.capped_warmup_seq_len
        model_specific_ceil_warmup_lengths = {
            "gemma-4-31b": 2048,
            "gemma-4-31B": 2048,
        }
        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(self.base_model_name, DEFAULT_VALUE)
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )
        to_warmup_seq_lens = self.filter_warmup_seq_lens(to_warmup_seq_lens)
        return to_warmup_seq_lens

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        return to_warmup_seq_lens

    def get_residual_mem_config(self, mode, prefetcher=None):
        """Override for decode: use L1 interleaved (Gemma4's 1344/dev doesn't
        shard cleanly with the base config's grid)."""
        from models.tt_transformers.tt.common import Mode

        if mode == Mode.DECODE:
            return ttnn.L1_MEMORY_CONFIG
        return super().get_residual_mem_config(mode, prefetcher)

    def get_attn_sdpa_output_mem_config(self, mode, batch_size_per_device_group=1, prefetcher=None, head_dim=None):
        """Use base config — it handles head_dim via parameter."""
        return super().get_attn_sdpa_output_mem_config(mode, batch_size_per_device_group, prefetcher, head_dim=head_dim)

    def get_attn_create_head_output_mem_config(self, mode, prefetcher=None, head_dim=None):
        """For non-standard head_dim: auto shard. For batch>16 with standard head_dim:
        use rectangular grid (paged_update_cache's fill_pad_writer overflows with non-rectangular grids)."""
        from models.tt_transformers.tt.common import Mode

        hd = head_dim if head_dim is not None else self.head_dim
        if mode == Mode.DECODE and prefetcher is None:
            if hd != self.head_dim:
                return ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            if self.max_batch_size > 16:
                # Use rectangular 4x8=32 core grid (rows 0-3 avoid all harvested cores)
                return ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, hd),
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
        return super().get_attn_create_head_output_mem_config(mode, prefetcher, head_dim=head_dim)

    def get_attn_qkv_program_config(self, mode, seq_len_or_batch, prefetcher):
        """Override for Gemma 4's non-standard dimensions."""
        from models.tt_transformers.tt.common import Mode

        if mode == Mode.PREFILL:
            # For seq_len > 128, minimal_matmul uses 'config' parameter.
            # Use the base config for this path - it generates proper DRAM-sharded configs.
            # For seq_len <= 128, ttnn.linear path has issues with the grid.
            seq_len = seq_len_or_batch if isinstance(seq_len_or_batch, int) else 256
            if seq_len > 128:
                return super().get_attn_qkv_program_config(mode, seq_len, prefetcher)
            return None
        return super().get_attn_qkv_program_config(mode, seq_len_or_batch, prefetcher)

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

    def _set_hf_params(self, checkpoint_dir):
        from transformers import AutoConfig

        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for Gemma 4 models.")
        else:
            self.hf_config = AutoConfig.from_pretrained(self.CKPT_DIR).to_dict()

        if "text_config" in self.hf_config or "vision_config" in self.hf_config:
            self._set_params_from_dict(self.hf_config)
        else:
            self._set_params_from_dict(self.hf_config)

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if is_vision:
            text_prefix = "model.vision_tower.vision_model.encoder."
        else:
            text_prefix = ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "Gemma4Attention": "attention",
            "TransformerBlock": "",
            "Gemma4TransformerBlock": "",
            "": "",
        }

        return text_prefix + layer_prefix + module_map.get(module_name, "")

    def load_state_dict(self):
        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for Gemma 4 models.")

        from transformers import Gemma4ForConditionalGeneration

        logger.info(f"Loading Gemma 4 model from {self.CKPT_DIR}")
        model = Gemma4ForConditionalGeneration.from_pretrained(
            self.CKPT_DIR,
            torch_dtype="auto",
        )
        if self.cache_hf_flag:
            self.cached_hf_model = model
        state_dict = model.state_dict()

        # Standardize multimodal keys: strip model.language_model. prefix etc.
        state_dict = standardize_hf_keys_multimodal(state_dict)

        # Convert to meta format WITH per-layer QKV permutation for Meta-style RoPE
        state_dict = split_hf_keys(state_dict)

        # Apply per-layer reverse_permute to Q/K weights (different head_dim per layer type)
        converted = {}
        for key, tensor in state_dict.items():
            if "vision_tower" in key or "visual" in key:
                converted[key] = tensor
                continue

            if ("q_proj.weight" in key or "k_proj.weight" in key) and "layers." in key:
                try:
                    parts = key.split(".")
                    layer_idx_pos = parts.index("layers") + 1
                    layer_num = int(parts[layer_idx_pos])
                    if self.is_layer_sliding(layer_num):
                        # Sliding layers: reverse_permute for Meta-style RoPE
                        layer_hd = self.head_dim  # 256
                        n_heads = tensor.shape[0] // layer_hd
                        converted[key] = reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
                    else:
                        # Full attention layers: NO permutation (use HF-style partial RoPE)
                        converted[key] = tensor
                except (ValueError, IndexError):
                    converted[key] = tensor
            elif ("q_norm.weight" in key or "k_norm.weight" in key) and "layers." in key:
                try:
                    parts = key.split(".")
                    layer_idx_pos = parts.index("layers") + 1
                    layer_num = int(parts[layer_idx_pos])
                    if self.is_layer_sliding(layer_num):
                        converted[key] = reverse_permute_1d(tensor)
                    else:
                        converted[key] = tensor  # No permute for full attention norms
                except (ValueError, IndexError):
                    converted[key] = tensor
            else:
                converted[key] = tensor
        state_dict = converted

        state_dict = map_hf_to_meta_keys(state_dict)

        # Handle Gemma 4-specific keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Preserve layer_scalar as-is
            if "layer_scalar" in k:
                new_state_dict[k] = v
                continue

            # For full attention layers without v_proj: duplicate K weights as V
            # Only handle text layers (starts with "layers.{N}")
            if "attention.wk.weight" in k and k.startswith("layers."):
                try:
                    layer_num = int(k.split(".")[1])
                    if not self.get_layer_has_v_proj(layer_num):
                        wv_key = k.replace("attention.wk.weight", "attention.wv.weight")
                        if wv_key not in state_dict:
                            new_state_dict[wv_key] = v.clone()
                except (ValueError, IndexError):
                    logger.debug(f"Could not parse layer index from key: {k}")

            new_state_dict[k] = v

        state_dict = new_state_dict

        # Remove text layers beyond n_layers
        keys_to_remove = []
        for k in state_dict.keys():
            if k.startswith("layers."):
                try:
                    layer_idx = int(k.split(".")[1])
                    if layer_idx >= self.n_layers:
                        keys_to_remove.append(k)
                except (ValueError, IndexError):
                    logger.debug(f"Could not parse layer index from key: {k}")
        for k in keys_to_remove:
            state_dict.pop(k)

        return state_dict

    def create_tokenizer(self):
        from transformers import AutoTokenizer

        logger.info(f"Tokenizer path: {self.TOKENIZER_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH, local_files_only=os.getenv("CI") == "true")
        if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
            tokenizer.stop_tokens = [tokenizer.eos_token_id]
        return tokenizer

    def get_hf_model_cls(self):
        from transformers import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration

    def reference_transformer(self, wrap=True, load_checkpoint=False):
        if self.dummy_weights and not load_checkpoint:
            raise NotImplementedError("Dummy weights not supported for Gemma 4.")

        from transformers import Gemma4ForConditionalGeneration

        model = Gemma4ForConditionalGeneration.from_pretrained(self.CKPT_DIR)
        if wrap:
            wrapper = HfModelWrapper(model, self.head_dim)
            return wrapper
        return model

    def reference_decoder(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.language_model.layers[i]
        rotary_emb = model.model.language_model.rotary_emb
        wrapper = HfGemma4DecoderWrapper(
            layer, self.head_dim, rotary_emb, self.layer_types[i] if self.layer_types else "sliding_attention"
        )
        return wrapper

    def reference_attention(self, layer_idx=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.language_model.layers[layer_idx].self_attn
        rotary_emb = model.model.language_model.rotary_emb
        wrapper = HfAttentionWrapper(layer, self.get_layer_head_dim(layer_idx), rotary_emb)
        return wrapper

    def reference_mlp(self, layer_idx=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.language_model.layers[layer_idx].mlp
        return layer


class HfGemma4DecoderWrapper:
    def __init__(self, decoder, head_dim, rotary_emb, layer_type):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.layer_type = layer_type
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        position_embeddings = self.rotary_emb(x, position_ids, self.layer_type)

        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

        result = self.decoder.forward(
            x,
            position_embeddings=position_embeddings,
            attention_mask={self.layer_type: mask},
            past_key_values=self.past_key_values,
            use_cache=True,
            position_ids=position_ids,
        )
        return result[0]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
