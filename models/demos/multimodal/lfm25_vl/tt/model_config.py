# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import os

import torch
from loguru import logger

import ttnn
from models.demos.multimodal.lfm25_vl.tt.load_checkpoints import convert_lfm_hf_to_meta
from models.tt_transformers.tt.load_checkpoints import convert_meta_to_hf
from models.tt_transformers.tt.model_config import HfAttentionWrapper, HfDecoderWrapper, HfModelWrapper
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs


class ModelArgs(TTModelArgs):
    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 8,
        optimizations=None,
        cache_hf=False,
    ):
        # Resolve HF_MODEL to a local snapshot path before super().__init__() so every HF call
        # (AutoConfig, tokenizer, weights) skips the refs/main lookup (absent on some CI machines).
        # Mirrors models.demos.multimodal.gemma3.tt.model_config.ModelArgs.
        hf_model = os.environ.get("HF_MODEL", "")
        if hf_model and not os.path.isabs(hf_model):
            snapshot = ModelArgs._resolve_hf_snapshot(hf_model)
            if snapshot:
                logger.info(f"[LFM2.5-VL] Resolved HF model '{hf_model}' to snapshot: {snapshot}")
                os.environ["HF_MODEL"] = str(snapshot)

        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            cache_hf=cache_hf,
        )

        if dummy_weights and self.tokenizer is None:
            self.tokenizer = self.create_tokenizer()

        text_config = self.hf_config.get("text_config", self.hf_config)

        # LFM2 uses "image_token_id" (not the generic "image_token_index" tt_transformers looks for).
        self.image_token_index = self.hf_config.get("image_token_id") or self.hf_config.get("image_token_index", 396)
        self.image_token_id = self.image_token_index

        # Hybrid decoder: which layers are ShortConv vs full self-attention.
        self.layer_types = text_config.get(
            "layer_types",
            ["full_attention"] * self.n_layers,
        )
        self.conv_L_cache = text_config.get("conv_L_cache", 3)
        self.conv_bias = text_config.get("conv_bias", False)

        # LLaMA-style SwiGLU dim auto-adjustment ("block_auto_adjust_ff_dim"): recompute hidden_dim
        # here since the base class just takes text_config["intermediate_size"] verbatim.
        if text_config.get("block_auto_adjust_ff_dim", False):
            ff_dim = text_config.get("intermediate_size", self.hidden_dim)
            ff_dim = int(2 * ff_dim / 3)
            ffn_dim_multiplier = text_config.get("block_ffn_dim_multiplier")
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            multiple_of = text_config.get("block_multiple_of", 256)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)
            self.hidden_dim = ff_dim
            self.unpadded_hidden_dim = ff_dim

        # LFM2 attention does not use tt_transformers' fused-qkv rope/paged-cache-update path.
        self.use_qk_fused = False

        # Vision tower / projector params, computed from vision_config (base class's default
        # _set_vision_params overwrite happens inside _set_params_from_dict, so re-derive here).
        self._set_vision_params(self.hf_config)

        self.downsample_factor = self.hf_config.get("downsample_factor", 2)
        self.projector_hidden_size = self.hf_config.get("projector_hidden_size", 2048)
        self.projector_use_layernorm = self.hf_config.get("projector_use_layernorm", False)
        self.projector_bias = self.hf_config.get("projector_bias", True)

    def _set_model_specific_params(self):
        self.rms_norm_add_unit_offset = False

    def _set_hf_params(self, checkpoint_dir):
        """Override the base class: LFM2.5-VL nests both ``text_config`` and ``vision_config``
        (unlike plain-text HF configs), and the base implementation stores ``self.hf_config`` as a
        ``PretrainedConfig`` object rather than a plain dict, which breaks the ``.get(...)`` calls
        used throughout this file. Mirrors ``models.demos.multimodal.gemma3.tt.model_config``.
        """

        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            text_config.update({k: v for k, v in base_config.items() if k not in ("text_config", "vision_config")})
            return text_config

        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            vision_config.update({k: v for k, v in base_config.items() if k not in ("text_config", "vision_config")})
            return vision_config

        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(
            checkpoint_dir, trust_remote_code=self.trust_remote_code_hf
        ).to_dict()

        if "text_config" in self.hf_config or "vision_config" in self.hf_config:
            self._set_params_from_dict(merge_text_config(self.hf_config))
            if "vision_config" in self.hf_config:
                self._set_vision_params({"vision_config": merge_vision_config(self.hf_config)})
            self.is_multimodal = True
        else:
            self._set_params_from_dict(self.hf_config)

    def _set_vision_params(self, config):
        vision_config = config.get("vision_config", config)

        self.vision_dim = vision_config.get("hidden_size", 1152)
        intermediate_size = vision_config.get("intermediate_size", self.vision_dim * 4)
        self.vision_mlp_ratio = intermediate_size // self.vision_dim
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_n_layers = vision_config.get("num_hidden_layers", 27)
        self.vision_patch_size = vision_config.get("patch_size", 16)
        self.vision_in_channels = vision_config.get("num_channels", 3)
        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.vision_n_global_layers = vision_config.get("n_global_layers", 8)
        self.vision_num_patches = vision_config.get("num_patches", 256)
        self.norm_eps_vision = vision_config.get("layer_norm_eps", 1e-6)

        # SigLIP2-NaFlex has no fixed "image size"; derive a square reference chunk edge from the
        # configured (fixed, non-NaFlex) patch grid so the rest of the vision stack (which expects
        # an "image_size"/"vision_chunk_size") can size its position-embedding table.
        self.vision_chunk_size = self.vision_patch_size * int(round(self.vision_num_patches**0.5))
        self.vision_max_num_chunks = config.get("max_tiles", 10)

        act_layer = str(vision_config.get("hidden_act", "gelu_pytorch_tanh")).lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "gelu_pytorch_tanh": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)

    @staticmethod
    def _snapshot_has_weights(snapshot_path):
        """A HF cache snapshot may exist with only config/tokenizer files (from earlier AutoConfig
        calls) but no model weights; resolving HF_MODEL to such a path breaks from_pretrained
        (it errors instead of downloading). Only treat a snapshot as usable if weights are present.
        """
        try:
            names = os.listdir(snapshot_path)
        except OSError:
            return False
        return any(
            n.endswith(".safetensors") or n.endswith(".safetensors.index.json") or n == "pytorch_model.bin"
            for n in names
        )

    @staticmethod
    def _resolve_hf_snapshot(hf_model_name):
        hf_cache = os.path.normpath(
            os.environ.get("HF_HUB_CACHE")
            or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        )
        model_slug = "models--" + hf_model_name.replace("/", "--")
        snapshots_dir = os.path.normpath(os.path.join(hf_cache, model_slug, "snapshots"))
        if not snapshots_dir.startswith(hf_cache + os.sep):
            return None
        if not os.path.isdir(snapshots_dir):
            return None
        snaps = [
            path
            for path in (os.path.join(snapshots_dir, s) for s in os.listdir(snapshots_dir))
            if os.path.isdir(path) and ModelArgs._snapshot_has_weights(path)
        ]
        return max(snaps, key=os.path.getmtime) if snaps else None

    def get_max_prefill_chunk_size(self):
        model_overrides = {
            "LFM2.5-VL-1.6B": {"N150": 128, "N300": 128, "T3K": 128, "P150": 128, "P150x4": 128},
        }
        model_name = self.base_model_name
        device_name = self.device_name
        if model_name in model_overrides and device_name in model_overrides[model_name]:
            return model_overrides[model_name][device_name] * 1024
        return super().get_max_prefill_chunk_size()

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if is_vision:
            text_prefix = "model.vision_tower.vision_model.encoder."
        else:
            text_prefix = ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "ShortConv": "conv",
            "TransformerBlock": "",
            "": "",
        }
        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "self_attn.",
            "TransformerBlock": "",
            "": "",
        }
        module_map = vision_module_map if is_vision else module_map
        return text_prefix + layer_prefix + module_map[module_name]

    def get_hf_model_cls(self):
        try:
            from transformers import Lfm2VlForConditionalGeneration

            return Lfm2VlForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForImageTextToText

            logger.warning(
                "transformers.Lfm2VlForConditionalGeneration not found (need transformers>=4.53 / v5.1 "
                "LFM2-VL support); falling back to AutoModelForImageTextToText."
            )
            return AutoModelForImageTextToText

    def _lfm_dummy_hf_model(self):
        from transformers import AutoConfig

        logger.info("LFM2.5-VL ModelArgs: building HF dummy model from config (dummy_weights=True)")
        config = AutoConfig.from_pretrained(self.CKPT_DIR, trust_remote_code=self.trust_remote_code_hf)
        if hasattr(config, "text_config") and config.text_config is not None:
            config.text_config.num_hidden_layers = self.n_layers
        model_cls = self.get_hf_model_cls()
        # transformers 5.x removed `from_config` from concrete PreTrainedModel classes
        # (only Auto* classes keep it); instantiate directly from the config in that case.
        if hasattr(model_cls, "from_config"):
            try:
                model = model_cls.from_config(
                    config, torch_dtype=torch.bfloat16, trust_remote_code=self.trust_remote_code_hf
                )
            except TypeError:
                model = model_cls.from_config(config)
        else:
            model = model_cls(config).to(torch.bfloat16)
        gc.collect()
        return model

    def load_state_dict(self):
        if self.dummy_weights:
            logger.info("LFM2.5-VL ModelArgs: using dummy_weights path; NOT loading checkpoints from HF_MODEL")
            model = self._lfm_dummy_hf_model()
            state_dict = model.state_dict()
            del model
            gc.collect()
        else:
            model_cls = self.get_hf_model_cls()
            model = model_cls.from_pretrained(
                self.CKPT_DIR,
                torch_dtype="auto",
                trust_remote_code=self.trust_remote_code_hf,
                local_files_only=os.getenv("CI") == "true",
            )
            if self.cache_hf_flag:
                self.cached_hf_model = model
            state_dict = model.state_dict()

        state_dict = convert_lfm_hf_to_meta(state_dict, self.head_dim)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any(r in k for r in remv):
                state_dict.pop(k)

        return state_dict

    # ------------------------------------------------------------------
    # HF submodule accessors, used by unit tests (PCC comparisons)
    # ------------------------------------------------------------------
    def reference_transformer(self, wrap=True, load_checkpoint=False):
        if self.dummy_weights and not load_checkpoint:
            model = self._lfm_dummy_hf_model()
        else:
            model = self.get_hf_model_cls().from_pretrained(self.CKPT_DIR, trust_remote_code=self.trust_remote_code_hf)
        model = model.float()
        if wrap:
            return HfModelWrapper(model, self.head_dim)
        return model

    @staticmethod
    def _lfm_language_model(model):
        # Lfm2VlForConditionalGeneration.model.language_model (Lfm2Model)
        return model.model.language_model

    @staticmethod
    def _lfm_vision_tower(model):
        vt = model.model.vision_tower
        return vt.vision_model if hasattr(vt, "vision_model") else vt

    @staticmethod
    def _lfm_multi_modal_projector(model):
        return model.model.multi_modal_projector

    def reference_mlp(self):
        # HF Lfm2MLP already uses llama-style parameter names (w1/w2/w3), which match the
        # meta-converted state dict directly -- no meta->HF key conversion needed.
        model = self.reference_transformer(wrap=False)
        return self._lfm_language_model(model).layers[0].feed_forward

    def _first_layer_of_type(self, layer_type):
        for i, lt in enumerate(self.layer_types):
            if lt == layer_type:
                return i
        raise ValueError(f"No layer of type {layer_type!r} in layer_types={self.layer_types}")

    def reference_short_conv(self, layer_idx=None):
        model = self.reference_transformer(wrap=False)
        layer_idx = self._first_layer_of_type("conv") if layer_idx is None else layer_idx
        return self._lfm_language_model(model).layers[layer_idx].conv

    def reference_attention(self):
        model = self.reference_transformer(wrap=False)
        layer_idx = self._first_layer_of_type("full_attention")
        layer = self._lfm_language_model(model).layers[layer_idx].self_attn
        rotary_emb = self._lfm_language_model(model).rotary_emb
        return HfAttentionWrapper(layer, self.head_dim, rotary_emb)

    def reference_decoder(self, i=None):
        model = self.reference_transformer(wrap=False)
        i = self._first_layer_of_type("full_attention") if i is None else i
        layer = self._lfm_language_model(model).layers[i]
        rotary_emb = self._lfm_language_model(model).rotary_emb
        return HfDecoderWrapper(layer, self.head_dim, rotary_emb)

    def reference_rms_norm(self):
        model = self.reference_transformer(wrap=False)
        layer = self._lfm_language_model(model).embedding_norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_model(self):
        model = self.reference_transformer(wrap=False)
        return self._lfm_vision_tower(model)

    def reference_vision_mlp(self):
        return self.reference_vision_model().encoder.layers[0].mlp

    def reference_vision_attention(self):
        return self.reference_vision_model().encoder.layers[0].self_attn

    def reference_projector(self):
        model = self.reference_transformer(wrap=False)
        return self._lfm_multi_modal_projector(model)
