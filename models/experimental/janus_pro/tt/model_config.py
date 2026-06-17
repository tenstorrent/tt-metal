# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Model configuration for Janus Pro (deepseek-community/Janus-Pro-7B).

Follows the Gemma3 ModelArgs conventions (same TTModelArgs base, same method
layout) but loads Janus Pro weights and uses the Janus vision-tower dimensions
and state-dict prefixes (``model.vision_model.*``).
"""

import gc
import os

from loguru import logger

import ttnn
from models.experimental.janus_pro.tt.load_checkpoints import convert_vision_hf_to_meta
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs


class ModelArgs(TTModelArgs):
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
        # Resolve HF_MODEL to a local snapshot path before super().__init__() so that
        # all HF calls (AutoConfig, tokenizer, weights) skip the refs/main lookup,
        # which is absent on some CI machines.
        hf_model = os.environ.get("HF_MODEL", "")
        if hf_model and not os.path.isabs(hf_model):
            snapshot = ModelArgs._resolve_hf_snapshot(hf_model)
            if snapshot:
                logger.info(f"[JanusPro] Resolved HF model '{hf_model}' to snapshot: {snapshot}")
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

    @staticmethod
    def _resolve_hf_snapshot(hf_model_name):
        hf_cache = os.path.normpath(
            os.environ.get("HF_HUB_CACHE")
            or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        )
        model_slug = "models--" + hf_model_name.replace("/", "--")
        snapshots_dir = os.path.normpath(os.path.join(hf_cache, model_slug, "snapshots"))
        # Prevent path traversal: ensure the resolved path stays within hf_cache.
        if not snapshots_dir.startswith(hf_cache + os.sep):
            return None
        if not os.path.isdir(snapshots_dir):
            return None
        snaps = [
            os.path.join(snapshots_dir, s)
            for s in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, s))
        ]
        return max(snaps, key=os.path.getmtime) if snaps else None

    def _set_model_specific_params(self):
        # Janus Pro uses a LLaMA-style text decoder: no Gemma-style RMSNorm unit offset.
        self.rms_norm_add_unit_offset = False

    def create_tokenizer(self):
        # Janus ships only a fast tokenizer (tokenizer.json) and no sentencepiece
        # tokenizer.model, so force use_fast to avoid the slow LlamaTokenizer path
        # that requires a vocab_file.
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.CKPT_DIR,
            use_fast=True,
            local_files_only=os.getenv("CI") == "true",
            trust_remote_code=self.trust_remote_code_hf,
        )

    def _set_vision_params(self, vision_config):
        self.vision_chunk_size = vision_config.get("image_size", 384)
        self.vision_dim = vision_config.get("hidden_size", 1024)
        self.vision_patch_size = vision_config.get("patch_size", 16)
        self.vision_in_channels = vision_config.get("num_channels", 3)

        self.vision_n_layers = vision_config.get("num_hidden_layers", 24)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads

        mlp_ratio = vision_config.get("mlp_ratio", 4.0)
        self.vision_mlp_ratio = mlp_ratio
        self.vision_hidden_dim = int(self.vision_dim * mlp_ratio)

        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.mm_tokens_per_image = vision_config.get("num_image_tokens", 576)

        # Kept for parity with the Gemma3 vision tower wiring (no cross-attention in Janus).
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = vision_config.get("vision_num_cross_attention_layers", 0)
        self.vision_n_global_layers = vision_config.get("n_global_layers", 0)

        act_layer = str(vision_config.get("hidden_act", "gelu")).lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)

    def _set_hf_params(self, checkpoint_dir):
        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            # Merge non-nested keys into vision_config
            vision_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return vision_config

        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.CKPT_DIR).to_dict()

        if "text_config" in self.hf_config or "vision_config" in self.hf_config:
            self._set_params_from_dict(self.hf_config)
            if "vision_config" in self.hf_config:
                merged_vision_config = merge_vision_config(self.hf_config)
                self._set_vision_params(merged_vision_config)
        else:
            self._set_params_from_dict(self.hf_config)

    def load_state_dict(self):
        from transformers import JanusForConditionalGeneration

        model = JanusForConditionalGeneration.from_pretrained(
            self.CKPT_DIR,
            torch_dtype="auto",
        )
        if self.cache_hf_flag:
            self.cached_hf_model = model
        state_dict = model.state_dict()
        state_dict = convert_vision_hf_to_meta(state_dict, self.head_dim)

        if not self.cache_hf_flag:
            del model
            gc.collect()

        return state_dict

    def reference_vision_transformer(self, wrap=False, load_checkpoint=False):
        from transformers import JanusForConditionalGeneration

        return JanusForConditionalGeneration.from_pretrained(self.CKPT_DIR)

    def reference_siglip_patch_embed(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.embeddings.patch_embedding

    def reference_vision_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.embeddings
