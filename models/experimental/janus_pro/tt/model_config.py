# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

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
        # LLaMA-style text decoder: RMSNorm without a unit offset.
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

    def _set_vision_params(self, config):
        vision_config = config.get("vision_config", config)

        # JanusVisionEmbeddings (patch_embedding, position_embedding)
        self.vision_chunk_size = vision_config.get("image_size", 384)
        self.vision_dim = vision_config.get("hidden_size", 1024)
        self.vision_patch_size = vision_config.get("patch_size", 16)
        self.vision_in_channels = vision_config.get("num_channels", 3)
        self.mm_tokens_per_image = vision_config.get("num_image_tokens", 576)

        # JanusVisionEncoder (ModuleList of encoder layers)
        self.vision_n_layers = vision_config.get("num_hidden_layers", 24)

        # JanusVisionEncoderLayer (layer_norm1/2) and JanusVisionModel (post_layernorm)
        self.vision_layer_norm_eps = vision_config.get("layer_norm_eps", 1e-6)

        # JanusVisionAttention (q/k/v_proj, q/k_norm, projection_layer)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_attention_bias = vision_config.get("attention_bias", True)
        self.vision_use_qk_norm = vision_config.get("use_qk_norm", False)
        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.vision_projection_dropout = vision_config.get("projection_dropout", 0.0)

        # JanusVisionMLP (fc1, activation_fn, fc2; activation_fn also used by JanusVisionAlignerMLP)
        mlp_ratio = vision_config.get("mlp_ratio", 4.0)
        self.vision_mlp_ratio = mlp_ratio
        self.vision_hidden_dim = int(self.vision_dim * mlp_ratio)
        act_layer = str(vision_config.get("hidden_act", "gelu")).lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)
        self.vision_hidden_dropout = vision_config.get("hidden_dropout_rate", 0.0)

        # JanusVisionAlignerMLP (fc1, hidden_layers; vision-to-text projection)
        self.vision_projection_dim = vision_config.get("projection_dim", 2048)
        self.vision_aligner_depth = vision_config.get("depth", 2)

        # Not in Janus HF config; placeholders for base ModelArgs.__repr__
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = vision_config.get("vision_num_cross_attention_layers", 0)

    def get_hf_model_cls(self):
        from transformers import JanusForConditionalGeneration

        return JanusForConditionalGeneration

    def load_state_dict(self):
        # Weight-seeding path: take weights from the base model at its native checkpoint dtype.
        # Do NOT route through reference_vision_transformer() below, which upcasts to float32 for
        # the golden reference only. Mirrors models/demos/multimodal/gemma3, which likewise keeps
        # the float32 upcast off the weight path.
        model = super().reference_vision_transformer(wrap=False)
        return convert_vision_hf_to_meta(model.state_dict(), self.head_dim)

    def reference_vision_transformer(self, wrap=True, load_checkpoint=False):
        # Golden-reference path: force float32 so the reference matches float32 test inputs (a stable
        # PCC baseline). Float an ISOLATED model instance, never self.cached_hf_model (which seeds the
        # TT weights via load_state_dict), so the upcast cannot leak into the weight path regardless of
        # call order. Mirrors models/demos/multimodal/gemma3.tt.model_config.reference_vision_transformer.
        model = self.get_hf_model_cls().from_pretrained(
            self.CKPT_DIR, torch_dtype="auto", local_files_only=os.getenv("CI") == "true"
        )
        model = model.float()
        if wrap:
            from models.tt_transformers.tt.model_config import HfModelWrapper

            return HfModelWrapper(model, self.head_dim, use_hf_rope=self.use_hf_rope)
        return model

    def reference_siglip_patch_embed(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.embeddings.patch_embedding

    def reference_vision_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.embeddings

    def reference_vision_layernorm(self, layer_name="layer_norm1"):
        model = self.reference_vision_model()
        if layer_name == "layer_norm1":
            return model.encoder.layers[0].layer_norm1
        elif layer_name == "layer_norm2":
            return model.encoder.layers[0].layer_norm2
        return model.post_layernorm

    def reference_vision_attention(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder.layers[0].self_attn

    def reference_vision_mlp(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder.layers[0].mlp

    def reference_vision_encoder_block(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder.layers[0]

    def reference_vision_encoder(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model.encoder

    def reference_vision_model(self):
        model = self.reference_vision_transformer(wrap=False)
        return model.model.vision_model

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if is_vision:
            prefix = "model.vision_model.encoder."
        else:
            prefix = ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        text_module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",
        }
        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "attn.",
            "TransformerBlock": "",
            "": "",
        }
        module_map = vision_module_map if is_vision else text_module_map

        return prefix + layer_prefix + module_map[module_name]
