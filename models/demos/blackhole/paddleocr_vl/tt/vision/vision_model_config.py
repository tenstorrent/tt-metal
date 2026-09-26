# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Args object driving PaddleOCR-VL's vision tower on Blackhole.

Follows ``qwen36/tt/vision/vision_model_config.py`` closely (same tower
shape, same modules); differences are naming, not math -- see
``load_state_dict`` below and the ``out_hidden_size`` publish.
"""

from __future__ import annotations

import math

from loguru import logger

import ttnn
from models.demos.qwen3_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs


class VisionModelOptimizations:
    """Accuracy-first: OCR pays for MLP precision in dropped characters."""

    def __init__(self, model_name):
        self.bfp4_mlp = False


class VisionModelArgs(ModelArgs):
    # The base class validates the TEXT config's 2 KV heads against the mesh. The
    # vision tower overrides the head counts below with its own 16 MHA heads, so
    # only the base check needs relaxing.
    SUPPORTS_KV_REPLICATION = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vision_cfg = self.hf_config.vision_config

        # PatchMerger reads out_hidden_size off the vision config; PaddleOCR-VL
        # does not carry one because the projector's output width is defined by
        # the text side. Publish it so the shared module needs no special case.
        if not hasattr(vision_cfg, "out_hidden_size"):
            vision_cfg.out_hidden_size = self.hf_config.text_config.hidden_size

        self.n_vision_layers = vision_cfg.num_hidden_layers
        self.patch_size = vision_cfg.patch_size
        self.spatial_merge_size = vision_cfg.spatial_merge_size
        self.vision_layer_norm_eps = vision_cfg.layer_norm_eps

        self.dim = vision_cfg.hidden_size
        self.unpadded_hidden_dim = vision_cfg.intermediate_size
        self.hidden_dim = nearest_multiple(self.unpadded_hidden_dim, self.tile_size * self.num_devices)
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding vision hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")

        self.n_heads = vision_cfg.num_attention_heads
        self.n_kv_heads = vision_cfg.num_attention_heads  # MHA, not GQA
        self.head_dim = vision_cfg.hidden_size // vision_cfg.num_attention_heads

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size
        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding vision head dim from {self.head_dim} to {self.padded_head_dim}")

        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        self.optimizations = VisionModelOptimizations(self.model_name)

        # Rebuilt at the tower's depth (27), not the text stack's (18), or
        # VisionAttention's layer-18+ lookup raises KeyError.
        self.model_config["DECODERS_OPTIMIZATIONS"] = DecodersPrecision(self.n_vision_layers, self.model_name)

        num_rows = lambda seq_len: min(seq_len, 1024 if self.is_galaxy else 2048)
        k_dim = self.dim // self.cluster_shape[0] if self.is_galaxy else self.dim
        n_dim = self.dim // self.cluster_shape[1] if self.is_galaxy else self.dim
        self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
            m=num_rows(seq_len),
            k=k_dim,
            n=n_dim,
            grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
            in0_block_w=1 if self.is_galaxy else max(1, self.dim // 1024),
            fuse_batch=seq_len <= 1024,
        )

        tp = self.cluster_shape[1]
        assert self.n_heads % tp == 0, f"vision n_heads ({self.n_heads}) must be divisible by TP={tp}"
        assert self.qkv_size % tp == 0, f"vision qkv_size ({self.qkv_size}) must be divisible by TP={tp}"
        assert self.dim % tp == 0, f"vision dim ({self.dim}) must be divisible by TP={tp}"
        assert self.hidden_dim % tp == 0, f"vision hidden_dim ({self.hidden_dim}) must be divisible by TP={tp}"

        mlp_size = vision_cfg.hidden_size * (vision_cfg.spatial_merge_size**2)
        assert mlp_size % tp == 0, f"vision merger mlp_size ({mlp_size}) must be divisible by TP={tp}"
        assert (
            vision_cfg.out_hidden_size % tp == 0
        ), f"vision out_hidden_size ({vision_cfg.out_hidden_size}) must be divisible by TP={tp}"

    def load_state_dict(self):
        """Load the checkpoint, converting the text QKV with the *text* head dim.

        The base implementation permutes the text q/k projections into meta RoPE
        format using ``self.head_dim``, which this subclass has redefined to the
        vision tower's 72. Left alone it computes 2048/72 = 28 text heads and
        dies on a reshape. Restore the text value for the duration of the load;
        the vision projections are untouched by that path and get their own
        permute in ``weight_mapping._to_meta_rope_format``.
        """
        vision_head_dim = self.head_dim
        self.head_dim = self.hf_config.text_config.head_dim
        try:
            return super().load_state_dict()
        finally:
            self.head_dim = vision_head_dim

    def prepare_residual_tensor_prefill(self, x_bsh):
        """Shard the patch sequence along hidden, which is the blocks' I/O contract."""
        x_1BSH = x_bsh.unsqueeze(0)
        return ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.cluster_shape),
        )

    def is_distributed_norm(self, mode):
        return False

    def get_state_dict_prefix(self, module_name, layer_num=None, deepstack_merger_num=None):
        """Keys follow the qwen36 tower's layout; see ``tt/weight_mapping.py``."""
        layer_prefix = f"visual.blocks.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "VisionAttention": "attention",
            "VisionBlock": "",
            "VisionTransformer": "visual",
            "PatchMerger": "visual.merger",
            "norm1": "norm1",
            "norm2": "norm2",
            "": "",
        }
        return layer_prefix + module_map[module_name]

    # ---- references for PCC comparison -------------------------------------

    def reference_vision_model(self):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(self.CKPT_DIR, dtype="auto")
        return model.model.visual.vision_model

    def reference_vision_block(self, layer_num=0):
        return self.reference_vision_model().encoder.layers[layer_num]

    def reference_mlp(self):
        return self.reference_vision_block().mlp

    def reference_attention(self):
        return self.reference_vision_block().attn

    def reference_patch_merger(self):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(self.CKPT_DIR, dtype="auto")
        return model.model.projector
