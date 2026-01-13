# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

from loguru import logger

import ttnn
from models.demos.qwen25_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.model_config import ModelArgs


class ModelOptimizations:
    def __init__(self, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        self.bfp4_mlp = "Qwen2.5-VL-72B" in model_name


class VisionModelArgs(ModelArgs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Core dimensions from HF config
        self.vision_dim = self.hf_config.vision_config.hidden_size
        self.vision_unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        self.vision_hidden_dim = nearest_multiple(  # pad to a tile multiple per device
            self.vision_unpadded_hidden_dim, self.tile_size * self.num_devices
        )
        if self.vision_hidden_dim != self.vision_unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.vision_unpadded_hidden_dim} to {self.vision_hidden_dim}")
        self.vision_head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_heads
        self.vision_n_heads = self.hf_config.vision_config.num_heads
        self.vision_n_kv_heads = self.hf_config.vision_config.num_heads

        self.vision_padded_head_dim = math.ceil(self.vision_head_dim / self.tile_size) * self.tile_size

        if self.vision_padded_head_dim != self.vision_head_dim:
            logger.info(f"padding head dim from {self.vision_head_dim} to {self.vision_padded_head_dim}")

        self.vision_qkv_size = self.vision_padded_head_dim * (2 * self.vision_n_kv_heads + self.vision_n_heads)
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        self.optimizations = ModelOptimizations(
            self.model_name
        )  # todo)) implement finer grained control similar to tt_transformers'

        num_rows = lambda seq_len: min(seq_len, 1024 if self.is_galaxy else 2048)
        k_dim = self.vision_dim // self.cluster_shape[0] if self.is_galaxy else self.vision_dim
        n_dim = self.vision_dim // self.cluster_shape[1] if self.is_galaxy else self.vision_dim
        self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
            m=num_rows(seq_len),
            k=k_dim,
            n=n_dim,
            grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
            in0_block_w=1 if self.is_galaxy else self.vision_dim // 1024,
            fuse_batch=seq_len <= 1024,
        )

        assert self.vision_n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

    def prepare_residual_tensor_prefill(self, x_bsh, force_replicated=False):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim
        """

        x_1BSH = x_bsh.unsqueeze(0)

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # todo)) refactor this code to make the intent clear, which is data parallelism
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        )
        return xs_1BSH

    # Visual model does not use distributed norm for now
    def is_distributed_norm(self, mode):
        return False

    def get_state_dict_prefix(self, module_name, layer_num=None):
        layer_prefix = f"visual.blocks.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "VisionAttention": "attention",
            "VisionBlock": "",
            "VisionTransformer": "visual",
            "PatchMerger": "visual.merger",
            "": "",  # If no module is given, just get layer prefix
        }
        return layer_prefix + module_map[module_name]

    def reference_vision_model(self, depth=None):
        # Workaround until Qwen2.5-VL is fully integrated into a HF release
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration as AutoModelForCausalLM,
        )

        print("Loading Qwen2.5-VL model: ", AutoModelForCausalLM)
        config = AutoModelForCausalLM.config_class.from_pretrained(self.CKPT_DIR)
        config.vision_config.depth = depth if depth is not None else config.vision_config.depth
        model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR, config=config)
        return model.visual

    def reference_vision_block(self, layer_num=0):
        return self.reference_vision_model().blocks[layer_num]

    def reference_mlp(self):
        return self.reference_vision_block().mlp

    def reference_attention(self):
        return self.reference_vision_block().attn

    def reference_rms_norm(self):
        return self.reference_vision_block().norm2

    def reference_patch_merger(self):
        return self.reference_vision_model().merger

    def reference_patch_embed(self):
        return self.reference_vision_model().patch_embed
