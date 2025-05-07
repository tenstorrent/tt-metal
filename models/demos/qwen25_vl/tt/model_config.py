# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from loguru import logger

from models.demos.qwen25_vl.tt.common import nearest_multiple

from models.tt_transformers.tt.load_checkpoints import (
    load_hf_state_dict,
    convert_hf_to_meta,
    standardize_hf_keys,
)
from models.tt_transformers.tt.model_config import ModelArgs


class VisionModelArgs(ModelArgs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Core dimensions from HF config
        self.dim = self.hf_config.vision_config.hidden_size
        self.unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(  # pad to a tile multiple per device
            self.unpadded_hidden_dim, self.tile_size * self.num_devices
        )
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_heads
        self.n_heads = self.hf_config.vision_config.num_heads
        self.n_kv_heads = self.hf_config.vision_config.num_heads

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size

        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding head dim from {self.head_dim} to {self.padded_head_dim}")

        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

    # Visual model does not use distributed norm for now
    def is_distributed_norm(self, mode):
        return False

    def load_state_dict(self):
        assert False, "FIXME"
        if self.from_hf_url:
            # Special case Qwen2.5-VL models until they are fully integrated into a HF release
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration as AutoModelForCausalLM,
            )

            print("Loading Qwen2.5-VL model: ", AutoModelForCausalLM)
            model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR)
            state_dict = {k: v for k, v in model.state_dict().items() if k.startswith("visual.")}
        else:
            state_dict = load_hf_state_dict(self.CKPT_DIR)
        state_dict = standardize_hf_keys(state_dict)
        state_dict = convert_hf_to_meta(state_dict, self.head_dim)
        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict

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
