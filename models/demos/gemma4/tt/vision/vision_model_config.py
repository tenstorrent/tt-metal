# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.qwen3_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.model_config import ModelArgs


class ModelOptimizations:
    def __init__(self, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        self.bfp4_mlp = False
        # self.bfp4_mlp = "Qwen3-VL-32B" in model_name


class VisionModelArgs(ModelArgs):
    def __init__(self, *args, mesh_config=None, ccl_manager=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Tensor parallelism: the vision tower is sharded across every device of the
        # mesh (attention by heads, MLP by intermediate dim), and each block ends in an
        # all-reduce so activations stay replicated at the block boundaries. Batch is
        # always 1 — callers loop over users.
        self.tp = self.cluster_shape[1] if self.cluster_shape else 1
        if self.tp > 1:
            self.mesh_config = mesh_config or MeshConfig(self.mesh_device.shape, decode=ModeConfig(tp=self.tp))
            self.ccl_manager = ccl_manager or CCLManager(self.mesh_device)
        else:
            self.mesh_config = None
            self.ccl_manager = None

        # Core dimensions from HF config
        self.dim = self.hf_config.vision_config.hidden_size
        self.unpadded_hidden_dim = self.hf_config.vision_config.intermediate_size
        self.hidden_dim = nearest_multiple(  # pad to a tile multiple per device
            self.unpadded_hidden_dim, self.tile_size * self.tp
        )
        if self.hidden_dim != self.unpadded_hidden_dim:
            logger.info(f"padding hidden dim from {self.unpadded_hidden_dim} to {self.hidden_dim}")
        self.head_dim = self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_key_value_heads
        self.n_heads = self.hf_config.vision_config.num_key_value_heads
        self.n_kv_heads = self.hf_config.vision_config.num_key_value_heads

        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size

        if self.padded_head_dim != self.head_dim:
            logger.info(f"padding head dim from {self.head_dim} to {self.padded_head_dim}")

        self.qkv_size = self.padded_head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.MAX_QKV_MM_SEQ_LEN = self.MAX_QKV_MM_SEQ_LEN

        self.optimizations = ModelOptimizations(
            self.model_name
        )  # todo)) implement finer grained control similar to tt_transformers'

        num_rows = lambda seq_len: min(seq_len, 1024 if self.is_galaxy else 2048)
        k_dim = self.dim // self.cluster_shape[0] if self.is_galaxy else self.dim
        n_dim = self.dim // self.cluster_shape[1] if self.is_galaxy else self.dim
        self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
            m=num_rows(seq_len),
            k=k_dim,
            n=n_dim,
            grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
            in0_block_w=1 if self.is_galaxy else self.dim // 1024,
            fuse_batch=seq_len <= 1024,
        )

        assert self.n_heads % self.tp == 0, f"n_heads ({self.n_heads}) must be divisible by TP ({self.tp})"
        assert self.n_kv_heads % self.tp == 0, f"n_kv_heads ({self.n_kv_heads}) must be divisible by TP ({self.tp})"

    def prepare_residual_tensor_prefill(self, x_bsh):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim

        Activations are replicated: the tower is tensor-parallel, so every device
        needs the full hidden dim at each block boundary.
        """
        x_1BSH = x_bsh.unsqueeze(0)

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return xs_1BSH

    # Visual model does not use distributed norm for now
    def is_distributed_norm(self, mode):
        return False

    def get_state_dict_prefix(self, module_name, layer_num=None, deepstack_merger_num=None):
        layer_prefix = f"visual.encoder.layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "mlp",
            "Gemma4VisionMLP": "mlp",
            "VisionAttention": "self_attn",
            "VisionBlock": "",
            "VisionTransformer": "visual",
            "PatchMerger": "visual.merger",
            "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm",
            "pre_feedforward_layernorm": "pre_feedforward_layernorm",
            "post_feedforward_layernorm": "post_feedforward_layernorm",
            "DeepstackMerger": f"visual.deepstack_merger_list.{deepstack_merger_num}",
            "": "",  # If no module is given, just get layer prefix
        }
        return layer_prefix + module_map[module_name]

    def reference_vision_model(self, depth=None):
        # Workaround until Qwen2.5-VL is fully integrated into a HF release
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as AutoModelForCausalLM

        print("Loading Gemma-4 model: ", AutoModelForCausalLM)
        config = AutoModelForCausalLM.config_class.from_pretrained(self.CKPT_DIR)
        config.vision_config.num_hidden_layers = depth if depth is not None else config.vision_config.num_hidden_layers
        model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR, config=config)
        return model.model.vision_tower

    def reference_vision_block(self, layer_num=0):
        return self.reference_vision_model().encoder.layers[layer_num]

    def reference_mlp(self):
        return self.reference_vision_block().mlp

    def reference_attention(self):
        return self.reference_vision_block().self_attn

    def reference_rms_norm(self):
        return self.reference_vision_block().norm2

    def reference_pooler(self):
        return self.reference_vision_model().pooler

    def reference_rotary_emb(self):
        return self.reference_vision_model().encoder.rotary_emb

    def reference_patch_merger(self):
        return self.reference_vision_model().merger

    def reference_deepstack_merger(self):
        return self.reference_vision_model().deepstack_merger_list[0]

    def reference_patch_embedder(self):
        return self.reference_vision_model().patch_embedder

    def reference_patch_embed(self):
        return self.reference_vision_model().patch_embed
