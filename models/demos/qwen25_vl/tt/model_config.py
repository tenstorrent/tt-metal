# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import os

from loguru import logger

import ttnn
from models.demos.qwen25_vl.tt.common import nearest_multiple
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from models.tt_transformers.tt.model_config import DecodersPrecision
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs


def find_qwen_vl_width_grid(width: int, tile_size: int, max_rows: int = 4, max_cols: int = 8) -> tuple[int, int]:
    """Find the largest row/column grid whose width shard remains tile-aligned."""
    for rows in range(max_rows, 0, -1):
        for cols in range(max_cols, 0, -1):
            if width % (tile_size * rows * cols) == 0:
                return rows, cols
    raise ValueError(f"Could not find a tile-aligned grid for width={width}")


# Default data-parallel split of a 32-chip Wormhole Galaxy per base model; each DP lane is a
# T3K-like 1x8 submesh (see tt_transformers create_submeshes).
GALAXY_DEFAULT_DATA_PARALLEL = {
    "Qwen2.5-VL-3B": 4,
    "Qwen2.5-VL-7B": 4,
    "olmOCR-2-7B": 4,
    "Qwen2.5-VL-32B": 4,
    "Qwen2.5-VL-72B": 4,
}


def default_data_parallel(hf_model: str, num_devices: int) -> int:
    """Data-parallel lanes to use for ``hf_model`` on ``num_devices`` (TT_DATA_PARALLEL overrides)."""
    override = os.environ.get("TT_DATA_PARALLEL") or os.environ.get("DATA_PARALLEL")
    if override:
        return int(override)
    if num_devices != 32:
        return 1
    for base_name, data_parallel in GALAXY_DEFAULT_DATA_PARALLEL.items():
        if base_name in hf_model:
            return data_parallel
    return 1


def qwen25_vl_mesh_shape():
    """Mesh shape to open for MESH_DEVICE (falls back to the number of visible devices).

    Galaxy data-parallel runs open the 32 chips directly as the 4x8 view that create_submeshes
    splits into 1x8 lanes, so the parent mesh is never reshaped after open (a reshape leaves the
    real-time profiler bound to the old coordinates).
    """
    mesh_device_env = os.environ.get("MESH_DEVICE")
    if mesh_device_env == "TG":
        return (4, 8) if default_data_parallel(os.environ.get("HF_MODEL", ""), 32) > 1 else (8, 4)
    return {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(mesh_device_env, len(ttnn.get_device_ids()))


class ModelArgs(TTModelArgs):
    LOCAL_HF_PARAMS = {
        **TTModelArgs.LOCAL_HF_PARAMS,
        "olmOCR-2-7B-1025": "models/tt_transformers/model_params/Qwen2.5-VL-7B-Instruct",
    }

    def _lm_head_grid_dims(self):
        # On a full 32-device mesh the LM head is width-sharded over the mesh columns; pick a
        # tile-aligned grid for that per-device width (the base search can reach 0 rows, e.g. dim=3584).
        if self.num_devices == 32:
            return find_qwen_vl_width_grid(self.dim // self.cluster_shape[1], ttnn.TILE_SIZE, max_rows=4, max_cols=8)
        return super()._lm_head_grid_dims()


class ModelOptimizations:
    def __init__(self, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        self.bfp4_mlp = "Qwen2.5-VL-72B" in model_name


class VisionModelArgs(ModelArgs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vision_depth = self.hf_config.vision_config.depth
        if vision_depth > self.n_layers:
            self.model_config["DECODERS_OPTIMIZATIONS"] = DecodersPrecision.accuracy(
                num_decoders=vision_depth, model_name=self.model_name
            )

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
        # WO weight K dimension is n_heads * padded_head_dim (e.g. 1536), not vision_dim (1280)
        # Use in0_block_w=2 which safely divides all possible Kt values
        wo_k_dim = self.vision_n_heads * self.vision_padded_head_dim

        self.model_config["VISION_WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
            m=num_rows(seq_len),
            k=wo_k_dim,
            n=n_dim,
            grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
            in0_block_w=1 if self.is_galaxy else 2,
            fuse_batch=seq_len <= 1024,
        )

        assert self.vision_n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"

        # Enable fused QK ops (rotary embedding + paged cache update) for decode.
        # The base class disables this for multimodal models, but for Qwen2.5-VL the
        # fused path only affects decode (not prefill), and M-RoPE differences are
        # handled in the prefill code path which uses pre-computed rotation matrices.
        self.use_qk_fused = True

        # Minimal matmul configs for text decoder MLP on N300 (experimental, gated behind env var)
        self.use_minimal_matmul = os.getenv("TT_MINIMAL_MATMUL") == "1" and not self.is_galaxy

        if self.use_minimal_matmul:
            logger.info("Minimal matmul enabled for text decoder MLP (experimental)")

            def prefill_ff1_ff3_minimal_matmul_config(seq_len):
                if seq_len <= 4096:
                    return ttnn.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=8,
                        N_block_size=8,
                        subblock_h=4,
                        subblock_w=2,
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    )
                elif seq_len <= 16384:
                    return ttnn.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=8,
                        N_block_size=8,
                        subblock_h=2,
                        subblock_w=4,
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    )
                else:
                    return ttnn.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=8,
                        N_block_size=8,
                        subblock_h=4,
                        subblock_w=2,
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    )

            self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"] = prefill_ff1_ff3_minimal_matmul_config

            def prefill_ff2_minimal_matmul_config(seq_len):
                if seq_len <= 4096:
                    return ttnn.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=8,
                        N_block_size=8,
                        subblock_h=4,
                        subblock_w=2,
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    )
                elif seq_len <= 16384:
                    return ttnn.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=8,
                        N_block_size=8,
                        subblock_h=2,
                        subblock_w=4,
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    )
                else:
                    return ttnn.MinimalMatmulConfig(
                        M_block_size=8,
                        K_block_size=8,
                        N_block_size=8,
                        subblock_h=4,
                        subblock_w=2,
                        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    )

            self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"] = prefill_ff2_minimal_matmul_config

    def prepare_residual_tensor_prefill(self, x_bsh, force_replicated=False):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim
        """

        x_1BSH = x_bsh.unsqueeze(0)

        if force_replicated:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        else:
            mesh_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
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
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration as AutoModelForCausalLM,
        )

        print("Loading Qwen2.5-VL model: ", AutoModelForCausalLM)
        config = AutoModelForCausalLM.config_class.from_pretrained(self.CKPT_DIR)
        config.vision_config.depth = depth if depth is not None else config.vision_config.depth
        vision_model = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)

        if self.dummy_weights:
            import torch

            return vision_model.to(dtype=torch.float32)

        # Load only vision weights to reduce host memory usage.
        key_prefixes = ("visual.", "model.visual.")
        vision_state_dict = load_hf_state_dict_filtered(self.CKPT_DIR, key_prefixes)
        if not vision_state_dict:
            logger.warning(
                "No vision weights found in {} for prefixes {}. Vision model will use default init.",
                self.CKPT_DIR,
                key_prefixes,
            )
            return vision_model

        prefix_stripped_state_dict = {}
        for key, value in vision_state_dict.items():
            if key.startswith("model.visual."):
                key = key[len("model.visual.") :]
            elif key.startswith("visual."):
                key = key[len("visual.") :]
            prefix_stripped_state_dict[key] = value

        model_keys = set(vision_model.state_dict().keys())
        filtered_state_dict = {k: v for k, v in prefix_stripped_state_dict.items() if k in model_keys}

        logger.info(
            "Filtered vision weights: checkpoint_keys={}, prefix_stripped_keys={}, matched_model_keys={}, model_keys={}",
            len(vision_state_dict),
            len(prefix_stripped_state_dict),
            len(filtered_state_dict),
            len(model_keys),
        )
        if not filtered_state_dict:
            logger.warning(
                "No matching vision weights found after filtering for {}. Check prefixes or checkpoint contents.",
                key_prefixes,
            )
            return vision_model

        load_result = vision_model.load_state_dict(filtered_state_dict, strict=False)
        if load_result.missing_keys:
            logger.warning(
                "Vision model missing {} of {} weights after filtered load.",
                len(load_result.missing_keys),
                len(model_keys),
            )
        if load_result.unexpected_keys:
            logger.warning(
                "Vision model received {} unexpected keys after filtered load.",
                len(load_result.unexpected_keys),
            )

        return vision_model

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
