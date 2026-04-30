# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Molmo2-8B TTNN implementation."""


import ttnn


class Molmo2Config:
    """Configuration for Molmo2-8B TTNN model.

    Attribute layout mirrors what tt_transformers Attention/MLP classes expect from a
    `configuration` argument, so those forward methods can be reused verbatim.
    """

    # ------------------------------------------------------------------ #
    # Text decoder dimensions (from text_config)
    # ------------------------------------------------------------------ #
    dim = 4096
    n_heads = 32
    n_kv_heads = 8
    head_dim = 128  # 4096 / 32; already tile-aligned (128 = 4 × 32)
    padded_head_dim = 128
    norm_eps = 1e-6
    vocab_size = 152064  # 151936 base + 128 new tokens
    n_layers = 36
    rope_theta = 1_000_000.0
    max_seq_len = 36864
    intermediate_size = 12288
    image_patch_id = 151938

    # ------------------------------------------------------------------ #
    # ViT encoder dimensions (from vit_config)
    # ------------------------------------------------------------------ #
    vit_hidden = 1152
    vit_n_heads = 16
    vit_head_dim = 72
    vit_padded_head_dim = 96  # ceil(72 / 32) × 32
    vit_intermediate = 4304
    vit_n_layers = 25  # only 25 of 27 blocks are built
    vit_capture_layers = (24, 18)  # HF vit_layers = [-3, -9] → [24, 18]
    vit_pos_embed_seq = 729  # 27 × 27 patches per crop
    vit_norm_eps = 1e-6
    vit_qkv_size = 3 * vit_head_dim * vit_n_heads  # = 3456

    # ------------------------------------------------------------------ #
    # Image pooling adapter dimensions
    # ------------------------------------------------------------------ #
    pool_n_heads = 16
    pool_head_dim = 72
    pool_padded_head_dim = 96
    pool_hidden = 1152
    pool_dim = 2304  # 2 × vit_hidden; input dim for wq/wk/wv
    pool_qkv_size = 3 * pool_head_dim * pool_n_heads  # = 3456

    # ------------------------------------------------------------------ #
    # Image projector dimensions
    # ------------------------------------------------------------------ #
    proj_intermediate = 12288

    # ------------------------------------------------------------------ #
    # T3K / multi-device settings
    # ------------------------------------------------------------------ #
    cluster_shape = [1, 8]
    is_multichip = True

    # ------------------------------------------------------------------ #
    # Inference settings
    # ------------------------------------------------------------------ #
    max_batch_size = 1
    dummy_weights = False

    # ------------------------------------------------------------------ #
    # CCL settings
    # ------------------------------------------------------------------ #
    min_kv_prefill_shard_seqlen = 1024
    ccl_dtype = ttnn.bfloat8_b
    num_reduce_scatter_links = 1
    num_all_gather_links = 1
    tile_size = 32
    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(self, mesh_device=None):
        self.mesh_device = mesh_device

        if mesh_device is not None:
            self.num_devices = mesh_device.get_num_devices()
        else:
            self.num_devices = 8  # T3K default

        self.is_multichip = self.num_devices > 1
        # When num_devices=1, each "local" = all heads (no tensor parallel)
        self.n_local_heads = self.n_heads // max(1, self.num_devices)
        self.n_local_kv_heads = self.n_kv_heads // max(1, self.num_devices)
        # QKV size per device (full size when single device)
        self.qkv_size = self.head_dim * (self.n_heads + 2 * self.n_kv_heads)  # 6144
        # For weight sharding: column-parallel splits output dim across devices
        self.qkv_size_per_device = self.qkv_size // max(1, self.num_devices)
        self.tile_padded_batch_rows = self.max_batch_size * self.tile_size
        # cluster_shape for ShardTensor2dMesh
        if self.num_devices == 1:
            self.cluster_shape = [1, 1]
        elif self.num_devices == 2:
            self.cluster_shape = [1, 2]
        else:
            self.cluster_shape = [1, 8]

        # Compute kernel configs (Wormhole B0)
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self._model_config = self._build_model_config()

    # ------------------------------------------------------------------ #
    # Interface expected by tt_transformers attention/MLP
    # ------------------------------------------------------------------ #

    def _build_model_config(self):
        cfg = {}
        cfg["ATTN_W_LAYOUT_TILE"] = ttnn.TILE_LAYOUT
        cfg["USE_FUSED_ALL_GATHER_MATMUL"] = False
        cfg["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.DRAM_MEMORY_CONFIG
        cfg["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            exp_approx_mode=False,
            q_chunk_size=256,
            k_chunk_size=256,
        )
        cfg["SDPA_DECODE_COMPUTE_PROGCFG"] = self.compute_kernel_config_hifi4

        # chunk=256 is L1-safe (1.08 MB < 1.43 MB max).
        # S is padded to the next power-of-2 in model.py forward_prefill before the
        # decoder loop, ensuring S_pad/256 is always a power-of-2 (no partial tiles,
        # no problematic Q-tile counts like 19 that caused TTNN SDPA deadlocks).
        def sdpa_prog(seq_len):
            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=256,
                k_chunk_size=256,
            )

        cfg["SDPA_PROGCFG"] = sdpa_prog

        # Phase 1: let TTNN auto-select matmul program configs.
        # Hand-crafted configs require seq_len divisible by specific grid factors and
        # would fail for arbitrary lengths (e.g. 2701 for 30-frame video).
        cfg["XQKV_PREFILL_PROGCFG"] = lambda seq_len: None
        cfg["WO_PREFILL_PROGCFG"] = lambda seq_len: None

        cfg["SCORES_BATCHED_MM_OUTPUT_MEMCFG"] = lambda batch: ttnn.DRAM_MEMORY_CONFIG
        cfg["QKV_OUT_GATHERED_MEMCFG"] = lambda n_devices: ttnn.L1_MEMORY_CONFIG
        cfg["GATHER_USERS_MEMCFG"] = lambda n_devices: ttnn.L1_MEMORY_CONFIG
        cfg["DECODE_RESIDUAL_MEMCFG"] = ttnn.L1_MEMORY_CONFIG
        cfg["ATTN_OUTPUT_PROGCFG"] = None
        cfg["XQKV_DECODE_PROGCFG"] = None
        return cfg

    def get_model_config(self):
        return self._model_config

    def ccl_topology(self):
        return ttnn.Topology.Ring

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM sharded memory config (simplified for correctness)."""
        return ttnn.DRAM_MEMORY_CONFIG

    def get_state_dict_prefix(self, class_name, layer_num=None):
        """Return the state dict key prefix for the given class and layer."""
        if class_name in ("TtMolmo2TextAttention",):
            if layer_num is not None:
                return f"model.transformer.blocks.{layer_num}.self_attn"
        elif class_name in ("TtMolmo2TextMLP",):
            if layer_num is not None:
                return f"model.transformer.blocks.{layer_num}.mlp"
        elif class_name in ("TtMolmo2ViTBlock", "ViTAttention"):
            if layer_num is not None:
                return f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"
        elif class_name == "TtMolmo2ImagePooling2D":
            return "model.vision_backbone.image_pooling_2d"
        elif class_name == "TtMolmo2ImageProjector":
            return "model.vision_backbone.image_projector"
        return ""
