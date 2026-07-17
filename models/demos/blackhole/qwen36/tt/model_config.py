# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B model configuration for Blackhole P150.

Subclasses ``tt_transformers.ModelArgs`` (the framework convention). ``HF_MODEL`` is
the single source of truth and must be exported (the base raises a clear error if it
is unset). It may be a local checkpoint directory OR a hub id; a hub id is resolved to
a local snapshot directory via ``snapshot_download`` (same as the vLLM wrapper) because
``AutoConfig.from_pretrained`` on a bare hub id is unreliable in this transformers
version. Config, weights and the tokenizer are loaded with ``trust_remote_code=True``.
The base class resolves the checkpoint from ``HF_MODEL`` (``self.CKPT_DIR``) and parses the HF config via
``AutoConfig``. No JSON config override and no ``checkpoint_dir`` constructor param remain.

Everything Qwen3.5-specific (hybrid Gated DeltaNet + Gated Full Attention layers,
DeltaNet key/value heads + conv kernel, partial rotary factor) is read from the
parsed HF text config and set on top of the base params after ``super().__init__()``.

Weight loading (``load_state_dict``) and the bfp8/bf16 cache path
(``weight_cache_path``) are overridden: the 9B uses its own remapped key scheme,
NOT the framework's meta-style wq/wk/wv keys. Weights come from
``transformers.AutoModelForCausalLM.from_pretrained`` (resolves to the text-only
``Qwen3_5ForCausalLM`` — no vision tower) and are remapped to the internal scheme.
"""
import os
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs

# l1_small_size the GDN prefill depthwise ttnn.conv1d requires.
GDN_CONV1D_L1_SMALL_SIZE = 24576


class Qwen36ModelArgs(ModelArgs):
    """Model configuration for Qwen3.5-9B on Blackhole P150."""

    def __init__(
        self,
        mesh_device=None,
        max_batch_size=1,
        max_seq_len=2048,
        **kwargs,
    ):
        # HF_MODEL (set in the environment) is the single source of truth: the base
        # ModelArgs reads it into self.CKPT_DIR. It defaults to the Qwen/Qwen3.6-27B hub
        # id when unset, so no local checkpoint path is ever hardcoded.
        # Unless HF_MODEL already points at a local checkpoint dir (one containing
        # config.json), resolve it to a local snapshot dir via snapshot_download (same
        # as the vLLM wrapper): AutoConfig.from_pretrained on a bare hub id is unreliable
        # in this transformers version, but works on a directory path. The config.json
        # check (rather than os.path.isdir) avoids being fooled by a stray relative dir
        # created by the weight tensor cache when an unresolved hub id was used before.
        hf_model = os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.6-27B")
        if not os.path.isfile(os.path.join(hf_model, "config.json")):
            from huggingface_hub import snapshot_download

            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            os.environ["HF_MODEL"] = snapshot_download(hf_model, local_files_only=offline)
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)

        # The base resolves the checkpoint dir from HF_MODEL into self.CKPT_DIR; mirror
        # it onto self.checkpoint_dir so weight_cache_path / load_state_dict keep working.
        self.checkpoint_dir = self.CKPT_DIR

        # ------------------------------------------------------------------
        # Qwen3.5-specific params the base does not know about. Read from the
        # parsed HF text config (Qwen3_5TextConfig). The base already set
        # head_dim, dim, n_heads, n_kv_heads, n_layers, vocab_size, norm_eps, etc.
        # ------------------------------------------------------------------
        text_config = self.hf_config.get_text_config()

        # RoPE — both rope_theta and partial_rotary_factor live nested under rope_parameters in
        # these checkpoints; read them there FIRST. Some configs (e.g. 3.6-27B) also hoist
        # partial_rotary_factor to the top level, but others (3.5-27B / 3.5-27B-FP8) only nest it —
        # reading the top level alone silently fell back to 1.0 for those, rotating the full
        # head_dim instead of the trained 0.25 fraction and scrambling RoPE at long context
        # (fine short, degrades past ~32k). Prefer nested, then top-level, then the 1.0 default.
        rope_params = getattr(text_config, "rope_parameters", None) or {}
        self.rope_theta = rope_params.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = rope_params.get(
            "partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 1.0)
        )
        self.rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # M-RoPE (multimodal rotary). The 3 sections (T, H, W) sum to rope_head_dim // 2 and drive
        # the interleaved-mrope cos/sin (modeling_qwen3_5.Qwen3_5RotaryEmbedding). For the "default"
        # rope type Qwen3.5 uses, attention_scaling is 1.0 (so text cos/sin are unchanged). The
        # spatial_merge_size + image/video token ids let the model derive the 3D position ids on
        # host from input_ids + image_grid_thw (no dependency on mm_token_type_ids from the caller).
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        self.rope_attention_scaling = 1.0
        vision_config = getattr(self.hf_config, "vision_config", None)
        self.spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
        self.image_token_id = getattr(self.hf_config, "image_token_id", None)
        self.video_token_id = getattr(self.hf_config, "video_token_id", None)

        # DeltaNet-specific parameters (base does not know about these)
        self.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        self.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)
        self.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)

        # Layer type list — base only reads layer_types into a local (to derive
        # sliding_window_pattern); the 9B needs the full list to dispatch DeltaNet
        # vs. full-attention layers.
        self.attention_type_list = getattr(text_config, "layer_types", None) or (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
        )

        # Derived
        self.linear_q_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_v_dim = self.linear_num_value_heads * self.linear_value_head_dim

        # Blackhole P150 device config (lazy import to allow CPU-only testing)
        if mesh_device is not None:
            import ttnn

            self.weight_dtype = ttnn.bfloat8_b
            self.act_dtype = ttnn.bfloat16
        else:
            self.weight_dtype = None
            self.act_dtype = None

        # ------------------------------------------------------------------
        # Tensor-parallel (multi-device) config. Inert on a single device:
        # the entire block only runs when num_devices > 1, so the validated
        # 9B single-device path is byte-for-byte unchanged. For 27B on a (1,4)
        # mesh this sets the per-device sharded dims + DRAM-sharded matmul
        # configs ported from models/demos/qwen35_27b. See tt/tp_common.py.
        # ------------------------------------------------------------------
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        if mesh_device is not None and self.num_devices > 1:
            self._init_tp_config(mesh_device)

    def _init_tp_config(self, mesh_device):
        """Set per-device sharded dims + DRAM-sharded matmul/mem configs for TP.

        Only called when num_devices > 1. All dims are derived from the
        HF-config values already parsed above (dim, n_heads, head_dim, and the
        linear_* GDN dims) so the same code serves any Qwen3.5 size whose head
        counts divide evenly by the device count.
        """
        import ttnn
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        tp = self.num_devices
        self.cluster_shape = list(mesh_device.shape)

        # GDN dims (named to match the qwen35_27b reference helpers)
        self.gdn_nk = self.linear_num_key_heads
        self.gdn_dk = self.linear_key_head_dim
        self.gdn_nv = self.linear_num_value_heads
        self.gdn_dv = self.linear_value_head_dim
        self.gdn_conv_kernel_size = self.linear_conv_kernel_dim
        self.gdn_key_dim = self.linear_q_dim  # = nk * dk  (q and k are equal)
        self.gdn_value_dim = self.linear_v_dim  # = nv * dv
        self.gdn_qkv_dim = self.linear_q_dim + self.linear_k_dim + self.linear_v_dim
        self.gdn_z_dim = self.linear_v_dim
        self.gdn_chunk_size = 128  # gated_delta_attn_seq kernel requires 128

        # Per-device (sharded) dims
        assert self.n_heads % tp == 0, f"n_heads {self.n_heads} not divisible by TP={tp}"
        assert self.gdn_nk % tp == 0 and self.gdn_nv % tp == 0, "GDN head counts must divide by TP"
        self.n_local_heads = self.n_heads // tp
        self.n_local_kv_heads = max(1, self.n_kv_heads // tp)
        self.kv_replication = tp > self.n_kv_heads  # False at TP=4 (4 KV heads)
        self.gdn_nk_tp = self.gdn_nk // tp
        self.gdn_nv_tp = self.gdn_nv // tp
        self.gdn_qkv_dim_tp = self.gdn_qkv_dim // tp
        self.gdn_z_dim_tp = self.gdn_z_dim // tp
        self.gdn_qkvz_dim_tp = (self.gdn_qkv_dim + self.gdn_z_dim) // tp
        # Per-device width of the [qkv|z|a|b] fused in-projection: folding the tiny a/b (decay/beta)
        # projection into qkvz removes a whole decode matmul while keeping the (good) K=dim. Default
        # (was QWEN36_GDN_FUSE_AB); gdn/tp.py fuses whenever the qkvz weight is DRAM-sharded.
        self.gdn_qkvzab_dim_tp = self.gdn_qkvz_dim_tp + 2 * self.gdn_nv_tp
        self.gdn_value_dim_tp = self.gdn_value_dim // tp
        self.gdn_key_dim_tp = self.gdn_key_dim // tp
        self.attn_out_dim_tp = (self.n_heads * self.head_dim) // tp
        kv_dim_per_device = self.n_local_kv_heads * self.head_dim

        # DRAM-sharded weight memory configs ─ column-parallel: [hidden, out_tp]
        self.gdn_qkvz_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.gdn_qkvz_dim_tp)
        self.gdn_qkvzab_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.gdn_qkvzab_dim_tp)
        self.attn_qg_weight_memcfg = tpc.create_dram_sharded_mem_config(
            self.dim, self.n_local_heads * self.head_dim * 2
        )
        self.attn_k_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        self.attn_v_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, kv_dim_per_device)
        # Fused [q+gate | k | v] in-projection (P4: QWEN36_FUSED_QKV) — one column-parallel matmul.
        self.attn_qkv_fused_dim_tp = self.n_local_heads * self.head_dim * 2 + 2 * kv_dim_per_device
        self.attn_qkv_fused_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.attn_qkv_fused_dim_tp)
        self.mlp_w1_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)
        self.mlp_w3_weight_memcfg = tpc.create_dram_sharded_mem_config(self.dim, self.hidden_dim // tp)
        # row-parallel out-projections: DRAM-INTERLEAVED (None -> plain ttnn.linear); DRAM-sharding narrow-K here loses to the interleaved 1D kernel and adds 2 reshards/layer.
        self.gdn_out_weight_memcfg = None
        self.attn_wo_weight_memcfg = None
        self.mlp_w2_weight_memcfg = tpc.create_dram_sharded_mem_config(self.hidden_dim // tp, self.dim)

        # DRAM-sharded matmul program configs (decode, m=1)
        M = 1
        self.gdn_qkvz_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.gdn_qkvz_dim_tp)
        self.gdn_qkvzab_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.gdn_qkvzab_dim_tp)
        self.gdn_out_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.gdn_value_dim_tp, self.dim)
        self.attn_qg_progcfg = tpc.create_dram_sharded_matmul_program_config(
            M, self.dim, self.n_local_heads * self.head_dim * 2
        )
        self.attn_k_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_v_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, kv_dim_per_device)
        self.attn_qkv_fused_progcfg = tpc.create_dram_sharded_matmul_program_config(
            M, self.dim, self.attn_qkv_fused_dim_tp
        )
        self.attn_wo_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.attn_out_dim_tp, self.dim)
        self.mlp_w1_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.hidden_dim // tp)
        self.mlp_w3_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.dim, self.hidden_dim // tp)
        self.mlp_w2_progcfg = tpc.create_dram_sharded_matmul_program_config(M, self.hidden_dim // tp, self.dim)

        # 1D decode MLP matmuls (DEFAULT): small grids beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny (M<=1) decode matmuls. Interleaved weights.
        # decode_grid_w = the device worker-grid width (11 on BH P150, 8 on WH). Shaping the 1D-mcast
        # grid WIDE-first (up to this many cols) beats the old cols<=8 shaping by ~2% on this matmul —
        # a wide-short grid shortens the in0 multicast column (test_mlp_matmul_sweep wide1d_* vs
        # forced1d_*). Applied to gate/up ONLY (the swept, verified projections); the others below keep
        # the legacy cols<=8 shaping (grid_w default) until their shapes are swept too.
        self.decode_grid_w = mesh_device.compute_with_storage_grid_size().x
        self.mlp_1d_decode = True
        # gate/up: num_cores=44 -> 11x4 on BH, the fastest measured config (wide1d_11x4c, 42.8us vs
        # 43.9us for the old 8x4=forced1d_32c). On WH (decode_grid_w=8) this falls back to 8x6.
        self.mlp_w1_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=44,
            fused_activation=ttnn.UnaryOpType.SILU,
            grid_w=self.decode_grid_w,
        )
        self.mlp_w3_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.hidden_dim // tp, num_cores=44, grid_w=self.decode_grid_w
        )
        # down: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~63us, +28% vs
        # the old 8x2). On WH (decode_grid_w=8) this falls back to 8x5.
        self.mlp_w2_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.hidden_dim // tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        # Input-projection 1D decode (DEFAULT): same idea for attn QKV+gate and GDN QKVZAB in-projections.
        # Weights load interleaved (prefill AGMM verified bit-identical); tuned grids per test_mlp_matmul_sweep.
        self.proj_1d_decode = True
        self.attn_qkv_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.attn_qkv_fused_dim_tp, num_cores=64
        )
        # gdn_qkvz: num_cores=44 -> 11x4 on BH, the fastest measured config (wide1d_11x4c, ~59us, +22%
        # vs the old 8x5). On WH (decode_grid_w=8) this falls back to 8x6.
        self.gdn_qkvz_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.gdn_qkvzab_dim_tp, num_cores=44, grid_w=self.decode_grid_w
        )
        # Output projections (attn wo, GDN o_proj): already interleaved+auto (no weight relayout, not in
        # the prefill AGMM fusion), so this just swaps ttnn-auto for a tuned ~32-core 1D decode grid.
        # attn_wo: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4). On WH (decode_grid_w=8) this falls back to 8x5.
        self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.attn_out_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )
        # gdn_out: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4; same 1536x5120 shape as attn_wo). On WH (decode_grid_w=8) this falls back to 8x5.
        self.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.gdn_value_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        # Prefill matmul factory (M = seq_len)
        self._prefill_grid = tpc.prefill_grid_default()
        self.prefill_progcfg = lambda seq_len, k, n: tpc.create_prefill_matmul_program_config(
            seq_len, k, n, grid_size=self._prefill_grid
        )

        # Activation shard configs
        self.act_shard_hidden = tpc.create_activation_shard_config(self.dim)
        self.act_shard_gdn_value = tpc.create_activation_shard_config(self.gdn_value_dim_tp)
        self.act_shard_attn_out = tpc.create_activation_shard_config(self.attn_out_dim_tp)

        # KV-cache height-shard config for paged_update_cache (decode). The op
        # dispatches one user per core, so the grid must have exactly
        # max_batch_size cores (B=32 → 8x4; B=1 → 1x1).
        _B = max(1, self.max_batch_size)
        _cols = next(c for c in range(min(8, _B), 0, -1) if _B % c == 0)
        _rows = _B // _cols
        self.kv_update_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(tpc.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(x=_cols, y=_rows),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _set_hf_params(self, checkpoint_dir):
        # Load the HF config with trust_remote_code=True. Set the
        # flag before delegating so the base AutoConfig.from_pretrained call uses it.
        self.trust_remote_code_hf = True
        super()._set_hf_params(checkpoint_dir)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "full_attention"

    def is_deltanet_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "linear_attention"

    def weight_cache_path(self, dtype=None):
        """Weight tensor cache dir, rooted at model_cache_path (TT_CACHE_PATH + device), NOT the HF
        snapshot (often read-only in CI -> caching there silently never persists); falls back to the
        checkpoint dir. TP caches qualified by mesh shape: per-device layouts differ by mesh and
        as_tensor reloads a cache file as-is, IGNORING mesh_mapper (single device keeps the
        unqualified path so validated 9B behavior is unchanged)."""
        if dtype is None:
            dtype = self.weight_dtype
        import ttnn

        if dtype == ttnn.bfloat8_b:
            suffix = "tensor_cache_bfp8"
        else:
            suffix = "tensor_cache_bf16"
        if self.num_devices > 1:
            suffix += "_mesh" + "x".join(str(d) for d in self.cluster_shape)
        root = getattr(self, "model_cache_path", None) or Path(self.checkpoint_dir)
        return Path(root) / suffix

    def load_state_dict(self):
        """Load + remap this checkpoint's weights via transformers from_pretrained.

        HF_MODEL (self.CKPT_DIR) is the single source — a hub name or local path.
        AutoModelForCausalLM resolves to the TEXT-ONLY Qwen3_5ForCausalLM (no vision
        tower is built), whose state_dict uses the `model.` prefix; remap_qwen36_state_dict
        normalizes that to the internal key scheme. This OVERRIDES the base meta-key
        (wq/wk/wv) loader — the 9B uses its own scheme.
        """
        from models.demos.blackhole.qwen36.tt.weight_mapping import (
            is_fp8_checkpoint,
            load_qwen36_state_dict_fp8,
            remap_qwen36_state_dict,
        )

        # Block-wise FP8 checkpoints (e.g. Qwen3.5-27B-FP8) cannot go through
        # AutoModelForCausalLM here; dequant + remap to the TP key scheme that
        # the multi-device weight loaders consume.
        if is_fp8_checkpoint(self.CKPT_DIR):
            return load_qwen36_state_dict_fp8(self.CKPT_DIR)

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR, dtype="auto", trust_remote_code=True)
        state_dict = remap_qwen36_state_dict(model.state_dict())
        del model
        return state_dict
