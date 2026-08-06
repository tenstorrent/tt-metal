# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B config for Blackhole P150.

Subclasses tt_transformers.ModelArgs. HF_MODEL env var is canonical (hub id or local dir);
hub ids are snapshot_download'd first (AutoConfig on bare hub id is unreliable here).
Qwen3.5-specific params (GDN, partial RoPE, layer types) come from HF text config.
load_state_dict/weight_cache_path override the base meta-key (wq/wk/wv) scheme.
"""
import os
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs

# l1_small_size the GDN prefill depthwise ttnn.conv1d requires.
GDN_CONV1D_L1_SMALL_SIZE = 24576


class Qwen36ModelArgs(ModelArgs):
    """Qwen3.5-9B ModelArgs for Blackhole P150."""

    # Opt into base ModelArgs TP > n_kv_heads path; attention/tp.py replicates via replicate_kv_weight.
    SUPPORTS_KV_REPLICATION = True

    def __init__(
        self,
        mesh_device=None,
        max_batch_size=1,
        max_seq_len=2048,
        **kwargs,
    ):
        # HF_MODEL is canonical (defaults to Qwen/Qwen3.6-27B). Snapshot hub ids unless
        # config.json exists locally (avoids cache-dir false positives).
        hf_model = os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.6-27B")
        if not os.path.isfile(os.path.join(hf_model, "config.json")):
            from huggingface_hub import snapshot_download

            offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
            os.environ["HF_MODEL"] = snapshot_download(hf_model, local_files_only=offline)
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        if mesh_device is not None:
            self.model_config["SAMPLING_AG_CONFIG"]["allow_force_argmax"] = True

        # Mirror CKPT_DIR -> checkpoint_dir for weight_cache_path / load_state_dict.
        self.checkpoint_dir = self.CKPT_DIR

        # Qwen3.5-specific params from HF text config (base sets dim, heads, layers, etc.).
        text_config = self.hf_config.get_text_config()

        # RoPE: read partial_rotary_factor from rope_parameters first (some configs nest only there).
        # Top-level-only read silently used 1.0 and broke long-context RoPE on 3.5-27B.
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

        # Full layer_types list for DeltaNet vs full-attn dispatch.
        self.attention_type_list = getattr(text_config, "layer_types", None) or (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
        )

        # Derived
        self.linear_q_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_v_dim = self.linear_num_value_heads * self.linear_value_head_dim

        # Lazy import for CPU-only testing.
        if mesh_device is not None:
            import ttnn

            self.weight_dtype = ttnn.bfloat8_b
            self.act_dtype = ttnn.bfloat16
        else:
            self.weight_dtype = None
            self.act_dtype = None

        # TP config (num_devices>1 only). 27B (1,4) sharded dims + DRAM matmul cfgs; see tp_common.py.
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        if mesh_device is not None and self.num_devices > 1:
            self._init_tp_config(mesh_device)

    def _init_tp_config(self, mesh_device):
        """Per-device sharded dims + DRAM matmul/mem configs for TP (num_devices>1)."""
        import ttnn
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        tp = self.num_devices
        self.cluster_shape = list(mesh_device.shape)

        # GDN dims (match qwen35_27b reference names).
        self.gdn_nk = self.linear_num_key_heads
        self.gdn_dk = self.linear_key_head_dim
        self.gdn_nv = self.linear_num_value_heads
        self.gdn_dv = self.linear_value_head_dim
        self.gdn_conv_kernel_size = self.linear_conv_kernel_dim
        self.gdn_key_dim = self.linear_q_dim  # q and k equal
        self.gdn_value_dim = self.linear_v_dim
        self.gdn_qkv_dim = self.linear_q_dim + self.linear_k_dim + self.linear_v_dim
        self.gdn_z_dim = self.linear_v_dim
        self.gdn_chunk_size = 128  # GDN seq kernel requires 128

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
        # WORMHOLE: pad this width up so its TILE COUNT factors well for the prefill matmul. The
        # 2048x4096x6176 in-proj is K-PASS bound, not subblock bound — its cost tracks
        # per_core_N / out_block_w — and 6176 = 193 tiles is PRIME, so at 8 columns per_core_N=25 whose
        # only divisors are 1/5/25; out_block_w can only be 5 (five passes over K, each re-reading the
        # 16MB DRAM-resident in0) because 25 overflows L1. prefill_kpass_width picks the smallest wider
        # width that minimises those passes: 216 tiles => per_core_N=27, out_block_w=9, 3 passes, still
        # all 64 cores. See its docstring for the width sweep this is derived from.
        #
        # CONFIRMED IN THE REAL LAYER (single-layer GDN prefill, seq 2048, N300):
        #     op 287  2,155us @ 36.5% of peak FLOPs  ->  1,512us @ 58.3%   (-643us)
        #     layer   18,644us -> 17,916us  (-728us, -3.9%)
        # It becomes the most efficient matmul in the layer, above the o_proj's 48.6%. PCC is unchanged:
        # the pad rows sit past every slice offset in _project_qkvzab, so they are never read.
        #
        # KNOWN COST, ACCEPTED: this width also sizes gdn_qkvzab_weight_memcfg and both decode
        # progcfgs, and the same weight serves decode — so decode's in-proj does ~11.9% more work and
        # reads ~11.9% more weight bytes. Prefill is the current target and decode is a later pass; when
        # decode is tuned, re-measure. If the decode cost turns out to matter, the fix is a second
        # unpadded weight for decode (+~12.6MB/layer of DRAM) rather than giving up the prefill win.
        #
        # Deliberately NOT env-gated. Unlike the runtime dtype switches elsewhere in this model
        # (QWEN35_GDN_OUT_FP32 etc.), this changes weight GEOMETRY — the DRAM-sharded memcfg, both
        # decode progcfgs and the weight cache key — so a flag would let two processes build different
        # weights for the same model. To revert, delete this block; prefill_kpass_width() is a no-op on
        # widths that already factor well, so nothing else needs touching. Blackhole is excluded: its
        # grid, L1 budget and TP all differ and its blocking was tuned at the unpadded width.
        self.gdn_qkvzab_pad_tiles = 0
        if not tpc.is_blackhole():
            _pad_to = tpc.prefill_kpass_width(self.gdn_qkvzab_dim_tp)
            if _pad_to > self.gdn_qkvzab_dim_tp:
                self.gdn_qkvzab_pad_tiles = (_pad_to - self.gdn_qkvzab_dim_tp) // tpc.TILE_SIZE
                self.gdn_qkvzab_dim_tp = _pad_to
        self.gdn_value_dim_tp = self.gdn_value_dim // tp
        self.gdn_key_dim_tp = self.gdn_key_dim // tp
        self.attn_out_dim_tp = (self.n_heads * self.head_dim) // tp
        kv_dim_per_device = self.n_local_kv_heads * self.head_dim

        # DRAM-sharded weights: column-parallel [hidden, out_tp]
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

        # DRAM-sharded matmul progcfgs (decode, M=1)
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

        # Prefill matmul factory (M = seq_len). Shared by MLP down-proj, attention in/out-proj, and
        # GDN in/out-proj. On Blackhole the grid is already wider (8x10) with more L1 headroom, so the
        # full per_core_N-wide output/intermediate CB fits (validated there); WH's grid tops out at
        # 8x8 with less L1/core, and a full decoder layer's combined GDN+attention+MLP CBs on this
        # config alone were measured to overflow (1.68-1.85MB vs WH's 1.5MB max) — halve it there (see
        # create_prefill_matmul_program_config's halve_out_block).
        self._prefill_grid = tpc.prefill_grid_default()
        self.prefill_tuning = tpc.prefill_tuning(tp)
        self.prefill_progcfg = lambda seq_len, k, n: tpc.create_prefill_matmul_program_config(
            seq_len,
            k,
            n,
            grid_size=self._prefill_grid,
            tuning=self.prefill_tuning,
            halve_out_block=not tpc.is_blackhole(),
        )

        # Activation shard configs
        self.act_shard_hidden = tpc.create_activation_shard_config(self.dim)
        self.act_shard_gdn_value = tpc.create_activation_shard_config(self.gdn_value_dim_tp)
        self.act_shard_attn_out = tpc.create_activation_shard_config(self.attn_out_dim_tp)

        # KV-cache height shard for paged_update_cache (one user per core).
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
        # trust_remote_code before base AutoConfig load.
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
        """Load + remap weights via the text-only HF Qwen3_5ForCausalLM.
        Overrides base meta-key loader."""
        from models.demos.blackhole.qwen36.tt.weight_mapping import (
            is_fp8_checkpoint,
            load_qwen36_state_dict_fp8,
            remap_qwen36_state_dict,
        )

        # Block FP8 checkpoints: dequant + remap for TP loaders (skip the HF model).
        if is_fp8_checkpoint(self.CKPT_DIR):
            return load_qwen36_state_dict_fp8(self.CKPT_DIR)

        # Import the HF classes directly rather than going through AutoModelForCausalLM.
        # Serving out-of-tree, vllm.transformers_utils.config registers vLLM's OWN
        # Qwen3_5Config for model_type "qwen3_5" into transformers' AutoConfig
        # (AutoConfig.register(..., exist_ok=True)), so AutoConfig hands back vLLM's class.
        # transformers only unwraps a composite config to its text sub-config when
        # `model_class.config_class == config.sub_configs["text_config"]` — an identity
        # check that cannot hold across libraries — so the composite config would reach
        # Qwen3_5ForCausalLM and fail on `config.vocab_size` (which lives one level down,
        # in text_config). Naming the classes here keeps config and model from the same
        # library, matching vision/vision_model_config.py::reference_vision_model.
        #
        # Qwen3_5TextConfig.from_pretrained picks the `text_config` sub-dict on composite
        # (3.6 VLM) checkpoints via base_config_key, and reads a text-only (3.5) config.json
        # as-is, so both checkpoint layouts land on the config Qwen3_5ForCausalLM expects.
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        text_config = Qwen3_5TextConfig.from_pretrained(self.CKPT_DIR)
        assert text_config.vocab_size == self.vocab_size and text_config.hidden_size == self.dim, (
            f"HF text config disagrees with model args: vocab_size {text_config.vocab_size} vs "
            f"{self.vocab_size}, hidden_size {text_config.hidden_size} vs {self.dim}"
        )
        model = Qwen3_5ForCausalLM.from_pretrained(self.CKPT_DIR, config=text_config, dtype="auto")
        state_dict = remap_qwen36_state_dict(model.state_dict())
        del model
        return state_dict
