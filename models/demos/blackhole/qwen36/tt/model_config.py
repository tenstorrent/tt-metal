# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Config for the Qwen3.5 / 3.6 family (9B, 27B, 35B-A3B) on Blackhole (P150, P150x4) and Wormhole (N150, N300, T3K).

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
    """ModelArgs for the Qwen3.5 / 3.6 family (9B / 27B / 35B-A3B; dense + MoE); tuning gated in tp_common.py."""

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
            _snap = snapshot_download(hf_model, local_files_only=offline)
            os.environ["HF_MODEL"] = _snap
            # ModelArgs takes model_name from HF_MODEL's BASENAME, which the line above just turned
            # into an opaque snapshot hash ("c202236235..."), not "Qwen3.5-9B". Everything that keys
            # off the hub name then misses -- notably the dummy_weights=True path, whose
            # LOCAL_HF_PARAMS[model_name] lookup raised KeyError and took out every vision test that
            # builds from dummy weights (test_vision_block, test_wrapped_model: 4 failures on both
            # the 9B and the 27B). Alias the hash to whatever the friendly name already maps to, so
            # the rewrite stays invisible to those lookups. Additive and idempotent: it only ever
            # adds a key for a snapshot this process resolved, and never rebinds an existing one.
            _friendly = hf_model.strip("/").split("/")[-1]
            _params = ModelArgs.LOCAL_HF_PARAMS.get(_friendly)
            if _params is not None:
                ModelArgs.LOCAL_HF_PARAMS.setdefault(os.path.basename(_snap.rstrip("/")), _params)
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

        # ------------------------------------------------------------------
        # MoE (Qwen3.5-MoE / Qwen3-Next sparse layers). All read from the parsed
        # HF text config. Absent on the dense 9B/27B, where num_experts defaults
        # to 0 → is_moe_layer() is False everywhere and the validated dense
        # Qwen36MLP path is byte-for-byte unchanged. For the 35B-A3B every layer
        # is MoE (decoder_sparse_step=1, mlp_only_layers=[]) with a gated shared
        # expert; see tt/moe/.
        # ------------------------------------------------------------------
        self.moe_num_experts = getattr(text_config, "num_experts", 0) or 0
        self.moe_top_k = getattr(text_config, "num_experts_per_tok", 0) or 0
        self.moe_intermediate_size = getattr(text_config, "moe_intermediate_size", 0) or 0
        self.moe_shared_intermediate_size = getattr(text_config, "shared_expert_intermediate_size", None)
        self.moe_norm_topk_prob = bool(getattr(text_config, "norm_topk_prob", True))
        self.moe_decoder_sparse_step = getattr(text_config, "decoder_sparse_step", 1) or 1
        self.mlp_only_layers = set(getattr(text_config, "mlp_only_layers", None) or [])

        # MTP (multi-token prediction) head. Every Qwen3.5/3.6 checkpoint ships a single-layer
        # MTP head (mtp.*) that reuses the main embedding + LM head; it is the speculative-decode
        # drafter. mtp_use_dedicated_embeddings=False means it shares tok_embeddings.
        self.mtp_num_hidden_layers = getattr(text_config, "mtp_num_hidden_layers", 0)
        self.mtp_use_dedicated_embeddings = getattr(text_config, "mtp_use_dedicated_embeddings", False)
        self.has_mtp = self.mtp_num_hidden_layers > 0
        if self.has_mtp:
            assert (
                self.mtp_num_hidden_layers == 1
            ), f"Only single-layer MTP is supported (got mtp_num_hidden_layers={self.mtp_num_hidden_layers})"
            assert (
                not self.mtp_use_dedicated_embeddings
            ), "mtp_use_dedicated_embeddings=True is unsupported (would need a separate MTP embedding/head)"

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
        # 35B-A3B has 2 KV heads on 4 devices -> True (each KV head replicated across tp/n_kv_heads
        # devices); dense 27B has n_kv_heads >= tp -> False (even shard).
        self.kv_replication = tp > self.n_kv_heads
        self.gdn_nk_tp = self.gdn_nk // tp
        self.gdn_nv_tp = self.gdn_nv // tp
        self.gdn_qkv_dim_tp = self.gdn_qkv_dim // tp
        # Native depthwise conv1d (prefill) keeps all qkv_dim_tp channels resident per core (L1_FULL);
        # the 35B-A3B channel count overflows L1 on BH. Split the conv over N channel chunks (exact —
        # depthwise is per-channel-independent) so each call fits. 27B (chunks=1) is unchanged.
        self.gdn_conv_channel_chunks = 2 if self.moe_num_experts > 0 else 1
        self.gdn_z_dim_tp = self.gdn_z_dim // tp
        self.gdn_qkvz_dim_tp = (self.gdn_qkv_dim + self.gdn_z_dim) // tp
        # Per-device width of the [qkv|z|a|b] fused in-projection. Folding the tiny a/b (decay/beta)
        # projection into qkvz removes a whole decode matmul while keeping K=dim.
        if tpc.is_blackhole():
            self.gdn_ab_gap = 0
        else:
            self.gdn_ab_gap = -(-self.gdn_nv_tp // 32) * 32 - self.gdn_nv_tp
        self.gdn_qkvzab_dim_tp = self.gdn_qkvz_dim_tp + 2 * self.gdn_nv_tp + self.gdn_ab_gap
        # No pad: geometry, cache key and decode progcfgs all use the natural width.
        # The 2048x4096x6176 in-proj is K-PASS bound, not subblock bound -- cost tracks
        self.gdn_qkvzab_pad_tiles = 0
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

        # 1D decode MLP matmuls: small grids beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny (M<=1) decode matmuls. Interleaved weights.
        self.decode_grid_w = mesh_device.compute_with_storage_grid_size().x
        self.mlp_1d_decode = True
        # gate/up: 44 cores (11x4) on BH, the fastest measured (42.8us vs 43.9 for 8x4). On WH,
        # swept at the exact production shape (M=32 K=4096 N=6144, test_mlp_decode_matmul_sweep.py):
        _gateup_9b = tpc.wh_9b_n300(self)
        _gateup_cores = 56 if _gateup_9b else (44 if tpc.is_blackhole() else 64)
        # T3K only: lift create_matmul_1d_decode_progcfg's in0_block_w cap from 8 to 16. N=68 tiles
        # here has no divisor between 35 and 64, so the core count is pinned at 34 and the K-block
        # count is the only lever left. MEASURED 58.0 -> 52.1 us per matmul (1.11x) at identical
        # PCC, and this config is used by the gate/up of EVERY layer, so it lands in verify as well
        # as in the drafter. Left at 8 elsewhere: the other configs' shapes were not swept.
        _gateup_cap = 16 if tpc.wh_t3k(self) else 8
        self.mlp_w1_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=_gateup_cores,
            fused_activation=ttnn.UnaryOpType.SILU,
            grid_w=self.decode_grid_w,
            fp32_acc=not _gateup_9b,
            in0_block_w_cap=_gateup_cap,
        )
        self.mlp_w3_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=_gateup_cores,
            grid_w=self.decode_grid_w,
            fp32_acc=not _gateup_9b,
            in0_block_w_cap=_gateup_cap,
        )
        # down: 33 cores (11x3) on BH, fastest measured (~63us). On WH this falls back to 8x5.
        self.mlp_w2_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.hidden_dim // tp, self.dim, num_cores=33 if tpc.is_blackhole() else 64, grid_w=self.decode_grid_w
        )

        # Input-projection 1D decode (DEFAULT): same idea for attn QKV+gate and GDN QKVZAB in-projections.
        # Weights load interleaved (prefill AGMM verified bit-identical); tuned grids per test_mlp_matmul_sweep.
        self.proj_1d_decode = True
        self.attn_qkv_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.attn_qkv_fused_dim_tp, num_cores=64
        )
        # gdn_qkvz: 44 cores (11x4) on BH, fastest measured (~59us). On WH the full 8x8=64-core grid
        # measured 150.4us vs 8x6's 156.5us (-3.9%, no accuracy cost), matching attn_qkv above.
        self.gdn_qkvz_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.gdn_qkvzab_dim_tp, num_cores=44 if tpc.is_blackhole() else 64, grid_w=self.decode_grid_w
        )
        # Output projections (attn wo, GDN o_proj): already interleaved+auto (no weight relayout, not in
        # the prefill AGMM fusion), so this just swaps ttnn-auto for a tuned ~32-core 1D decode grid.
        self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.attn_out_dim_tp, self.dim, num_cores=33 if tpc.is_blackhole() else 48, grid_w=self.decode_grid_w
        )
        # gdn_out: 33 cores (11x3) on BH, fastest measured (~24us; same 1536x5120 shape as attn_wo).
        self.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.gdn_value_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        # Prefill matmul factory (M = seq_len), shared by the MLP down-proj and the attention/GDN
        # in/out-projections. Blackhole's wider 8x10 grid fits the full per_core_N-wide CB; N300's
        self._prefill_grid = tpc.prefill_grid_default()
        self.prefill_tuning = tpc.prefill_tuning(tp)
        if self.moe_num_experts > 0:
            # The dense TP=4 tuning picks in0_block_w = min(cap, k_tiles // grid), which the
            # 35B-A3B's attention/GDN prefill K dims don't divide (Kt % in0_block_w != 0). Force
            # the divisor path (largest divisor of k_tiles ≤ cap) so the block always divides;
            # dense 9B/27B keep their tuned block.
            self.prefill_tuning = {**self.prefill_tuning, "in0_block_w_divisor": True}
        self.prefill_progcfg = lambda seq_len, k, n: tpc.create_prefill_matmul_program_config(
            seq_len,
            k,
            n,
            grid_size=self._prefill_grid,
            tuning=self.prefill_tuning,
            halve_out_block=tpc.wh_9b_n300(self),
        )
        # WORMHOLE ONLY: one-K-pass factory for the GDN in-projection, paired with
        # COMPUTE_HIFI2_NO_FP32_ACC (gdn/tp.py's _col_proj passes both -- the blocking is only legal
        self.gdn_qkvzab_prefill_progcfg = (
            None
            if tpc.is_blackhole()
            else (
                lambda seq_len, k, n: tpc.create_prefill_kpass1_matmul_program_config(
                    seq_len, k, n, grid_size=self._prefill_grid
                )
            )
        )
        # WORMHOLE ONLY: same one-K-pass fix for the fused attention QKV(+gate) in-projection.
        # N=5120 (160 tiles) -> per_core_N=20 at 8 cols; fp32 dest acc caps the subblock at 4 (five
        self.attn_qkv_fused_prefill_progcfg = (
            None
            if tpc.is_blackhole()
            else (
                lambda seq_len, k, n: tpc.create_prefill_kpass1_matmul_program_config(
                    seq_len, k, n, grid_size=self._prefill_grid
                )
            )
        )
        # WORMHOLE ONLY: same one-K-pass fix for the attention wo (output) projection. N=4096 (128
        # tiles) -> per_core_N=16; fp32 dest acc on gives two K passes, off gives one. MEASURED
        self.attn_wo_prefill_progcfg = (
            None
            if tpc.is_blackhole()
            else (
                lambda seq_len, k, n: tpc.create_prefill_kpass1_matmul_program_config(
                    seq_len, k, n, grid_size=self._prefill_grid
                )
            )
        )

        # Activation shard configs
        self.act_shard_hidden = tpc.create_activation_shard_config(self.dim)
        self.act_shard_gdn_value = tpc.create_activation_shard_config(self.gdn_value_dim_tp)
        self.act_shard_attn_out = tpc.create_activation_shard_config(self.attn_out_dim_tp)
        # Decode token embedding: width-sharded L1 on dim_tp, 32 cores (8x4). Interleaved lands on
        # 1 core / ~21us at B=32; this layout is 3.0us and the all-gather consumes it directly
        # (test_embedding_decode_sweep.py). None outside wh_9b_n300.
        self.emb_decode_memcfg = tpc.create_activation_shard_config(self.dim // tp) if tpc.wh_9b_n300(self) else None

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
        # Disjoint K/V grids for the fused paged-cache write (paged_fused_update_cache requires its
        # two inputs' shard grids to be non-overlapping), collapsing 2 device programs into 1.
        self.kv_cache_write_fused_enabled = (
            tpc.wh_9b_n300(self) and 2 * _rows <= mesh_device.compute_with_storage_grid_size().y
        )
        if self.kv_cache_write_fused_enabled:
            # K -> SHIFTED half (it reshards from interleaved regardless, so the origin costs nothing)
            self.kv_cache_write_k_shard_cfg = ttnn.create_sharded_memory_config(
                shape=(tpc.TILE_SIZE, self.head_dim),
                core_grid=ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, _rows), ttnn.CoreCoord(_cols - 1, 2 * _rows - 1))}
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            # V -> NATURAL half == kv_update_shard_cfg's grid == what the head split already emits,
            # so forward_decode's equality guard skips V's reshard entirely.
            self.kv_cache_write_v_shard_cfg = ttnn.create_sharded_memory_config(
                shape=(tpc.TILE_SIZE, self.head_dim),
                core_grid=ttnn.CoreRangeSet(
                    {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_cols - 1, _rows - 1))}
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        # Permuted-head_dim full-width RoPE. ON for Wormhole 9B N300 (tpc.rope_permuted_enabled);
        # see that helper for the measurements and attention/rope_tp.py's rope_channel_perm for the
        # derivation.
        self.rope_permuted_enabled = tpc.rope_permuted_enabled(self)
        # The ONE grid the decode rotary runs on, Q and K both: the natural one-user-per-core height
        # shard, so a single cos/sin pair serves both.
        self.rope_k_shard_cfg = self.kv_update_shard_cfg
        # LOAD-BEARING: this grid becomes the grid of the q that reaches SDPA, which (see above)
        # misreads any q not rooted at (0,0).
        if self.rope_permuted_enabled:
            _rope_q_origin = self.rope_k_shard_cfg.shard_spec.grid.bounding_box().start
            assert (_rope_q_origin.x, _rope_q_origin.y) == (0, 0), (
                f"rope_k_shard_cfg must start at core (0,0) -- got {_rope_q_origin}. This grid is the "
                "grid of the q handed to paged_scaled_dot_product_attention_decode, which ignores the "
                "shard origin and reads from absolute (0,0) outward (silently, no assert). See the "
                "root-cause note above and tests/perf/test_sdpa_decode_sharded_q_origin.py."
            )

        # Attention projection weights stay bfloat8_b. bfp4 is a real -41% to -43% on these
        # DRAM-bandwidth-bound decode matmuls, but costs 1.6-2.8% teacher-forced perplexity -- an
        # order of magnitude past every other accuracy trade here (test_decode_weight_dtype_sweep.py).

    def _set_hf_params(self, checkpoint_dir):
        # trust_remote_code before base AutoConfig load.
        self.trust_remote_code_hf = True
        super()._set_hf_params(checkpoint_dir)

    def _set_params_from_dict(self, config):
        # Qwen3.5-MoE checkpoints have NO dense `intermediate_size` (every layer is
        # sparse MoE), but the base ModelArgs still requires it (or ffn_dim_multiplier)
        # to derive the dense `hidden_dim`. That hidden_dim is vestigial here — MoE
        # layers route through tt/moe, not the dense MLP memcfgs — so inject the
        # per-expert intermediate as a tile-aligned stand-in purely to satisfy the
        # base. The dense 9B/27B carry a real intermediate_size and are untouched.
        if not config.get("intermediate_size") and config.get("moe_intermediate_size"):
            config = {**config, "intermediate_size": config["moe_intermediate_size"]}
        super()._set_params_from_dict(config)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "full_attention"

    def is_deltanet_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "linear_attention"

    def is_moe_layer(self, layer_idx: int) -> bool:
        """True when this layer uses the sparse MoE MLP instead of the dense SwiGLU.

        Follows the HF Qwen3-Next / Qwen3.5-MoE rule: a layer is MoE when there
        are experts, it is not forced dense (mlp_only_layers), and it falls on
        the decoder_sparse_step cadence. On the dense 9B/27B num_experts==0 so
        this is always False and the Qwen36MLP path is byte-for-byte unchanged.
        """
        if self.moe_num_experts <= 0:
            return False
        if layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.moe_decoder_sparse_step == 0

    def is_distributed_norm(self, mode):
        """Force the distributed-norm path for multi-device MoE prefill.

        The prefill norm-all-gather fusion (all_gather_minimal_matmul_async in-proj) needs the norm
        to honor enable_all_gather and leave its output hidden-fractured for the fused matmul to
        gather. The base enables the distributed-norm path only for dim>4096 (an L1 heuristic the
        dense 27B's 5120 hits but the MoE 35B-A3B's 2048 misses) — on the miss it force-gathers the
        norm output, so the AGMM in-proj double-gathers (K mismatch). Only the MoE configs need this
        override (dense variants either hit the dim>4096 heuristic like the 27B, or are validated on
        the base path), so gate it on moe_num_experts to avoid diverging the dense path from base.
        """
        from models.tt_transformers.tt.common import Mode

        if self.moe_num_experts > 0 and self.is_multichip and mode == Mode.PREFILL:
            return True
        return super().is_distributed_norm(mode)

    @property
    def base_model_name(self):
        # get_base_model_name() strips "-A3B" (Qwen3.6-35B-A3B -> Qwen3.6-35B), so the emitted
        # benchmark name misses this checkpoint's perf targets; keep the full name for MoE.
        # getattr: read during base __init__ before moe_num_experts is set.
        if getattr(self, "moe_num_experts", 0) > 0:
            return self.model_name
        return super().base_model_name

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
            load_mtp_tensors,
            load_qwen36_state_dict_fp8,
            remap_qwen36_state_dict,
        )

        # Block FP8 checkpoints: dequant + remap for TP loaders (skip the HF model).
        # The FP8 loader already keeps mtp.* (read raw from safetensors), so no extra merge.
        if is_fp8_checkpoint(self.CKPT_DIR):
            return load_qwen36_state_dict_fp8(self.CKPT_DIR)

        # Name the HF classes directly rather than going through AutoModelForCausalLM: under vLLM,
        # vllm.transformers_utils.config registers its OWN Qwen3_5Config for model_type "qwen3_5",
        # so AutoConfig hands back vLLM's class. transformers only unwraps a composite config to its
        # text sub-config when `model_class.config_class == config.sub_configs["text_config"]`, an
        # identity check that cannot hold across libraries -- the composite config would then reach
        # Qwen3_5ForCausalLM and fail on `config.vocab_size`, which lives in text_config.
        #
        # Qwen3_5TextConfig.from_pretrained picks the `text_config` sub-dict on composite
        # (3.6 VLM) checkpoints via base_config_key, and reads a text-only (3.5) config.json
        # as-is, so both checkpoint layouts land on the config Qwen3_5ForCausalLM expects.
        # The 35B-A3B is a Qwen3.5-MoE checkpoint (model_type qwen3_5_moe): its sparse experts and
        # gated shared expert live under the MoE config that the dense Qwen3_5TextConfig silently
        # drops — that class would build a dense mlp.gate_proj and leave mlp.shared_expert/experts
        # unloaded. Pick the MoE text class for MoE configs; the dense/vision path keeps Qwen3_5.
        if self.moe_num_experts > 0:
            from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM as _HFForCausalLM
            from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as _HFTextConfig
        else:
            from transformers.models.qwen3_5 import Qwen3_5ForCausalLM as _HFForCausalLM
            from transformers.models.qwen3_5 import Qwen3_5TextConfig as _HFTextConfig

        text_config = _HFTextConfig.from_pretrained(self.CKPT_DIR)
        assert text_config.vocab_size == self.vocab_size and text_config.hidden_size == self.dim, (
            f"HF text config disagrees with model args: vocab_size {text_config.vocab_size} vs "
            f"{self.vocab_size}, hidden_size {text_config.hidden_size} vs {self.dim}"
        )
        model = _HFForCausalLM.from_pretrained(self.CKPT_DIR, config=text_config, dtype="auto")
        state_dict = remap_qwen36_state_dict(model.state_dict())
        del model
        # AutoModelForCausalLM drops mtp.* before remap; read the drafter weights directly.
        if getattr(self, "has_mtp", False):
            state_dict.update(load_mtp_tensors(self.CKPT_DIR))
        return state_dict
