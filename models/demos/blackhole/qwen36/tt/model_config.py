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
        # Per-device width of the [qkv|z|a|b] fused in-projection. Folding the tiny a/b (decay/beta)
        # projection into qkvz removes a whole decode matmul while keeping K=dim.
        #
        # gdn_ab_gap zero-pads between a and b so b starts on a tile boundary. A ttnn.slice's START
        # offset must be tile-aligned to stay tile-native (a ragged END is free), so an unaligned b
        # costs an Untilize->Slice->Tilize round-trip in _project_qkvzab. Measured at T3K TP=8
        # (gdn_nv_tp=6, so b sits at offset 6): aligned slice 2.1us vs 47.4us unaligned, and the
        # a/b pair 50.5 -> 12.1us for one extra tile of matmul N that measured free. Nothing else
        # (Nv, state, conv1d, GQA ratio) changes; gap 0 restores the pre-gap width, which gdn/tp.py
        # handles with its enclosing-tile slice dance.
        #
        # Two gates, one mechanism: the 9B and 27B were measured separately, on different hosts at
        # different TP, so each keeps its own provenance rather than a shared "align whenever
        # nv_tp % 32" rule that would silently re-tune the unmeasured one. Mutually exclusive by
        # construction, asserted below. Blackhole is excluded: its decode grid (num_cores=44) is
        # tuned against the unpadded width. Gate on dim, not model_name -- HF_MODEL is often a
        # hashed snapshot dir (same reasoning as _qkv_l1_tuned_for_this_model in gdn/tp.py).
        _ab_gap_9b = (not tpc.is_blackhole()) and self.dim <= 4096
        _gap_9b = (-(-self.gdn_nv_tp // 32) * 32 - self.gdn_nv_tp) if _ab_gap_9b else 0
        _ab_gap_27b = (not tpc.is_blackhole()) and self.dim > 4096
        _gap_27b = (-(-self.gdn_nv_tp // 32) * 32 - self.gdn_nv_tp) if _ab_gap_27b else 0
        assert not (_ab_gap_9b and _ab_gap_27b), "a/b gap gates must be mutually exclusive"
        self.gdn_ab_gap = _gap_9b + _gap_27b
        self.gdn_qkvzab_dim_tp = self.gdn_qkvz_dim_tp + 2 * self.gdn_nv_tp + self.gdn_ab_gap
        # No pad: geometry, cache key and decode progcfgs all use the natural width.
        #
        # The 2048x4096x6176 in-proj is K-PASS bound, not subblock bound -- cost tracks
        # ceil(per_core_N / out_block_w), each pass re-reading the 16MB DRAM-resident in0. Padding
        # to 216 tiles once bought a better blocking, but only because fp32_dest_acc_en capped the
        # output subblock at 4. With COMPUTE_HIFI2_NO_FP32_ACC the cap rises to 8 and the natural
        # width reaches ONE K pass, which beats the padded config outright. MEASURED (N150, M=2048
        # K=4096; full sweep in tests/perf/test_gdn_inproj_sweep.py):
        #     N=6912  blk_w=9   fp32_acc ON   3 passes  1493us
        #     N=6176  blk_w=25  fp32_acc OFF  1 pass    1255us   -16.0%, and 12% less work
        # Confirmed in-layer (Tracy, seq 2048, N300): 1,514 -> 1,265us, every other matmul in the
        # layer unchanged. The trade is accuracy: PCC vs fp32 0.99997 -> 0.99992 (test_gdn_tp min
        # 0.99959).
        #
        # Kept as 0 rather than deleted: load_gdn_weights_tp qualifies the weight cache key with
        # `.pad{N}`, so a previously cached padded weight cannot be reloaded at the wrong width. To
        # restore the pad, set it from prefill_kpass_width AND drop gdn_qkvzab_prefill_progcfg.
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
        # decode_grid_w is the device worker-grid width (11 on BH P150, 8 on WH); shaping the 1D-mcast
        # grid wide-first shortens the in0 multicast column, worth ~2% (test_mlp_matmul_sweep).
        self.decode_grid_w = mesh_device.compute_with_storage_grid_size().x
        self.mlp_1d_decode = True
        # gate/up: 44 cores (11x4) on BH, the fastest measured (42.8us vs 43.9 for 8x4). On WH,
        # swept at the exact production shape (M=32 K=4096 N=6144, test_mlp_decode_matmul_sweep.py):
        # 56 cores + fp32_dest_acc_en=False beats 64 cores + fp32_acc=True by ~3-4% on both, at a
        # 4th-decimal PCC cost (gate 0.99032 -> 0.98959) that LoFi+bfp4 already dominates.
        #
        # Scoped to Wormhole 9B on N300: the 56/fp32F pair was swept at the 9B's hidden_dim/tp=6144;
        # the 27B's is 2176, so neither the core count nor the subblock choice transfers.
        # fp32_acc MUST stay in lockstep with mlp.py's compute_kernel_config_gateup_decode on the
        # same gate -- a mismatch pairs a subblock cap of 8 with an fp32-acc kernel.
        _gateup_9b = tpc.wh_9b_n300(self)
        _gateup_cores = 56 if _gateup_9b else (44 if tpc.is_blackhole() else 64)
        self.mlp_w1_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=_gateup_cores,
            fused_activation=ttnn.UnaryOpType.SILU,
            grid_w=self.decode_grid_w,
            fp32_acc=not _gateup_9b,
        )
        self.mlp_w3_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=_gateup_cores,
            grid_w=self.decode_grid_w,
            fp32_acc=not _gateup_9b,
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
        # attn_wo: 33 cores (11x3) on BH, fastest measured (~24us). On WH, 8x6 (>=40 cores) is 51.0us
        # vs 8x5's 51.8 -- ~1.5%, near noise, but consistently faster across 48/56/64 and PCC-neutral.
        self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.attn_out_dim_tp, self.dim, num_cores=33 if tpc.is_blackhole() else 48, grid_w=self.decode_grid_w
        )
        # gdn_out: 33 cores (11x3) on BH, fastest measured (~24us; same 1536x5120 shape as attn_wo).
        self.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.gdn_value_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        # Prefill matmul factory (M = seq_len), shared by the MLP down-proj and the attention/GDN
        # in/out-projections. Blackhole's wider 8x10 grid fits the full per_core_N-wide CB; N300's
        # 8x8 does not -- a full layer's combined GDN+attention+MLP CBs measured 1.68-1.85MB against
        # N300's 1.5MB max, so halve the output block there (create_prefill_matmul_program_config).
        # The gate is wh_9b_n300, not is_blackhole: that overflow was measured on N300's grid, so
        # T3K takes the un-halved block. Don't widen it back without measuring T3K's L1 headroom.
        self._prefill_grid = tpc.prefill_grid_default()
        self.prefill_tuning = tpc.prefill_tuning(tp)
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
        # with fp32 dest accumulation off). Worth -16.0%; measurements in the pad_tiles note above.
        # Scoped to this one matmul: one K pass needs the full per_core_N-wide CB and WH's L1 has no
        # room for every projection in a layer to do that (hence halve_out_block). None on Blackhole,
        # whose prefill takes the fused all_gather_matmul path with its own grid and progcfg.
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
        # K passes), dropping it raises the cap to 8 so out_block_w=20 divides evenly (one pass).
        # MEASURED (N300, M=2048 K=4096 N=5120; tests/perf/test_attn_qkv_inproj_sweep.py):
        # 1683.7 -> 1011.5us, -39.9%, pcc 0.99997 -> 0.99992. in0=L1 is within noise of DRAM here
        # (one pass already reads in0 once), so this needs no norm-side change. None on Blackhole.
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
        # (N300, M=2048 K=2048 N=4096): 558.6 -> 517.2us, -7.4%, pcc 0.99997 -> 0.99993. None on
        # Blackhole -- only attention/tp.py's _wo_proj reads this, and then only when set.
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
        # two inputs' shard grids to be non-overlapping), collapsing 2 device programs into 1. The
        # NATURAL half is rows 0.._rows-1 -- kv_update_shard_cfg's grid, and what
        # nlp_create_qkv_heads_decode emits; the SHIFTED half is the next _rows rows.
        #
        # WHICH TENSOR GETS WHICH HALF IS NOT ARBITRARY -- V TAKES THE NATURAL ONE. K arrives
        # interleaved (via q/k-norm + RoPE) so it owes an InterleavedToSharded either way and the
        # origin is free for it; V arrives already sharded on the natural half, so leaving it there
        # skips its reshard entirely (forward_decode's equality guard short-circuits). Measured
        # 26.7us/4 programs -> 26.0us/3 (-3.0%), cache contents bit-identical.
        #
        # DO NOT instead make nlp_create_qkv_heads_decode emit onto the shifted half via its
        # memory_config: it SILENTLY RETURNS NaN. The tensor comes back with the right shape and
        # shard_spec, but that kernel writes shards from absolute core (0,0) outward and leaves a
        # non-(0,0)-origin grid unwritten. Measured: natural max|diff| 0.000000, origin (0,1) NaN.
        #
        # Scoped to wh_9b_n300 plus the geometric precondition that the grid has room for both
        # halves; outside it, forward_decode falls back to two separate paged_update_cache calls.
        # Verified bit-identical against those (test_kv_cache_sdpa_decode_sweep.py, N300, B=32,
        # per-user-distinct page table), 22.4us -> 18.1us (-19.1%).
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
        #
        # TRIED AND REJECTED: pointing this at the SHIFTED half so K's rotary output lands in the
        # fused KV write's layout and saves K's reshard. It structurally works, but batched decode's
        # worst per-user PCC drops 0.9456 -> 0.870.
        #
        # ROOT CAUSE: paged_scaled_dot_product_attention_decode IGNORES THE SHARD-GRID ORIGIN of a
        # height-sharded q -- it enumerates batch->core from absolute (0,0) via the origin-discarding
        # grid_to_cores overload, never reading bounding_box().start_coord, and its one grid check is
        # commented out. q on rows 1.. is therefore read from rows 0.. Because Q and K share this
        # grid, aiming it at the shifted half drags Q there too (Q goes from the rotary into SDPA
        # still sharded). Standalone repro: natural q 1.000000, shifted 0.0855-0.1093 across all 8
        # combinations of cache dtype x position layout x call order
        # (test_sdpa_decode_sharded_q_origin.py). The rotary itself is bit-exact and is not at fault.
        #
        # WHY THE RESHARD STAYS. The fused write needs K and V on disjoint grids and both arrive on
        # the natural one, so exactly one must move. Fixing Q's placement instead costs more than it
        # saves: moving Q back after the rotary costs the op it saves, and SDPAProgramConfig
        # .sub_core_grids (the one origin-aware path) confines SDPA to B cores, ~76us against the
        # ~1us Reshard. Dropping the fused write is the same program count and slower (18.3us + ~1us
        # reshard vs 20.3us). This is a local optimum -- keep it; the guard below makes it safe.
        self.rope_k_shard_cfg = self.kv_update_shard_cfg
        # LOAD-BEARING: this grid becomes the grid of the q that reaches SDPA, which (see above)
        # misreads any q not rooted at (0,0). kv_update_shard_cfg is origin-anchored today, but it
        # derives from max_batch_size and nothing in the type system ties the two together -- so
        # fail loudly rather than serve quietly-wrong tokens.
        #
        # Scoped to the permuted-RoPE path: rope_k_shard_cfg is only read when rope_permuted_enabled
        # is on, while _init_tp_config runs for every TP config, so asserting unconditionally would
        # police a precondition the other configs never rely on.
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

        # Name the HF classes directly rather than going through AutoModelForCausalLM: under vLLM,
        # vllm.transformers_utils.config registers its OWN Qwen3_5Config for model_type "qwen3_5",
        # so AutoConfig hands back vLLM's class. transformers only unwraps a composite config to its
        # text sub-config when `model_class.config_class == config.sub_configs["text_config"]`, an
        # identity check that cannot hold across libraries -- the composite config would then reach
        # Qwen3_5ForCausalLM and fail on `config.vocab_size`, which lives in text_config.
        #
        # Qwen3_5TextConfig.from_pretrained picks the `text_config` sub-dict on composite (3.6 VLM)
        # checkpoints and reads a text-only (3.5) config.json as-is, so both layouts work.
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
