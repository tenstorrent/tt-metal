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
        #
        # gdn_ab_gap: zero-column gap inserted between a and b so b starts on a tile boundary
        # (gdn_nv_tp isn't necessarily one). Splitting the fused ab tensor into a=[0:Nv)/b=[Nv:2*Nv)
        # forces an Untilize->Slice->TilizeWithValPadding round-trip in _project_qkvzab, because a
        # ttnn.slice's STARTING offset must be tile-aligned to stay tile-native (a non-tile-aligned
        # END is free -- confirmed empirically: slice[0:24] and slice[32:56] on a TILE tensor both
        # dispatch as a single SliceDeviceOperation, only slice[24:48] triggers the 3-op
        # untilize/retilize sequence). Moving b to start at the next tile boundary makes both a and b
        # single-op tile-native slices with no change to Nv/v/state/conv1d/the GQA ratio anywhere
        # else -- see tests/perf/test_gdn_ab_pad_check.py, which confirmed this padding lands inside
        # per-core tile capacity already being paid for.
        #
        # TWO INDEPENDENT GATES, ONE MECHANISM. The 9B and the 27B were measured separately, on
        # different hosts, at different TP -- so each keeps its own scope test and its own
        # provenance rather than being folded into a single "align whenever nv_tp % 32" rule that
        # would silently re-tune whichever model was not measured. They are mutually exclusive by
        # construction (dim <= 4096 vs dim > 4096), asserted below, so the two contributions sum.
        # Blackhole is excluded from both: its decode grid (num_cores=44) is tuned against the
        # unpadded width. Gate on dim, not model_name: HF_MODEL is often a hashed snapshot dir --
        # same reasoning as _qkv_l1_tuned_for_this_model in gdn/tp.py. 0 restores the pre-gap width
        # exactly, and gdn/tp.py falls back to its enclosing-tile ab slice dance when the gap is 0.
        #
        # GATE 1 -- WORMHOLE 9B (dim 4096), from the base branch.
        _ab_gap_9b = (not tpc.is_blackhole()) and self.dim <= 4096
        _gap_9b = (-(-self.gdn_nv_tp // 32) * 32 - self.gdn_nv_tp) if _ab_gap_9b else 0
        #
        # GATE 2 -- WORMHOLE 27B (dim 5120). Same defect, measured independently at T3K TP=8, where
        # gdn_nv_tp is 6 so `b` lands at offset 6 inside the tile. MEASURED (seq 2048, single-layer
        # profile) on the two identical 6-wide tensors:
        #     a  (offset 0,  aligned)    slice                              2.1us
        #     b  (offset 6, misaligned)  untilize 13.9 + slice 27.9 + tilize 5.6 = 47.4us
        # -- that row-major slice moves 24KB at ~0.9 GB/s, which is the tell. Aligning b took the
        # pair 50.5us -> 12.1us and removed 2 device ops, for one extra tile of matmul N (65 -> 66)
        # that measured free: the in-projection was 566.4us before and 565.6us after.
        _ab_gap_27b = (not tpc.is_blackhole()) and self.dim > 4096
        _gap_27b = (-(-self.gdn_nv_tp // 32) * 32 - self.gdn_nv_tp) if _ab_gap_27b else 0
        assert not (_ab_gap_9b and _ab_gap_27b), "a/b gap gates must be mutually exclusive"
        self.gdn_ab_gap = _gap_9b + _gap_27b
        self.gdn_qkvzab_dim_tp = self.gdn_qkvz_dim_tp + 2 * self.gdn_nv_tp + self.gdn_ab_gap
        # NO PAD (was: prefill_kpass_width padded 6176 -> 6912 on Wormhole). Kept at 0 so the weight
        # geometry, cache key and decode progcfgs all use the natural width; see the history below,
        # because the reason this is safe now is NOT the reason it was originally padded.
        #
        # The 2048x4096x6176 in-proj is K-PASS bound, not subblock bound: its cost tracks
        # ceil(per_core_N / out_block_w), each pass re-reading the 16MB DRAM-resident in0. 6176 = 193
        # tiles is PRIME, so at 8 columns per_core_N=25, whose only divisors are 1/5/25. The pad existed
        # because out_block_w=25 (one pass) overflowed L1, leaving 5 as the best available: padding to
        # 216 tiles bought per_core_N=27 / out_block_w=9 / 3 passes and measured -643us on the op.
        #
        # That L1 overflow was an artifact of fp32_dest_acc_en, not a property of the shape. With
        # COMPUTE_HIFI2_NO_FP32_ACC the intermediate CB halves and the output-subblock ceiling rises
        # from 4 to 8, so per_core_N=25 gets out_subblock_w=5 and out_block_w=25 — ONE K pass — at the
        # unpadded width. MEASURED (device kernel duration, N150, M=2048 K=4096; the full sweep is in
        # tests/perf/test_gdn_inproj_sweep.py):
        #     N=6912  sub_w=3 blk_w=9   fp32_acc ON   3 passes  1493us   <- the padded config
        #     N=6912  sub_w=3 blk_w=27  fp32_acc OFF  1 pass    1403us
        #     N=6176  sub_w=5 blk_w=25  fp32_acc OFF  1 pass    1255us   -16.0%, and 12% less work
        # So the pad is now strictly worse: one K pass makes the prime tile count harmless, and the
        # unpadded width does 11.9% fewer FLOPs. Reverting it also gives back the pad's accepted costs
        # — ~12.6MB/layer of weight DRAM and decode's ~11.9% extra in-proj work.
        #
        # CONFIRMED IN THE REAL LAYER (Tracy, single-layer GDN prefill, seq 2048, N300, Qwen3.5-9B):
        #     op 277  2048x4096x6912  1,514us @ 58.2% of peak FLOPs
        #          -> 2048x4096x6176  1,265us @ 62.2%   (-249us, -16.4%; layer share 9.3% -> 7.9%)
        # Every other matmul in the layer is byte-for-byte unchanged (o_proj 536->538us, MLP
        # 1328/1181/1105 -> 1332/1182/1104), which is the check that the scoping held. The two
        # ReduceScatter rows moved -164us/-17us; those are CCL capture variance, not this change.
        #
        # The trade is accuracy, not geometry: PCC against an fp32 reference goes 0.99997 -> 0.99992.
        # test_gdn_tp (11 tests, decode + prefill, N300) passes with min PCC 0.99959.
        #
        # gdn_qkvzab_pad_tiles is kept (as 0) rather than deleted because load_gdn_weights_tp reads it
        # to qualify the weight cache key with `.pad{N}` — 0 selects the unpadded cache file, so a
        # previously cached padded weight cannot be silently reloaded at the wrong width. To restore
        # the pad, set it from prefill_kpass_width here AND drop gdn_qkvzab_prefill_progcfg below.
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
        # 43.9us for the old 8x4=forced1d_32c). On WH (decode_grid_w=8), re-swept at the EXACT
        # production shape (M=32 K=4096 N=6144, test_mlp_decode_matmul_sweep.py, 3 independent runs
        # each): num_cores=56 (not 64 -- the prior "N~5504 representative shape" note predates this
        # exact width) paired with fp32_dest_acc_en=False beats today's cores=64/fp32_acc=True by
        # ~3-4% on BOTH gate and up, reproducibly. fp32_acc=False is passed here too so this
        # progcfg's own subblock choice is consistent with the runtime compute config that's paired
        # with it (mlp.py's compute_kernel_config_gateup_decode) -- at this N/core count it doesn't
        # change the chosen blocking (per_core_N=4 has no divisor between 4 and 8), but keeping the
        # two in sync avoids a silent mismatch if either changes later. PCC cost is small and
        # consistent with LoFi+bfp4 already dominating this matmul's error budget (gate 0.99032 ->
        # 0.98959, up 0.99320 -> 0.99268 -- 4th-decimal, same class as other accepted trades here).
        #
        # SCOPED TO WORMHOLE 9B ON N300 (tpc.wh_9b_n300 -- see that helper for why each of the three
        # conditions is load-bearing). The 56/fp32F pair was swept at the 9B's hidden_dim/tp = 6144;
        # the 27B's is 2176, so neither the core count nor the subblock choice transfers. Outside the
        # scope this restores the previously shipped values exactly: 44 cores on Blackhole, 64 on
        # other Wormhole meshes, fp32_dest_acc_en left on.
        # NOTE: fp32_acc here MUST stay in lockstep with mlp.py's compute_kernel_config_gateup_decode,
        # which is gated on the same helper -- a mismatch silently pairs a subblock cap of 8 with an
        # fp32-acc kernel (or vice versa).
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
        # down: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~63us, +28% vs
        # the old 8x2). On WH (decode_grid_w=8) this falls back to 8x5; MEASURED full 8x8 grid is
        # -1.4% vs 33, smaller than gate/up but consistently faster, same PCC.
        self.mlp_w2_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.hidden_dim // tp, self.dim, num_cores=33 if tpc.is_blackhole() else 64, grid_w=self.decode_grid_w
        )

        # Input-projection 1D decode (DEFAULT): same idea for attn QKV+gate and GDN QKVZAB in-projections.
        # Weights load interleaved (prefill AGMM verified bit-identical); tuned grids per test_mlp_matmul_sweep.
        self.proj_1d_decode = True
        self.attn_qkv_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.attn_qkv_fused_dim_tp, num_cores=64
        )
        # gdn_qkvz: num_cores=44 -> 11x4 on BH, the fastest measured config (wide1d_11x4c, ~59us, +22%
        # vs the old 8x5). On WH (decode_grid_w=8), 44 was never independently tuned -- it only ever
        # fell back to whatever min(grid_w, 44) -> 8x6=48 cores happened to produce. MEASURED (WH,
        # M=32 K=4096 N=6176, same fp32_dest_acc_en=True/PCC as today): the full 8x8=64-core grid is
        # 150.4us vs 8x6's 156.5us, -3.9%, zero accuracy cost -- matches attn_qkv_decode_1d_progcfg
        # above, which already uses num_cores=64 for the same reason.
        self.gdn_qkvz_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.dim, self.gdn_qkvzab_dim_tp, num_cores=44 if tpc.is_blackhole() else 64, grid_w=self.decode_grid_w
        )
        # Output projections (attn wo, GDN o_proj): already interleaved+auto (no weight relayout, not in
        # the prefill AGMM fusion), so this just swaps ttnn-auto for a tuned ~32-core 1D decode grid.
        # attn_wo: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4). On WH (decode_grid_w=8) 33 falls back to 8x5, again never independently
        # tuned there. MEASURED (WH, M=32 K=2048 N=4096): 8x5 is 51.8us vs 8x6 (num_cores>=40) at
        # 51.0us -- only ~1.5%, much smaller than gdn_qkvz's -3.9% and close to noise, but
        # consistently the faster side across 48/56/64 cores, zero accuracy cost either way.
        self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.attn_out_dim_tp, self.dim, num_cores=33 if tpc.is_blackhole() else 48, grid_w=self.decode_grid_w
        )
        # gdn_out: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4; same 1536x5120 shape as attn_wo). On WH (decode_grid_w=8) this falls back to 8x5.
        self.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.gdn_value_dim_tp, self.dim, num_cores=33, grid_w=self.decode_grid_w
        )

        # Prefill matmul factory (M = seq_len). Shared by MLP down-proj, attention in/out-proj, and
        # GDN in/out-proj. On Blackhole the grid is already wider (8x10) with more L1 headroom, so the
        # full per_core_N-wide output/intermediate CB fits (validated there); N300's grid tops out at
        # 8x8 with less L1/core, and a full decoder layer's combined GDN+attention+MLP CBs on this
        # config alone were measured to overflow (1.68-1.85MB vs N300's 1.5MB max) — halve it there
        # (see create_prefill_matmul_program_config's halve_out_block).
        #
        # DELIBERATE narrowing (Wormhole gating audit, item 10): this used to be
        # `not tpc.is_blackhole()`, so T3K halved its output block too even though the 1.68-1.85MB
        # vs. 1.5MB overflow above was measured on N300's 8x8 grid specifically, not T3K's. Narrowed
        # to wh_9b_n300 on purpose -- T3K now gets the un-halved (Blackhole-style) block, accepting
        # that tradeoff without T3K L1-budget measurements of its own. Don't revert this to
        # is_blackhole() without measuring T3K's actual grid/L1 headroom first.
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
        # WORMHOLE ONLY: dedicated one-K-pass factory for the GDN in-projection, paired with
        # COMPUTE_HIFI2_NO_FP32_ACC (gdn/tp.py's _col_proj passes both together — the blocking is only
        # legal with fp32 dest accumulation off). Worth -16.0% on that matmul; see the
        # gdn_qkvzab_pad_tiles note above for the measurements and the accuracy trade.
        #
        # Scoped to this ONE matmul on purpose. prefill_progcfg above is shared by the MLP down-proj and
        # both attention projections, which were tuned with fp32 dest accumulation ON and are NOT
        # covered by tests/perf/test_gdn_inproj_sweep.py. Widening this needs its own sweep per shape:
        # one K pass means the full per_core_N-wide CB, and WH's L1 does not have room for every
        # projection in a layer to do that (which is exactly why halve_out_block exists above).
        #
        # None on Blackhole: BH prefill takes the fused all_gather_matmul_prefill path, which pins its
        # own grid and progcfg, so this would never be consulted there.
        self.gdn_qkvzab_prefill_progcfg = (
            None
            if tpc.is_blackhole()
            else (
                lambda seq_len, k, n: tpc.create_prefill_kpass1_matmul_program_config(
                    seq_len, k, n, grid_size=self._prefill_grid
                )
            )
        )
        # WORMHOLE ONLY: same one-K-pass fix, for the fused attention QKV(+gate) in-projection.
        # attn_qkv_fused_dim_tp=5120 (160 tiles) -> per_core_N=20 at 8 cols. With fp32_dest_acc_en=True
        # (today) the subblock cap is 4, giving out_subblock_w=4 / out_block_w=4 -- FIVE K passes.
        # Dropping fp32 dest acc raises the cap to 8, so out_subblock_w=5 (20%5==0) and
        # out_block_w=per_core_N=20 divides evenly -- ONE K pass.
        #
        # MEASURED (device kernel duration, N300, M=2048 K=4096 N=5120;
        # tests/perf/test_attn_qkv_inproj_sweep.py):
        #     baseline fp32T/pkT in0=DRAM   1683.7us   <- today (matches the 1,706us layer profile)
        #     kpass1   fp32F/pkT in0=DRAM   1011.5us   -39.9%   pcc=0.99992 (vs 0.99997 baseline)
        # in0=L1 measured within noise of in0=DRAM at this blocking (one K pass already reads in0
        # once), so no companion input-placement change is needed — unlike the GDN in-proj, this one
        # needs no norm-side changes at all.
        #
        # None on Blackhole: BH prefill takes the fused all_gather_matmul_prefill path (_fuse_agmm),
        # which pins its own grid and progcfg, so this would never be consulted there.
        self.attn_qkv_fused_prefill_progcfg = (
            None
            if tpc.is_blackhole()
            else (
                lambda seq_len, k, n: tpc.create_prefill_kpass1_matmul_program_config(
                    seq_len, k, n, grid_size=self._prefill_grid
                )
            )
        )
        # WORMHOLE ONLY: same one-K-pass fix, for the attention wo (output) projection.
        # K=attn_out_dim_tp=2048, N=dim=4096 (128 tiles) -> per_core_N=16 at 8 cols: fp32 dest acc on
        # caps out_subblock_w at 4 (blk_w=8, TWO K passes); off, cap rises to 8 (blk_w=16, ONE pass).
        #
        # MEASURED (device kernel duration, N300, M=2048 K=2048 N=4096;
        # tests/perf/test_attn_qkv_inproj_sweep.py):
        #     baseline fp32T/pkT in0=DRAM   558.6us   <- today (matches the 531-536us layer profile)
        #     kpass1   fp32F/pkT in0=DRAM   517.2us   -7.4%   pcc=0.99993 (vs 0.99997 baseline)
        # Smaller win than the QKV in-proj (2->1 pass here vs 5->1 there), but same fix, same low risk.
        #
        # None on Blackhole: this progcfg is only consulted by attention/tp.py's _wo_proj when set.
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
        # Decode token embedding: width-sharded L1 on dim_tp, 32 cores (8x4). Interleaved embedding
        # is 1 core / ~21us at B=32 (token-tile split); this layout is 3.0us and the pre-norm
        # all-gather consumes it directly (test_embedding_decode_sweep.py). None outside wh_9b_n300.
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
        # Disjoint K/V grids for the fused paged-cache write (ttnn.experimental.
        # paged_fused_update_cache requires its two inputs' shard grids to be non-overlapping). Two
        # halves of the device grid: the NATURAL half (rows 0.._rows-1, which is exactly
        # kv_update_shard_cfg's grid and exactly what nlp_create_qkv_heads_decode emits) and the
        # SHIFTED half (the next _rows rows). Using both doubles the cores for this one op (together =
        # the full device grid at B=32) to collapse 2 device programs into 1.
        #
        # WHICH TENSOR GETS WHICH HALF IS NOT ARBITRARY -- V TAKES THE NATURAL ONE:
        #   * K arrives INTERLEAVED at the write (it went through q/k-norm + RoPE), so it owes an
        #     InterleavedToSharded no matter which half it targets -- the grid origin is FREE for K.
        #   * V arrives ALREADY SHARDED on the natural half straight from the head split, so pointing
        #     V at the natural half means it needs NO reshard at all (forward_decode's equality guard
        #     then short-circuits), while pointing it at the shifted half forces a Reshard op.
        # This was originally assigned the other way round, which cost one extra device op per layer.
        # MEASURED (test_attn_head_split_v_reshard_sweep.py::test_kv_write_grid_swap_removes_v_reshard,
        # N300, B=32): 26.7us/4 programs -> 26.0us/3 programs (-3.0%), K and V cache contents
        # bit-identical either way and no NaN.
        #
        # DO NOT instead try to make nlp_create_qkv_heads_decode emit onto the shifted half by passing
        # that grid as its memory_config: it SILENTLY RETURNS NaN. The tensor comes back with the
        # right shape and the right shard_spec (so a structural check passes) but that kernel writes
        # its shards from absolute core (0,0) outward, leaving a non-(0,0)-origin grid unwritten.
        # MEASURED: natural grid -> max|diff vs torch| 0.000000; grid at origin (0,1) -> NaN.
        #
        # WH-only (gated with is_blackhole below): unverified there from this host, and Blackhole's
        # paged decode takes a different pad-first branch anyway (_WH_KV_PAD_NOTE in attention/tp.py).
        # VERIFIED (test_kv_cache_sdpa_decode_sweep.py, N300, B=32): bit-identical cache contents vs
        # the two separate paged_update_cache calls (checked against a per-user-distinct page table,
        # not a degenerate shared one), device kernel duration 22.4us -> 18.1us (-19.1%).
        # SCOPED TO WORMHOLE 9B ON N300 (tpc.wh_9b_n300), plus the geometric precondition that the
        # worker grid actually has room for both disjoint halves. Outside the scope, forward_decode
        # falls back to the two separate paged_update_cache calls exactly as before.
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
        # The ONE grid the decode-mode rotary runs on, for Q and K both: the NATURAL
        # one-user-per-core height shard. Sharing it between Q and K is what lets a single cos/sin
        # pair serve both (the sharded rotary lays its kernels on the input's grid, so cos/sin must
        # live on the same cores as the tensor being rotated).
        #
        # TRIED AND REJECTED: pointing this at kv_cache_write_k_shard_cfg (the SHIFTED half) so K's
        # rotary output would land in the fused KV write's layout directly and save K's reshard --
        # the rotary copies its output shard spec from its input, so it structurally works, and
        # test_attention_tp.py passes on it at PCC 1.00000. It is still WRONG: on
        # test_model_tp.py::test_model_tp_decode_batched (B=8, the batched-vs-B1-oracle contract)
        # the worst per-user PCC at the first decode step went 0.9436 (baseline) -> 0.870 with the
        # shifted grid, while this natural grid measures 0.9456, i.e. baseline to within noise.
        #
        # ROOT CAUSE (2026-08-20 -- the "KV-cache write corruption" reading above is WRONG; K was
        # never the problem). The real mechanism, bisected and then reproduced standalone:
        #
        #   ttnn's paged_scaled_dot_product_attention_decode IGNORES THE SHARD-GRID ORIGIN of a
        #   height-sharded q. It enumerates batch->core as {i % grid_x, i / grid_x} from absolute
        #   core (0,0) (sdpa_decode_program_factory.cpp ~183 and ~301-315 -- the body of the
        #   origin-discarding grid_to_cores(num_cores,x,y,row_wise) overload, core_coord.cpp
        #   ~498-517, inlined), and q's shard grid is only ever queried for num_cores()/shape, never
        #   bounding_box().start_coord. There is no assert: the one grid-related check is commented
        #   out (sdpa_decode_device_operation.cpp ~361). So q on rows 1.. is read from rows 0..
        #
        # Since this grid is shared by Q AND K, aiming it at the shifted half does not just relocate
        # K -- it drags Q onto row 1, and Q goes from the rotary straight into SDPA still sharded
        # (attention/tp.py:856 -> :945, nothing in between). That, not the cache write, is the bug.
        #
        # EVIDENCE (three files, each eliminating one candidate):
        #   * test_rope_shifted_grid_bisect_qk.py -- real TPAttention, B=8, worst per-user PCC vs a
        #     B=1 oracle: natural Q+K 1.000000; shifted Q+K badly broken (0.03-0.14, and WHICH user is
        #     worst moves -- the value depends on what was last freed at the L1 address SDPA wrongly
        #     reads, so do not treat any single figure as canonical); shifted Q+K with **Q alone**
        #     moved back to the natural grid before SDPA 1.000000. It also reads back Q and K as they
        #     leave the rotary: both BIT-EXACT (max|diff| = 0) against the natural run in the failing
        #     variant -- so the rotary is innocent and SDPA is handed correct data.
        #   * test_sdpa_decode_sharded_q_origin.py -- standalone, no model: natural-grid q PCC
        #     1.000000, shifted-grid q PCC 0.0855-0.1093, across all 8 combinations of {bf16, bfp8}
        #     cache x {uniform, per-user} positions x call order. Not dtype- or position-dependent.
        # Two candidates were checked and CLEARED along the way, so nobody re-treads them: the rotary
        # itself (its output is bit-exact on either grid, per the capture above) and program-cache
        # state (results are bit-identical across warm/cold/warm caches). An early "which decode step
        # breaks flips between runs" observation was an ARTEFACT of the same stale-address mechanism:
        # freeing a natural-grid q leaves a correct copy at the address SDPA wrongly reads, so the
        # failure can masquerade as passing depending on allocation order -- which is also why a first
        # standalone probe wrongly exonerated SDPA. See that file's docstring for how to avoid the trap.
        #
        # WHY THE RESHARD STAYS ANYWAY. paged_fused_update_cache needs K and V on DISJOINT grids, and
        # both K (from the rotary) and V (from the head split) naturally arrive on the natural grid --
        # so exactly one of them must always move. The shifted rope grid is the only arrangement that
        # makes BOTH free, and it needs Q's placement fixed, for which the options are:
        #   (i)  move Q back to the natural grid after the rotary -- costs the op it saves (net zero;
        #        this is exactly the measured-clean `shifted_k_only` variant above);
        #   (ii) pass the q grid as SDPAProgramConfig.sub_core_grids (the one origin-aware path) --
        #        correct, but it confines SDPA to the B cores q lives on, and SDPA is ~76us on the
        #        full grid, so this loses far more than the ~0.6-1us Reshard it saves;
        #   (iii) fix the ttnn kernel -- a C++ change, out of scope here.
        # Dropping the fused write instead (two paged_update_cache calls, both on the natural grid, no
        # reshard at all) is the same program count and measured slower overall (18.3us fused + ~1us
        # reshard vs 20.3us unfused). So the shipped arrangement is a local optimum: KEEP IT.
        #
        # The guard below is what makes that safe rather than lucky.
        self.rope_k_shard_cfg = self.kv_update_shard_cfg
        # LOAD-BEARING, NOT COSMETIC: the decode rotary's grid becomes the grid of the q that reaches
        # SDPA, and per the root-cause note above SDPA silently misreads a q whose grid does not start
        # at (0,0). kv_update_shard_cfg is origin-anchored today (built from a ttnn.CoreGrid, i.e.
        # rooted at (0,0)), so this holds -- but it is derived from max_batch_size, and nothing else in
        # the type system ties the two together. Fail loudly rather than serve quietly-wrong tokens.
        #
        # SCOPED TO THE PERMUTED-ROPE PATH, deliberately: rope_k_shard_cfg is only ever READ when
        # rope_permuted_enabled is on (attention/tp.py::_rope_decode returns via
        # apply_partial_rope_decode before touching it otherwise), and that flag is itself gated to
        # wh_9b_n300 + rope_head_dim < head_dim (tp_common.rope_permuted_enabled). _init_tp_config,
        # by contrast, runs for EVERY TP config (27B on T3K/P150x4 included), so asserting
        # unconditionally here would police a precondition those configs never rely on -- and could
        # fail a model that does not even take this code path. Same scoping rule as every other decode
        # change in this file.
        if self.rope_permuted_enabled:
            _rope_q_origin = self.rope_k_shard_cfg.shard_spec.grid.bounding_box().start
            assert (_rope_q_origin.x, _rope_q_origin.y) == (0, 0), (
                f"rope_k_shard_cfg must start at core (0,0) -- got {_rope_q_origin}. This grid is the "
                "grid of the q handed to paged_scaled_dot_product_attention_decode, which ignores the "
                "shard origin and reads from absolute (0,0) outward (silently, no assert). See the "
                "root-cause note above and tests/perf/test_sdpa_decode_sharded_q_origin.py."
            )

        # Attention projection weights (QKV in-proj + wo out-proj) stay bfloat8_b. bfp4 is faster
        # there -- these decode matmuls are DRAM-bandwidth-bound, so a narrower weight dtype is a real
        # -41% to -43% -- but the end-to-end quality cost is disproportionate: it moves the model's
        # teacher-forced perplexity by 1.6-2.8%, an order of magnitude past every accuracy trade this
        # codebase has otherwise accepted. Not worth it (tests/perf/test_decode_weight_dtype_sweep.py
        # has the full measurement, both speed and quality).

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
