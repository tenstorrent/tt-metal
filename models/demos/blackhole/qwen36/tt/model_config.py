# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Config for the Qwen3.5 / 3.6 family on Blackhole (9B single-device, 27B / 35B-A3B TP).

Subclasses tt_transformers.ModelArgs. HF_MODEL env var is canonical (hub id or local dir);
hub ids are snapshot_download'd first (AutoConfig on bare hub id is unreliable here).
Qwen3.5-specific params (GDN, partial RoPE, layer types) come from HF text config.
load_state_dict/weight_cache_path override the base meta-key (wq/wk/wv) scheme.
"""

import os
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs

# Qwen3.6-27B Blackhole (P300X2/P150x4 TP) serving optimizations are ON BY DEFAULT. These are the
# validated "optimized" config from BENCHMARKS.md (TTFT 1.4-4.2x, TPOT 1.3-2.2x vs the reference
# build); each is set via setdefault so any flag can still be overridden from the environment (set a
# flag to its old value to disable it). Runs at import of this module, before any layer __init__
# reads a flag. QWEN_SDPA_BF8 is the one precision change (bf16->bf8 KV; decode PCC 0.9999,
# 64k retrieval matches bf16) and is included by request; export QWEN_SDPA_BF8=0 for bf16 KV.
_QWEN36_SERVING_OPT_DEFAULTS = {
    "QWEN36_GDN_OUT_MODE": "agmm",
    "QWEN36_GDN_CONV": "kda",
    "QWEN36_AGMM_LAYOUT": "nt11x8",
    "QWEN36_SDPA_K_CHUNK": "256",
    "QWEN36_GDN_DECODE_FUSED": "2",
    "QWEN36_GDN_SLOT_DEVICE_COPY": "2",
    "TT_SDPA_GQA_MCAST": "1",
    "TT_GDN_SCAN_MCAST": "1",
    "TT_SDPA_GQA_MCAST_QPAIR": "1",
    "QWEN36_GDN_PROJ_CHUNKS": "1",
    "QWEN36_GDN_GB_BF16": "1",
    "QWEN36_KDA_TILE_IN": "1",
    "QWEN36_AGMM_BARRIER": "1",
    "QWEN36_PREFILL_LOGITS_FAST": "1",
    "QWEN36_PREFILL_BUCKET_TRACE": "1",
    "QWEN_SDPA_BF8": "1",
    # Speculative-decode drafter (demo/spec path): "dflash2" = the DFlash block-diffusion drafter
    # (tt/dflash2_decode.py) on the same verify/accept substrate; "mtp" = the native MTP head.
    # Measured 2026-09-12 (ISL 130, lossless): DFlash v1 K=11 107.8 tok/s vs native MTP 80.9.
    "QWEN36_DRAFTER": "dflash2",
    # The drafter checkpoint MUST match the served target (a local dir or an HF repo id resolved through
    # the HF cache): z-lab/Qwen3.6-27B-DFlash for Qwen3.6-27B; incoai/Qwen3.8-27B-DFlash2 for Qwen3.8-27B
    # (the *DFlash2* drafters target 3.8, a different checkpoint).
    "DFLASH_WEIGHTS": "z-lab/Qwen3.6-27B-DFlash",
    # Draft block: 12 -> K=11 -> T=12 fused verify SDPA (the measured sweet spot for the block-16 v1
    # drafter); clamped to the checkpoint's own block_size (8 for DFlash2 -> K=7).
    "QWEN36_DFLASH_BLOCK": "12",
}
for _k, _v in _QWEN36_SERVING_OPT_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# l1_small_size the GDN prefill depthwise ttnn.conv1d requires.
GDN_CONV1D_L1_SMALL_SIZE = 24576

# The 27B class (hidden 5120 / 64 layers) must run the TP code path even on ONE die: the single-device
# 9B port (Qwen36GatedAttention / Qwen36GatedDeltaNet) hangs on 27B in ttnn_gated_deltanet. The 9B
# (hidden 4096 / 32 layers) keeps its validated single-device path.
_TP1_AUTO_MIN_DIM = 5120
_TP1_AUTO_MIN_LAYERS = 64


def mtp_requested_by_env():
    """Should the MTP drafter head (weights + its own paged KV cache) be built? OFF unless a speculative
    mode asks for it: QWEN36_MTP=1 (or any value other than 0/empty) explicitly; with QWEN36_MTP unset,
    QWEN36_DRAFTER=mtp. Plain serving never loads it (merge review 2026-09-21, major 1)."""
    env = os.environ.get("QWEN36_MTP")
    if env is not None and env != "":
        return env != "0"
    return os.environ.get("QWEN36_DRAFTER", "").lower() == "mtp"


def tp_path_forced_for_single_device(dim, n_layers):
    """Should a (1,1) mesh take the tensor-parallel code path (TP=1 mode)?

    QWEN36_FORCE_TP_PATH=1 forces it on, =0 forces it off (the single-device 9B port); unset -> ON for
    the 27B class (dim >= 5120 or n_layers >= 64), OFF otherwise."""
    env = os.environ.get("QWEN36_FORCE_TP_PATH", os.environ.get("QWEN36_TP1"))
    if env is not None and env != "":
        return env == "1"
    return dim >= _TP1_AUTO_MIN_DIM or n_layers >= _TP1_AUTO_MIN_LAYERS


def tp_path_forced_for_single_device_from_hf(hf_model=None):
    """Same decision from the HF config.json of `hf_model` (default $HF_MODEL) -- for code that must know
    the mode BEFORE the mesh/model exist (e.g. the demo's device_params). Falls back to the env-only
    rule when the config cannot be read."""
    import json

    hf_model = hf_model or os.environ.get("HF_MODEL", "")
    try:
        with open(os.path.join(hf_model, "config.json")) as f:
            cfg = json.load(f)
        cfg = cfg.get("text_config", cfg)
        return tp_path_forced_for_single_device(int(cfg.get("hidden_size", 0)), int(cfg.get("num_hidden_layers", 0)))
    except (OSError, ValueError, TypeError):
        return tp_path_forced_for_single_device(0, 0)


class Qwen36ModelArgs(ModelArgs):
    """ModelArgs for the Qwen3.5 / 3.6 family on Blackhole (9B / 27B / 35B-A3B; dense + MoE)."""

    # Opt into base ModelArgs TP > n_kv_heads path; attention/tp.py replicates via replicate_kv_weight.
    SUPPORTS_KV_REPLICATION = True

    def __init__(
        self,
        mesh_device=None,
        max_batch_size=1,
        max_seq_len=2048,
        enable_mtp=None,
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
        # drafter AND the substrate every speculative decoder (SpeculativeDecoder, DFlash2Decoder,
        # the DFlash2 serving class) is built on. mtp_use_dedicated_embeddings=False means it
        # shares tok_embeddings.
        # Loading the head is OFF by default: plain serving (Qwen36ForCausalLM -- the p150x2 `tp2` /
        # `p1d1` profiles, the plain p300x2 package) must not pay its weights, its extra paged KV
        # cache and its construction. A speculative mode requests it: QWEN36_MTP=1 explicitly, or
        # QWEN36_DRAFTER=mtp (the native drafter) with QWEN36_MTP unset; the DFlash2 paths set
        # QWEN36_MTP=1 themselves (qwen36_vllm_dflash.py when speculation is on, text_demo.py unless
        # QWEN36_SPEC=0). enable_mtp=True/False overrides the environment either way.
        self.mtp_num_hidden_layers = getattr(text_config, "mtp_num_hidden_layers", 0)
        self.mtp_use_dedicated_embeddings = getattr(text_config, "mtp_use_dedicated_embeddings", False)
        if enable_mtp is None:
            enable_mtp = mtp_requested_by_env()
        self.has_mtp = self.mtp_num_hidden_layers > 0 and bool(enable_mtp)
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

        # TP config. `use_tp_path` selects the tensor-parallel CODE PATH (TPAttention / TPGatedDeltaNet /
        # tp_common progcfgs, chunk-outer traced prefill, batched paged decode with per-slot GDN state).
        # It is on for every multi-device mesh and, on a (1,1) mesh, for the 27B class or when
        # QWEN36_FORCE_TP_PATH=1 (see tp_path_forced_for_single_device). `num_devices` keeps meaning the
        # REAL device count: collectives, shard math and the fused-collective gates key on it, so every
        # TP=1 fallback keys on `tp1` (= TP path on one die), never on a faked device count.
        self.num_devices = mesh_device.get_num_devices() if mesh_device is not None else 1
        self.use_tp_path = mesh_device is not None and (
            self.num_devices > 1 or tp_path_forced_for_single_device(self.dim, self.n_layers)
        )
        self.tp1 = self.use_tp_path and self.num_devices == 1
        if self.tp1 and os.environ.get("QWEN36_MTP") is None and self.has_mtp:
            # TP=1 mode (one die): the MTP head / speculative verify are the multi-device TP path and are not
            # validated on one die, so the drafter weights are not loaded by default (plain decode, the PD
            # profile's decode node). QWEN36_MTP=1 opts back in explicitly.
            self.has_mtp = False
        if self.use_tp_path:
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
        # Per-device width of the [qkv|z|a|b] fused in-projection: folding the tiny a/b (decay/beta)
        # projection into qkvz removes a whole decode matmul while keeping the (good) K=dim. Default
        # (was QWEN36_GDN_FUSE_AB); gdn/tp.py fuses whenever the qkvz weight is DRAM-sharded.
        self.gdn_qkvzab_dim_tp = self.gdn_qkvz_dim_tp + 2 * self.gdn_nv_tp
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
        self.decode_grid_h = mesh_device.compute_with_storage_grid_size().y  # rows available to the 1D grids (10 on BH)
        self.mlp_1d_decode = True
        # gate/up: num_cores=44 -> 11x4 on BH, the fastest measured config (wide1d_11x4c, 42.8us vs
        # 43.9us for the old 8x4=forced1d_32c). On WH (decode_grid_w=8) this falls back to 8x6.
        # TP=1: N=17408 (544 tiles) on 44 cores -> per_core_N=13 (prime, 1-wide subblock); 77 cores (11x7) ->
        # per_core_N=8 -> 4-wide subblock. Same subblock-rescue for w2 / wo / gdn_out below (44 cores -> 4/core).
        # TP=1 grids re-swept on one BH die at the TP=1 shapes (tests/decode_matmul_sweep_tp1_scratch.py, traced
        # replays, every candidate checked bit-identical to the legacy grid -- fp32 dest accumulation makes the grid
        # a pure partition of N): gate/up bf4 [5120,17408] 186.5 -> 177.3 us on the full 11x10 grid (per_core_N 5);
        # down bf8 [17408,5120] 269 -> 249 us, wo [8192,5120] 130 -> 121 us and gdn_out [6144,5120] 99 -> 92 us on
        # an 11x3 grid (per_core_N 5); qkvzab [5120,16480] 254 -> 244 us (11x10); attn qkv [5120,14336] 242 -> 215 us
        # (11x9). ~4 ms of a ~74 ms fused-GDN decode step; served TPOT 77.2 -> 75.1 ms (1 user), 90.3 -> 87.4 (8 users), greedy
        # streams identical to the previous grids on the 21-prompt corpus. QWEN36_TP1_DECODE_GRIDS=0 restores the old grids.
        _tp1_grids = tp == 1 and os.environ.get("QWEN36_TP1_DECODE_GRIDS", "1") == "1"
        _gu_cores = (110 if _tp1_grids else 77) if tp == 1 else 44
        _rp_cores = (33 if _tp1_grids else 44) if tp == 1 else 33  # 33 = the swept 11x3 grid (per_core_N 5)
        self.mlp_w1_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=_gu_cores,
            fused_activation=ttnn.UnaryOpType.SILU,
            grid_w=self.decode_grid_w,
            grid_h=self.decode_grid_h,
        )
        self.mlp_w3_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.hidden_dim // tp,
            num_cores=_gu_cores,
            grid_w=self.decode_grid_w,
            grid_h=self.decode_grid_h,
        )
        # down: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~63us, +28% vs
        # the old 8x2). On WH (decode_grid_w=8) this falls back to 8x5.
        self.mlp_w2_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.hidden_dim // tp,
            self.dim,
            num_cores=_rp_cores,
            grid_w=self.decode_grid_w,
            grid_h=self.decode_grid_h,
        )

        # Input-projection 1D decode (DEFAULT): same idea for attn QKV+gate and GDN QKVZAB in-projections.
        # Weights load interleaved (prefill AGMM verified bit-identical); tuned grids per test_mlp_matmul_sweep.
        self.proj_1d_decode = True
        # TP=1: the per-device N is 4x TP=4's (14336 = 448 tiles); on the 64-core grid per_core_N=7 (prime ->
        # 1-wide subblock). 56 cores (8x7) give per_core_N=8 -> a 4-wide subblock. Untuned otherwise.
        self.attn_qkv_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.attn_qkv_fused_dim_tp,
            num_cores=(99 if _tp1_grids else 56) if tp == 1 else 64,
            grid_w=self.decode_grid_w if _tp1_grids else 8,
            grid_h=self.decode_grid_h,
        )
        # gdn_qkvz: num_cores=44 -> 11x4 on BH, the fastest measured config (wide1d_11x4c, ~59us, +22%
        # vs the old 8x5). On WH (decode_grid_w=8) this falls back to 8x6.
        self.gdn_qkvz_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.dim,
            self.gdn_qkvzab_dim_tp,
            num_cores=110 if _tp1_grids else 44,
            grid_w=self.decode_grid_w,
            grid_h=self.decode_grid_h,
        )
        # Output projections (attn wo, GDN o_proj): already interleaved+auto (no weight relayout, not in
        # the prefill AGMM fusion), so this just swaps ttnn-auto for a tuned ~32-core 1D decode grid.
        # attn_wo: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4). On WH (decode_grid_w=8) this falls back to 8x5.
        self.attn_wo_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M, self.attn_out_dim_tp, self.dim, num_cores=_rp_cores, grid_w=self.decode_grid_w, grid_h=self.decode_grid_h
        )
        # gdn_out: num_cores=33 -> 11x3 on BH, the fastest measured config (wide1d_11x3c, ~24us, +25%
        # vs the old 8x4; same 1536x5120 shape as attn_wo). On WH (decode_grid_w=8) this falls back to 8x5.
        self.gdn_out_decode_1d_progcfg = tpc.create_matmul_1d_decode_progcfg(
            M,
            self.gdn_value_dim_tp,
            self.dim,
            num_cores=_rp_cores,
            grid_w=self.decode_grid_w,
            grid_h=self.decode_grid_h,
        )

        # Prefill matmul factory (M = seq_len)
        self._prefill_grid = tpc.prefill_grid_default()
        self.prefill_tuning = tpc.prefill_tuning(tp)
        if self.moe_num_experts > 0:
            # The dense TP=4 tuning picks in0_block_w = min(cap, k_tiles // grid), which the
            # 35B-A3B's attention/GDN prefill K dims don't divide (Kt % in0_block_w != 0). Force
            # the divisor path (largest divisor of k_tiles ≤ cap) so the block always divides;
            # dense 9B/27B keep their tuned block.
            self.prefill_tuning = {**self.prefill_tuning, "in0_block_w_divisor": True}
        self.prefill_progcfg = lambda seq_len, k, n: tpc.create_prefill_matmul_program_config(
            seq_len, k, n, grid_size=self._prefill_grid, tuning=self.prefill_tuning
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
        if getattr(self, "use_tp_path", self.num_devices > 1):
            suffix += "_mesh" + "x".join(str(d) for d in self.cluster_shape)
            if self.num_devices == 1:
                # TP code path on ONE die: its layouts (layers.N/tp/* shards, `.il` fused weights, the
                # `<norm>_distributed` copies) must not share a dir with the single-device 9B port, which
                # would otherwise resolve to the same `<TT_CACHE_PATH>/P150/tensor_cache_bfp8`.
                suffix += "_tp1"
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
