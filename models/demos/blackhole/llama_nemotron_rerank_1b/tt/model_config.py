# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shapes and precision settings for llama-nemotron-rerank-1b-v2 on one Blackhole P150."""

from dataclasses import dataclass, field

import ttnn

MODEL_ID = "nvidia/llama-nemotron-rerank-1b-v2"
PAD_TOKEN_ID = 128001
# Sequence-length buckets. A request is padded to the smallest bucket that fits it.
SEQ_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)


# minimal_matmul block sizes (M_block, K_block, N_block, subblock_h, subblock_w) in tiles,
# per weight kind and activation rows M = batch * seq_len, from perf_tools/sweep_mm.py on a P150
# (fp32 accumulation, so subblock_h * subblock_w <= 4; 512 / 16384 rows re-swept with bfp8 activations).
# A row count not listed uses the largest listed M that is <= it.
MM_CONFIGS = {
    "qkv": {
        512: (4, 8, 8, 1, 4),
        2048: (8, 8, 8, 1, 4),
        4096: (16, 8, 4, 1, 4),
        8192: (8, 16, 4, 1, 4),
        16384: (8, 32, 4, 1, 4),
    },
    "wo": {
        512: (4, 8, 8, 2, 2),
        2048: (8, 8, 8, 2, 2),
        4096: (16, 8, 4, 2, 2),
        8192: (4, 8, 8, 1, 4),
        16384: (8, 8, 4, 1, 4),
    },
    # 16384 rows with fp32 accumulation next to the resident residual (sweep_mm.py --l1-reserve-mb 64):
    # (4, 32, 8, 1, 4) does not fit; (2, 32, 8, 1, 4) 3690 us alone and 32x512 148.5 M cycles (75.1 % of SoL),
    # but under continuous replays the board throttles it hard: 32x512 111 ms per replay at first, 335 ms
    # after ~12 replays, against 116 -> 182-188 ms for (4, 16, 8, 1, 4) (the best K 16 block, 4042 us alone).
    # 5120 rows (10 x 512; sweep_mm.py --l1-reserve-mb 21): (4, 32, 8, 1, 4) HiFi2 1169 vs 1271 us, LoFi 789 vs 896.
    "ff13s": {
        512: (8, 16, 8, 1, 4),
        2048: (4, 16, 8, 1, 4),
        4096: (4, 16, 8, 1, 4),
        5120: (4, 32, 8, 1, 4),
        8192: (4, 16, 8, 1, 4),
        16384: (4, 16, 8, 1, 4),
    },
    "ff13": {512: (4, 8, 8, 1, 4), 4096: (4, 16, 16, 1, 4), 16384: (8, 16, 8, 1, 4)},
    "ff2": {
        512: (4, 16, 8, 2, 2),
        2048: (4, 8, 8, 2, 2),
        4096: (16, 8, 4, 1, 4),
        8192: (4, 16, 4, 1, 4),
        16384: (8, 16, 8, 1, 4),
    },
    "score": {512: (4, 8, 1, 4, 1), 16384: (16, 8, 1, 4, 1)},  # 16384 rows: 176 -> 151 us
    # bf16 DST accumulation (see RerankConfig.bf16_acc_kinds): 8-tile subblocks. FF2 at LoFi, 16384 rows:
    # (8,16,8,1,4) fp32 ~1118 -> (8,8,8,1,8) bf16 986 us (perf_tools/sweep_mm.py).
    "ff2_bf16acc": {16384: (8, 8, 8, 1, 8)},
    # FF1+FF3 fused SwiGLU, 16384 rows: (4,32,8,1,4) fp32 3677 -> (4,16,8,1,8) bf16 3611 us.
    "ff13s_bf16acc": {16384: (4, 16, 8, 1, 8)},
}


# matmul_qkv_rope blocks (M_block, K_block, N_block, subblock_h, subblock_w) per rows, from
# perf_tools/probe_qkv_rope.py; N_block must be even (whole heads).
QKV_ROPE_CONFIGS = {512: (2, 8, 2, 1, 2), 2048: (4, 8, 4, 1, 4), 4096: (8, 8, 4, 1, 4), 16384: (8, 16, 2, 4, 1)}
# Weight CB depth in K blocks for matmul_qkv_rope: 32x512 861 -> 851 us (4).
QKV_ROPE_IN1_DEPTH = 4
# Fidelity of the rotation's multiplies: HiFi2 keeps Q / K PCC at 0.999996 (HiFi4 0.999998), 32x512 910 -> 887 us.
QKV_ROPE_FIDELITY = "HiFi2"


# matmul_add_norm blocks (M_block, K_block, N_block, subblock_h, subblock_w); M_block (and with it
# subblock_h) is lowered to a divisor of the rows per core. N_block 7 = the 64 columns over 10 core
# rows, so no padding column is computed. 16384 rows (perf_tools/probe_mm_add_norm.py, gamma folded):
# WO 681 us, FF2 1216 us, against 1131 / 1547 us for the matmul plus fused_add_rmsnorm.
MM_ADD_NORM_CONFIGS = {"wo": (4, 8, 7, 4, 1), "ff2": (8, 8, 7, 4, 1)}
# Weight (in1) CB depth in K blocks: the x = 0 core reads ahead while the row multicasts.
# 16384 rows: WO 593 -> 574 us (4), FF2 1124 -> 1116 us (3; 4 does not fit next to the residual).
MM_ADD_NORM_IN1_DEPTH = {"wo": 4, "ff2": 3}
# Fidelity of matmul_add_norm's y multiplies (h * inv, * gamma). HiFi2 for both fused ops dropped MIRACL
# batch 32 Pearson to 0.99947 before gamma folding; with it folded it passes (RerankConfig.y_hifi2_kinds).
MM_ADD_NORM_Y_FIDELITY = "HiFi4"


def sdpa_chunks(batch, seq_len):
    """(q_chunk, k_chunk) per shape, from perf_tools/sweep_sdpa.py and probe_mask.py on a P150."""
    if seq_len <= 512:
        return min(seq_len, 128), min(seq_len, 256 if batch == 1 else 512)
    return 512, 256


# Blocks for >= 8192 rows when the residual stream is L1-resident (FU-05: CBs must end below it):
# perf_tools/sweep_mm.py --l1-reserve-mb 64 --xdtype bfloat8_b --odtype bfloat8_b at 16384 rows.
MM_CONFIGS_L1_RESIDUAL = {
    "qkv": (8, 16, 2, 2, 2),
    "wo": (8, 8, 4, 2, 2),
    "ff13s": (2, 16, 8, 1, 4),
    "ff2": (4, 16, 4, 1, 4),
}


def mm_block_config(kind, rows, l1_residual=False):
    if l1_residual and rows >= 8192 and kind in MM_CONFIGS_L1_RESIDUAL:
        return MM_CONFIGS_L1_RESIDUAL[kind]
    table = MM_CONFIGS[kind]
    keys = [m for m in sorted(table) if m <= rows] or [min(table)]
    return table[keys[-1]]


def prompt_template(query: str, passage: str) -> str:
    """The text format the reranker was trained on (from the model card)."""
    return f"question:{query} \n \n passage:{passage}"


@dataclass
class RerankConfig:
    dim: int = 2048
    n_layers: int = 16
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 64
    hidden_dim: int = 8192
    vocab_size: int = 128256
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    rope_scale_factor: float = 32.0
    rope_orig_context_len: int = 8192
    max_seq_len: int = 8192

    # Weight dtype per tensor group.
    wqkv_dtype: object = ttnn.bfloat8_b
    wo_dtype: object = ttnn.bfloat8_b
    w13_dtype: object = ttnn.bfloat8_b
    w2_dtype: object = ttnn.bfloat8_b
    # Activation dtype between ops.
    act_dtype: object = ttnn.bfloat16
    # dtype of the SwiGLU product (FF2's input). bfp8: FF2 at 32x512 38.7 -> 28.0 M cycles
    # (minimal_matmul runs best with in0 and weight in one dtype); MIRACL Pearson 0.99966.
    ff2_in_dtype: object = ttnn.bfloat8_b
    # dtype of the RMSNorm outputs that feed QKV and FF1/FF3 (the final norm, feeding the score head, stays bf16).
    # bfp8: 32x512 256.5 -> 251.7 M cycles; MIRACL Pearson 0.99956 (b1, b8) / 0.99963 (b32).
    norm_out_dtype: object = ttnn.bfloat8_b
    # dtype of the final RMSNorm output (the score head's input) on the resident-residual path. bfp8: 32x512
    # 151.6 -> 151.4 M cycles; MIRACL batch 32 Pearson 0.99962 (unchanged).
    final_norm_dtype: object = ttnn.bfloat8_b
    # Output dtypes of the tensors between ops (each is written once and read once).
    # bfp8 into / out of fused_heads_rope gives garbage K / V (Q correct): open issue, kept bf16.
    qkv_out_dtype: object = ttnn.bfloat16  # QKV matmul -> head split + RoPE
    heads_out_dtype: object = ttnn.bfloat16  # Q / K / V -> SDPA
    # MIRACL batch 8: WO bfp8 Pearson 0.99963, FF2 bfp8 0.99966 (baseline 0.99966).
    wo_out_dtype: object = ttnn.bfloat8_b  # WO matmul -> residual add
    ff2_out_dtype: object = ttnn.bfloat8_b  # FF2 matmul -> residual add
    # Matmul math fidelity.
    mm_fidelity: object = ttnn.MathFidelity.HiFi2
    # Matmul kinds (qkv, wo, ff13s, ff13, ff2) that run at LoFi instead of mm_fidelity, comma separated.
    # MIRACL Pearson vs fp32: ff2 0.99959 (baseline 0.99962), ff13s 0.99928, qkv+wo 0.99920, all 0.99918.
    lofi_kinds: str = "ff2"
    # Layers (comma-separated indices, "" = all) where lofi_kinds applies; ff2 always follows lofi_kinds.
    lofi_layers: str = ""
    # Matmul kinds that accumulate in bf16 DST (8-tile subblocks) instead of fp32 (4-tile), comma separated;
    # applies from 16384 rows (MM_CONFIGS "<kind>_bf16acc"). Off: with packed rows every pair runs at 32 x 512,
    # and ff13s in bf16 drops MIRACL Pearson to 0.99945 (fp32: 0.99959), under the 0.9995 gate. FF1+FF3
    # 32x512 77.75 (bf16) vs 78.93 M cycles. (FF2 on the resident path is matmul_add_norm, which ignores this.)
    bf16_acc_kinds: str = ""
    # HiFi2 SDPA raised the MIRACL mean score error 0.23 -> 0.30 at no measurable speed gain.
    sdpa_fidelity: object = ttnn.MathFidelity.HiFi4
    # Fidelity of SDPA's compute config; None = sdpa_fidelity. ttnn's SDPA ignores
    # SDPAProgramConfig.matmul_math_fidelity (only ring-joint SDPA reads it), so this sets its matmuls too.
    # 32x512 SDPA: HiFi2 14.73, LoFi 14.92 vs 14.24 M cycles (slower); max_cores_per_head_batch 32 and
    # segmented_accumulation: no change.
    sdpa_softmax_fidelity: object = None
    sdpa_max_cores_per_head_batch: int = 16
    sdpa_k_splits: int = 1
    sdpa_segmented_accumulation: bool = False
    # SDPA softmax exp in the SFPU's approximate mode.
    sdpa_exp_approx: bool = False
    # Per fused kind ("wo", "ff2"), comma separated: the y multiplies at HiFi2 instead of MM_ADD_NORM_Y_FIDELITY.
    # Both (with gamma folded): 32x512 150.67 -> 150.38 M cycles; MIRACL Pearson batch 32 0.99960
    # (nDCG@10 0.6834), batch 8 0.99967 (0.6745).
    y_hifi2_kinds: str = "ff2,wo"
    # SDPA q / k chunk sizes; 0 = the sdpa_chunks table.
    sdpa_q_chunk: int = 0
    sdpa_k_chunk: int = 0
    # Padding mask dtype. The mask holds only 0 and a large negative value, which bfp4 stores exactly;
    # SDPA reads it once per (head, q chunk, k chunk), so its size is SDPA's main data movement.
    mask_dtype: object = ttnn.bfloat4_b
    # Pass attn_mask_block_map so SDPA skips (q chunk, k chunk) blocks that hold no real query or key.
    sdpa_block_map: bool = True
    # Masked SDPA at batch 8 / 32 and 128-512 tokens runs tt/custom_ops/sdpa_mask_cache: ttnn's SDPA with a reader
    # that reads the [B, 1, S, S] mask from DRAM once per batch per core instead of once per Q head (bit-identical
    # output). One 32x512 call: 1253 -> 835 us (DRAM output), 1019 -> 780 us (L1 output).
    sdpa_mask_cache: bool = True
    # sdpa_mask_cache reads the pair index and writes mask tiles that lie within one pair (all 0) or across pairs
    # (all -inf) in L1, reading only tiles that straddle a pair boundary from the mask in DRAM.
    sdpa_pair_tiles: bool = True
    # SDPA writes heads side by side ([B, 1, S, n_heads * head_dim]) so nlp_concat_heads is not needed.
    sdpa_concat_heads: bool = True
    # Residual add + following RMSNorm as one op (tt/custom_ops/fused_add_rmsnorm) writing both outputs.
    fuse_add_norm: bool = True
    # Head split + RoPE on Q and K as one op (tt/custom_ops/fused_heads_rope) instead of three.
    fuse_heads_rope: bool = True
    # QKV matmul + head split + RoPE as one program (tt/custom_ops/matmul_qkv_rope); supersedes
    # fuse_heads_rope. 32x512: 1406 -> 924 us per layer (perf_tools/probe_qkv_rope.py).
    fuse_qkv_rope: bool = True
    # The fused op gives each core whole tile-rows, so below one row per core (4096 rows on 128 cores)
    # the stock add + rms_norm spread wider and win (1x512: 8.8 vs 9.7 ms; 1x2048: 33.4 vs 42.2 ms).
    # 8x512: 49.9 -> 50.9 % of SoL with the fused op; 32x512: 238 -> 226 ms.
    fuse_add_norm_min_rows: int = 4096
    # Math fidelity of the fused add + RMSNorm's multiplies (x^2, x * inv, * gamma). HiFi2: op 561 -> 533 us
    # but MIRACL Pearson 0.99951, at the 0.9995 gate; kept HiFi4.
    add_norm_fidelity: object = ttnn.MathFidelity.HiFi4
    # dtype of the residual stream written by the fused add + RMSNorm.
    residual_dtype: object = ttnn.bfloat16
    # Fused path: keep the residual stream in L1 (one buffer, updated in place by the fused add + norm).
    residual_l1: bool = False
    # Fused path: residual stream resident in L1, sharded like matmul_add_norm's grid; WO and FF2 run
    # as matmul_add_norm (matmul + residual add + RMSNorm in one program). Supersedes residual_l1.
    resident_residual: bool = True
    # Resident path: embed_rmsnorm gathers each core's slice of the token embeddings straight into its
    # residual shard (instead of ttnn.embedding to DRAM + to_resident + resident_rmsnorm).
    # 32x512: 645 -> 304 us for the three (perf_tools/probe_embed_rmsnorm.py).
    embed_gather: bool = True
    # Fused kinds ("wo", "ff2") whose activation blocks are multicast down each core column
    # (tt/custom_ops/kernels/dm_in0_mcast.cpp) instead of relayed core to core.
    # 16384 rows (perf_tools/probe_mm_add_norm.py --in0-mcast): WO 602 -> 593, FF2 1129 -> 1124 us.
    in0_mcast_kinds: str = "wo,ff2"
    # Fold every RMSNorm's gamma into the rows of the matmul that consumes it (QKV, FF1+FF3, score head);
    # the norms then skip their gamma multiply. MIRACL Pearson batch 32 0.99963, batch 8 0.99967.
    fold_norm_gamma: bool = True
    # Up to this many rows RMSNorm runs block-sharded over an 8-wide grid (with the interleaved <-> sharded
    # conversions): stock rms_norm gives each core whole tile-rows, 16 cores at 512 rows.
    # perf_tools/probe_norm.py: 128 rows 33.7 -> 10.7 us, 512 rows 33.9 -> 18.3 us.
    sharded_norm_max_rows: int = 512
    # minimal_matmul with the swept block sizes instead of ttnn.linear's own config.
    use_minimal_matmul: bool = True
    # FF1 + FF3 + silu(gate) * up as one minimal_matmul (fuse_swiglu) over an interleaved weight.
    fuse_swiglu: bool = True
    # The fused SwiGLU's silu in the SFPU's approximate mode.
    swiglu_approx: bool = False
    # Activations up to this many bytes are kept in L1 (interleaved) instead of DRAM; 0 = always DRAM.
    l1_act_bytes: int = 8_500_000
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_hf(cls, hf_config, **overrides):
        c = cls(
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            n_kv_heads=hf_config.num_key_value_heads,
            head_dim=getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads),
            hidden_dim=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
            norm_eps=hf_config.rms_norm_eps,
        )
        rope = getattr(hf_config, "rope_scaling", None) or getattr(hf_config, "rope_parameters", None) or {}
        c.rope_theta = rope.get("rope_theta", getattr(hf_config, "rope_theta", c.rope_theta))
        c.rope_scale_factor = rope.get("factor", c.rope_scale_factor)
        c.rope_orig_context_len = rope.get("original_max_position_embeddings", c.rope_orig_context_len)
        for k, v in overrides.items():
            setattr(c, k, v)
        return c

    def compute_config(self, fidelity, fp32_acc=True, packer_l1_acc=True, approx=False):
        return ttnn.init_device_compute_kernel_config(
            ttnn.device.Arch.BLACKHOLE,
            math_fidelity=fidelity,
            math_approx_mode=approx,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=packer_l1_acc,
        )


def config_from_overrides(items):
    """``["mm_fidelity=LoFi", "w2_dtype=bfloat4_b", "l1_act_bytes=8000000"]`` -> RerankConfig."""
    cfg = RerankConfig()
    for item in items or []:
        key, val = item.split("=", 1)
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            new = val.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            new = int(float(val))
        elif isinstance(cur, float):
            new = float(val)
        elif isinstance(cur, ttnn.MathFidelity) or key.endswith("fidelity"):
            new = getattr(ttnn.MathFidelity, val)
        elif isinstance(cur, str):
            new = val
        else:
            new = getattr(ttnn, val)
        setattr(cfg, key, new)
    return cfg
