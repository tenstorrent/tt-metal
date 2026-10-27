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
# (HiFi2, fp32 accumulation, so subblock_h * subblock_w <= 4).
# A row count not listed uses the largest listed M that is <= it.
MM_CONFIGS = {
    "qkv": {512: (4, 8, 8, 1, 4), 4096: (16, 8, 4, 1, 4), 16384: (8, 16, 4, 1, 4)},
    "wo": {512: (4, 8, 8, 2, 2), 4096: (16, 8, 4, 2, 2), 16384: (8, 8, 8, 1, 4)},
    "ff13s": {512: (4, 16, 8, 2, 2), 4096: (4, 16, 8, 1, 4), 16384: (4, 16, 8, 1, 4)},
    "ff13": {512: (4, 8, 8, 1, 4), 4096: (4, 16, 16, 1, 4), 16384: (8, 16, 8, 1, 4)},
    "ff2": {512: (4, 8, 8, 2, 2), 4096: (16, 8, 4, 1, 4), 16384: (8, 16, 4, 1, 4)},
    "score": {512: (4, 8, 1, 4, 1)},
}


def sdpa_chunks(batch, seq_len):
    """(q_chunk, k_chunk) per shape, from perf_tools/sweep_sdpa.py and probe_mask.py on a P150."""
    if seq_len <= 512:
        return min(seq_len, 128), min(seq_len, 256 if batch == 1 else 512)
    return 512, 256


def mm_block_config(kind, rows):
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
    # Matmul math fidelity.
    mm_fidelity: object = ttnn.MathFidelity.HiFi2
    # Matmul kinds (qkv, wo, ff13s, ff13, ff2) that run at LoFi instead of mm_fidelity, comma separated.
    # MIRACL Pearson vs fp32: ff2 0.99959 (baseline 0.99962), ff13s 0.99928, qkv+wo 0.99920, all 0.99918.
    lofi_kinds: str = "ff2"
    # HiFi2 SDPA raised the MIRACL mean score error 0.23 -> 0.30 at no measurable speed gain.
    sdpa_fidelity: object = ttnn.MathFidelity.HiFi4
    # Padding mask dtype. The mask holds only 0 and a large negative value, which bfp4 stores exactly;
    # SDPA reads it once per (head, q chunk, k chunk), so its size is SDPA's main data movement.
    mask_dtype: object = ttnn.bfloat4_b
    # Pass attn_mask_block_map so SDPA skips (q chunk, k chunk) blocks that hold no real query or key.
    sdpa_block_map: bool = True
    # SDPA writes heads side by side ([B, 1, S, n_heads * head_dim]) so nlp_concat_heads is not needed.
    sdpa_concat_heads: bool = True
    # Residual add + following RMSNorm as one op (tt/custom_ops/fused_add_rmsnorm) writing both outputs.
    fuse_add_norm: bool = True
    # The fused op gives each core whole tile-rows, so below ~2 rows per core the stock add + rms_norm
    # spread wider and win (1x512: 8.8 vs 9.7 ms; 1x2048: 33.4 vs 42.2 ms; 32x512: 238 vs 226 ms).
    fuse_add_norm_min_rows: int = 8192
    # Up to this many rows RMSNorm runs block-sharded over an 8-wide grid (with the interleaved <-> sharded
    # conversions): stock rms_norm gives each core whole tile-rows, 16 cores at 512 rows.
    # perf_tools/probe_norm.py: 128 rows 33.7 -> 10.7 us, 512 rows 33.9 -> 18.3 us.
    sharded_norm_max_rows: int = 512
    # minimal_matmul with the swept block sizes instead of ttnn.linear's own config.
    use_minimal_matmul: bool = True
    # FF1 + FF3 + silu(gate) * up as one minimal_matmul (fuse_swiglu) over an interleaved weight.
    fuse_swiglu: bool = True
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

    def compute_config(self, fidelity, fp32_acc=True):
        return ttnn.init_device_compute_kernel_config(
            ttnn.device.Arch.BLACKHOLE,
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=True,
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
        elif isinstance(cur, ttnn.MathFidelity):
            new = getattr(ttnn.MathFidelity, val)
        elif isinstance(cur, str):
            new = val
        else:
            new = getattr(ttnn, val)
        setattr(cfg, key, new)
    return cfg
