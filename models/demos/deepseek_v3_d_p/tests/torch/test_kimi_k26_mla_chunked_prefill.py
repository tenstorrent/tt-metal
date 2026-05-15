# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Torch reference test for Kimi K2.6 Multi-Latent Attention (MLA) with chunked prefill.

For each total ISL in {5K, 10K, ..., 55K}, the test compares:
  - mla_full_attention      : one-shot causal MLA over the full sequence
  - mla_chunked_prefill     : iterative MLA in 5K chunks, building a growing KVPE cache

The chunked-prefill implementation below is the golden reference for a future ttnn
chunked-prefill MLA op. This test will be re-targeted at that op once it exists, by
swapping mla_chunked_prefill for the device implementation.

Kimi K2.6 attention dims are taken from moonshotai/Kimi-K2.6 config.json (text_config);
the attention stack matches the DeepSeek V3 MLA family, so we reuse
DeepseekV3Attention with Kimi-K2.6 hyperparameters and random weights.
"""

from typing import Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.reference.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    apply_rotary_pos_emb,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

# Kimi K2.6 attention dims (text_config slice of moonshotai/Kimi-K2.6 config.json).
KIMI_K26_TEXT_CONFIG = dict(
    hidden_size=7168,
    num_attention_heads=64,
    num_key_value_heads=64,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    rms_norm_eps=1e-5,
    rope_theta=50000.0,
    max_position_embeddings=262144,
    rope_scaling={
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 64.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    attention_bias=False,
    attention_dropout=0.0,
    initializer_range=0.02,
    num_hidden_layers=1,
    vocab_size=163840,
)

CHUNK_SIZE = 5 * 1024
MAX_TOTAL_SEQ = 55 * 1024

# CPU SDPA materializes Q@K^T in fp32 — chunk both axes to keep peak working set bounded.
HEAD_CHUNK = 8
SEQ_CHUNK = 2048


def build_kimi_k26_attention(seed: int = 42) -> DeepseekV3Attention:
    """Instantiate a DeepseekV3Attention sized for Kimi K2.6 with deterministic random weights."""
    config = DeepseekV3Config(**KIMI_K26_TEXT_CONFIG)
    torch.manual_seed(seed)
    attn = DeepseekV3Attention(config, layer_idx=0)

    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=config.initializer_range)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, DeepseekV3RMSNorm) and hasattr(m, "weight"):
            nn.init.ones_(m.weight)

    attn.apply(init)
    return attn.eval().to(torch.bfloat16)


def _project_qkv_for_chunk(
    attn: DeepseekV3Attention,
    hidden_chunk: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Q (absorbed into latent space) and KVPE for one chunk.

    Args:
        hidden_chunk: [bsz, chunk_len, hidden_size]
        position_ids: [bsz, chunk_len] absolute positions for RoPE

    Returns:
        q_latent:   [bsz, num_heads, chunk_len, kv_lora_rank + qk_rope_head_dim]
        kvpe_chunk: [bsz, 1,         chunk_len, kv_lora_rank + qk_rope_head_dim]
    """
    bsz, chunk_len, _ = hidden_chunk.shape

    # Q LoRA + absorption into latent K space (so attention is over [kv_lora_rank + qk_rope_head_dim]).
    q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_chunk)))
    q = q.view(bsz, chunk_len, attn.num_heads, attn.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

    kv_b1 = attn.kv_b_proj.weight.view(attn.num_heads, -1, attn.kv_lora_rank)[:, : attn.qk_nope_head_dim]
    q_nope = torch.matmul(q_nope, kv_b1)

    # KV LoRA → latent k_nope + rope k_pe (shared across heads).
    compressed = attn.kv_a_proj_with_mqa(hidden_chunk)
    compressed, k_pe = torch.split(compressed, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, 1, chunk_len, attn.qk_rope_head_dim)
    k_nope = attn.kv_a_layernorm(compressed).view(bsz, 1, chunk_len, attn.kv_lora_rank)

    # RoPE: cos/sin sized to cover the largest absolute position touched by this chunk.
    max_pos = int(position_ids.max().item()) + 1
    cos, sin = attn.rotary_emb(k_nope, seq_len=max_pos, meta_style=True)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids, meta_style=True)

    q_latent = k_pe.new_empty(bsz, attn.num_heads, chunk_len, attn.kv_lora_rank + attn.qk_rope_head_dim)
    q_latent[:, :, :, : attn.kv_lora_rank] = q_nope
    q_latent[:, :, :, attn.kv_lora_rank :] = q_pe

    kvpe_chunk = k_pe.new_empty(bsz, 1, chunk_len, attn.kv_lora_rank + attn.qk_rope_head_dim)
    kvpe_chunk[:, :, :, : attn.kv_lora_rank] = k_nope
    kvpe_chunk[:, :, :, attn.kv_lora_rank :] = k_pe
    return q_latent, kvpe_chunk


def _chunked_sdpa_and_oproj(
    attn: DeepseekV3Attention,
    q_latent: torch.Tensor,
    kvpe_cache: torch.Tensor,
    abs_q_start: int,
) -> torch.Tensor:
    """
    Run causal SDPA for Q[abs_q_start : abs_q_start + q_len) against K[0 : abs_q_start + q_len),
    then expand V from latent to v_head_dim and apply o_proj. Head- and Q-seq-chunked
    internally to keep CPU SDPA's working set bounded.

    Args:
        q_latent:    [bsz, num_heads, q_len, kv_lora_rank + qk_rope_head_dim]
        kvpe_cache:  [bsz, 1, k_len, kv_lora_rank + qk_rope_head_dim], with k_len >= abs_q_start + q_len
        abs_q_start: absolute position of q_latent[..., 0, :] in the full sequence

    Returns: [bsz, q_len, num_heads * v_head_dim] after o_proj.
    """
    bsz, num_heads, q_len, _ = q_latent.shape
    k_total = abs_q_start + q_len
    assert kvpe_cache.shape[-2] >= k_total, "KVPE cache must cover up to last Q position"
    value_cache = kvpe_cache[..., : attn.kv_lora_rank]

    out_latent = torch.empty(bsz, num_heads, q_len, attn.kv_lora_rank, dtype=q_latent.dtype, device=q_latent.device)
    # PyTorch SDPA's is_causal=True aligns Q at top-left of the K window (mask[i,j] = j<=i),
    # which is only correct when abs_q_start == 0. For cached/chunked prefill we need the
    # absolute-position causal mask: Q at absolute position p attends K positions 0..p.
    for h0 in range(0, num_heads, HEAD_CHUNK):
        h1 = min(h0 + HEAD_CHUNK, num_heads)
        for s0 in range(0, q_len, SEQ_CHUNK):
            s1 = min(s0 + SEQ_CHUNK, q_len)
            k_upto = abs_q_start + s1  # exclusive
            q_h = q_latent[:, h0:h1, s0:s1, :]
            k_h = kvpe_cache[:, :, :k_upto, :].expand(bsz, h1 - h0, -1, -1)
            v_h = value_cache[:, :, :k_upto, :].expand(bsz, h1 - h0, -1, -1)
            q_abs = torch.arange(abs_q_start + s0, abs_q_start + s1, device=q_h.device)
            k_abs = torch.arange(k_upto, device=q_h.device)
            attn_mask = k_abs.unsqueeze(0) <= q_abs.unsqueeze(1)  # [Lq, Lk] bool
            out_latent[:, h0:h1, s0:s1, :] = F.scaled_dot_product_attention(
                q_h, k_h, v_h, attn_mask=attn_mask, scale=attn.softmax_scale
            )

    kv_b2 = attn.kv_b_proj.weight.view(attn.num_heads, -1, attn.kv_lora_rank)[:, -attn.v_head_dim :].transpose(1, 2)
    out = torch.matmul(out_latent, kv_b2)
    out = out.transpose(1, 2).contiguous().reshape(bsz, q_len, attn.num_heads * attn.v_head_dim)
    return attn.o_proj(out)


def mla_full_attention(attn: DeepseekV3Attention, hidden_states: torch.Tensor) -> torch.Tensor:
    """One-shot causal MLA over the full sequence — the per-token golden output."""
    bsz, total_seq, _ = hidden_states.shape
    position_ids = torch.arange(total_seq, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
    q_latent, kvpe_full = _project_qkv_for_chunk(attn, hidden_states, position_ids)
    return _chunked_sdpa_and_oproj(attn, q_latent, kvpe_full, abs_q_start=0)


def mla_chunked_prefill(
    attn: DeepseekV3Attention,
    hidden_states: torch.Tensor,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative chunked prefill: walk the sequence in `chunk_size`-token windows, project
    Q/K/V for each chunk with absolute position ids, write the new K/V into a preallocated
    KVPE cache, and run causal SDPA of the chunk's Q against the cache populated up through
    the chunk's end. Concatenated outputs match the one-shot MLA path.

    Args:
        hidden_states: [bsz, total_seq, hidden_size]
        chunk_size:    new-ISL per step (50K cache + 5K new ISL → chunk_size=5K, total_seq=55K)

    Returns:
        output:     [bsz, total_seq, hidden_size]
        kvpe_cache: [bsz, 1, total_seq, kv_lora_rank + qk_rope_head_dim]
    """
    bsz, total_seq, _ = hidden_states.shape
    cache_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
    kvpe_cache = torch.zeros(bsz, 1, total_seq, cache_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    outs = []
    for start in range(0, total_seq, chunk_size):
        end = min(start + chunk_size, total_seq)
        chunk = hidden_states[:, start:end, :]
        position_ids = torch.arange(start, end, dtype=torch.long, device=chunk.device).unsqueeze(0).expand(bsz, -1)

        q_latent, kvpe_chunk = _project_qkv_for_chunk(attn, chunk, position_ids)
        kvpe_cache[:, :, start:end, :] = kvpe_chunk

        out = _chunked_sdpa_and_oproj(attn, q_latent, kvpe_cache[:, :, :end, :], abs_q_start=start)
        outs.append(out)

    return torch.cat(outs, dim=1), kvpe_cache


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "total_seq",
    list(range(CHUNK_SIZE, MAX_TOTAL_SEQ + 1, CHUNK_SIZE)),
    ids=lambda s: f"seq{s // 1024}k",
)
def test_kimi_k26_mla_chunked_prefill(total_seq: int) -> None:
    bsz = 1
    hidden_size = KIMI_K26_TEXT_CONFIG["hidden_size"]

    attn = build_kimi_k26_attention(seed=42)

    torch.manual_seed(0)
    hidden_states = torch.randn(bsz, total_seq, hidden_size, dtype=torch.bfloat16)

    logger.info(
        f"Kimi K2.6 MLA chunked-prefill: total_seq={total_seq} "
        f"({total_seq // 1024}K), chunk_size={CHUNK_SIZE} ({CHUNK_SIZE // 1024}K), "
        f"n_chunks={(total_seq + CHUNK_SIZE - 1) // CHUNK_SIZE}"
    )

    with torch.no_grad():
        ref_output = mla_full_attention(attn, hidden_states)
        chunked_output, _ = mla_chunked_prefill(attn, hidden_states, CHUNK_SIZE)

    _, pcc_msg = assert_with_pcc(ref_output, chunked_output, 0.99)
    logger.info(f"PCC(full vs chunked) at seq{total_seq // 1024}k: {pcc_msg}")
