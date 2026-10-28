#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for MiniMaxAI/MiniMax-M3.1-Flash, layer by layer (engine.py), from transformers' minimax_m3_vl
code. Same architecture and sizes as MiniMax-M3 (see generate_golden_minimax_m3.py for the attention, indexer and MoE);
what differs is in the config and the weights:

  * every layer is minimax_m3_sparse (M3: layers 0-2 full_attention), so every layer has a lightning indexer and
    stores idx_keys and indexer.output; layers 0-2 still have the dense MLP
  * RoPE theta 1e7 (M3: 5e6), still partial: dims [0:64] of each 128-dim head
  * no MTP layer; the DSpark draft head (dspark/) is neither built nor read
  * the checkpoint is quantization-aware trained and stored quantized (below)

Stored tensors (N = isl, i = layer; 60 layers, all minimax_m3_sparse; 0-2 dense MLP, 3-59 MoE;
B = min(16, ceil(N / 128)) selected key blocks):

  hidden/embed.safetensors
    model.embed_tokens.output                   [1, N, 6144]       bf16  embed_tokens (no scaling)
  hidden/layer_XX.safetensors
    model.layers.{i}.output                     [1, N, 6144]       bf16  all     h + mlp(post_attn_norm(h)),
                                                                         h = x + attn(input_norm(x))
    model.layers.{i}.self_attn.output           [1, N, 6144]       bf16  all     --sublayers: attention after o_proj
    model.layers.{i}.mlp.output                 [1, N, 6144]       bf16  all     --sublayers: dense MLP (0-2) /
                                                                         whole MoE block (3-59), before the residual
    model.layers.{i}.mlp.shared_experts.output  [N, 6144]          bf16  3-59    --sublayers: shared expert
    model.layers.{i}.mlp.experts.output         [N, 6144]          bf16  3-59    --sublayers: routed experts, BEFORE
                                                                         x routed_scaling_factor (2.0)
  cache/layer_XX.safetensors
    past_key_values.layers.{i}.keys             [1, 4, N, 128]     bf16  all     k_norm + partial RoPE (theta 1e7)
    past_key_values.layers.{i}.values           [1, 4, N, 128]     bf16  all     v_proj, no norm, no RoPE
    past_key_values.layers.{i}.idx_keys         [1, 1, N, 128]     bf16  all     indexer keys
    model.layers.{i}.self_attn.indexer.output   [1, N, B]          int64 all     selected key-block ids per query
    model.layers.{i}.mlp.gate.output            [N, 128]           fp32  3-59    router logits
    model.layers.{i}.mlp.experts.top_k_index    [N, 4]             int32 3-59    chosen experts, ascending id
    model.layers.{i}.mlp.experts.top_k_weights  [N, 4]             fp32  3-59    normalized sigmoid weights
  hidden/final.safetensors
    model.last_hidden_state                     [1, N, 6144]       bf16  final norm
    logits                                      [1, 1, 200064]     fp32  last position, untied lm_head, no softcap

Weights (quant_method "mxfp8", moe_quant_format "nvfp4-pack-quantized"). transformers 5.12 maps "mxfp8" to its FP8
quantizer, which knows nothing of the NVFP4 experts, so convert() dequantizes both itself:

  * MXFP8, every linear but the routed experts and the routers (attention q/k/v/o, indexer q/k, the dense MLP and
    the shared expert): float8_e4m3fn weight x a power-of-two scale per 1x32 block along the input dim (E8M0,
    stored uint8 as 2^(byte - 127)), by transformers' own Fp8Dequantize, as for Mistral-Medium-3.5.
  * NVFP4, the routed experts (w1 / w3 / w2 = gate / up / down): E2M1 values packed two per byte (low nibble first,
    unpacked by Fp8Dequantize's own table) x a float8_e4m3fn scale per 16 inputs / weight_global_scale (one fp32
    per tensor; w1 and w3 share it). The compressed-tensors convention: the global scale is 448 * 6 / amax(W), so it
    divides. Checked on the real checkpoint (layer 5, expert 7): every 16-block reaches |E2M1| = 6, and the result has
    std 0.028, like the MXFP8 shared expert's 0.026 (multiplying instead gives ~1e7). input_global_scale (the
    activation global scale an NVFP4 x NVFP4 kernel uses; 1.0 there) is recorded in metadata.json and dropped.
  * Routers (fp32), e_score_correction_bias, norms, embed_tokens and lm_head are unquantized in the checkpoint.

Everything is fp32 product, cast once to the compute dtype. Activations, K/V and the indexer keys stay in the
compute dtype: this is the dequantized model, not what a serving stack computes.

HF_GOLDEN_SERVING=hopper emulates vLLM serving this checkpoint on Hopper (H100 / H200, TP8, the vLLM recipe's
`--kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype": "fp8"}' --moe-backend marlin`), as implemented in
vLLM PR #63232 (branch minimax-m31-flash, a0ea9410; vllm/models/minimax_m3/), and writes isl<N>-serving-hopper
beside the plain trace. What it changes, each from that source:

  * KV cache float8_e4m3fn, scale 1.0 (the checkpoint ships none), round-to-nearest-even, saturating at +-448. K is
    k_norm + RoPE in fp32 (cos / sin from a bf16 cache), rounded straight to e4m3; V is v_proj's bf16, rounded to
    e4m3. Prefill attention reads K / V back from that cache, the current chunk's included; Q stays bf16 (fp32 norm +
    RoPE, rounded once). Stored keys / values are the cache contents.
  * Indexer: index_k likewise straight from fp32 to e4m3 (the cache), index_q rounded to bf16 and then to e4m3, no
    scales; scores are their dot products, fp32 accumulation.
  * Per-head block selection: vLLM gives each KV head its own index head (index_n_heads == num_key_value_heads == 4)
    and its own top-16 blocks (max over each block's 128 keys, own block forced in). transformers instead takes the
    max over the 4 index heads and gives every head one shared selection. Attention here is block-sparse per KV head
    accordingly: query head h reads KV head h // 16 and that head's blocks. Scores and softmax in fp32, P rounded to
    bf16 before PV, fp32 accumulation, normalized after.
  * Router in fp32: the gate weight (fp32 in the checkpoint) and e_score_correction_bias stay fp32, logits are an fp32
    GEMM of the bf16 input, as vLLM's GateLinear. (transformers casts both to the compute dtype: the bias, ~12.5, then
    has a bf16 step of 0.06, comparable to the sigmoid score gaps that pick the experts.)
  * NVFP4 block scales: Marlin re-encodes them as S0E5M3 after a per-layer power-of-two rescale, flushing tiny ones
    to 0; done here too (layer 5 expert 7's scales are 30..448, so none there; the count is in metadata.json).

Unchanged, because vLLM on Hopper runs them weight-only with bf16 activations: the MXFP8 linears (Marlin W8A16; no
activation quantization, the 1x32 E8M0 scales exact in bf16) and the NVFP4 experts (Marlin W4A16;
input_global_scale unused). Not emulated: summation order (the TP8 partial sums and fused all-reduce + RMSNorm,
Marlin applying the NVFP4 global scale after the GEMM, fused q/k/v/index projections), bf16 rounding inside the
SwiGLU-OAI kernel, Hopper's FP8 MMA accumulator width, and the order of tied top-k blocks (a bitonic sort upstream).
--selftest checks the plain model; at startup the serving attention and indexer are checked against the installed
transformers' with quantization off and the selection shared (check_serving_attention).

The prefix is the chat template's (its default system prompt, the template's only thinking mode, adaptive) up to the
ai turn, then an empty </mm:think>, as MiniMax-M3's script renders thinking_mode="disabled".

    python generate_golden_minimax_m3_1_flash.py --isl 10240
    HF_GOLDEN_INDEXER_CHUNK=4096 python generate_golden_minimax_m3_1_flash.py --isl 56320 --attn-chunk 4096
    HF_GOLDEN_SERVING=hopper python generate_golden_minimax_m3_1_flash.py --isl 10240 --attn-chunk 4096
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_golden_minimax_m3 import NOTES as M3_NOTES  # noqa: E402
from generate_golden_minimax_m3 import MiniMaxM3Golden  # noqa: E402

NOTES = {
    **M3_NOTES,
    "model.layers.{i}.self_attn.output": (
        "Attention output after o_proj (no post-attention norm, no output gate), before the residual add. Every layer "
        "is minimax_m3_sparse: attention only sees the key blocks the indexer selected."
    ),
    "model.layers.{i}.mlp.experts.output": (
        "Routed experts: sum over the top-k of top_k_weights * expert(x), tokens flattened, BEFORE the "
        "routed_scaling_factor (2.0) is applied. Expert weights are the NVFP4 checkpoint dequantized."
    ),
    "past_key_values.layers.{i}.keys": (
        "K for every position [1, num_key_value_heads, N, head_dim]: k_proj, per-head k_norm (Gemma RMSNorm, "
        "x (1 + w)), then RoPE on the first rotary_dim = 64 dims only (half-split rotation within those 64; "
        "theta 1e7), the last 64 dims pass through."
    ),
    "past_key_values.layers.{i}.idx_keys": (
        "Lightning-indexer keys [1, 1, N, index_head_dim] (every layer): index_k_proj, index_k_norm (Gemma "
        "RMSNorm), RoPE on the first 64 dims like the attention keys. The state the indexer carries to the next "
        "token alongside K/V."
    ),
}

# HF_GOLDEN_SERVING=hopper: what the stored tensors are instead.
SERVING_NOTES = {
    "model.layers.{i}.self_attn.output": (
        "Attention output after o_proj, before the residual add. Block-sparse per KV head: query head h attends "
        "the blocks KV head h // 16's index head selected, causally, reading K / V from the e4m3 cache."
    ),
    "past_key_values.layers.{i}.keys": (
        "The e4m3 KV cache's K (scale 1), stored in bf16 (exact): k_proj, k_norm and RoPE (theta 1e7, dims "
        "[0:64]) in fp32, rounded straight to float8_e4m3fn (nearest even, saturating at 448)."
    ),
    "past_key_values.layers.{i}.values": "The e4m3 KV cache's V (scale 1): v_proj's bf16 rounded to float8_e4m3fn.",
    "past_key_values.layers.{i}.idx_keys": (
        "The e4m3 index-K cache (no scale): index_k_proj, index_k_norm and RoPE in fp32, rounded straight to "
        "float8_e4m3fn."
    ),
    "model.layers.{i}.self_attn.indexer.output": (
        "Key blocks each KV head may attend [1, num_key_value_heads, N, min(16, ceil(N / 128))], int64: index head g "
        "(one per KV head) scores block b as the max over its keys of e4m3(bf16(q_idx)) . e4m3(k_idx) (fp32), the "
        "query's own block forced in; top-k per head in descending score order, -1 where fewer blocks exist. Ties "
        "may come in any order: compare as sets."
    ),
    "model.layers.{i}.mlp.gate.output": (
        "Router logits [N, num_local_experts], fp32: the fp32 checkpoint router applied to the bf16 input in fp32, "
        "before sigmoid and the fp32 e_score_correction_bias."
    ),
}

FP8_MAX = 448.0


def fp8_e4m3(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """x rounded to float8_e4m3fn with scale 1, nearest even, saturating at +-448 (CUDA's __NV_SATFINITE), as
    ``dtype`` (exact)."""
    return x.float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(dtype)


def serving_indexer_forward(chunk_of, quantize: bool = True, per_head: bool = True):
    """MiniMaxM3VLIndexer.forward as vLLM runs it on Hopper (see the header): fp32 norm + RoPE, e4m3 index q / k,
    and with ``per_head`` one top-k per index head, [B, H_idx, S_q, topk]. quantize=False, per_head=False is
    upstream's function (rounding aside: fp32 norm output instead of the compute dtype's), [B, S_q, topk]. Scores
    are built ``chunk_of()`` query rows at a time, as chunked_indexer_forward."""
    import torch.nn.functional as F
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, past_key_values, position_ids):
        batch, q_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        idx_q = self.q_proj(hidden_states).view(batch, q_len, -1, self.head_dim)
        idx_q = self.q_norm(idx_q.float()).transpose(1, 2)
        idx_k = self.k_proj(hidden_states).view(batch, q_len, 1, self.head_dim)
        idx_k = self.k_norm(idx_k.float()).transpose(1, 2)
        cos, sin = (t[..., : self.head_dim].float() for t in position_embeddings)
        idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos, sin)
        idx_q = idx_q.to(dtype)
        idx_q, idx_k = (fp8_e4m3(idx_q, dtype), fp8_e4m3(idx_k, dtype)) if quantize else (idx_q, idx_k.to(dtype))
        if past_key_values is not None:
            idx_k = past_key_values.layers[self.layer_idx].update_index(idx_k)

        k_len = idx_k.shape[2]
        num_key_blocks = -(-k_len // self.block_size)
        pad = num_key_blocks * self.block_size - k_len
        k_float_t = idx_k.float().transpose(-1, -2)
        k_positions = torch.arange(k_len, device=idx_q.device)
        topk = min(self.topk_blocks, num_key_blocks)
        chunk = chunk_of() or q_len
        out = []
        for s in range(0, q_len, chunk):
            e = min(s + chunk, q_len)
            pos = position_ids[:, s:e]
            scores = torch.matmul(idx_q[:, :, s:e].float(), k_float_t)
            token_future = k_positions[None, None, None, :] > pos[:, None, :, None]
            scores = scores.masked_fill(token_future, float("-inf"))
            if pad:
                scores = F.pad(scores, (0, pad), value=float("-inf"))
            scores = scores.view(batch, self.num_heads, e - s, num_key_blocks, self.block_size)
            block_scores = scores.amax(dim=-1)  # [B, H_idx, rows, blocks]
            if not per_head:
                block_scores = block_scores.amax(dim=1, keepdim=True)
            del scores, token_future
            q_block = pos // self.block_size
            if self.local_blocks > 0:
                local = torch.arange(self.local_blocks, device=idx_q.device)
                local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)
                local_idx = local_idx[:, None].expand(-1, block_scores.shape[1], -1, -1)
                block_scores.scatter_(-1, local_idx, float("inf"))
            topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
            out.append(topk_indices.masked_fill(topk_scores == float("-inf"), -1))
        out = torch.cat(out, dim=2)
        return out if per_head else out[:, 0]

    return forward


def block_sparse_attention(q, k, v, blocks, position_ids, scaling, block_size, chunk):
    """q [B, H, S_q, D]; k, v [B, KV, S_k, D]; blocks [B, KV or 1, S_q, T] key-block ids, -1 = none -> [B, S_q, H, D].
    Query head h attends KV head h // (H / KV) over that head's selected blocks (a single selection: every head),
    causally. Scores and softmax in fp32, P rounded to q's dtype before PV, fp32 accumulation, normalized after, as
    vLLM's Triton block-sparse attention; ``chunk`` query rows at a time."""
    batch, n_heads, q_len, head_dim = q.shape
    n_kv, k_len = k.shape[1], k.shape[2]
    rep = n_heads // n_kv
    n_blocks = -(-k_len // block_size)
    k_positions = torch.arange(k_len, device=q.device)
    out = q.new_empty(batch, q_len, n_heads, head_dim)
    for g in range(n_kv):
        heads = slice(g * rep, (g + 1) * rep)
        sel = blocks[:, min(g, blocks.shape[1] - 1)]
        k_g, v_g = k[:, g, None].float(), v[:, g, None].float()
        for s in range(0, q_len, chunk):
            e = min(s + chunk, q_len)
            idx = sel[:, s:e]
            keep = torch.zeros(batch, e - s, n_blocks + 1, dtype=torch.bool, device=q.device)
            keep.scatter_(-1, idx.masked_fill(idx < 0, n_blocks), True)
            keep = keep[..., :n_blocks].repeat_interleave(block_size, dim=-1)[..., :k_len]
            keep &= k_positions[None, None, :] <= position_ids[:, s:e, None]
            scores = torch.matmul(q[:, heads, s:e].float(), k_g.transpose(-1, -2)) * scaling
            scores = scores.masked_fill(~keep[:, None], float("-inf"))
            p = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
            denom = p.sum(dim=-1, keepdim=True)
            o = torch.matmul(p.to(q.dtype).float(), v_g) / denom
            out[:, s:e, heads] = o.transpose(1, 2).to(q.dtype)
            del scores, p, keep
    return out


def serving_attention_forward(chunk_of, quantize: bool = True):
    """MiniMaxM3VLAttention.forward as vLLM runs it on Hopper (see the header): fp32 q / k norm + RoPE, e4m3 K / V
    into the cache and read back from it, block-sparse attention on the indexer's selection (per KV head when it
    returns [B, H_idx, S_q, T]). quantize=False with a shared selection is upstream's function."""
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        dtype = hidden_states.dtype
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape).float()).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape).float()).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = (t.float() for t in position_embeddings)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.to(dtype)
        if quantize:
            key_states, value_states = fp8_e4m3(key_states, dtype), fp8_e4m3(value_states, dtype)
        else:
            key_states = key_states.to(dtype)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            k_len, q_len = key_states.shape[2], query_states.shape[2]
            position_ids = torch.arange(k_len - q_len, k_len, device=query_states.device)
        position_ids = (position_ids if position_ids.ndim > 1 else position_ids.unsqueeze(0)).expand(
            query_states.shape[0], -1
        )
        blocks = self.indexer(hidden_states, position_embeddings, past_key_values, position_ids)
        if blocks.dim() == 3:  # one selection for every head
            blocks = blocks[:, None]
        if blocks.shape[1] not in (1, key_states.shape[1]):
            raise SystemExit(f"{blocks.shape[1]} index heads for {key_states.shape[1]} KV heads: vLLM needs them 1:1")
        attn_output = block_sparse_attention(
            query_states,
            key_states,
            value_states,
            blocks,
            position_ids,
            self.scaling,
            self.indexer.block_size,
            chunk_of() or query_states.shape[2],
        )
        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), None

    return forward


def check_serving_attention(text_config) -> None:
    """The serving attention and indexer re-implement upstream's around the changes in the header; fail loudly if the
    installed transformers computes something else. On a random fp32 tiny layer (16-key blocks, 4 kept, so the
    selection cuts), with quantization off and the selection shared, they must give upstream's block ids exactly and
    its eager attention output to fp32 rounding, for chunk sizes that do and do not divide the sequence; and with
    identical index heads, the per-head selections must each equal the shared one."""
    from transformers import DynamicCache
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLAttention,
        MiniMaxM3VLRotaryEmbedding,
    )

    cfg = copy.deepcopy(text_config)
    cfg.index_block_size, cfg.index_topk_blocks = 16, 4
    cfg._attn_implementation = "eager"
    gen = torch.Generator().manual_seed(0)
    attn = MiniMaxM3VLAttention(cfg, layer_idx=0).float()
    for p in attn.parameters():
        p.data = torch.randn(p.shape, generator=gen) * 0.5
    n = 300
    x = torch.randn(1, n, cfg.hidden_size, generator=gen)
    pos = torch.arange(n)[None]
    pe = MiniMaxM3VLRotaryEmbedding(cfg)(x, pos)
    upstream_attn, upstream_indexer = type(attn).forward, type(attn.indexer).forward

    def run(attn_fn, indexer_fn):
        type(attn).forward, type(attn.indexer).forward = attn_fn, indexer_fn
        try:
            cache = DynamicCache(config=cfg)
            with torch.no_grad():
                out, _ = attn(x, pe, None, past_key_values=cache, position_ids=pos)
                blocks = indexer_fn(attn.indexer, x, pe, None, pos)
            return out, blocks
        finally:
            type(attn).forward, type(attn.indexer).forward = upstream_attn, upstream_indexer

    ref, ref_blocks = run(upstream_attn, upstream_indexer)
    for chunk in (64, 77, 4096):
        out, blocks = run(
            serving_attention_forward(lambda chunk=chunk: chunk, quantize=False),
            serving_indexer_forward(lambda chunk=chunk: chunk, quantize=False, per_head=False),
        )
        if not torch.equal(blocks, ref_blocks) or not torch.allclose(out, ref, rtol=1e-4, atol=1e-5):
            raise SystemExit(
                f"serving attention / indexer (chunk {chunk}, unquantized, shared selection) differ from this "
                f"transformers' MiniMaxM3VLAttention: max|diff| {(out - ref).abs().max():.3g}, blocks equal "
                f"{torch.equal(blocks, ref_blocks)}; upstream changed: update serving_*_forward"
            )
    heads, dim = attn.indexer.num_heads, attn.indexer.head_dim
    attn.indexer.q_proj.weight.data = attn.indexer.q_proj.weight.data[:dim].repeat(heads, 1)
    _, shared = run(upstream_attn, upstream_indexer)
    with torch.no_grad():
        per_head = serving_indexer_forward(lambda: 77, quantize=False)(attn.indexer, x, pe, None, pos)
    if not all(torch.equal(per_head[:, h], shared) for h in range(heads)):
        raise SystemExit("serving indexer: identical index heads give per-head selections unlike the shared one")


def marlin_flush_scales(sd: dict[str, torch.Tensor]) -> int:
    """Zero the NVFP4 block scales vLLM's Marlin MoE (Hopper) loses: it multiplies a layer's e4m3 scales (w1 + w3
    together, w2 apart) by the power of two sf that brings their max into [224, 448], then re-encodes them as S0E5M3,
    which holds none below 2: a scale with scale * sf * 2^7 < 2 becomes 0 (marlin_utils_fp4.py). Returns how many."""
    flushed = 0
    for group in (("w1", "w3"), ("w2",)):
        keys = [k for k in sd if k.endswith(".weight_scale") and k.split(".")[-2] in group]
        if not keys:
            continue
        top = max(sd[k].float().max().item() for k in keys)
        sf = 2.0 ** math.floor(math.log2(FP8_MAX / top)) if 0 < top < FP8_MAX else 1.0
        for k in keys:
            s = sd[k].float()
            flush = (s > 0) & (s * sf * 2**7 < 2)
            if flush.any():
                flushed += int(flush.sum())
                sd[k] = s.masked_fill(flush, 0).to(sd[k].dtype)
    return flushed


def nvfp4_dequantize(packed, scale, global_scale, dtype: torch.dtype) -> torch.Tensor:
    """compressed-tensors nvfp4-pack-quantized -> ``dtype``: E2M1 pairs (low nibble first) [out, in / 2] uint8 x
    float8_e4m3fn scale [out, in / 16] / global_scale, in fp32, cast once."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize

    values = Fp8Dequantize(None)._unpack_fp4(packed)
    group = values.shape[-1] // scale.shape[-1]
    scale = scale.to(torch.float32).repeat_interleave(group, dim=-1)
    return (values * scale / global_scale.to(torch.float32)).to(dtype)


class MiniMaxM31FlashGolden(MiniMaxM3Golden):
    hf_repo = "MiniMaxAI/MiniMax-M3.1-Flash"
    prefix = (
        "]~!b[]~b]system\nYour model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026. "
        "Founded in early 2022, MiniMax is a global AI foundation model company committed to advancing the frontiers "
        "of AI towards AGI.\n\n<thinking_instructions>\nYou have a thinking capability that allows you to reason step "
        "by step before responding. When thinking is enabled, wrap your reasoning in <mm:think></mm:think> tags "
        "before your response. When thinking is disabled, begin your response directly after the </mm:think> prefix. "
        "When thinking is adaptive, decide on your own whether to think for the current turn.\nCurrent thinking mode: "
        "adaptive. You are encouraged to think for complex decision-making, multi-step reasoning, or when analyzing "
        "function/tool results.\n</thinking_instructions>\n\n<model_id>8bb8149fe</model_id>[e~[\n]~b]developer\nYou "
        "are a helpful assistant.[e~[\n]~b]user\nContinue this WikiText-2 dump verbatim: a sequence of Wikipedia "
        "articles, one after another, each starting with a = Title = line, in raw formatting ( @-@ , = = headings "
        "= = ).[e~[\n]~b]ai\n</mm:think>"
    )
    recipe = dataclasses.replace(MiniMaxM3Golden.recipe, notes=NOTES)

    def __init__(self):
        super().__init__()
        # HF_GOLDEN_SERVING=hopper: emulate vLLM on Hopper (see the header); "" (default): the dequantized model.
        self.serving = os.environ.get("HF_GOLDEN_SERVING", "")
        if self.serving not in ("", "hopper"):
            raise SystemExit(f"HF_GOLDEN_SERVING={self.serving!r}: only 'hopper' is emulated")
        self.attn_chunk = 0
        if self.serving:
            from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
                MiniMaxM3VLAttention,
                MiniMaxM3VLIndexer,
            )

            check_serving_attention(self.tiny_config().text_config)
            # self.attn_chunk is set from --attn-chunk once the args are parsed
            MiniMaxM3VLAttention.forward = serving_attention_forward(lambda: self.attn_chunk)
            MiniMaxM3VLIndexer.forward = serving_indexer_forward(lambda: self.attn_chunk)
            self.recipe = dataclasses.replace(self.recipe, notes={**NOTES, **SERVING_NOTES})
            print("[serving] hopper: e4m3 KV + index caches, per-head block selection, fp32 router", flush=True)

    def trace_name(self, args):
        return super().trace_name(args) + (f"-serving-{self.serving}" if self.serving else "")

    def build(self, config, dtype):
        model = super().build(config, dtype)
        if self.serving:  # the router (weight and e_score_correction_bias) stays fp32, as vLLM's GateLinear
            plan = model._get_dtype_plan
            model._get_dtype_plan = lambda d: {**plan(d), "mlp.gate": torch.float32}
        return model

    def generate(self, args):
        self.dtype = getattr(torch, args.compute_dtype)
        self.input_global_scales = set()
        self.marlin_flushed = 0
        # Serving attention and indexer: --attn-chunk query rows at a time, 4096 by default (whole [4, N, N] fp32
        # indexer scores would be ~51 GB at isl 56320).
        self.attn_chunk = args.attn_chunk or 4096
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        meta["quantized_weights"] = {
            "mxfp8": (
                "attention q/k/v/o, indexer q/k, dense MLP, shared expert: float8_e4m3fn x 2^(e8m0 - 127) per 1x32 "
                "block, by transformers' Fp8Dequantize"
            ),
            "nvfp4": (
                "routed experts: e2m1 x float8_e4m3fn scale per 16 / weight_global_scale (compressed-tensors "
                "nvfp4-pack-quantized)"
            ),
            "dequantized_to": args.compute_dtype,
        }
        meta["nvfp4_input_global_scales"] = sorted(self.input_global_scales)
        if self.serving:
            meta["serving"] = {
                "emulates": (
                    "vLLM PR #63232 (branch minimax-m31-flash, a0ea9410) on Hopper, TP8: --kv-cache-dtype fp8, "
                    "indexer_kv_dtype fp8, --moe-backend marlin"
                ),
                "kv_cache": "float8_e4m3fn, scale 1.0; K from fp32 norm + RoPE; prefill reads K / V from the cache",
                "indexer": "index q / k float8_e4m3fn, no scale; one top-k per KV head (no max over index heads)",
                "router": "fp32 weight, bias and logits",
                "weight_only": "MXFP8 linears (Marlin W8A16) and NVFP4 experts (Marlin W4A16): bf16 activations",
                "marlin_flushed_nvfp4_scales": self.marlin_flushed,
                "attn_chunk": self.attn_chunk,
            }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def convert(self, sd, prefix, module):
        """MXFP8 (``weight`` + uint8 ``weight_scale_inv``) and NVFP4 (``weight_packed`` + ``weight_scale`` +
        ``weight_global_scale``) -> ``weight`` in the compute dtype, under the checkpoint's names; then MiniMax-M3's
        renaming and gate/up fusion. The unquantized --selftest checkpoint passes through."""
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        dq = Fp8Dequantize(None)
        if self.serving:
            self.marlin_flushed += marlin_flush_scales(sd)
        for scale in [k for k in sd if k.endswith(".weight_scale_inv")]:
            weight = scale.removesuffix("_scale_inv")
            sd[weight] = dq._dequantize_one(sd[weight], sd.pop(scale), output_dtype=self.dtype)
        for packed in [k for k in sd if k.endswith(".weight_packed")]:
            weight = packed.removesuffix("_packed")
            sd[weight] = nvfp4_dequantize(
                sd.pop(packed), sd.pop(f"{weight}_scale"), sd.pop(f"{weight}_global_scale"), self.dtype
            )
            self.input_global_scales.add(sd.pop(weight.removesuffix("weight") + "input_global_scale").item())
        return super().convert(sd, prefix, module)

    def selftest(self, args, compute, isl=64):
        if self.serving:  # its reference is one upstream forward, which serving changes on purpose
            raise SystemExit("--selftest checks the plain model: unset HF_GOLDEN_SERVING")
        return super().selftest(args, compute, isl)

    def tiny_config(self):
        config = super().tiny_config()
        text = config.text_config
        text.layer_types = ["minimax_m3_sparse"] * text.num_hidden_layers  # M3's dense MLPs on 0-1 stay
        text.rope_parameters["rope_theta"] = 1e7
        return config


if __name__ == "__main__":
    sys.exit(MiniMaxM31FlashGolden().main())
