#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill goldens for deepseek-ai/DeepSeek-V4.1-Flash-1017, layer by layer (engine.py), from DeepSeek's own reference
implementation shipped in the repo (inference/model.py, engram.py; pinned with the checkpoint's revision).
transformers has no deepseek_v41 (huggingface/transformers#48755, open), so this is the one first-party
implementation; the engine drives its Blocks one at a time. Text only: the vision tower and the DSpark draft layers
(mtp.*) are never built nor read.

Stored tensors (N = isl, i = layer; 40 layers, all MoE (384 routed experts, top 6, + 1 shared); attention: a
128-token sliding window on every layer, plus compressed KV on layers 2-39: ratio 2 on 2-19, ratio 1 on 20-39;
Engram n-gram memory added to the residual streams before layers 1 and 14):

  hidden/embed.safetensors
    model.embed_tokens.output                          [1, N, 5120]       bf16  embeddings; layer 0 copies them
                                                                                into hc_mult = 4 residual streams
  hidden/layer_XX.safetensors
    model.layers.{i}.output                            [1, N, 4, 5120]    bf16  all     the 4 residual streams
    model.layers.{i}.engram.output                     [1, N, 4, 5120]    bf16  1, 14   --sublayers: streams after
                                                                                        the Engram write
    model.layers.{i}.attn_norm.output                  [1, N, 5120]       bf16  all     --sublayers: attention input
                                                                                        (hc_pre collapse + RMSNorm)
    model.layers.{i}.attn.output                       [1, N, 5120]       bf16  all     --sublayers: after wo_b
    model.layers.{i}.ffn_norm.output                   [1, N, 5120]       bf16  all     --sublayers: MoE input
    model.layers.{i}.ffn.shared_experts.output         [N, 5120]          bf16  all     --sublayers
    model.layers.{i}.ffn.output                        [1, N, 5120]       bf16  all     --sublayers: routed + shared
  cache/layer_XX.safetensors
    model.layers.{i}.pre_mix                           [1, N, 4]          fp32  all     the stream mix the NEXT
                                                                                        layer's attention collapses
                                                                                        with (with .output, the state
                                                                                        a layer starts from)
    model.layers.{i}.attn.kv_norm.output               [1, N, 512]        bf16  all     the shared K=V of every
                                                                                        position, before RoPE
    past_key_values.layers.{i}.window_kv               [1, 128, 512]      bf16  all     the sliding-window ring, post
                                                                                        RoPE: slot p % 128 holds p
    past_key_values.layers.{i}.compressed_kv           [1, N // r, 512]   bf16  2, 8, 14, 20  compressed KV, post RoPE
                                                                                        (read by the following layers
                                                                                        of the same ratio)
    past_key_values.layers.{i}.index_k                 [1, N // r, 128]   bf16  2, 8, 14, 20  indexer keys, post RoPE
    past_key_values.layers.{i}.compressor_{kv,score}_state [1, N % r, 512] fp32 2, 8, 14  the incomplete group
                                                                                        (only when N % 2 != 0)
    model.layers.{i}.attn.indexer.output               [1, N, 512]        int32 2, 8, 14, 20, 24, 28, 32, 36  chosen
                                                                                        compressed positions + N, -1
                                                                                        none (read by the following
                                                                                        layers)
    model.layers.{i}.attn.indexer.candidate_blocks     [1, N, ceil(N / 8)] bool 20      the first-level candidate
                                                                                        blocks of 8 that indexers 24-36
                                                                                        choose inside
    model.layers.{i}.engram.hash_ids                   [1, N, 24]         int32 1, 14   the n-gram table rows read
    model.layers.{i}.ffn.gate.output                   [N, 384]           fp32  all     router logits, before
                                                                                        sqrt(softplus(.))
    model.layers.{i}.ffn.gate.top_k_index              [N, 6]             int32 all     chosen experts, ascending
    model.layers.{i}.ffn.gate.top_k_weights            [N, 6]             fp32  all     their weights
  hidden/final.safetensors
    model.last_hidden_state                            [1, N, 5120]       bf16  norm(hc_pre(streams, last pre_mix))
    logits                                             [1, 1, 129280]     fp32  last position, head (untied, fp32)

What is model-specific, and why it is handled here:

  * The reference code (inference/model.py, engram.py, convert.py; vision.py and image_processor.py only because
    model.py imports them) is downloaded at the checkpoint's own revision. Its kernel.py (TileLang, CUDA only) is
    replaced by the torch functions below, the reference's own torch (attention, MoE, Engram, compressor, indexer)
    runs unchanged, and model.py's sha256 goes to metadata.json.
  * Numerics: the unquantized model, as the DeepSeek-V4-Flash golden, in bf16 only (the reference's Engram lookup
    returns bf16 rows whatever the model dtype, so --compute-dtype float32 is refused). Weights are dequantized (fp8
    e4m3 x e8m0 per 32x32 block; experts fp4 e2m1 x e8m0 per 1x32, unpacked by transformers' Fp8Dequantize and
    checked at run time against convert.py's own FP4_TABLE decode on the first expert: metadata.json
    "dequant_check"), so every Linear
    takes the reference's plain F.linear path; the reference's in-graph activation rounding (act_quant /
    fp4_act_quant: fp8 window KV and attention output, fp4 indexer q / k and compressed KV) is a no-op. sparse_attn
    is computed exactly (fp32 softmax with the per-head sink, fp32 PV; the TileLang kernel rounds probabilities to
    bf16) and hc_split_sinkhorn is the kernel's arithmetic in torch; on CUDA, with tilelang installed, both are
    checked against DeepSeek's kernels at startup (metadata.json "kernel_check").
  * Cross-layer state: the compressed KV and index keys come from kv sources (2, 8, 14, 20), the top-k positions from
    index sources (2, 8, ..., 36), the candidate blocks from layer 20; the reference hands them down through its
    module-global shared_attn, which survives each layer's release here because the next layers hold references.
    The stream mix (pre_mix) is carried as state.carry.
  * Engram: two n-gram hash tables, fp8 [~384M, 256] (~98 GB each) in their own shards (47, 48). They are never
    loaded: each is memory-mapped from its shard and only the rows the hashed n-grams hit ([N, 24] per layer) are read
    and dequantized as the reference's ParallelEngramEmbedding does. The hashes come from the reference's
    NgramHashState (compressed tokenizer vocab, per-layer multipliers, prime buckets) on the prefilled ids. The layer
    stages exclude engram.* (a tuple of prefixes), and its small weights are read from that shard too.
  * Memory at N = 56320 (H100): one layer is ~30 GB of bf16 weights. Per query row everything is independent, so the
    indexer scores ([N, 32 heads, N / r]: ~200 GB whole) and the sparse attention gather run --attn-chunk (default
    1024) query rows at a time, and hc_post ([N, 4, 4, 5120] fp32: ~18 GB whole) 4096 tokens at a time; the --selftest
    reference runs them whole, so this is tested. The gate's bias_vl (image tokens only) is dropped.

    python generate_golden_deepseek_v4_1_flash.py --selftest                     # first: tiny model, every path
    python generate_golden_deepseek_v4_1_flash.py --isl 1024 --num-layers 3 --root /tmp/smoke  # Engram, ratio 2
    python generate_golden_deepseek_v4_1_flash.py --isl 56320
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.util
import json
import re
import struct
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import TEXT, cached_file, resolve_revision  # noqa: E402
from engine import LayerwiseGolden, Recipe, RunState  # noqa: E402

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
CODE_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"  # for --selftest: the reference code of REPO at this commit
CODE_FILES = ("model.py", "engram.py", "convert.py", "vision.py", "image_processor.py", "kernel.py")
HC_POST_CHUNK = 4096  # tokens per hc_post call (header: memory)
FP8_BLOCK = 32

# Checkpoint weights stored fp8 (with a 32x32 .scale) and fp4 (int8 pairs, 1x32 .scale), for --selftest's checkpoint.
FP8_WEIGHTS = re.compile(
    r"\.(attn\.(wq_a|wq_b|wkv|wo_a|wo_b)|attn\.indexer\.wq_b|ffn\.shared_experts\.w[123]|engram\.wkv)\.weight$"
)
FP4_WEIGHTS = re.compile(r"\.ffn\.experts\.\d+\.w[123]\.weight$")
# ModelArgs (inference/config.json) <-> config.json text_config, checked against each other at build.
ARG_KEYS = {
    "dim": "hidden_size",
    "vocab_size": "vocab_size",
    "n_heads": "num_attention_heads",
    "head_dim": "head_dim",
    "rope_head_dim": "qk_rope_head_dim",
    "q_lora_rank": "q_lora_rank",
    "o_lora_rank": "o_lora_rank",
    "o_groups": "o_groups",
    "window_size": "sliding_window",
    "moe_inter_dim": "moe_intermediate_size",
    "n_routed_experts": "n_routed_experts",
    "n_activated_experts": "num_experts_per_tok",
    "route_scale": "routed_scaling_factor",
    "swiglu_limit": "swiglu_limit",
    "score_func": "scoring_func",
    "norm_eps": "rms_norm_eps",
    "compress_ratios": "compress_ratios",
    "compress_rope_theta": "compress_rope_theta",
    "rope_theta": "rope_theta",
    "kv_source_layers": "kv_source_layer_ids",
    "index_source_layers": "index_source_layer_ids",
    "candidate_source_layer": "candidate_source_layer_id",
    "candidate_topk_blocks": "candidate_topk_blocks",
    "candidate_block_size": "candidate_block_size",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "engram_layer_ids": "engram_layer_ids",
    "engram_num_embeddings": "engram_num_embeddings",
    "engram_max_ngram_size": "engram_max_ngram_size",
    "engram_vocab_size": "engram_vocab_size",
    "engram_n_heads": "engram_n_heads",
    "engram_head_dim": "engram_head_dim",
    "engram_compressed_vocab_size": "engram_compressed_vocab_size",
}

NOTES = {
    "model.embed_tokens.output": "Token embeddings (no scale). Layer 0 copies them into hc_mult residual streams.",
    "model.layers.{i}.output": (
        "The hc_mult residual streams after the layer [1, N, hc_mult, dim]: attention and MoE each collapse the "
        "streams (hc_pre, by the previous sub-block's pre mix), and hc_post writes their output back as post * out "
        "+ comb @ streams (Sinkhorn-normalized comb). With model.layers.{i}.pre_mix, the next layer's input."
    ),
    "model.layers.{i}.pre_mix": (
        "The pre mix [1, N, hc_mult] this layer's MoE site computed, fp32: the next layer's attention collapses the "
        "streams with it (the final norm reads the last layer's). Layer 0 starts from the one-hot [1, 0, 0, 0]."
    ),
    "model.layers.{i}.engram.output": (
        "Streams after the Engram write (--sublayers): h + sigmoid(signed sqrt(normalized h . key)) * value, per "
        "stream, with key / value from wkv(the dequantized n-gram rows)."
    ),
    "model.layers.{i}.attn_norm.output": "Attention input (--sublayers): RMSNorm(hc_pre(streams, pre_mix)).",
    "model.layers.{i}.attn.output": (
        "Attention output after wo_b (--sublayers): sparse attention over the window and the chosen compressed "
        "positions with a per-head sink, inverse RoPE on the output's rope dims, grouped low-rank wo_a, wo_b."
    ),
    "model.layers.{i}.ffn_norm.output": "MoE input (--sublayers): RMSNorm(hc_pre(streams, the attention's pre mix)).",
    "model.layers.{i}.ffn.shared_experts.output": "The shared expert (--sublayers), [N, dim].",
    "model.layers.{i}.ffn.output": (
        "Routed experts (weighted, fp32 sum) + the shared expert, before hc_post (--sublayers)."
    ),
    "model.layers.{i}.attn.kv_norm.output": (
        "The layer's own K=V for every position [1, N, head_dim]: kv_norm(wkv(x)), before RoPE (the window cache "
        "holds it after RoPE on its last rope_head_dim dims)."
    ),
    "past_key_values.layers.{i}.window_kv": (
        "The sliding-window ring [1, window_size, head_dim] after prefill, post RoPE: slot p % window_size holds "
        "position p (the last window_size positions)."
    ),
    "past_key_values.layers.{i}.compressed_kv": (
        "kv sources only: compressed KV [1, N // r, head_dim], one latent per r tokens (softmax-gated pooling for r "
        "> 1, a projection for r = 1), RMSNorm, RoPE at compress_rope_theta / YaRN on group j's first position "
        "j * r. The following layers of the same ratio attend to this one."
    ),
    "past_key_values.layers.{i}.index_k": (
        "kv sources only: indexer keys [1, N // r, index_head_dim] = k_norm(wk(compressor latent before RoPE)), then "
        "RoPE. The following index sources score against them."
    ),
    "past_key_values.layers.{i}.compressor_kv_state": "kv sources with r > 1: kv of the incomplete group (N % r).",
    "past_key_values.layers.{i}.compressor_score_state": "kv sources with r > 1: gate scores of the incomplete group.",
    "model.layers.{i}.attn.indexer.output": (
        "index sources only: the index_topk compressed positions each query attends to, ascending, shifted by N (the "
        "window KV comes first in the attention's KV), -1 where fewer are reachable. int32. The following layers "
        "reuse it."
    ),
    "model.layers.{i}.attn.indexer.candidate_blocks": (
        "candidate source only: which blocks of candidate_block_size compressed positions each query keeps "
        "(candidate_topk_blocks best by max score, its newest block always); later index sources choose inside them."
    ),
    "model.layers.{i}.engram.hash_ids": (
        "Engram layers: the table rows [1, N, (max_ngram - 1) * n_heads] the 2..4-grams ending at each position hash "
        "to (compressed token ids, per-layer multipliers, xor, a prime bucket per (n-gram, head), + offsets)."
    ),
    "model.layers.{i}.ffn.gate.output": (
        "Router logits [N, n_routed_experts] = fp32(ffn_norm output) @ fp32(gate.weight).T, before sqrt(softplus). "
        "Selection: top-k of the scores + gate.bias; weights from the unbiased scores, renormalized, x route_scale."
    ),
    "model.layers.{i}.ffn.gate.top_k_index": "The chosen experts per token, ascending by id. int32.",
    "model.layers.{i}.ffn.gate.top_k_weights": "Weights of top_k_index, same order, fp32.",
    "model.last_hidden_state": "norm(hc_pre(last streams, last pre_mix)).",
    "logits": "Last position only: head (untied, fp32 weight), fp32.",
}


# ---- the reference code, and the kernels it imports ---------------------------------------------------------------

_SPARSE_CHUNK = [0]  # query rows per sparse_attn step; 0: all at once (the --selftest reference)


def _act_quant_noop(x, *args, **kwargs):
    """act_quant / fp4_act_quant: in-place activation rounding in the reference, a no-op here (header: numerics).
    The not-in-place form only feeds fp8 / fp4 GEMMs, which bf16 weights never reach."""
    inplace = kwargs.get("inplace", args[-1] if args and isinstance(args[-1], bool) else False)
    if not inplace:
        raise SystemExit("act_quant out of place: a quantized GEMM was reached, but every weight is dequantized")
    return x


def _no_gemm(*args, **kwargs):
    raise SystemExit("fp8_gemm / fp4_gemm reached, but every weight is dequantized")


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.sparse_attn in torch: each query's heads attend to the kv rows topk_idxs names (-1: none) plus a per-head
    sink logit; a row with no valid index gives 0, as the kernel. q [b, m, h, d], kv [b, n, d], topk_idxs [b, m, k].
    Exact softmax in fp32 (the kernel rounds the probabilities to bf16 before PV)."""
    b, m, _, _ = q.shape
    out = torch.empty_like(q)
    step = _SPARSE_CHUNK[0] or m
    sink = attn_sink.float()[None, None, :, None]
    batch = torch.arange(b, device=q.device)[:, None, None]
    for s in range(0, m, step):
        e = min(s + step, m)
        idx = topk_idxs[:, s:e].to(q.device).long()
        rows = kv[batch, idx.clamp_min(0)].float()  # [b, c, k, d]
        scores = torch.einsum("bchd,bckd->bchk", q[:, s:e].float(), rows) * softmax_scale
        scores = scores.masked_fill((idx < 0)[:, :, None, :], -torch.inf)
        peak = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)  # the kernel's finite floor
        p = (scores - peak).exp()
        denom = p.sum(dim=-1, keepdim=True) + (sink - peak).exp()
        out[:, s:e] = (torch.einsum("bchk,bckd->bchd", p, rows) / denom).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """kernel.hc_split_sinkhorn in torch, op for op: pre = sigmoid + eps, post = 2 sigmoid, comb = row softmax + eps,
    then alternating column / row normalization (column first), each with + eps in the denominator."""
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def torch_kernels() -> types.ModuleType:
    kernel = types.ModuleType("kernel")
    kernel.act_quant = kernel.fp4_act_quant = _act_quant_noop
    kernel.fp8_gemm = kernel.fp4_gemm = _no_gemm
    kernel.sparse_attn, kernel.hc_split_sinkhorn = sparse_attn, hc_split_sinkhorn
    return kernel


def fetch_code(repo: str, revision: str) -> Path:
    """The reference's inference/ directory at ``revision``, from the HF cache or the hub."""
    paths = [Path(hf_hub_download(repo, f"inference/{f}", revision=revision)) for f in CODE_FILES]
    return paths[0].parent


def load_reference(code: Path) -> types.ModuleType:
    """Import inference/model.py (as "model", with engram / vision / image_processor beside it) over torch_kernels().
    One reference per process: model.py keeps module globals (world_size, shared_attn, default_dtype)."""
    loaded = sys.modules.get("model")
    if loaded is not None:
        if Path(loaded.__file__).resolve() != (code / "model.py").resolve():
            raise SystemExit(f"another module named 'model' is imported ({loaded.__file__})")
        return loaded
    sys.modules["kernel"] = torch_kernels()
    sys.path.insert(0, str(code))
    return importlib.import_module("model")


def check_kernels(code: Path, device: torch.device) -> str:
    """sparse_attn and hc_split_sinkhorn against DeepSeek's TileLang kernels, on random inputs shaped like the model's
    (head_dim 512, -1 indices, a row with none)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(1, 96, 16, 512, generator=g).to(device, torch.bfloat16)
    kv = torch.randn(1, 160, 512, generator=g).to(device, torch.bfloat16)
    sink = torch.randn(16, generator=g).to(device)
    idx = torch.randint(-1, 160, (1, 96, 192), generator=g, dtype=torch.int32).to(device)
    idx[:, 0] = -1
    mixes = torch.randn(1, 64, 24, generator=g).to(device)
    scale, base = torch.rand(3, generator=g).to(device), torch.randn(24, generator=g).to(device)
    try:  # an unavailable kernel (no tilelang, a JIT or launch failure) is recorded, not fatal: torch runs either way
        spec = importlib.util.spec_from_file_location("deepseek_v41_kernel", code / "kernel.py")
        real = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(real)
        want = real.sparse_attn(q, kv, sink, idx, 512**-0.5).float()
        want_hc = real.hc_split_sinkhorn(mixes, scale, base)
    except Exception as e:
        return f"not run ({type(e).__name__}: {str(e)[:200]})"
    _SPARSE_CHUNK[0] = 32
    got = sparse_attn(q, kv, sink, idx, 512**-0.5).float()
    attn_err = ((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).item()
    hc_err = max((a - b).abs().max().item() for a, b in zip(hc_split_sinkhorn(mixes, scale, base), want_hc))
    if attn_err > 1e-2 or hc_err > 1e-5:
        raise SystemExit(
            f"torch kernels differ from DeepSeek's: sparse_attn RMSE ratio {attn_err:.2e}, sinkhorn {hc_err:.2e}"
        )
    return f"match DeepSeek's TileLang kernels (sparse_attn RMSE ratio {attn_err:.2e}, hc_split_sinkhorn {hc_err:.1e})"


# ---- weights --------------------------------------------------------------------------------------------------------


def dequantize(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """fp8 e4m3 [out, in] or fp4 e2m1 pairs [out, in / 2] (int8, low nibble first, unpacked by transformers'
    Fp8Dequantize) x its e8m0 block scale, the block shape read off the scale's ([32, 32] / [1, 32] here): convert.py's
    own block expansion. Exact: an fp8 / fp4 value times a power of two."""
    if weight.dtype in (torch.int8, torch.uint8):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize

        values = Fp8Dequantize(None)._unpack_fp4(weight.view(torch.uint8)).float()
    else:
        values = weight.float()
    bo, bi = values.shape[0] // scale.shape[0], values.shape[1] // scale.shape[1]
    blocks = values.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * scale.float()[:, None, :, None]
    return blocks.flatten(2, 3).flatten(0, 1).to(dtype)


def fp4_table_decode(convert, packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """convert.py's own fp4 decode (its FP4_TABLE, low nibble first, as cast_e2m1fn_to_e4m3fn unpacks) x the scale."""
    x = packed.view(torch.uint8).cpu()
    values = torch.stack([convert.FP4_TABLE[(x & 0x0F).long()], convert.FP4_TABLE[(x >> 4).long()]], dim=-1).flatten(1)
    return values * scale.cpu().float().repeat_interleave(values.shape[1] // scale.shape[1], dim=1)


def check_dequant(convert, packed: torch.Tensor, scale: torch.Tensor) -> str:
    if not torch.equal(dequantize(packed, scale, torch.float32).cpu(), fp4_table_decode(convert, packed, scale)):
        raise SystemExit("fp4 dequantization differs from convert.py's FP4_TABLE decode: update dequantize")
    return "fp4 experts bit-identical to convert.py's FP4_TABLE decode (first expert)"


def safetensors_entries(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


class HostEngramTable(torch.nn.Module):
    """ParallelEngramEmbedding (world size 1) over a memory map of its shard: only the rows looked up are read. The
    same arithmetic: fp8 rows x their e8m0 scale per block of 32, to bf16."""

    def __init__(self, path: Path, name: str, block_size: int):
        super().__init__()
        start, entries = safetensors_entries(path)
        self.maps = {}
        for part, dtype in (("weight", "F8_E4M3"), ("scale", "F8_E8M0")):
            e = entries[f"{name}.{part}"]
            assert e["dtype"] == dtype, (name, part, e["dtype"])
            a, b = e["data_offsets"]
            self.maps[part] = np.memmap(path, dtype=np.uint8, mode="r", offset=start + a, shape=tuple(e["shape"]))
            assert b - a == int(np.prod(e["shape"]))
        self.block_size, self.rows = block_size, entries[f"{name}.weight"]["shape"][0]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        flat = indices.reshape(-1).cpu().numpy()
        rows, inverse = np.unique(flat, return_inverse=True)
        assert rows.min() >= 0 and rows.max() < self.rows, "engram hash outside its table"
        dev = indices.device
        weight = torch.from_numpy(np.ascontiguousarray(self.maps["weight"][rows])).view(torch.float8_e4m3fn).to(dev)
        scale = torch.from_numpy(np.ascontiguousarray(self.maps["scale"][rows])).view(torch.float8_e8m0fnu).to(dev)
        inverse = torch.from_numpy(inverse).to(dev)
        values = weight[inverse].float().unflatten(-1, (-1, self.block_size)) * scale[inverse].float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)


# ---- per-instance replacements for the memory at long ISL (header: memory) ----------------------------------------


def chunked_indexer_forward(ds, self, x, qr, latent, start_pos, offset, chunk):
    """Indexer.forward (prefill, start_pos 0), the scores and top-k ``chunk`` query rows at a time: each row's scores,
    causal mask, candidate blocks and top-k depend on that row only. Everything else as the reference, line for line."""
    if start_pos != 0:
        raise SystemExit("chunked indexer: prefill only")
    assert self.freqs_cis is not None
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, seqlen
    if self.owns_k and latent is not None:
        freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio]
        k = self.k_norm(self.wk(latent))
        ds.apply_rotary_emb(k[..., -rd:], freqs)
        ds.fp4_act_quant(k, ds.fp4_block_size, True)
        self.k_cache[:bsz, : k.size(1)] = k
    if self.owns_k:
        ds.shared_attn.index_k = self.k_cache
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
    ds.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    ds.fp4_act_quant(q, ds.fp4_block_size, True)
    index_k = ds.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    width, topk = index_k.size(1), min(self.index_topk, end_pos // ratio)
    positions = torch.arange(width, device=x.device)
    out = torch.empty(bsz, seqlen, topk, dtype=torch.int32, device=x.device)
    candidates = (
        torch.empty(bsz, seqlen, width, dtype=torch.bool, device=x.device) if self.is_candidate_source else None
    )
    for s in range(0, seqlen, chunk):
        e = min(s + chunk, seqlen)
        index_score = torch.einsum("bshd,btd->bsht", q[:, s:e], index_k)
        index_score = (index_score.relu_() * weights[:, s:e].unsqueeze(-1)).sum(dim=2)
        compress_lens = (torch.arange(s + 1, e + 1, device=x.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(positions >= compress_lens, -torch.inf)
        if self.is_candidate_source:
            candidates[:, s:e] = ds.select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~ds.shared_attn.candidates[:, s:e], -torch.inf)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        out[:, s:e] = torch.where(idxs < compress_lens, idxs + offset, -1).int()
    if self.is_candidate_source:
        ds.shared_attn.candidates = candidates
    return out


def chunked_hc_post(ds, self, x, residual, post, comb, chunk=HC_POST_CHUNK):
    """Block.hc_post, ``chunk`` tokens at a time (every op in it is per token)."""
    n = x.shape[1]
    if n <= chunk:
        return ds.Block.hc_post(self, x, residual, post, comb)
    parts = [
        ds.Block.hc_post(
            self, x[:, s : s + chunk], residual[:, s : s + chunk], post[:, s : s + chunk], comb[:, s : s + chunk]
        )
        for s in range(0, n, chunk)
    ]
    return torch.cat(parts, dim=1)


# ---- the engine's model -------------------------------------------------------------------------------------------


class _NoHashState(torch.nn.Module):
    """Stands in for NgramHashState while the model is built on meta (it needs the tokenizer and real tensors); the
    hashes are computed outside the model (engram_hashes)."""

    def __init__(self, *args, **kwargs):
        super().__init__()


class GoldenModel(torch.nn.Module):
    """What the engine reads off a transformers model, around DeepSeek's Transformer."""

    def __init__(self, transformer, fp32_names: list[str], init_buffers):
        super().__init__()
        self.model, self.fp32_names, self.init_buffers = transformer, fp32_names, init_buffers

    def get_decoder(self):
        return self.model

    def _get_dtype_plan(self, dtype):
        # Parameters the reference declares fp32 (sinks, mHC, router bias, compressor pooling, head) stay fp32.
        return self.fp32_names if dtype != torch.float32 else []

    def _initialize_weights(self, module, _is_remote_code):
        self.init_buffers(module)

    def is_remote_code(self):
        return True


class DeepseekV41Golden(LayerwiseGolden):
    hf_repo = REPO
    # BOS only (the tokenizer adds none: add_bos_token false). V4-Flash's measured best frames the text as the model's
    # own chat reply; V4.1's encoding adds a reasoning-effort system preamble, so that is not carried over unmeasured.
    prefix = "<｜begin▁of▁sentence｜>"
    recipe = Recipe(
        modules=("attn.indexer",),  # + kv_norm, the gate and the Engram hashes (register_hooks)
        sublayers=("attn", "ffn", "engram", "attn_norm", "ffn_norm", "ffn.shared_experts"),
        cache=(),  # the reference keeps its caches in module buffers: see cache_tensors
        keep_fp32=("ffn.gate.output", "ffn.gate.top_k_weights", "pre_mix", "_state"),
        config=tuple(ARG_KEYS.values()) + ("num_hidden_layers", "rope_scaling", "max_position_embeddings"),
        notes=NOTES,
    )

    def __init__(self):
        register_config()
        self.kernel_check = self.dequant_check = None

    # ---- run ----------------------------------------------------------------------------------------------------

    def generate(self, args):
        if args.compute_dtype != "bfloat16":
            raise SystemExit("--compute-dtype bfloat16 only: the reference's Engram lookup returns bf16 rows")
        self.args, self.compute = args, getattr(torch, args.compute_dtype)
        self.device = torch.device(
            args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.chunk = args.attn_chunk or 1024
        args.revision = resolve_revision(args.hf_repo, args.revision)  # pinned once: weights and code alike
        local = args.revision == "local"
        self.code = fetch_code(self.hf_repo if local else args.hf_repo, CODE_REVISION if local else args.revision)
        self.ds = load_reference(self.code)
        self.convert_py = importlib.import_module("convert")
        if self.kernel_check is None:
            self.kernel_check = check_kernels(self.code, self.device) if self.device.type == "cuda" else "not run (CPU)"
            print(f"[kernels] {self.kernel_check}", flush=True)
        _SPARSE_CHUNK[0] = self.chunk
        self.engram_paths: dict[int, Path] = {}
        self.downloads = args.tmp / "hf_golden_engram"  # only what fetch() downloads here is ever deleted
        out = super().generate(args)
        meta = json.loads((out / "metadata.json").read_text())
        source = f"{self.hf_repo} @ {CODE_REVISION}" if local else f"{args.hf_repo} @ {args.revision}"
        meta["implementation"] = {
            "source": f"inference/ of {source}",
            "model_py_sha256": hashlib.sha256((self.code / "model.py").read_bytes()).hexdigest(),
            "activation_quantization": "off: act_quant / fp4_act_quant are no-ops (the unquantized model)",
            "kernels": "torch sparse_attn (exact fp32 softmax) and hc_split_sinkhorn in place of kernel.py",
            "kernel_check": self.kernel_check,
            "dequant_check": self.dequant_check,
            "query_chunk": self.chunk,
            "hc_post_chunk": HC_POST_CHUNK,
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return out

    def model_args(self, text_config):
        """ModelArgs from the repo's inference/config.json, checked against config.json, for this run: bf16 weights
        (dequantized), text only, one sequence of isl tokens."""
        # The raw config.json, not the PretrainedConfig, which may move or normalize keys (rope_theta, ...).
        text = json.loads(self.fetch("config.json").read_text())["text_config"]
        cfg = json.loads(self.fetch("inference/config.json").read_text())
        for arg, key in ARG_KEYS.items():
            if cfg.get(arg) != text.get(key):
                raise SystemExit(
                    f"inference/config.json {arg} = {cfg.get(arg)} but config.json {key} = {text.get(key)}"
                )
        cfg.update(
            n_layers=text_config.num_hidden_layers,  # --num-layers
            dtype="bf16",
            expert_dtype=None,
            max_batch_size=1,
            max_seq_len=self.args.isl,
            vision_n_layers=0,
            dspark_block_size=0,
        )
        return self.ds.ModelArgs(**cfg)

    def build(self, config, dtype):
        text = config.get_text_config()
        self.margs = margs = self.model_args(text)
        real_hash = self.ds.NgramHashState
        self.ds.NgramHashState = _NoHashState
        try:
            with self.ds.set_dtype(torch.bfloat16):  # as generate.py builds it; the Loader casts to --compute-dtype
                t = self.ds.Transformer(margs)
        finally:
            self.ds.NgramHashState = real_hash
            self.ds.precompute_freqs_cis.cache_clear()  # lru_cache'd: it now holds meta tables
        self.transformer, self.blocks = t, t.layers
        fp32 = [n for n, p in t.named_parameters() if p.dtype == torch.float32]
        return GoldenModel(t, fp32, self.init_buffers)

    def init_buffers(self, module):
        """The reference's non-persistent buffers, which the Loader left uninitialized: RoPE tables as
        Attention.__init__ computes them, zero caches (-inf gate scores) in the compute dtype (fp32 states stay)."""
        for name, buf in list(module.named_buffers(recurse=False)):
            if name == "freqs_cis":
                continue
            dtype = buf.dtype if buf.dtype == torch.float32 or not buf.is_floating_point() else self.compute
            new = torch.zeros(buf.shape, dtype=dtype, device=buf.device)
            module._buffers[name] = new.fill_(-torch.inf) if name == "score_state" else new
        if isinstance(module, self.ds.Attention):
            a = self.margs
            seq, theta = (a.original_seq_len, a.compress_rope_theta) if module.compress_ratio else (0, a.rope_theta)
            freqs = self.ds.precompute_freqs_cis.__wrapped__(  # not its lru_cache: real tensors
                module.rope_head_dim, a.max_seq_len, seq, theta, a.rope_factor, a.beta_fast, a.beta_slow
            )
            module._buffers["freqs_cis"] = freqs.to(module._buffers["freqs_cis"].device)

    def layer_types(self, text_config) -> list[str]:
        n = text_config.num_hidden_layers
        return [
            ("sliding_attention" if r == 0 else f"compressed_attention_{r}")
            + ("+engram" if i in text_config.engram_layer_ids else "")
            for i, r in enumerate(text_config.compress_ratios[:n])
        ]

    def stages(self, model, backbone, text_config):
        """Embedding, the Blocks (Engram layers without engram.*: read from their own shard in convert), the final
        norm (after hc_pre), the head. Checkpoint names are the reference's."""
        t = backbone
        stages = [("embed", t.embed, "embed.")]
        for i, block in enumerate(t.layers):
            block.forward = functools.partial(self.block_forward, block)
            block.hc_post = functools.partial(chunked_hc_post, self.ds, block)
            p = f"layers.{i}."
            prefix = (p + "attn.", p + "ffn.", p + "attn_norm.", p + "ffn_norm.", p + "hc_") if block.engram else p
            stages.append((f"layer {i}", block, prefix))
        t.norm.forward = functools.partial(self.final_forward, t.norm)
        return stages + [("norm", t.norm, "norm."), ("lm_head", t.head, "head.")]

    def block_forward(self, block, x, pre_mix, hash_ids=None):
        """Transformer.forward's loop body for one layer: layer 0 copies the embeddings into the streams; an Engram
        layer writes its n-gram lookup into the streams first. Returns (streams, pre_mix)."""
        with torch.device(x.device):  # get_window_topk_idxs builds on the default device
            if x.dim() == 3:
                x = x.unsqueeze(2).repeat(1, 1, block.hc_mult, 1)
            if block.engram is not None:
                x = block.engram(x, hash_ids, None)
            return self.ds.Block.forward(block, x, 0, pre_mix, None)

    def final_forward(self, norm, x):
        return self.ds.RMSNorm.forward(norm, self.ds.Block.hc_pre(None, x, self.state.carry["pre_mix"]))

    def logits(self, text_config, lm_head, hidden):
        return lm_head(hidden)[:, None]  # ParallelHead takes the last position itself

    def prepare(self, backbone, text_config, hidden, position_ids):
        self.ds.shared_attn = self.ds.SharedAttentionRuntime()
        self.hashes = None
        self.state = RunState(
            position_ids=position_ids,
            cache=None,
            masks={},
            rope={},
            carry={"pre_mix": self.ds.make_identity_pre_mix(hidden, self.margs.hc_mult)},
        )
        return self.state

    def engram_hashes(self, state):
        """NgramHashState over the prefilled ids: [1, N, n_engram_layers, n_hash_cols]."""
        if self.hashes is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                self.args.hf_repo, **({} if self.args.revision == "local" else {"revision": self.args.revision})
            )
            layout = self.transformer.engram_layout
            hasher = self.ds.NgramHashState(self.margs, layout, tok)
            self.hashes = hasher(state.input_ids.cpu(), 0, None).to(state.input_ids.device)
        return self.hashes

    def layer_kwargs(self, i, kind, state):
        block = self.blocks[i]
        hash_ids = None
        if block.engram is not None:
            hash_ids = self.engram_hashes(state)[:, :, block.engram.layer_hash_index, :]
        return dict(pre_mix=state.carry["pre_mix"], hash_ids=hash_ids)

    def after_layer(self, i, output, state):
        hidden, state.carry["pre_mix"] = output
        return hidden

    def convert(self, sd, prefix, module):
        """A Block: dequantize every <name>.weight with a <name>.scale, drop the image-token router bias, attach the
        Engram (small weights and the memory-mapped table, from its own shard), and the chunked indexer."""
        if not isinstance(module, self.ds.Block):
            return sd
        sd.pop("ffn.gate.bias_vl", None)
        for scale in sorted(k for k in sd if k.endswith(".scale")):
            weight = scale.removesuffix("scale") + "weight"
            if self.dequant_check is None and sd[weight].dtype == torch.int8:
                self.dequant_check = check_dequant(self.convert_py, sd[weight], sd[scale])
                print(f"[dequant] {self.dequant_check}", flush=True)
            sd[weight] = dequantize(sd.pop(weight), sd.pop(scale), self.compute)
        if module.engram is not None:
            sd.update(self.attach_engram(module))
        if module.attn.indexer is not None:
            module.attn.indexer.forward = functools.partial(
                chunked_indexer_forward, self.ds, module.attn.indexer, chunk=self.chunk
            )
        return sd

    def attach_engram(self, block) -> dict[str, torch.Tensor]:
        name = f"layers.{block.layer_id}.engram."
        index = self.fetch("model.safetensors.index.json")
        path = self.fetch(json.loads(index.read_text())["weight_map"][name + "embed.weight"])
        self.engram_paths[block.layer_id] = path
        with safe_open(str(path), "pt") as f:
            small = {
                k: f.get_tensor(name + k).to(self.device) for k in ("wkv.weight", "wkv.scale", "q_weight", "k_weight")
            }
        block.engram.embed = HostEngramTable(path, name + "embed", self.ds.fp8_block_size)
        wkv = dequantize(small.pop("wkv.weight"), small.pop("wkv.scale"), self.compute)
        return {"engram.wkv.weight": wkv, **{f"engram.{k}": v for k, v in small.items()}}

    def fetch(self, filename: str) -> Path:
        """A checkpoint file, from disk if it is there, else downloaded into --tmp (the Engram shards are not stage
        files, so the engine never fetches or deletes them; free_cache does)."""
        repo, rev = self.args.hf_repo, self.args.revision
        path = cached_file(repo, filename, rev)
        if path is not None and path.exists():
            return path
        return Path(hf_hub_download(repo, filename, revision=rev, local_dir=self.downloads))

    def register_hooks(self, i, layer, captured, sublayers):
        handles = super().register_hooks(i, layer, captured, sublayers)
        key = f"model.layers.{i}"

        def kv(_m, _a, out):  # cloned: the window K=V is rotated in place right after
            captured[f"{key}.attn.kv_norm.output"] = out.clone()

        def routing(gate, args, out):
            # Gate returns (weights, indices); its logits are this same fp32 GEMM, before score_func.
            captured[f"{key}.ffn.gate.output"] = F.linear(args[0].float(), gate.weight.float()) / gate.gate_temp
            index, order = out[1].sort(dim=-1)
            captured[f"{key}.ffn.gate.top_k_index"] = index.to(torch.int32)
            captured[f"{key}.ffn.gate.top_k_weights"] = out[0].gather(-1, order)

        handles += [layer.attn.kv_norm.register_forward_hook(kv), layer.ffn.gate.register_forward_hook(routing)]
        if layer.engram is not None:

            def hashes(_m, args):
                captured[f"{key}.engram.hash_ids"] = args[1].to(torch.int32)

            handles.append(layer.engram.register_forward_pre_hook(hashes))
        return handles

    def cache_tensors(self, i, cache):
        return self.state_of(self.blocks[i], i, self.state.carry["pre_mix"], self.state.position_ids.shape[-1])

    def state_of(self, block, i, pre_mix, n) -> dict[str, torch.Tensor]:
        """What a layer leaves for decode, read off the reference's buffers (see the header)."""
        a, out = block.attn, {f"model.layers.{i}.pre_mix": pre_mix}
        out[f"past_key_values.layers.{i}.window_kv"] = a.window_kv_cache[:1]
        if a.is_kv_source:
            r = a.compress_ratio
            out[f"past_key_values.layers.{i}.compressed_kv"] = a.compress_kv_cache[:1, : n // r]
            if r > 1 and n % r:
                out[f"past_key_values.layers.{i}.compressor_kv_state"] = a.compressor.kv_state[:1, : n % r]
                out[f"past_key_values.layers.{i}.compressor_score_state"] = a.compressor.score_state[:1, : n % r]
            if a.indexer is not None and a.indexer.owns_k:
                out[f"past_key_values.layers.{i}.index_k"] = a.indexer.k_cache[:1, : n // r]
        if a.indexer is not None and a.indexer.is_candidate_source:
            out[f"model.layers.{i}.attn.indexer.candidate_blocks"] = self.ds.shared_attn.candidates[
                :1, :, :: a.indexer.candidate_block_size
            ]
        return out

    def free_cache(self, i, cache):
        """The layer's Engram shard, if this run downloaded it."""
        path = self.engram_paths.pop(i, None)
        if path is not None and path.is_relative_to(self.downloads):
            self.blocks[i].engram.embed.maps.clear()
            path.unlink()

    # ---- --selftest ---------------------------------------------------------------------------------------------

    def selftest(self, args, compute, isl=64):
        """A tiny random checkpoint in the real one's layout (fp8 / fp4 weights with e8m0 scales, Engram tables in
        their own shards, the image router bias), streamed layer by layer with the chunked indexer / attention /
        hc_post, must give what one forward of the reference Transformer gives, whole and unchunked, from weights
        dequantized by convert.py's own formulas: exactly on CPU, to a tolerance on GPU."""
        device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        if compute != "bfloat16":
            print(
                f"[selftest] SKIP {device} {compute}: the reference runs bf16 only (its Engram lookup returns bf16)",
                flush=True,
            )
            return True
        self.ds = load_reference(fetch_code(self.hf_repo, CODE_REVISION))
        self.convert_py = importlib.import_module("convert")
        with tempfile.TemporaryDirectory(prefix="hf_golden_selftest_") as tmp:
            ckpt = Path(tmp) / "ckpt"
            margs = self.write_tiny_checkpoint(ckpt, isl)
            ids = torch.randint(
                0, margs.vocab_size, (4 * isl,), generator=torch.Generator().manual_seed(1), dtype=torch.int32
            )
            meta = {"source": "selftest", "source_sha256": "0" * 64, "synthetic": "random ids"}
            save_file({"token_ids": ids}, str(ckpt / f"{Path(TEXT).stem}.tokens.safetensors"), metadata=meta)
            run = argparse.Namespace(**{**vars(args), "hf_repo": str(ckpt), "isl": isl, "root": Path(tmp)})
            run.tmp, run.num_layers, run.sublayers, run.compute_dtype, run.prefix = Path(tmp), None, True, compute, ""
            run.attn_chunk = 16  # < isl: the chunked paths really chunk
            trace = self.generate(run)
            got = {}
            for f in trace.rglob("*.safetensors"):
                got.update(load_file(str(f)))
            prefilled = got.pop("input_ids", None)
            ref = self.reference_forward(ckpt, margs, ids[:isl], device, getattr(torch, compute))
        errors = [f"tensor names differ: {sorted(set(got) ^ set(ref))}"] if set(got) != set(ref) else []
        if prefilled is None or not torch.equal(prefilled, ids[:isl]):
            errors.append("input_ids: missing or not the ids the reference prefilled")
        for k in sorted(set(got) & set(ref)):
            if (e := self.compare(k, got[k], ref[k], exact=device.type == "cpu")) is not None:
                errors.append(f"{k}: {e}")
        print(f"[selftest] {'PASS' if not errors else 'FAIL'} {device} {compute}: {len(got)} tensors", flush=True)
        for e in errors:
            print(f"    {e}", flush=True)
        return not errors

    def tiny_args(self, isl):
        """Every mechanism of the real config, small: sliding-only layers 0-1, ratio 2 on 2-4 (source 2), ratio 1 on
        5-7 (source 5, also the candidate source), index sources 2, 4, 5, 7 (4 and 7 reuse keys; 7 chooses inside
        candidates), Engram on 1 and 4, YaRN (original_seq_len < isl), top-k and candidate counts that really cut.
        The real tokenizer: the Engram hash needs its compressed vocab."""
        cfg = json.loads(
            Path(hf_hub_download(self.hf_repo, "inference/config.json", revision=CODE_REVISION)).read_text()
        )
        cfg.update(
            dim=64,
            moe_inter_dim=32,
            n_layers=8,
            n_mtp_layers=0,
            n_heads=4,
            n_routed_experts=8,
            n_activated_experts=2,
            q_lora_rank=32,
            head_dim=32,
            rope_head_dim=16,
            o_groups=2,
            o_lora_rank=16,
            window_size=8,
            compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1],
            kv_source_layers=[2, 5],
            index_source_layers=[2, 4, 5, 7],
            candidate_source_layer=5,
            candidate_topk_blocks=3,
            candidate_block_size=4,
            original_seq_len=32,
            rope_factor=4,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=8,
            engram_layer_ids=[1, 4],
            engram_max_ngram_size=3,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_vocab_size=40,
            vision_n_layers=0,
            dspark_block_size=0,
            dspark_target_layer_ids=[],
            max_batch_size=1,
            max_seq_len=isl,
        )
        layout = self.ds.EngramLayout.from_args(self.ds.ModelArgs(**{**cfg, "engram_num_embeddings": [0, 0]}))
        cfg["engram_num_embeddings"] = [sum(p for per in layer for p in per) for layer in layout.primes]
        return cfg

    def write_tiny_checkpoint(self, ckpt: Path, isl: int):
        """config.json (deepseek_v41) + inference/config.json + tokenizer + shards, weights quantized as the real
        checkpoint stores them. Returns the run's ModelArgs."""
        cfg = self.tiny_args(isl)
        (ckpt / "inference").mkdir(parents=True)
        (ckpt / "inference/config.json").write_text(
            json.dumps({**cfg, "dtype": "fp8", "expert_dtype": "fp4"}, indent=2)
        )
        text = {key: cfg[arg] for arg, key in ARG_KEYS.items()}
        text.update(num_hidden_layers=cfg["n_layers"], max_position_embeddings=1048576, model_type="deepseek_v41_text")
        config = {
            "model_type": "deepseek_v41",
            "architectures": ["DeepseekV41ForCausalLM"],
            "transformers_version": "5.6.0",  # as the real one's; without it transformers warns of a Mistral regex
            "text_config": text,
        }
        (ckpt / "config.json").write_text(json.dumps(config, indent=2))
        for f in ("tokenizer.json", "tokenizer_config.json"):
            (ckpt / f).write_bytes(Path(hf_hub_download(self.hf_repo, f, revision=CODE_REVISION)).read_bytes())
        margs = self.ds.ModelArgs(**{**cfg, "dtype": "bf16", "expert_dtype": None})
        real_hash = self.ds.NgramHashState
        self.ds.NgramHashState = _NoHashState
        try:
            with self.ds.set_dtype(torch.bfloat16), torch.device("meta"):
                shapes = {k: (v.shape, v.dtype) for k, v in self.ds.Transformer(margs).state_dict().items()}
        finally:
            self.ds.NgramHashState = real_hash
            self.ds.precompute_freqs_cis.cache_clear()  # lru_cache'd: it now holds meta tables
        g = torch.Generator().manual_seed(0)

        def pow2(shape, lo, hi):
            return (2.0 ** torch.randint(lo, hi, shape, generator=g)).to(torch.float8_e8m0fnu)

        sd = {}
        for k, (shape, dtype) in shapes.items():
            out_dim, in_dim = (shape[0], shape[-1]) if len(shape) else (0, 0)
            if FP4_WEIGHTS.search(k):
                sd[k] = torch.randint(-128, 128, (out_dim, in_dim // 2), generator=g, dtype=torch.int8)
                sd[k.removesuffix("weight") + "scale"] = pow2((out_dim, in_dim // FP8_BLOCK), -8, -5)
            elif FP8_WEIGHTS.search(k):
                sd[k] = (torch.randn(shape, generator=g) * 2).to(torch.float8_e4m3fn)
                sd[k.removesuffix("weight") + "scale"] = pow2(
                    (-(-out_dim // FP8_BLOCK), -(-in_dim // FP8_BLOCK)), -7, -4
                )
            elif k.endswith("engram.embed.weight"):
                sd[k] = torch.randn(shape, generator=g).to(torch.float8_e4m3fn)
            elif k.endswith("engram.embed.scale"):
                sd[k] = pow2(shape, -3, 0)
            elif k.endswith("norm.weight") or k.endswith(("q_weight", "k_weight")):
                sd[k] = (1 + 0.1 * torch.randn(shape, generator=g)).to(torch.bfloat16)
            else:
                t = torch.randn(shape, generator=g) * (0.5 if k.endswith(("_scale", "_base", "attn_sink")) else 0.05)
                fp32 = dtype == torch.float32 and not k.endswith("head.weight") and ".compressor." not in k
                sd[k] = t.to(torch.float32 if fp32 else torch.bfloat16)
        for i in range(cfg["n_layers"]):
            sd[f"layers.{i}.ffn.gate.bias_vl"] = torch.randn(cfg["n_routed_experts"], generator=g)
        # Shards as the real checkpoint's: each Engram layer's tensors on their own, the rest split in two.
        files = {
            k: (f"engram{m[1]}" if (m := re.match(r"layers\.(\d+)\.engram\.", k)) else f"part{len(k) % 2}") for k in sd
        }
        names = sorted(set(files.values()))
        weight_map = {k: f"model-{names.index(f) + 1:05d}-of-{len(names):05d}.safetensors" for k, f in files.items()}
        for f in set(weight_map.values()):
            save_file({k: v for k, v in sd.items() if weight_map[k] == f}, str(ckpt / f), metadata={"format": "pt"})
        (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}, indent=2))
        return margs

    def reference_forward(self, ckpt, margs, ids, device, dtype):
        """One Transformer.forward of the reference, whole sequence, its own Indexer / hc_post / sparse_attn unchunked,
        weights dequantized by convert.py's formulas (its wo_a block expansion; its FP4_TABLE for the experts)."""
        from transformers import AutoTokenizer

        sd = {}
        for f in sorted(set(json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"].values())):
            sd.update(load_file(str(ckpt / f)))
        for k in [k for k in sd if k.endswith(".scale") and not k.endswith("engram.embed.scale")]:
            w, s = sd[k.removesuffix("scale") + "weight"], sd.pop(k)
            if w.dtype == torch.int8:
                sd[k.removesuffix("scale") + "weight"] = fp4_table_decode(self.convert_py, w, s).to(torch.bfloat16)
            else:
                bo, bi = w.size(0) // s.size(0), w.size(1) // s.size(1)
                full = w.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float() * s[:, None, :, None].float()
                sd[k.removesuffix("scale") + "weight"] = full.flatten(2, 3).flatten(0, 1).bfloat16()
        for k in [k for k in sd if k.endswith("bias_vl")]:
            del sd[k]
        tok = AutoTokenizer.from_pretrained(str(ckpt))
        with self.ds.set_dtype(torch.bfloat16):
            model = self.ds.Transformer(margs, tok)
        model.load_state_dict(sd, strict=True)
        if dtype == torch.float32:
            model = model.float()
        model = model.to(device).eval()
        captured: dict[str, torch.Tensor] = {}

        def put(key):
            return lambda _m, _a, o: captured.__setitem__(key, o[0] if isinstance(o, tuple) else o)

        model.embed.register_forward_hook(put("model.embed_tokens.output"))
        model.norm.register_forward_hook(put("model.last_hidden_state"))
        pre_mix = {}
        for i, layer in enumerate(model.layers):
            layer.register_forward_hook(put(f"model.layers.{i}.output"))
            layer.register_forward_hook(lambda _m, _a, o, i=i: pre_mix.__setitem__(i, o[1]))
            self.register_hooks(i, layer, captured, sublayers=True)
        _SPARSE_CHUNK[0] = 0
        try:
            with torch.device(device), torch.no_grad():
                _, logits, _ = model(ids[None].long().to(device), 0)
        finally:
            _SPARSE_CHUNK[0] = self.chunk
        captured["logits"] = logits[:, None]
        for i, layer in enumerate(model.layers):
            captured.update(self.state_of(layer, i, pre_mix[i], ids.numel()))
        return {k: v.cpu() for k, v in captured.items()}


def register_config() -> None:
    """config.json's deepseek_v41 / deepseek_v41_text, which transformers does not know, as plain configs: the engine
    reads the layer count, sizes and context off them; the model itself is built from inference/config.json."""
    from transformers import AutoConfig, PretrainedConfig

    class DeepseekV41TextConfig(PretrainedConfig):
        model_type = "deepseek_v41_text"

        def __init__(self, **kwargs):
            # Kept as given: PretrainedConfig would standardize them into rope_parameters (no model reads them here).
            rope = {k: kwargs.pop(k) for k in ("rope_scaling", "rope_theta") if k in kwargs}
            super().__init__(**kwargs)
            for k, v in rope.items():
                object.__setattr__(self, k, v)

    class DeepseekV41Config(PretrainedConfig):
        model_type = "deepseek_v41"
        sub_configs = {"text_config": DeepseekV41TextConfig}

        def __init__(self, text_config=None, vision_config=None, **kwargs):
            if not isinstance(text_config, PretrainedConfig):
                text_config = DeepseekV41TextConfig(**(text_config or {}))
            self.text_config, self.vision_config = text_config, vision_config
            super().__init__(**kwargs)

        def get_text_config(self, *args, **kwargs):
            return self.text_config

    for cls in (DeepseekV41TextConfig, DeepseekV41Config):
        AutoConfig.register(cls.model_type, cls, exist_ok=True)


if __name__ == "__main__":
    sys.exit(DeepseekV41Golden().main())
