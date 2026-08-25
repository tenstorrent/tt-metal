# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained torch reference (CPU golden) for the MiniMax-M3 **text** model.

This is the correctness oracle for prefill: a plain-PyTorch forward over the real
``language_model.*`` checkpoint weights, producing per-layer KV cache (post-RoPE K, raw V)
and optionally logits. It exists because the shipped HF checkpoint is a *vision-language*
package (``model_type=minimax_m3_vl``, arch ``MiniMaxM3SparseForConditionalGeneration``) that
ships only config + image/video processors and **no** ``modeling_*.py`` — so
``AutoModelForCausalLM.from_pretrained`` cannot build it. This module rebuilds just the text
tower from the weights directly.

The math is composed from the team's **already-PCC-validated** unit-test references so it is
exactly what the TT model is checked against:
  * dense decoder / MoE / norms / RoPE  -> ``tests/unit/test_model_vs_ref.py``
  * per-head QK-norm + partial RoPE      -> ``tests/unit/test_msa_layer_vs_ref.py``
Weights are used in **raw HF layout** (no Meta swizzle — that conversion is only for the device
path), so the K/V this produces matches the layout the old HF-based golden produced.

ATTENTION (dense + MSA, the real schedule)
------------------------------------------
Layers with ``sparse_attention_freq[i] == 0`` (layers 0-2) use full causal attention. Layers with
``sparse_attention_freq[i] == 1`` (layers 3-59) use MSA sparse attention: a lightweight index
branch (``index_q/k_proj`` -> per-head norm -> partial RoPE) scores key blocks, the top
``sparse_topk_blocks`` (16) blocks of ``sparse_block_size`` (128) — plus the force-local current
block (no sink/init block, matching upstream) — are selected per GQA group, and attention runs causally over only the
selected blocks. This mirrors the device chain (``msa.py`` + ``indexer_score_msa`` /
``sparse_sdpa_msa``) and the validated torch reference in ``tests/unit/test_msa_layer_vs_ref.py``.

Because MSA always keeps a *causal* mask and selects up to ``MSA_DENSE_EQUIV_TOKENS = 2048`` keys,
for a context of <= 2048 tokens every block is selected and MSA reduces **exactly** to full causal
attention. Above 2048 it is genuinely sparse.

``dense_only=True`` (CLI ``--dense-only``) forces full causal attention on every layer, skipping
the index branch + block selection entirely. At <= 2048 this is identical to the default; above
2048 it does NOT match the real sparse model — use it to exercise the larger-sequence pipeline
while MSA prefill is not yet device-tested, not as a valid >2048 golden.

NOTE on the index branch: ``msa.py`` flags the index-branch RoPE as not-yet-verified against the
HF modeling source. It only affects which blocks are selected, so it is a no-op for <= 2048 (all
blocks selected) and only matters for >2048 sparsity. This reference matches the device's choice.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

# MSA equivalence bound: sparse_topk_blocks(16) * sparse_block_size(128). At or below this many
# context tokens, MSA's top-k block selection covers every block -> identical to dense attention.
MSA_DENSE_EQUIV_TOKENS = 2048

# -------------------------------------------------------------------------------------------------
# Long-context memory controls. The only O(S^2) tensors in this reference are the attention scores
# ([heads, S, S]) and the MSA index scores ([groups, S, S]); at S=55k, fp32 those are hundreds of
# GB. We never materialize them: queries are processed in row-chunks of ATTN_Q_CHUNK (exact per-row
# softmax, so the result is numerically identical to the un-chunked path), and the FFNs are computed
# in token-chunks of FFN_TOKEN_CHUNK. Both are pure memory/speed knobs — they do NOT change outputs.
# Defaults keep peak host RAM well under ~40GB at S=55k in fp32; raise them (via env) for speed on
# machines with more RAM, or lower them if you still OOM.
ATTN_Q_CHUNK = int(os.environ.get("REF_ATTN_Q_CHUNK", "256"))
FFN_TOKEN_CHUNK = int(os.environ.get("REF_FFN_TOKEN_CHUNK", "4096"))


@dataclass
class MiniMaxM3Config:
    """The subset of ``text_config`` the text forward needs."""

    hidden_size: int = 6144
    num_hidden_layers: int = 60
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    rotary_dim: int = 64  # partial_rotary_factor 0.5 * head_dim
    rope_theta: float = 5_000_000.0
    rms_norm_eps: float = 1e-6
    vocab_size: int = 200064
    # MoE
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 2.0
    # swigluoai activation
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    # per-layer schedules (0 -> dense, 1 -> MoE / sparse-attn)
    moe_layer_freq: Optional[list] = None
    sparse_attention_freq: Optional[list] = None
    # MSA (sparse attention) — sparse_attention_config
    sparse_index_dim: int = 128
    sparse_num_index_heads: int = 4
    sparse_block_size: int = 128
    sparse_topk_blocks: int = 16
    sparse_local_block: int = 1  # current block is always selected (force-local); no sink/init block (upstream)

    @classmethod
    def from_hf_config(cls, hf_config: dict) -> "MiniMaxM3Config":
        """Build from the flat ``text_config`` dict (the inner block of the VL config.json)."""
        tc = hf_config.get("text_config", hf_config)
        head_dim = tc.get("head_dim", 128)
        rotary_dim = tc.get("rotary_dim")
        if rotary_dim is None:
            rotary_dim = int(round(tc.get("partial_rotary_factor", 0.5) * head_dim))
        sac = tc.get("sparse_attention_config", {}) or {}
        return cls(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            rope_theta=tc.get("rope_theta", 5_000_000.0),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            vocab_size=tc["vocab_size"],
            num_local_experts=tc.get("num_local_experts", 128),
            num_experts_per_tok=tc.get("num_experts_per_tok", 4),
            routed_scaling_factor=tc.get("routed_scaling_factor", 2.0),
            swiglu_alpha=tc.get("swiglu_alpha", 1.702),
            swiglu_limit=tc.get("swiglu_limit", 7.0),
            moe_layer_freq=list(tc.get("moe_layer_freq", [])),
            sparse_attention_freq=list(sac.get("sparse_attention_freq", [])),
            sparse_index_dim=sac.get("sparse_index_dim", 128),
            sparse_num_index_heads=sac.get("sparse_num_index_heads", 4),
            sparse_block_size=sac.get("sparse_block_size", 128),
            sparse_topk_blocks=sac.get("sparse_topk_blocks", 16),
            sparse_local_block=sac.get("sparse_local_block", 1),
        )

    def is_moe_layer(self, i: int) -> bool:
        return bool(self.moe_layer_freq[i]) if self.moe_layer_freq else False

    def is_sparse_attn_layer(self, i: int) -> bool:
        return bool(self.sparse_attention_freq[i]) if self.sparse_attention_freq else False


# --------------------------------------------------------------------------------------------
# Weight sources: stream raw HF weights on demand so the 60-layer / 426GB checkpoint never has
# to be fully resident. Each ``get`` returns a fresh fp32 tensor; the caller drops it per layer.
# --------------------------------------------------------------------------------------------
class SafetensorsWeights:
    """Lazy, mmap-backed accessor over a sharded ``language_model.*`` checkpoint."""

    def __init__(self, model_path, prefix: str = "language_model.", dtype: torch.dtype = torch.float32):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.model_path = Path(model_path)
        self.prefix = prefix
        self.dtype = dtype
        index_path = self.model_path / "model.safetensors.index.json"
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self._handles: dict = {}

    def _handle(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = self._handles[shard] = self._safe_open(str(self.model_path / shard), framework="pt")
        return h

    def has(self, name: str) -> bool:
        return (self.prefix + name) in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        key = self.prefix + name
        return self._handle(self.weight_map[key]).get_tensor(key).to(self.dtype)


class DictWeights:
    """Wrap an in-memory ``state_dict`` (raw HF keys, no prefix) — used by unit tests."""

    def __init__(self, state_dict: dict, dtype: torch.dtype = torch.float32):
        self.state_dict = state_dict
        self.dtype = dtype

    def has(self, name: str) -> bool:
        return name in self.state_dict

    def get(self, name: str) -> torch.Tensor:
        return self.state_dict[name].to(self.dtype)


# --------------------------------------------------------------------------------------------
# Math primitives — verbatim from the validated torch references.
# --------------------------------------------------------------------------------------------
def gemma_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Full-width RMSNorm with Gemma ``(1 + w)`` scale (use_gemma_norm=True)."""
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)) * (1.0 + w.float())


def gemma_head_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm over head_dim with ``(1 + w)`` scale (qk_norm_type=per_head)."""
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)) * (1.0 + w.float())


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def partial_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """RoPE on the leading ``rotary_dim`` channels (rotate-half / NeoX); the rest pass through."""
    rot, passth = t[..., :rotary_dim], t[..., rotary_dim:]
    return torch.cat([rot * cos + _rotate_half(rot) * sin, passth], dim=-1)


def build_rope(seq_len: int, rotary_dim: int, theta: float, device=None, dtype=torch.float32):
    """HF-convention cos/sin for partial RoPE, shaped ``[1, 1, S, rotary_dim]``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """swigluoai: clamp, gated-SiLU on ``gate``, ``(up + 1)`` multiplicative skip."""
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1.0) * (gate * torch.sigmoid(alpha * gate))


def ffn(x: torch.Tensor, w_gate, w_up, w_down, alpha: float, limit: float) -> torch.Tensor:
    """SwiGLU FFN. ``w_gate``=w1/gate_proj, ``w_up``=w3/up_proj, ``w_down``=w2/down_proj.

    Computed in token-chunks of ``FFN_TOKEN_CHUNK`` so the ``[tokens, intermediate]`` activations
    never fully materialize (at S=55k the dense-MLP intermediate is 12288-wide). Chunking is along
    tokens only, so the output is numerically identical to the single-shot path.
    """
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    n = flat.shape[0]
    out_dim = w_down.shape[0]
    if n <= FFN_TOKEN_CHUNK:
        y = swiglu_oai(flat @ w_gate.t(), flat @ w_up.t(), alpha, limit) @ w_down.t()
    else:
        y = torch.empty(n, out_dim, dtype=flat.dtype, device=flat.device)
        for i in range(0, n, FFN_TOKEN_CHUNK):
            j = min(i + FFN_TOKEN_CHUNK, n)
            fc = flat[i:j]
            y[i:j] = swiglu_oai(fc @ w_gate.t(), fc @ w_up.t(), alpha, limit) @ w_down.t()
    return y.reshape(*lead, out_dim)


def msa_block_selection(index_q, index_k, scale, block_size, topk_blocks):
    """MSA indexer block selection, mirroring ``indexer_score_msa`` + top-k.

    ``index_q`` ``[1, G, S, d]`` (per GQA group), ``index_k`` ``[1, 1, S, d]`` (one shared head,
    broadcast over G). Single-shot prefill so query row ``s`` attends to keys ``[0, s]``
    (chunk_start=0). Returns a per-group boolean ``[G, S, nblk]`` of selected key blocks.

    Pipeline: scaled index dot -> causal ``-inf`` for future tokens -> block max-pool (score_type
    "max") -> force-local the current block (+inf) -> top-k blocks. Matches upstream
    ``minimax_m3_vl`` MiniMaxM3VLIndexer: ONLY the local block is forced, there is no sink/init block.
    """
    G, S, T = index_q.shape[1], index_q.shape[2], index_k.shape[2]
    nblk = (T + block_size - 1) // block_size
    tpad = nblk * block_size

    scores = scale * (index_q.float() @ index_k.float().transpose(-1, -2))  # [1,G,S,T]
    kpos = torch.arange(T, device=scores.device)
    qpos = torch.arange(S, device=scores.device)
    scores = scores.masked_fill(kpos[None, None, None, :] > qpos[None, None, :, None], float("-inf"))
    if tpad > T:  # pad the partial trailing block with -inf so block max-pool ignores it
        scores = torch.cat([scores, scores.new_full((1, G, S, tpad - T), float("-inf"))], dim=-1)

    bs = scores.view(1, G, S, nblk, block_size).max(-1).values  # block max-pool [1,G,S,nblk]
    local = (qpos // block_size).clamp(max=nblk - 1)
    bs[0, :, torch.arange(S, device=scores.device), local] = float("inf")  # force-local current block

    k = min(topk_blocks, nblk)
    sel = bs.topk(k, dim=-1).indices  # [1,G,S,k]
    selected = torch.zeros(1, G, S, nblk, dtype=torch.bool, device=scores.device)
    selected.scatter_(-1, sel, True)
    return selected[0]  # [G, S, nblk]


def msa_block_selection_chunk(iq_g, ik_shared, scale, block_size, topk_blocks, q_start, seq_len):
    """Per-query-chunk MSA block selection -> token-level ``[C, seq_len]`` bool mask.

    Identical math to :func:`msa_block_selection`, but for a single GQA group and only the ``C``
    query rows ``[q_start, q_start + C)``, so the score tensor is ``O(C * seq_len)`` instead of
    ``O(S * seq_len)``. This is what makes >2048-token prefill fit in modest RAM.

    ``iq_g``       ``[C, d]``  one group's post-norm/post-RoPE index-q rows for this chunk.
    ``ik_shared``  ``[T, d]``  the single shared index-k over the full context (``T == seq_len``).
    Returns the selected key TOKENS (blocks expanded and trimmed to ``seq_len``), ready to use as an
    attention mask. Only the local (current) block is force-selected — no sink/init block (upstream).
    """
    C = iq_g.shape[0]
    T = ik_shared.shape[0]
    nblk = (T + block_size - 1) // block_size
    tpad = nblk * block_size

    scores = scale * (iq_g.float() @ ik_shared.float().t())  # [C, T]
    kpos = torch.arange(T, device=scores.device)
    qpos = torch.arange(q_start, q_start + C, device=scores.device)
    scores = scores.masked_fill(kpos[None, :] > qpos[:, None], float("-inf"))  # causal (future keys)
    if tpad > T:  # pad partial trailing block with -inf so block max-pool ignores it
        scores = torch.cat([scores, scores.new_full((C, tpad - T), float("-inf"))], dim=-1)

    bs = scores.view(C, nblk, block_size).max(-1).values  # block max-pool [C, nblk]
    local = (qpos // block_size).clamp(max=nblk - 1)
    bs[torch.arange(C, device=scores.device), local] = float("inf")  # force-local current block

    k = min(topk_blocks, nblk)
    sel = bs.topk(k, dim=-1).indices  # [C, k]
    selected = torch.zeros(C, nblk, dtype=torch.bool, device=scores.device)
    selected.scatter_(-1, sel, True)
    return selected.repeat_interleave(block_size, dim=-1)[:, :seq_len]  # [C, seq_len] bool


# --------------------------------------------------------------------------------------------
# The model.
# --------------------------------------------------------------------------------------------
class MiniMaxM3TextModel:
    """MiniMax-M3 text reference (dense layers 0-2 + MSA sparse layers 3-59, the real schedule).

    Streams weights per layer via a weight source. ``dense_only`` forces full causal attention
    everywhere."""

    def __init__(self, config: MiniMaxM3Config, weights):
        self.cfg = config
        self.w = weights

    # ---- MSA index branch (per-group index_q + one shared index_k) ----
    def _index_branch(self, x, p, cos, sin):
        cfg = self.cfg
        B, S, _ = x.shape
        nidx, d = cfg.sparse_num_index_heads, cfg.sparse_index_dim
        iq = (x @ self.w.get(p + "self_attn.index_q_proj.weight").t()).view(B, S, nidx, d).transpose(1, 2)
        ik = (x @ self.w.get(p + "self_attn.index_k_proj.weight").t()).view(B, S, 1, d).transpose(1, 2)
        iq = partial_rope(
            gemma_head_rms_norm(iq, self.w.get(p + "self_attn.index_q_norm.weight"), cfg.rms_norm_eps),
            cos,
            sin,
            cfg.rotary_dim,
        )
        ik = partial_rope(
            gemma_head_rms_norm(ik, self.w.get(p + "self_attn.index_k_norm.weight"), cfg.rms_norm_eps),
            cos,
            sin,
            cfg.rotary_dim,
        )
        return iq, ik

    # ---- attention (GQA; full causal, or MSA block-sparse causal) ----
    def _attention(self, x, p, cos, sin, sparse: bool):
        cfg = self.cfg
        B, S, _ = x.shape
        nq, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        rep = nq // nkv

        q = (x @ self.w.get(p + "self_attn.q_proj.weight").t()).view(B, S, nq, hd).transpose(1, 2)
        k = (x @ self.w.get(p + "self_attn.k_proj.weight").t()).view(B, S, nkv, hd).transpose(1, 2)
        v = (x @ self.w.get(p + "self_attn.v_proj.weight").t()).view(B, S, nkv, hd).transpose(1, 2)

        q = gemma_head_rms_norm(q, self.w.get(p + "self_attn.q_norm.weight"), cfg.rms_norm_eps)
        k = gemma_head_rms_norm(k, self.w.get(p + "self_attn.k_norm.weight"), cfg.rms_norm_eps)
        q = partial_rope(q, cos, sin, cfg.rotary_dim)
        k = partial_rope(k, cos, sin, cfg.rotary_dim)

        # KV cache stores post-RoPE K and raw V (HF layout): [B, nkv, S, hd]. Same for dense/MSA.
        # MSA layers additionally cache the post-norm/post-RoPE index_k (single shared head [B,1,S,d]);
        # dense layers leave it None. See _index_branch.
        k_cache, v_cache = k, v
        index_k = None

        scale = hd**-0.5

        # MSA index branch (per-GQA-group block selection). We keep index_q/index_k around and
        # compute the block mask per query-chunk below (never the full [G, S, S]).
        iq = ik_shared = None
        if sparse:
            iq, ik = self._index_branch(x, p, cos, sin)  # iq [B,G,S,d], ik [B,1,S,d]
            index_k = ik  # cache the post-norm/post-RoPE shared index key
            ik_shared = ik[0, 0]  # [S, d] (B==1 for the sparse path, as in the device pipeline)

        # Attend per GQA group, and within each group per query row-chunk, so the score tensor peaks
        # at [rep, ATTN_Q_CHUNK, S] instead of [rep, S, S] — the difference between a few hundred MB
        # and hundreds of GB at S=55k. Exact per-row softmax => bit-identical to the un-chunked path.
        #
        # Causality is enforced at the token level here (future keys -> -inf). For MSA, the block
        # mask (top-k past blocks + the force-local current block) is intersected on top. Both masks
        # produce the same -inf set as the old causal_bias + block-mask formulation; below the
        # top-k*block_size (=2048) bound every block is selected, so MSA == full causal attention.
        out = torch.empty(B, nq, S, hd, dtype=q.dtype, device=x.device)
        kpos = torch.arange(S, device=x.device)
        for g in range(nkv):
            kgt = k[:, g].transpose(-1, -2)  # [B, hd, S]
            vg = v[:, g]  # [B, S, hd]
            for qs in range(0, S, ATTN_Q_CHUNK):
                qe = min(qs + ATTN_Q_CHUNK, S)
                qg = q[:, g * rep : (g + 1) * rep, qs:qe, :]  # [B, rep, C, hd]
                scores = (qg @ kgt) * scale  # [B, rep, C, S]
                future = kpos[None, :] > kpos[qs:qe][:, None]  # [C, S] bool: key later than query
                scores = scores.masked_fill(future[None, None], float("-inf"))
                if sparse:
                    sel = msa_block_selection_chunk(  # [C, S] bool: selected key tokens
                        iq[0, g, qs:qe], ik_shared, scale, cfg.sparse_block_size, cfg.sparse_topk_blocks, qs, S
                    )
                    scores = scores.masked_fill(~sel[None, None], float("-inf"))  # drop unselected blocks
                attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                out[:, g * rep : (g + 1) * rep, qs:qe, :] = attn @ vg
                del scores, attn

        o = out.transpose(1, 2).reshape(B, S, nq * hd)
        attn_out = o @ self.w.get(p + "self_attn.o_proj.weight").t()
        return attn_out, k_cache, v_cache, index_k

    # ---- dense MLP ----
    def _dense_mlp(self, x, p):
        cfg = self.cfg
        return ffn(
            x,
            self.w.get(p + "mlp.gate_proj.weight"),
            self.w.get(p + "mlp.up_proj.weight"),
            self.w.get(p + "mlp.down_proj.weight"),
            cfg.swiglu_alpha,
            cfg.swiglu_limit,
        )

    # ---- sparse MoE (sigmoid routing + correction bias + shared expert) ----
    def _moe(self, x, p):
        cfg = self.cfg
        flat = x.reshape(-1, cfg.hidden_size)  # [T, H]
        T = flat.shape[0]

        gate_w = self.w.get(p + "block_sparse_moe.gate.weight")
        bias = self.w.get(p + "block_sparse_moe.e_score_correction_bias")
        scores = torch.sigmoid(flat.float() @ gate_w.float().t())  # [T, E]
        sel = torch.topk(scores + bias.float(), cfg.num_experts_per_tok, dim=-1).indices  # bias selects only
        tw = torch.gather(scores, 1, sel)  # un-biased scores weight the combine
        tw = (tw / tw.sum(-1, keepdim=True)) * cfg.routed_scaling_factor  # [T, topk]

        # Per-expert membership + combine weight, so each expert FFN runs once over its tokens.
        member = torch.zeros(T, cfg.num_local_experts, dtype=torch.bool)
        member.scatter_(1, sel, True)
        weight = torch.zeros(T, cfg.num_local_experts, dtype=flat.dtype)
        weight.scatter_(1, sel, tw.to(flat.dtype))

        out = torch.zeros_like(flat)
        for e in range(cfg.num_local_experts):
            idx = member[:, e].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue  # skip (and don't even load) experts with no routed tokens
            ep = p + f"block_sparse_moe.experts.{e}."
            y = ffn(
                flat[idx],
                self.w.get(ep + "w1.weight"),
                self.w.get(ep + "w3.weight"),
                self.w.get(ep + "w2.weight"),
                cfg.swiglu_alpha,
                cfg.swiglu_limit,
            )
            out[idx] += weight[idx, e].unsqueeze(-1) * y

        sp = p + "block_sparse_moe.shared_experts."
        out += ffn(
            flat,
            self.w.get(sp + "gate_proj.weight"),
            self.w.get(sp + "up_proj.weight"),
            self.w.get(sp + "down_proj.weight"),
            cfg.swiglu_alpha,
            cfg.swiglu_limit,
        )
        return out.reshape(x.shape)

    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        dense_only: bool = False,
        compute_logits: bool = False,
        kv_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], None]] = None,
        compute_dtype: torch.dtype = torch.float32,
    ):
        """Single-shot prefill forward (no cache reuse). ``input_ids`` is ``[B, S]`` (B==1).

        Uses the real attention schedule (dense for layers 0-2, MSA sparse for 3-59) unless
        ``dense_only=True``, which forces full causal attention on every layer.

        Returns ``(kv_caches, logits)`` where ``kv_caches`` is a list of ``(k, v)`` per layer
        (each ``[B, num_kv_heads, S, head_dim]``, post-RoPE K + raw V) and ``logits`` is
        ``[B, S, vocab]`` or ``None``. If ``kv_callback`` is given it is called as
        ``kv_callback(layer_idx, k, v, index_k)`` after each layer (for streaming saves); ``index_k`` is
        the post-norm/post-RoPE MSA index key ``[B, 1, S, sparse_index_dim]`` on sparse layers, else None.
        """
        cfg = self.cfg
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, S = input_ids.shape
        assert B == 1, "reference prefill supports batch size 1"

        embed = self.w.get("model.embed_tokens.weight")
        x = embed[input_ids.reshape(-1)].reshape(B, S, cfg.hidden_size).to(compute_dtype)
        del embed
        return self.forward_hidden(x, dense_only=dense_only, compute_logits=compute_logits, kv_callback=kv_callback)

    def forward_hidden(
        self,
        x: torch.Tensor,
        *,
        dense_only: bool = False,
        compute_logits: bool = False,
        kv_callback: Optional[Callable[[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]], None]] = None,
    ):
        """Transformer-stack forward from post-embedding hidden states ``x`` ``[B, S, H]``.

        Same contract as the TT ``prefill_forward`` (embedding applied upstream).
        """
        cfg = self.cfg
        B, S, _ = x.shape
        compute_dtype = x.dtype
        cos, sin = build_rope(S, cfg.rotary_dim, cfg.rope_theta, device=x.device, dtype=compute_dtype)

        kv_caches = []
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            sparse = cfg.is_sparse_attn_layer(i) and not dense_only
            normed = gemma_rms_norm(x, self.w.get(p + "input_layernorm.weight"), cfg.rms_norm_eps)
            attn_out, k_cache, v_cache, index_k = self._attention(normed, p, cos, sin, sparse)
            x = x + attn_out

            normed2 = gemma_rms_norm(x, self.w.get(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps)
            ffn_out = self._moe(normed2, p) if cfg.is_moe_layer(i) else self._dense_mlp(normed2, p)
            x = x + ffn_out

            k_cache = k_cache.to(compute_dtype)
            v_cache = v_cache.to(compute_dtype)
            if index_k is not None:
                index_k = index_k.to(compute_dtype)
            if kv_callback is not None:
                kv_callback(i, k_cache, v_cache, index_k)
            kv_caches.append((k_cache, v_cache))

        logits = None
        if compute_logits:
            h = gemma_rms_norm(x, self.w.get("model.norm.weight"), cfg.rms_norm_eps)
            logits = h @ self.w.get("lm_head.weight").t()
        return kv_caches, logits


def load_text_model(model_path, *, compute_dtype: torch.dtype = torch.float32) -> MiniMaxM3TextModel:
    """Build the reference text model from a MiniMax-M3(-VL) checkpoint directory.

    Reads ``config.json`` (text_config) and streams ``language_model.*`` weights via mmap.
    """
    model_path = Path(model_path)
    hf_config = json.loads((model_path / "config.json").read_text())
    cfg = MiniMaxM3Config.from_hf_config(hf_config)
    weights = SafetensorsWeights(model_path, prefix="language_model.", dtype=compute_dtype)
    return MiniMaxM3TextModel(cfg, weights)
