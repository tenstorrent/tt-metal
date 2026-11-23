# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP64 CPU chain of Kimi-K3.1 layers 0-3 from prompt token ids: golden inputs and outputs of MoE layers 1-2 and of
DSA layer 3, every AttnRes read included. CPU only; never imports ttnn.

A vLLM trace records neither the AttnRes state nor the module inputs of layers 1-3, so nothing below layer 3 can be
teacher-fed from it. This chain computes them instead, in FP64 from the token ids (no GPU activation enters it):

    embed_tokens -> layer 0 (KDA, dense FFN) -> layer 1 (KDA, MoE) -> layer 2 (KDA, MoE) -> layer 3 (DSA)

with the AttnRes residual bookkeeping of ``reference/kimi_k3/attn_res/hf_walk.py`` (block size 12: layer 0 seals the
embedding; every later read mixes the sealed snapshot and the live in-block sum). AttnRes acts row by row, so the
only state carried across rows is each KDA layer's recurrent and convolution state and DSA layer 3's keys.

The run is segment-major and resumable (``tests/kda/long_context_golden.py``'s manifest helpers): each segment of
rows passes through all four layers, then atomically writes its products (``segment_<i>.pt``, fp32) and the carried
state after it (``state_<i>.pt``: the three KDA states and the segment's DSA keys, FP64), so an interruption costs
one segment and a resume is bit-identical to an uninterrupted run (same thread count). Per row, the products are:

* MoE layers 1 and 2: the gate input ``moe_input_L`` (post ``post_attention_layernorm``, what the gate and the
  routed / shared experts read), ``moe_output_L`` (routed + shared, before the residual add), the FP64 routing
  ``routing_ids_L`` (top-16, best first), ``routing_weights_L`` and ``routing_margin_L`` (16th minus 17th choice
  score: how near the selection is to a swap).
* DSA layer 3: ``dsa_input_3``, the AttnRes read before ``input_layernorm``; the KV latent (``kv_a_layernorm``), the
  unrotated k_pe and the indexer key (state file, FP64); every row's top-2048 by the FP64 indexer score over all
  causal keys, best first (``topk_indices_3``, -1 where a row has fewer causal keys, and ``topk_scores_3``); and on
  the sampled rows only (quadratic otherwise) the FP64 gated attention output ``attn_output_3``.

Each segment is cross-checked against the GPU trace (bf16; differences are expected, nothing is gated on them
except the token embedding, which must equal the trace's ``decoder_input_layer_0`` bit for bit): layer 0 attention
and FFN outputs, routing recall of layers 1-2 (and 3 on sampled rows, whose FFN read only needs those rows'
attention), layer 3 keys (PCC, NRMSE) and top-k recall. The results go into the manifest's segment entries.

``--kda-precision fp32|vllm`` runs KDA layers 0-2 at another precision (fp32 reference; or vLLM's FlashKDA rounding
points, ``kda_vllm_emulation``) with everything else FP64, recorded in the manifest's model identity; scoring such a
run against the FP64 chain (``chain_vs_trace --candidate``) isolates what that KDA precision does to the products
over long context.

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain run --trace T --count N --out G
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict, required_kda_weight_names
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import pcc
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    MANIFEST,
    atomic_save,
    resume_manifest,
    segment_path,
    token_ids_sha256,
    write_manifest,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import dsa_fp64_golden as dsa
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_vllm_emulation import kda_flashkda_emulation
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import Trace, checkpoint_tensors, missed_keys, topk_recall

C = KimiK31Config
F64 = torch.float64
KDA_LAYERS, MOE_LAYERS, DSA_LAYER = (0, 1, 2), (1, 2), 3
# Bump when anything the run stores changes its bytes (math, rounding, layout of the payloads).
PRODUCER_VERSION = 2  # 2: buffered indexer blocks of 4096 keys (FP64 summation order of the scores)
# Keys per indexer score block. Each block's [rows x heads, keys] FP64 scores go into one reused buffer: a fresh
# allocation per block page-faults, and 4096 keys measured 48.7 M keys/s per row against 19.6 for unbuffered 128K
# blocks (16 threads, logs/indexer-buffers-probe-*). The rate is then the FP64 matmul's.
KEY_BLOCK = 4096
QUERY_BLOCK = 16  # query rows per indexer block
# Precision of the KDA layers 0-2; everything else stays FP64. "fp64" is the golden. "fp32" runs the same reference
# in fp32 (fp32 state); "vllm" emulates vLLM's FlashKDA rounding points (``kda_vllm_emulation``). The non-golden arms
# measure how a KDA precision alone moves the chain's products over long context.
KDA_PRECISIONS = ("fp64", "fp32", "vllm")


# --- math, FP64 ----------------------------------------------------------------------------------------------------


def kda_layer(precision: str, hidden: torch.Tensor, weights: dict, config: KDAConfig, state: KDAReferenceState | None):
    """One KDA layer over ``hidden`` ``[1, T, d]`` at ``precision`` (``KDA_PRECISIONS``); the output in FP64 and the
    carried state in the arm's own dtypes."""
    if precision == "fp64":
        return kda_forward_reference(hidden, weights, config, state, dtype=F64)
    if precision == "fp32":
        out, state = kda_forward_reference(hidden, weights, config, state, dtype=torch.float32)
    elif precision == "vllm":
        out, state = kda_flashkda_emulation(hidden, weights, config, state)
    else:
        raise ValueError(f"KDA precision {precision!r} not in {KDA_PRECISIONS}")
    return out.to(F64), state


rms64 = dsa.rms64  # RMSNorm in FP64: x rsqrt(mean(x^2) + eps) weight


def attn_res64(live: torch.Tensor, sealed: torch.Tensor, query: tuple[torch.Tensor, torch.Tensor], eps: float):
    """The AttnRes read in FP64: each candidate (the sealed snapshots ``[n, S, d]``, then the live sum ``[n, d]``)
    scores ``proj . rms_norm(candidate; norm)``; the read mixes the raw candidates by the softmax of those scores.
    ``query``: (``*_res_norm.weight``, ``*_res_proj.weight``)."""
    norm, proj = query
    candidates = torch.cat([sealed, live.unsqueeze(1)], dim=1)
    scores = rms64(candidates, norm, eps) @ proj.reshape(-1)
    return torch.einsum("ns,nsd->nd", scores.softmax(-1), candidates)


def situ64(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SiTU-and-mul with K3's beta / linear beta: ``b tanh(g / b) sigmoid(g) * lb tanh(u / lb)``."""
    beta, linear = C.ACTIVATION_SITU_BETA, C.ACTIVATION_SITU_LINEAR_BETA
    return beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * (linear * torch.tanh(up / linear))


def dense_mlp64(x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """Layer 0's dense FFN (and the shared experts): ``down(situ(gate x, up x))``."""
    return situ64(x @ w["gate_proj.weight"].T, x @ w["up_proj.weight"].T) @ w["down_proj.weight"].T


class MoE64:
    """A K3 latent MoE layer in FP64: sigmoid router with selection bias, top-k, weights renormalized x ROUTE_SCALE;
        routed experts at the latent width (down-project, the experts' weighted sum, ``routed_expert_norm``, up-project)
        plus the shared experts on the full hidden. The routed experts stay in their checkpoint dtype (bf16; FP64 for
        all 896 of a layer would be ~237 GB) and each is cast to FP64 when it computes its rows, one expert at a time,
    into reused buffers: a fresh 264 MB FP64 tensor per weight page-faults (35 ms per expert against 1.7 ms for a copy
    into a buffer, logs/cast-probe-*), which made the casts cost ~24 s per layer and segment.

        ``w``: the non-routed weights by their checkpoint names under ``block_sparse_moe.``; ``experts[e]``: routed
        expert e's (w1 gate, w3 up, w2 down)."""

    def __init__(self, w: dict[str, torch.Tensor], experts: list[tuple[torch.Tensor, ...]], top_k: int, eps: float):
        self.w = {n: t.to(F64) for n, t in w.items()}
        self.experts, self.top_k, self.eps = experts, top_k, eps
        self.buffers = [torch.empty(t.shape, dtype=F64) for t in experts[0]]

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(ids ``[n, k]`` best first, weights ``[n, k]``, margin ``[n]``: k-th minus (k+1)-th choice score)."""
        scores = (x @ self.w["gate.weight"].T).sigmoid()
        choice = scores + self.w["gate.e_score_correction_bias"]
        top = choice.topk(min(self.top_k + 1, choice.shape[-1]), dim=-1)
        ids = top.indices[:, : self.top_k]
        margin = (
            top.values[:, self.top_k - 1] - top.values[:, self.top_k]
            if top.values.shape[-1] > self.top_k
            else torch.full_like(top.values[:, 0], float("inf"))
        )
        weights = scores.gather(1, ids)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * C.ROUTE_SCALE
        return ids, weights, margin

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ids, weights, margin = self.route(x)
        latent = x @ self.w["routed_expert_down_proj.weight"].T
        y = torch.zeros_like(latent)
        for e in ids.unique().tolist():
            rows, slot = (ids == e).nonzero(as_tuple=True)
            w1, w3, w2 = (b.copy_(t) for b, t in zip(self.buffers, self.experts[e]))
            h = situ64(latent[rows] @ w1.T, latent[rows] @ w3.T) @ w2.T
            y.index_add_(0, rows, h * weights[rows, slot].unsqueeze(-1))
        y = rms64(y, self.w["routed_expert_norm.weight"], self.eps) @ self.w["routed_expert_up_proj.weight"].T
        shared = {k: self.w[f"shared_experts.{k}"] for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
        return {"output": y + dense_mlp64(x, shared), "ids": ids, "weights": weights, "margin": margin}


def index_topk(
    index_q: torch.Tensor, index_w: torch.Tensor, keys: torch.Tensor, rows: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``topk`` causal keys of query rows ``rows`` (absolute positions) by the FP64 indexer score
    ``sum_h w_h relu(q_h . k)`` (``dsa_fp64_golden.index_scores``, batched): (indices ``[n, topk]`` int64 best first,
    -1 past a row's causal keys; scores ``[n, topk]``, NaN there). ``keys``: the index keys of rows ``[0, >max(rows)]``.
    """
    n, heads, dim = index_q.shape
    causal = int(rows.max()) + 1
    scores = torch.empty(n, causal, dtype=F64)
    q, mix = index_q.reshape(n * heads, dim), index_w.unsqueeze(1)
    full = (torch.empty(n * heads, KEY_BLOCK, dtype=F64), torch.empty(n, 1, KEY_BLOCK, dtype=F64))
    for lo in range(0, causal, KEY_BLOCK):
        width = min(KEY_BLOCK, causal - lo)
        s, o = (
            full
            if width == KEY_BLOCK
            else (torch.empty(n * heads, width, dtype=F64), torch.empty(n, 1, width, dtype=F64))
        )
        torch.matmul(q, keys[lo : lo + width].T, out=s)
        torch.bmm(mix, s.clamp_min_(0).view(n, heads, width), out=o)
        scores[:, lo : lo + width] = o[:, 0]
    scores.masked_fill_(torch.arange(causal) > rows.unsqueeze(1), -float("inf"))
    top = scores.topk(min(topk, causal), dim=-1)
    indices = top.indices.masked_fill(top.values == -float("inf"), -1)
    values = top.values.masked_fill(indices < 0, float("nan"))
    pad = topk - indices.shape[1]
    return F.pad(indices, (0, pad), value=-1), F.pad(values, (0, pad), value=float("nan"))


def dsa_topk(
    hidden: torch.Tensor, w: dict, dims: dsa.Dims, eps: float, index_keys: torch.Tensor, start: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's FP64 indexer top-k (``index_topk``) for the normed DSA input ``hidden`` of rows
    ``[start, start + n)``, in blocks of ``QUERY_BLOCK`` rows; ``index_keys``: every row's index key so far."""
    indices, scores = [], []
    for lo in range(0, hidden.shape[0], QUERY_BLOCK):
        hi = min(lo + QUERY_BLOCK, hidden.shape[0])
        qr = rms64(hidden[lo:hi] @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"], eps)
        index_q = (qr @ w["indexer.wq_b.weight"].T).reshape(-1, dims.index_heads, dims.index_dim)
        index_w = (hidden[lo:hi] @ w["indexer.weights_proj.weight"].T) * (
            dims.index_heads**-0.5 * dims.index_dim**-0.5
        )
        i, s = index_topk(index_q, index_w, index_keys, torch.arange(start + lo, start + hi), dims.topk)
        indices.append(i)
        scores.append(s)
    return torch.cat(indices), torch.cat(scores)


# --- model ---------------------------------------------------------------------------------------------------------


@dataclass
class ChainModel:
    """Weights of layers 0-3 the chain reads (checkpoint dtypes; FP64 casts happen at use or once at load).

    ``layers[l]``: ``input_layernorm``, ``attn_res`` (norm, proj; l >= 1), ``mlp_res`` (norm, proj),
    ``post_attention_layernorm``, and ``kda`` (l in 0-2), ``mlp`` (l = 0, dense FFN), ``moe`` (l in 1-2, unless
    loaded without routed experts), ``dsa`` (l = 3) and ``gate`` / ``gate_bias`` (l in 1-3: the router alone)."""

    embedding: torch.Tensor
    layers: dict[int, dict]
    kda_config: KDAConfig
    dsa_dims: dsa.Dims
    eps: float
    block_size: int = C.ATTN_RES_BLOCK_SIZE
    top_k: int = C.NUM_EXPERTS_PER_TOKEN
    digests: dict[str, str] = field(default_factory=dict)
    kda_precision: str = "fp64"

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, threads: int = 16, log: Callable[[str], None] = print, routed_experts: bool = True
    ) -> "ChainModel":
        """``routed_experts=False`` leaves out the MoE layers' routed experts (~145 GB resident) and their ``moe``
        entry, for a consumer that reads the chain's stored MoE outputs instead; every other digest is the same."""
        config = json.loads((checkpoint / "config.json").read_text())
        kda_config = KDAConfig.from_model_config(config)
        eps = float(config.get("text_config", config)["rms_norm_eps"])
        embedding = checkpoint_tensors(checkpoint, None, ["embed_tokens.weight"])["embed_tokens.weight"]
        layers, digests = {}, {"embed_tokens": tensor_digest({"w": embedding})}
        for l in range(DSA_LAYER + 1):
            layers[l], d = load_layer(
                checkpoint,
                l,
                kda=l in KDA_LAYERS,
                dense=l == 0,
                moe=l in MOE_LAYERS,
                dsa_attn=l == DSA_LAYER,
                threads=threads,
                log=log,
                routed_experts=routed_experts,
            )
            digests.update(d)
        return cls(embedding, layers, kda_config, dsa.Dims(eps=eps), eps, digests=digests)

    def identity(self) -> dict:
        identity = {
            "kda_config": asdict(self.kda_config),
            "dsa_dims": asdict(self.dsa_dims),
            "eps": self.eps,
            "block_size": self.block_size,
            "weights_sha256": dict(sorted(self.digests.items())),
        }
        if self.kda_precision != "fp64":  # absent for the golden, so existing FP64 runs keep their identity
            identity["kda_precision"] = self.kda_precision
        return identity


def load_layer(
    checkpoint: Path,
    l: int,
    *,
    kda: bool,
    dense: bool,
    moe: bool,
    dsa_attn: bool,
    threads: int,
    log: Callable[[str], None] = print,
    routed_experts: bool = True,
) -> tuple[dict, dict[str, str]]:
    """Layer ``l``'s weights as ``ChainModel.layers[l]`` holds them, and their digests: the norms and AttnRes queries;
    ``kda`` (KDA attention), ``dsa_attn`` (DSA attention), ``dense`` (dense FFN) or ``moe`` (the full MoE). Every MoE
    or DSA layer also gets its router alone (``gate`` / ``gate_bias``: a DSA layer routes sampled rows for a
    cross-check, ``kda_chain_feed`` routes the MoE layers). ``routed_experts=False`` leaves out a MoE layer's routed
    experts and its ``moe`` entry (and their digest); every other digest is the same."""
    config = json.loads((checkpoint / "config.json").read_text())
    kda_config = KDAConfig.from_model_config(config)
    eps = float(config.get("text_config", config)["rms_norm_eps"])
    names = ["input_layernorm.weight", "mlp_res_norm.weight", "mlp_res_proj.weight"]
    names += ["post_attention_layernorm.weight", "self_attention_res_norm.weight"]
    names += ["self_attention_res_proj.weight"]
    if kda:
        names += [f"self_attn.{n}" for n in required_kda_weight_names(kda_config)]
    if dense:
        names += [f"mlp.{n}" for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")]
    if moe or dsa_attn:
        names += [f"block_sparse_moe.gate.{n}" for n in ("weight", "e_score_correction_bias")]
    if moe:
        names += [f"block_sparse_moe.{n}" for n in MoE_NON_ROUTED if not n.startswith("gate.")]
    if dsa_attn:
        names += [f"self_attn.{n}" for n in dsa.WEIGHTS]
    t = checkpoint_tensors(checkpoint, l, names)
    digests = {f"layers.{l}": tensor_digest(t)}
    layer = {
        "input_layernorm": t["input_layernorm.weight"].to(F64),
        "attn_res": (t["self_attention_res_norm.weight"].to(F64), t["self_attention_res_proj.weight"].to(F64)),
        "mlp_res": (t["mlp_res_norm.weight"].to(F64), t["mlp_res_proj.weight"].to(F64)),
        "post_attention_layernorm": t["post_attention_layernorm.weight"].to(F64),
    }
    if kda:
        weights = {n: t[f"self_attn.{n}"] for n in required_kda_weight_names(kda_config)}
        layer["kda"] = {n: v.to(F64) for n, v in normalize_kda_state_dict(weights, kda_config).items()}
    if dense:
        layer["mlp"] = {n: t[f"mlp.{n}"].to(F64) for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")}
    if moe and routed_experts:
        num_experts = int(config.get("text_config", config)["num_experts"])
        experts, digests[f"layers.{l}.experts"] = load_experts(checkpoint, l, num_experts, threads, log)
        w = {n: t[f"block_sparse_moe.{n}"] for n in MoE_NON_ROUTED}
        layer["moe"] = MoE64(w, experts, C.NUM_EXPERTS_PER_TOKEN, eps)
    if dsa_attn:
        layer["dsa"] = {n: t[f"self_attn.{n}"].to(F64) for n in dsa.WEIGHTS}
    if moe or dsa_attn:
        layer["gate"] = t["block_sparse_moe.gate.weight"].to(F64)
        layer["gate_bias"] = t["block_sparse_moe.gate.e_score_correction_bias"].to(F64)
    log(f"loaded layer {l} weights")
    return layer, digests


MoE_NON_ROUTED = (
    "gate.weight",
    "gate.e_score_correction_bias",
    "routed_expert_down_proj.weight",
    "routed_expert_up_proj.weight",
    "routed_expert_norm.weight",
    "shared_experts.gate_proj.weight",
    "shared_experts.up_proj.weight",
    "shared_experts.down_proj.weight",
)


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    """Content identity of named tensors (names, dtypes, shapes, bytes)."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(memoryview(t.view(torch.uint8).numpy()))
    return h.hexdigest()


def load_experts(checkpoint: Path, layer: int, count: int, threads: int, log: Callable[[str], None]):
    """The ``count`` routed experts of ``layer`` in their checkpoint dtype, ``[(w1, w3, w2)]`` by expert, read in parallel groups,
    and the digest of their per-expert digests."""
    began = time.perf_counter()

    def group(lo: int) -> list[tuple[tuple[torch.Tensor, ...], str]]:
        names = [
            f"block_sparse_moe.experts.{e}.{m}.weight"
            for e in range(lo, min(lo + 32, count))
            for m in ("w1", "w3", "w2")
        ]
        t = checkpoint_tensors(checkpoint, layer, names)
        out = []
        for e in range(lo, min(lo + 32, count)):
            ws = {m: t[f"block_sparse_moe.experts.{e}.{m}.weight"] for m in ("w1", "w3", "w2")}
            out.append((tuple(ws.values()), tensor_digest(ws)))
        return out

    experts, digests = [], []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, part in enumerate(pool.map(group, range(0, count, 32))):
            experts += [e for e, _ in part]
            digests += [d for _, d in part]
            if i % 7 == 6:
                log(f"layer {layer}: {len(experts)}/{count} experts read {time.perf_counter() - began:.0f} s")
    log(f"layer {layer}: {len(experts)} experts read in {time.perf_counter() - began:.0f} s")
    return experts, hashlib.sha256("".join(digests).encode()).hexdigest()


# --- one segment through the chain ---------------------------------------------------------------------------------


@dataclass
class Carried:
    """What a segment hands the next: KDA states (by layer) and DSA layer 3's FP64 keys of every row so far."""

    kda: dict[int, KDAReferenceState | None]
    keys: dict[str, list[torch.Tensor]]

    @classmethod
    def empty(cls) -> "Carried":
        return cls({l: None for l in KDA_LAYERS}, {k: [] for k in ("latent", "kpe", "index_key")})


def chain_segment(
    model: ChainModel,
    token_ids: torch.Tensor,
    start: int,
    carried: Carried,
    sampled: list[int],
    stage: Callable[[str], None] = lambda name: None,
) -> tuple[dict, dict]:
    """Rows ``[start, start + len(token_ids))`` through layers 0-3; updates ``carried`` in place. Returns (products,
    cross-check values for this segment's rows: layer 0 outputs and the sampled rows' layer 3 routing)."""
    eps, dims = model.eps, model.dsa_dims
    emb = model.embedding[token_ids.long()]
    x = emb.to(F64)
    out, extra = {}, {"embedding": emb}
    sealed = x.new_zeros(x.shape[0], 0, x.shape[1])
    live = x
    for l in KDA_LAYERS:
        L = model.layers[l]
        hidden = live if sealed.shape[1] == 0 else attn_res64(live, sealed, L["attn_res"], eps)
        hidden = rms64(hidden, L["input_layernorm"], eps)
        if l % model.block_size == 0:
            sealed, live = torch.cat([sealed, live.unsqueeze(1)], dim=1), None
        attn, carried.kda[l] = kda_layer(
            model.kda_precision, hidden.unsqueeze(0), L["kda"], model.kda_config, carried.kda[l]
        )
        attn = attn[0]
        live = attn if live is None else live + attn
        stage(f"layer {l} kda")
        normed = rms64(attn_res64(live, sealed, L["mlp_res"], eps), L["post_attention_layernorm"], eps)
        if l == 0:
            ffn = dense_mlp64(normed, L["mlp"])
            extra.update({"attn_output_0": attn, "ffn_output_0": ffn})
            stage("layer 0 dense ffn")
        else:
            moe = L["moe"](normed)
            ffn = moe["output"]
            out.update({f"moe_input_{l}": normed, f"moe_output_{l}": ffn, f"routing_ids_{l}": moe["ids"].int()})
            out.update({f"routing_weights_{l}": moe["weights"], f"routing_margin_{l}": moe["margin"]})
            stage(f"layer {l} moe")
        live = live + ffn
    L = model.layers[DSA_LAYER]
    read = attn_res64(live, sealed, L["attn_res"], eps)
    hidden = rms64(read, L["input_layernorm"], eps)
    keys = dsa.layer_keys(hidden, L["dsa"], dims)
    for k, v in keys.items():
        carried.keys[k].append(v)
    index_keys = torch.cat(carried.keys["index_key"])
    out["dsa_input_3"] = read
    stage("layer 3 keys")
    n = x.shape[0]
    indices, out["topk_scores_3"] = dsa_topk(hidden, L["dsa"], dims, eps, index_keys, start)
    out["topk_indices_3"] = indices.int()
    stage("layer 3 indexer top-k")
    local = [r - start for r in sampled if start <= r < start + n]
    out["sampled_rows"] = torch.tensor([start + r for r in local], dtype=torch.int64)
    if local:
        latent, kpe = torch.cat(carried.keys["latent"]), torch.cat(carried.keys["kpe"])
        q = dsa.layer_queries(hidden[local], L["dsa"], dims)
        attn_rows = []
        for i, r in enumerate(local):
            sel = out["topk_indices_3"][r].long()
            sel = sel[sel >= 0]
            attn_rows.append(dsa.attend(q, i, latent[sel], kpe[sel], L["dsa"], dims))
        out["attn_output_3"] = torch.stack(attn_rows)
        normed = rms64(
            attn_res64(live[local] + out["attn_output_3"], sealed[local], L["mlp_res"], eps),
            L["post_attention_layernorm"],
            eps,
        )
        gate = (normed @ L["gate"].T).sigmoid().add(L["gate_bias"])
        extra["routing_ids_3_sampled"] = gate.topk(model.top_k, dim=-1).indices
    else:
        out["attn_output_3"] = hidden.new_zeros(0, hidden.shape[1])
        extra["routing_ids_3_sampled"] = torch.zeros(0, model.top_k, dtype=torch.int64)
    stage("layer 3 sampled attention")
    return out, extra


# Products stored FP64 (small, or exact for a downstream FP64 consumer); every other float product is stored fp32.
KEEP_FP64 = ("attn_output_3",)
SAMPLED = ("sampled_rows", "attn_output_3")  # products of the sampled rows only
KEYS = ("latent", "kpe", "index_key")  # per-row products held in the state files


def storable(products: dict) -> dict:
    return {k: v.to(torch.float32) if v.dtype == F64 and k not in KEEP_FP64 else v for k, v in products.items()}


# --- cross-checks against the GPU trace ----------------------------------------------------------------------------


def nrmse(got: torch.Tensor, want: torch.Tensor) -> float:
    """||got - want|| / ||want|| over the whole tensor, in FP64."""
    return float((got.double() - want.double()).norm() / want.double().norm())


def trace_checks(trace: Trace, start: int, end: int, products: dict, extra: dict, keys: dict, dims: dsa.Dims) -> dict:
    """This segment's chain against the trace (bf16 GPU streams; cross-checks, not targets)."""
    lora, topk = dims.lora, dims.topk
    emb = trace.rows("decoder_input_layer_0", start, end)
    c = {"embedding_ulp_mismatches": int((emb.view(torch.int16) != extra["embedding"].view(torch.int16)).sum())}
    for name, stream in (("attn_output_0", "attn_output_layer_0"), ("ffn_output_0", "decoder_output_layer_0")):
        want = extra[name]
        got = trace.rows(stream, start, end)
        c[f"{name}_pcc"], c[f"{name}_nrmse"] = pcc(want, got), nrmse(got, want)
        rows = (got.double() - want).norm(dim=-1) / want.norm(dim=-1)
        c[f"{name}_row_nrmse_median"] = float(rows.median())
    for l in MOE_LAYERS:
        recall = topk_recall(
            products[f"routing_ids_{l}"].long(), trace.rows(f"expert_ids_layer_{l}", start, end).long()
        )
        c[f"routing_recall_{l}"], c[f"routing_set_equal_{l}"] = float(recall.mean()), float(
            (recall == 1).float().mean()
        )
    if products["sampled_rows"].numel():
        want = torch.cat([trace.rows("expert_ids_layer_3", r, r + 1) for r in products["sampled_rows"].tolist()])
        recall = topk_recall(extra["routing_ids_3_sampled"], want.long())
        c["routing_recall_3_sampled"], c["routing_sampled_rows"] = float(recall.mean()), int(recall.numel())
    kv = trace.rows(f"kv_post_transform_layer_{DSA_LAYER}", start, end)
    ik = trace.rows(f"indexer_k_layer_{DSA_LAYER}", start, end)
    for name, got, want in (
        ("kv_latent", kv[:, :lora], keys["latent"]),
        ("k_pe", kv[:, lora:], keys["kpe"]),
        ("index_key", ik, keys["index_key"]),
    ):
        c[f"{name}_pcc"], c[f"{name}_nrmse"] = pcc(want, got), nrmse(got, want)
    trace_sel = trace.rows(f"dsa_topk_indices_layer_{DSA_LAYER}", start, end).long()
    golden_sel = products["topk_indices_3"].long()
    recall = topk_recall(trace_sel, golden_sel)
    full = torch.arange(start, end) >= topk - 1  # rows with a full selection (row + 1 >= top-k keys)
    c["topk_recall_min_partial"] = float(recall[~full].min()) if (~full).any() else float("nan")
    if full.any():
        c["topk_recall_mean"], c["topk_recall_p01"] = float(recall[full].mean()), float(recall[full].quantile(0.01))
        rank = missed_keys(trace_sel[full], golden_sel[full], products["topk_scores_3"][full].double())["rank"]
        c["topk_misses"], c["topk_miss_top_half_fraction"] = int(rank.numel()), (
            float((2 * rank < topk).float().mean()) if rank.numel() else 0.0
        )
    return c


# --- the run ---------------------------------------------------------------------------------------------------------


def state_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"state_{index:06d}.pt"


def load_carried(run_dir: Path, done: int) -> Carried:
    carried = Carried.empty()
    for index in range(done):
        state = torch.load(state_path(run_dir, index), weights_only=True)
        for k in carried.keys:
            carried.keys[k].append(state["keys"][k])
    if done:
        carried.kda = {int(l): KDAReferenceState(**s) for l, s in state["kda"].items()}
    return carried


def run_chain(
    *,
    run_dir: Path,
    model: ChainModel,
    token_ids: torch.Tensor,
    segment: int,
    sampled: list[int],
    identity: dict,
    trace: Trace | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (or resume) the chain over ``token_ids`` and return the manifest. A resume refuses a directory whose
    identity, tokens, segment size, sampled rows or thread count differ (the thread count fixes FP64 summation
    order, so a resumed run stays bit-identical to an uninterrupted one)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    count = int(token_ids.numel())
    expected = {
        "producer": "fp64_chain",
        "producer_version": PRODUCER_VERSION,
        "identity": identity,
        "model": model.identity(),
        "count": count,
        "token_ids_sha256": token_ids_sha256(token_ids.tolist()),
        "segment": segment,
        "sampled_rows": sampled,
        "dtype": "float64",
        "threads": torch.get_num_threads(),
    }
    segments = -(-count // segment)
    manifest = resume_manifest(run_dir, expected, segments)
    done = len(manifest["segments"])
    carried = load_carried(run_dir, done)
    if done:
        log(f"resuming at segment {done}/{segments} (row {manifest['segments'][-1]['end']})")
    for index in range(done, segments):
        start, end = index * segment, min((index + 1) * segment, count)
        began = last = time.perf_counter()
        stages = {}

        def stage(name: str) -> None:
            nonlocal last
            now = time.perf_counter()
            stages[name] = round(now - last, 2)
            last = now
            log(f"segment {index + 1}/{segments} rows [{start}, {end}) {name} {stages[name]:.1f} s")

        products, extra = chain_segment(model, token_ids[start:end], start, carried, sampled, stage)
        keys = {k: v[-1] for k, v in carried.keys.items()}
        payload = {"start": start, "end": end, **storable(products)}
        atomic_save(payload, segment_path(run_dir, index))
        atomic_save(
            {"start": start, "end": end, "keys": keys, "kda": {l: vars(s) for l, s in carried.kda.items()}},
            state_path(run_dir, index),
        )
        stage("save")
        entry = {"start": start, "end": end, "seconds": round(time.perf_counter() - began, 2), "stages": stages}
        if trace is not None:
            entry["checks"] = trace_checks(trace, start, end, products, extra, keys, model.dsa_dims)
            stage("trace checks")
            if entry["checks"]["embedding_ulp_mismatches"]:
                raise ValueError(f"rows [{start}, {end}): the token embedding differs from the trace's: {entry}")
        entry["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        if "bytes_per_row" not in manifest:  # per-row streams only (not the sampled rows' attention)
            per_row = {k: v for k, v in payload.items() if torch.is_tensor(v) and k not in SAMPLED}
            per_row.update({f"state.{k}": v for k, v in keys.items()})
            manifest["bytes_per_row"] = {k: v.element_size() * v[0].numel() for k, v in per_row.items()}
        manifest["segments"].append(entry)
        write_manifest(run_dir, manifest)
        log(
            f"segment {index + 1}/{segments} done {entry['seconds']:.1f} s ({(end - start) / max(entry['seconds'], 0.01):.1f} rows/s) "
            f"rss {entry['max_rss_gb']} GB " + json.dumps(entry.get("checks", {}))
        )
    return manifest


def load_products(run_dir: Path, names: list[str], start: int = 0, end: int | None = None) -> dict[str, torch.Tensor]:
    """Per-row products (segment files) and keys (``latent`` / ``kpe`` / ``index_key``, state files) of rows
    ``[start, end)``; the run must hold them."""
    if any(n in SAMPLED for n in names):
        raise ValueError(f"{[n for n in names if n in SAMPLED]} are sampled-row products; read them with load_sampled")
    manifest = json.loads((Path(run_dir) / MANIFEST).read_text())
    done = manifest["segments"][-1]["end"] if manifest["segments"] else 0
    end = done if end is None else end
    if not 0 <= start < end <= done:
        raise ValueError(f"rows [{start}, {end}) are not inside the {done} rows of {run_dir}")
    seg = manifest["segment"]
    parts: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    for index in range(start // seg, -(-end // seg)):
        lo = index * seg
        # Memory-mapped, and only the files holding a requested name: a segment file is ~0.6 GB at 4096 rows, so a
        # full read of 255 of them is a ~160 GB pass for a few columns (review r1 m1). Slices are copied by the cat.
        data = {}
        if any(n not in KEYS for n in names):
            data.update(torch.load(segment_path(run_dir, index), weights_only=True, mmap=True))
        if any(n in KEYS for n in names):
            data.update(torch.load(state_path(run_dir, index), weights_only=True, mmap=True)["keys"])
        for n in names:
            parts[n].append(data[n][max(start, lo) - lo : end - lo])
    return {n: torch.cat(v) for n, v in parts.items()}


def load_sampled(run_dir: Path) -> dict[str, torch.Tensor]:
    """The sampled-row products of every complete segment: ``sampled_rows`` (absolute rows, ascending) and
    ``attn_output_3`` (FP64 ``[rows, hidden]``)."""
    manifest = json.loads((Path(run_dir) / MANIFEST).read_text())
    parts = []
    for i in range(len(manifest["segments"])):
        p = torch.load(segment_path(run_dir, i), weights_only=True, mmap=True)  # only the sampled rows are read
        parts.append({k: p[k].clone() for k in SAMPLED})
    return {k: torch.cat([p[k] for p in parts]) for k in SAMPLED}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--trace", type=Path, required=True, help="vLLM trace: token ids and the cross-check streams")
    parser.add_argument("--count", type=int, help="prompt rows to run (default: the trace's prompt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--segment", type=int, default=4096)
    parser.add_argument("--sampled", type=int, default=512, help="rows with the layer 3 attention output")
    parser.add_argument("--threads", type=int, default=16, help="CPU threads; a resume must use the same count")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get("KIMI_K31_HF_MODEL", "/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized")
        ),
    )
    parser.add_argument("--no-checks", action="store_true", help="skip the per-segment trace cross-checks")
    parser.add_argument("--kda-precision", choices=KDA_PRECISIONS, default="fp64", help="KDA layers 0-2 only")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    trace = Trace(args.trace)
    token_ids = torch.tensor(trace.metadata["token_ids"][: trace.prompt_len], dtype=torch.int64)
    count = args.count or trace.prompt_len
    if count > token_ids.numel():
        raise ValueError(f"{trace.path} has {token_ids.numel()} prompt tokens, fewer than --count {count}")
    log(f"loading weights from {args.checkpoint}")
    model = ChainModel.from_checkpoint(args.checkpoint, args.threads, log)
    model.kda_precision = args.kda_precision
    log(f"weights loaded; rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB")
    identity = {
        "model": "Kimi-K3.1",
        "checkpoint": str(args.checkpoint),
        "token_source": f"vllm trace {trace.path.name} metadata token_ids[:{count}]",
        "trace_commit": trace.metadata.get("trace_commit"),
    }
    run_chain(
        run_dir=args.out,
        model=model,
        token_ids=token_ids[:count],
        segment=args.segment,
        sampled=dsa.sample_rows(count, args.sampled, 0),
        identity=identity,
        trace=None if args.no_checks else trace,
        log=log,
    )


if __name__ == "__main__":
    main()
