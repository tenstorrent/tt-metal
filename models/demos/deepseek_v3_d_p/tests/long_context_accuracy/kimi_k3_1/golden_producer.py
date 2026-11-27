# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 FP64 chain golden producer: layers 0-3 from prompt token ids on CPU, the golden inputs and outputs of MoE
layers 1-2 and of DSA layer 3, every AttnRes read included. CPU only; never imports ttnn.

A vLLM trace records neither the AttnRes state nor the module inputs of layers 1-3, so nothing below layer 3 can be
teacher-fed from it. This chain computes them instead, in FP64 from the token ids (no GPU activation enters it):

    embed_tokens -> layer 0 (KDA, dense FFN) -> layer 1 (KDA, MoE) -> layer 2 (KDA, MoE) -> layer 3 (DSA)

with the AttnRes residual bookkeeping of ``reference/kimi_k3/attn_res/hf_walk.py`` (block size 12: layer 0 seals the
embedding; every later read mixes the sealed snapshot and the live in-block sum). AttnRes acts row by row, so the
only state carried across rows is each KDA layer's recurrent and convolution state and DSA layer 3's keys.

The run is segment-major and resumable (``fp64_golden.start_or_resume``): each segment of rows passes through all four
layers, then atomically writes its products (``segment_<i>.pt``, fp32) and the carried state after it
(``state_<i>.pt``: the three KDA states and the segment's DSA keys, FP64), so an interruption costs one segment and a
resume is bit-identical to an uninterrupted run (same thread count). Per row, the products are:

* MoE layers 1 and 2: the gate input ``moe_input_L`` (post ``post_attention_layernorm``, what the gate and the
  routed / shared experts read), ``moe_output_L`` (routed + shared, before the residual add), the FP64 routing
  ``routing_ids_L`` (top-16, best first), ``routing_weights_L`` and ``routing_margin_L`` (16th minus 17th choice
  score: how near the selection is to a swap).
* DSA layer 3: ``dsa_input_3``, the AttnRes read before ``input_layernorm``; the KV latent (``kv_a_layernorm``), the
  unrotated k_pe and the indexer key (state file, FP64); every row's top-2048 by the FP64 indexer score over all
  causal keys, best first (``topk_indices_3``, -1 where a row has fewer causal keys, and ``topk_scores_3``); and on
  the sampled rows only (quadratic otherwise) the FP64 gated attention output ``attn_output_3``.

The golden is written in the long-context golden registry's format (``common/prefill/long_context_accuracy/
fp64_golden.py``; its id is its directory name); the reader is ``golden.py`` next to this file. The manifest's
``producer`` is ``PRODUCER``, ``fp64_chain`` (version 2): the name this chain had (``tests/kimi_k3/fp64_chain.py`` in
draft PR #66789) when it wrote the registered golden ``k31-l0-3-fp64-1m``. It names the producer's numerics, not the
module's location, so it stays while the bytes do; the reader requires it. That golden was written before the
registry existed and registered by rewriting its manifest (``manifest.orig.json`` holds the original), so its
manifest differs from what this module writes in form only: ``count`` became ``rows``, ``model.weights_sha256`` moved
to ``identity.weights_sha256``, and ``golden_id`` and ``products`` were added.

Each segment is cross-checked against the GPU trace (bf16; differences are expected, nothing is gated on them
except the token embedding, which must equal the trace's ``decoder_input_layer_0`` bit for bit): layer 0 attention
and FFN outputs, routing recall of layers 1-2 (and 3 on sampled rows, whose FFN read only needs those rows'
attention), layer 3 keys (PCC, NRMSE) and top-k recall. The results go into the manifest's segment entries.

    nice -n 19 python -m models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer \\
        --trace T --count 1044480 --threads 32 --out /mnt/weka/model-cache/scratch/accuracy-goldens/k31-l0-3-fp64-1m

``--segments N`` stops after N segments; a later run resumes.
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
from safetensors import safe_open

from models.demos.common.prefill.long_context_accuracy.fp64_golden import (
    manifest_problems,
    product_schema,
    publish,
    start_or_resume,
    token_ids_sha256,
    write_manifest,
)
from models.demos.common.prefill.long_context_accuracy.topk_selection import missed_keys, topk_recall
from models.demos.common.prefill.long_context_accuracy.windowed_errors import nrmse, pcc, row_nrmse, row_sums
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict, required_kda_weight_names
from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config

C = KimiK31Config
F64 = torch.float64
PRODUCER = "fp64_chain"
# Bump when anything the run stores changes its bytes (math, rounding, layout of the payloads).
PRODUCER_VERSION = 2  # 2: buffered indexer blocks of 4096 keys (FP64 summation order of the scores)
KDA_LAYERS, MOE_LAYERS, DSA_LAYER = (0, 1, 2), (1, 2), 3
# Keys per indexer score block. Each block's [rows x heads, keys] FP64 scores go into one reused buffer: a fresh
# allocation per block page-faults, and 4096 keys measured 48.7 M keys/s per row against 19.6 for unbuffered 128K
# blocks (16 threads, logs/indexer-buffers-probe-*). The rate is then the FP64 matmul's.
KEY_BLOCK = 4096
QUERY_BLOCK = 16  # query rows per indexer block
SAMPLED_ROWS = 512  # rows with the layer 3 attention output, evenly spread over the prompt

# Layer 3's attention weights under the checkpoint's ``self_attn.``.
DSA_WEIGHTS = (
    "q_a_proj.weight",  # q down
    "q_a_layernorm.weight",  # q norm
    "q_b_proj.weight",  # q up (96 x (128 nope + 64 rope))
    "kv_a_proj_with_mqa.weight",  # kv down (512 latent + 64 rope)
    "kv_a_layernorm.weight",  # latent norm
    "kv_b_proj.weight",  # kv up (96 x (128 k + 128 v))
    "g_proj.weight",  # output gate
    "o_proj.weight",  # output projection
    "indexer.wq_b.weight",  # index query (32 x 128)
    "indexer.wk.weight",  # index key
    "indexer.k_norm.weight",  # index key LayerNorm weight
    "indexer.k_norm.bias",  # index key LayerNorm bias
    "indexer.weights_proj.weight",  # index head weights
)
# A MoE layer's non-routed weights under the checkpoint's ``block_sparse_moe.``.
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
DENSE_MLP = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")


@dataclass(frozen=True)
class Dims:
    """Layer 3's dimensions; K3.1's by default, smaller ones in the tests."""

    heads: int = C.NUM_ATTENTION_HEADS
    nope: int = C.QK_NOPE_HEAD_DIM
    rope: int = C.QK_ROPE_HEAD_DIM
    v: int = C.V_HEAD_DIM
    lora: int = C.KV_LORA_RANK
    index_heads: int = C.INDEX_N_HEADS
    index_dim: int = C.INDEX_HEAD_DIM
    topk: int = C.INDEX_TOPK
    eps: float = C.RMS_NORM_EPS


# --- math, FP64 ----------------------------------------------------------------------------------------------------


def rms64(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in FP64: ``x rsqrt(mean(x^2) + eps) weight``."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


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
    plus the shared experts on the full hidden. The routed experts stay in their checkpoint dtype (bf16; FP64 for all
    896 of a layer would be ~237 GB) and each is cast to FP64 when it computes its rows, one expert at a time, into
    reused buffers: a fresh 264 MB FP64 tensor per weight page-faults (35 ms per expert against 1.7 ms for a copy into a
    buffer, logs/cast-probe-*), which made the casts cost ~24 s per layer and segment.

    ``w``: the non-routed weights by their checkpoint names under ``block_sparse_moe.``; ``experts[e]``: routed expert
    e's (w1 gate, w3 up, w2 down)."""

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
        shared = {k: self.w[f"shared_experts.{k}"] for k in DENSE_MLP}
        return {"output": y + dense_mlp64(x, shared), "ids": ids, "weights": weights, "margin": margin}


def layer_keys(x: torch.Tensor, w: dict, dims: Dims) -> dict[str, torch.Tensor]:
    """Layer 3's keys per row of the normed input ``x`` (any float dtype, computed in FP64): KV latent, k_pe, indexer
    key (LayerNorm with bias)."""
    x = x.double()
    kv = x @ w["kv_a_proj_with_mqa.weight"].T
    latent = rms64(kv[:, : dims.lora], w["kv_a_layernorm.weight"], dims.eps)
    key = F.layer_norm(
        x @ w["indexer.wk.weight"].T, (dims.index_dim,), w["indexer.k_norm.weight"], w["indexer.k_norm.bias"], dims.eps
    )
    return {"latent": latent, "kpe": kv[:, dims.lora :], "index_key": key}


def layer_queries(x: torch.Tensor, w: dict, dims: Dims) -> dict[str, torch.Tensor]:
    """Layer 3's attention query (nope, rope) and output gate per row of the normed input ``x``, in FP64."""
    x = x.double()
    qr = rms64(x @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"], dims.eps)
    q = (qr @ w["q_b_proj.weight"].T).reshape(-1, dims.heads, dims.nope + dims.rope)
    return {"q_nope": q[..., : dims.nope], "q_pe": q[..., dims.nope :], "gate": torch.sigmoid(x @ w["g_proj.weight"].T)}


def attend(q: dict, row: int, latent: torch.Tensor, kpe: torch.Tensor, w: dict, dims: Dims) -> torch.Tensor:
    """Gated MLA output ``[hidden]`` of query row ``row`` of ``q`` (``layer_queries``) over the given keys
    (``latent`` [n, lora], ``kpe`` [n, rope]), absorbed: q_nope W_k scores the latent and W_v maps the latent mix;
    softmax scale (nope + rope)^-0.5, NoPE."""
    kvb = w["kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)
    q_latent = torch.einsum("hd,hdl->hl", q["q_nope"][row], kvb[:, : dims.nope])
    scores = (q_latent @ latent.T + q["q_pe"][row] @ kpe.T) * (dims.nope + dims.rope) ** -0.5
    mixed = scores.softmax(-1) @ latent  # [heads, lora]
    out = torch.einsum("hl,hvl->hv", mixed, kvb[:, dims.nope :]).reshape(-1) * q["gate"][row]
    return out @ w["o_proj.weight"].T


def index_topk(
    index_q: torch.Tensor, index_w: torch.Tensor, keys: torch.Tensor, rows: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``topk`` causal keys of query rows ``rows`` (absolute positions) by the FP64 indexer score
    ``sum_h w_h relu(q_h . k)``, batched: (indices ``[n, topk]`` int64 best first, -1 past a row's causal keys; scores
    ``[n, topk]``, NaN there). ``keys``: the index keys of rows ``[0, >max(rows)]``."""
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


def sample_rows(count: int, n: int) -> list[int]:
    """``n`` rows evenly spread over ``[0, count)``, both ends included, distinct."""
    return sorted(set(torch.linspace(0, count - 1, n).round().long().tolist()))


# --- model ---------------------------------------------------------------------------------------------------------


def checkpoint_tensors(checkpoint: Path, layer: int | None, names: list[str]) -> dict[str, torch.Tensor]:
    """Tensors ``layers.<layer>.<name>`` (``layer`` None: ``<name>`` under the model root, e.g. ``embed_tokens.weight``)
    of a K3 / K3.1 checkpoint (either key root), keyed by ``name``, read through its safetensors index opening each
    shard once."""
    weight_map = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    base = "" if layer is None else f"layers.{layer}."
    roots = [r for r in ("model.", "language_model.model.") if f"{r}{base}{names[0]}" in weight_map]
    if not roots:
        raise ValueError(f"{checkpoint} has no {base}{names[0]}")
    full = {f"{roots[0]}{base}{n}": n for n in names}
    by_shard: dict[str, list[str]] = {}
    for key in full:
        by_shard.setdefault(weight_map[key], []).append(key)
    out = {}
    for shard, keys in by_shard.items():
        with safe_open(str(Path(checkpoint) / shard), framework="pt") as handle:
            out.update({full[k]: handle.get_tensor(k) for k in keys})
    return out


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    """Content identity of named tensors (names, dtypes, shapes, bytes)."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(memoryview(t.view(torch.uint8).numpy()))
    return h.hexdigest()


def layer_weight_names(layer: int, kda_config: KDAConfig | None) -> list[str]:
    """Layer-local checkpoint names of the weights the chain reads from layer ``layer`` and digests as
    ``layers.<layer>`` (the routed experts are digested apart, as ``layers.<layer>.experts``). ``kda_config`` names
    the KDA weights of layers ``KDA_LAYERS``; layer ``DSA_LAYER`` needs none."""
    names = ["input_layernorm.weight", "mlp_res_norm.weight", "mlp_res_proj.weight"]
    names += ["post_attention_layernorm.weight", "self_attention_res_norm.weight", "self_attention_res_proj.weight"]
    if layer in KDA_LAYERS:
        names += [f"self_attn.{n}" for n in required_kda_weight_names(kda_config)]
    if layer == 0:
        names += [f"mlp.{n}" for n in DENSE_MLP]
    if layer in MOE_LAYERS or layer == DSA_LAYER:  # layer 3's router routes the sampled rows (a cross-check)
        names += [f"block_sparse_moe.{n}" for n in MoE_NON_ROUTED[:2]]
    if layer in MOE_LAYERS:
        names += [f"block_sparse_moe.{n}" for n in MoE_NON_ROUTED[2:]]
    if layer == DSA_LAYER:
        names += [f"self_attn.{n}" for n in DSA_WEIGHTS]
    return names


@dataclass
class ChainModel:
    """Weights of layers 0-3 the chain reads (checkpoint dtypes; FP64 casts happen at use or once at load).

    ``layers[l]``: ``input_layernorm``, ``attn_res`` (norm, proj; l >= 1), ``mlp_res`` (norm, proj),
    ``post_attention_layernorm``, and ``kda`` (l in 0-2), ``mlp`` (l = 0, dense FFN), ``moe`` (l in 1-2), ``dsa`` and
    ``gate`` / ``gate_bias`` (l = 3; the gate only to route sampled rows for the cross-check)."""

    embedding: torch.Tensor
    layers: dict[int, dict]
    kda_config: KDAConfig
    dsa_dims: Dims
    eps: float
    block_size: int = C.ATTN_RES_BLOCK_SIZE
    top_k: int = C.NUM_EXPERTS_PER_TOKEN
    digests: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, threads: int = 16, log: Callable[[str], None] = print) -> "ChainModel":
        config = json.loads((checkpoint / "config.json").read_text())
        kda_config = KDAConfig.from_model_config(config)
        eps = float(config.get("text_config", config)["rms_norm_eps"])
        num_experts = int(config.get("text_config", config)["num_experts"])
        embedding = checkpoint_tensors(checkpoint, None, ["embed_tokens.weight"])["embed_tokens.weight"]
        layers, digests = {}, {"embed_tokens": tensor_digest({"w": embedding})}
        for l in range(DSA_LAYER + 1):
            t = checkpoint_tensors(checkpoint, l, layer_weight_names(l, kda_config))
            digests[f"layers.{l}"] = tensor_digest(t)
            layer = {
                "input_layernorm": t["input_layernorm.weight"].to(F64),
                "attn_res": (t["self_attention_res_norm.weight"].to(F64), t["self_attention_res_proj.weight"].to(F64)),
                "mlp_res": (t["mlp_res_norm.weight"].to(F64), t["mlp_res_proj.weight"].to(F64)),
                "post_attention_layernorm": t["post_attention_layernorm.weight"].to(F64),
            }
            if l in KDA_LAYERS:
                kda = {n: t[f"self_attn.{n}"] for n in required_kda_weight_names(kda_config)}
                layer["kda"] = {n: v.to(F64) for n, v in normalize_kda_state_dict(kda, kda_config).items()}
            if l == 0:
                layer["mlp"] = {n: t[f"mlp.{n}"].to(F64) for n in DENSE_MLP}
            if l in MOE_LAYERS:
                experts, digests[f"layers.{l}.experts"] = load_experts(checkpoint, l, num_experts, threads, log)
                w = {n: t[f"block_sparse_moe.{n}"] for n in MoE_NON_ROUTED}
                layer["moe"] = MoE64(w, experts, C.NUM_EXPERTS_PER_TOKEN, eps)
            if l == DSA_LAYER:
                layer["dsa"] = {n: t[f"self_attn.{n}"].to(F64) for n in DSA_WEIGHTS}
                layer["gate"] = t["block_sparse_moe.gate.weight"].to(F64)
                layer["gate_bias"] = t["block_sparse_moe.gate.e_score_correction_bias"].to(F64)
            layers[l] = layer
            log(f"loaded layer {l} weights")
        return cls(embedding, layers, kda_config, Dims(eps=eps), eps, digests=digests)

    def config(self) -> dict:
        """The manifest's ``model``: the configuration the products depend on besides the weights."""
        return {
            "kda_config": asdict(self.kda_config),
            "dsa_dims": asdict(self.dsa_dims),
            "eps": self.eps,
            "block_size": self.block_size,
        }


def load_experts(checkpoint: Path, layer: int, count: int, threads: int, log: Callable[[str], None]):
    """The ``count`` routed experts of ``layer`` in their checkpoint dtype, ``[(w1, w3, w2)]`` by expert, read in
    parallel groups, and the digest of their per-expert digests."""
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
        return cls({l: None for l in KDA_LAYERS}, {k: [] for k in KEYS})


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
        attn, carried.kda[l] = kda_forward_reference(
            hidden.unsqueeze(0), L["kda"], model.kda_config, carried.kda[l], dtype=F64
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
    keys = layer_keys(hidden, L["dsa"], dims)
    for k, v in keys.items():
        carried.keys[k].append(v)
    index_keys = torch.cat(carried.keys["index_key"])
    out["dsa_input_3"] = read
    stage("layer 3 keys")
    n = x.shape[0]
    indices, scores = [], []
    for lo in range(0, n, QUERY_BLOCK):
        hi = min(lo + QUERY_BLOCK, n)
        qr = rms64(hidden[lo:hi] @ L["dsa"]["q_a_proj.weight"].T, L["dsa"]["q_a_layernorm.weight"], eps)
        index_q = (qr @ L["dsa"]["indexer.wq_b.weight"].T).reshape(-1, dims.index_heads, dims.index_dim)
        index_w = (hidden[lo:hi] @ L["dsa"]["indexer.weights_proj.weight"].T) * (
            dims.index_heads**-0.5 * dims.index_dim**-0.5
        )
        i, s = index_topk(index_q, index_w, index_keys, torch.arange(start + lo, start + hi), dims.topk)
        indices.append(i)
        scores.append(s)
    out["topk_indices_3"], out["topk_scores_3"] = torch.cat(indices).int(), torch.cat(scores)
    stage("layer 3 indexer top-k")
    local = [r - start for r in sampled if start <= r < start + n]
    out["sampled_rows"] = torch.tensor([start + r for r in local], dtype=torch.int64)
    if local:
        latent, kpe = torch.cat(carried.keys["latent"]), torch.cat(carried.keys["kpe"])
        q = layer_queries(hidden[local], L["dsa"], dims)
        attn_rows = []
        for i, r in enumerate(local):
            sel = out["topk_indices_3"][r].long()
            sel = sel[sel >= 0]
            attn_rows.append(attend(q, i, latent[sel], kpe[sel], L["dsa"], dims))
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


def product_schemas(payload: dict, keys: dict, kda: dict) -> dict:
    """The manifest's ``products``: the segment file's row and sampled-row products, the state file's per-row keys
    and its KDA states after the segment (``kda.<layer>.<name>``, segment-end)."""
    schemas = {
        name: product_schema("sampled_row" if name in SAMPLED else "row", t)
        for name, t in payload.items()
        if torch.is_tensor(t)
    }
    schemas.update({name: product_schema("row", t) for name, t in keys.items()})
    for l, state in kda.items():
        schemas.update({f"kda.{l}.{name}": product_schema("segment_end", t) for name, t in state.items()})
    return dict(sorted(schemas.items()))


# --- cross-checks against the GPU trace ----------------------------------------------------------------------------


class Trace:
    """A ``chunked_group_a_v1`` vLLM trace: each stream a directory of row shards listed in ``index.json``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.streams = json.loads((self.path / "index.json").read_text())["tensor_streams"]
        self.metadata = json.loads((self.path / "metadata.json").read_text())

    @property
    def prompt_len(self) -> int:
        return int(self.metadata["n_prompt_tokens"])

    def rows(self, key: str, start: int, end: int) -> torch.Tensor:
        """Rows ``[start, end)`` of one stream in its stored dtype."""
        parts = []
        for chunk in self.streams[key]["chunks"]:
            lo, hi = chunk["row_start"], chunk["row_end"]
            if hi <= start or lo >= end:
                continue
            with safe_open(str(self.path / chunk["path"]), framework="pt") as handle:
                parts.append(handle.get_slice(key)[max(start, lo) - lo : min(end, hi) - lo])
        if not parts:
            raise ValueError(f"{self.path.name}: no {key} rows in [{start}, {end})")
        return torch.cat(parts)


def trace_checks(trace: Trace, start: int, end: int, products: dict, extra: dict, keys: dict, dims: Dims) -> dict:
    """This segment's chain against the trace (bf16 GPU streams; cross-checks, not targets)."""
    lora, topk = dims.lora, dims.topk
    emb = trace.rows("decoder_input_layer_0", start, end)
    c = {"embedding_ulp_mismatches": int((emb.view(torch.int16) != extra["embedding"].view(torch.int16)).sum())}
    for name, stream in (("attn_output_0", "attn_output_layer_0"), ("ffn_output_0", "decoder_output_layer_0")):
        sums = row_sums(extra[name], trace.rows(stream, start, end))
        c[f"{name}_pcc"], c[f"{name}_nrmse"] = pcc(sums), nrmse(sums)
        c[f"{name}_row_nrmse_median"] = float(row_nrmse(sums, 0.0).median())
    for l in MOE_LAYERS:
        recall = topk_recall(
            products[f"routing_ids_{l}"].long(), trace.rows(f"expert_ids_layer_{l}", start, end).long()
        )
        c[f"routing_recall_{l}"] = float(recall.mean())
        c[f"routing_set_equal_{l}"] = float((recall == 1).float().mean())
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
        sums = row_sums(want, got)
        c[f"{name}_pcc"], c[f"{name}_nrmse"] = pcc(sums), nrmse(sums)
    trace_sel = trace.rows(f"dsa_topk_indices_layer_{DSA_LAYER}", start, end).long()
    golden_sel = products["topk_indices_3"].long()
    recall = topk_recall(trace_sel, golden_sel)
    full = torch.arange(start, end) >= topk - 1  # rows with a full selection (row + 1 >= top-k keys)
    c["topk_recall_min_partial"] = float(recall[~full].min()) if (~full).any() else float("nan")
    if full.any():
        c["topk_recall_mean"], c["topk_recall_p01"] = float(recall[full].mean()), float(recall[full].quantile(0.01))
        rank = missed_keys(trace_sel[full], golden_sel[full], products["topk_scores_3"][full].double())["rank"]
        c["topk_misses"] = int(rank.numel())
        c["topk_miss_top_half_fraction"] = float((2 * rank < topk).float().mean()) if rank.numel() else 0.0
    return c


# --- the run ---------------------------------------------------------------------------------------------------------


def segment_path(run_dir: Path, index: int) -> Path:
    """Segment ``index``'s row and sampled-row products."""
    return Path(run_dir) / f"segment_{index:06d}.pt"


def state_path(run_dir: Path, index: int) -> Path:
    """The state after segment ``index``: ``keys`` (the segment's DSA keys) and ``kda`` (each KDA layer's state)."""
    return Path(run_dir) / f"state_{index:06d}.pt"


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
    max_segments: int | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Run (or resume) the chain over ``token_ids`` into the golden ``run_dir`` and return the manifest; with
    ``max_segments``, stop once that many segments are complete. ``identity`` names the model, checkpoint and prompt;
    the weights' digests are added to it. A resume refuses a directory whose identity, tokens, segment size, sampled
    rows or thread count differ (``fp64_golden.start_or_resume``)."""
    run_dir = Path(run_dir)
    count = int(token_ids.numel())
    expected = {
        "golden_id": run_dir.name,
        "producer": PRODUCER,
        "producer_version": PRODUCER_VERSION,
        "identity": {**identity, "weights_sha256": dict(sorted(model.digests.items()))},
        "model": model.config(),
        "rows": count,
        "token_ids_sha256": token_ids_sha256(token_ids),
        "segment": segment,
        "sampled_rows": sampled,
        "dtype": "float64",
        "threads": torch.get_num_threads(),
    }
    segments = -(-count // segment)
    stop = segments if max_segments is None else min(segments, max_segments)
    manifest = start_or_resume(run_dir, expected, token_ids, stop)
    done = len(manifest["segments"])
    carried = load_carried(run_dir, done)
    if done:
        log(f"resuming at segment {done}/{segments} (row {manifest['segments'][-1]['end']})")
    for index in range(done, stop):
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
        kda = {l: vars(s) for l, s in carried.kda.items()}
        payload = {"start": start, "end": end, **storable(products)}
        publish(lambda path: torch.save(payload, path), segment_path(run_dir, index))
        publish(
            lambda path: torch.save({"start": start, "end": end, "keys": keys, "kda": kda}, path),
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
        if "products" not in manifest:
            per_row = {k: v for k, v in payload.items() if torch.is_tensor(v) and k not in SAMPLED}
            per_row.update({f"state.{k}": v for k, v in keys.items()})
            manifest["bytes_per_row"] = {k: v.element_size() * v[0].numel() for k, v in per_row.items()}
            manifest["products"] = product_schemas(payload, keys, kda)
        manifest["segments"].append(entry)
        write_manifest(run_dir, manifest)
        log(
            f"segment {index + 1}/{segments} done {entry['seconds']:.1f} s ({(end - start) / entry['seconds']:.1f} "
            f"rows/s) rss {entry['max_rss_gb']} GB " + json.dumps(entry.get("checks", {}))
        )
    if len(manifest["segments"]) == segments:
        problems = manifest_problems(run_dir.name, manifest)
        if problems:
            raise ValueError(f"the golden in {run_dir} breaks the registry contract: {problems}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, required=True, help="vLLM trace: token ids and the cross-check streams")
    parser.add_argument("--count", type=int, help="prompt rows to run (default: the trace's prompt)")
    parser.add_argument("--out", type=Path, required=True, help="the golden directory; its name is the golden id")
    parser.add_argument("--segment", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads; a resume must use the same count")
    parser.add_argument("--segments", type=int, help="stop after this many complete segments (resumable)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            os.environ.get("KIMI_K31_HF_MODEL", "/mnt/weka/model-weights/llm/moonshotai/Kimi-K3.1-Pro-dequantized")
        ),
    )
    parser.add_argument("--no-checks", action="store_true", help="skip the per-segment trace cross-checks")
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
        sampled=sample_rows(count, SAMPLED_ROWS),
        identity=identity,
        trace=None if args.no_checks else trace,
        max_segments=args.segments,
        log=log,
    )


if __name__ == "__main__":
    main()
