# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side preparation and sharding plan of one Hy4 attention layer (CPU, torch). Mesh 4x8: rows = users, columns = TP8 (head groups).

Every entry is ``HostW(tensor, shard_dim)``: ``tensor`` in the layout the device matmul wants (x @ W, i.e. [in, out], batch dims in front), ``shard_dim`` the tensor dim
that is split over the 8 columns (None = replicated); rows always replicate. Column c owns heads 8c..8c+7 (``cfg.heads_per_col``).

  q_a      [1,1,6144,2048]  N-sharded (256 per column) -> all_gather -> q_a_layernorm -> q_b   (a replicated q_a would cost 25 MB x 78 layers per chip)
  q_b      [1,1,2048,16384] N-sharded: column c = the 8 heads' [192 nope | 64 rope] blocks
  kv_a     [1,1,6144,576]   replicated (3.5 MB / layer): 512 latent | 64 rope key
  w_uk     [1,64,192,512]   head-sharded: q_nope[h] @ w_uk[h] = absorbed query (512)            (kv_b_proj rows h*448 .. h*448+191, transposed view)
  w_uv_t   [1,64,512,256]   head-sharded: o_latent[h] @ w_uv_t[h] = per-head value output (256)  (kv_b_proj rows h*448+192 .. h*448+447, transposed)
  gate     [1,1,6144,16384] N-sharded (elementwise gate, sigmoid)
  o_proj   [1,1,16384,6144] K-sharded -> partial sums -> all-reduce
  sink     fp32 [64] (the logit-domain value of the checkpoint) -> sparse_sdpa wants sink / scale, bf16 [1,1,1,H]
  indexer (full layers only): wq_b [1,1,2048,4096], wk [1,1,6144,128], weights_proj [1,1,6144,32] (x 32^-.5 * 128^-.5), k_norm weight/bias; all replicated
"""
from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class HostW:
    tensor: torch.Tensor
    shard_dim: Optional[int]  # tensor dim split over the TP columns; None = replicated


def _norm_w(w):
    """rms/layer norm weight [K] -> [1,1,K/32,32] (the row-major layout ttnn norm ops take)."""
    return w.reshape(1, 1, -1, 32).contiguous()


def build_attn_host_weights(
    cfg, sd: Dict[str, torch.Tensor], tp: int = 8, sink_scale: Optional[float] = None
) -> Dict[str, HostW]:
    """sd: checkpoint tensors of the layer keyed by the path after ``model.layers.<i>.`` (``self_attn.q_a_proj.weight`` ...)."""
    H, nope, rope, vd, kvl = (
        cfg.num_attention_heads,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        cfg.v_head_dim,
        cfg.kv_lora_rank,
    )
    g = lambda k: sd[f"self_attn.{k}"]
    out: Dict[str, HostW] = {}
    out["q_a"] = HostW(g("q_a_proj.weight").float().T.reshape(1, 1, cfg.hidden_size, cfg.q_lora_rank), 3)
    out["q_a_norm"] = HostW(_norm_w(g("q_a_layernorm.weight").float()), None)
    out["q_b"] = HostW(g("q_b_proj.weight").float().T.reshape(1, 1, cfg.q_lora_rank, H * (nope + rope)), 3)
    out["kv_a"] = HostW(g("kv_a_proj_with_mqa.weight").float().T.reshape(1, 1, cfg.hidden_size, kvl + rope), None)
    out["kv_a_norm"] = HostW(_norm_w(g("kv_a_layernorm.weight").float()), None)
    kvb = g("kv_b_proj.weight").float().reshape(H, nope + vd, kvl)  # [64, 448, 512]
    out["w_uk"] = HostW(kvb[:, :nope, :].reshape(1, H, nope, kvl).contiguous(), 1)
    out["w_uv_t"] = HostW(kvb[:, nope:, :].transpose(1, 2).reshape(1, H, kvl, vd).contiguous(), 1)
    out["gate"] = HostW(g("linear_gate.weight").float().T.reshape(1, 1, cfg.hidden_size, H * vd), 3)
    out["o_proj"] = HostW(g("o_proj.weight").float().T.reshape(1, 1, H * vd, cfg.hidden_size), 2)
    scale = (nope + rope) ** -0.5 if sink_scale is None else sink_scale
    out["sink_over_scale"] = HostW((g("learnable_sink_param").float() / scale).reshape(1, 1, 1, H), 3)
    hc = H // tp
    dec = torch.zeros(tp, 32)  # decode: the column's 8 heads zero-padded to the 32 heads the stock op needs
    dec[:, :hc] = out["sink_over_scale"].tensor.reshape(tp, hc)
    out["sink_dec"] = HostW(dec.reshape(1, 1, 1, tp * 32), 3)
    if "self_attn.indexer.wq_b.weight" in sd:
        ni, di = cfg.index_n_heads, cfg.index_head_dim
        out["idx_wq_b"] = HostW(g("indexer.wq_b.weight").float().T.reshape(1, 1, cfg.q_lora_rank, ni * di), None)
        out["idx_wk"] = HostW(g("indexer.wk.weight").float().T.reshape(1, 1, cfg.hidden_size, di), None)
        out["idx_wp"] = HostW(
            (g("indexer.weights_proj.weight").float() * (ni**-0.5) * (di**-0.5)).T.reshape(
                1, 1, cfg.hidden_size, ni
            ),
            None,
        )
        out["idx_k_norm_w"] = HostW(_norm_w(g("indexer.k_norm.weight").float()), None)
        out["idx_k_norm_b"] = HostW(_norm_w(g("indexer.k_norm.bias").float()), None)
    return out


def column_shard(w: HostW, col: int, tp: int = 8) -> torch.Tensor:
    """The slice column ``col`` holds (what ShardTensor2dMesh(dims=(None, shard_dim)) hands it)."""
    if w.shard_dim is None:
        return w.tensor
    n = w.tensor.shape[w.shard_dim] // tp
    return w.tensor.narrow(w.shard_dim, col * n, n)
