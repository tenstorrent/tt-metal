# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 chunked-prefill attention block (SP rows x TP cols).

Input ``x`` is the input-normed hidden state ``[1, 1, S_local, H]``: sequence block-cyclic over SP rows,
replicated over TP cols. Per chip:

  qkv  = x @ wqkv_c                        (column-parallel; per-col [q_c | k_c | v_c])
  q,k,v = nlp_create_qkv_heads
  q = rope(q_norm(q)), k = rope(k_norm(k)), v = v_norm(v)      (per-head RMSNorm; v_norm has no scale)
  cache <- k, v                            (update_padded_kv_cache, block-cyclic)
  o = chunk_attention(q, cache)            (ring SDPA over SP, scale 1.0)
  out = allreduce_TP(concat_heads(o) @ wo_c)

RoPE runs in the Meta interleaved convention: q/k projection rows and q/k norm weights are
reverse-permuted per head (dot products are invariant; the norms commute with the permutation).
Full layers have no V projection (``attention_k_eq_v``): V is the *raw* K projection, so the fused weight
carries an unpermuted copy of k_proj in the V slot.

KV-head placement: TP >= n_kv -> each col gets n_kv/TP heads; TP > n_kv (full layers at TP=4) -> each col
gets the single GQA head its Q heads use (duplicated across cols).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4_26b_d_p.reference.config import Gemma4TextConfig
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

from .kv_cache import Gemma4KVCache
from .sdpa import chunk_attention


def kv_heads_for_col(col: int, tp: int, n_q: int, n_kv: int) -> list[int]:
    nq_l = n_q // tp
    if n_kv >= tp:
        per = n_kv // tp
        return list(range(col * per, (col + 1) * per))
    return [(col * nq_l) // (n_q // n_kv)]


@dataclass
class AttentionWeights:
    wqkv: ttnn.Tensor
    wo: ttnn.Tensor
    q_norm: ttnn.Tensor
    k_norm: ttnn.Tensor


def _norm_weight(w, mesh_device):
    return ttnn.from_torch(
        w.reshape(1, 1, -1, ttnn.TILE_SIZE),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def load_attention_weights(mesh_device, cfg: Gemma4TextConfig, layer_idx: int, sd: dict, weight_dtype=ttnn.bfloat8_b):
    """``sd``: the layer's ``self_attn.*`` sub-state (HF names)."""
    tp = mesh_device.shape[1]
    n_q, n_kv, hd = cfg.num_attention_heads, cfg.layer_kv_heads(layer_idx), cfg.layer_head_dim(layer_idx)
    nq_l = n_q // tp
    wq = sd["q_proj.weight"].float()
    wk = sd["k_proj.weight"].float()
    wv = wk if cfg.layer_k_eq_v(layer_idx) else sd["v_proj.weight"].float()
    wq_m = reverse_permute(wq, n_q, wq.shape[0], wq.shape[1])
    wk_m = reverse_permute(wk, n_kv, wk.shape[0], wk.shape[1])
    heads = lambda w, idx: torch.cat([w[h * hd : (h + 1) * hd] for h in idx], 0)
    cols = []
    for c in range(tp):
        kv_idx = kv_heads_for_col(c, tp, n_q, n_kv)
        q_c = wq_m[c * nq_l * hd : (c + 1) * nq_l * hd]
        cols.append(torch.cat([q_c, heads(wk_m, kv_idx), heads(wv, kv_idx)], 0).T)
    wqkv = torch.cat(cols, -1)[None, None]
    wo = sd["o_proj.weight"].float().T[None, None]  # [1,1,nq*hd, H], rows sharded over TP
    col_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 3))
    row_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 2))
    to = lambda t, m: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=weight_dtype, mesh_mapper=m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return AttentionWeights(
        wqkv=to(wqkv, col_map),
        wo=to(wo, row_map),
        q_norm=_norm_weight(reverse_permute_1d(sd["q_norm.weight"].float()), mesh_device),
        k_norm=_norm_weight(reverse_permute_1d(sd["k_norm.weight"].float()), mesh_device),
    )


class TtAttention:
    def __init__(self, mesh_device, cfg: Gemma4TextConfig, layer_idx: int, state_dict, ccl_manager, tp_topology, weight_dtype=ttnn.bfloat8_b):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.tp = mesh_device.shape[1]
        self.sp_axis = 0
        self.nq_l = cfg.num_attention_heads // self.tp
        self.nkv_l = len(kv_heads_for_col(0, self.tp, cfg.num_attention_heads, cfg.layer_kv_heads(layer_idx)))
        self.window = cfg.sliding_window if cfg.is_sliding(layer_idx) else None
        self.eps = cfg.rms_norm_eps
        self.ccl = ccl_manager
        self.tp_topology = tp_topology
        self.w = load_attention_weights(mesh_device, cfg, layer_idx, state_dict, weight_dtype)
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def __call__(self, x, rope, trans_mat, kv_cache: Gemma4KVCache, *, cache_layer: int, kv_actual: int, user: int = 0, valid_end: int | None = None):
        """x [1,1,S_local,H] -> [1,1,S_local,H]. ``cache_layer``: this layer's index within ``kv_cache``."""
        S_local = x.shape[2]
        sp = self.mesh_device.shape[self.sp_axis]
        xqkv = ttnn.linear(x, self.w.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=self.nq_l, num_kv_heads=self.nkv_l, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xqkv.deallocate(True)
        qn = ttnn.rms_norm(q, epsilon=self.eps, weight=self.w.q_norm)
        kn = ttnn.rms_norm(k, epsilon=self.eps, weight=self.w.k_norm)
        vn = ttnn.rms_norm(v, epsilon=self.eps)
        for t in (q, k, v):
            t.deallocate(True)
        rope_kw = dict(kv_actual_global=kv_actual, cluster_axis=self.sp_axis)
        qr = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(qn, rope[0], rope[1], trans_mat, **rope_kw)
        kr = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(kn, rope[0], rope[1], trans_mat, **rope_kw)
        qn.deallocate(True)
        kn.deallocate(True)

        slot = user * kv_cache.num_layers + cache_layer
        for cache, t in ((kv_cache.k, kr), (kv_cache.v, vn)):
            src = ttnn.typecast(t, cache.dtype) if t.dtype != cache.dtype else t
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache, src, slot_idx=user, layer_idx=cache_layer, num_layers=kv_cache.num_layers,
                kv_actual_global=kv_actual, cluster_axis=self.sp_axis, valid_global=valid_end,
            )
            if src is not t:
                src.deallocate(True)

        o = chunk_attention(
            qr, kv_cache, kr, vn,
            kv_actual=kv_actual, logical_n=kv_actual + S_local * sp, window=self.window,
            layer_slot=slot, num_slots_layers=kv_cache.num_users * kv_cache.num_layers,
            mesh_device=self.mesh_device, ccl_manager=self.ccl, sp_axis=self.sp_axis, scale=1.0,
        )
        for t in (qr, kr, vn):
            t.deallocate(True)
        oc = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o.deallocate(True)
        out = ttnn.linear(oc, self.w.wo, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg)
        oc.deallocate(True)
        if self.tp > 1:
            red = ttnn.all_reduce(out, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.deallocate(True)
            out = red
        return out
