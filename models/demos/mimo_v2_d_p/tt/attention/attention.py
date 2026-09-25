# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 chunked-prefill attention block (SP rows x TP cols), GA and SWA.

Input ``x`` is the input-normed hidden ``[1, 1, S_local, H]``: sequence block-cyclic over SP rows,
replicated over TP cols. Per chip (TP col c):

  qkv  = x @ wqkv_c          column-parallel, per col [q_c | k_c | v_c]; v rows zero-padded 128 -> 192 per
                             head so one nlp_create_qkv_heads splits all three, ``attention_value_scale``
                             folded into v, rope sub-block of q/k rows permuted to the Meta convention
  q, k = rope(q), rope(k)    first 64 of 192 dims, in place (indexed, block-cyclic cos/sin)
  cache <- k, v              update_padded_kv_cache (v sliced to 128 on GA layers)
  o    = ring SDPA           scale 192^-0.5; SWA: window 128 + per-head sink
  out  = allreduce_TP(concat_heads(o[..., :128]) @ wo_c)

KV-head placement: TP <= n_kv -> each col owns n_kv/TP heads (2x2: GA 2, SWA 4; Galaxy TP=4: GA 1,
SWA 2); TP > n_kv -> each col gets the single GQA head its Q heads use (duplicated).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tt.ffn import all_reduce_tp
from models.demos.mimo_v2_d_p.tt.mm_configs import best_mm_config
from models.demos.mimo_v2_d_p.tt.rope import permute_heads
from models.demos.mimo_v2_d_p.tt.weight_cache import cache_name

from .kv_cache import MiMoKVCache
from .sdpa import ring_attention


def kv_heads_for_col(col: int, tp: int, n_q: int, n_kv: int) -> list[int]:
    nq_l = n_q // tp
    if n_kv >= tp:
        per = n_kv // tp
        return list(range(col * per, (col + 1) * per))
    return [(col * nq_l) // (n_q // n_kv)]


def cache_v_dim(spec) -> int:
    """V head dim as stored in the cache / fed to SDPA (the ring sliding path accepts VDH < DH)."""
    return spec.v_head_dim


@dataclass
class AttentionWeights:
    wqkv: ttnn.Tensor
    wo: ttnn.Tensor
    sink: ttnn.Tensor | None


def attention_host_weights(cfg: MiMoTextConfig, layer_idx: int, sd: dict, tp: int) -> dict:
    """Host (torch) TP layout of one attention layer — what the device tensors are cut from.

    wqkv [H, tp * cols_per_col]: per TP col c ``[q heads of c | k heads | v heads]`` (column-sharded over TP);
    q/k rope rows permuted to the Meta convention, v scaled by ``attention_value_scale`` and zero-padded
    128 -> 192 per head (one nlp_create_qkv_heads splits all three). wo [n_q * vd, H] (row-sharded over TP).
    sink [1, n_q, 1, 1] = sink / scale (the op re-applies the SDPA scale), or None.
    """
    spec = cfg.layer_attn(layer_idx)
    n_q, n_kv, hd, vd = spec.n_q, spec.n_kv, spec.head_dim, spec.v_head_dim
    nq_l = n_q // tp
    w = sd["qkv_proj.weight"].float()
    wq, wk, wv = w.split([n_q * hd, n_kv * hd, n_kv * vd], 0)
    wq = permute_heads(wq, n_q, hd, spec.rope_dim)
    wk = permute_heads(wk, n_kv, hd, spec.rope_dim)
    wv = wv * cfg.attention_value_scale
    wv = torch.nn.functional.pad(wv.view(n_kv, vd, -1), (0, 0, 0, hd - vd)).reshape(n_kv * hd, -1)  # per-head pad
    heads = lambda t, idx, d: torch.cat([t[h * d : (h + 1) * d] for h in idx], 0)
    cols = []
    for c in range(tp):
        kv_idx = kv_heads_for_col(c, tp, n_q, n_kv)
        cols.append(torch.cat([wq[c * nq_l * hd : (c + 1) * nq_l * hd], heads(wk, kv_idx, hd), heads(wv, kv_idx, hd)], 0).T)
    sink = sd["attention_sink_bias"].float().view(1, n_q, 1, 1) * (hd**0.5) if spec.has_sink else None
    return {"wqkv": torch.cat(cols, -1), "wo": sd["o_proj.weight"].float().T.contiguous(), "sink": sink}


def load_attention_weights(mesh_device, cfg: MiMoTextConfig, layer_idx: int, sd: dict, weight_dtype=ttnn.bfloat8_b, cache_prefix=None):
    """``sd``: the layer's ``self_attn.*`` sub-state (HF names; fused qkv in global [Q|K|V] order)."""
    host = attention_host_weights(cfg, layer_idx, sd, mesh_device.shape[1])
    shape = tuple(mesh_device.shape)
    to = lambda t, dims, name, dt=weight_dtype: ttnn.as_tensor(
        t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dt, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=dims),
        cache_file_name=cache_name(mesh_device, cache_prefix, f"attn.{name}"),
    )
    sink = None if host["sink"] is None else to(host["sink"], (None, 1), "sink", ttnn.bfloat16)
    return AttentionWeights(wqkv=to(host["wqkv"][None, None], (None, 3), "wqkv"), wo=to(host["wo"][None, None], (None, 2), "wo"), sink=sink)


class TtAttention:
    def __init__(self, mesh_device, cfg: MiMoTextConfig, layer_idx: int, state_dict, ccl_manager, weight_dtype=ttnn.bfloat8_b,
                 cache_prefix=None):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.spec = cfg.layer_attn(layer_idx)
        self.tp = mesh_device.shape[1]
        self.sp_axis = 0
        self.nq_l = self.spec.n_q // self.tp
        self.nkv_l = len(kv_heads_for_col(0, self.tp, self.spec.n_q, self.spec.n_kv))
        self.window = self.spec.window
        self.v_dim = cache_v_dim(self.spec)
        self.scale = self.spec.head_dim**-0.5
        self.ccl = ccl_manager
        self.w = load_attention_weights(mesh_device, cfg, layer_idx, state_dict, weight_dtype, cache_prefix)
        self._pcs = {}
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def _pc(self, name, a, w):
        key = (name, a.shape[2])
        if key not in self._pcs:
            self._pcs[key] = best_mm_config(self.mesh_device, a.shape[2], a.shape[3], w.shape[3])
        return self._pcs[key]

    def __call__(self, x, rope, trans_mat, kv_cache: MiMoKVCache, *, cache_layer: int, kv_actual: int, user: int = 0, valid_end: int | None = None):
        """x [1,1,S_local,H] -> [1,1,S_local,H] (replicated over TP)."""
        S_local = x.shape[2]
        sp = self.mesh_device.shape[self.sp_axis]
        xqkv = ttnn.linear(x, self.w.wqkv, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg, program_config=self._pc("qkv", x, self.w.wqkv))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=self.nq_l, num_kv_heads=self.nkv_l, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xqkv.deallocate(True)
        # Partial rope: only the first rope_dim (64 of 192) dims of each head are rotated; the rest are copied.
        rope_kw = dict(kv_actual_global=kv_actual, cluster_axis=self.sp_axis, rotary_dim=rope[0].shape[-1], rotary_offset=0)
        qr = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(q, rope[0], rope[1], trans_mat, **rope_kw)
        kr = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(k, rope[0], rope[1], trans_mat, **rope_kw)
        q.deallocate(True)
        k.deallocate(True)
        if self.v_dim != v.shape[3]:
            vs = ttnn.slice(v, [0, 0, 0, 0], [1, self.nkv_l, S_local, self.v_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v.deallocate(True)
            v = vs

        slot = user * kv_cache.num_layers + cache_layer
        for cache, t in ((kv_cache.k, kr), (kv_cache.v, v)):
            src = ttnn.typecast(t, cache.dtype) if t.dtype != cache.dtype else t
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache, src, slot_idx=user, layer_idx=cache_layer, num_layers=kv_cache.num_layers,
                kv_actual_global=kv_actual, cluster_axis=self.sp_axis, valid_global=valid_end,
            )
            if src is not t:
                src.deallocate(True)
        kr.deallocate(True)
        v.deallocate(True)

        o = ring_attention(
            qr, kv_cache, kv_actual=kv_actual, logical_n=kv_actual + S_local * sp, window=self.window, sink=self.w.sink,
            layer_slot=slot, mesh_device=self.mesh_device, ccl_manager=self.ccl, sp_axis=self.sp_axis, scale=self.scale,
        )
        qr.deallocate(True)
        vd = self.spec.v_head_dim
        if o.shape[3] != vd:
            os_ = ttnn.slice(o, [0, 0, 0, 0], [1, self.nq_l, S_local, vd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o.deallocate(True)
            o = os_
        oc = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o.deallocate(True)
        out = ttnn.linear(oc, self.w.wo, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_cfg, program_config=self._pc("o", oc, self.w.wo))
        oc.deallocate(True)
        return all_reduce_tp(out, self.mesh_device)
